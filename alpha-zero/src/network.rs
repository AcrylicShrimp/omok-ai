use environment::Environment;
use network_utils::{Conv2DPadding, WeightInitializer};
use tensorflow::{
    ops::{
        constant, leaky_relu, mean, reshape, softmax, softmax_cross_entropy_with_logits, tanh,
        Placeholder,
    },
    DataType, Operation, Scope, Status, Variable,
};

pub struct Network {
    pub op_input: Operation,
    pub op_v_output: Operation,
    pub op_p_output: Operation,
    pub op_p_loss: Operation,
    pub variables: Vec<Variable>,
}

impl Network {
    pub const INPUT_SIZE: i64 = Environment::BOARD_SIZE as i64;
    pub const INPUT_CHANNELS: i64 = 2;

    pub const RESIDUAL_FILTER_SIZE: i64 = 3;
    pub const RESIDUAL_CHANNELS: i64 = 128;
    pub const RESIDUAL_MIDDLE_CHANNELS: i64 = 32;
    pub const RESIDUAL_STRIDE: i64 = 1;
    pub const RESIDUAL_COUNT: i64 = 7;

    pub const V_CONV_FILTER_SIZE: i64 = 1;
    pub const V_CONV_CHANNELS: i64 = 1;
    pub const V_CONV_STRIDE: i64 = 1;

    pub const V_FLATTEN_SIZE: i64 =
        Environment::BOARD_SIZE as i64 * Environment::BOARD_SIZE as i64 * Self::V_CONV_CHANNELS;

    pub const V_FC0_SIZE: i64 = 1;

    pub const P_CONV_FILTER_SIZE: i64 = 1;
    pub const P_CONV_CHANNELS: i64 = 32;
    pub const P_CONV_STRIDE: i64 = 1;

    pub const P_FLATTEN_SIZE: i64 =
        Environment::BOARD_SIZE as i64 * Environment::BOARD_SIZE as i64 * Self::P_CONV_CHANNELS;

    pub const P_FC0_SIZE: i64 = Environment::BOARD_SIZE as i64 * Environment::BOARD_SIZE as i64;
    pub const P_OUTPUT_SIZE: i64 = Environment::BOARD_SIZE as i64;

    pub fn new(
        op_p_label: Operation,
        scope: &mut Scope,
        input_name: impl AsRef<str>,
        v_output_name: impl AsRef<str>,
        p_output_name: impl AsRef<str>,
        p_loss_name: impl AsRef<str>,
    ) -> Result<Self, Status> {
        let mut variables = Vec::new();
        let op_input = Placeholder::new()
            .dtype(DataType::Float)
            .shape([-1, Self::INPUT_SIZE, Self::INPUT_SIZE, Self::INPUT_CHANNELS])
            .build(&mut scope.with_op_name(input_name.as_ref()))?;

        let conv = network_utils::conv2d(
            "conv",
            DataType::Float,
            op_input.clone(),
            Self::INPUT_CHANNELS,
            Self::RESIDUAL_CHANNELS,
            &[1, 1],
            &[1, 1],
            Conv2DPadding::Same,
            WeightInitializer::He,
            scope,
        )?;
        let activation = leaky_relu(conv.output, &mut scope.with_op_name("conv_activation"))?;
        variables.push(conv.w);
        variables.push(conv.b);

        let mut previous = activation;

        for i in 0..Self::RESIDUAL_COUNT {
            let residual = network_utils::conv2d_bottleneck_residual(
                format!("residual_{}", i),
                DataType::Float,
                previous,
                Self::RESIDUAL_CHANNELS,
                Self::RESIDUAL_MIDDLE_CHANNELS,
                &[Self::RESIDUAL_FILTER_SIZE, Self::RESIDUAL_FILTER_SIZE],
                &[Self::RESIDUAL_STRIDE, Self::RESIDUAL_STRIDE],
                Conv2DPadding::Same,
                WeightInitializer::He,
                scope,
            )?;
            let activation = leaky_relu(
                residual.output,
                &mut scope.with_op_name(&format!("residual_{}_activation", i)),
            )?;

            variables.push(residual.w0);
            variables.push(residual.b0);

            variables.push(residual.depthwise_w1);
            variables.push(residual.pointwise_w1);
            variables.push(residual.b1);

            variables.push(residual.w2);
            variables.push(residual.b2);

            previous = activation;
        }

        let v_conv = network_utils::conv2d(
            "v_conv",
            DataType::Float,
            previous.clone(),
            Self::RESIDUAL_CHANNELS,
            Self::V_CONV_CHANNELS,
            &[Self::V_CONV_FILTER_SIZE, Self::V_CONV_FILTER_SIZE],
            &[Self::V_CONV_STRIDE, Self::V_CONV_STRIDE],
            Conv2DPadding::Same,
            WeightInitializer::He,
            scope,
        )?;
        let v_conv_activation =
            leaky_relu(v_conv.output, &mut scope.with_op_name("v_conv_activation"))?;
        variables.push(v_conv.w);
        variables.push(v_conv.b);

        let v_flatten = reshape(
            v_conv_activation,
            constant(&[-1, Self::V_FLATTEN_SIZE], scope)?,
            &mut scope.with_op_name("v_flatten"),
        )?;

        let v_fc0 = network_utils::fc(
            "v_fc0",
            DataType::Float,
            v_flatten,
            Self::V_FLATTEN_SIZE,
            Self::V_FC0_SIZE,
            WeightInitializer::He,
            scope,
        )?;
        let v_fc0_activation = tanh(
            v_fc0.output,
            &mut scope.with_op_name(v_output_name.as_ref()),
        )?;
        variables.push(v_fc0.w);
        variables.push(v_fc0.b);

        let p_conv = network_utils::conv2d(
            "p_conv",
            DataType::Float,
            previous.clone(),
            Self::RESIDUAL_CHANNELS,
            Self::P_CONV_CHANNELS,
            &[Self::P_CONV_FILTER_SIZE, Self::P_CONV_FILTER_SIZE],
            &[Self::P_CONV_STRIDE, Self::P_CONV_STRIDE],
            Conv2DPadding::Same,
            WeightInitializer::He,
            scope,
        )?;
        let p_conv_activation =
            leaky_relu(p_conv.output, &mut scope.with_op_name("p_conv_activation"))?;
        variables.push(p_conv.w);
        variables.push(p_conv.b);

        let p_flatten = reshape(
            p_conv_activation,
            constant(&[-1, Self::P_FLATTEN_SIZE], scope)?,
            &mut scope.with_op_name("p_flatten"),
        )?;

        let p_fc0 = network_utils::fc(
            "p_fc0",
            DataType::Float,
            p_flatten,
            Self::P_FLATTEN_SIZE,
            Self::P_FC0_SIZE,
            WeightInitializer::He,
            scope,
        )?;
        let p_fc0_activation = softmax(
            p_fc0.output.clone(),
            &mut scope.with_op_name("p_fc0_activation"),
        )?;
        variables.push(p_fc0.w);
        variables.push(p_fc0.b);

        let p_output = reshape(
            p_fc0_activation.clone(),
            constant(&[-1, Self::P_OUTPUT_SIZE, Self::P_OUTPUT_SIZE], scope)?,
            &mut scope.with_op_name(p_output_name.as_ref()),
        )?;

        let p_loss = mean(
            softmax_cross_entropy_with_logits(p_fc0.output, op_p_label, scope)?,
            constant(&[0], scope)?,
            &mut scope.with_op_name(p_loss_name.as_ref()),
        )?;

        Ok(Self {
            op_input,
            op_v_output: v_fc0_activation,
            op_p_output: p_output,
            op_p_loss: p_loss,
            variables,
        })
    }
}
