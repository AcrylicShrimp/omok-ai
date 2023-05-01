use environment::Environment;
use network_utils::{Conv2DPadding, WeightInitializer};
use tensorflow::{
    ops::{
        constant, leaky_relu, reshape, softmax, softmax_cross_entropy_with_logits, tanh,
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
    pub const INPUT_CHANNELS: i64 = 1;

    pub const CONV0_FILTER_SIZE: i64 = 8;
    pub const CONV0_CHANNELS: i64 = 16;
    pub const CONV0_STRIDE: i64 = 2;

    pub const CONV1_FILTER_SIZE: i64 = 4;
    pub const CONV1_CHANNELS: i64 = 32;
    pub const CONV1_STRIDE: i64 = 1;

    pub const FLATTEN_SIZE: i64 = 288;

    pub const V_FC0_SIZE: i64 = 256;
    pub const V_FC1_SIZE: i64 = 128;
    pub const V_FC2_SIZE: i64 = 1;

    pub const P_FC0_SIZE: i64 = 256;
    pub const P_FC1_SIZE: i64 = 128;
    pub const P_FC2_SIZE: i64 = Environment::BOARD_SIZE as i64 * Environment::BOARD_SIZE as i64;
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

        let conv0 = network_utils::conv2d(
            "conv0",
            DataType::Float,
            op_input.clone(),
            Self::INPUT_CHANNELS,
            Self::CONV0_CHANNELS,
            &[Self::CONV0_FILTER_SIZE, Self::CONV0_FILTER_SIZE],
            &[Self::CONV0_STRIDE, Self::CONV0_STRIDE],
            Conv2DPadding::Valid,
            WeightInitializer::He,
            scope,
        )?;
        let conv0_activation =
            leaky_relu(conv0.output, &mut scope.with_op_name("conv0_activation"))?;
        variables.push(conv0.w);
        variables.push(conv0.b);

        let conv1 = network_utils::conv2d(
            "conv1",
            DataType::Float,
            conv0_activation,
            Self::CONV0_CHANNELS,
            Self::CONV1_CHANNELS,
            &[Self::CONV1_FILTER_SIZE, Self::CONV1_FILTER_SIZE],
            &[Self::CONV1_STRIDE, Self::CONV1_STRIDE],
            Conv2DPadding::Valid,
            WeightInitializer::He,
            scope,
        )?;
        let conv1_activation =
            leaky_relu(conv1.output, &mut scope.with_op_name("conv1_activation"))?;
        variables.push(conv1.w);
        variables.push(conv1.b);

        let flatten = reshape(
            conv1_activation,
            constant(&[-1, Self::FLATTEN_SIZE], scope)?,
            &mut scope.with_op_name("reshape"),
        )?;

        let v_fc0 = network_utils::fc(
            "v_fc0",
            DataType::Float,
            flatten.clone(),
            Self::FLATTEN_SIZE,
            Self::V_FC0_SIZE,
            WeightInitializer::He,
            scope,
        )?;
        let v_fc0_activation =
            leaky_relu(v_fc0.output, &mut scope.with_op_name("v_fc0_activation"))?;
        variables.push(v_fc0.w);
        variables.push(v_fc0.b);

        let v_fc1 = network_utils::fc(
            "v_fc1",
            DataType::Float,
            v_fc0_activation,
            Self::V_FC0_SIZE,
            Self::V_FC1_SIZE,
            WeightInitializer::He,
            scope,
        )?;
        let v_fc1_activation =
            leaky_relu(v_fc1.output, &mut scope.with_op_name("v_fc1_activation"))?;
        variables.push(v_fc1.w);
        variables.push(v_fc1.b);

        let v_fc2 = network_utils::fc(
            "v_fc2",
            DataType::Float,
            v_fc1_activation,
            Self::V_FC1_SIZE,
            Self::V_FC2_SIZE,
            WeightInitializer::Xavier,
            scope,
        )?;
        let v_fc2_activation = tanh(
            v_fc2.output,
            &mut scope.with_op_name(v_output_name.as_ref()),
        )?;
        variables.push(v_fc2.w);
        variables.push(v_fc2.b);

        let p_fc0 = network_utils::fc(
            "p_fc0",
            DataType::Float,
            flatten,
            Self::FLATTEN_SIZE,
            Self::P_FC0_SIZE,
            WeightInitializer::He,
            scope,
        )?;
        let p_fc0_activation =
            leaky_relu(p_fc0.output, &mut scope.with_op_name("p_fc0_activation"))?;
        variables.push(p_fc0.w);
        variables.push(p_fc0.b);

        let p_fc1 = network_utils::fc(
            "p_fc1",
            DataType::Float,
            p_fc0_activation,
            Self::P_FC0_SIZE,
            Self::P_FC1_SIZE,
            WeightInitializer::He,
            scope,
        )?;
        let p_fc1_activation =
            leaky_relu(p_fc1.output, &mut scope.with_op_name("p_fc1_activation"))?;
        variables.push(p_fc1.w);
        variables.push(p_fc1.b);

        let p_fc2 = network_utils::fc(
            "p_fc2",
            DataType::Float,
            p_fc1_activation,
            Self::P_FC1_SIZE,
            Self::P_FC2_SIZE,
            WeightInitializer::He,
            scope,
        )?;
        let p_fc2_activation = softmax(
            p_fc2.output.clone(),
            &mut scope.with_op_name("p_fc2_activation"),
        )?;
        variables.push(p_fc2.w);
        variables.push(p_fc2.b);

        let p_output = reshape(
            p_fc2_activation.clone(),
            constant(&[-1, Self::P_OUTPUT_SIZE, Self::P_OUTPUT_SIZE], scope)?,
            &mut scope.with_op_name(p_output_name.as_ref()),
        )?;

        let p_loss = softmax_cross_entropy_with_logits(
            p_fc2.output,
            op_p_label,
            &mut scope.with_op_name(p_loss_name.as_ref()),
        )?;

        Ok(Self {
            op_input,
            op_v_output: v_fc2_activation,
            op_p_output: p_output,
            op_p_loss: p_loss,
            variables,
        })
    }
}
