use tensorflow::{
    ops::{
        bias_add, broadcast_to, constant, mat_mul, mul, Conv2D as TFConv2D, RandomStandardNormal,
    },
    DataType, Operation, Scope, Status, Variable,
};

#[derive(Debug, Clone)]
pub struct Conv2D {
    pub w: Variable,
    pub b: Variable,
    pub output: Operation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Conv2DPadding {
    Valid,
    Same,
}

#[derive(Debug, Clone)]
pub struct Fc {
    pub w: Variable,
    pub b: Variable,
    pub output: Operation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Represents weight initialization method.
pub enum WeightInitializer {
    /// LeCun initializer. It is suitable for SELU activation.
    LeCun,
    /// Xavier initializer; also known as Glorot initializer. It is suitable for sigmoid or tanh activation.
    Xavier,
    /// He initializer. It is suitable for ReLU-like activation.
    He,
}

impl WeightInitializer {
    pub fn build_constant(&self, fan_in: i64, fan_out: i64) -> f32 {
        match self {
            WeightInitializer::LeCun => 1f32 / f32::sqrt(fan_in as f32),
            WeightInitializer::Xavier => 2f32 / f32::sqrt((fan_in + fan_out) as f32),
            WeightInitializer::He => 2f32 / f32::sqrt(fan_in as f32),
        }
    }
}

pub fn conv2d(
    name: impl AsRef<str>,
    data_type: DataType,
    x: Operation,
    input_channels: i64,
    output_channels: i64,
    filter_size: &[i64; 2],
    stride: &[i64; 2],
    padding: Conv2DPadding,
    weight_init: WeightInitializer,
    scope: &mut Scope,
) -> Result<Conv2D, Status> {
    let name = name.as_ref();
    let w = Variable::builder()
        .data_type(data_type)
        .shape(&[
            filter_size[1],
            filter_size[0],
            input_channels,
            output_channels,
        ])
        .initial_value(mul(
            RandomStandardNormal::new().dtype(data_type).build(
                constant(
                    &[
                        filter_size[1],
                        filter_size[0],
                        input_channels,
                        output_channels,
                    ],
                    scope,
                )?,
                scope,
            )?,
            constant(
                weight_init.build_constant(
                    filter_size[1] * filter_size[0] * input_channels,
                    filter_size[1] * filter_size[0] * output_channels,
                ),
                scope,
            )?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_w", name)))?;
    let b = Variable::builder()
        .data_type(data_type)
        .shape(&[output_channels])
        .initial_value(broadcast_to(
            constant(&[0f32], scope)?,
            constant(&[output_channels], scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_b", name)))?;
    let conv = TFConv2D::new()
        .data_format("NHWC")
        .strides([1, stride[1], stride[0], 1])
        .padding(match padding {
            Conv2DPadding::Valid => "VALID",
            Conv2DPadding::Same => "SAME",
        })
        .build(
            x,
            w.output().clone(),
            &mut scope.with_op_name(&format!("{}_conv2d", name)),
        )?;
    let conv_biased = bias_add(
        conv,
        b.output().clone(),
        &mut scope.with_op_name(&format!("{}_bias_add", name)),
    )?;
    Ok(Conv2D {
        w,
        b,
        output: conv_biased,
    })
}

pub fn fc(
    name: impl AsRef<str>,
    data_type: DataType,
    x: Operation,
    inputs: i64,
    outputs: i64,
    weight_init: WeightInitializer,
    scope: &mut Scope,
) -> Result<Fc, Status> {
    let name = name.as_ref();
    let w = Variable::builder()
        .data_type(data_type)
        .shape(&[inputs, outputs])
        .initial_value(mul(
            RandomStandardNormal::new()
                .dtype(data_type)
                .build(constant(&[inputs, outputs], scope)?, scope)?,
            constant(weight_init.build_constant(inputs, outputs), scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_w", name)))?;
    let b = Variable::builder()
        .data_type(data_type)
        .shape(&[outputs])
        .initial_value(broadcast_to(
            constant(&[0f32], scope)?,
            constant(&[outputs], scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_b", name)))?;
    let mm = mat_mul(
        x,
        w.output().clone(),
        &mut scope.with_op_name(&format!("{}_mat_mul", name)),
    )?;
    let mm_biased = bias_add(
        mm,
        b.output().clone(),
        &mut scope.with_op_name(&format!("{}_bias_add", name)),
    )?;
    Ok(Fc {
        w,
        b,
        output: mm_biased,
    })
}
