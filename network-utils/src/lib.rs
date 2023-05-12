use tensorflow::{
    ops::{
        add, assign, bias_add, broadcast_to, constant, leaky_relu, mat_mul, mul, reshape,
        Conv2D as TFConv2D, DepthwiseConv2dNative as TFDepthwiseConv2dNative, FusedBatchNormV3,
        Identity, MaxPool, RandomStandardNormal,
    },
    DataType, Operation, Scope, Status, Variable,
};

#[derive(Debug, Clone)]
pub struct Conv2D {
    pub w: Variable,
    pub b: Variable,
    pub output: Operation,
}

#[derive(Debug, Clone)]
pub struct SeparableConv2D {
    pub depthwise_w: Variable,
    pub pointwise_w: Variable,
    pub b: Variable,
    pub output: Operation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Conv2DPadding {
    Valid,
    Same,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PoolPadding {
    Valid,
    Same,
}

#[derive(Debug, Clone)]
pub struct Fc {
    pub w: Variable,
    pub b: Variable,
    pub output: Operation,
}

#[derive(Debug, Clone)]
pub struct Conv2DResidual {
    pub w0: Variable,
    pub b0: Variable,
    pub w1: Variable,
    pub b1: Variable,
    pub output: Operation,
}

#[derive(Debug, Clone)]
pub struct Conv2DBottleneckResidual {
    pub w0: Variable,
    pub b0: Variable,
    pub depthwise_w1: Variable,
    pub pointwise_w1: Variable,
    pub b1: Variable,
    pub w2: Variable,
    pub b2: Variable,
    pub output: Operation,
}

#[derive(Debug, Clone)]
pub struct BatchNorm {
    pub scale: Variable,
    pub offset: Variable,
    pub mean: Variable,
    pub variance: Variable,
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

pub fn separable_conv2d(
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
) -> Result<SeparableConv2D, Status> {
    let name = name.as_ref();

    let depthwise_w = Variable::builder()
        .data_type(data_type)
        .shape(&[filter_size[1], filter_size[0], input_channels, 1])
        .initial_value(mul(
            RandomStandardNormal::new().dtype(data_type).build(
                constant(&[filter_size[1], filter_size[0], input_channels, 1], scope)?,
                scope,
            )?,
            constant(
                weight_init.build_constant(
                    filter_size[1] * filter_size[0] * input_channels,
                    filter_size[1] * filter_size[0] * 1,
                ),
                scope,
            )?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_w", name)))?;
    let depthwise_conv = TFDepthwiseConv2dNative::new()
        .data_format("NHWC")
        .strides([1, stride[1], stride[0], 1])
        .padding(match padding {
            Conv2DPadding::Valid => "VALID",
            Conv2DPadding::Same => "SAME",
        })
        .build(
            x,
            depthwise_w.output().clone(),
            &mut scope.with_op_name(&format!("{}_depthwise_conv2d", name)),
        )?;

    let pointwise_w = Variable::builder()
        .data_type(data_type)
        .shape(&[1, 1, input_channels, output_channels])
        .initial_value(mul(
            RandomStandardNormal::new().dtype(data_type).build(
                constant(&[1, 1, input_channels, output_channels], scope)?,
                scope,
            )?,
            constant(
                weight_init.build_constant(1 * 1 * input_channels, 1 * 1 * output_channels),
                scope,
            )?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_w", name)))?;
    let pointwise_conv = TFConv2D::new()
        .data_format("NHWC")
        .strides([1, 1, 1, 1])
        .padding("SAME")
        .build(
            depthwise_conv,
            pointwise_w.output().clone(),
            &mut scope.with_op_name(&format!("{}_pointwise_conv2d", name)),
        )?;

    let b = Variable::builder()
        .data_type(data_type)
        .shape(&[output_channels])
        .initial_value(broadcast_to(
            constant(&[0f32], scope)?,
            constant(&[output_channels], scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_b", name)))?;
    let conv_biased = bias_add(
        pointwise_conv,
        b.output().clone(),
        &mut scope.with_op_name(&format!("{}_bias_add", name)),
    )?;
    Ok(SeparableConv2D {
        depthwise_w,
        pointwise_w,
        b,
        output: conv_biased,
    })
}

pub fn max_pool(
    name: impl AsRef<str>,
    x: Operation,
    filter_size: &[i64; 2],
    stride: &[i64; 2],
    padding: PoolPadding,
    scope: &mut Scope,
) -> Result<Operation, Status> {
    let name = name.as_ref();
    let pool = MaxPool::new()
        .data_format("NHWC")
        .ksize([1, filter_size[1], filter_size[0], 1])
        .strides([1, stride[1], stride[0], 1])
        .padding(match padding {
            PoolPadding::Valid => "VALID",
            PoolPadding::Same => "SAME",
        })
        .build(x, &mut scope.with_op_name(&format!("{}_max_pool", name)))?;
    Ok(pool)
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

pub fn conv2d_residual(
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
) -> Result<Conv2DResidual, Status> {
    let conv0 = conv2d(
        &format!("{}_conv0", name.as_ref()),
        data_type,
        x.clone(),
        input_channels,
        output_channels,
        filter_size,
        stride,
        Conv2DPadding::Same,
        WeightInitializer::He,
        scope,
    )?;
    let relu = leaky_relu(
        conv0.output,
        &mut scope.with_op_name(&format!("{}_relu", name.as_ref())),
    )?;
    let conv1 = conv2d(
        &format!("{}_conv1", name.as_ref()),
        data_type,
        relu,
        output_channels,
        output_channels,
        filter_size,
        stride,
        padding,
        weight_init,
        scope,
    )?;
    let add = add(
        conv1.output,
        x,
        &mut scope.with_op_name(&format!("{}_add", name.as_ref())),
    )?;
    Ok(Conv2DResidual {
        w0: conv0.w,
        b0: conv0.b,
        w1: conv1.w,
        b1: conv1.b,
        output: add,
    })
}

pub fn conv2d_bottleneck_residual(
    name: impl AsRef<str>,
    data_type: DataType,
    x: Operation,
    channels: i64,
    middle_channels: i64,
    filter_size: &[i64; 2],
    stride: &[i64; 2],
    padding: Conv2DPadding,
    weight_init: WeightInitializer,
    scope: &mut Scope,
) -> Result<Conv2DBottleneckResidual, Status> {
    let conv0 = conv2d(
        &format!("{}_conv0", name.as_ref()),
        data_type,
        x.clone(),
        channels,
        middle_channels,
        &[1, 1],
        &[1, 1],
        Conv2DPadding::Same,
        WeightInitializer::He,
        scope,
    )?;
    let activation0 = leaky_relu(
        conv0.output,
        &mut scope.with_op_name(&format!("{}_activation0", name.as_ref())),
    )?;

    let conv1 = separable_conv2d(
        &format!("{}_conv1", name.as_ref()),
        data_type,
        activation0,
        middle_channels,
        middle_channels,
        filter_size,
        stride,
        padding,
        WeightInitializer::He,
        scope,
    )?;
    let activation1 = leaky_relu(
        conv1.output,
        &mut scope.with_op_name(&format!("{}_activation1", name.as_ref())),
    )?;

    let conv2 = conv2d(
        &format!("{}_conv2", name.as_ref()),
        data_type,
        activation1,
        middle_channels,
        channels,
        &[1, 1],
        &[1, 1],
        Conv2DPadding::Same,
        weight_init,
        scope,
    )?;

    let add = add(
        conv2.output,
        x,
        &mut scope.with_op_name(&format!("{}_add", name.as_ref())),
    )?;

    Ok(Conv2DBottleneckResidual {
        w0: conv0.w,
        b0: conv0.b,
        depthwise_w1: conv1.depthwise_w,
        pointwise_w1: conv1.pointwise_w,
        b1: conv1.b,
        w2: conv2.w,
        b2: conv2.b,
        output: add,
    })
}

pub fn batch_norm(
    name: impl AsRef<str>,
    data_type: DataType,
    x: Operation,
    shape: &[i64; 4],
    is_training: bool,
    scope: &mut Scope,
) -> Result<BatchNorm, Status> {
    let name = name.as_ref();
    let scale = Variable::builder()
        .data_type(data_type)
        .shape(&[shape[3]])
        .initial_value(broadcast_to(
            constant(&[1f32], scope)?,
            constant(&[shape[3]], scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_scale", name)))?;
    let offset = Variable::builder()
        .data_type(data_type)
        .shape(&[shape[3]])
        .initial_value(broadcast_to(
            constant(&[0f32], scope)?,
            constant(&[shape[3]], scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_offset", name)))?;
    let mean = Variable::builder()
        .data_type(data_type)
        .shape(&[shape[3]])
        .initial_value(broadcast_to(
            constant(&[0f32], scope)?,
            constant(&[shape[3]], scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_mean", name)))?;
    let variance = Variable::builder()
        .data_type(data_type)
        .shape(&[shape[3]])
        .initial_value(broadcast_to(
            constant(&[1f32], scope)?,
            constant(&[shape[3]], scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name(&format!("{}_variance", name)))?;

    if is_training {
        let bn = FusedBatchNormV3::new()
            .data_format("NHWC")
            .is_training(is_training)
            .build(
                x,
                scale.output().clone(),
                offset.output().clone(),
                mean.output().clone(),
                variance.output().clone(),
                &mut scope.with_op_name(&format!("{}_bn", name)),
            )?;
        let output = Identity::new()
            .add_control_input(assign(
                mean.output().clone(),
                bn.output(1),
                &mut scope.with_op_name(&format!("{}_update_mean", name)),
            )?)
            .add_control_input(assign(
                variance.output().clone(),
                bn.output(2),
                &mut scope.with_op_name(&format!("{}_update_variance", name)),
            )?)
            .build(
                bn.output(0),
                &mut scope.with_op_name(&format!("{}_output", name)),
            )?;
        Ok(BatchNorm {
            scale,
            offset,
            mean,
            variance,
            output,
        })
    } else {
        let bn = FusedBatchNormV3::new()
            .data_format("NHWC")
            .is_training(is_training)
            .build(
                x,
                scale.output().clone(),
                offset.output().clone(),
                mean.output().clone(),
                variance.output().clone(),
                &mut scope.with_op_name(&format!("{}_bn", name)),
            )?;
        Ok(BatchNorm {
            scale,
            offset,
            mean,
            variance,
            output: bn,
        })
    }
}

pub fn batch_norm_fc(
    name: impl AsRef<str>,
    data_type: DataType,
    x: Operation,
    channels: i64,
    is_training: bool,
    scope: &mut Scope,
) -> Result<BatchNorm, Status> {
    let name = name.as_ref();
    let before_reshape = reshape(
        x,
        constant(&[-1, 1, 1, channels], scope)?,
        &mut scope.with_op_name(&format!("{}_input_reshape", name)),
    )?;
    let mut bn = batch_norm(
        name,
        data_type,
        before_reshape,
        &[-1, 1, 1, channels],
        is_training,
        scope,
    )?;
    let output = reshape(
        bn.output,
        constant(&[-1, channels], scope)?,
        &mut scope.with_op_name(&format!("{}_output_reshape", name)),
    )?;
    bn.output = output;
    Ok(bn)
}
