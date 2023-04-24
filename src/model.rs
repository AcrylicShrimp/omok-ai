use tensorflow::{
    ops::{
        bias_add, broadcast_to, constant, leaky_relu, mat_mul, mul, reshape, get_element_at_index, Conv2D, Placeholder,
        RandomStandardNormal,
    },
    DataType, Scope, Status, Variable,
};

use crate::environment::Environment;

pub struct Model {
    pub scope: Scope,
    pub variables: Vec<Variable>,
    pub target_variables: Vec<Variable>,
}

pub fn create_model() -> Result<Model, Status> {
    let mut scope = Scope::new_root_scope();
    let variables = build_graph("input", "output", &mut scope)?;
    let target_variables = build_graph("target_input", "target_output", &mut scope)?;
    Ok(Model {
        scope,
        variables,
        target_variables,
    })
}

fn build_graph(
    input_name: impl AsRef<str>,
    output_name: impl AsRef<str>,
    scope: &mut Scope,
) -> Result<Vec<Variable>, Status> {
    let mut variables = Vec::new();

    let input = Placeholder::new()
        .dtype(DataType::Int64)
        .shape([
            -1i64,
            Environment::BOARD_SIZE as i64,
            Environment::BOARD_SIZE as i64,
            1,
        ])
        .build(&mut scope.with_op_name(input_name.as_ref()))?;

    let weight = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[
            Environment::BOARD_SIZE as i64,
            Environment::BOARD_SIZE as i64,
            3i64,
            3i64,
        ])
        .initial_value(mul(
            RandomStandardNormal::new().dtype(DataType::Float).build(
                constant(
                    &[
                        Environment::BOARD_SIZE as i64,
                        Environment::BOARD_SIZE as i64,
                        3,
                        3,
                    ],
                    scope,
                )?,
                scope,
            )?,
            constant(2f32 / f32::sqrt(3f32 * 3f32 * 3f32), scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name("embedding"))?;
    let weight = reshape(
        weight.output().clone(),
        constant(&[
            Environment::BOARD_SIZE as i64,
            Environment::BOARD_SIZE as i64,
            3i64,
            3i64,
        ], scope)?,
        scope,
    )?;

    let embedding = get_element_at_index(weight, input.clone(), scope)?;

    let filter = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[8i64, 8, 1, 16])
        .initial_value(mul(
            RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(constant(&[8i64, 8, 1, 16], scope)?, scope)?,
            constant(2f32 / f32::sqrt(8f32 * 8f32 * 1f32), scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name("filter_1"))?;
    let filter_bias = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[16i64])
        .initial_value(broadcast_to(
            constant(&[0f32], scope)?,
            constant(&[16i64], scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name("filter_bias_1"))?;
    let conv = Conv2D::new()
        .data_format("NHWC")
        .strides([1i64, 2, 2, 1])
        .padding("VALID")
        .build(
            embedding.clone(),
            filter.output().clone(),
            &mut scope.with_op_name("conv_1"),
        )?;
    let conv = bias_add(
        conv,
        filter_bias.output().clone(),
        &mut scope.with_op_name("conv_bias_1"),
    )?;
    let conv = leaky_relu(conv, &mut scope.with_op_name("conv_activation_1"))?;
    variables.push(filter);
    variables.push(filter_bias);

    let filter = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[4i64, 4, 16, 32])
        .initial_value(mul(
            RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(constant(&[4i64, 4, 16, 32], scope)?, scope)?,
            constant(2f32 / f32::sqrt(4f32 * 4f32 * 16f32), scope)?,
            scope,
        )?)
        .build(scope)?;
    let filter_bias = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[32i64])
        .initial_value(broadcast_to(
            constant(&[0f32], scope)?,
            constant(&[32i64], scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name("filter_bias_2"))?;
    let conv = Conv2D::new()
        .data_format("NHWC")
        .strides([1i64, 1, 1, 1])
        .padding("VALID")
        .build(
            conv,
            filter.output().clone(),
            &mut scope.with_op_name("conv_2"),
        )?;
    let conv = bias_add(
        conv,
        filter_bias.output().clone(),
        &mut scope.with_op_name("conv_bias_2"),
    )?;
    let conv = leaky_relu(conv, &mut scope.with_op_name("conv_activation_2"))?;
    variables.push(filter);
    variables.push(filter_bias);

    let flatten = reshape(
        conv,
        constant(&[-1i64, 288], scope)?,
        &mut scope.with_op_name("reshape"),
    )?;

    let w = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[288i64, 256])
        .initial_value(mul(
            RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(constant(&[288i64, 256], scope)?, scope)?,
            constant(2f32 / f32::sqrt(288f32), scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name("w_1"))?;
    let b = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[256i64])
        .initial_value(broadcast_to(
            constant(&[0f32], scope)?,
            constant(&[256i64], scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name("b_1"))?;
    let mm = bias_add(
        mat_mul(flatten, w.output().clone(), &mut scope.with_op_name("mm_1"))?,
        b.output().clone(),
        &mut scope.with_op_name("mm_bias_1"),
    )?;
    let mm = leaky_relu(mm, &mut scope.with_op_name("mm_activation_1"))?;
    variables.push(w);
    variables.push(b);

    let w = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[
            256i64,
            Environment::BOARD_SIZE as i64 * Environment::BOARD_SIZE as i64,
        ])
        .initial_value(mul(
            RandomStandardNormal::new().dtype(DataType::Float).build(
                constant(
                    &[
                        256i64,
                        Environment::BOARD_SIZE as i64 * Environment::BOARD_SIZE as i64,
                    ],
                    scope,
                )?,
                scope,
            )?,
            constant(2f32 / f32::sqrt(256f32), scope)?,
            scope,
        )?)
        .build(&mut scope.with_op_name("w_2"))?;
    let b = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[Environment::BOARD_SIZE as i64 * Environment::BOARD_SIZE as i64])
        .initial_value(broadcast_to(
            constant(&[0f32], scope)?,
            constant(
                &[Environment::BOARD_SIZE as i64 * Environment::BOARD_SIZE as i64],
                scope,
            )?,
            scope,
        )?)
        .build(&mut scope.with_op_name("b_2"))?;
    let mm = bias_add(
        mat_mul(mm, w.output().clone(), &mut scope.with_op_name("mm_2"))?,
        b.output().clone(),
        &mut scope.with_op_name("mm_bias_2"),
    )?;
    variables.push(w);
    variables.push(b);

    reshape(
        mm,
        constant(
            &[
                -1i64,
                Environment::BOARD_SIZE as i64,
                Environment::BOARD_SIZE as i64,
            ],
            scope,
        )?,
        &mut scope.with_op_name(output_name.as_ref()),
    )?;

    Ok(variables)
}
