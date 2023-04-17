use std::io::Error as IOError;
use tensorflow::{
    ops::{
        bias_add, broadcast_to, constant, leaky_relu, mat_mul, reshape, Conv2D, Placeholder,
        RandomStandardNormal,
    },
    DataType, Operation, Scope, Status, Variable,
};

pub struct Model {
    pub scope: Scope,
    pub input: Operation,
    pub output: Operation,
    pub variables: Vec<Variable>,
}

pub fn create_model() -> Result<Model, Status> {
    let mut scope = Scope::new_root_scope();
    let mut variables = Vec::new();

    let input = Placeholder::new()
        .dtype(DataType::Float)
        .shape([-1i64, 19, 19, 4])
        .build(&mut scope)?;

    let filter = Variable::builder()
        .data_type(DataType::Float)
        .initial_value(
            RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(constant(&[8i64, 8, 4, 16], &mut scope)?, &mut scope)?,
        )
        .build(&mut scope)?;
    let filter_bias = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[8i64 * 8 * 4 * 16])
        .initial_value(broadcast_to(
            constant(&[0f32], &mut scope)?,
            constant(&[8i64 * 8 * 4 * 16], &mut scope)?,
            &mut scope,
        )?)
        .build(&mut scope)?;
    let conv = Conv2D::new()
        .data_format("NHWC")
        .strides([1i64, 2, 2, 1])
        .padding("VALID")
        .build(input.clone(), filter.output().clone(), &mut scope)?;
    let conv = bias_add(conv, filter_bias.output().clone(), &mut scope)?;
    let conv = leaky_relu(conv, &mut scope)?;
    variables.push(filter);
    variables.push(filter_bias);

    let filter = Variable::builder()
        .data_type(DataType::Float)
        .initial_value(
            RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(constant(&[4i64, 4, 16, 32], &mut scope)?, &mut scope)?,
        )
        .build(&mut scope)?;
    let filter_bias = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[4i64 * 4 * 16 * 32])
        .initial_value(broadcast_to(
            constant(&[0f32], &mut scope)?,
            constant(&[4i64 * 4 * 16 * 32], &mut scope)?,
            &mut scope,
        )?)
        .build(&mut scope)?;
    let conv = Conv2D::new()
        .data_format("NHWC")
        .strides([1i64, 1, 1, 1])
        .padding("VALID")
        .build(conv, filter.output().clone(), &mut scope)?;
    let conv = bias_add(conv, filter_bias.output().clone(), &mut scope)?;
    let conv = leaky_relu(conv, &mut scope)?;
    variables.push(filter);
    variables.push(filter_bias);

    let flatten = reshape(conv, constant(&[-1i64, 288], &mut scope)?, &mut scope)?;

    let w = Variable::builder()
        .data_type(DataType::Float)
        .initial_value(
            RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(constant(&[288i64, 256], &mut scope)?, &mut scope)?,
        )
        .build(&mut scope)?;
    let b = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[256i64])
        .initial_value(broadcast_to(
            constant(&[0f32], &mut scope)?,
            constant(&[256i64], &mut scope)?,
            &mut scope,
        )?)
        .build(&mut scope)?;
    let mm = bias_add(
        mat_mul(flatten, w.output().clone(), &mut scope)?,
        b.output().clone(),
        &mut scope,
    )?;
    let mm = leaky_relu(mm, &mut scope)?;
    variables.push(w);
    variables.push(b);

    let w = Variable::builder()
        .data_type(DataType::Float)
        .initial_value(
            RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(constant(&[256i64, 361], &mut scope)?, &mut scope)?,
        )
        .build(&mut scope)?;
    let b = Variable::builder()
        .data_type(DataType::Float)
        .shape(&[361i64])
        .initial_value(broadcast_to(
            constant(&[0f32], &mut scope)?,
            constant(&[361i64], &mut scope)?,
            &mut scope,
        )?)
        .build(&mut scope)?;
    let mm = bias_add(
        mat_mul(mm, w.output().clone(), &mut scope)?,
        b.output().clone(),
        &mut scope,
    )?;
    let mm = leaky_relu(mm, &mut scope)?;
    variables.push(w);
    variables.push(b);

    let output = reshape(mm, constant(&[-1i64, 19, 19], &mut scope)?, &mut scope)?;

    Ok(Model {
        scope,
        input,
        output,
        variables,
    })
}

// pub fn load_graph(path: impl AsRef<Path>) -> IOError<Graph> {}
