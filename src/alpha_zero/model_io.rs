use bincode::{deserialize_from, serialize_into};
use serde::{Deserialize, Serialize};
use std::{fs::File, path::Path};
use tensorflow::{
    ops::{assign, NoOp, Placeholder},
    Operation, Scope, Session, SessionRunArgs, Status, Tensor, Variable,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelIOError {
    #[error("Tensorflow error: {0}")]
    Tensorflow(#[from] Status),
    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),
    #[error("Bincode error: {0}")]
    Bincode(#[from] bincode::Error),
}

#[derive(Serialize, Deserialize)]
pub struct SavedData {
    pub variable_names: Vec<String>,
    pub parameters: Vec<Vec<f32>>,
}

pub struct ModelIO {
    pub variables: Vec<Variable>,
    pub op_variable_inputs: Vec<Operation>,
    pub op_load_variables: Operation,
}

impl ModelIO {
    pub fn new(variables: Vec<Variable>, scope: &mut Scope) -> Result<Self, Status> {
        let mut op_variable_inputs = Vec::with_capacity(variables.len());
        let mut op_load_variables = NoOp::new();

        for variable in &variables {
            let op_variable_input = Placeholder::new()
                .dtype(variable.data_type())
                .shape(variable.shape().clone())
                .build(scope)?;
            op_variable_inputs.push(op_variable_input.clone());

            let op_load_variable =
                assign(variable.output().clone(), op_variable_input.clone(), scope)?;
            op_load_variables = op_load_variables.add_control_input(op_load_variable);
        }

        let op_load_variables = op_load_variables.build(scope)?;

        Ok(Self {
            variables,
            op_variable_inputs,
            op_load_variables,
        })
    }

    /// Save variables with bincode + serde to a single file.
    pub fn save(&self, session: &Session, path: impl AsRef<Path>) -> Result<(), ModelIOError> {
        let mut save_run_args = SessionRunArgs::new();
        let mut fetch_tokens = Vec::with_capacity(self.variables.len());

        for variable in &self.variables {
            fetch_tokens.push(save_run_args.request_fetch(&variable.output().operation, 0));
            save_run_args.add_target(&variable.output().operation);
        }

        session.run(&mut save_run_args)?;

        let mut parameters = Vec::with_capacity(self.variables.len());

        for fetch_token in fetch_tokens {
            let variable_input = save_run_args.fetch::<f32>(fetch_token)?;
            parameters.push(variable_input.to_vec());
        }

        let saved_data = SavedData {
            variable_names: self
                .variables
                .iter()
                .map(|variable| variable.name().to_string())
                .collect(),
            parameters,
        };

        let file = File::create(path)?;
        serialize_into(file, &saved_data)?;

        Ok(())
    }

    pub fn load(&self, session: &Session, path: impl AsRef<Path>) -> Result<(), ModelIOError> {
        let file = File::open(path)?;
        let saved_data: SavedData = deserialize_from(file)?;

        let mut tensor_inputs = Vec::with_capacity(self.variables.len());

        for (variable, parameter) in self.variables.iter().zip(saved_data.parameters) {
            let shape = Option::<Vec<Option<i64>>>::from(variable.shape().clone())
                .unwrap()
                .iter()
                .map(|x| x.unwrap() as u64)
                .collect::<Vec<_>>();
            let mut tensor_input = Tensor::new(&shape);
            tensor_input.copy_from_slice(&parameter);
            tensor_inputs.push(tensor_input);
        }

        let mut load_run_args = SessionRunArgs::new();
        load_run_args.add_target(&self.op_load_variables);

        for (op_variable_input, tensor_input) in self.op_variable_inputs.iter().zip(&tensor_inputs)
        {
            load_run_args.add_feed(op_variable_input, 0, tensor_input);
        }

        session.run(&mut load_run_args)?;

        Ok(())
    }
}
