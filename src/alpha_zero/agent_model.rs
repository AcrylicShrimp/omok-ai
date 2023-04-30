use super::Network;
use environment::Environment;
use std::path::Path;
use tensorflow::{
    ops::{add, constant, mean, reshape, square, sub, Placeholder},
    train::{AdadeltaOptimizer, MinimizeOptions, Optimizer},
    DataType, Operation, SaveModelError, SavedModelBuilder, SavedModelSaver, Scope, Session,
    Status, Variable,
};

pub struct AgentModel {
    pub scope: Scope,
    pub op_input: Operation,
    pub op_v_output: Operation,
    pub op_p_output: Operation,
    pub op_z_input: Operation,
    pub op_v_loss: Operation,
    pub op_pi_input: Operation,
    pub op_p_loss: Operation,
    pub op_loss: Operation,
    pub op_minimize: Operation,
    pub variables: Vec<Variable>,
    pub saver: SavedModelSaver,
}

impl AgentModel {
    pub const LEARNING_RATE: f32 = 0.01;

    pub fn new(mut scope: Scope) -> Result<Self, Status> {
        let op_pi_input = Placeholder::new()
            .dtype(DataType::Float)
            .shape([
                -1,
                Environment::BOARD_SIZE as i64,
                Environment::BOARD_SIZE as i64,
            ])
            .build(&mut scope.with_op_name("pi_input"))?;
        let op_pi_input_flatten = reshape(
            op_pi_input.clone(),
            constant(
                &[
                    -1,
                    Environment::BOARD_SIZE as i64 * Environment::BOARD_SIZE as i64,
                ],
                &mut scope,
            )?,
            &mut scope,
        )?;

        let network = Network::new(
            op_pi_input_flatten,
            &mut scope,
            "input",
            "v_output",
            "p_output",
            "p_loss",
        )?;

        let op_z_input = Placeholder::new()
            .dtype(DataType::Float)
            .shape([-1i64, 1])
            .build(&mut scope.with_op_name("z_input"))?;
        let op_v_loss = mean(
            square(
                sub(op_z_input.clone(), network.op_v_output.clone(), &mut scope)?,
                &mut scope,
            )?,
            constant(&[0], &mut scope)?,
            &mut scope,
        )?;

        let op_loss = add(
            op_v_loss.clone(),
            network.op_p_loss.clone(),
            &mut scope.with_op_name("loss"),
        )?;

        let mut optimizer = AdadeltaOptimizer::new();
        optimizer.set_learning_rate(constant(Self::LEARNING_RATE, &mut scope)?);

        let (optimizer_vars, op_minimize) = optimizer.minimize(
            &mut scope,
            op_loss.output(0),
            MinimizeOptions::default().with_variables(&network.variables),
        )?;

        let mut variables = Vec::new();
        variables.extend(network.variables.clone());
        variables.extend(optimizer_vars.clone());

        let mut saver_builder = SavedModelBuilder::new();
        saver_builder
            .add_collection("network_variables", &network.variables)
            .add_collection("optimizer_variables", &optimizer_vars)
            .add_tag("serve")
            .add_tag("train");
        let saver = saver_builder.inject(&mut scope)?;

        Ok(Self {
            scope,
            op_input: network.op_input,
            op_v_output: network.op_v_output,
            op_p_output: network.op_p_output,
            op_z_input,
            op_v_loss,
            op_pi_input,
            op_p_loss: network.op_p_loss,
            op_loss,
            op_minimize,
            variables,
            saver,
        })
    }

    pub fn save(&self, session: &Session, path: impl AsRef<Path>) -> Result<(), SaveModelError> {
        self.saver.save(session, &self.scope.graph(), path)
    }
}
