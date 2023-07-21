use super::{ModelIO, Network};
use environment::Environment;
use tensorflow::{
    ops::{add, constant, mean, reshape, square, sub, Placeholder},
    train::{AdadeltaOptimizer, MinimizeOptions, Optimizer},
    DataType, Operation, Scope, Session, SessionRunArgs, Status, Tensor, Variable,
};

pub struct AgentModel {
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
    pub io: ModelIO,
}

impl AgentModel {
    pub const LEARNING_RATE: f32 = 0.1;

    pub fn new(scope: &mut Scope) -> Result<Self, Status> {
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
                scope,
            )?,
            scope,
        )?;

        let network = Network::new(
            op_pi_input_flatten,
            scope,
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
                sub(op_z_input.clone(), network.op_v_output.clone(), scope)?,
                scope,
            )?,
            constant(&[0, 1], scope)?,
            &mut scope.with_op_name("v_loss"),
        )?;

        let op_loss = add(
            op_v_loss.clone(),
            network.op_p_loss.clone(),
            &mut scope.with_op_name("loss"),
        )?;

        let mut optimizer = AdadeltaOptimizer::new();
        optimizer.set_learning_rate(constant(Self::LEARNING_RATE, scope)?);

        let (optimizer_vars, op_minimize) = optimizer.minimize(
            scope,
            op_loss.output(0),
            MinimizeOptions::default().with_variables(&network.variables),
        )?;

        let io = ModelIO::new(network.variables.clone(), scope)?;

        let mut variables = Vec::new();
        variables.extend(network.variables);
        variables.extend(optimizer_vars.clone());

        Ok(Self {
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
            io,
        })
    }

    pub fn evaluate_p(&self, session: &Session, input: Tensor<f32>) -> Result<Tensor<f32>, Status> {
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&self.op_input, 0, &input);
        run_args.add_target(&self.op_p_output);

        let p_fetch_token = run_args.request_fetch(&self.op_p_output, 0);
        session.run(&mut run_args)?;

        Ok(run_args.fetch(p_fetch_token)?)
    }

    pub fn evaluate_pv(
        &self,
        session: &Session,
        input: Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>), Status> {
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&self.op_input, 0, &input);
        run_args.add_target(&self.op_p_output);
        run_args.add_target(&self.op_v_output);

        let p_fetch_token = run_args.request_fetch(&self.op_p_output, 0);
        let v_fetch_token = run_args.request_fetch(&self.op_v_output, 0);
        session.run(&mut run_args)?;

        Ok((
            run_args.fetch(p_fetch_token)?,
            run_args.fetch(v_fetch_token)?,
        ))
    }

    pub fn train(
        &self,
        session: &Session,
        input: Tensor<f32>,
        policy_target: Tensor<f32>,
        value_target: Tensor<f32>,
    ) -> Result<(f32, f32, f32), Status> {
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&self.op_input, 0, &input);
        run_args.add_feed(&self.op_pi_input, 0, &policy_target);
        run_args.add_feed(&self.op_z_input, 0, &value_target);
        run_args.add_target(&self.op_minimize);
        session.run(&mut run_args)?;

        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&self.op_input, 0, &input);
        run_args.add_feed(&self.op_pi_input, 0, &policy_target);
        run_args.add_feed(&self.op_z_input, 0, &value_target);
        run_args.add_target(&self.op_p_loss);
        run_args.add_target(&self.op_v_loss);
        run_args.add_target(&self.op_loss);

        let p_loss_fetch_token = run_args.request_fetch(&self.op_p_loss, 0);
        let v_loss_fetch_token = run_args.request_fetch(&self.op_v_loss, 0);
        let loss_fetch_token = run_args.request_fetch(&self.op_loss, 0);
        session.run(&mut run_args)?;

        let p_loss = run_args.fetch::<f32>(p_loss_fetch_token)?;
        let v_loss = run_args.fetch::<f32>(v_loss_fetch_token)?;
        let loss = run_args.fetch::<f32>(loss_fetch_token)?;

        Ok((p_loss[0], v_loss[0], loss[0]))
    }
}

unsafe impl Send for AgentModel {}
unsafe impl Sync for AgentModel {}
