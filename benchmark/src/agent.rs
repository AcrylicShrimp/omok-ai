use std::path::Path;

use alpha_zero::{ActionSamplingMode, AgentModel, MCTSExecutor};
use tensorflow::{Scope, Session, SessionOptions};

pub struct Agent {
    pub agent: alpha_zero::Agent,
    pub agent_model: AgentModel,
    pub session: Session,
    pub mcts_executor: MCTSExecutor,
}

impl Agent {
    pub const EPSILON: f32 = 0.0;
    pub const ALPHA: f32 = 1.0;

    pub fn new(path: impl AsRef<Path>) -> Self {
        let mut scope = Scope::new_root_scope();
        let agent_model = AgentModel::new(&mut scope).unwrap();
        let session = Session::new(&SessionOptions::new(), &scope.graph()).unwrap();

        agent_model.io.load(&session, path).unwrap();

        let agent = alpha_zero::Agent::new(&agent_model, &session).unwrap();

        Self {
            agent_model,
            session,
            agent,
            mcts_executor: MCTSExecutor::new(),
        }
    }

    pub fn make_move(&self, mcts_count: usize, mcts_batch_size: usize) -> usize {
        self.mcts_executor
            .run(
                mcts_count,
                mcts_batch_size,
                Self::EPSILON,
                Self::ALPHA,
                &self.agent_model,
                &self.session,
                &self.agent,
            )
            .unwrap();
        self.agent
            .sample_action(ActionSamplingMode::Best)
            .unwrap()
            .0
    }

    pub fn reset(&mut self) {
        self.agent = alpha_zero::Agent::new(&self.agent_model, &self.session).unwrap();
    }
}
