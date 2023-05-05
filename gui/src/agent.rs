use alpha_zero::{AgentModel, BoardState, MCTSExecutor};
use atomic_float::AtomicF32;
use environment::{Environment, GameStatus};
use mcts::{State, MCTS};
use parking_lot::RwLock;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::atomic::Ordering;
use tensorflow::{Scope, Session, SessionOptions, SessionRunArgs, Tensor};

pub struct Agent {
    agent: AgentModel,
    mcts_executor: MCTSExecutor,
    session: Session,
    thread_pool: ThreadPool,
}

impl Agent {
    pub fn new(env: Environment) -> Self {
        let mut scope = Scope::new_root_scope();
        let agent = AgentModel::new(&mut scope).unwrap();
        let session = Session::new(&SessionOptions::new(), &scope.graph()).unwrap();
        let mcts = MCTS::<BoardState>::new(BoardState {
            env,
            status: GameStatus::InProgress,
            policy: RwLock::new(
                [1f32 / (Environment::BOARD_SIZE * Environment::BOARD_SIZE) as f32;
                    Environment::BOARD_SIZE * Environment::BOARD_SIZE],
            ),
            z: AtomicF32::new(0f32),
        });

        agent.io.load(&session, "saves/alpha-zero").unwrap();

        Self {
            agent,
            mcts_executor: MCTSExecutor::new(mcts),
            session,
            thread_pool: ThreadPoolBuilder::new().build().unwrap(),
        }
    }

    pub fn compute_probabilities(
        &self,
    ) -> [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE] {
        self.mcts_executor
            .run(
                &self.thread_pool,
                &self.agent,
                &self.session,
                16000 / MCTSExecutor::NN_EVALUATION_BATCH_SIZE,
            )
            .unwrap();

        // Get the policy from the root node. Policy is the visit count of the children.
        let mut policy = {
            let root = self.mcts_executor.mcts.root();
            let children = root.children.read();
            let mut policy = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];

            for child in children.iter() {
                policy[child.action.unwrap()] = child.n.load(Ordering::Relaxed) as f32;
            }

            policy
        };

        // Normalize the policy if the policy is not all zero.
        // This is necessary because the policy is the visit count of the children.
        let sum = policy.iter().sum::<f32>();
        if f32::EPSILON <= sum {
            for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                policy[action] /= sum;
            }
        }

        policy
    }

    pub fn make_move(&self) -> usize {
        let policy = self.compute_probabilities();
        policy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }

    pub fn sync_move(&mut self, env: &Environment, last_action: usize) {
        let has_child = {
            let children = self.mcts_executor.mcts.root().children.read();
            children.iter().any(|node| node.action == Some(last_action))
        };

        if !has_child {
            // Encode the board state.
            let mut board_tensor = Tensor::new(&[
                1,
                Environment::BOARD_SIZE as u64,
                Environment::BOARD_SIZE as u64,
                2,
            ]);
            self.mcts_executor.mcts.root().state.env.encode_board(
                self.mcts_executor.mcts.root().state.env.turn,
                &mut board_tensor[..],
            );

            // Evaluate the NN with the child state to get the policy and value.
            let mut eval_run_args = SessionRunArgs::new();
            eval_run_args.add_feed(&self.agent.op_input, 0, &board_tensor);
            eval_run_args.add_target(&self.agent.op_p_output);
            eval_run_args.add_target(&self.agent.op_v_output);

            let p_fetch_token = eval_run_args.request_fetch(&self.agent.op_p_output, 0);
            let v_fetch_token = eval_run_args.request_fetch(&self.agent.op_v_output, 0);

            self.session.run(&mut eval_run_args).unwrap();

            let p = eval_run_args.fetch::<f32>(p_fetch_token).unwrap();
            let v = eval_run_args.fetch::<f32>(v_fetch_token).unwrap();

            let mut pi = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
            pi.copy_from_slice(&p[..]);

            // Filter out illegal actions.
            pi[last_action] = 0.0;
            for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                if !self
                    .mcts_executor
                    .mcts
                    .root()
                    .state
                    .is_available_action(action)
                {
                    pi[action] = 0.0;
                }
            }

            // Re-normalize the policy if the policy is not all zero.
            let sum = pi.iter().sum::<f32>();
            if f32::EPSILON <= sum {
                for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                    pi[action] /= sum;
                }
            }

            // Make the child node.
            match self.mcts_executor.mcts.expand(
                self.mcts_executor.mcts.root(),
                last_action,
                BoardState {
                    env: env.clone(),
                    status: GameStatus::InProgress,
                    policy: RwLock::new(pi),
                    z: AtomicF32::new(v[0]),
                },
            ) {
                Some(child) => {
                    // Perform backup from the leaf node.
                    child.propagate(v[0]);
                }
                None => {}
            }
        }

        let child_index = {
            let children = self.mcts_executor.mcts.root().children.read();
            children
                .iter()
                .position(|node| node.action == Some(last_action))
                .unwrap()
        };

        self.mcts_executor.mcts.transition(child_index);
    }
}
