mod agent_model;
mod network;

pub use agent_model::*;
pub use network::*;

use environment::{compute_state, Environment, GameStatus, Turn};
use mcts::{Node, Policy, State, MCTS};
use rand::{distributions::Uniform, thread_rng, Rng};
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    ThreadPoolBuilder,
};
use std::{collections::VecDeque, sync::atomic::Ordering};
use tensorflow::{Scope, Session, SessionOptions, SessionRunArgs, Status, Tensor};

#[derive(Clone)]
pub struct BoardState {
    pub board: [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE],
    pub policy: BoardPolicy,
    pub reward: f32,
    pub is_terminal: bool,
    pub available_actions_len: usize,
}

impl State for BoardState {
    type Policy = BoardPolicy;

    fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn available_actions_len(&self) -> usize {
        self.available_actions_len
    }

    fn is_available_action(&self, action: usize) -> bool {
        f32::abs(self.board[action]) < f32::EPSILON
    }
}

#[derive(Clone)]
pub struct BoardPolicy {
    pub pi: [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE],
}

impl Policy for BoardPolicy {
    fn get(&self, action: usize) -> f32 {
        self.pi[action]
    }
}

pub struct Transition {
    pub board: [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE],
    pub policy: [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE],
    pub z: f32,
}

pub struct Train {
    pub session: Session,
    pub agent: AgentModel,
    pub replay_memory: VecDeque<Transition>,
}

impl Train {
    pub const EPISODE_COUNT: usize = 100;
    pub const EVALUATE_COUNT: usize = 300;
    pub const C_PUCT: f32 = 1.0;

    pub fn new() -> Result<Self, Status> {
        let agent = AgentModel::new(Scope::new_root_scope())?;
        let session = Session::new(&SessionOptions::new(), &agent.scope.graph())?;

        let mut init_run_args = SessionRunArgs::new();

        for variable in &agent.variables {
            init_run_args.add_target(&variable.initializer());
        }

        session.run(&mut init_run_args)?;

        Ok(Self {
            session,
            agent: AgentModel::new(Scope::new_root_scope())?,
            replay_memory: VecDeque::with_capacity(5_000),
        })
    }

    pub fn train(&mut self, iteration_count: usize) -> Result<(), Status> {
        let thread_pool = ThreadPoolBuilder::new().build().unwrap();

        for _ in 0..iteration_count {
            for _ in 0..Self::EPISODE_COUNT {
                let mut env = Environment::new();
                let mut mcts = {
                    let state = BoardState {
                        board: [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE],
                        policy: BoardPolicy {
                            pi: [1f32 / (Environment::BOARD_SIZE * Environment::BOARD_SIZE) as f32;
                                Environment::BOARD_SIZE * Environment::BOARD_SIZE],
                        },
                        reward: 0f32,
                        is_terminal: false,
                        available_actions_len: env.legal_move_count,
                    };
                    MCTS::<BoardState>::new(state)
                };

                loop {
                    let mut board = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                    env.copy_board(env.turn, &mut board);

                    thread_pool.install(|| -> Result<(), Status> {
                        // Run MCTS to get the best action.
                        (0..Self::EVALUATE_COUNT).into_par_iter().try_for_each(
                            |_| -> Result<(), Status> {
                                let node = mcts.select_leaf(|parent, children| {
                                    let parent_n = parent.n.load(Ordering::Relaxed);
                                    children
                                        .iter()
                                        .map(|child| compute_ucb_1(parent_n, child, Self::C_PUCT))
                                        .enumerate()
                                        .max_by(|(_, a), (_, b)| f32::total_cmp(a, b))
                                        .unwrap()
                                        .0
                                });

                                if node.state.is_terminal() {
                                    // If the leaf node is terminal state, we don't need to expand it.
                                    // Instead we perform backup from the leaf node.
                                    node.propagate(node.state.reward);
                                    return Ok(());
                                }

                                // Since the leaf node doesn't have terminal state, we need to expand it.
                                // Select any possible action. It is safe to call unwrap() because
                                // non-terminal state always has at least one possible action.
                                let rng = thread_rng();
                                let action = rng
                                    .sample_iter(Uniform::new(
                                        0,
                                        Environment::BOARD_SIZE * Environment::BOARD_SIZE,
                                    ))
                                    .filter(|&action| node.state.is_available_action(action))
                                    .next();
                                let action = action.unwrap();

                                // Make board state for the action.
                                let mut board_tensor = Tensor::new(&[
                                    1,
                                    Environment::BOARD_SIZE as u64,
                                    Environment::BOARD_SIZE as u64,
                                    1,
                                ]);
                                board_tensor.copy_from_slice(&node.state.board);
                                board_tensor[action] = 1.0;

                                let result = compute_state(
                                    &board_tensor[..].try_into().unwrap(),
                                    action,
                                    node.state.available_actions_len - 1,
                                );

                                // Flip the board. Child node always has the opposite turn of the parent.
                                for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                                    board_tensor[action] *= -1.0;
                                }

                                // Evaluate the NN with the child state to get the policy and value.
                                let mut eval_run_args = SessionRunArgs::new();
                                eval_run_args.add_feed(&self.agent.op_input, 0, &board_tensor);
                                eval_run_args.add_target(&self.agent.op_p_output);
                                eval_run_args.add_target(&self.agent.op_v_output);

                                let p_fetch_token =
                                    eval_run_args.request_fetch(&self.agent.op_p_output, 0);
                                let v_fetch_token =
                                    eval_run_args.request_fetch(&self.agent.op_v_output, 0);

                                self.session.run(&mut eval_run_args)?;

                                let p = eval_run_args.fetch::<f32>(p_fetch_token)?;
                                let v = eval_run_args.fetch::<f32>(v_fetch_token)?;

                                let mut pi =
                                    [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                                pi.copy_from_slice(&p[..]);

                                // Filter out illegal actions.
                                for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                                    if !node.state.is_available_action(action) {
                                        pi[action] = 0.0;
                                    }
                                }

                                // Re-normalize the policy if the policy is not all zero.
                                let sum = pi.iter().sum::<f32>();
                                if f32::EPSILON <= sum {
                                    for action in
                                        0..Environment::BOARD_SIZE * Environment::BOARD_SIZE
                                    {
                                        pi[action] /= sum;
                                    }
                                }

                                let mut board =
                                    [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                                board.copy_from_slice(&board_tensor[..]);

                                // Make the child node.
                                let reward = if let Some(reward) = result {
                                    // If the child node is terminal state, we need to
                                    // flip the sign of the reward. It is because the
                                    // child node has the opposite turn of the parent.
                                    -reward
                                } else {
                                    // Since the value is from side of the child node,
                                    // we don't need to flip the sign of the reward.
                                    v[0]
                                };

                                match mcts.expand(
                                    node,
                                    action,
                                    BoardState {
                                        board,
                                        policy: BoardPolicy { pi },
                                        reward,
                                        is_terminal: result.is_some(),
                                        available_actions_len: if result.is_some() {
                                            0
                                        } else {
                                            node.state.available_actions_len - 1
                                        },
                                    },
                                ) {
                                    Some(child) => {
                                        // Perform backup from the leaf node.
                                        child.propagate(reward);
                                    }
                                    None => {
                                        // If the child node is already expanded by other thread,
                                        // silently ignore it.
                                    }
                                }

                                Ok(())
                            },
                        )
                    })?;

                    // Get the policy from the root node. Policy is the visit count of the children.
                    let policy = {
                        let root = mcts.root();
                        let children = root.children.read();
                        let mut policy = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];

                        for child in children.iter() {
                            policy[child.action.unwrap()] = child.n.load(Ordering::Relaxed) as f32;
                        }

                        // Re-normalize the policy if the policy is not all zero.
                        let sum = policy.iter().sum::<f32>();
                        if f32::EPSILON <= sum {
                            for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                                policy[action] /= sum;
                            }
                        }

                        policy
                    };

                    // Select the best action; it is the most visited child.
                    let (children_index, best_action) = {
                        let children = mcts.root().children.read();
                        let (index, node) = children
                            .iter()
                            .enumerate()
                            .max_by(|&(_, a), &(_, b)| {
                                a.n.load(Ordering::Relaxed)
                                    .cmp(&b.n.load(Ordering::Relaxed))
                            })
                            .unwrap();
                        (index, node.action.unwrap())
                    };

                    // Play the action.
                    let (z, is_terminal) = match env.place_stone(best_action) {
                        GameStatus::InProgress => (0f32, false),
                        GameStatus::Draw => (0f32, true),
                        GameStatus::BlackWin => {
                            (if env.turn == Turn::Black { 1f32 } else { -1f32 }, true)
                        }
                        GameStatus::WhiteWin => {
                            (if env.turn == Turn::White { 1f32 } else { -1f32 }, true)
                        }
                    };

                    self.replay_memory.push_back(Transition {
                        board: env.board,
                        policy,
                        z,
                    });

                    if is_terminal {
                        break;
                    }

                    // The game is continued. Re-root the tree.
                    mcts.transition(children_index);
                }
            }

            println!("Self-play finished; training the agent...");
        }

        Ok(())
    }
}

fn compute_ucb_1(parent_n: u64, node: &Node<BoardState>, c: f32) -> f32 {
    let n = node.n.load(Ordering::Relaxed);
    let q_s_a = node.w.load(Ordering::Relaxed) as f32 / (n as f32 + f32::EPSILON);
    let p_s_a = node.p;
    let bias = f32::sqrt(parent_n as f32) / (1 + n) as f32;
    q_s_a + c * p_s_a * bias - node.v_loss.load(Ordering::Relaxed) as f32 * 0.1f32
}
