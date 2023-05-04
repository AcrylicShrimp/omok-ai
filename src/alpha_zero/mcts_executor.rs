use super::{mcts_node::BoardState, AgentModel};
use atomic_float::AtomicF32;
use bitvec::vec::BitVec;
use environment::Environment;
use mcts::{Node, NodePtr, State, MCTS};
use parking_lot::RwLock;
use rand::{seq::SliceRandom, thread_rng};
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    ThreadPool,
};
use std::{
    sync::{atomic::Ordering, mpsc::sync_channel},
    thread,
};
use tensorflow::{Session, SessionRunArgs, Status, Tensor};

pub struct MCTSExecutor {
    pub mcts: MCTS<BoardState>,
}

impl MCTSExecutor {
    pub const NN_EVALUATION_BATCH_SIZE: usize = 128;
    pub const NN_EVALUATION_BACKLOG_SIZE: usize = 256;

    pub const C_PUCT: f32 = 1.0;
    pub const V_LOSS: f32 = 0.5f32;

    pub fn new(mcts: MCTS<BoardState>) -> Self {
        Self { mcts }
    }

    pub fn run(
        &self,
        thread_pool: &ThreadPool,
        agent: &AgentModel,
        session: &Session,
        count: usize,
    ) -> Result<(), Status> {
        let (tx, rx) = sync_channel(Self::NN_EVALUATION_BACKLOG_SIZE);

        thread::scope(|s| {
            let mcts_execution_handle = s.spawn(move || {
                thread_pool.install(move || {
                    let tx = tx;
                    (0..count)
                        .into_par_iter()
                        .try_for_each(|_| -> Result<(), Status> {
                            let node = self.mcts.select_leaf(|parent, children| {
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
                                node.propagate(node.state.z.load(Ordering::Relaxed));
                                node.v_loss.fetch_sub(1, Ordering::Relaxed);
                                return Ok(());
                            }

                            // Select any possible action.
                            // Since the leaf node doesn't have terminal state, we need to expand it.
                            let mut rng = thread_rng();
                            let action = {
                                let mut bits = BitVec::<usize>::repeat(
                                    false,
                                    Environment::BOARD_SIZE * Environment::BOARD_SIZE,
                                );

                                for children in node.children.read().iter() {
                                    bits.set(children.action.unwrap(), true);
                                }

                                let available_actions = (0..Environment::BOARD_SIZE
                                    * Environment::BOARD_SIZE)
                                    .filter(|&action| {
                                        node.state.is_available_action(action) && !bits[action]
                                    })
                                    .collect::<Vec<_>>();
                                available_actions.choose(&mut rng).cloned()
                            };
                            let action = if let Some(action) = action {
                                action
                            } else {
                                // There's no action for now.
                                // Note that this not means the game is over.
                                return Ok(());
                            };

                            // Place the stone.
                            let mut env = node.state.env.clone();
                            let status = env.place_stone(action).unwrap();

                            // Encode the board state.
                            let mut board_tensor =
                                [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2];
                            env.encode_board(env.turn, &mut board_tensor[..]);

                            // Pre-expand the node.
                            let expanded_child = match self.mcts.expand(
                                node,
                                action,
                                f32::MIN,
                                u64::MAX - 1,
                                BoardState {
                                    env,
                                    status,
                                    policy: RwLock::new(
                                        [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE],
                                    ),
                                    z: AtomicF32::new(0f32),
                                },
                            ) {
                                Some(child) => {
                                    node.v_loss.fetch_sub(1, Ordering::Relaxed);
                                    child
                                }
                                None => {
                                    // The node is already expanded by other thread.
                                    // We don't need to expand it again.
                                    node.v_loss.fetch_sub(1, Ordering::Relaxed);
                                    return Ok(());
                                }
                            };

                            // Request NN evaluation.
                            tx.send(NNEvalRequest {
                                node: expanded_child,
                                input: board_tensor,
                                terminal_reward: if status.is_terminal() {
                                    Some(1f32)
                                } else {
                                    None
                                },
                            })
                            .unwrap();

                            Ok(())
                        })
                })
            });
            let nn_evaluation_handle = s.spawn(move || -> Result<(), Status> {
                let mut input_tensor = Tensor::new(&[
                    Self::NN_EVALUATION_BATCH_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    2,
                ]);
                let mut requests = Vec::with_capacity(Self::NN_EVALUATION_BATCH_SIZE);

                loop {
                    let request = match rx.recv() {
                        Ok(request) => request,
                        Err(_) => {
                            // The sender is dropped.
                            // We don't need to evaluate the NN anymore.
                            break;
                        }
                    };
                    requests.push(request);

                    if requests.len() < Self::NN_EVALUATION_BATCH_SIZE {
                        continue;
                    }

                    for (batch_index, request) in requests.iter().enumerate() {
                        input_tensor[batch_index
                            * (Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2)
                            ..(batch_index + 1)
                                * (Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2)]
                            .copy_from_slice(&request.input[..]);
                    }

                    let mut eval_run_args = SessionRunArgs::new();
                    eval_run_args.add_feed(&agent.op_input, 0, &input_tensor);
                    eval_run_args.add_target(&agent.op_p_output);
                    eval_run_args.add_target(&agent.op_v_output);

                    let p_fetch_token = eval_run_args.request_fetch(&agent.op_p_output, 0);
                    let v_fetch_token = eval_run_args.request_fetch(&agent.op_v_output, 0);

                    session.run(&mut eval_run_args)?;

                    let p = eval_run_args.fetch::<f32>(p_fetch_token)?;
                    let v = eval_run_args.fetch::<f32>(v_fetch_token)?;

                    for (batch_index, request) in requests.iter().enumerate() {
                        let node = &*request.node;
                        let p = &p[batch_index * (Environment::BOARD_SIZE * Environment::BOARD_SIZE)
                            ..(batch_index + 1)
                                * (Environment::BOARD_SIZE * Environment::BOARD_SIZE)];
                        let v = v[batch_index];
                        let reward = request.terminal_reward.unwrap_or_else(|| v);

                        // Update the pre-expanded child node.
                        node.state.policy.write().copy_from_slice(p);
                        node.state.z.store(reward, Ordering::Relaxed);

                        // Reset the node.
                        node.n.store(0, Ordering::Relaxed);
                        node.w.store(0f32, Ordering::Relaxed);

                        // Perform backup from the expanded child node.
                        node.propagate(reward);
                    }

                    requests.clear();
                }

                if !requests.is_empty() {
                    let mut input_tensor = Tensor::new(&[
                        requests.len() as u64,
                        Environment::BOARD_SIZE as u64,
                        Environment::BOARD_SIZE as u64,
                        2,
                    ]);

                    for (batch_index, request) in requests.iter().enumerate() {
                        input_tensor[batch_index
                            * (Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2)
                            ..(batch_index + 1)
                                * (Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2)]
                            .copy_from_slice(&request.input[..]);
                    }

                    let mut eval_run_args = SessionRunArgs::new();
                    eval_run_args.add_feed(&agent.op_input, 0, &input_tensor);
                    eval_run_args.add_target(&agent.op_p_output);
                    eval_run_args.add_target(&agent.op_v_output);

                    let p_fetch_token = eval_run_args.request_fetch(&agent.op_p_output, 0);
                    let v_fetch_token = eval_run_args.request_fetch(&agent.op_v_output, 0);

                    session.run(&mut eval_run_args)?;

                    let p = eval_run_args.fetch::<f32>(p_fetch_token)?;
                    let v = eval_run_args.fetch::<f32>(v_fetch_token)?;

                    for (batch_index, request) in requests.iter().enumerate() {
                        let node = &*request.node;
                        let p = &p[batch_index * (Environment::BOARD_SIZE * Environment::BOARD_SIZE)
                            ..(batch_index + 1)
                                * (Environment::BOARD_SIZE * Environment::BOARD_SIZE)];
                        let v = v[batch_index];
                        let reward = request.terminal_reward.unwrap_or_else(|| v);

                        // Update the pre-expanded child node.
                        node.state.policy.write().copy_from_slice(p);
                        node.state.z.store(reward, Ordering::Relaxed);

                        // Reset the node.
                        node.n.store(0, Ordering::Relaxed);
                        node.w.store(0f32, Ordering::Relaxed);

                        // Perform backup from the expanded child node.
                        node.propagate(reward);
                    }
                }

                Ok(())
            });

            mcts_execution_handle.join().unwrap()?;
            nn_evaluation_handle.join().unwrap()?;

            Ok(())
        })
    }
}

struct NNEvalRequest {
    pub node: NodePtr<BoardState>,
    pub input: [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2],
    pub terminal_reward: Option<f32>,
}

fn compute_ucb_1<S>(parent_n: u64, node: &Node<S>, c: f32) -> f32
where
    S: State,
{
    let n = node.n.load(Ordering::Relaxed);
    let q_s_a = node.w.load(Ordering::Relaxed) as f32 / (n as f32 + f32::EPSILON);
    let p_s_a = node.p;
    let bias = f32::sqrt(parent_n as f32) / (1 + n) as f32;
    q_s_a + c * p_s_a * bias - node.v_loss.load(Ordering::Relaxed) as f32 * MCTSExecutor::V_LOSS
}
