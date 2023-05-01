mod agent_model;
mod network;

pub use agent_model::*;
pub use network::*;

use environment::{compute_state, Environment, GameStatus, Turn};
use mcts::{Node, Policy, State, MCTS};
use rand::{
    seq::{IteratorRandom, SliceRandom},
    thread_rng, Rng,
};
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    ThreadPool, ThreadPoolBuilder,
};
use std::{
    collections::VecDeque,
    fs::{create_dir_all, remove_dir_all},
    path::Path,
    sync::atomic::Ordering,
};
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
    pub const REPLAY_MEMORY_SIZE: usize = 5_000;
    pub const EPISODE_COUNT: usize = 100;
    pub const EVALUATE_COUNT: usize = 300;
    pub const TRAINING_COUNT: usize = 100;
    pub const BATCH_SIZE: usize = 32;
    pub const C_PUCT: f32 = 1.0;
    pub const V_LOSS: f32 = 1f32;

    pub const TEST_EVALUATE_COUNT: usize = 300;

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
            agent,
            replay_memory: VecDeque::with_capacity(Self::REPLAY_MEMORY_SIZE),
        })
    }

    pub fn train(&mut self, iteration_count: usize) -> Result<(), Status> {
        let thread_pool = ThreadPoolBuilder::new().build().unwrap();
        let mut rng = thread_rng();
        let mut recent_losses = VecDeque::with_capacity(100);

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

                                // Select any possible action.
                                // Since the leaf node doesn't have terminal state, we need to expand it.
                                let mut rng = thread_rng();
                                let action = {
                                    let children = node.children.read();
                                    let available_actions = (0..Environment::BOARD_SIZE
                                        * Environment::BOARD_SIZE)
                                    .filter(|&action| node.state.is_available_action(action))
                                        .filter(|&action| {
                                            children
                                                .iter()
                                                .all(|child| child.action != Some(action))
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

                                // Make board state for the action; first, copy the board state.
                                let mut board_tensor = Tensor::new(&[
                                    1,
                                    Environment::BOARD_SIZE as u64,
                                    Environment::BOARD_SIZE as u64,
                                    1,
                                ]);
                                board_tensor.copy_from_slice(&node.state.board);

                                // Flip the board. Child node always has the opposite turn of the parent.
                                for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                                    board_tensor[action] *= -1.0;
                                }

                                // Mark the action as taken.
                                board_tensor[action] = 1.0;

                                // Check if the game is over.
                                let result = compute_state(
                                    &board_tensor[..].try_into().unwrap(),
                                    action,
                                    node.state.available_actions_len - 1,
                                );

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
                                pi[action] = 0.0;
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

                                let reward = result.unwrap_or_else(|| v[0]);

                                // Make the child node.
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

                    if self.replay_memory.len() == Self::REPLAY_MEMORY_SIZE {
                        self.replay_memory.pop_front();
                    }

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

            for _ in 0..Self::TRAINING_COUNT {
                let transition = self
                    .replay_memory
                    .iter()
                    .choose_multiple(&mut rng, Self::BATCH_SIZE);

                debug_assert!(transition.len() == Self::BATCH_SIZE);

                let mut tensor_input = Tensor::new(&[
                    Self::BATCH_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    1,
                ]);
                let mut tensor_z_input = Tensor::new(&[Self::BATCH_SIZE as u64, 1]);
                let mut tensor_pi_input = Tensor::new(&[
                    Self::BATCH_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                ]);

                for batch_index in 0..Self::BATCH_SIZE {
                    let transition = transition[batch_index];

                    tensor_input[batch_index * Environment::BOARD_SIZE * Environment::BOARD_SIZE
                        ..(batch_index + 1) * Environment::BOARD_SIZE * Environment::BOARD_SIZE]
                        .copy_from_slice(&transition.board[..]);
                    tensor_z_input[batch_index] = transition.z;
                    tensor_pi_input[batch_index * Environment::BOARD_SIZE * Environment::BOARD_SIZE
                        ..(batch_index + 1) * Environment::BOARD_SIZE * Environment::BOARD_SIZE]
                        .copy_from_slice(&transition.policy[..]);
                }

                let mut train_run_args = SessionRunArgs::new();
                train_run_args.add_feed(&self.agent.op_input, 0, &tensor_input);
                train_run_args.add_feed(&self.agent.op_z_input, 0, &tensor_z_input);
                train_run_args.add_feed(&self.agent.op_pi_input, 0, &tensor_pi_input);
                train_run_args.add_target(&self.agent.op_minimize);
                self.session.run(&mut train_run_args)?;

                let mut loss_run_args = SessionRunArgs::new();
                loss_run_args.add_feed(&self.agent.op_input, 0, &tensor_input);
                loss_run_args.add_feed(&self.agent.op_z_input, 0, &tensor_z_input);
                loss_run_args.add_feed(&self.agent.op_pi_input, 0, &tensor_pi_input);
                loss_run_args.add_target(&self.agent.op_loss);

                let fetch_token = loss_run_args.request_fetch(&self.agent.op_loss, 0);
                self.session.run(&mut loss_run_args)?;

                let loss = loss_run_args.fetch::<f32>(fetch_token)?;

                if recent_losses.len() == 100 {
                    recent_losses.pop_front();
                }

                recent_losses.push_back(loss[0]);
            }

            println!(
                "Loss: {}",
                recent_losses.iter().sum::<f32>() / recent_losses.len() as f32
            );

            let mut win = 0;
            let mut lose = 0;
            let mut draw = 0;

            for _ in 0..100 {
                let result = self.play_against_random_player(&thread_pool)?;
                if result == 1 {
                    win += 1;
                } else if result == -1 {
                    lose += 1;
                } else {
                    draw += 1;
                }
            }

            println!(
                "[Playing against random move player] Win: {}, Lose: {}, Draw: {}",
                win, lose, draw
            );

            self.save("alpha-zero");
            println!("Model saved.");
        }

        Ok(())
    }

    fn play_against_random_player(&self, thread_pool: &ThreadPool) -> Result<i32, Status> {
        let mut rng = thread_rng();
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
            thread_pool.install(|| -> Result<(), Status> {
                // Run MCTS to get the best action.
                (0..Self::TEST_EVALUATE_COUNT).into_par_iter().try_for_each(
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

                        // Select any possible action.
                        // Since the leaf node doesn't have terminal state, we need to expand it.
                        let mut rng = thread_rng();
                        let action = {
                            let children = node.children.read();
                            let available_actions = (0..Environment::BOARD_SIZE
                                * Environment::BOARD_SIZE)
                            .filter(|&action| node.state.is_available_action(action))
                                .filter(|&action| {
                                    children.iter().all(|child| child.action != Some(action))
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

                        // Make board state for the action; first, copy the board state.
                        let mut board_tensor = Tensor::new(&[
                            1,
                            Environment::BOARD_SIZE as u64,
                            Environment::BOARD_SIZE as u64,
                            1,
                        ]);
                        board_tensor.copy_from_slice(&node.state.board);

                        // Flip the board. Child node always has the opposite turn of the parent.
                        for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                            board_tensor[action] *= -1.0;
                        }

                        // Mark the action as taken.
                        board_tensor[action] = 1.0;

                        // Check if the game is over.
                        let result = compute_state(
                            &board_tensor[..].try_into().unwrap(),
                            action,
                            node.state.available_actions_len - 1,
                        );

                        // Evaluate the NN with the child state to get the policy and value.
                        let mut eval_run_args = SessionRunArgs::new();
                        eval_run_args.add_feed(&self.agent.op_input, 0, &board_tensor);
                        eval_run_args.add_target(&self.agent.op_p_output);
                        eval_run_args.add_target(&self.agent.op_v_output);

                        let p_fetch_token = eval_run_args.request_fetch(&self.agent.op_p_output, 0);
                        let v_fetch_token = eval_run_args.request_fetch(&self.agent.op_v_output, 0);

                        self.session.run(&mut eval_run_args)?;

                        let p = eval_run_args.fetch::<f32>(p_fetch_token)?;
                        let v = eval_run_args.fetch::<f32>(v_fetch_token)?;

                        let mut pi = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                        pi.copy_from_slice(&p[..]);

                        // Filter out illegal actions.
                        pi[action] = 0.0;
                        for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                            if !node.state.is_available_action(action) {
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

                        let mut board = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                        board.copy_from_slice(&board_tensor[..]);

                        let reward = result.unwrap_or_else(|| v[0]);

                        // Make the child node.
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

            match env.place_stone(best_action) {
                GameStatus::InProgress => {}
                GameStatus::Draw => {
                    return Ok(0);
                }
                GameStatus::BlackWin => {
                    return Ok(1);
                }
                GameStatus::WhiteWin => {
                    return Ok(1);
                }
            }

            mcts.transition(children_index);

            let legal_moves = env
                .legal_moves
                .iter()
                .enumerate()
                .filter_map(|(index, is_legal)| if *is_legal { Some(index) } else { None })
                .collect::<Vec<_>>();
            let random_move = legal_moves[rng.gen_range(0..legal_moves.len())];

            match env.place_stone(random_move) {
                GameStatus::InProgress => {}
                GameStatus::Draw => {
                    return Ok(0);
                }
                GameStatus::BlackWin => {
                    return Ok(-1);
                }
                GameStatus::WhiteWin => {
                    return Ok(-1);
                }
            }

            {
                let has_random_move_children = {
                    let children = mcts.root().children.read();
                    children.iter().any(|node| node.action == Some(random_move))
                };
                if !has_random_move_children {
                    // Make board state for the action; first, copy the board state.
                    let mut board_tensor = Tensor::new(&[
                        1,
                        Environment::BOARD_SIZE as u64,
                        Environment::BOARD_SIZE as u64,
                        1,
                    ]);
                    board_tensor.copy_from_slice(&mcts.root().state.board);

                    // Flip the board. Child node always has the opposite turn of the parent.
                    for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                        board_tensor[action] *= -1.0;
                    }

                    // Mark the action as taken.
                    board_tensor[random_move] = 1.0;

                    // Check if the game is over.
                    let result = compute_state(
                        &board_tensor[..].try_into().unwrap(),
                        random_move,
                        mcts.root().state.available_actions_len - 1,
                    );

                    // Evaluate the NN with the child state to get the policy and value.
                    let mut eval_run_args = SessionRunArgs::new();
                    eval_run_args.add_feed(&self.agent.op_input, 0, &board_tensor);
                    eval_run_args.add_target(&self.agent.op_p_output);
                    eval_run_args.add_target(&self.agent.op_v_output);

                    let p_fetch_token = eval_run_args.request_fetch(&self.agent.op_p_output, 0);
                    let v_fetch_token = eval_run_args.request_fetch(&self.agent.op_v_output, 0);

                    self.session.run(&mut eval_run_args)?;

                    let p = eval_run_args.fetch::<f32>(p_fetch_token)?;
                    let v = eval_run_args.fetch::<f32>(v_fetch_token)?;

                    let mut pi = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                    pi.copy_from_slice(&p[..]);

                    // Filter out illegal actions.
                    pi[random_move] = 0.0;
                    for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                        if !mcts.root().state.is_available_action(action) {
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

                    let mut board = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                    board.copy_from_slice(&board_tensor[..]);

                    let reward = result.unwrap_or_else(|| v[0]);

                    // Make the child node.
                    match mcts.expand(
                        mcts.root(),
                        random_move,
                        BoardState {
                            board,
                            policy: BoardPolicy { pi },
                            reward,
                            is_terminal: result.is_some(),
                            available_actions_len: if result.is_some() {
                                0
                            } else {
                                mcts.root().state.available_actions_len - 1
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
                }
            }

            let children_index = {
                let children = mcts.root().children.read();
                children
                    .iter()
                    .position(|node| node.action == Some(random_move))
                    .unwrap()
            };

            mcts.transition(children_index);
        }
    }

    fn save(&self, name: impl AsRef<Path>) {
        let path = Path::new("saves").join(name);

        if path.exists() {
            remove_dir_all(&path).unwrap();
        } else {
            let base = path.parent().unwrap();

            if !base.exists() {
                create_dir_all(&base).unwrap();
            }
        }

        self.agent.save(&self.session, &path).unwrap();
    }
}

fn compute_ucb_1(parent_n: u64, node: &Node<BoardState>, c: f32) -> f32 {
    let n = node.n.load(Ordering::Relaxed);
    let q_s_a = node.w.load(Ordering::Relaxed) as f32 / (n as f32 + f32::EPSILON);
    let p_s_a = node.p;
    let bias = f32::sqrt(parent_n as f32) / (1 + n) as f32;
    q_s_a + c * p_s_a * bias - node.v_loss.load(Ordering::Relaxed) as f32 * Train::V_LOSS
}
