mod agent_model;
mod model_io;
mod network;

pub use agent_model::*;
pub use model_io::*;
pub use network::*;

use atomic_float::AtomicF32;
use bitvec::vec::BitVec;
use environment::{Environment, GameStatus, Stone, Turn};
use mcts::{Node, PolicyRef, State, MCTS};
use parking_lot::{RwLock, RwLockReadGuard};
use rand::{
    distributions::WeightedIndex,
    prelude::Distribution,
    seq::{IteratorRandom, SliceRandom},
    thread_rng, Rng,
};
use rayon::{
    prelude::{IntoParallelIterator, ParallelIterator},
    ThreadPool, ThreadPoolBuilder,
};
use std::{
    collections::VecDeque,
    fs::{create_dir_all, remove_dir_all, remove_file},
    path::Path,
    sync::atomic::Ordering,
};
use tensorflow::{Scope, Session, SessionOptions, SessionRunArgs, Status, Tensor};

pub struct BoardState {
    pub env: Environment,
    pub status: GameStatus,
    pub policy: RwLock<[f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE]>,
    pub z: AtomicF32,
}

impl State for BoardState {
    type PolicyRef<'s> = BoardPolicy<'s>;

    fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    fn policy<'s>(&'s self) -> Self::PolicyRef<'s> {
        BoardPolicy {
            policy: self.policy.read(),
        }
    }

    fn available_actions_len(&self) -> usize {
        self.env.legal_move_count as usize
    }

    fn is_available_action(&self, action: usize) -> bool {
        self.env.board[action] == Stone::Empty
    }
}

impl Clone for BoardState {
    fn clone(&self) -> Self {
        Self {
            env: self.env.clone(),
            status: self.status.clone(),
            policy: RwLock::new(self.policy.read().clone()),
            z: AtomicF32::new(self.z.load(Ordering::Relaxed)),
        }
    }
}

pub struct BoardPolicy<'s> {
    pub policy: RwLockReadGuard<'s, [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE]>,
}

impl<'s> PolicyRef<'s> for BoardPolicy<'s> {
    fn get(&self, action: usize) -> f32 {
        self.policy[action]
    }
}

pub struct Transition {
    pub env: Environment,
    pub policy: [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE],
    pub z: f32,
}

pub struct Train {
    pub session: Session,
    pub agent: AgentModel,
    pub replay_memory: VecDeque<Transition>,
}

impl Train {
    pub const MODEL_NAME: &'static str = "alpha-zero";

    pub const REPLAY_MEMORY_SIZE: usize = 5_000;
    pub const EPISODE_COUNT: usize = 50;
    pub const EVALUATE_COUNT: usize = 800;
    pub const TRAINING_COUNT: usize = 100;
    pub const BATCH_SIZE: usize = 64;
    pub const C_PUCT: f32 = 1.0;
    pub const V_LOSS: f32 = 0.5f32;

    pub const TEST_EVALUATE_COUNT: usize = 1000;

    pub const TEMPERATURE: f32 = 1.0;
    pub const TEMPERATURE_THRESHOLD: usize = 30;

    pub fn new() -> Result<Self, Status> {
        let mut scope = Scope::new_root_scope();
        let agent = AgentModel::new(&mut scope)?;
        let session = Session::new(&SessionOptions::new(), &scope.graph())?;

        let mut init_run_args = SessionRunArgs::new();

        for variable in &agent.variables {
            init_run_args.add_target(&variable.initializer());
        }

        session.run(&mut init_run_args)?;

        let this = Self {
            session,
            agent,
            replay_memory: VecDeque::with_capacity(Self::REPLAY_MEMORY_SIZE),
        };

        // Load the parameters if it exists.
        this.load(Self::MODEL_NAME);

        Ok(this)
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
                        env: env.clone(),
                        status: GameStatus::InProgress,
                        policy: RwLock::new(
                            [1f32 / (Environment::BOARD_SIZE * Environment::BOARD_SIZE) as f32;
                                Environment::BOARD_SIZE * Environment::BOARD_SIZE],
                        ),
                        z: AtomicF32::new(0f32),
                    };
                    MCTS::<BoardState>::new(state)
                };
                let mut turn_count = 0usize;

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
                                    node.propagate(node.state.z.load(Ordering::Relaxed));
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
                                let mut board_tensor = Tensor::new(&[
                                    1,
                                    Environment::BOARD_SIZE as u64,
                                    Environment::BOARD_SIZE as u64,
                                    2,
                                ]);
                                env.encode_board(env.turn, &mut board_tensor[..]);

                                // Pre-expand the node.
                                // This helps to other threads to avoid expanding the same node.
                                let expanded_child = match mcts.expand(
                                    node,
                                    action,
                                    f32::MIN,
                                    u64::MAX - 1,
                                    BoardState {
                                        env,
                                        status,
                                        policy: RwLock::new(
                                            [0f32;
                                                Environment::BOARD_SIZE * Environment::BOARD_SIZE],
                                        ),
                                        z: AtomicF32::new(0f32),
                                    },
                                ) {
                                    Some(child) => child,
                                    None => {
                                        // The node is already expanded by other thread.
                                        // We don't need to expand it again.
                                        return Ok(());
                                    }
                                };

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

                                let reward = if status.is_terminal() { 1f32 } else { v[0] };

                                // Update the pre-expanded child node.
                                expanded_child.state.policy.write().copy_from_slice(&pi[..]);
                                expanded_child.state.z.store(reward, Ordering::Relaxed);

                                // Reset the node.
                                expanded_child.n.store(0, Ordering::Relaxed);
                                expanded_child.w.store(0f32, Ordering::Relaxed);

                                // Perform backup from the expanded child node.
                                expanded_child.propagate(reward);

                                Ok(())
                            },
                        )
                    })?;

                    // Get the policy from the root node. Policy is the visit count of the children.
                    let mut policy = {
                        let root = mcts.root();
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

                    let action = if turn_count < Self::TEMPERATURE_THRESHOLD {
                        // Apply Boltzmann exploration.
                        let inv_tau = Self::TEMPERATURE.recip();

                        for index in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                            policy[index] = policy[index].powf(inv_tau);
                        }

                        // Re-normalize the policy if the policy is not all zero.
                        let sum = policy.iter().sum::<f32>();
                        if f32::EPSILON <= sum {
                            for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                                policy[action] /= sum;
                            }
                        }

                        // Sample an action from the policy.
                        let dist = WeightedIndex::new(&policy).unwrap();
                        dist.sample(&mut rng)
                    } else {
                        // Find best action.
                        policy
                            .iter()
                            .enumerate()
                            .max_by(|&(_, a), &(_, b)| f32::total_cmp(a, b))
                            .unwrap()
                            .0
                    };

                    turn_count += 1;

                    let children_index = {
                        // TODO: We must ensure that the action is in the children.
                        let children = mcts.root().children.read();
                        let (index, _) = children
                            .iter()
                            .enumerate()
                            .find(|&(_, child)| child.action == Some(action))
                            .unwrap();
                        index
                    };

                    // Clone the environment to store.
                    // This is required because we have to store the environment
                    // before the action is applied.
                    let env_before_action = env.clone();

                    // Play the action.
                    let (z, is_terminal) = match env.place_stone(action).unwrap() {
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
                        env: env_before_action,
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
                    2,
                ]);
                let mut tensor_z_input = Tensor::new(&[Self::BATCH_SIZE as u64, 1]);
                let mut tensor_pi_input = Tensor::new(&[
                    Self::BATCH_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                ]);

                for batch_index in 0..Self::BATCH_SIZE {
                    let transition = transition[batch_index];

                    transition.env.encode_board(
                        transition.env.turn,
                        &mut tensor_input[batch_index
                            * Environment::BOARD_SIZE
                            * Environment::BOARD_SIZE
                            * 2
                            ..(batch_index + 1)
                                * Environment::BOARD_SIZE
                                * Environment::BOARD_SIZE
                                * 2],
                    );
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

            self.save(Self::MODEL_NAME);
            println!("Model saved.");
        }

        Ok(())
    }

    fn play_against_random_player(&self, thread_pool: &ThreadPool) -> Result<i32, Status> {
        let mut rng = thread_rng();
        let mut env = Environment::new();
        let mut mcts = {
            let state = BoardState {
                env: env.clone(),
                status: GameStatus::InProgress,
                policy: RwLock::new(
                    [1f32 / (Environment::BOARD_SIZE * Environment::BOARD_SIZE) as f32;
                        Environment::BOARD_SIZE * Environment::BOARD_SIZE],
                ),
                z: AtomicF32::new(0f32),
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
                            node.propagate(node.state.z.load(Ordering::Relaxed));
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
                        let mut board_tensor = Tensor::new(&[
                            1,
                            Environment::BOARD_SIZE as u64,
                            Environment::BOARD_SIZE as u64,
                            2,
                        ]);
                        env.encode_board(env.turn, &mut board_tensor[..]);

                        // Pre-expand the node.
                        // This helps to other threads to avoid expanding the same node.
                        let expanded_child = match mcts.expand(
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
                            Some(child) => child,
                            None => {
                                // The node is already expanded by other thread.
                                // We don't need to expand it again.
                                return Ok(());
                            }
                        };

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

                        let reward = if status.is_terminal() { 1f32 } else { v[0] };

                        // Update the pre-expanded child node.
                        expanded_child.state.policy.write().copy_from_slice(&pi[..]);
                        expanded_child.state.z.store(reward, Ordering::Relaxed);

                        // Reset the node.
                        expanded_child.n.store(0, Ordering::Relaxed);
                        expanded_child.w.store(0f32, Ordering::Relaxed);

                        // Perform backup from the expanded child node.
                        expanded_child.propagate(reward);

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

            match env.place_stone(best_action).unwrap() {
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

            let legal_moves = (0..Environment::BOARD_SIZE * Environment::BOARD_SIZE)
                .filter(|&action| env.board[action] == Stone::Empty)
                .collect::<Vec<_>>();
            let random_move = legal_moves[rng.gen_range(0..legal_moves.len())];

            match env.place_stone(random_move).unwrap() {
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

            let has_random_move_children = {
                let children = mcts.root().children.read();
                children.iter().any(|node| node.action == Some(random_move))
            };
            if !has_random_move_children {
                // Encode the board state.
                let mut board_tensor = Tensor::new(&[
                    1,
                    Environment::BOARD_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    2,
                ]);
                env.encode_board(env.turn, &mut board_tensor[..]);

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

                // Make the child node.
                match mcts.expand(
                    mcts.root(),
                    random_move,
                    0f32,
                    0,
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

    pub fn save(&self, name: impl AsRef<Path>) {
        let path = Path::new("saves").join(name);

        if path.exists() {
            if path.is_dir() {
                remove_dir_all(&path).unwrap();
            } else {
                remove_file(&path).unwrap();
            }
        } else {
            let base = path.parent().unwrap();

            if !base.exists() {
                create_dir_all(&base).unwrap();
            }
        }

        self.agent.io.save(&self.session, &path).unwrap();
    }

    pub fn load(&self, name: impl AsRef<Path>) {
        let path = Path::new("saves").join(name);

        if !path.exists() {
            return;
        }

        self.agent.io.load(&self.session, &path).unwrap();
    }
}

fn compute_ucb_1<S>(parent_n: u64, node: &Node<S>, c: f32) -> f32
where
    S: State,
{
    let n = node.n.load(Ordering::Relaxed);
    let q_s_a = node.w.load(Ordering::Relaxed) as f32 / (n as f32 + f32::EPSILON);
    let p_s_a = node.p;
    let bias = f32::sqrt(parent_n as f32) / (1 + n) as f32;
    q_s_a + c * p_s_a * bias - node.v_loss.load(Ordering::Relaxed) as f32 * Train::V_LOSS
}
