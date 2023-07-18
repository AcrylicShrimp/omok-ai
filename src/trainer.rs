use crate::{
    plot::Plotter,
    utils::{flip_horizontal, flip_vertical, rotate_180, rotate_270, rotate_90},
};
use alpha_zero::{AgentModel, BoardState, MCTSExecutor, ParallelMCTSExecutor};
use atomic_float::AtomicF32;
use environment::{Environment, GameStatus, Stone};
use mcts::{State, MCTS};
use parking_lot::RwLock;
use rand::{
    distributions::WeightedIndex, prelude::Distribution, seq::IteratorRandom, thread_rng, Rng,
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::{
    collections::VecDeque,
    fs::{create_dir_all, remove_dir_all, remove_file},
    io::Write,
    path::Path,
    sync::atomic::Ordering,
};
use tensorflow::{Scope, Session, SessionOptions, SessionRunArgs, Status, Tensor};

pub struct Transition {
    pub env: Environment,
    pub policy: [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE],
    pub z: f32,
}

pub struct Trainer {
    pub session: Session,
    pub agent: AgentModel,
    pub plotter: Plotter,
    pub replay_memory: VecDeque<Transition>,
}

impl Trainer {
    pub const MODEL_NAME: &'static str = "alpha-zero";

    pub const REPLAY_MEMORY_SIZE: usize = 600_000;
    pub const EPISODE_COUNT: usize = 50;
    pub const EVALUATE_COUNT: usize = 600;
    pub const EVALUATE_BATCH_SIZE: usize = 16;
    pub const TRAINING_COUNT: usize = 600;
    pub const TRAINING_BATCH_SIZE: usize = 128;

    pub const TEST_EVALUATE_COUNT: usize = 800;

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

        let mut plotter = Plotter::new();
        if Path::new("plots").join("losses").exists() {
            plotter.load("plots/losses").unwrap();
        }

        let this = Self {
            session,
            agent,
            plotter,
            replay_memory: VecDeque::with_capacity(Self::REPLAY_MEMORY_SIZE),
        };

        // Load the parameters if it exists.
        this.load(Self::MODEL_NAME);

        Ok(this)
    }

    pub fn train(&mut self, iteration_count: usize) -> Result<(), Status> {
        let thread_pool = ThreadPoolBuilder::new().build().unwrap();
        let parallel_mcts_executor = ParallelMCTSExecutor::new();
        let mut rng = thread_rng();
        let mut recent_losses = VecDeque::with_capacity(100);

        for iteration in 0..iteration_count {
            println!("========================================");
            println!("[iter={}] Entering self-play phase.", iteration + 1);

            let mut finished_episode_count = 0usize;
            let mut env_list = vec![Environment::new(); Self::EPISODE_COUNT];
            let mut mcts_list = Vec::with_capacity(Self::EPISODE_COUNT);
            let mut turn_count_list = vec![0; Self::EPISODE_COUNT];
            let mut transition_list = Vec::with_capacity(Self::EPISODE_COUNT);

            for env in &env_list {
                mcts_list.push(MCTS::<BoardState>::new(BoardState {
                    env: env.clone(),
                    status: GameStatus::InProgress,
                    policy: RwLock::new({
                        let mut input_tensor = Tensor::new(&[
                            1,
                            Environment::BOARD_SIZE as u64,
                            Environment::BOARD_SIZE as u64,
                            2,
                        ]);

                        env.encode_board(env.turn, &mut input_tensor[..]);

                        // Prepare the evaluation run arguments.
                        let mut eval_run_args = SessionRunArgs::new();
                        eval_run_args.add_feed(&self.agent.op_input, 0, &input_tensor);
                        eval_run_args.add_target(&self.agent.op_p_output);
                        eval_run_args.add_target(&self.agent.op_v_output);

                        let p_fetch_token = eval_run_args.request_fetch(&self.agent.op_p_output, 0);

                        // Evaluate the network.
                        self.session.run(&mut eval_run_args)?;

                        let p = eval_run_args.fetch::<f32>(p_fetch_token)?;
                        let mut policy = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];

                        policy[..].copy_from_slice(&p[..]);

                        policy
                    }),
                    z: AtomicF32::new(0f32),
                }));
                transition_list.push(Vec::with_capacity(64));
            }

            while finished_episode_count < Self::EPISODE_COUNT {
                parallel_mcts_executor.execute(
                    Self::EVALUATE_COUNT,
                    Self::EVALUATE_BATCH_SIZE,
                    &self.session,
                    &self.agent,
                    &mcts_list,
                )?;

                for (env, (mcts, (turn_count, transitions))) in env_list.iter_mut().zip(
                    mcts_list
                        .iter_mut()
                        .zip(turn_count_list.iter_mut().zip(transition_list.iter_mut())),
                ) {
                    if mcts.root().state.status.is_terminal() {
                        continue;
                    }

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

                    let action = if *turn_count < Self::TEMPERATURE_THRESHOLD {
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

                    *turn_count += 1;

                    let children_index = {
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
                        GameStatus::BlackWin => (1f32, true),
                        GameStatus::WhiteWin => (1f32, true),
                    };

                    transitions.push(Transition {
                        env: env_before_action,
                        policy,
                        z,
                    });

                    // Re-root the tree.
                    mcts.transition(children_index);

                    if is_terminal {
                        finished_episode_count += 1;
                        print!(
                            "\r[iter={}] Self-playing... [episode={}/{}]",
                            iteration + 1,
                            finished_episode_count,
                            Self::EPISODE_COUNT
                        );
                        std::io::stdout().flush().unwrap();
                        continue;
                    }
                }
            }

            // Update z in the game history, so that the agent can learn from it.
            for mut transitions in transition_list.into_iter() {
                let mut z = transitions.last().unwrap().z;

                for transition in transitions.iter_mut().rev() {
                    transition.z = z;
                    z = -z;
                }

                // Augment the replay memory, by rotating and flipping the board.
                // We can generate extra 5 boards from one board.
                // 3 from rotation, and 2 from flipping.
                let mut augmented_replay_memory = Vec::with_capacity(transitions.len() * 5);

                for transition in transitions.iter() {
                    augmented_replay_memory.push(Transition {
                        env: {
                            let mut env = transition.env.clone();
                            rotate_90(
                                &transition.env.board,
                                &mut env.board,
                                Environment::BOARD_SIZE,
                            );
                            env
                        },
                        policy: {
                            let mut policy =
                                [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                            rotate_90(&transition.policy, &mut policy, Environment::BOARD_SIZE);
                            policy
                        },
                        z: transition.z,
                    });
                    augmented_replay_memory.push(Transition {
                        env: {
                            let mut env = transition.env.clone();
                            rotate_180(
                                &transition.env.board,
                                &mut env.board,
                                Environment::BOARD_SIZE,
                            );
                            env
                        },
                        policy: {
                            let mut policy =
                                [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                            rotate_180(&transition.policy, &mut policy, Environment::BOARD_SIZE);
                            policy
                        },
                        z: transition.z,
                    });
                    augmented_replay_memory.push(Transition {
                        env: {
                            let mut env = transition.env.clone();
                            rotate_270(
                                &transition.env.board,
                                &mut env.board,
                                Environment::BOARD_SIZE,
                            );
                            env
                        },
                        policy: {
                            let mut policy =
                                [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                            rotate_270(&transition.policy, &mut policy, Environment::BOARD_SIZE);
                            policy
                        },
                        z: transition.z,
                    });
                    augmented_replay_memory.push(Transition {
                        env: {
                            let mut env = transition.env.clone();
                            flip_horizontal(
                                &transition.env.board,
                                &mut env.board,
                                Environment::BOARD_SIZE,
                            );
                            env
                        },
                        policy: {
                            let mut policy =
                                [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                            flip_horizontal(
                                &transition.policy,
                                &mut policy,
                                Environment::BOARD_SIZE,
                            );
                            policy
                        },
                        z: transition.z,
                    });
                    augmented_replay_memory.push(Transition {
                        env: {
                            let mut env = transition.env.clone();
                            flip_vertical(
                                &transition.env.board,
                                &mut env.board,
                                Environment::BOARD_SIZE,
                            );
                            env
                        },
                        policy: {
                            let mut policy =
                                [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                            flip_vertical(&transition.policy, &mut policy, Environment::BOARD_SIZE);
                            policy
                        },
                        z: transition.z,
                    });
                }

                self.replay_memory.extend(transitions);
                self.replay_memory.extend(augmented_replay_memory);
            }

            while Self::REPLAY_MEMORY_SIZE < self.replay_memory.len() {
                self.replay_memory.pop_front();
            }

            println!();
            println!("[iter={}] Entering training phase.", iteration + 1);

            for _ in 0..Self::TRAINING_COUNT {
                let transition = self
                    .replay_memory
                    .iter()
                    .choose_multiple(&mut rng, Self::TRAINING_BATCH_SIZE);

                debug_assert!(transition.len() == Self::TRAINING_BATCH_SIZE);

                let mut tensor_input = Tensor::new(&[
                    Self::TRAINING_BATCH_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    2,
                ]);
                let mut tensor_z_input = Tensor::new(&[Self::TRAINING_BATCH_SIZE as u64, 1]);
                let mut tensor_pi_input = Tensor::new(&[
                    Self::TRAINING_BATCH_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                ]);

                for batch_index in 0..Self::TRAINING_BATCH_SIZE {
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
                loss_run_args.add_target(&self.agent.op_v_loss);
                loss_run_args.add_target(&self.agent.op_p_loss);
                loss_run_args.add_target(&self.agent.op_loss);

                let v_loss_fetch_token = loss_run_args.request_fetch(&self.agent.op_v_loss, 0);
                let p_loss_fetch_token = loss_run_args.request_fetch(&self.agent.op_p_loss, 0);
                let loss_fetch_token = loss_run_args.request_fetch(&self.agent.op_loss, 0);
                self.session.run(&mut loss_run_args)?;

                let v_loss = loss_run_args.fetch::<f32>(v_loss_fetch_token)?;
                let p_loss = loss_run_args.fetch::<f32>(p_loss_fetch_token)?;
                let loss = loss_run_args.fetch::<f32>(loss_fetch_token)?;

                if recent_losses.len() == 100 {
                    recent_losses.pop_front();
                }

                recent_losses.push_back((v_loss[0], p_loss[0], loss[0]));
            }

            let (v_loss, p_loss, loss) = (
                recent_losses.iter().map(|loss| loss.0).sum::<f32>() / recent_losses.len() as f32,
                recent_losses.iter().map(|loss| loss.1).sum::<f32>() / recent_losses.len() as f32,
                recent_losses.iter().map(|loss| loss.2).sum::<f32>() / recent_losses.len() as f32,
            );

            println!(
                "[iter={}] Loss: {} [v_loss={:.4}, p_loss={:.4}]",
                iteration + 1,
                loss,
                v_loss,
                p_loss,
            );

            self.plotter.add_loss((v_loss, p_loss, loss));
            self.plotter.save("losses").unwrap();
            self.plotter.draw_plot("loss.svg");

            self.save(Self::MODEL_NAME);
            println!("[iter={}] Model saved.", iteration + 1);

            if iteration % 10 == 0 {
                println!(
                    "[iter={}] Playing against random move player.",
                    iteration + 1
                );

                let mut win = 0;
                let mut lose = 0;
                let mut draw = 0;

                for game in 0..100 {
                    print!(
                        "\r[iter={}] Playing... [game={}/{}]",
                        iteration + 1,
                        game + 1,
                        100
                    );
                    std::io::stdout().flush().unwrap();

                    let result = self.play_against_random_player(&thread_pool)?;
                    if result == 1 {
                        win += 1;
                    } else if result == -1 {
                        lose += 1;
                    } else {
                        draw += 1;
                    }
                }

                println!();
                println!(
                    "[iter={}] Win: {}, Lose: {}, Draw: {}",
                    iteration + 1,
                    win,
                    lose,
                    draw
                );
            }
        }

        Ok(())
    }

    fn play_against_random_player(&self, thread_pool: &ThreadPool) -> Result<i32, Status> {
        let mut rng = thread_rng();
        let mut env = Environment::new();
        let mut mcts_executor = MCTSExecutor::new(MCTS::<BoardState>::new(BoardState {
            env: env.clone(),
            status: GameStatus::InProgress,
            policy: RwLock::new({
                let mut input_tensor = Tensor::new(&[
                    1,
                    Environment::BOARD_SIZE as u64,
                    Environment::BOARD_SIZE as u64,
                    2,
                ]);

                env.encode_board(env.turn, &mut input_tensor[..]);

                // Prepare the evaluation run arguments.
                let mut eval_run_args = SessionRunArgs::new();
                eval_run_args.add_feed(&self.agent.op_input, 0, &input_tensor);
                eval_run_args.add_target(&self.agent.op_p_output);
                eval_run_args.add_target(&self.agent.op_v_output);

                let p_fetch_token = eval_run_args.request_fetch(&self.agent.op_p_output, 0);

                // Evaluate the network.
                self.session.run(&mut eval_run_args)?;

                let p = eval_run_args.fetch::<f32>(p_fetch_token)?;
                let mut policy = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];

                policy[..].copy_from_slice(&p[..]);
                policy
            }),
            z: AtomicF32::new(0f32),
        }));

        loop {
            mcts_executor.run(
                thread_pool,
                &self.agent,
                &self.session,
                Self::TEST_EVALUATE_COUNT,
            )?;

            let (children_index, best_action) = {
                let children = mcts_executor.mcts.root().children.read();
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

            mcts_executor.mcts.transition(children_index);

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
                let children = mcts_executor.mcts.root().children.read();
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
                mcts_executor.mcts.root().state.env.encode_board(
                    mcts_executor.mcts.root().state.env.turn,
                    &mut board_tensor[..],
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
                    if !mcts_executor.mcts.root().state.is_available_action(action) {
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
                match mcts_executor.mcts.expand(
                    mcts_executor.mcts.root(),
                    random_move,
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
                let children = mcts_executor.mcts.root().children.read();
                children
                    .iter()
                    .position(|node| node.action == Some(random_move))
                    .unwrap()
            };

            mcts_executor.mcts.transition(children_index);
        }
    }

    pub fn save(&self, name: impl AsRef<Path>) {
        let path_base = Path::new("saves");
        let path_model = path_base.join(name);
        if path_base.exists() {
            if path_model.exists() {
                if path_model.is_dir() {
                    remove_dir_all(&path_model).unwrap();
                } else {
                    remove_file(&path_model).unwrap();
                }
            }
        } else {
            if !path_base.exists() {
                create_dir_all(path_base).unwrap();
            }
        }

        self.agent.io.save(&self.session, &path_model).unwrap();
    }

    pub fn load(&self, name: impl AsRef<Path>) {
        let path = Path::new("saves").join(name);

        if !path.exists() {
            return;
        }

        self.agent.io.load(&self.session, &path).unwrap();
    }
}
