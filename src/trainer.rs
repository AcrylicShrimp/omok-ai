use crate::{
    plot::Plotter,
    utils::{flip_horizontal, flip_vertical, rotate_180, rotate_270, rotate_90},
};
use alpha_zero::{
    encode_nn_input, encode_nn_targets, ActionSamplingMode, Agent, AgentModel, EnvTurnMode,
    ParallelMCTSExecutor,
};
use environment::{Environment, GameStatus, Stone};
use rand::{seq::IteratorRandom, thread_rng, Rng};
use std::{
    collections::VecDeque,
    fs::{create_dir_all, remove_dir_all, remove_file},
    io::Write,
    path::Path,
};
use tensorflow::{Scope, Session, SessionOptions, SessionRunArgs, Status};

pub struct Transition {
    pub env: Environment,
    pub policy: [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE],
    pub z: f32,
}

pub struct Trainer {
    pub session: Session,
    pub agent_model: AgentModel,
    pub plotter: Plotter,
    pub replay_memory: VecDeque<Transition>,
}

impl Trainer {
    pub const MODEL_NAME: &'static str = "alpha-zero";

    pub const REPLAY_MEMORY_SIZE: usize = 250_000;
    pub const EPISODE_COUNT: usize = 500;
    pub const EVALUATE_COUNT: usize = 600;
    pub const EVALUATE_BATCH_SIZE: usize = 8;
    pub const TRAINING_COUNT: usize = 1000;
    pub const TRAINING_BATCH_SIZE: usize = 256;

    pub const TEST_EVALUATE_COUNT: usize = 600;

    pub const TEMPERATURE: f32 = 2.0;
    pub const TEMPERATURE_THRESHOLD: usize = 100;

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
            agent_model: agent,
            plotter,
            replay_memory: VecDeque::with_capacity(Self::REPLAY_MEMORY_SIZE),
        };

        // Load the parameters if it exists.
        this.load(Self::MODEL_NAME);

        Ok(this)
    }

    pub fn train(&mut self, iteration_count: usize) -> Result<(), Status> {
        let parallel_mcts_executor = ParallelMCTSExecutor::new();
        let mut rng = thread_rng();
        let mut recent_losses = VecDeque::with_capacity(100);

        for iteration in 0..iteration_count {
            println!("========================================");
            println!("[iter={}] Entering self-play phase.", iteration + 1);

            // Empty the replay memory.
            self.replay_memory.clear();

            let mut finished_episode_count = 0usize;
            let mut agents = Vec::with_capacity(Self::EPISODE_COUNT);
            let mut turn_counts = vec![0; Self::EPISODE_COUNT];
            let mut transition_list = Vec::with_capacity(Self::EPISODE_COUNT);

            for _ in 0..Self::EPISODE_COUNT {
                agents.push(Agent::new(&self.agent_model, &self.session)?);
                transition_list.push(Vec::with_capacity(64));
            }

            while !agents.is_empty() {
                parallel_mcts_executor.execute(
                    Self::EVALUATE_COUNT,
                    Self::EVALUATE_BATCH_SIZE,
                    &self.agent_model,
                    &self.session,
                    &agents,
                )?;

                let mut index = 0;

                while index < agents.len() {
                    let agent = &mut agents[index];
                    let turn_count = &mut turn_counts[index];
                    let transitions = &mut transition_list[index];

                    let (action, policy) = agent
                        .sample_action(if *turn_count < Self::TEMPERATURE_THRESHOLD {
                            ActionSamplingMode::Boltzmann(Self::TEMPERATURE)
                        } else {
                            ActionSamplingMode::Best
                        })
                        .unwrap();

                    *turn_count += 1;

                    // Clone the environment to store.
                    // This is required because we have to store the environment
                    // before the action is applied.
                    let env_before_action = agent.env.clone();

                    // Play the action.
                    let (z, is_terminal) = match agent.play_action(action).unwrap() {
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

                    if is_terminal {
                        finished_episode_count += 1;

                        agents.swap_remove(index);
                        turn_counts.swap_remove(index);

                        print!(
                            "\r[iter={}] Self-playing... [episode={}/{}]",
                            iteration + 1,
                            finished_episode_count,
                            Self::EPISODE_COUNT
                        );
                        std::io::stdout().flush().unwrap();

                        continue;
                    }

                    index += 1;
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
                let transitions = self
                    .replay_memory
                    .iter()
                    .choose_multiple(&mut rng, Self::TRAINING_BATCH_SIZE);

                debug_assert!(!transitions.is_empty());

                let input = encode_nn_input(
                    transitions.len(),
                    EnvTurnMode::Player,
                    transitions.iter().map(|&transition| &transition.env),
                );
                let (policy_target, value_target) = encode_nn_targets(
                    transitions.len(),
                    transitions.iter().map(|&transition| &transition.policy),
                    transitions.iter().map(|&transition| transition.z),
                );

                let (policy_loss, value_loss, loss) =
                    self.agent_model
                        .train(&self.session, input, policy_target, value_target)?;

                recent_losses.push_back((value_loss, policy_loss, loss));

                while 100 < recent_losses.len() {
                    recent_losses.pop_front();
                }
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
            self.plotter.draw_plot("plots/loss.svg");

            self.save(Self::MODEL_NAME);
            println!("[iter={}] Model saved.", iteration + 1);

            if iteration % 10 == 0 {
                println!(
                    "[iter={}] Playing against random move player.",
                    iteration + 1
                );

                let (black_win, white_win, draw) =
                    self.play_against_random_player(100, &parallel_mcts_executor)?;

                println!();
                println!(
                    "[iter={}] Win: {}, Lose: {}, Draw: {}",
                    iteration + 1,
                    black_win,
                    white_win,
                    draw
                );
            }
        }

        Ok(())
    }

    fn play_against_random_player(
        &self,
        episode_count: usize,
        parallel_mcts_executor: &ParallelMCTSExecutor,
    ) -> Result<(u32, u32, u32), Status> {
        let mut rng = thread_rng();
        let mut black_win = 0u32;
        let mut white_win = 0u32;
        let mut draw = 0u32;
        let mut agents = Vec::with_capacity(episode_count);

        for _ in 0..episode_count {
            agents.push(Agent::new(&self.agent_model, &self.session)?);
        }

        while !agents.is_empty() {
            parallel_mcts_executor.execute(
                Self::TEST_EVALUATE_COUNT,
                Self::EVALUATE_BATCH_SIZE,
                &self.agent_model,
                &self.session,
                &agents,
            )?;

            let mut index = 0;

            while index < agents.len() {
                let agent = &mut agents[index];
                let (best_action, _) = agent.sample_action(ActionSamplingMode::Best).unwrap();
                let is_terminal = match agent.play_action(best_action).unwrap() {
                    GameStatus::InProgress => false,
                    GameStatus::Draw => {
                        draw += 1;
                        true
                    }
                    GameStatus::BlackWin => {
                        black_win += 1;
                        true
                    }
                    GameStatus::WhiteWin => {
                        white_win += 1;
                        true
                    }
                };

                if is_terminal {
                    agents.swap_remove(index);
                    continue;
                }

                let legal_moves = (0..Environment::BOARD_SIZE * Environment::BOARD_SIZE)
                    .filter(|&action| agent.env.board[action] == Stone::Empty)
                    .collect::<Vec<_>>();
                let random_action = legal_moves[rng.gen_range(0..legal_moves.len())];

                agent.ensure_action_exists(random_action, &self.agent_model, &self.session)?;

                let is_terminal = match agent.play_action(random_action).unwrap() {
                    GameStatus::InProgress => false,
                    GameStatus::Draw => {
                        draw += 1;
                        true
                    }
                    GameStatus::BlackWin => {
                        black_win += 1;
                        true
                    }
                    GameStatus::WhiteWin => {
                        white_win += 1;
                        true
                    }
                };

                if is_terminal {
                    agents.swap_remove(index);
                    continue;
                }

                index += 1;
            }
        }

        Ok((black_win, white_win, draw))
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

        self.agent_model
            .io
            .save(&self.session, &path_model)
            .unwrap();
    }

    pub fn load(&self, name: impl AsRef<Path>) {
        let path = Path::new("saves").join(name);

        if !path.exists() {
            return;
        }

        self.agent_model.io.load(&self.session, &path).unwrap();
    }
}
