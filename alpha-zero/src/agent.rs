use crate::{encode_nn_input, AgentModel, BoardState, EnvTurnMode};
use atomic_float::AtomicF32;
use environment::{Environment, GameStatus};
use mcts::{State, MCTS};
use parking_lot::RwLock;
use rand::{distributions::WeightedIndex, prelude::*};
use std::{iter::once, sync::atomic::Ordering};
use tensorflow::{Session, Status};

pub struct Agent {
    pub env: Environment,
    pub mcts: MCTS<BoardState>,
}

impl Agent {
    pub fn new(agent_model: &AgentModel, session: &Session) -> Result<Self, Status> {
        let env = Environment::new();

        let input = encode_nn_input(1, EnvTurnMode::Player, once(&env));
        let p = agent_model.evaluate_p(session, input)?;
        let policy = {
            let mut policy = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
            policy.copy_from_slice(&p[..]);
            policy
        };

        let mcts = MCTS::new(BoardState {
            env: env.clone(),
            status: GameStatus::InProgress,
            policy: RwLock::new(policy),
            z: AtomicF32::new(0f32),
        });

        Ok(Self { env, mcts })
    }

    /// Computes the policy from MCTS tree.
    /// Returns `None` if:
    /// - The tree is empty.
    /// - All children in the tree have 0 visits (never explored).
    ///
    /// To ensure that the policy is not empty, run MCTS for a few iterations first. See [MCTSExecutor](super::MCTSExecutor) or [ParallelMCTSExecutor](super::ParallelMCTSExecutor) to perform MCTS.
    pub fn compute_policy(
        &self,
    ) -> Option<[f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE]> {
        let root = self.mcts.root();

        let mut sum = 0f32;
        let mut policy = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];

        {
            let children = root.children.read();

            if children.is_empty() {
                return None;
            }

            for child in children.iter() {
                let action = child.action.unwrap();
                let n = child.n.load(Ordering::Relaxed) as f32;
                sum += n;
                policy[action] = n;
            }
        }

        if sum < f32::EPSILON {
            return None;
        }

        let sum_inv = sum.recip();

        for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
            policy[action] *= sum_inv;
        }

        Some(policy)
    }

    /// Samples a single action from the computed policy and returns the action and the policy.
    /// Returns `None` if the policy is empty.
    /// Note that the policy returned by this function is not affected by the temperature;
    /// the temperature is only used to sample the action.
    pub fn sample_action(
        &self,
        mode: ActionSamplingMode,
    ) -> Option<(
        usize,
        [f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE],
    )> {
        let policy = if let Some(policy) = self.compute_policy() {
            policy
        } else {
            return None;
        };

        Some((
            match mode {
                ActionSamplingMode::Best => {
                    policy
                        .iter()
                        .enumerate()
                        .max_by(|&(_, a), &(_, b)| f32::total_cmp(a, b))
                        .unwrap()
                        .0
                }
                ActionSamplingMode::Boltzmann(temperature) => {
                    let mut sum = 0f32;
                    let mut heated_policy =
                        [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
                    let temperature_inv = temperature.recip();

                    for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                        let prob = policy[action];

                        if prob < f32::EPSILON {
                            continue;
                        }

                        let heated = (policy[action] * temperature_inv).exp();
                        sum += heated;
                        heated_policy[action] = heated;
                    }

                    let sum_inv = sum.recip();

                    for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                        heated_policy[action] *= sum_inv;
                    }

                    let dist = WeightedIndex::new(&heated_policy).unwrap();

                    dist.sample(&mut rand::thread_rng())
                }
            },
            policy,
        ))
    }

    /// Creates a new child node for the given action if it doesn't exist yet.
    /// This function is intended to be called during game play.
    /// When opponent plays, it's possible that action is not in the tree yet,
    /// because opponent can play any action, not just the ones that were
    /// explored by MCTS.
    pub fn ensure_action_exists(
        &mut self,
        action: usize,
        agent_model: &AgentModel,
        session: &Session,
    ) -> Result<(), Status> {
        if Environment::BOARD_SIZE * Environment::BOARD_SIZE <= action {
            return Ok(());
        }

        let mut env = self.env.clone();
        env.place_stone(action);

        let input = encode_nn_input(1, EnvTurnMode::Opponent, once(&env));
        let p = agent_model.evaluate_p(session, input)?;
        let mut policy = {
            let mut policy = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
            policy.copy_from_slice(&p[..]);
            policy
        };

        // Filter out illegal actions.
        policy[action] = 0.0;
        for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
            if !self.mcts.root().state.is_available_action(action) {
                policy[action] = 0.0;
            }
        }

        let sum = policy.iter().sum::<f32>();

        // Re-normalize the policy if the policy is not all zero.
        if f32::EPSILON <= sum {
            let sum_inv = sum.recip();

            for action in 0..Environment::BOARD_SIZE * Environment::BOARD_SIZE {
                policy[action] *= sum_inv;
            }
        }

        // Make the child node.
        self.mcts.expand(
            self.mcts.root(),
            action,
            BoardState {
                env,
                status: GameStatus::InProgress,
                policy: RwLock::new(policy),
                z: AtomicF32::new(0.0),
            },
        );

        Ok(())
    }

    /// Plays a single action and returns the game status.
    /// Returns `None` if:
    /// - The action is illegal.
    /// - The action is not in the tree.
    /// - The game is already finished.
    ///
    /// To ensure that the action is in the tree, use `ensure_action_exists` first.
    pub fn play_action(&mut self, action: usize) -> Option<GameStatus> {
        if self.mcts.root().state.status.is_terminal() {
            return None;
        }

        let children_index = {
            let children = self.mcts.root().children.read();
            if let Some(index) = children
                .iter()
                .enumerate()
                .find(|(_, &child)| child.action == Some(action))
                .map(|(index, _)| index)
            {
                index
            } else {
                return None;
            }
        };
        let status = if let Some(status) = self.env.place_stone(action) {
            status
        } else {
            return None;
        };

        self.mcts.transition(children_index);
        Some(status)
    }
}

/// A method to sample actions from the policy.
pub enum ActionSamplingMode {
    /// Selects the action with the highest probability.
    Best,
    /// Selects the action using Boltzmann distribution with the given temperature.
    Boltzmann(f32),
}
