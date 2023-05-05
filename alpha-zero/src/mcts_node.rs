use atomic_float::AtomicF32;
use environment::{Environment, GameStatus, Stone};
use mcts::{PolicyRef, State};
use parking_lot::{RwLock, RwLockReadGuard};
use std::sync::atomic::Ordering;

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
