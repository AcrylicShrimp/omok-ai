#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod agent;

use agent::Agent;
use environment::{Environment, GameStatus, Stone, Turn};
use std::sync::Mutex;

struct Application {
    agent: Agent,
    env_status: GameStatus,
}

impl Application {
    pub const MCTS_COUNT: usize = 600;
    pub const MCTS_BATCH_SIZE: usize = 16;

    pub fn new() -> Self {
        let mut agent = Agent::new();

        let action = agent.make_move(Self::MCTS_COUNT, Self::MCTS_BATCH_SIZE);
        let env_status = agent.agent.play_action(action).unwrap();

        Self { agent, env_status }
    }

    fn on_click_button(&mut self, x: usize, y: usize) -> &mut Self {
        match self.env_status {
            GameStatus::InProgress => {
                if self.agent.agent.env.turn != Turn::White {
                    return self;
                }

                let index = y * Environment::BOARD_SIZE + x;
                self.place_stone(index);
            }
            _ => {
                // Reset game
                self.agent = Agent::new();

                let action = self
                    .agent
                    .make_move(Self::MCTS_COUNT, Self::MCTS_BATCH_SIZE);
                self.env_status = self.agent.agent.play_action(action).unwrap();
            }
        }

        self
    }

    fn place_stone(&mut self, index: usize) {
        self.agent
            .agent
            .ensure_action_exists(index, &self.agent.agent_model, &self.agent.session)
            .unwrap();
        self.env_status = match self.agent.agent.play_action(index) {
            Some(status) => status,
            None => return,
        };

        if self.env_status.is_terminal() {
            return;
        }

        let action = self
            .agent
            .make_move(Self::MCTS_COUNT, Self::MCTS_BATCH_SIZE);
        self.env_status = self.agent.agent.play_action(action).unwrap();
    }
}

#[derive(serde::Serialize)]
struct ClickResponse {
    board: Vec<i32>,
    game_status: GameStatus,
}

fn main() {
    tauri::Builder::default()
        .manage(Mutex::new(Application::new()))
        .invoke_handler(tauri::generate_handler![on_click])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
fn on_click(state: tauri::State<Mutex<Application>>, x: usize, y: usize) -> ClickResponse {
    println!("{} {}", x, y);

    let mut state = state.lock().unwrap();
    state.on_click_button(x, y);

    ClickResponse {
        board: state
            .agent
            .agent
            .env
            .board
            .iter()
            .map(|&stone| match stone {
                Stone::Empty => 0,
                Stone::Black => 1,
                Stone::White => -1,
            })
            .collect(),
        game_status: state.env_status,
    }
}
