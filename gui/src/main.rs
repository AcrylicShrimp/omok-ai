#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

mod agent;

use agent::Agent;
use environment::{Environment, GameStatus, Stone, Turn};
use std::sync::Mutex;

struct Application {
    env: Environment,
    env_status: GameStatus,
    agent: Agent,
}

impl Application {
    pub fn new() -> Self {
        let mut env = Environment::new();
        let mut agent = Agent::new(env.clone());

        let action = agent.make_move();
        env.place_stone(action);
        agent.sync_move(&env, action);

        Self {
            env,
            env_status: GameStatus::InProgress,
            agent,
        }
    }

    fn on_click_button(&mut self, x: usize, y: usize) -> &mut Self {
        match self.env_status {
            GameStatus::InProgress => {
                if self.env.turn != Turn::White {
                    return self;
                }

                let index = y * Environment::BOARD_SIZE + x;
                self.place_stone(index);
            }
            _ => {
                // Reset game
                self.env = Environment::new();
                self.agent = Agent::new(self.env.clone());

                let action = self.agent.make_move();

                self.env.place_stone(action);
                self.agent.sync_move(&self.env, action);
                self.env_status = GameStatus::InProgress;
            }
        }

        self
    }

    fn place_stone(&mut self, index: usize) {
        self.env_status = match self.env.place_stone(index) {
            Some(status) => status,
            None => return,
        };

        if self.env_status != GameStatus::InProgress {
            return;
        }

        let action = self.agent.make_move();
        self.env_status = self.env.place_stone(action).unwrap();
        self.agent.sync_move(&self.env, action);
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
