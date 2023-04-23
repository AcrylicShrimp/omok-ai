#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use std::sync::Mutex;
use tauri::Manager;

use crate::environment::*;
use crate::model::*;

mod environment;
mod model;

struct Application {
    env: Environment,
    env_status: GameStatus,
    model: Model,
}

impl Application {
    pub fn new() -> Self {
        let mut env = Environment::new();
        let mut model = Model::load();

        let action = model.make_move(&env);
        env.place_stone(action);

        Self {
            env,
            env_status: GameStatus::InProgress,
            model,
        }
    }

    fn on_click_button(&mut self, x: usize, y: usize) -> &mut Self {
        match self.env_status {
            GameStatus::InProgress => {
                let index = y * Environment::BOARD_SIZE + x;

                if self.env.turn != Turn::White || !self.env.legal_moves[index] {
                    return self;
                }

                self.place_stone(index);
            }
            _ => {
                // Reset game
                self.env = Environment::new();

                let action = self.model.make_move(&self.env);
                self.env.place_stone(action);
                self.env_status = GameStatus::InProgress;
            }
        }

        self
    }

    fn place_stone(&mut self, index: usize) {
        self.env_status = self.env.place_stone(index);

        if self.env_status != GameStatus::InProgress {
            return;
        }

        let action = self.model.make_move(&self.env);
        self.env_status = self.env.place_stone(action);
    }
}

#[derive(serde::Serialize)]
struct ClickResponse {
    board: Vec<f32>,
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
        board: state.env.board.into(),
        game_status: state.env_status,
    }
}
