#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

mod environment;
mod model;

use crate::{environment::Environment, model::Model};
use eframe::{run_native, App, NativeOptions};
use egui::{vec2, Button, Color32, PointerButton, RichText, Rounding, Widget};
use environment::{GameStatus, Turn};

fn main() -> Result<(), eframe::Error> {
    let options = NativeOptions {
        initial_window_size: Some(vec2(800.0, 600.0)),
        ..Default::default()
    };
    run_native(
        "Omok AI",
        options,
        Box::new(|_cc| Box::new(Application::new())),
    )
}

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

    fn on_click_button(&mut self, x: usize, y: usize) {
        match self.env_status {
            GameStatus::InProgress => {
                let index = y * Environment::BOARD_SIZE + x;

                if self.env.turn != Turn::White || !self.env.legal_moves[index] {
                    return;
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

impl App for Application {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.label(
                RichText::new(match self.env_status {
                    GameStatus::InProgress => "Your move!",
                    GameStatus::Draw => "Draw",
                    GameStatus::BlackWin => "AI wins!",
                    GameStatus::WhiteWin => "You win!",
                })
                .heading()
                .color(Color32::WHITE),
            );

            for y in 0..Environment::BOARD_SIZE {
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing = vec2(0.0, 0.0);
                        for x in 0..Environment::BOARD_SIZE {
                            let stone = self.env.board[y * Environment::BOARD_SIZE + x];

                            let color = if f32::abs(stone) < f32::EPSILON {
                                Color32::WHITE
                            } else if 0f32 < stone {
                                Color32::BLUE
                            } else {
                                Color32::from_rgb(255u8, 127u8, 0)
                            };

                            let button = Button::new("")
                                .fill(color)
                                .frame(true)
                                .rounding(Rounding::none())
                                .min_size(vec2(30.0, 30.0));
                            if button.ui(ui).clicked_by(PointerButton::Primary) {
                                self.on_click_button(x, y);
                            }
                        }
                    });
                });
            }
        });
    }
}
