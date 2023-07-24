use serde::{Deserialize, Serialize};
use std::fmt::Display;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stone {
    Empty,
    Black,
    White,
}

impl Display for Stone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "-"),
            Self::Black => write!(f, "X"),
            Self::White => write!(f, "O"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Turn {
    Black,
    White,
}

impl Turn {
    pub fn opponent(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
}

impl Display for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Black => write!(f, "Black"),
            Self::White => write!(f, "White"),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GameStatus {
    InProgress,
    Draw,
    BlackWin,
    WhiteWin,
}

impl GameStatus {
    pub fn is_terminal(self) -> bool {
        match self {
            GameStatus::InProgress => false,
            _ => true,
        }
    }
}

#[derive(Clone)]
pub struct Environment {
    pub turn: Turn,
    pub legal_move_count: u16,
    pub board: [Stone; Self::BOARD_SIZE * Self::BOARD_SIZE],
}

impl Environment {
    pub const BOARD_SIZE: usize = 15;
    pub const SERIAL_STONE_COUNT: usize = 5;

    pub fn new() -> Self {
        Environment {
            turn: Turn::Black,
            legal_move_count: (Self::BOARD_SIZE * Self::BOARD_SIZE) as u16,
            board: [Stone::Empty; Self::BOARD_SIZE * Self::BOARD_SIZE],
        }
    }

    pub fn encode_board(&self, turn: Turn, mut dst: impl AsMut<[f32]>) {
        let dst = dst.as_mut();
        dst.fill(0f32);

        let black_offset = match turn {
            Turn::Black => 0,
            Turn::White => 1,
        };
        let white_offset = match turn {
            Turn::Black => 1,
            Turn::White => 0,
        };

        for (index, &stone) in self.board.iter().enumerate() {
            let offset = match stone {
                Stone::Empty => continue,
                Stone::Black => black_offset,
                Stone::White => white_offset,
            };
            dst[index * 2 + offset] = 1f32;
        }
    }

    pub fn place_stone(&mut self, index: usize) -> Option<GameStatus> {
        if self.board[index] != Stone::Empty {
            return None;
        }

        self.legal_move_count -= 1;
        self.board[index] = match self.turn {
            Turn::Black => Stone::Black,
            Turn::White => Stone::White,
        };

        let horizontal_count = 1
            + self.count_serial_stones(
                self.turn,
                index,
                &[(-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0)],
            )
            + self.count_serial_stones(self.turn, index, &[(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);
        let vertical_count = 1
            + self.count_serial_stones(
                self.turn,
                index,
                &[(0, -1), (0, -2), (0, -3), (0, -4), (0, -5)],
            )
            + self.count_serial_stones(self.turn, index, &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);
        let diagonal_lt_rb_count = 1
            + self.count_serial_stones(
                self.turn,
                index,
                &[(-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5)],
            )
            + self.count_serial_stones(self.turn, index, &[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]);
        let diagonal_lb_rt_count =
            1 + self.count_serial_stones(
                self.turn,
                index,
                &[(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5)],
            ) + self.count_serial_stones(
                self.turn,
                index,
                &[(1, -1), (2, -2), (3, -3), (4, -4), (5, -5)],
            );

        let turn = self.turn;
        self.turn = self.turn.opponent();

        Some(
            if horizontal_count == Self::SERIAL_STONE_COUNT
                || vertical_count == Self::SERIAL_STONE_COUNT
                || diagonal_lt_rb_count == Self::SERIAL_STONE_COUNT
                || diagonal_lb_rt_count == Self::SERIAL_STONE_COUNT
            {
                match turn {
                    Turn::Black => GameStatus::BlackWin,
                    Turn::White => GameStatus::WhiteWin,
                }
            } else if self.legal_move_count == 0 {
                GameStatus::Draw
            } else {
                GameStatus::InProgress
            },
        )
    }

    fn count_serial_stones(&self, turn: Turn, index: usize, offset: &[(isize, isize)]) -> usize {
        let stone = match turn {
            Turn::Black => Stone::Black,
            Turn::White => Stone::White,
        };

        let x = (index % Self::BOARD_SIZE) as isize;
        let y = (index / Self::BOARD_SIZE) as isize;
        let mut count = 0;

        for &(offset_x, offset_y) in offset {
            let x = x + offset_x;
            let y = y + offset_y;
            if x < 0 || Self::BOARD_SIZE as isize <= x || y < 0 || Self::BOARD_SIZE as isize <= y {
                break;
            }

            if self.board[(y * Self::BOARD_SIZE as isize + x) as usize] != stone {
                break;
            }

            count += 1;
        }

        count
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_place_stone() {
        let mut env = Environment::new();
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(0), Some(GameStatus::InProgress));
        assert_eq!(env.board[0], Stone::Black);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(1), Some(GameStatus::InProgress));
        assert_eq!(env.board[1], Stone::White);
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(2), Some(GameStatus::InProgress));
        assert_eq!(env.board[2], Stone::Black);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(3), Some(GameStatus::InProgress));
        assert_eq!(env.board[3], Stone::White);
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(4), Some(GameStatus::InProgress));
        assert_eq!(env.board[4], Stone::Black);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(5), Some(GameStatus::InProgress));
        assert_eq!(env.board[5], Stone::White);
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(6), Some(GameStatus::InProgress));
        assert_eq!(env.board[6], Stone::Black);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(7), Some(GameStatus::InProgress));
        assert_eq!(env.board[7], Stone::White);
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(8), Some(GameStatus::InProgress));
        assert_eq!(env.board[8], Stone::Black);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(9), Some(GameStatus::InProgress));
        assert_eq!(env.board[9], Stone::White);
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(10), Some(GameStatus::InProgress));
        assert_eq!(env.board[10], Stone::Black);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(11), Some(GameStatus::InProgress));
        assert_eq!(env.board[11], Stone::White);
        assert_eq!(env.turn, Turn::Black);
    }

    #[test]
    fn test_game_ending_horizontal() {
        let mut env = Environment::new();

        assert_eq!(
            env.place_stone(0 + 0 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );
        assert_eq!(
            env.place_stone(0 + 1 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );

        assert_eq!(
            env.place_stone(1 + 0 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );
        assert_eq!(
            env.place_stone(1 + 1 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );

        assert_eq!(
            env.place_stone(2 + 0 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );
        assert_eq!(
            env.place_stone(2 + 1 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );

        assert_eq!(
            env.place_stone(3 + 0 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );
        assert_eq!(
            env.place_stone(3 + 1 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );

        assert_eq!(
            env.place_stone(4 + 0 * Environment::BOARD_SIZE),
            Some(GameStatus::BlackWin)
        );
    }

    #[test]
    fn test_game_ending_vertical() {
        let mut env = Environment::new();

        assert_eq!(
            env.place_stone(0 + 0 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );
        assert_eq!(
            env.place_stone(2 + 0 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );

        assert_eq!(
            env.place_stone(0 + 1 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );
        assert_eq!(
            env.place_stone(2 + 1 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );

        assert_eq!(
            env.place_stone(0 + 2 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );
        assert_eq!(
            env.place_stone(2 + 2 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );

        assert_eq!(
            env.place_stone(0 + 3 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );
        assert_eq!(
            env.place_stone(2 + 3 * Environment::BOARD_SIZE),
            Some(GameStatus::InProgress)
        );

        assert_eq!(
            env.place_stone(0 + 4 * Environment::BOARD_SIZE),
            Some(GameStatus::BlackWin)
        );
    }

    #[test]
    fn test_game_ending_lt_rb() {
        let mut env = Environment::new();

        for index in 0..Environment::BOARD_SIZE * (Environment::SERIAL_STONE_COUNT - 1) {
            env.place_stone(index);
        }

        assert_eq!(
            env.place_stone(Environment::BOARD_SIZE * (Environment::SERIAL_STONE_COUNT - 1) + 4),
            Some(GameStatus::BlackWin)
        );
    }

    #[test]
    fn test_game_ending_lb_rt() {
        let mut env = Environment::new();

        for index in 0..Environment::BOARD_SIZE * (Environment::SERIAL_STONE_COUNT - 1) {
            env.place_stone(index);
        }

        assert_eq!(
            env.place_stone(Environment::BOARD_SIZE * (Environment::SERIAL_STONE_COUNT - 1)),
            Some(GameStatus::BlackWin)
        );
    }

    #[test]
    fn encoding_0() {
        let mut env = Environment::new();
        env.place_stone(0);

        let mut encoded = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2];
        env.encode_board(Turn::Black, &mut encoded);

        let mut expected = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2];
        expected[0] = 1.0;

        assert_eq!(encoded, expected);
    }

    #[test]
    fn encoding_1() {
        let mut env = Environment::new();
        env.place_stone(0);
        env.place_stone(10);
        env.place_stone(2);
        env.place_stone(30);

        let mut encoded = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2];
        env.encode_board(Turn::Black, &mut encoded);

        let mut expected = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2];
        expected[0 * 2 + 0] = 1.0;
        expected[10 * 2 + 1] = 1.0;
        expected[2 * 2 + 0] = 1.0;
        expected[30 * 2 + 1] = 1.0;

        assert_eq!(encoded, expected);
    }

    #[test]
    fn encoding_2() {
        let mut env = Environment::new();
        env.place_stone(0);
        env.place_stone(10);
        env.place_stone(2);
        env.place_stone(30);

        let mut encoded = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2];
        env.encode_board(Turn::White, &mut encoded);

        let mut expected = [0f32; Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2];
        expected[0 * 2 + 1] = 1.0;
        expected[10 * 2 + 0] = 1.0;
        expected[2 * 2 + 1] = 1.0;
        expected[30 * 2 + 0] = 1.0;

        assert_eq!(encoded, expected);
    }
}
