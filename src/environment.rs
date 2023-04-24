#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Turn {
    Black,
    White,
}

impl Turn {
    pub fn opponent(self) -> Self {
        match self {
            Turn::Black => Turn::White,
            Turn::White => Turn::Black,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GameStatus {
    InProgress,
    Draw,
    BlackWin,
    WhiteWin,
}

pub struct Environment {
    pub turn: Turn,
    pub board: [i32; Self::BOARD_SIZE * Self::BOARD_SIZE], // 19 * 19 board, 0 for empty, 1 for black, -1 for white
    pub legal_moves: [bool; Self::BOARD_SIZE * Self::BOARD_SIZE],
    pub legal_move_count: usize,
}

impl Environment {
    pub const BOARD_SIZE: usize = 19;
    pub const SERIAL_STONE_COUNT: usize = 5;

    pub fn new() -> Self {
        Environment {
            turn: Turn::Black,
            board: [0i32; Self::BOARD_SIZE * Self::BOARD_SIZE],
            legal_moves: [true; Self::BOARD_SIZE * Self::BOARD_SIZE],
            legal_move_count: Self::BOARD_SIZE * Self::BOARD_SIZE,
        }
    }

    pub fn copy_board(&self, turn: Turn, slice: &mut [i32; Self::BOARD_SIZE * Self::BOARD_SIZE]) {
        slice.copy_from_slice(&self.board[..]);

        if turn == Turn::White {
            // Make the board from the perspective of opponent
            for i in 0..slice.len() {
                if slice[i] == 1i32 {
                    slice[i] = 2i32;
                }
                if slice[i] == 2i32 {
                    slice[i] = 1i32;
                }
            }
        }
    }

    pub fn place_stone(&mut self, index: usize) -> GameStatus {
        self.board[index] = match self.turn {
            Turn::Black => 1i32,
            Turn::White => 2i32,
        };

        self.legal_moves[index] = false;
        self.legal_move_count -= 1;

        let turn = self.turn;
        self.turn = match self.turn {
            Turn::Black => Turn::White,
            Turn::White => Turn::Black,
        };

        let horizontal_count = 1
            + self.count_serial_stones(turn, index, &[(-1, 0), (-2, 0), (-3, 0), (-4, 0), (-5, 0)])
            + self.count_serial_stones(turn, index, &[(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]);
        let vertical_count = 1
            + self.count_serial_stones(turn, index, &[(0, -1), (0, -2), (0, -3), (0, -4), (0, -5)])
            + self.count_serial_stones(turn, index, &[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);
        let diagonal_lt_rb_count =
            1 + self.count_serial_stones(
                turn,
                index,
                &[(-1, -1), (-2, -2), (-3, -3), (-4, -4), (-5, -5)],
            ) + self.count_serial_stones(turn, index, &[(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]);
        let diagonal_lb_rt_count = 1
            + self.count_serial_stones(turn, index, &[(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5)])
            + self.count_serial_stones(turn, index, &[(1, -1), (2, -2), (3, -3), (4, -4), (5, -5)]);

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
        }
    }

    fn count_serial_stones(&self, turn: Turn, index: usize, offset: &[(isize, isize)]) -> usize {
        let is_black = match turn {
            Turn::Black => true,
            Turn::White => false,
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

            let stone = self.board[(y * Self::BOARD_SIZE as isize + x) as usize];
            if stone == 0i32 {
                break;
            }

            let is_black_stone = stone == 1i32;
            if is_black_stone != is_black {
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

        assert_eq!(env.place_stone(0), GameStatus::InProgress);
        assert_eq!(env.board[0], 1i32);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(1), GameStatus::InProgress);
        assert_eq!(env.board[1], 2i32);
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(2), GameStatus::InProgress);
        assert_eq!(env.board[2], 1i32);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(3), GameStatus::InProgress);
        assert_eq!(env.board[3], 2i32);
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(4), GameStatus::InProgress);
        assert_eq!(env.board[4], 1i32);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(5), GameStatus::InProgress);
        assert_eq!(env.board[5], 2i32);
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(6), GameStatus::InProgress);
        assert_eq!(env.board[6], 1i32);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(7), GameStatus::InProgress);
        assert_eq!(env.board[7], 2i32);
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(8), GameStatus::InProgress);
        assert_eq!(env.board[8], 1i32);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(9), GameStatus::InProgress);
        assert_eq!(env.board[9], 2i32);
        assert_eq!(env.turn, Turn::Black);

        assert_eq!(env.place_stone(10), GameStatus::InProgress);
        assert_eq!(env.board[10], 1i32);
        assert_eq!(env.turn, Turn::White);

        assert_eq!(env.place_stone(11), GameStatus::InProgress);
        assert_eq!(env.board[11], 2i32);
        assert_eq!(env.turn, Turn::Black);
    }

    #[test]
    fn test_game_ending_horizontal() {
        let mut env = Environment::new();

        assert_eq!(env.place_stone(0 + 0 * 19), GameStatus::InProgress);
        assert_eq!(env.place_stone(0 + 1 * 19), GameStatus::InProgress);

        assert_eq!(env.place_stone(1 + 0 * 19), GameStatus::InProgress);
        assert_eq!(env.place_stone(1 + 1 * 19), GameStatus::InProgress);

        assert_eq!(env.place_stone(2 + 0 * 19), GameStatus::InProgress);
        assert_eq!(env.place_stone(2 + 1 * 19), GameStatus::InProgress);

        assert_eq!(env.place_stone(3 + 0 * 19), GameStatus::InProgress);
        assert_eq!(env.place_stone(3 + 1 * 19), GameStatus::InProgress);

        assert_eq!(env.place_stone(4 + 0 * 19), GameStatus::BlackWin);
    }

    #[test]
    fn test_game_ending_vertical() {
        let mut env = Environment::new();

        assert_eq!(env.place_stone(0 + 0 * 19), GameStatus::InProgress);
        assert_eq!(env.place_stone(2 + 0 * 19), GameStatus::InProgress);

        assert_eq!(env.place_stone(0 + 1 * 19), GameStatus::InProgress);
        assert_eq!(env.place_stone(2 + 1 * 19), GameStatus::InProgress);

        assert_eq!(env.place_stone(0 + 2 * 19), GameStatus::InProgress);
        assert_eq!(env.place_stone(2 + 2 * 19), GameStatus::InProgress);

        assert_eq!(env.place_stone(0 + 3 * 19), GameStatus::InProgress);
        assert_eq!(env.place_stone(2 + 3 * 19), GameStatus::InProgress);

        assert_eq!(env.place_stone(0 + 4 * 19), GameStatus::BlackWin);
    }

    #[test]
    fn test_game_ending_lt_rb() {
        let mut env = Environment::new();

        for index in 0..Environment::BOARD_SIZE * (Environment::SERIAL_STONE_COUNT - 1) {
            env.place_stone(index);
        }

        assert_eq!(
            env.place_stone(Environment::BOARD_SIZE * (Environment::SERIAL_STONE_COUNT - 1) + 4),
            GameStatus::BlackWin
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
            GameStatus::BlackWin
        );
    }
}
