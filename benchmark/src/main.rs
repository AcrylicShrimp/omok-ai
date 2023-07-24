use agent::Agent;
use environment::GameStatus;

mod agent;

const LEFT_AGENT_PATH: &str = "saves/alpha-zero";
const RIGHT_AGENT_PATH: &str = "saves/alpha-zero-other";

const MCTS_COUNT: usize = 800;
const MCTS_BATCH_SIZE: usize = 8;

const GAME_COUNT: usize = 100;

fn main() {
    let mut left = Agent::new(LEFT_AGENT_PATH);
    let mut right = Agent::new(RIGHT_AGENT_PATH);

    println!("Playing {} games...", GAME_COUNT);

    let mut left_wins = 0;
    let mut right_wins = 0;
    let mut draws = 0;

    for _ in 0..GAME_COUNT / 2 {
        match play_game(&mut left, &mut right) {
            1 => {
                left_wins += 1;
            }
            -1 => {
                right_wins += 1;
            }
            _ => {
                draws += 1;
            }
        }

        left.reset();
        right.reset();
    }

    for _ in 0..GAME_COUNT / 2 {
        match play_game(&mut right, &mut left) {
            1 => {
                right_wins += 1;
            }
            -1 => {
                left_wins += 1;
            }
            _ => {
                draws += 1;
            }
        }

        left.reset();
        right.reset();
    }

    println!("Left wins: {}", left_wins);
    println!("Right wins: {}", right_wins);
    println!("Draws: {}", draws);
}

fn play_game(left: &mut Agent, right: &mut Agent) -> i32 {
    loop {
        let left_action = left.make_move(MCTS_COUNT, MCTS_BATCH_SIZE);
        if let Some(status) = left.agent.play_action(left_action) {
            match status {
                GameStatus::InProgress => {}
                GameStatus::Draw => {
                    return 0;
                }
                GameStatus::BlackWin => {
                    return 1;
                }
                GameStatus::WhiteWin => {
                    return -1;
                }
            }
        }

        right
            .agent
            .ensure_action_exists(left_action, &right.agent_model, &right.session)
            .unwrap();
        right.agent.play_action(left_action).unwrap();

        let right_action = right.make_move(MCTS_COUNT, MCTS_BATCH_SIZE);
        if let Some(status) = right.agent.play_action(right_action) {
            match status {
                GameStatus::InProgress => {}
                GameStatus::Draw => {
                    return 0;
                }
                GameStatus::BlackWin => {
                    return 1;
                }
                GameStatus::WhiteWin => {
                    return -1;
                }
            }
        }

        left.agent
            .ensure_action_exists(right_action, &left.agent_model, &left.session)
            .unwrap();
        left.agent.play_action(right_action).unwrap();
    }
}
