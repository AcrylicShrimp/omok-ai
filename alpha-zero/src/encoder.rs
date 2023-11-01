use environment::{Environment, Turn};
use tensorflow::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EnvTurnMode {
    Player,
    Opponent,
}

pub fn encode_nn_input<'a>(
    input_count: usize,
    env_turn_mode: EnvTurnMode,
    env_iter: impl Iterator<Item = &'a Environment>,
) -> Tensor<f32> {
    let mut input = Tensor::new(&[
        input_count as _,
        Environment::BOARD_SIZE as _,
        Environment::BOARD_SIZE as _,
        3,
    ]);

    for (index, env) in env_iter.enumerate() {
        env.encode_board(
            match env_turn_mode {
                EnvTurnMode::Player => env.turn,
                EnvTurnMode::Opponent => env.turn.opponent(),
            },
            &mut input[index * Environment::BOARD_SIZE * Environment::BOARD_SIZE * 3
                ..index * Environment::BOARD_SIZE * Environment::BOARD_SIZE * 3
                    + Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2],
        );

        // encode the turn as a single layer
        let value = match env.turn {
            Turn::Black => 1f32,
            Turn::White => 0f32,
        };

        input[index * Environment::BOARD_SIZE * Environment::BOARD_SIZE * 3
            + Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2
            ..(index + 1) * Environment::BOARD_SIZE * Environment::BOARD_SIZE * 3]
            .fill(value);
    }

    input
}

pub fn encode_nn_targets<'a, const N: usize>(
    input_count: usize,
    pi_iter: impl Iterator<Item = &'a [f32; N]>,
    z_iter: impl Iterator<Item = f32>,
) -> (Tensor<f32>, Tensor<f32>) {
    let mut policy_target = Tensor::new(&[
        input_count as _,
        Environment::BOARD_SIZE as _,
        Environment::BOARD_SIZE as _,
    ]);
    let mut value_target = Tensor::new(&[input_count as _, 1]);

    for (index, (z, pi)) in (z_iter.zip(pi_iter)).enumerate() {
        policy_target[index * Environment::BOARD_SIZE * Environment::BOARD_SIZE
            ..(index + 1) * Environment::BOARD_SIZE * Environment::BOARD_SIZE]
            .copy_from_slice(pi);
        value_target[index] = z;
    }

    (policy_target, value_target)
}
