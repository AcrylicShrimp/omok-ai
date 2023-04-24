use crate::environment::Environment;
use std::path::Path;
use tensorflow::{
    Operation, SavedModelBundle, Scope, Session, SessionOptions, SessionRunArgs, Tensor,
};

pub struct Model {
    pub session: Session,
    pub op_input: Operation,
    pub op_output: Operation,
}

impl Model {
    pub fn load() -> Self {
        let mut scope = Scope::new_root_scope();
        let bundle = SavedModelBundle::load(
            &SessionOptions::new(),
            &["serve", "train"],
            &mut scope.graph_mut(),
            Path::new("saves").join("model"),
        )
        .unwrap();

        let op_input = scope.graph().operation_by_name_required("input").unwrap();
        let op_output = scope.graph().operation_by_name_required("output").unwrap();

        Model {
            session: bundle.session,
            op_input,
            op_output,
        }
    }

    pub fn eval_action_values(&mut self, env: &Environment) -> Vec<f32> {
        let mut board = [0i32; Environment::BOARD_SIZE * Environment::BOARD_SIZE];
        env.copy_board(env.turn, &mut board);

        let mut input = Tensor::new(&[1, 19, 19, 2]);

        let mut indexed_board = [0i32; Environment::BOARD_SIZE * Environment::BOARD_SIZE * 2];
        for y in 0..Environment::BOARD_SIZE {
            for x in 0..Environment::BOARD_SIZE {
                indexed_board[(y * Environment::BOARD_SIZE + x) * 2] =
                    (y * Environment::BOARD_SIZE + x) as i32;
                indexed_board[(y * Environment::BOARD_SIZE + x) * 2 + 1] =
                    board[y * Environment::BOARD_SIZE + x];
            }
        }

        input.copy_from_slice(&indexed_board[..]);

        let mut eval_run_args = SessionRunArgs::new();
        eval_run_args.add_feed(&self.op_input, 0, &input);

        let fetch_token = eval_run_args.request_fetch(&self.op_output, 0);
        self.session.run(&mut eval_run_args).unwrap();

        let output = eval_run_args.fetch::<f32>(fetch_token).unwrap();
        output.to_vec()
    }

    pub fn make_move(&mut self, env: &Environment) -> usize {
        self.eval_action_values(env)
            .iter()
            .enumerate()
            .filter(|(i, _)| env.legal_moves[*i])
            .max_by(|(_, a), (_, b)| f32::total_cmp(a, b))
            .unwrap()
            .0
    }
}
