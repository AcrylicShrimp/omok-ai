mod environment;
mod model;
mod train;

use model::create_model;
use tensorflow::Status;
use train::TrainSession;

fn main() -> Result<(), Status> {
    let mut train_session = TrainSession::new(create_model()?)?;
    train_session.init()?;
    train_session.warm_up();
    train_session.perform_episodes(1000_0000)?;
    Ok(())
}
