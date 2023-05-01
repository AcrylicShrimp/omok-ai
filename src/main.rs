mod alpha_zero;
mod environment;

use alpha_zero::Train;
use tensorflow::Status;

fn main() -> Result<(), Status> {
    let mut train = Train::new()?;
    train.train(10_000)?;
    Ok(())
}
