mod plot;
mod trainer;
mod utils;
mod config;

use tensorflow::Status;
use trainer::Trainer;
use clap::{Arg,Command};

fn cli() -> Command {
    Command::new("omok-ai")
        .arg(
            Arg::new("config")
                .help("Name of the config file")
                .short('c')
                .long("config")
                .default_value("default"),
        )
}

fn main() -> Result<(), Status> {
    let args = cli().get_matches();
    let mut train = Trainer::new()?;

    let config_name = args.get_one::<String>("config").unwrap();
    let config = config::Config::new(config_name);
    train.train(10_000)?;
    Ok(())
}
