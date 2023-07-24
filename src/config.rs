use serde::{Deserialize, Serialize};
use std::{default::Default, fs, path::Path};
use toml;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub parameters: Parameters,
    // TODO: implement enviroment or make it separate from the config file.
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Parameters {
    // Network Parameters
    pub model_name: String,

    // Self Play Parameters
    pub replay_memory_size: usize,
    pub episode_count: usize,
    pub evaluate_count: usize,
    pub evaluate_batch_size: usize,
    pub epsilon: f32,
    pub alpha: f32,
    pub temperature: f32,
    pub temperature_threshold: usize,

    // Training Parameters
    pub parameter_update_count: usize,
    pub parameter_update_batch_size: usize,

    // Test Play Parameters
    pub test_evaluate_count: usize,

    // Plot Parameters
    pub max_losses: usize,
}

impl Config {
    pub fn new(name: &str) -> Self {
        let path_base = Path::new("config");
        let path_file = path_base.join(Path::new(name).with_extension("toml"));

        if !path_base.exists() {
            fs::create_dir_all(path_base).unwrap();
        }

        if !path_file.exists() {
            Self::save(path_file.to_str().unwrap().to_string()).unwrap();
        }

        match Self::load(path_file.to_str().unwrap().to_string()) {
            Ok(config) => config,
            Err(_) => {
                println!("failed to load configuration, using default config");
                Self::default()
            }
        }
    }

    pub fn load(path: String) -> Result<Self, std::io::Error> {
        let path = Path::new(&path);
        let contents = fs::read_to_string(path)?;
        let config = toml::from_str(&contents).unwrap();
        Ok(config)
    }

    pub fn save(path: String) -> Result<(), std::io::Error> {
        let path = Path::new(&path);
        let contents = toml::to_string(&Config::default()).unwrap();
        fs::write(path, contents)?;
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            parameters: Parameters::default(),
        }
    }
}

impl Default for Parameters {
    fn default() -> Parameters {
        Parameters {
            // Network Parameters
            model_name: "alpha-zero".to_owned(),

            // Self Play Parameters
            replay_memory_size: 600_000,
            episode_count: 50,
            evaluate_count: 600,
            evaluate_batch_size: 16,
            epsilon: 0.25,
            alpha: 0.03,
            temperature: 1.0,
            temperature_threshold: 30,

            // Training Parameters
            parameter_update_count: 600,
            parameter_update_batch_size: 128,

            // Test Play Parameters
            test_evaluate_count: 800,

            // Plot Parameters
            max_losses: 1024 * 1024,
        }
    }
}
