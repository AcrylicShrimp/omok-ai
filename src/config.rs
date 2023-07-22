use serde::{Deserialize,Serialize};

use std::{
    fs,
    path::Path,
    default::Default,
};

use toml;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub parameters: Parameters,
    //pub enviroment: Enviroment,    // TODO: implement enviroment or make it separate from the
                                     //       config file
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Parameters {
    // Trainer Parameters
    pub model_name: String,
    pub replay_memory_size: usize,
    pub episode_count: usize,
    pub evaluate_count: usize,
    pub evaluate_batch_size: usize,
    pub training_count: usize,
    pub training_batch_size: usize,
    pub test_evaluate_count: usize,
    pub temperature: f32,
    pub temperature_threshold: usize,

    //Plot Parameters
    pub max_losses: usize,
}

impl Config {
    pub fn new(name: &str) -> Self {
        let path_base = Path::new("config");
        let path_file = path_base.join(name.to_owned() + ".toml");

        if !path_base.exists() {
            fs::create_dir_all(path_base).unwrap();
        }
        if !path_file.exists() {
            Self::save(path_file.to_str().unwrap().to_string()).unwrap();
        }
            
        match Self::load(path_file.to_str().unwrap().to_string()){
            Ok(_config) => _config,
            Err(_e) => {
                println!("Error loading config, Fallback to default");
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
    fn default() -> Parameters
    {
        Parameters {
            model_name: "alpha-zero".to_owned(),
            replay_memory_size: 600_000,
            episode_count: 50,
            evaluate_count: 600,
            evaluate_batch_size: 16,
            training_count: 600,
            training_batch_size: 128,
            test_evaluate_count: 800,
            temperature: 1.0,
            temperature_threshold: 30,

            max_losses: 1024 * 1024,
        }
    }
}
 
