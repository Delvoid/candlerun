use config::{Config, ConfigError};
use dirs::config_dir;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::model::InferenceConfig;

const CONFIG_FOLDER_PATH: &str = "/edgerunner_test";
const CONFIG_FILE_PATH: &str = "/config";

#[derive(Debug, Deserialize, Serialize)]
pub struct HyperspaceConfig {
    pub model: InferenceConfig,
}

impl HyperspaceConfig {
    pub fn new(config_file_path_without_extension: &str) -> Result<Self, ConfigError> {
        let settings = Config::builder()
            .add_source(config::File::with_name(config_file_path_without_extension))
            .build()?;
        settings.try_deserialize()
    }

    // load the config file from the default location or create it if it does not exist
    pub fn load_or_create_default_config_file() -> Result<Self, ConfigError> {
        let config_folder_path = format!(
            "{}{}",
            config_dir().unwrap().to_str().unwrap(),
            CONFIG_FOLDER_PATH
        );
        let config_file_path = format!("{}{}", config_folder_path, CONFIG_FILE_PATH);

        Self::create_default_config_file(&config_folder_path, &config_file_path);
        Self::new(&config_file_path)
    }

    // Utility function to create the default config file if it does not exist.
    fn create_default_config_file(folder_path: &str, file_path: &str) {
        let config_default = include_str!("../../assets/config.toml");
        let file_with_extension_toml = format!("{}.toml", file_path);

        if !Path::new(folder_path).exists() {
            fs::create_dir_all(folder_path).expect("Error creating folder!");
        }

        if Path::new(&file_with_extension_toml).exists() {
            return;
        }

        let mut config_file =
            File::create(&file_with_extension_toml).expect("Error creating config file!");
        config_file
            .write_all(config_default.as_bytes())
            .expect("Error writing into the config file!");
    }
}
