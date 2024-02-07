use clap::{value_parser, Arg, Command};

use conf::{model::InferenceConfig, which::Which};
use model::loader::LoadModel;

pub mod conf;
pub mod log_util;
pub mod model;
pub mod runner;
pub mod system_benchmark;
pub mod util;

#[derive(Debug)]
pub struct ArgsResult {
    pub prompt: String,
    pub config: InferenceConfig,
}

pub fn is_model_cached(which: &Which) -> bool {
    LoadModel::check_cache(which, None).unwrap()
}

pub fn get_args() -> anyhow::Result<ArgsResult> {
    let matches = Command::new("runner")
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .value_name("MODEL")
                .help("The model to use")
                .value_parser(value_parser!(Which))
                .default_value("7b-mistral-instruct"),
        )
        .arg(
            Arg::new("prompt")
                .short('p')
                .long("prompt")
                .value_name("PROMPT")
                .num_args(1..)
                .value_parser(value_parser!(String))
                .help("The prompt to use")
                .default_value("How does this work?"),
        )
        .get_matches();

    let model = matches.get_one::<Which>("model").unwrap();

    let prompt_values = matches
        .get_many::<String>("prompt")
        .unwrap()
        .cloned()
        .collect::<Vec<String>>();
    let prompt = prompt_values.join(" "); //

    Ok(ArgsResult {
        prompt,
        config: InferenceConfig {
            which: *model,
            ..Default::default()
        },
    })
}
