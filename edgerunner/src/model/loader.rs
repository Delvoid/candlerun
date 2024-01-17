use std::path::PathBuf;

use crate::{
    conf::{model::InferenceConfig, which::Which},
    runner::device::device,
    util::format_size,
};
use anyhow::{Error, Result};
use candle_core::{
    quantized::{ggml_file, gguf_file},
    Device,
};
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::{Cache, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct LoadModel;

impl LoadModel {
    pub fn load_model(config: &InferenceConfig) -> Result<Model> {
        println!("Config: {:?}", config);
        let model_path = config.model()?;

        let model_weights = Self::load_model_weights(&model_path)?;

        let tokenizer = config.tokenizer()?;

        Ok(Model::new(tokenizer, model_weights))
    }

    fn load_model_weights(model_path: &PathBuf) -> Result<ModelWeights> {
        let mut file = std::fs::File::open(model_path)?;
        let start = std::time::Instant::now();

        match model_path.extension().and_then(|v| v.to_str()) {
            Some("gguf") => {
                let model = gguf_file::Content::read(&mut file)?;
                let mut total_size_in_bytes = 0;
                for (_, tensor) in model.tensor_infos.iter() {
                    let elem_count = tensor.shape.elem_count();
                    total_size_in_bytes +=
                        elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.blck_size();
                }
                println!(
                    "loaded {:?} tensors ({}) in {:.2}s",
                    model.tensor_infos.len(),
                    &format_size(total_size_in_bytes),
                    start.elapsed().as_secs_f32(),
                );
                Ok(ModelWeights::from_gguf(model, &mut file)?)
            }
            Some("ggml" | "bin") | Some(_) | None => {
                let model = ggml_file::Content::read(&mut file)?;
                let mut total_size_in_bytes = 0;
                for (_, tensor) in model.tensors.iter() {
                    let elem_count = tensor.shape().elem_count();
                    total_size_in_bytes +=
                        elem_count * tensor.dtype().type_size() / tensor.dtype().blck_size();
                }
                println!(
                    "loaded {:?} tensors ({}) in {:.2}s",
                    model.tensors.len(),
                    &format_size(total_size_in_bytes),
                    start.elapsed().as_secs_f32(),
                );
                println!("params: {:?}", model.hparams);

                let default_gqa = 8;
                Ok(ModelWeights::from_ggml(model, default_gqa)?)
            }
        }
    }

    pub fn check_cache(which: &Which, _revision: Option<&str>) -> Result<bool> {
        let home_dir =
            dirs::home_dir().ok_or_else(|| Error::msg("Could not find home directory."))?;

        let cache_dir = home_dir.join(".cache/huggingface/hub");

        let cache = Cache::new(cache_dir);

        let (repo_name, model_filename) = which.get_repo_and_filename();

        println!("Repo name: {}", repo_name);
        println!("Model filename: {}", model_filename);

        let repo = Repo::new(repo_name.to_string(), RepoType::Model);

        Ok(cache.repo(repo).get(model_filename).is_some())
    }
}

#[derive(Debug)]
pub struct Model {
    pub tokenizer: Tokenizer,
    pub weights: ModelWeights,
    pub device: Device,
}

impl Model {
    pub fn new(tokenizer: Tokenizer, weights: ModelWeights) -> Self {
        let device = device(true).expect("Could not get device");
        Self {
            tokenizer,
            weights,
            device,
        }
    }
}
