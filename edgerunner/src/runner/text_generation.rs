use std::{
    io::Write,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::{generation::LogitsProcessor, models::quantized_llama::ModelWeights};
use tokenizers::Tokenizer;

use crate::{conf::which::Which, model::prompt::GeneratedPrompt};

use super::token_output_stream::TokenOutputStream;

pub struct TextGeneration {
    model: ModelWeights,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: ModelWeights,
        device: Device,
        tokenizer: Tokenizer,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            device,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
        }
    }

    fn check_stop_flag(&self, stop_flag: &Arc<AtomicBool>) -> Result<()> {
        if stop_flag.load(Ordering::SeqCst) {
            println!("Operation was stopped by the user");
            Err(anyhow::Error::msg("Operation was stopped by the user"))
        } else {
            Ok(())
        }
    }
    pub fn run(
        &mut self,
        prompt: GeneratedPrompt,
        sample_len: usize,
        which: &Which,
        stop_flag: Arc<AtomicBool>,
        on_token: impl Fn(&str),
    ) -> Result<(String, f64, f64, f64, f64)> {
        // check if model is available
        if !which.is_available() {
            return Err(anyhow::Error::msg(format!(
                "Model {:?} is not available",
                which
            )));
        }

        self.tokenizer.clear();

        let pre_prompt_tokens: Vec<u32> = vec![];
        let prompt_str = prompt.as_str();
        println!("Prompt: {}", prompt_str);
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;

        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();

        let to_sample = sample_len.saturating_sub(1);
        let prompt_tokens = if prompt_tokens.len() + to_sample > 4096 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - 4096;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };

        self.check_stop_flag(&stop_flag)?;

        let mut all_tokens = vec![];

        let start_prompt_processor = std::time::Instant::now();
        let mut next_token = {
            let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            self.logits_processor.sample(&logits)?
        };

        self.check_stop_flag(&stop_flag)?;

        let prompt_dt = start_prompt_processor.elapsed();
        all_tokens.push(next_token);
        if let Some(t) = self.tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        let eos_token = if which.is_open_chat() {
            "<|end_of_turn|>"
        } else {
            "</s>"
        };

        let eos_token = *self
            .tokenizer
            .tokenizer()
            .get_vocab(true)
            .get(eos_token)
            .unwrap();

        let start_post_prompt = std::time::Instant::now();

        let mut full_response = "".to_string();
        let mut sampled = 0;
        for index in 0..to_sample {
            self.check_stop_flag(&stop_flag)?;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, prompt_tokens.len() + index)?;
            let logits = logits.squeeze(0)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = all_tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &all_tokens[start_at..],
                )?
            };

            next_token = self.logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                on_token(&t);
                full_response += &t;
            }
            sampled += 1;
            if next_token == eos_token {
                break;
            }
            self.check_stop_flag(&stop_flag)?;
        }

        if let Some(rest) = self
            .tokenizer
            .decode_rest()
            .map_err(candle_core::Error::msg)?
        {
            on_token(&rest);
            full_response += &rest;
        }

        std::io::stdout().flush()?;
        let dt = start_post_prompt.elapsed();

        Ok((
            full_response,
            prompt_tokens.len() as f64,
            prompt_dt.as_secs_f64(),
            sampled as f64,
            dt.as_secs_f64(),
        ))
    }
}
