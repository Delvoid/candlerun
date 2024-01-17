use std::io::Write;

use edgerunner_test::{
    get_args,
    model::{loader::LoadModel, prompt::handle_user_input},
    runner::text_generation::TextGeneration,
};

fn main() {
    if let Err(e) = get_args().map(|args| {
        println!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );
        // TODO:: currently defaulting to chat prompt type
        let prompt = handle_user_input(args.config.which, &args.prompt).unwrap();

        // if model is not set in config it uses the which model details
        let model = LoadModel::load_model(&args.config).unwrap();

        let mut pipeline = TextGeneration::new(
            model.weights,
            model.device,
            model.tokenizer,
            args.config.repeat_penalty,
            args.config.repeat_last_n,
            args.config.seed,
            args.config.temperature,
            args.config.top_p,
        );
        let (response, prompt_tokens, prompt_secs, sampled, sampled_secs) = pipeline
            .run(prompt, args.config.sample_len, &args.config.which, |t| {
                print!("{t}");
                std::io::stdout().flush().unwrap();
            })
            .unwrap();
        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens,
            prompt_tokens / prompt_secs,
        );
        println!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled / sampled_secs
        );
        println!("Full response was {} bytes", response.as_bytes().len());
    }) {
        println!("Error: {}", e);
    }
}
