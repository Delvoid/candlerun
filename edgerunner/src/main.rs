use std::{
    io::Write,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

use edgerunner::{
    get_args,
    log_util::set_env_logger,
    model::{loader::LoadModel, prompt::handle_user_input},
    runner::text_generation::TextGeneration,
};
use log::debug;

fn main() {
    set_env_logger();
    let stop_flag = Arc::new(AtomicBool::new(false));

    // uncomment to simulate stop flag
    // let stop_flag_clone = stop_flag.clone();
    // simulate_stop_flag(stop_flag_clone);

    if let Err(e) = get_args().map(|args| {
        debug!(
            "avx: {}, neon: {}, simd128: {}, f16c: {}",
            candle_core::utils::with_avx(),
            candle_core::utils::with_neon(),
            candle_core::utils::with_simd128(),
            candle_core::utils::with_f16c()
        );

        // let system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman:";

        debug!("Args: {:?}", args);
        // TODO:: currently defaulting to chat prompt type
        let prompt = handle_user_input(args.config.which, &args.prompt, None).unwrap();

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
            .run(
                prompt,
                args.config.sample_len,
                &args.config.which,
                stop_flag,
                |t| {
                    print!("{t}");
                    std::io::stdout().flush().unwrap();
                },
            )
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

#[allow(dead_code)]
fn simulate_stop_flag(stop_flag: Arc<AtomicBool>) {
    // Spawn a thread to simulate external stop after a certain duration
    thread::spawn(move || {
        // Set the duration for the simulated delay (e.g., 5 seconds)
        thread::sleep(Duration::from_secs(10));
        stop_flag.store(true, Ordering::SeqCst);
        println!("Stop flag set by simulation thread");
    });
}
