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
    conf::{model::InferenceConfig, which::Which},
    model::{loader::LoadModel, prompt::handle_user_input},
    runner::text_generation::TextGeneration,
};
fn main() {
    // use deault model
    run_inference(None);

    // use selected zephyr model
    // run_inference(Some(Which::Zephyr7bBeta));
}

fn run_inference(which: Option<Which>) {
    let mut config = InferenceConfig::default();
    let stop_flag = Arc::new(AtomicBool::new(false));

    // uncomment to simulate stop flag
    // let stop_flag_clone = stop_flag.clone();
    // simulate_stop_flag(stop_flag_clone);

    if let Some(which) = which {
        config.which = which;
    }

    println!("Config: {:?}", config);

    let model = LoadModel::load_model(&config).expect("Error loading model");

    let prompt = "How does this work?".to_string();
    // TODO:: currently defaulting to chat prompt type - need to fix
    let prompt = handle_user_input(config.which, &prompt).unwrap();

    let mut pipeline = TextGeneration::new(
        model.weights,
        model.device,
        model.tokenizer,
        config.repeat_penalty,
        config.repeat_last_n,
        config.seed,
        config.temperature,
        config.top_p,
    );

    let (response, prompt_tokens, prompt_secs, sampled, sampled_secs) = pipeline
        .run(prompt, config.sample_len, &config.which, stop_flag, |t| {
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
