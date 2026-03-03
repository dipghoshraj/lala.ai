use std::io::{self, Write};
use crate::agent::model::ModelWrapper;
use llama_cpp::standard_sampler::StandardSampler;

pub fn run(model_path: &str) -> anyhow::Result<()> {
    let model = ModelWrapper::load(model_path)?;
    let mut session = model.create_session()?;


    println!("🦙 lala-agent ready (/exit to quit)");

    loop {
        print!(">> ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input == "/exit" {
            break;
        }

        println!("{}", input);

        let prompt = format!(
            "<s>[INST]\nYou are a friendly AI assistant. \
        Explain things clearly and naturally. Respond in full sentences and use emojis occasionally.\n\n{}\n[/INST]",
            input
        );
        // let prompt = format!("{} {}\n[/INST]", system_prompt, input);
        session.session.advance_context(&prompt)?;

        // let response = session.complete(input, 512)?;
        let mut stream = session.session.start_completing_with(StandardSampler::default(), 512)?;
        println!("--- response ---");

        let mut buffer = String::new();
        while let Some(token) = stream.next_token() {
            let text = model.model.token_to_piece(token);
            buffer.push_str(&text);

            // Stop when stop token appears
            if buffer.contains("[/INST]") {
                break;
            }

            print!("{}", text);
            io::stdout().flush().unwrap();
        }
        println!();
    }

    Ok(())
}
