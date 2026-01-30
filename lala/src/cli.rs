use std::io::{self, Write};
use crate::agent::model::ModelWrapper;
use llama_cpp::standard_sampler::StandardSampler;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};


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
        

        let prompt = if input.is_empty() {
            "User: Hello, who are you?\nAssistant:".to_string()
        } else {
            format!("User: {}\nAssistant:", input)
        };
        session.session.advance_context(&prompt)?;

        // let response = session.complete(input, 512)?;
        let mut stream = session.session.start_completing_with(StandardSampler::default(), 512)?;
        println!("--- response ---");
        while let Some(token) = stream.next_token() {


            let text = model.model.token_to_piece(token);
            print!("{}", text);

            io::stdout().flush().unwrap();
        }
        println!();
    }

    Ok(())
}
