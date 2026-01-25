mod chat;

use chat::Llama;


fn main() {
    println!("Hello, world!");

    let mut llama = Llama::load("model.bin")?;
    let response = llama.chat("Hello, how are you?")?;
    println!("Assistant: {}", response);
    Ok(())
}
