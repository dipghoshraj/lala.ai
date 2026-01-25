
fn main(){
    cc::Build::new().cpp(true)
    .files([
        "llama.cpp/llama.cpp",
        "llama.cpp/ggml.c",
        "llama.cpp/ggml-alloc.c",
        "llama.cpp/ggml-backend.c",
    ]).include("llama.cpp").flag("-O3").compile("llama");

    println!("cargo:rerun-if-changed=llama.cpp/");

}