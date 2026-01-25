use std::ffi::c_void;

#[repr(C)] pub struct llama_model;
#[repr(C)] pub struct llama_context;


pub type llama_token = i32;

#[repr(C)]
pub struct llama_model_params {
    pub n_gpu_layers: i32,
}
impl Default for llama_model_params {
    fn default() -> Self { Self { n_gpu_layers: 0 } }
}


#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_threads: u32,
}
impl Default for llama_context_params {
    fn default() -> Self { Self { n_ctx: 4096, n_threads: 8 } }
}


extern "C" {
    pub fn llama_backend_init(numa: bool);
    pub fn llama_load_model_from_file(path: *const i8, params: llama_model_params) -> *mut llama_model;
    pub fn llama_new_context_with_model(model: *mut llama_model, params: llama_context_params) -> *mut llama_context;
    pub fn llama_tokenize(ctx: *mut llama_context, text: *const i8, tokens: *mut llama_token, n_tokens: i32, add_bos: bool, special: bool) -> i32;
    pub fn llama_decode(ctx: *mut llama_context, batch: *mut llama_batch) -> i32;
    pub fn llama_get_logits(ctx: *mut llama_context) -> *const f32;
    pub fn llama_n_vocab(ctx: *mut llama_context) -> i32;
    pub fn llama_token_to_piece(ctx: *mut llama_context, token: llama_token, buf: *mut i8, len: i32) -> i32;
    pub fn llama_token_eos(ctx: *mut llama_context) -> llama_token;
    pub fn llama_sample_temp(ctx: *mut llama_context, cands: *mut llama_token_data_array, temp: f32);
    pub fn llama_sample_top_p(ctx: *mut llama_context, cands: *mut llama_token_data_array, top_p: f32, min_keep: usize);
    pub fn llama_sample_token(ctx: *mut llama_context, cands: *mut llama_token_data_array) -> llama_token;
}
