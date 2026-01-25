use crate::chat::ffi::*;

use anyhow::{Result, anyhow};
use std::ffi::CString;


pub struct Model {
    pub ptr: *mut llama_model,
}

impl Model {
    pub fn load(path: &str) -> Result<Self> {
        unsafe {
            let path = CString::new(path)?;
            let model = llama_load_model_from_file(path.as_ptr(), llama_model_params::default());
            if model.is_null() { return Err(anyhow!("failed to load model")); }
            Ok(Self { ptr: model })
        }
    }
}


pub struct Context {
    pub ptr: *mut llama_context,
    pub n_past: i32,
}

impl Context {
    pub fn new(model: &Model) -> Result<Self> {
        unsafe {
            let ctx = llama_new_context_with_model(model.ptr, llama_context_params::default());
            if ctx.is_null() { return Err(anyhow!("failed to create context")); }
            Ok(Self { ptr: ctx, n_past: 0 })
        }
    }
}
