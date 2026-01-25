use crate::chat::ffi::*;
use crate::chat::model::{Model, Context};

use anyhow::{Result, anyhow};
use std::ffi::{CString, CStr};


pub struct Llama {
    ctx: Context,
}

impl {
    pub fn load(path: &str) -> Result<Self> {
        unsafe { llama_backend_init(false); }
        let model = Model::load(path)?;
        let ctx = Context::new(&model)?;
        Ok(Self { ctx })
    }

    fn eval(&mut self, tokens: &[llama_token]) -> Result<()> {
        let mut pos: Vec<i32> = (self.ctx.n_past..self.ctx.n_past + tokens.len() as i32).collect();
        let mut seq: Vec<i32> = vec![0; tokens.len()];
        let mut logits: Vec<bool> = vec![false; tokens.len()];
        *logits.last_mut().unwrap() = true;

        let mut batch = llama_batch {
            n_tokens: tokens.len() as i32,
            token: tokens.as_ptr() as *mut _,
            pos: pos.as_mut_ptr(),
            seq_id: seq.as_mut_ptr(),
            logits: logits.as_mut_ptr(),
        };

        unsafe {
            if llama_decode(self.ctx.ptr, &mut batch) != 0 {
                return Err(anyhow!("decode failed"));
            }
        }

        self.ctx.n_past += tokens.len() as i32;
        Ok(())
    }

    pub fn chat(&mut self, user: &str) -> Result<String> {
        unimplemented!()
    }

}