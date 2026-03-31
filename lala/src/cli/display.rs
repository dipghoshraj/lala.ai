use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

// ── Braille spinner frames ────────────────────────────────────────────────────
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

// ── ANSI colour codes ─────────────────────────────────────────────────────────
pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";
pub const BOLD_YELLOW: &str = "\x1b[1;33m";
pub const DIM_YELLOW: &str = "\x1b[2;33m";
pub const BOLD_CYAN: &str = "\x1b[1;36m";
pub const CYAN: &str = "\x1b[36m";
pub const BOLD_GREEN: &str = "\x1b[1;32m";
pub const BOLD_RED: &str = "\x1b[1;31m";
pub const YELLOW: &str = "\x1b[33m";
pub const SECTION_WIDTH: usize = 60;

/// Run `f` while displaying a braille spinner labelled `label`.
/// Stops and erases the spinner line before returning.
pub fn with_spinner<F, T>(label: &str, f: F) -> T
where
    F: FnOnce() -> T,
{
    let running = Arc::new(AtomicBool::new(true));
    let spin_flag = Arc::clone(&running);
    let label_owned = label.to_string();
    let clear_len = label.len() + 8;
    let spinner_handle = thread::spawn(move || {
        let mut i = 0usize;
        loop {
            if !spin_flag.load(Ordering::Relaxed) {
                break;
            }
            print!("\r  {} {}...", SPINNER[i % SPINNER.len()], label_owned);
            io::stdout().flush().ok();
            i += 1;
            thread::sleep(Duration::from_millis(80));
        }
        print!("\r{}\r", " ".repeat(clear_len));
        io::stdout().flush().ok();
    });
    let result = f();
    running.store(false, Ordering::Relaxed);
    spinner_handle.join().ok();
    result
}

/// Print a titled section with a coloured header and separator.
pub fn print_section(header: &str, header_color: &str, text_color: &str, content: &str) {
    let sep = "─".repeat(SECTION_WIDTH);
    println!("\n{}{} {}{}", header_color, "▷", header, RESET);
    println!("{}{}{}", header_color, sep, RESET);
    println!("{}{}{}", text_color, content.trim(), RESET);
    println!("{}{}{}", header_color, sep, RESET);
    println!();
}

/// Print an informational message.
pub fn info(msg: &str) {
    println!("  {}ℹ{} {}", BOLD_CYAN, RESET, msg);
}

/// Print a success message.
pub fn success(msg: &str) {
    println!("  {}✓{} {}", BOLD_GREEN, RESET, msg);
}

/// Print a warning message.
pub fn warn(msg: &str) {
    println!("  {}⚠{} {}", YELLOW, RESET, msg);
}

/// Print an error message.
pub fn error(msg: &str) {
    eprintln!("  {}✗{} {}", BOLD_RED, RESET, msg);
}

/// Print a progress indicator: [current/total] filename
pub fn progress(current: usize, total: usize, label: &str) {
    println!(
        "  {}[{}/{}]{} {}",
        DIM, current, total, RESET, label
    );
}
