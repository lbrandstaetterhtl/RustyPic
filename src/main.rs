use std::io::Write;
use crate::models::AppConfig;

mod models;
mod handle;
mod operate;
mod emnist;

fn main() {
    let config = AppConfig::load().expect("failed to load config");
    config.save().expect("failed to load config properly");
    clearscreen::clear().expect("failed to clear screen");
    loop {
        print!("\n>: ");
        std::io::stdout().flush().unwrap();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        let result = handle::base(input);

        match result {
            Ok(value) => {
                match value {
                    Some(value) => {
                        println!("Operation ended with message: {}", value);
                    }
                    None => {}
                }
            }
            Err(error) => {
                println!("Error: {}", error);
            }
        }
    }
}
