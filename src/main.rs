use std::io::Write;
use std::path::Path;
use crate::handle::base;
use crate::models::AppConfig;

mod models;
mod handle;
mod operate;
mod emnist;

fn main() {
    let config = AppConfig::load().expect("failed to load config");
    config.save().expect("failed to save config");

    loop {
        print!(">: ");
        std::io::stdout().flush().unwrap();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        let result = base(input);

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
