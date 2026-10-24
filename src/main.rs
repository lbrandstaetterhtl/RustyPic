use std::io::Write;
use crate::handle::base;

mod models;
mod handle;
mod operate;

fn main() {
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
