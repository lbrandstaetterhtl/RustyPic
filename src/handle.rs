use std::collections::HashMap;
use std::process::exit;
use crate::models::{Input, Command};
use crate::operate::grayscale;

pub fn base(input_str: String) -> Result<Option<String>, String> {
    let parts: Vec<String> = shell_words::split(input_str.trim())
        .map_err(|e| format!("Parse error: {e}"))?;
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();

    let input_result = Input::parse(parts);

    match input_result {
        Ok(input) => {
            let handle_fn = input.command.handle_fn;

            let result = handle_fn(input.args_values);

            result
        },
        Err(e) => Err(e)
    }
}

pub fn handle_grayscale(args_values: HashMap<String, String>) -> Result<Option<String>, String> {
    let source_option = args_values.get("--source");
    let output_option = args_values.get("--output");

    let source: String = match source_option {
        Some(s) => {s.to_string()},
        None => {return Err("Value for '--source' not found".to_string())},
    };

    let output = match output_option {
        Some(o) => {o.to_string()},
        None => {return Err("Value for '--output' not found".to_string())},
    };

    grayscale(source, output)
}

pub fn handle_exit(_: HashMap<String, String>) -> Result<Option<String>, String> {
    exit(0);
}

impl Input {
    fn parse(mut input_parts: Vec<&str>) -> Result<Input, String> {

        if input_parts.is_empty() || input_parts[0].is_empty() {
            return Err("No command given".to_string());
        }
        
        let commands = Command::get_all();

        let mut args_values: HashMap<String, String> = HashMap::new();
        let mut input_result: Result<Input, String> = Err("Not a valid command".to_string());

        for command in commands {
            if input_parts[0] == command.name {
                input_parts.remove(0);

                if command.args.is_none() {
                    return Ok(Input::new(command, args_values));
                }
                else {
                    let mut i = 0;
                   while i < input_parts.len() {
                        for (arg, has_value) in command.args.clone().unwrap() {
                            if arg == input_parts[i] {
                                if has_value {
                                    if i+1 >= input_parts.len() {
                                        return Err(format!("Missing value for argument '{}'", arg).to_string());
                                    }

                                    args_values.insert(arg.to_string(), input_parts[i+1].trim_matches('"').to_string());
                                    i += 1;
                                    break;
                                }
                                else {
                                    args_values.insert(arg.to_string(), "".to_string());
                                    break;
                                }
                            }
                        }
                       i += 1;
                    }
                }

                input_result = Ok(Input::new(command, args_values.clone()));
                break;
            }
        }

        input_result
    }
}