use std::collections::HashMap;
use std::process::exit;
use crate::models::{Input, Command};
use crate::operate::{invert, grayscale, resolve, train_ocr, test_ocr};

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

pub fn get_values(name: String, args_values: HashMap<String, String>) -> Result<Vec<String>, String> {
    let command_result = Command::get(name.as_str());

    let command: Command =  match command_result {
        Ok(c) => {c},
        Err(e) => {return Err(e)}
    };

    let mut values: Vec<String> = Vec::new();
    for (arg, has_value) in command.args.unwrap() {
        if args_values.contains_key(arg) {
            if has_value {
                let value_option = args_values.get(arg);
                match value_option {
                    Some(v) => { values.push(v.to_string()); },
                    None => { return Err(format!("No value for arg '{}'", arg)) }
                }
            } else {
                values.push("1".to_string());
            }
        }
        else if !args_values.contains_key(arg) && !has_value{
            values.push("0".to_string());
        }
    }

    Ok(values)
}

pub fn handle_grayscale(args_values: HashMap<String, String>) -> Result<Option<String>, String> {
    let values_result = get_values("grayscale".to_string(), args_values);

    let values = match values_result {
        Ok(v) => v,
        Err(e) => {return Err(e)}
    };

    if values.len() == 0 || values.len() < 2 || values.len() > 2 {
        return Err("Invalid grayscale arguments".to_string());
    }

    let source = values[0].clone();
    let output = values[1].clone();

    grayscale(source, output)
}

pub fn handle_invert(args_values: HashMap<String, String>) -> Result<Option<String>, String> {
    let values_result = get_values("invert".to_string(), args_values);

    let values = match values_result {
        Ok(v) => v,
        Err(e) => return Err(e)
    };

    if values.len() == 0 || values.len() < 2 || values.len() > 2 {
        return Err("Invalid invert arguments".to_string());
    }

    let source = values[0].clone();
    let output = values[1].clone();
    invert(source, output)
}

pub fn handle_color(args_values: HashMap<String, String>) -> Result<Option<String>, String> {
    let values_result = get_values("color".to_string(), args_values);

    let values = match values_result {
        Ok(v) => v,
        Err(e) => {return Err(e)}
    };

    if values.len() == 0 || values.len() < 2 || values.len() > 2 {
        return Err("Invalid color arguments".to_string());
    }

    let source = values[0].clone();
    let output = values[1].clone();
    let all_as_integer = values[2].clone().parse::<i32>().map_err(|_| "Invalid grayscale arguments")?;
    let all = if all_as_integer == 1 { true } else { false };

    panic!("not implemented");
}

pub fn handle_ocr(args_values: HashMap<String, String>) -> Result<Option<String>, String> {
    let values = match get_values("ocr".to_string(), args_values.clone()) {
        Ok(v) => v,
        Err(e) => {return Err(e)}
    };

    if args_values.contains_key("--train") {
        return train_ocr()
    }
    else if args_values.contains_key("--test") {
        return  test_ocr()
    }
    else if args_values.contains_key("--resolve") {
        let source = values[0].clone();
        return  resolve(source)
    }

    Err("Invalid ocr arguments".to_string())
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
                                        return Err(format!("Value for '{}' not found", arg).to_string());
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