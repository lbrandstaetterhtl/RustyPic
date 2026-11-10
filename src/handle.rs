use std::collections::HashMap;
use std::process::exit;
use crate::models::{Input, Command, OcrTrainTestEnum, ValueTypes, ArgsHandler};
use crate::operate;
use crate::operate::{invert, grayscale, train_ocr, test_ocr, color};

pub fn base(input_str: String) -> Result<Option<String>, String> {
    let parts: Vec<String> = shell_words::split(input_str.trim())
        .map_err(|e| format!("Parse error: {e}"))?;
    let parts: Vec<&str> = parts.iter().map(String::as_str).collect();

    let input_result = Input::parse(parts);

    match input_result {
        Ok(input) => {
            let handle_fn = input.command.handle_fn;

            let args_values = input.args_values.unwrap_or_else(|| HashMap::new());
            let result = handle_fn(args_values);

            result
        },
        Err(e) => Err(e)
    }
}

pub fn handle_grayscale(args_values: HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String> {
    let source = args_values.require_str("--source").map_err(|e| e)?;
    let output = args_values.require_str("--output").map_err(|e| e)?;

    grayscale(source, output)
}

pub fn handle_invert(args_values:HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String> {
    let source = args_values.require_str("--source").map_err(|e| e)?;
    let output = args_values.require_str("--output").map_err(|e| e)?;

    invert(source, output)
}

pub fn handle_color(args_values: HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String> {;
    let source = args_values.require_str("--source").map_err(|e| e)?;
    let output = args_values.require_str("--output").map_err(|e| e)?;

    let all = args_values.get_int("--all");
    let red = args_values.get_int("-r");
    let green = args_values.get_int("-g");
    let blue = args_values.get_int("-b");

    color(source, output, all, red, green, blue)
}

pub fn handle_ocr(args_values: HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String> {
    let train = args_values.get_data("--train");
    let test = args_values.get_data("--test");
    let debug = args_values.get_bool("--debug");
    let resolve = args_values.get_str("--resolve");

    if let Some(t) = train {
        if let Some(r) = train_ocr(t).map_err(|e| e)? { println!("{}", r); }
    }

    if let Some(t) = test {
        if let Some(r) = test_ocr(t).map_err(|e| e)? { println!("{}",r ); }
    }

    if let Some(r) = resolve {
        if let Some(rr) = operate::resolve(r, debug).map_err(|e| e)? { println!("{}", rr); }
    }

    Ok(None)
}

pub fn handle_exit(_: HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String> {
    exit(0);
}

pub fn handle_clear(_: HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String> {
    clearscreen::clear().map_err(|e| e.to_string())?;

    Ok(None)
}

impl Input {
    fn parse(mut input_parts: Vec<&str>) -> Result<Input, String> {

        if input_parts.is_empty() || input_parts[0].is_empty() {
            return Err("No command given".to_string());
        }
        
        let commands = Command::get_all();

        let mut args_values: HashMap<&'static str, ValueTypes> = HashMap::new();
        let mut input_result: Result<Input, String> = Err("Not a valid command".to_string());
        for command in commands {
            if input_parts[0] == command.name {
                input_parts.remove(0);
                let args_types = match command.args_types.clone() {
                    Some(a) => a,
                    None => {return Ok(Input::new(command, None));}
                };

                let mut i = 0;
                while i < input_parts.len() {
                    let value: ValueTypes;
                    let mut matched = false;
                    for (arg, value_type) in &args_types {
                        if *arg == input_parts[i] {
                            let mut consumed_value = false;
                            match value_type {
                                ValueTypes::Int(None) => {
                                    if i + 1 >= input_parts.len() || input_parts[i+1].starts_with("--") {
                                        return Err(format!("Value for arg {} not found", arg))
                                    }

                                    value = ValueTypes::Int(Some(input_parts[i+1].parse::<i64>().map_err(|e| e.to_string())?));
                                    consumed_value = true;
                                },
                                ValueTypes::Bool(None) => {
                                    if i + 1 >= input_parts.len() || input_parts[i+1].starts_with("--") {
                                        value = ValueTypes::Bool(Some(true));
                                        consumed_value = false;
                                    }
                                    else {
                                        value = ValueTypes::Bool(Some(input_parts[i + 1].parse::<bool>().map_err(|e| e.to_string())?));
                                        consumed_value = true;
                                    }
                                },
                                ValueTypes::String(None) => {
                                    if i + 1 >= input_parts.len() || input_parts[i+1].starts_with("--") {
                                        return Err(format!("Value for arg {} not found", arg))
                                    }

                                    value = ValueTypes::String(Some(input_parts[i+1].to_string()));
                                    consumed_value = true;
                                },
                                ValueTypes::Data(None) => {
                                    if i + 1 >= input_parts.len() || input_parts[i+1].starts_with("--") {
                                        return Err(format!("Value for arg {} not found", arg))
                                    }

                                    value = ValueTypes::Data(Some(OcrTrainTestEnum::encode(input_parts[i+1]).map_err(|e| e.to_string())?));
                                    consumed_value = true;
                                }
                                _ => {return Err(format!("Could not parse value {} for args {}", input_parts[i+1], arg))}
                            }

                            args_values.insert(arg, value);
                            i+= if consumed_value {2} else {1};
                            matched = true;
                            break
                        }
                    }
                    if !matched {
                        println!("Ignoring unknown arg {}", input_parts[i]);
                        i += 1;
                    }
                }

                input_result = Ok(Input::new(command, Some(args_values)));
                break;
            }
        }

        input_result
    }
}