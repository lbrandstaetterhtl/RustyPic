use std::collections::HashMap;
use std::process::exit;
use crate::models::{Input, Command, OcrTrainTestEnum, ValueTypes};
use crate::models::OcrTrainTestEnum::Combined;
use crate::operate;
use crate::operate::{invert, grayscale, train_ocr, test_ocr};

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
    let mut source = "".to_string();
    let mut output = "".to_string();

    for (a, v) in args_values {
        if a == "--source" {
            source = match v.as_str() {
                Some(s) => s,
                None => return Err("Couldn't get source value from input".to_string())
            };
        }
        else if a == "--output" {
            output = match v.as_str() {
                Some(s) => s,
                None => return Err("Couldn't get output value from input".to_string())
            }
        }
    }

    grayscale(source, output)
}

pub fn handle_invert(args_values:HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String> {
    let mut source = "".to_string();
    let mut output = "".to_string();

    for (a, v) in args_values {
        if a == "--source" {
            source = match v.as_str() {
                Some(s) => s,
                None => return Err("Couldn't get source value from input".to_string())
            };
        }
        else if a == "--output" {
            output = match v.as_str() {
                Some(s) => s,
                None => return Err("Couldn't get output value from input".to_string())
            }
        }
    }

    invert(source, output)
}

pub fn handle_color(args_values: HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String> {;

    panic!("not implemented");
}

pub fn handle_ocr(args_values: HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String> {

    let mut train: OcrTrainTestEnum = Combined;
    let mut train_bool = false;
    let mut test: OcrTrainTestEnum = Combined;
    let mut test_bool = false;
    let mut debug = false;
    let mut resolve = String::new();
    let mut resolve_bool = false;
    for (a, v) in args_values {
        if a == "--train" {
            train = v.as_data().unwrap_or_else(|| Combined);
            train_bool = true;
        }

        if a == "--test" {
            test = v.as_data().unwrap_or_else(|| Combined);
            test_bool = true;
        }

        if a == "--debug" {
            debug = v.as_bool().unwrap_or_else(|| false);
        }

        if a == "--resolve" {
            resolve = match v.as_str() {
                Some(r) => r,
                None => return Err("Couldn't get resolve value from input".to_string())
            };
            resolve_bool = true;
        }
    }

    if train_bool {
        let _ = match train_ocr(train).map_err(|e| e.to_string())? {
            Some(r) => {println!("{}", r)}
            None => {}
        };
    }

    if test_bool {
        let _ = match test_ocr(test).map_err(|e| e.to_string())? {
            Some(r) => {println!("{}", r)},
            None => {}
        };
    }

    if resolve_bool {
        let _ = match operate::resolve(resolve, debug).map_err(|e| e.to_string())? {
            Some(r) => { print!("{}", r);},
            None => {}
        };
    }

    Ok(None)
}

pub fn handle_exit(_: HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String> {
    exit(0);
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

                            if i + 1 >= input_parts.len() {
                                return Err(format!("Value for arg {} not found", arg))
                            }

                            match value_type {
                                ValueTypes::Int(None) => {value = ValueTypes::Int(Some(input_parts[i+1].parse::<i64>().map_err(|e| e.to_string())?))},
                                ValueTypes::Bool(None) => {value = ValueTypes::Bool(Some(input_parts[i+1].parse::<bool>().map_err(|e| e.to_string())?))},
                                ValueTypes::String(None) => {value = ValueTypes::String(Some(input_parts[i+1].to_string()))},
                                ValueTypes::Data(None) => {value = ValueTypes::Data(Some(OcrTrainTestEnum::encode(input_parts[i+1]).map_err(|e| e.to_string())?))}
                                _ => {return Err(format!("Could not parse value {} for args {}", input_parts[i+1], arg))}
                            }

                            args_values.insert(arg, value);
                            i+=2;
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