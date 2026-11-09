use std::collections::HashMap;
use std::fs;
use std::hash::Hash;
use std::path::{Path, PathBuf};
use candle_core::{DType, Device, Module, Tensor};
use candle_core::op::Op;
use candle_nn::{Conv2d, Conv2dConfig, Linear, VarBuilder, conv2d, linear, VarMap};
use serde::{Deserialize, Serialize};
use crate::{emnist, handle};
use crate::emnist::Source;
use crate::models::OcrTrainTestEnum::{Captcha, Combined, Emnist};

#[derive(Clone)]
pub struct Command {
    pub name: &'static str,
    pub description: &'static str,
    pub args_types: Option<HashMap<&'static str, ValueTypes>>,
    pub handle_fn: fn(args_values: HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String>,
}

#[derive(Clone)]
pub enum ValueTypes {
    Int(Option<i64>),
    Bool(Option<bool>),
    String(Option<String>),
    Data(Option<OcrTrainTestEnum>)
}

#[derive(Serialize, Deserialize)]
pub struct AppConfig {
    pub model_path: String,
    pub epochs: u32,
    pub batch_size: usize,
    pub device: String,
    pub learning_rate: f64,
    pub debug_dir: String
}

pub struct Input {
    pub command: Command,
    pub args_values: Option<HashMap<&'static str, ValueTypes>>
}

#[derive(Serialize, Deserialize)]
pub struct ModelConfig {
    pub kernel_size: usize,
    pub conv_count: u32,
    pub input_size: usize,
    pub vocab_str: String,
    pub pool_after: Vec<bool>
}

pub struct CNN {
    pub var_map: VarMap,
    pub conv_list: Vec<Conv2d>,
    pub pool_after: Vec<bool>,
    pub fc: Linear,
    pub vocab: Vocab
}

pub struct Vocab {
    pub to_idx: HashMap<char, u32>,
    pub to_char: Vec<char>,
}

#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
    pub area: u32,
}

impl ValueTypes {
    pub fn as_int(&self) -> Option<i64> {
        if let ValueTypes::Int(n) = self { *n } else { None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let ValueTypes::Bool(b) = self { *b } else { None }
    }
    pub fn as_str(&self) -> Option<String> {
        if let ValueTypes::String(Some(s)) = self { Some(s.clone()) } else { None }
    }
    pub fn as_data(&self) -> Option<OcrTrainTestEnum> {
        if let ValueTypes::Data(Some(d)) = self { Some(d.clone()) } else { None }
    }
}

#[derive(Clone)]
pub enum OcrTrainTestEnum
{
    Emnist,
    Captcha,
    Combined
}

impl OcrTrainTestEnum {
    pub fn encode(str: &str) -> Result<Self, String> {
        if str.to_lowercase().eq("emnist") {
            return Ok(Emnist)
        }
        else if str.to_lowercase().eq("captcha") {
            return Ok(Captcha)
        }
        else if str.to_lowercase().eq("combined") {
            return Ok(Combined)
        }

        Err(format!("Could not convert {} to any train/test option", str))
    }

    pub fn decode(&self) -> Result<String, String> {
        match self {
            Emnist => Ok("emnist".to_string()),
            Captcha => Ok("captcha".to_string()),
            Combined => Ok("combined".to_string())
        }
    }

    pub fn to_data_source(&self) -> Result<emnist::Source, String> {
        match self {
            Emnist => Ok(Source::Emnist),
            Captcha => Ok(Source::Captcha),
            Combined => Ok(Source::Combined)
        }
    }
}

impl BBox {
    pub fn width(&self) -> u32 {
        self.max_x - self.min_x + 1
    }

    pub fn height(&self) -> u32 {
        self.max_y - self.min_y + 1
    }

    pub fn fill(&self) -> f32 {
        self.area as f32 / (self.width() * self.height()) as f32
    }
}


impl AppConfig {
    pub fn get_config_dir() -> Result<String, String> {
        let base = std::env::var("LOCALAPPDATA")
            .map_err(|_| "LOCALAPPDATA not set".to_string())?;

        Ok(PathBuf::from(base).join("RustyPic").to_str().unwrap().to_string())
    }

    fn get_model_dir() -> Result<String, String> {
        let app_data = match AppConfig::get_config_dir() {
            Ok(d) => d,
            Err(e) => return Err(e.to_string())
        };

        Ok(PathBuf::from(app_data).join("Models").to_str().unwrap().to_string())
    }

    fn get_debug_dir() -> Result<String, String> {
        let app_data = AppConfig::get_config_dir().map_err(|e| e.to_string())?;

        Ok(PathBuf::from(app_data).join("debug").to_str().unwrap().to_string())
    }

    pub fn device(&self) -> Result<Device, String> {
        match self.device.as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" => Device::new_cuda(0).map_err(|e| e.to_string()),
            "metal" => Device::new_metal(0).map_err(|e| e.to_string()),
            other => Err(format!("unknown device: {}", other)),
        }
    }

    pub fn build_default() -> Result<AppConfig, String> {
        let model_dir = Self::get_model_dir()?;
        let debug_dir = Self::get_debug_dir()?;

        fs::create_dir_all(&model_dir).map_err(|e| format!("failed to create model dir: {}", e))?;
        fs::create_dir_all(&debug_dir).map_err(|e| format!("failed to create debug dir: {}", e))?;

        let model_path = model_dir + "\\model.safetensors";

        let config = AppConfig {model_path, epochs: 10, batch_size: 64, device: "cpu".to_string(), learning_rate: 0.001, debug_dir};

        match config.save() {
            Ok(_) => Ok(config),
            Err(e) => Err(e.to_string())
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let config_dir = Self::get_config_dir()?;
        fs::create_dir_all(&config_dir).map_err(|e| e.to_string())?;

        let path = Path::new(&config_dir).join("AppConfig.json");

        let text = serde_json::to_string_pretty(self).unwrap();
        fs::write(path, text).map_err(|e| e.to_string())
    }

    pub fn load() -> Result<AppConfig, String> {
        let config_dir = Self::get_config_dir()?;
        fs::create_dir_all(&config_dir).map_err(|e| e.to_string())?;

        let path = Path::new(&config_dir).join("AppConfig.json");


        if !path.exists() {
            let config = Self::build_default()?;
            config.save()?;
            return Ok(config);
        }

        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }
}

impl ModelConfig {
    pub fn load(path: &Path) -> Result<ModelConfig, String> {
        if !path.exists() {
            let config = match Self::build_default() {
                Ok(config) => config,
                Err(msg) => return Err(msg)
            };

            let result = match config.save(path) {
                Ok(_) => Ok(config),
                Err(msg) => Err(msg)
            };

            return result;
        }

        let text = fs::read_to_string(path).map_err(|err| err.to_string())?;
        serde_json::from_str(&text).map_err(|err| err.to_string())
    }

    pub fn build_default() -> Result<ModelConfig, String> {
        Ok(ModelConfig {kernel_size: 3, conv_count: 3, input_size: 28, vocab_str: String::from("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"), pool_after: vec![true, true, false] })
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|err| err.to_string())?;

        fs::write(path, text).map_err(|err| err.to_string())
    }
}

impl Vocab {
    pub fn new(chars: &str) -> Self {
        let to_char: Vec<char> = chars.chars().collect();
        let to_idx = to_char.iter().enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        Self { to_idx, to_char }
    }

    pub fn encode(&self, c: char) -> Option<u32> {
        self.to_idx.get(&c).copied()
    }

    pub fn decode(&self, i: u32) -> char {
        self.to_char[i as usize]
    }

    pub fn len(&self) -> usize {
        self.to_char.len()
    }
}

impl CNN {
    pub fn new(vocab: Vocab, var_builder: VarBuilder, pool_after: Vec<bool>, conv_count: u32, kernel_size: usize, input_size: usize, var_map: VarMap) -> Result<Self, String> {
        let config = Conv2dConfig {padding: 0, stride: 1, ..Default::default()};
        let mut conv_list: Vec<Conv2d> = Vec::new();
        let mut in_ch: usize = 1;
        let mut out_ch: usize = 16;
        let mut conv_counter: u32 = 0;
        let mut size = input_size;

        while conv_counter < conv_count {
            let conv = match conv2d(in_ch, out_ch, kernel_size, config, var_builder.pp(format!("conv{}", conv_counter))) {
                Ok(conv) => conv,
                Err(e) => return Err(e.to_string())
            };
            conv_list.push(conv);

            if size < kernel_size {
                return Err(format!("conv{} would need {}x{} but only {}x{} left", conv_counter, kernel_size, kernel_size, size, size));
            }
            size = size - kernel_size + 1;

            if pool_after[conv_counter as usize] {
                if size < 2 {
                    return Err(format!("pool after conv{} is impossible at size {}", conv_counter, size));
                }
                size /= 2;
            }

            in_ch = out_ch;
            out_ch *= 2;
            conv_counter += 1;
        }


        let flat = in_ch * size * size;
        let fc = match linear(flat, vocab.len(), var_builder.pp("fc")) {
            Ok(v) => v,
            Err(e) => return Err(e.to_string())
        };

        Ok(Self { var_map, conv_list, pool_after, fc, vocab })
    }

    pub fn forward(&self, tensor: &Tensor) -> Result<Tensor, String> {
        let mut step: Tensor = tensor.clone();
        let mut conv_counter: u32 = 0;

        while conv_counter < self.conv_list.len() as u32 {
            let conv = self.conv_list.get(conv_counter as usize).unwrap();

            step = match conv.forward(&step) {
                Ok(v) => v,
                Err(e) => return Err(e.to_string())
            };

            step = match step.relu() {
                Ok(v) => v,
                Err(e) => return Err(e.to_string())
            };

            if self.pool_after[conv_counter as usize] {
                step = match step.max_pool2d(2) {
                    Ok(v) => v,
                    Err(e) => return Err(e.to_string())
                }
            }

            conv_counter += 1;
        }

        step = match step.flatten_from(1) {
            Ok(v) => v,
            Err(e) => return Err(e.to_string())
        };

        let result = match self.fc.forward(&step) {
            Ok(v) => v,
            Err(e) => return Err(e.to_string())
        };

        Ok(result)
    }

    pub fn load(device: &Device) -> Result<Self, String> {
        let app_config = AppConfig::load()?;

        let config_path = format!("{}.json", app_config.model_path);

        let config = ModelConfig::load(Path::new(config_path.as_str())).map_err(|err| err.to_string())?;

        let vocab = Vocab::new(config.vocab_str.as_str());
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);

        let mut cnn = match Self::new(vocab, var_builder, config.pool_after, config.conv_count, config.kernel_size, config.input_size, var_map) {
            Ok(v) => v,
            Err(e) => return Err(e.to_string())
        };

        if Path::new(&app_config.model_path).exists() {
            if let Err(e) = cnn.var_map.load(app_config.model_path) {
                return Err(e.to_string());
            }
        }

        Ok(cnn)
    }

    pub fn save(&self) -> Result<(), String> {
        let app_config = AppConfig::load()?;

        let _ = match self.var_map.save(app_config.model_path) {
            Ok(v) => v,
            Err(e) => return Err(e.to_string())
        };

        Ok(())
    }

}

impl Input {
    pub fn new(command: Command, args_values: Option<HashMap<&'static str, ValueTypes>>) -> Input {
        Input{command, args_values}
    }
}

impl Command {
    pub fn get_all() -> Vec<Command> {
        let mut default_args: HashMap<&'static str, ValueTypes> = HashMap::new();
        default_args.insert("--source", ValueTypes::String(None));
        default_args.insert("--output", ValueTypes::String(None));
        let grayscale = Command::new("grayscale", "", Some(default_args.clone()), handle::handle_grayscale);

        let exit = Command::new("exit", "", None, handle::handle_exit);

        let invert = Command::new("invert", "", Some(default_args), handle::handle_invert);

        let mut color_args: HashMap<&'static str, ValueTypes> = HashMap::new();
        color_args.insert("--source", ValueTypes::String(None));
        color_args.insert("--output", ValueTypes::String(None));
        color_args.insert("--all", ValueTypes::Bool(None));
        color_args.insert("-r", ValueTypes::Int(None));
        color_args.insert("-g", ValueTypes::Int(None));
        color_args.insert("-b", ValueTypes::Int(None));
        let color = Command::new("color", "", Some(color_args), handle::handle_color);

        let mut ocr_args: HashMap<&'static str, ValueTypes> = HashMap::new();
        ocr_args.insert("--train", ValueTypes::Data(None));
        ocr_args.insert("--test", ValueTypes::Data(None));
        ocr_args.insert("--resolve", ValueTypes::String(None));
        ocr_args.insert("--debug", ValueTypes::Bool(None));
        let ocr = Command::new("ocr", "", Some(ocr_args), handle::handle_ocr);

        let mut commands = Vec::new();

        commands.push(grayscale);
        commands.push(invert);
        commands.push(color);
        commands.push(ocr);
        commands.push(exit);

        commands
    }

    pub fn get(name: &str) -> Result<Command, String> {
        let commands = Self::get_all();

        let mut command_result: Result<Command, String> = Err(format!("No such command: {}", name));
        for command in commands.iter() {
            if command.name == name {
                command_result = Ok(command.clone())
            }
        }

        command_result
    }

    fn new(name: &'static str, description: &'static str, args_types: Option<HashMap<&'static str, ValueTypes>>, handle_fn: fn(args_values: HashMap<&'static str, ValueTypes>) -> Result<Option<String>, String>) -> Command {
        Command{name, description, args_types, handle_fn }
    }
}

