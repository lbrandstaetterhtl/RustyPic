use std::collections::HashMap;
use crate::handle;

pub struct Command {
    pub name: &'static str,
    pub description: &'static str,
    pub args: Option<Vec<(&'static str, bool)>>,
    pub handle_fn: fn(args_values: HashMap<String, String>) -> Result<Option<String>, String>,
}

pub struct Input {
    pub command: Command,
    pub args_values: HashMap<String, String>
}

impl Input {
    pub fn new(command: Command, args_values: HashMap<String, String>) -> Input {
        Input{command, args_values}
    }
}

impl Command {
    pub fn get_all() -> Vec<Command> {
        let grayscale = Command::new("grayscale", "", Some(vec![("--source", true), ("--output", true)]), handle::handle_grayscale);
        let exit = Command::new("exit", "", None, handle::handle_exit);

        let mut commands = Vec::new();

        commands.push(grayscale);
        commands.push(exit);

        commands
    }

    fn new(name: &'static str, description: &'static str, args: Option<Vec<(&'static str, bool)>>, handle_fn: fn(args_values: HashMap<String, String>) -> Result<Option<String>, String>) -> Command {
        Command{name, description, args, handle_fn }
    }
}