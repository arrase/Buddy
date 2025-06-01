import os
import sys
import configparser
import logging

def load_app_config():
    config = configparser.ConfigParser()
    config_path = "config.ini"

    if not os.path.exists(config_path):
        logging.warning(f"Configuration file '{config_path}' not found in current working directory ({os.getcwd()}).")
        return None, None, None

    read_files = config.read(config_path)
    if not read_files:
        logging.error(f"Could not parse configuration file: '{config_path}'. Ensure it is a valid INI format.")
        return None, None, None

    api_key = config.get("AISTUDIO", "api_key", fallback=None)
    planner_model_name = config.get("PLANNER", "model", fallback="gemini-1.5-flash-latest")
    executor_model_name = config.get("EXECUTOR", "model", fallback="gemini-1.5-flash-latest")

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        logging.error(f"API key not found in '{config_path}' under [AISTUDIO] section.")
        sys.exit("Error: API key is required. Please set it in the configuration file.")

    logging.info(f"Planner model: {planner_model_name}, Executor model: {executor_model_name}")

    return api_key, planner_model_name, executor_model_name