import os
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

    if not api_key:
        logging.error(f"API key not found in '{config_path}' under [AISTUDIO] section.")
    else:
        os.environ["GOOGLE_API_KEY"] = api_key
        logging.info("Google API key loaded from config and set in environment.")

    logging.info(f"Planner model: {planner_model_name}, Executor model: {executor_model_name}")

    return api_key, planner_model_name, executor_model_name

if __name__ == '__main__':
    logging.info("Attempting to load configuration...")
    key, planner_model, executor_model = load_app_config()
    if key:
        logging.info("Config loaded successfully.")
        logging.info(f"API Key Present: {'Yes' if key else 'No'}")
        logging.info(f"Planner Model: {planner_model}")
        logging.info(f"Executor Model: {executor_model}")
    else:
        logging.error("Failed to load configuration.")
