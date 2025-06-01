import os
import subprocess
import pytest

# Determine the correct command for buddy-ai based on how it's installed
# Using 'python -m buddy_ai.cli' for robustness in test environments
BUDDY_AI_CMD = ["python", "-m", "buddy_ai.cli"]

def run_buddy_ai_with_log_level(prompt, log_level=None, expect_error=True):
    env = os.environ.copy()
    if log_level:
        env["BUDDY_LOG_LEVEL"] = log_level
    else:
        # Ensure no stray env var from previous tests
        env.pop("BUDDY_LOG_LEVEL", None)

    config_path = "config.ini"
    dummy_config_content = (
        "[AISTUDIO]\n"
        "api_key = DUMMY_API_KEY_FOR_TESTING\n"
        "[PLANNER]\n"
        "model = dummy-planner-model\n"
        "[EXECUTOR]\n"
        "model = dummy-executor-model\n"
    )

    with open(config_path, "w") as f:
        f.write(dummy_config_content)

    command = BUDDY_AI_CMD + ["--prompt", prompt]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        stdout, stderr = process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = "", "CRITICAL: Test subprocess timed out."
        pytest.fail("Test subprocess timed out.")
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

    if expect_error and process.returncode == 0:
        pytest.fail(f"Command {' '.join(command)} with LOG_LEVEL='{log_level}' was expected to error but succeeded. Stderr:\n{stderr}\nStdout:\n{stdout}")
    elif not expect_error and process.returncode != 0:
        pytest.fail(f"Command {' '.join(command)} with LOG_LEVEL='{log_level}' was expected to succeed but errored. Stderr:\n{stderr}\nStdout:\n{stdout}")

    return stdout, stderr, process.returncode

def test_default_logging_level_is_warning():
    _stdout, stderr, _returncode = run_buddy_ai_with_log_level("test prompt default", expect_error=True)

    assert "Logging initialized with level WARNING." not in stderr
    assert "Debug logging test message." not in stderr
    assert "Google API key loaded" not in stderr
    assert "CRITICAL ERROR: API key not found" in stderr or "CRITICAL ERROR during graph execution" in stderr or "Error invoking structured planner LLM" in stderr

def test_log_level_override_info():
    _stdout, stderr, _returncode = run_buddy_ai_with_log_level("test prompt info", log_level="INFO", expect_error=True)

    assert "Logging initialized with level INFO." in stderr
    assert "Debug logging test message." not in stderr
    assert "Google API key loaded from config and set in environment." in stderr # Core message
    assert "CRITICAL ERROR: API key not found" in stderr or "CRITICAL ERROR during graph execution" in stderr or "Error invoking structured planner LLM" in stderr

def test_log_level_override_debug():
    _stdout, stderr, _returncode = run_buddy_ai_with_log_level("test prompt debug", log_level="DEBUG", expect_error=True)

    assert "Logging initialized with level DEBUG." in stderr
    assert "Debug logging test message." in stderr # Core message
    assert "Google API key loaded from config and set in environment." in stderr # Core message
    assert "CRITICAL ERROR: API key not found" in stderr or "CRITICAL ERROR during graph execution" in stderr or "Error invoking structured planner LLM" in stderr

def test_log_level_invalid_falls_back_to_warning():
    _stdout, stderr, _returncode = run_buddy_ai_with_log_level("test prompt invalid", log_level="INVALID_LEVEL", expect_error=True)

    assert "Logging initialized with level WARNING." not in stderr
    assert "Debug logging test message." not in stderr
    assert "Google API key loaded" not in stderr
    assert "CRITICAL ERROR: API key not found" in stderr or "CRITICAL ERROR during graph execution" in stderr or "Error invoking structured planner LLM" in stderr
