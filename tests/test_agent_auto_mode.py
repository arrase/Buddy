import unittest
import subprocess
import os
import pathlib
import shutil

# Assume the script is run from the project root directory
# where 'buddy_ai' and 'tests' are subdirectories.
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
TEST_DIR = PROJECT_ROOT / "tests"
OUTPUT_FILE_AUTO = TEST_DIR / "output_auto_test.txt"
LIST_TARGET_FILE = TEST_DIR / "test_list_target.txt"

class TestAgentAutoMode(unittest.TestCase):

    def _run_cli_command(self, prompt: str, auto_mode: bool = True, context: str = None):
        command = [
            "python", "-m", "buddy_ai.cli",
            "--prompt", prompt
        ]
        if auto_mode:
            command.append("--auto")
        if context:
            command.extend(["--context", context])

        # Ensure config.ini exists for the CLI to load
        config_path = PROJECT_ROOT / "config.ini"
        if not config_path.exists():
            # Create a dummy config if it doesn't exist, as CLI will try to load it
            with open(config_path, "w") as f:
                f.write("[buddy_ai]\n")
                f.write("api_key = YOUR_API_KEY_HERE\n") # Placeholder
                f.write("planner_model_name = gemini-pro\n") # Placeholder
                f.write("executor_model_name = gemini-pro\n") # Placeholder
            self.created_dummy_config = True


        # Run the command from the project root
        # The CLI and agent internally use paths relative to where they are run or CWD.
        # For file creation tests, we'll check paths relative to TEST_DIR or CWD (PROJECT_ROOT)
        # depending on how the prompt is phrased.
        # Let's ensure prompts create files in TEST_DIR for easy cleanup.
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT # Run from project root
        )
        return process

    def setUp(self):
        """Set up test fixtures, if any."""
        TEST_DIR.mkdir(exist_ok=True)
        # Clean up files from previous runs, if any
        if OUTPUT_FILE_AUTO.exists():
            OUTPUT_FILE_AUTO.unlink()
        if LIST_TARGET_FILE.exists():
            LIST_TARGET_FILE.unlink()
        self.created_dummy_config = False


    def tearDown(self):
        """Tear down test fixtures, if any."""
        if OUTPUT_FILE_AUTO.exists():
            OUTPUT_FILE_AUTO.unlink()
        if LIST_TARGET_FILE.exists():
            LIST_TARGET_FILE.unlink()

        # Clean up dummy config if created by a test
        # However, it's better if config.ini is handled outside,
        # or tests mock its loading. For now, if a test creates it, it tries to remove it.
        # This is risky if a real config.ini was meant to be there.
        # A better approach is to ensure a TEST_CONFIG_INI is used by the CLI via env var if possible.
        # For now, we'll only remove it if this specific test run created it.
        # config_path = PROJECT_ROOT / "config.ini"
        # if self.created_dummy_config and config_path.exists():
        #    config_path.unlink()
        #    self.created_dummy_config = False
        # Commenting out config removal as it's too risky. Assume config.ini is managed externally.


    def test_auto_mode_create_file(self):
        """Test that --auto mode successfully creates a file as per prompt."""
        prompt = f"Create a file named '{OUTPUT_FILE_AUTO.name}' in the '{TEST_DIR.name}' directory with the content 'Hello auto mode'"

        # For the agent to correctly place the file in TEST_DIR, it needs to know TEST_DIR exists
        # The prompt is now relative to CWD (PROJECT_ROOT).
        # So the file will be at PROJECT_ROOT / TEST_DIR.name / OUTPUT_FILE_AUTO.name
        # which is `tests/output_auto_test.txt`

        result = self._run_cli_command(prompt=prompt, auto_mode=True)

        self.assertEqual(result.returncode, 0, f"CLI command failed with error: {result.stderr}")

        # The path used in prompt should be relative to where ShellTool executes,
        # which is PROJECT_ROOT.
        expected_file_path = PROJECT_ROOT / TEST_DIR.name / OUTPUT_FILE_AUTO.name

        self.assertTrue(expected_file_path.exists(), f"File {expected_file_path} was not created.")

        content = expected_file_path.read_text()
        self.assertEqual(content.strip(), "Hello auto mode", f"File content mismatch. Got: '{content}'")

    def test_auto_mode_list_files(self):
        """Test that --auto mode can list files, implying shell execution."""
        # Create a target file to be listed
        LIST_TARGET_FILE.write_text("This is a test file for listing.")
        self.assertTrue(LIST_TARGET_FILE.exists())

        # Prompt to list files in the TEST_DIR.
        # The shell command executed by the agent will be relative to PROJECT_ROOT.
        prompt = f"List all files in the '{TEST_DIR.name}' directory."

        result = self._run_cli_command(prompt=prompt, auto_mode=True)

        self.assertEqual(result.returncode, 0, f"CLI command failed with error: {result.stderr}")

        # Check if the output contains the name of the test file
        self.assertIn(LIST_TARGET_FILE.name, result.stdout,
                      f"CLI output did not contain '{LIST_TARGET_FILE.name}'. Output:\n{result.stdout}")

if __name__ == "__main__":
    # This allows running the tests directly from this file
    unittest.main()
