import subprocess
import os
import shutil # For cleaning up temporary files/directories
import unittest

# Helper function for running commands (as provided in the prompt)
def run_buddyai_command(args_list):
    command = ['buddyai'] + args_list
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # Ensure GOOGLE_API_KEY is available to the subprocess
        # Tests might fail if the key is not set in the environment they run in.
        # For CI, this would be set as a secret. For local, it should be in the shell.
        env = os.environ.copy()
        if "GOOGLE_API_KEY" not in env:
            print("Warning: GOOGLE_API_KEY not found in environment. Tests requiring LLM calls may fail or be skipped.")
            # One might choose to skip tests or use a mock LLM if the key isn't present.
            # For now, tests will run and potentially fail if the key is missing and LLM is called.

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=project_root,
            env=env, # Pass the environment, which includes GOOGLE_API_KEY if set
            timeout=180 # Increased timeout for AI operations, especially planning
        )
        return result
    except FileNotFoundError:
        print("Error: 'buddyai' command not found. Make sure the package is installed with 'pip install -e .'")
        # Re-raise so the test framework knows this is a critical setup error.
        raise
    except subprocess.TimeoutExpired:
        print(f"Error: Command '{' '.join(command)}' timed out.")
        return subprocess.CompletedProcess(command, timeout=True, returncode=124, stdout="TimeoutExpired", stderr="TimeoutExpired")

class TestBuddyAICLI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # This method is run once before any tests in the class
        # You could add a check here to see if 'buddyai' is callable or if GOOGLE_API_KEY is set.
        # For now, we assume 'pip install -e .' has been run.
        if "GOOGLE_API_KEY" not in os.environ:
            print("\nWARNING: GOOGLE_API_KEY is not set. Some tests may fail or produce limited output if they rely on actual LLM calls.")
            print("Please set the GOOGLE_API_KEY environment variable for comprehensive testing.\n")


    def test_01_basic_prompt_execution(self):
        """Test basic prompt execution (e.g., a simple question)."""
        # This test relies on the LLM being available and correctly configured.
        if "GOOGLE_API_KEY" not in os.environ:
            self.skipTest("Skipping LLM-dependent test: GOOGLE_API_KEY not set.")

        result = run_buddyai_command(['--prompt', 'What is the capital of France?'])
        self.assertEqual(result.returncode, 0, f"buddyai command failed. Stderr: {result.stderr}")
        self.assertIn("Execution Plan", result.stdout, "Output should contain 'Execution Plan'")
        self.assertIn("Executing Step", result.stdout, "Output should contain 'Executing Step'")
        self.assertIn("Result:", result.stdout, "Output should contain 'Result:'")
        # The exact phrasing of the answer can vary. Check for the core information.
        self.assertIn("Paris", result.stdout, "Output should contain 'Paris'")

    def test_02_prompt_from_file(self):
        """Test executing a prompt read from a file."""
        if "GOOGLE_API_KEY" not in os.environ:
            self.skipTest("Skipping LLM-dependent test: GOOGLE_API_KEY not set.")

        prompt_content = "List files in the current directory using a shell command."
        test_prompt_file = "test_prompt.txt"

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(project_root, test_prompt_file)

        with open(file_path, "w") as f:
            f.write(prompt_content)

        result = run_buddyai_command(['--prompt', test_prompt_file])

        if os.path.exists(file_path):
            os.remove(file_path)

        self.assertEqual(result.returncode, 0, f"buddyai command failed. Stderr: {result.stderr}")
        self.assertIn("Execution Plan", result.stdout)
        # Check for a step involving a shell command like 'ls' or similar
        self.assertIn("ls", result.stdout.lower(), "Plan should include 'ls' command or similar for listing files.")
        self.assertIn("Executing shell command:", result.stdout, "Should show execution of a shell command")
        # Check if common files/dirs are listed (e.g., 'buddy_agent', 'tests', 'setup.py')
        # This part of the assertion is environment-dependent, so it's kept general.
        self.assertIn("buddy_agent", result.stdout, "Output from ls should contain 'buddy_agent'")
        self.assertIn("setup.py", result.stdout, "Output from ls should contain 'setup.py'")


    def test_03_context_from_file(self):
        """Test providing context from a file."""
        if "GOOGLE_API_KEY" not in os.environ:
            self.skipTest("Skipping LLM-dependent test: GOOGLE_API_KEY not set.")

        context_content = "My favorite programming language is Python."
        test_context_file = "test_context.txt"
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(project_root, test_context_file)

        with open(file_path, "w") as f:
            f.write(context_content)

        result = run_buddyai_command(['--prompt', 'Based on the context, what is my favorite programming language?', '--context', test_context_file])

        if os.path.exists(file_path):
            os.remove(file_path)

        self.assertEqual(result.returncode, 0, f"buddyai command failed. Stderr: {result.stderr}")
        self.assertIn("Execution Plan", result.stdout)
        self.assertIn("Result:", result.stdout)
        self.assertIn("Python", result.stdout, "Agent should answer 'Python' based on context")

    def test_04_shell_command_echo(self):
        """Test execution of a simple shell command like 'echo'."""
        if "GOOGLE_API_KEY" not in os.environ:
            self.skipTest("Skipping LLM-dependent test: GOOGLE_API_KEY not set.")

        test_phrase = "hello buddy from test"
        result = run_buddyai_command(['--prompt', f"Echo '{test_phrase}' to the terminal using a shell command."])

        self.assertEqual(result.returncode, 0, f"buddyai command failed. Stderr: {result.stderr}")
        self.assertIn("Execution Plan", result.stdout)
        self.assertIn(f"echo '{test_phrase}'", result.stdout.lower().replace('"', "'"), "Plan should include the echo command")
        self.assertIn("Executing shell command:", result.stdout, "Should show execution of echo")
        self.assertIn(test_phrase, result.stdout, f"Output should contain '{test_phrase}'")

    def test_05_failing_shell_command(self):
        """Test how the agent handles a shell command that fails (e.g., exit 1)."""
        if "GOOGLE_API_KEY" not in os.environ:
            self.skipTest("Skipping LLM-dependent test: GOOGLE_API_KEY not set.")

        # Using a command that is likely to exist and fail, like 'false' or 'exit 1'
        # Some shells might not have 'exit 1' as a direct executable in subprocess.run with shell=True
        # 'false' is a standard utility that does nothing and exits with a non-zero status.
        # Or, a more complex command that the LLM might generate for "Run a command that fails"
        prompt_for_failure = "Run a shell command that is designed to fail with a non-zero exit code, for example, the command `false`."
        result = run_buddyai_command(['--prompt', prompt_for_failure])

        self.assertEqual(result.returncode, 0, f"Agent itself failed. Stderr: {result.stderr}") # Agent should not crash
        self.assertIn("Execution Plan", result.stdout)
        self.assertIn("false", result.stdout.lower(), "Plan should include the 'false' command or similar")
        self.assertIn("Executing shell command:", result.stdout)
        # Check for indication of failure in the output
        self.assertIn("Return Code:", result.stdout, "Output should show a non-zero Return Code for the failed command")
        self.assertNotIn("Return Code: 0", result.stdout, "Return code for the failing step should not be 0")


    def test_06_no_prompt_error_handling(self):
        """Test CLI error handling when no prompt is provided."""
        result = run_buddyai_command([]) # No arguments
        self.assertNotEqual(result.returncode, 0, "Command should fail without a prompt")
        # Argparse usually prints to stderr for errors
        self.assertIn("usage: buddyai", result.stderr.lower(), "Stderr should contain usage information")
        self.assertIn("error: the following arguments are required: -p/--prompt", result.stderr.lower(), "Stderr should indicate prompt is required")

    def test_07_directory_context(self):
        """Test providing a directory as context."""
        if "GOOGLE_API_KEY" not in os.environ:
            self.skipTest("Skipping LLM-dependent test: GOOGLE_API_KEY not set.")

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        test_dir_name = "test_context_dir_07"
        test_dir_path = os.path.join(project_root, test_dir_name)

        os.makedirs(test_dir_path, exist_ok=True)
        with open(os.path.join(test_dir_path, "file1.txt"), "w") as f:
            f.write("Content of file1 for context.")
        with open(os.path.join(test_dir_path, "file2.md"), "w") as f:
            f.write("## Markdown Content\nMore context from file2.")

        result = run_buddyai_command(['--prompt', 'Summarize the content of file1.txt from the provided directory context.', '--context', test_dir_name])

        if os.path.exists(test_dir_path):
            shutil.rmtree(test_dir_path)

        self.assertEqual(result.returncode, 0, f"buddyai command failed. Stderr: {result.stderr}\nStdout: {result.stdout}")
        self.assertIn("Execution Plan", result.stdout)
        self.assertIn("Context Provided:", result.stdout) # Check that context loading is mentioned
        self.assertIn("file1.txt", result.stdout) # CLI should indicate context from file1.txt
        self.assertIn("file2.md", result.stdout)  # CLI should indicate context from file2.md
        self.assertIn("Content of file1 for context.", result.stdout, "Summary should reflect content of file1.txt")


if __name__ == "__main__":
    # This allows running the tests directly with `python tests/test_cli.py`
    # However, it's more common to use `python -m unittest tests.test_cli` or a test runner like pytest.
    unittest.main()
