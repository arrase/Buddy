import os
import subprocess
from langchain.tools import Tool
from rich.console import Console
from rich.markdown import Markdown

# Rich Console for internal logging within the tool
_tool_console = Console()

# Agent's Current Working Directory - Managed within this module
AGENT_WORKING_DIRECTORY = os.getcwd()

def execute_shell_command(command: str) -> str:
    """
    Executes a shell command, managing a virtual working directory, and returns its output.
    Handles 'cd' commands by changing the virtual working directory.
    Output is formatted as Markdown.
    """
    global AGENT_WORKING_DIRECTORY # Declare intention to modify global variable

    command_trimmed = command.strip()

    try:
        # Handle 'cd' command
        if command_trimmed.startswith("cd "):
            path_to_change = command_trimmed[3:].strip()

            if not path_to_change: # 'cd' or 'cd '
                 # Consider changing to user's home directory by default, like a real shell
                 # For now, let's require an argument or make it explicit 'cd ~'
                return ":warning: 'cd' requires an argument. Use 'cd ~' for home or 'cd /path/to/dir'."

            if path_to_change == "~" or path_to_change == "$HOME":
                new_dir = os.path.expanduser('~')
            elif os.path.isabs(path_to_change):
                new_dir = path_to_change
            else:
                new_dir = os.path.join(AGENT_WORKING_DIRECTORY, path_to_change)

            new_dir = os.path.normpath(new_dir)

            if os.path.isdir(new_dir):
                AGENT_WORKING_DIRECTORY = new_dir
                success_msg = f":heavy_check_mark: Changed virtual working directory to: `{AGENT_WORKING_DIRECTORY}`"
                _tool_console.print(Markdown(success_msg)) # Internal log
                return success_msg # Return this for execute_step to render
            else:
                error_msg = f":x: Error changing directory: `{new_dir}` is not a valid directory (from CWD: `{AGENT_WORKING_DIRECTORY}`)."
                _tool_console.print(Markdown(error_msg)) # Internal log
                return error_msg

        # For other commands, execute in AGENT_WORKING_DIRECTORY
        _tool_console.print(f"Executing: `{command}` in CWD: `{AGENT_WORKING_DIRECTORY}`")

        process = subprocess.run(
            command,
            shell=True, # Using shell=True for convenience; be mindful of security.
            check=False,
            capture_output=True,
            text=True,
            timeout=30, # 30-second timeout for commands
            cwd=AGENT_WORKING_DIRECTORY # Use the agent's current working directory
        )

        output_parts = []
        stdout_content = process.stdout.strip()
        stderr_content = process.stderr.strip()

        if stdout_content:
            output_parts.append(f"**Stdout:**\n```text\n{stdout_content}\n```")
        if stderr_content:
            output_parts.append(f"**Stderr:**\n```text\n{stderr_content}\n```")

        if process.returncode != 0:
            output_parts.append(f"**Return Code:** `{process.returncode}`")
        elif not output_parts: # Command succeeded but no output
            output_parts.append("*Command executed successfully with no output.*")

        returned_output_string = "\n\n".join(output_parts).strip()
        # _tool_console.print(Markdown(f"*Shell tool returning to executor:*\n{returned_output_string}")) # Optional: internal log

        return returned_output_string

    except subprocess.TimeoutExpired:
        error_message_md = f":x: **Timeout Error:** Command `{command}` timed out after 30 seconds (CWD: `{AGENT_WORKING_DIRECTORY}`)."
        _tool_console.print(Markdown(error_message_md)) # Internal log
        return error_message_md
    except Exception as e:
        error_message_md = f":x: **Execution Error:** While running `{command}` (CWD: `{AGENT_WORKING_DIRECTORY}`): `{e}`"
        _tool_console.print(Markdown(error_message_md)) # Internal log
        return error_message_md

shell_tool = Tool(
    name="ShellCommandExecutor",
    func=execute_shell_command,
    description=(
        "Executes a given shell command on a Linux system and returns its standard output and standard error. "
        "Use this for tasks requiring interaction with the operating system, such as file manipulation (ls, cat, echo, etc.), "
        "package management (apt, pip), running scripts (python script.py), or checking versions (python --version). "
        "The tool maintains a virtual current working directory. 'cd <path>' commands will change this virtual directory for subsequent commands. "
        "Always provide full commands. Example: `ls -la` or `python --version` or `cd my_folder`."
    )
)

# Example usage (for testing this module directly)
if __name__ == '__main__':
    test_console = Console()
    test_console.rule("[bold yellow]Testing Shell Tool Module")

    test_console.print(Markdown(f"Initial CWD: `{AGENT_WORKING_DIRECTORY}`"))

    # Test 1: List files
    test_console.print(Markdown("--- Test 1: `ls -la` ---"))
    output1 = shell_tool.run("ls -la")
    test_console.print(Markdown(output1))
    test_console.print("")

    # Test 2: Change directory
    test_console.print(Markdown("--- Test 2: `cd ..` ---"))
    output2 = shell_tool.run("cd ..")
    test_console.print(Markdown(output2))
    test_console.print(Markdown(f"CWD after `cd ..`: `{AGENT_WORKING_DIRECTORY}`"))
    test_console.print("")

    # Test 3: Command in new CWD
    test_console.print(Markdown("--- Test 3: `pwd` (after cd ..) ---"))
    output3 = shell_tool.run("pwd") # pwd might not be the best if shell=True has its own context sometimes
                                    # but for basic test it's ok. `ls` might be more reliable.
    test_console.print(Markdown(output3))
    test_console.print(Markdown(f"CWD check: `{AGENT_WORKING_DIRECTORY}` (should match PWD if PWD worked as expected)"))
    test_console.print("")

    # Test 4: Invalid command
    test_console.print(Markdown("--- Test 4: `nonexistentcommand` ---"))
    output4 = shell_tool.run("nonexistentcommand")
    test_console.print(Markdown(output4))
    test_console.print("")

    # Test 5: cd to a non-existent directory
    test_console.print(Markdown("--- Test 5: `cd /nonexistentdir123` ---"))
    output5 = shell_tool.run("cd /nonexistentdir123")
    test_console.print(Markdown(output5))
    test_console.print(Markdown(f"CWD after failed `cd`: `{AGENT_WORKING_DIRECTORY}`"))

    # Test 6: Timeout (create a command that sleeps)
    test_console.print(Markdown("--- Test 6: `sleep 35` (should timeout) ---"))
    # Note: The timeout in execute_shell_command is 30s.
    # This test might be slow. Consider reducing sleep for tests or making timeout configurable.
    # output6 = shell_tool.run("sleep 5") # Reduced for faster testing
    # test_console.print(Markdown(output6))
    test_console.print("Skipping sleep test for brevity in this example run.")
    test_console.rule(style="bold yellow")
