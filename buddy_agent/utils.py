import os
import sys
from typing import List # Added for parse_llm_list_output
from rich.console import Console
from rich.markdown import Markdown

# Console for utility functions, can be a module-level instance
_util_console = Console()

def parse_llm_list_output(text: str) -> List[str]:
    """Parses LLM text output that should be a list into a Python list."""
    lines = text.strip().split('\n')
    parsed_list = []
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line.startswith(("* ", "- ")):
            cleaned_line = cleaned_line[2:]
        elif cleaned_line and cleaned_line[0].isdigit() and cleaned_line[1:3] in (". ", "."):
            cleaned_line = cleaned_line.split(". ", 1)[-1]
        if cleaned_line.startswith('"') and cleaned_line.endswith('"'):
            cleaned_line = cleaned_line[1:-1]
        if cleaned_line.startswith("'") and cleaned_line.endswith("'"):
            cleaned_line = cleaned_line[1:-1]
        if cleaned_line:
            parsed_list.append(cleaned_line)
    return parsed_list

def read_file_content(filepath: str) -> str:
    """Reads and returns the content of a file. Exits on error."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        _util_console.print(Markdown(f":x: [bold red]File Read Error:[/bold red] File not found at `{filepath}`"))
        sys.exit(1)
    except IOError as e:
        _util_console.print(Markdown(f":x: [bold red]File Read Error:[/bold red] Error reading file `{filepath}`: {e}"))
        sys.exit(1)
    except Exception as e: # Catch any other unexpected errors
        _util_console.print(Markdown(f":x: [bold red]Unexpected File Read Error:[/bold red] Error reading file `{filepath}`: {e}"))
        sys.exit(1)


def read_directory_content(dir_path: str) -> str:
    """
    Recursively reads content of all files in a directory.
    Exits on error if the root directory is not found.
    Skips individual files if they cause errors.
    Returns content formatted with Markdown.
    """
    if not os.path.isdir(dir_path):
        _util_console.print(Markdown(f":x: [bold red]Directory Read Error:[/bold red] Directory not found at `{dir_path}`"))
        sys.exit(1)

    _util_console.print(Markdown(f":mag: Reading content from directory: `{dir_path}`"), style="dim")
    all_content = []
    for current_dir_path, _, files_in_current_dir in os.walk(dir_path):
        for file_name in files_in_current_dir:
            file_path_to_read = os.path.join(current_dir_path, file_name)
            try:
                file_content = ""
                try:
                    with open(file_path_to_read, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    all_content.append(f"\n**Source: `{file_path_to_read}`**\n```text\n{file_content}\n```")
                except FileNotFoundError:
                    _util_console.print(Markdown(f":warning: Skipping file `{file_path_to_read}`: File not found unexpectedly."), style="yellow")
                except IOError as e_io:
                    _util_console.print(Markdown(f":warning: Skipping file `{file_path_to_read}` due to IOError: {e_io}"), style="yellow")
                except Exception as e_other:
                     _util_console.print(Markdown(f":warning: Skipping file `{file_path_to_read}` due to unexpected error: {e_other}"), style="yellow")
            except SystemExit:
                _util_console.print(Markdown(f":warning: Critical error reading file `{file_path_to_read}` (process would exit). Propagating."), style="red")
                raise
    if not all_content:
        _util_console.print(Markdown(f":information_source: No readable files found in `{dir_path}` or its subdirectories."), style="dim")
        return ""
    return "\n".join(all_content)

if __name__ == '__main__':
    test_console = Console()
    test_console.rule("[bold yellow]Testing Utils Module[/bold yellow]")

    # Test parse_llm_list_output
    test_console.print("\n--- Test: parse_llm_list_output ---")
    test_text = '1. First item\n- Second item\n* "Third item"\n  Fourth item, still part of list'
    parsed = parse_llm_list_output(test_text)
    test_console.print("Original Text:")
    test_console.print(test_text)
    test_console.print("Parsed List:")
    test_console.print(parsed)
    assert parsed == ["First item", "Second item", "Third item", "Fourth item, still part of list"]

    if not os.path.exists("test_dir"):
        os.makedirs("test_dir/subdir")
    with open("test_dir/test_file1.txt", "w") as f:
        f.write("Hello from test_file1.txt")
    with open("test_dir/subdir/test_file2.txt", "w") as f:
        f.write("Content of test_file2.txt in subdir.")

    test_console.print("\n--- Test 1: Reading a single file (test_file1.txt) ---")
    content1 = read_file_content("test_dir/test_file1.txt")
    test_console.print(Markdown("**Content:**"))
    test_console.print(content1)

    test_console.print("\n--- Test 2: Reading a directory (test_dir) ---")
    dir_content = read_directory_content("test_dir")
    test_console.print(Markdown("**Aggregated Content (Markdown formatted by read_directory_content):**"))
    test_console.print(Markdown(dir_content))

    test_console.rule(style="bold yellow")
