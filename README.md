# BuddyAI Agent

BuddyAI is an AI agent built with Python using LangGraph. It employs a plan-and-execute architecture to understand user requests, formulate a plan, and execute it step-by-step. The agent can interact with the local shell environment to perform tasks.

This project uses Google's Generative AI models (e.g., Gemini series) via the `langchain_google_genai` library.

## Features

*   **Plan-and-Execute:** Breaks down complex tasks into manageable steps.
*   **Shell Interaction:** Can execute shell commands (e.g., `ls`, `cat`, `echo`, run scripts) in a controlled virtual working directory.
*   **File-based Inputs:** Accepts prompts and context from direct strings or by reading from files/directories.
*   **Rich Output:** Uses the `rich` library for formatted and user-friendly console output, showing the plan and step-by-step execution.

## Project Structure

The project is organized as a Python package named `buddy_agent`:

```
buddy-agent/
├── buddy_agent/            # Main package directory
│   ├── __init__.py
│   ├── cli/                # Command-line interface logic
│   │   ├── __init__.py
│   │   └── main.py
│   ├── core/               # Core agent logic (LangGraph graph, state, nodes)
│   │   ├── __init__.py
│   │   └── graph.py
│   ├── tools/              # Tools for the agent (e.g., shell executor)
│   │   ├── __init__.py
│   │   └── shell.py
│   └── utils/              # Utility functions (file reading, text parsing)
│       └── utils.py
├── tests/                  # Test scripts
│   └── test_cli.py
├── README.md
├── requirements.txt        # Project dependencies
└── setup.py                # Packaging and installation script
```

## Installation

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <repository-url>
    cd buddy-agent
    ```

2.  **Set up the Google API Key:**
    The agent requires a Google API key with access to Generative AI models (e.g., Gemini). Set this key as an environment variable:
    ```bash
    export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```
    Replace `"YOUR_GOOGLE_API_KEY"` with your actual key.

3.  **Install the package:**
    It's recommended to install the package in a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Install the package and its dependencies. For development, install in editable mode:
    ```bash
    pip install -e .
    ```
    For a standard installation:
    ```bash
    pip install .
    ```

## Usage

Once installed, the agent can be run using the `buddyai` command:

```bash
buddyai --prompt "Your task description here"
```

**Parameters:**

*   `-p, --prompt PROMPT`: (Required) The user's request for the agent. This can be:
    *   A direct string (e.g., `"What is the capital of France?"`).
    *   The path to a text file containing the prompt (e.g., `my_prompt.txt`).
*   `-c, --context CONTEXT`: (Optional) Path to a file or a directory to load additional context for the agent.
    *   If a file path, its content is loaded as context.
    *   If a directory path, the content of all files within that directory (recursively) is loaded.

**Examples:**

*   **Simple question:**
    ```bash
    buddyai --prompt "What is the current date and time?"
    ```
*   **Using a prompt file:**
    ```bash
    echo "Create a Python script that prints 'Hello, World!' and then run it." > my_task.txt
    buddyai --prompt my_task.txt
    ```
*   **Providing context from a file:**
    ```bash
    echo "The project ID is 'buddy-123'." > project_info.txt
    buddyai --prompt "What is the project ID?" --context project_info.txt
    ```
*   **Providing context from a directory:**
    ```bash
    mkdir my_docs
    echo "File A contains details about feature X." > my_docs/file_a.txt
    echo "File B contains details about feature Y." > my_docs/file_b.txt
    buddyai --prompt "Summarize the features described in the provided documents." --context my_docs/
    ```

## Development

### Running Tests

Tests are located in the `tests/` directory and use Python's `unittest` framework. To run the tests:

1.  Ensure the package is installed in editable mode (`pip install -e .`).
2.  Ensure `GOOGLE_API_KEY` is set in your environment.
3.  Run the tests from the project root:
    ```bash
    python -m unittest discover -s tests
    ```
    Or, to run a specific test file:
    ```bash
    python -m unittest tests.test_cli
    ```

---
*This README provides an overview of the BuddyAI agent, its structure, installation, and usage.*
