# CLI Command Generator

A Python-based tool that generates CLI commands from natural language descriptions using GPT-4. The tool leverages man pages and iterative refinement to produce accurate and secure command-line instructions.

## Features

- Natural language to CLI command conversion
- Automatic man page lookup for command validation
- Iterative refinement process for command accuracy
- Built-in safety checks and validation
- Verbose mode for debugging and learning

## Prerequisites

- Python 3.6+
- OpenAI API access
- Access to system man pages

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install openai
```
3. Set up your OpenAI API key:
```python
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### As a Command Line Tool

```bash
python terminal_agent.py [-v] 'your command prompt here'
```

Options:
- `-v`: Enable verbose mode to see the agent's reasoning process

Example:
```bash
python terminal_agent.py "find all PDF files modified in the last 24 hours"
```

### As a Python Module

```python
from terminal_agent import generate_cli_command

# Generate a command
command = generate_cli_command("find all PDF files modified in the last 24 hours")
print(command)

# With verbose mode
command = generate_cli_command("find all PDF files modified in the last 24 hours", verbose=True)
```

## Architecture

The tool consists of several key components:

- `CLITask`: Represents a command generation task with its requirements
- `CLIAgent`: Manages the interaction with the GPT model and tools
- `ManTool`: Provides access to system man pages for command validation
- `SimpleAgent`: Base class providing core agent functionality

## How It Works

1. The user provides a natural language description of the desired command
2. The agent iteratively:
   - Analyzes the request
   - Researches necessary commands using man pages
   - Validates potential solutions
   - Refines the command
3. Once confident, the agent outputs the final command

## Safety Features

- Maximum iteration limit to prevent infinite loops
- Man page validation of commands
- System instructions emphasizing security consciousness
- Input validation and sanitization

## Limitations

- Requires access to system man pages
- Man page output truncated to 500 lines
- Maximum 5 iterations per command generation
- Depends on OpenAI API availability

## Error Handling

The tool includes error handling for:
- Invalid commands
- Missing man pages
- API failures
- Invalid inputs

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

[Insert appropriate license information here]

## Acknowledgments

- Built using OpenAI's GPT-4 API
- Utilizes system man pages for command validation
