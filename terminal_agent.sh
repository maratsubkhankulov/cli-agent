#!/bin/bash

# Get the command from the Python script
generated_command=$(python terminal_agent.py "$@")

# Show the generated command
echo -e "\nGenerated command: $generated_command"

# Prompt for execution
read -p "Do you want to execute this command? (y/N): " response

if [[ $response =~ ^[Yy]$ ]]; then
    eval $generated_command
fi