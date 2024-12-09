import json
import os
import sys

from typing import Any, Literal, Optional, List
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage

def apply_tool_call_format(
    tool_call: ChatCompletionMessageToolCall, content: str
) -> dict:
    """
    Formats the response of a tool call to be returned to the model.
    Args:
        - tool_call (ChatCompletionMessageToolCall) : The tool call object
        - content (str) : This is the tool response (i.e. results from executing the tool)

    Returns:
        - dict : The formatted tool response to be returned to the model
    """
    return {
        "role": "tool",
        "content": content, # e.g. "5"
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name
    }

class SimpleAgent:
    def __init__(
        self,
        task: Any = None,
        model: Literal["gpt-4o-mini"] = "gpt-4o-mini",
        tools: Optional[List[Any]] = None,
        chat_history: Optional[List[dict]] = None,
    ):
        self.model = model
        self.task = task
        self.tools = tools
        self.client = OpenAI()
        self.chat_history = chat_history if chat_history else []

    def get_response(self, use_tool: bool = True) -> ChatCompletionMessage:
        """
        Get the response from the model via an API call, with the option of tool calling.

        Args:
            use_tool (bool): Whether to use tool calling or not

        Returns:
            ChatCompletionMessage: The response from the model
        """
        if use_tool:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.chat_history,
                tools=[tool.description for tool in self.tools],
                tool_choice="auto",
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.chat_history
            )

        return response.choices[0].message

    def execute_tool_calls(self, message: ChatCompletionMessage) -> List[str]:
        """
        Execute the tool calls in the message and return a list of tool_responses.

        Args:
            message (ChatCompletionMessage): The message containing the tool calls

        Returns:
            List[str]: A list of tool responses (as strings, we'll format them correctly in run())
        """
        tool_responses = []
        for tool_call in message.tool_calls:
            args = tool_call.function.arguments

            kwargs = json.loads(args)
            func = [tool for tool in self.tools if tool.name == tool_call.function.name][0]

            if self.verbose: 
                # Print tool name and arguments
                print(f"Executing tool: {tool_call.function.name}")
                print(f"Arguments: {kwargs}")

            result = func.execute(**kwargs, task=self.task)
            tool_responses.append(result)
        
        return tool_responses

    def run(self, with_tool: bool = True) -> ChatCompletionMessage:
        """
        Default implementation of run method.
        This can be overridden in subclasses for specific behavior.

        Args:
            with_tool (bool): Whether to use tool calling or not

        Returns:
            str: The response from the model
        """
        print(f"Running SimpleAgent...")
        instruction = self.task.current_task_instruction
        self.chat_history.append(apply_user_format(instruction))
        response = self.get_response(use_tool=with_tool)
        return response

class ManTool():
    name = "lookup_manpage"

    @staticmethod
    def execute(command: str, task: Any = None) -> str:
        """
        Looks up the manpage for a CLI command and returns the output, truncated to 500 lines.
        
        Args:
            command (str): The command to look up
            task (Any): Not used in this function
            
        Returns:
            str: The output of the manpage lookup, truncated to 500 lines
        """
        import subprocess
        try:
            result = subprocess.run(f"man {command}", shell=True, capture_output=True, text=True)
            output_lines = result.stdout.splitlines()
            truncated_output = "\n".join(output_lines[:500])
            return f"Manpage lookup successful.\nOutput: {truncated_output}\nErrors: {result.stderr}"
        except Exception as e:
            return f"Error looking up manpage: {str(e)}"

    @property
    def description(self):
        description = """This is a manpage lookup tool.
        
        What it does:
        Looks up the manpage for a given terminal command and returns its output.

        When it should be used:
        This tool should be used to understand the usage and options of a terminal command.

        Tool limitations:
        - Commands should be valid and available in the system's manpages
        - Some commands may have limited or no manpage information
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The CLI command to look up in the manpages"
                        },
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                }
            }
        }

class CLITask:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.is_solved = False
        self.final_command = None

    @property
    def system_instruction(self) -> dict:
        return {
            "role": "system",
            "content": """You are a CLI command generator. Your task is to:
            1. Understand the user's request for a CLI command
            2. Research and validate the command using the man pages when needed
            3. Determine if more research/validation is needed
            4. Once confident, output ONLY the final command with no explanation
            
            Format your response as either:
            - "NEED_MORE_INFO: <specific question or aspect to research>"
            - "FINAL_COMMAND: <the command>"
            
            Be precise and security-conscious when generating commands."""
        }

    @property
    def current_task_instruction(self) -> str:
        return f"Generate a CLI command for: {self.prompt}"

class CLIAgent(SimpleAgent):
    def __init__(
        self,
        task: CLITask,
        tools: List[Any] = [ManTool()],
        model: str = "gpt-4o-mini",
        chat_history: List[dict] = None,
        verbose: bool = True,
        max_iterations: int = 5,
    ):
        super().__init__(model=model, task=task, tools=tools, chat_history=chat_history)
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.start()

    def start(self):
        """Initialize the chat with system instructions"""
        self.chat_history = [
            self.task.system_instruction,
            {"role": "user", "content": self.task.current_task_instruction}
        ]

    def generate_reason(self) -> ChatCompletionMessage:
        """Generate reasoning about the command"""
        return self.get_response(use_tool=False)

    def generate_action(self) -> ChatCompletionMessage:
        """Generate and potentially test the command"""
        self.chat_history.append({
            "role": "user",
            "content": "Based on your reasoning, generate and optionally test the appropriate command."
        })
        return self.get_response(use_tool=True)

    def handle_tool_calls(self, message: ChatCompletionMessage):
        """
        Handle the tool calls from the model response. This function should:
        - Execute the tool calls
        - Append the tool calls and responses to the chat history

        Args:
            message (ChatCompletionMessage): The message containing the tool calls
        """
        # Execute the tool calls
        results = self.execute_tool_calls(message)

        # Append the tool calls and responses to the chat history
        tool_messages = [apply_tool_call_format(tool_call, result)
                         for tool_call, result in zip(message.tool_calls, results)]

        self.chat_history.extend(tool_messages)

        return self.get_response(use_tool=True)

    def run(self):
        """Run the CLI agent in a loop until it's confident about the command"""
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1

            # If verbose, print iteration number and message content
            if self.verbose:
                print(f"Iteration {iterations}")
            
            # Get the next response
            response = self.get_response(use_tool=True)
            self.chat_history.append(response)
            
            # Handle any tool calls
            if response.tool_calls:
                self.handle_tool_calls(response)
                continue
                
            # Check if we have a final command
            content = response.content.strip()
            if content.startswith("FINAL_COMMAND:"):
                self.task.final_command = content.replace("FINAL_COMMAND:", "").strip()
                self.task.is_solved = True
                break
            elif content.startswith("NEED_MORE_INFO:"):
                continue  # Loop continues with the updated chat history
            
        return self.task.final_command

def generate_cli_command(prompt: str, verbose: bool = False) -> str:
    """
    Generate a CLI command from a natural language description.
    
    Args:
        prompt (str): Natural language description of the desired command
        
    Returns:
        str: Generated CLI command only
    """
    task = CLITask(prompt)
    agent = CLIAgent(task=task, verbose=verbose)
    return agent.run()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python terminal_agent.py [-v] 'your command prompt here'")
        sys.exit(1)
        
    # Check for verbose flag
    verbose = False
    if '-v' in sys.argv:
        verbose = True
        sys.argv.remove('-v')
    
    # Join all arguments after the script name to handle prompts with spaces
    prompt = ' '.join(sys.argv[1:])
    print(generate_cli_command(prompt, verbose=verbose))
