"""
LLM Module — Qwen2.5:3b via Ollama with streaming
Provides a clean class interface for the rest of the application.
"""

import ollama
from typing import Generator, List, Dict

SYSTEM_PROMPT = """You are TABS, a highly intelligent personal AI voice assistant.
Your user's name is Mohammed Saqib Junaid Khan. You address him as Sir.
You are concise, warm, and direct in your spoken responses.
Since your responses will be converted to speech, avoid using markdown formatting,
bullet point symbols, code blocks, or special characters. Speak in natural sentences.
Keep responses conversational and appropriately brief unless deep explanation is requested.
You have a built-in Reminder System that handles creating, querying, editing, and deleting
reminders automatically. When the user sets or asks about reminders or events, the system
handles it directly. You do not need to simulate reminders yourself."""

MODEL_NAME = "qwen2.5:3b"
MAX_TOKENS = 1024


class LLMEngine:
    """Handles all LLM interactions with Qwen2.5:3b via Ollama."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, model: str = MODEL_NAME):
        self.model = model
        self.system_prompt = system_prompt
        print(f"✅ LLM Engine ready — model: {self.model}")

    def stream_response(self, messages: List[Dict]) -> Generator[str, None, None]:
        """
        Stream tokens from the LLM given a list of message dicts.
        Each dict has 'role' ('system'/'user'/'assistant') and 'content'.

        Yields individual token strings.
        """
        # Always inject the system prompt at the front
        full_messages = [
            {"role": "system", "content": self.system_prompt}
        ] + messages

        stream = ollama.chat(
            model=self.model,
            messages=full_messages,
            stream=True,
            options={"num_predict": MAX_TOKENS},
        )

        for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield content

    def get_full_response(self, messages: List[Dict]) -> str:
        """
        Non-streaming version — collects full response and returns as string.
        Useful for agent routing decisions.
        """
        return "".join(self.stream_response(messages))


if __name__ == "__main__":
    engine = LLMEngine()
    msgs = [{"role": "user", "content": "Say hello briefly."}]
    print("AI: ", end="", flush=True)
    for token in engine.stream_response(msgs):
        print(token, end="", flush=True)
    print()
