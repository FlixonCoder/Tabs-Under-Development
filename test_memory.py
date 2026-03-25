from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import asyncio

# 1. Define the Agent with Ollama backend
# Note: Use 'ollama_chat/' prefix for better reliability with tools
root_agent = Agent(
    name="LocalMemoryAgent",
    model=LiteLlm(model="ollama_chat/qwen2.5:3b"),
    instruction="You are an assistant that remembers user preferences across sessions.",
    description="A local agent with persistent multi-session memory."
)

async def main():
    # 2. Initialize Memory/Session Service
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent, 
        session_service=session_service, 
        app_name="my_local_memory_app"  # Add this parameter
    )

    # 3. Start a session for a specific user
    user_id = "user_123"
    session = await session_service.create_session(
        user_id=user_id, 
        app_name="my_local_memory_app"
    )

    # 4. Interact with the agent
    print("Agent is ready. Type 'exit' to quit.")
    while True:
        query = input("User: ")
        if query.lower() == "exit": break

        new_message = types.Content(role="user", parts=[types.Part(text=query)])

        async for event in runner.run_async(user_id=user_id, session_id=session.id, new_message=new_message):
            if event.is_final_response():
                print(f"Agent: {event.content.parts[0].text}")

if __name__ == "__main__":
    asyncio.run(main())