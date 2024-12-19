from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import messages_to_dict, AIMessage, HumanMessage

# Create a memory buffer with summary support
memory = ConversationSummaryBufferMemory(
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash"), return_messages=True
)

# Initialize the chat model
chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

while True:
    # Get user input
    user_input = input("\nYou: ")

    # Exit condition
    if user_input.lower() in ['bye', 'exit']:
        print("Goodbye!")
        print("Conversation History:", memory.buffer)
        break

    # Add user input to memory and predict response
    memory.save_context({"input": user_input}, {})
    response = chat.predict_messages(
        messages_to_dict(memory.load_memory_variables({})["history"])
    )
    print("\nAI:", response.content)
    # Save AI response to memory
    memory.save_context({}, {"response": response.content})
