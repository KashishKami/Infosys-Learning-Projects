from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain

# It creates a memory for the conversation
memory = ConversationBufferMemory(
    return_messages=True
)

# Initialize the chat model
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# It creates a conversation chain with the chat model and the memory
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    verbose = True
    #verbose =true shows all the past content along with each query!
)
#new version use pipe for chain!

while True:
    # Here we are taking user input
    user_input = input("\nYou: ")

    # Check for exit command
    if user_input.lower() in ['bye', 'exit']:
        print("Goodbye!")
        print(conversation.memory.buffer)
        break

    # Here we are getting the response from the AI
    response = conversation.predict(input=user_input)
    print("\nAI:", response)
