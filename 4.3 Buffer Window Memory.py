from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain

# Initialize memory and conversation chain
memory = ConversationBufferWindowMemory(k=2)  # Tracks or stores only the last 2 interactions
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
conversation = ConversationChain(
    llm=llm, 
    memory=memory,
    verbose = True
)

while True:
    # Get user input
    user_input = input("\nYou: ")
    if user_input.lower() in ['bye', 'exit']:
        print("Goodbye!")
        print(conversation.memory.buffer)
        break
    
    # Get AI response
    response = conversation.predict(input=user_input)
    print("\nAI:", response)
