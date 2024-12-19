from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain

# Initialize memory and conversation chain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
memory = ConversationSummaryBufferMemory(
    llm = llm,  #We have to pass the LLM model because the summary is also created by the LLM model.
    max_token_limit=100 # after this limit hit for the buffer, it'll makes the summary of the previous conversations!
)  

conversation = ConversationChain(
    llm=llm, 
    memory=memory, 
    verbose=True
)

while True:
    # Get user input
    user_input = input("\nYou: ")
    if user_input.lower() in ['bye', 'exit']:
        print("Goodbye!")
        print("\nConversation Summary:")
        print(conversation.memory.buffer)
        break
    
    # Get AI response
    response = conversation.predict(input=user_input)
    print("\nAI:", response)
