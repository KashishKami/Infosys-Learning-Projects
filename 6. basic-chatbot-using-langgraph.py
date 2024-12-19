#https://www.langchain.com/langgraph

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain.prompts.prompt import PromptTemplate

from typing import TypedDict

from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
google_api_key = os.getenv("GOOGLE_API_KEY")


# Managing the state of the chatbot
class ChatbotState(TypedDict):
    history: list

# Creating a node to generate a response using LLM.
def llm_node(state: ChatbotState) -> ChatbotState:
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

    # Get last user message from history
    user_message = state["history"][-1]
    # give us the last message from the history, which is a list!
    
    prompt = """
    You are a helpful chatbot. {userInput}
    """

    prompt_template = PromptTemplate(input_variables=["userInput"], template=prompt)
    chain = prompt_template | llm 
    response = chain.invoke({"userInput": user_message})
    
    state["history"].append(response.content)
    return state

def create_graph():
    # defining the graph
    graph = StateGraph(ChatbotState)

    # Add the node
    graph.add_node("llm_response", llm_node)

    # Define the flow of the graph
    graph.set_entry_point("llm_response")  # Starting node
    graph.add_edge("llm_response", END)   # Ending node
    
    return graph

if __name__ == "__main__":
    # Building the graph
    chatbot_graph = create_graph()
    app = chatbot_graph.compile()

    # Initialize the chatbot state with an empty history list
    state: ChatbotState = {"history": []}

    # Chat loop
    print("Chatbot: Hello!")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print(f"History: {state['history']}")
            print("Chatbot: Goodbye!")
            break
        
        # Update state with user input
        state["history"].append(user_input)
        
        # Run the graph
        state = app.invoke(state)
        
        # Print the last response
        print(f"Chatbot: {state['history'][-1]}")


        # Yes, the entire graph runs each time you invoke it. However:
        # At each iteration of the chat loop, the entire graph is executed from the entry point to the END. 
        # The graph is small (only llm_response → END), so it appears as if a single node is running.
        # If the graph had more nodes (e.g., conditionally branching nodes), those would also run as defined by the edges and flow.