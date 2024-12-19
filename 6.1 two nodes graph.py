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

# Node to uppercase the last chatbot response
def uppercase_response_node(state: ChatbotState) -> ChatbotState:
    # Ensure there is at least one response in history
    if len(state["history"]) > 1:
        # Get the last response (which is always appended after the LLM response)
        last_response_index = -1  # The last item in the history
        state["history"][last_response_index] = state["history"][last_response_index].upper()
    return state


def create_graph():
    # Define the graph
    graph = StateGraph(ChatbotState)

    # Add nodes to the graph
    graph.add_node("llm_response", llm_node)
    graph.add_node("uppercase_response", uppercase_response_node)  # Add the new node

    # Define the flow of the graph
    graph.set_entry_point("llm_response")      # Start at the LLM response
    graph.add_edge("llm_response", "uppercase_response")  # Go to uppercase node
    graph.add_edge("uppercase_response", END)  # End after uppercase

    return graph

if __name__ == "__main__":
    # Building the graph
    chatbot_graph = create_graph()
    app = chatbot_graph.compile()

    # Initialize the chatbot state with an empty history list
    state: ChatbotState = {"history": []}

    # Chat loop
    print("Chatbot: HELLO!")
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
        # If the graph had more nodes (e.g., conditionally branching nodes), those would also run as defined by the edges and flow.