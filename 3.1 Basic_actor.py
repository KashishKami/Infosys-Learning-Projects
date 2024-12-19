from dotenv import load_dotenv
import os

load_dotenv()

# Loading environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")





from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.tools.render import render_text_description



# 1. Creating tools
@tool("get_length_of_string")
def get_length_of_string(string: str) -> int:
    """Returns the length of the string in characters"""
    # print(f"Getting length of string: {string}")
    text = string.strip().strip('"')
    return len(text)

if __name__ == "__main__":

    # 2. LLM Model
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=GOOGLE_API_KEY
    )

    #   # llm initialization
    # llm = ChatGoogleGenerativeAI(
    #     temperature=0,
    #     model="genai-instruct-turbo-001",
    #     max_tokens=1024,
    # )


    # 3. List the tools inside the array.
    tools = [get_length_of_string]


    # 4. ReAct Promting Template 
    # Reason prompting
    # Action Prompting
    # ReAct
    template = """Answer the following questions as best you can.
    You have access to the following tools:
    {tools}
    
    Use the following format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action (just the value, no function call syntax)
    Observation: the result of the action
    ... (Thought/Action/Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    Thought: """
    
    # 5. Here we can't directly pass tools as it is array as llm receives only text as input
    prompt = PromptTemplate.from_template(template=template).partial(
        #Here we can't directly pass tools as it is array, as llm receives only text as input
        #that's why we are using render_text_description(tools) which will automically create description for the tools for us!
        tools=render_text_description(tools),
        tool_names=", ".join(t.name for t in tools)
    )
    
    from langchain_core.output_parsers import StrOutputParser

    agent = {"input": lambda x: x["input"]} | prompt | llm | StrOutputParser()

    res = agent.invoke({
        "input": "What is the length in characters of the text 'Kashish Yadav'?"
    })

    print(res)


  
    