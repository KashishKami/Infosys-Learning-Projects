from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded
if google_api_key is None:
    raise ValueError("API key not found. Please make sure GOOGLE_API_KEY is set in your .env file.")

if __name__ == "__main__":

    summary_prompt = """
    Give me the famous books written by the author {author}.
    """

    prompt_template = PromptTemplate(input_variables=["author"], template=summary_prompt)

    # Initialize the LLM with the API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=google_api_key
    )

    # Set up the chai
    chain = prompt_template | llm | StrOutputParser() 

    # Invoke the chain
    res = chain.invoke({"author": "J.K.Rowling"})

    print(res)
