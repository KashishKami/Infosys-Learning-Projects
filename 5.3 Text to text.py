from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
google_api_key = os.getenv("GOOGLE_API_KEY")


if __name__ == "__main__":

    desired_lang = input("Enter the language you want the translation: ")

    summary_prompt = """
    You are a trasnlator where you have given an input: {input_text}
    And you need to translate this input text into {translate} language.
    """

    prompt_template = PromptTemplate(input_variables=["input_text", "translate"], template=summary_prompt)

    # Initialize the LLM with the API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=google_api_key
    )

    # Set up the chain
    chain = prompt_template | llm | StrOutputParser()

    # Invoke the chain
    res = chain.invoke({
        "input_text": "You just answer. It's like a reflex. Do I look fat? No. Is she prettier than I am? No.",
        "translate": desired_lang
    })

    print(res)
