from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()



# Step 1: Load the PDF document
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "/Material/Excel_QS.pdf"
loader = PyPDFLoader(pdf_path)  #Loads the data with metadata!
docs = loader.load()

# print(docs[0].page_content[:100])




# Step 2: Splitting the document
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

# print(f"Number of Segments: {len(all_splits)}")

# print(f"Firstn page content:\n {all_splits[0].page_content}")




# Step 3: Create embeddings and storing them into a Vector database Chroma DB
# Use the Hugging face embeddings!
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


vector = embeddings.embed_query("hello, world!")
print(vector[:5])



# from langchain_community.vectorstores import Chroma
#vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key = openai_api_embed_key))


# Step 4: Retriever to retrieve the segment which is most similar to the query

#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

#retrieved_docs = retriever.invoke("How to share your work?")

#len(retrieved_docs)



