from dotenv import load_dotenv
import os

load_dotenv()

# Loading environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
google_api_key = os.getenv("GOOGLE_API_KEY")




if __name__ == "__main__":
    
    # 1. Loading Documents
    from langchain_community.document_loaders import TextLoader

    print("Loading Documents...")
    loader = TextLoader("/README.md")
    document = loader.load()  # Loads the documents with metadata
    print(f"Loaded {len(document)} documents")



    # 2. Splitting Documents
    from langchain.text_splitter import CharacterTextSplitter

    print("Splitting Documents...")
    splitter = CharacterTextSplitter()
    documents = splitter.split_documents(document)
    print(f"Split {len(document)} documents into {len(documents)} chunks")



    # 3. Embedding Documents
    from langchain_huggingface import HuggingFaceEmbeddings

    #print("Embedding started...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #print("Embedding finished...")



    # 4. Inserting Documents into VectorDB
    from langchain_pinecone import PineconeVectorStore

    print("Inserting Documents into VectorDB...")
    vector_db = PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print(f"Inserted {len(documents)} documents into VectorDB")



