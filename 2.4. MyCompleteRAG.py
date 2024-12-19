from dotenv import load_dotenv
import os

load_dotenv()

# Loading environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")
google_api_key = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")




if __name__ == "__main__":
    
    # 1. Loading Documents
    
    #from langchain_community.document_loaders import TextLoader
    # print("Loading Documents...")
    # loader = TextLoader("./README.md")
    # document = loader.load()  # Loads the documents with metadata
    # print(f"Loaded {len(document)} documents")

    from langchain_community.document_loaders import PyPDFLoader

    pdf_path = "/Material/Document 1.pdf"
    loader = PyPDFLoader(pdf_path)  #Loads the data with metadata!
    document = loader.load()
    print(f"Loaded {len(document)} documents")


    # 2. Splitting Documents
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    print("Splitting Documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True)
    chunks = splitter.split_documents(document)
    print(f"Split {len(document)} documents into {len(chunks)} chunks")

    #print(f"Firstn page content:\n {documents[5].page_content}")



    # 3. Embedding Documents
    from langchain_huggingface import HuggingFaceEmbeddings

    #print("Embedding started...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #print(f"Embedded {len(chunks)} chunks! ")



    # 4. Inserting Documents into VectorDB
    from langchain_pinecone import PineconeVectorStore

    print("Inserting Documents into VectorDB...")
    vector_db = PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
    print(f"Inserted {len(chunks)} documents into VectorDB")



    # 5. Retriver and Prompt Template
    from langchain import hub

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    prompt = hub.pull("rlm/rag-prompt")



    # 6. Format Function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    


    #7. LLM Model
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=google_api_key
    )
    
    
    # 7. RAG chain
    from langchain_core.output_parsers import StrOutputParser
    
    from langchain_core.runnables import RunnablePassthrough

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke("Define risk in detail.")

    print(response)

