import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tqdm import tqdm

DOCUMENTS_DIR = "docs/"
OLLAMA_MODEL = "mistral"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_DB_PATH = "./chroma_db"

def load_all_documents(directory):
    documents = []
    print(f"Loading documents from {directory}...")
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        elif filename.endswith(".docx"):
            try:
                loader = Docx2txtLoader(filepath)
            except ImportError:
                print("Please install 'python-docx' and 'docx2txt'")
                continue
        else:
            print(f"Skipping unsupported file: {filename}")
            continue
        try:
            documents.extend(loader.load())
            print(f"Loaded: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return documents

def split_documents_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, persist_directory):
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Loading existing ChromaDB from {persist_directory}...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        print("ChromaDB loaded.")
    else:
        print(f"Creating new ChromaDB at {persist_directory}...")
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        print("Generating embeddings and adding to ChromaDB:")
        for i, chunk in enumerate(tqdm(chunks, desc="Embedding Progress")):
            vectorstore.add_documents([chunk])
        vectorstore.persist()
        print("\nChromaDB created and persisted.")
    return vectorstore

def initialize_llm():
    print(f"Initializing Ollama LLM with model: {OLLAMA_MODEL}...")
    llm = Ollama(model=OLLAMA_MODEL)
    print("LLM initialized.")
    return llm

def setup_rag_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer in English.

    {context}

    Question: {question}
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("RAG chain setup complete.")
    return qa_chain

if __name__ == "__main__":
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"Error: Document directory '{DOCUMENTS_DIR}' not found.")
        exit()

    all_documents = load_all_documents(DOCUMENTS_DIR)
    if not all_documents:
        print("No documents loaded.")
        exit()
    all_chunks = split_documents_into_chunks(all_documents)
    vector_store = create_vector_store(all_chunks, CHROMA_DB_PATH)
    llm = initialize_llm()
    qa_chain = setup_rag_chain(llm, vector_store)

    print("\n--- RAG System Ready ---")
    print(f"Asking questions to '{OLLAMA_MODEL}' using documents from '{DOCUMENTS_DIR}'")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour Query: ")
        if user_query.lower() == 'exit':
            break
        print("Searching and generating response...")
        try:
            response = qa_chain({"query": user_query})
            print("\nAnswer:")
            print(response["result"])
            if response["source_documents"]:
                print("\nSource Documents Used:")
                for i, doc in enumerate(response["source_documents"]):
                    print(f"--- Document {i+1} ---")
                    print(f"Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"Page/Chunk Content (first 200 chars): {doc.page_content[:200]}...")
            else:
                print("No specific source documents were used.")
        except Exception as e:
            print(f"An error occurred: {e}")
