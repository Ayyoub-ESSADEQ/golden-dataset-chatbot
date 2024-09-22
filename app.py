from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.llamafile import Llamafile
from langchain.chains import RetrievalQA

file_path = "./data/constituents-financials.csv"

loader = CSVLoader(file_path=file_path)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

texts = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
)

vector_store = FAISS.from_documents(texts, embeddings)
vector_store_retriever = vector_store.from_documents(
    documents=texts,
    embedding=embeddings,
)

tiny_llama_llm = Llamafile()

retriever_dict = {
    "model_type": "FAISS",
    "input_value": vector_store_retriever,
    "input_type": "FAISS",
}

qa_chain = RetrievalQA.from_llm(llm=tiny_llama_llm, retriever=retriever_dict)

question = "What is the price for Aetna Inc ?"
result = qa_chain({"query": question})
print(result["result"])
