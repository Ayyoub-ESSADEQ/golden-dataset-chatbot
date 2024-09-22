from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.llamafile import Llamafile
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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


vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    collection_name="companies_data_collection",
    persist_directory="./chroma",
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},
)

tiny_llama_model = Llamafile()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm=tiny_llama_model,
    prompt=prompt,
)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)
response = rag_chain.invoke(
    {
        "input": "What is the sector on which Aetna Inc operates ?",
    }
)

print(response["answer"])