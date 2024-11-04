import streamlit as st
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# Load the PDF
doc = PyPDFLoader(r"C:\Users\moham\OneDrive\Desktop\tezee.pdf")
docs = doc.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Persist directory path
persist_directory = r"C:\Users\moham\Downloads\data"

# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)

# Define the Ollama LLM
local_model = "llama3.1"
llm = ChatOllama(model=local_model)

# Define the query prompt template
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Create the multi-query retriever
retriever = MultiQueryRetriever.from_llm(
    vectorstore.as_retriever(),
    llm,
    prompt=QUERY_PROMPT
)

# Define the RAG prompt and chain
template = """Answer the question based ONLY on the following context:
{context}
Question: {question} and give the answer directly
"""
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def stream_parser(stream):
    for chunk in stream:
        yield chunk['message']['content']

def rag_chain(question):
    # Execute the RAG chain
    response = chain.invoke({"question": question})
    return response

# Streamlit UI setup
st.title("Ollama RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_prompt := st.chat_input("What would you like to ask?"):
    # Display user prompt in chat message widget
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Add user's prompt to session state
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Display spinner while generating response
    with st.spinner('Generating response...'):
        # Retrieve response from model using RAG chain
        llm_response = rag_chain(user_prompt)

        # Display the response in the chat interface
        with st.chat_message("assistant"):
            st.markdown(llm_response)

        # Append the response to the session state
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
