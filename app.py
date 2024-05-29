import os
import weaviate
import streamlit as st
from io import StringIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from weaviate.embedded import EmbeddedOptions
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from streamlit_extras.stylable_container import stylable_container 
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import warnings
warnings.filterwarnings("ignore")

# Defining Streamlit Session Variables
if 'doc_names' not in st.session_state:
    st.session_state.doc_names = []
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "client" not in st.session_state:
    st.session_state.client = weaviate.Client(embedded_options = EmbeddedOptions())
if "conversational_rag_chain" not in st.session_state:
    st.session_state.conversational_rag_chain = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = ""
if "store" not in st.session_state:
    st.session_state.store = {}

# Function for retreiving chat history 
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Defining constants
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Contextualizing question using chat history
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", contextualize_q_system_prompt),MessagesPlaceholder("chat_history"),("human", "{input}"),])

# Control prompt that answers the question
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, recommend questions that are more relevant.\
Use three sentences maximum and keep the answer concise.\
{context}"""
qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt),MessagesPlaceholder("chat_history"),("human", "{input}"),])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Streamlit Page Setup
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: black;'>Semantic Search using GPT 3.5</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 900px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Displaying all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# If New file is uploaded Then
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None and uploaded_file.name not in st.session_state.doc_names:

    # Extracting text from the document in the form of strings
    stringio=StringIO(uploaded_file.getvalue().decode('utf-8'))
    read_data=stringio.read()

    # Storing document data
    if uploaded_file.name not in st.session_state.doc_names:
        st.session_state.uploaded_docs.append(read_data)
        st.session_state.doc_names.append(uploaded_file.name)

    # For every unique document uplaoded, the contents are chunked and the vectorstore is updated
    split_text = text_splitter.split_text(read_data)
    st.session_state.chunks += [Document(page_content=split_text[x], metadata={"pId":x,"docName":uploaded_file.name}) for x in range(len(split_text))]
    st.session_state.vectorstore = Weaviate.from_documents(client = st.session_state.client,documents = st.session_state.chunks,embedding = OpenAIEmbeddings(),by_text = False)
    retriever = st.session_state.vectorstore.as_retriever()

    # Updating rag chain after every upload
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    st.session_state.conversational_rag_chain = RunnableWithMessageHistory(rag_chain,get_session_history,input_messages_key="input",history_messages_key="chat_history",output_messages_key="answer",)

# Printing unique list of document names
st.sidebar.subheader("Unique list of Uploaded Files")
for i in range(len(st.session_state.doc_names)):
    with st.sidebar:
        st.subheader(str(i+1)+") "+st.session_state.doc_names[i].split(".")[0])

# Sidebar for getting document data on entering a query
with st.sidebar:
    # Enter Query
    text_search = st.text_input("Search relevant documents by Quering", value="")
if text_search != "":
    # Langchain Semantic Similarity
    docs = st.session_state.vectorstore.similarity_search(text_search, k=3)
    # Printing results
    for i in range(3):
        with st.sidebar:
            Page1 = st.container(border=True)   
            Page1 = stylable_container(key="Page1", css_styles=""" {box-shadow: rgba(0, 0, 0, 0.24) 0px 3px 15px;}""")
            Page1.write(str(i+1)+". Doc Name - "+ docs[i].metadata['docName'] + ",  \t\t\t Chunk ID - "+ str(docs[i].metadata['pId']))
            Page1.write("Chunk: "+ docs[i].page_content)

# User text input in the chat
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generating reponse from the rag chain pipeline using chat history
    with st.chat_message("assistant"):
        response = st.session_state.conversational_rag_chain.invoke({"input": st.session_state.messages[-1]['content']},config={"configurable": {"session_id": "abc123"}})["answer"]
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})



