__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

load_dotenv()

custom_prompt_template = """Use as informações a seguir para responder à pergunta do usuário. Se você não sabe responder, apenas diga que não sabe, não tente inventar uma resposta.


Context: {context}
Question: {question}

Apenas retorne as respostas e nada mais.
Helpful answer:
"""

DATA_PATH = 'content/'

st.set_page_config(page_title="ChatPDF gemini pro", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("ChatPDF com gemin Pro💬")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Clique no botão para processar o arquivo"}
    ]

with st.sidebar:
    st.title('Sidebar')

if st.button("Process"):
    st.text("Processing ...")    
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if texts:
        docsearch = Chroma.from_documents(texts, embeddings)
    else:
        st.error("Nenhum texto encontrado nos documentos.")
    if 'retriever' not in st.session_state:

        st.session_state.retriever = docsearch.as_retriever()


    message_history = ChatMessageHistory()
    if 'memory' not in st.session_state:

         st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

    if 'prompt' not in st.session_state:

         st.session_state.prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    response = f"Processing done.!"
    st.session_state.messages = [{"role": "assistant", "content": response}]
    
if user_question := st.chat_input("Sua pergunta"):
    st.session_state.messages.append({"role": "user", "content": user_question})
def get_text():
    global input_text
    input_text = st.text_input("Faça uma pergunta", key="input")
    return input_text
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Clique no botão para processar o arquivo"}
    ]
st.sidebar.button('Limpar histórico de conversa', on_click=clear_chat_history)

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, max_output_tokens=2048)
            chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        chain_type="stuff",
                        retriever= st.session_state.retriever,
                        verbose=True,
                        max_tokens_limit = 3000,
                        combine_docs_chain_kwargs={"prompt": st.session_state.prompt},
                        memory= st.session_state.memory,
                        return_source_documents=True,
                    )
            # st_callback = StreamlitCallbackHandler(st.container())
            res = chain(user_question)#, callbacks=[st_callback])
            answer = res["answer"]
            
            placeholder = st.empty()
            placeholder.markdown(answer)
            message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message) # Add response to message history 