from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
import streamlit as st
import os
import pickle
import warnings
from configEnv import settings
from htmlTemplates import css, bot_template, user_template
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import time
import io
import glob
from pdf2image import convert_from_bytes
from pytesseract import image_to_string
from langchain import HuggingFacePipeline, PromptTemplate
from PyPDF2 import PdfReader
import fitz  # PyMuPDF

#For link
import requests
from io import BytesIO
from langchain.chains import ConversationalRetrievalChain

DEFAULT_SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
""".strip()


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]
""".strip()


@st.cache_resource
def get_conversation_chain(_vectorstore):
    llm = ChatOpenAI()
    SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

    template = generate_prompt(
        """Use only the chat history and the following information
        {context}
        to answer in a helpful manner to the question.
        Keep your replies short, compassionate and informative.
        {chat_history}

    Question: {question}
    Response:
    """,
        system_prompt=SYSTEM_PROMPT,
    )
    prompt = PromptTemplate(template=template, input_variables=["context", "question","chat_history"])
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="Question",
        ai_prefix="Response",
        input_key="question",
        k=50,
        return_messages=True,
        output_key='answer'
    )

    chain = load_qa_chain(
        llm, chain_type="stuff", prompt=prompt, memory=memory, verbose=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True,
    )
    return qa_chain

# Function to create or load the vector store


def get_vectorstore(texts, pdf):
    for single_pdf in pdf:
        store_name = ""
        if(isinstance(single_pdf, str)):
            store_name = single_pdf
        else:
            store_name = single_pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl") and os.path.getsize(f"{store_name}.pkl") > 0:
            with open(f"{store_name}.pkl", "rb") as f:
                doc_search = pickle.load(f)
                return doc_search
        else:
            with open(f"{store_name}.pkl", "wb") as f:
                embeddings = OpenAIEmbeddings()
                doc_search = FAISS.from_texts(texts, embeddings)
                pickle.dump(doc_search, f)
                return doc_search



# Function to handle user input and display the conversation
def handle_userinput(user_question):
    response = st.session_state["conversation"]({'question': user_question})
    st.session_state["chat_history"] = response['chat_history']

    for i, message in enumerate(st.session_state["chat_history"]):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace(
            "{{MSG}}", message.content), unsafe_allow_html=True)
