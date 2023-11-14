from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
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

from PyPDF2 import PdfReader
import fitz  # PyMuPDF


# Helper function to convert PDF to images and extract text using Tesseract OCR


def get_text_from_any_pdf(pdf_bytes):
    images = convert_pdf_to_img(pdf_bytes)
    final_text = ""
    for pg, img in enumerate(images):
        final_text += convert_image_to_text(img)
    return final_text

# Helper function to convert PDF to images


def convert_pdf_to_img(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    return images

# Helper function to convert image to text using Tesseract OCR


def convert_image_to_text(img):
    text = pytesseract.image_to_string(img)
    return text

# Main function to extract text from a PDF file


def pdf_to_text(pdf_bytes):
    return get_text_from_any_pdf(pdf_bytes)

# Function to create or load the conversation chain


@st.cache_resource
def get_conversation_chain(_vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to create or load the vector store


def get_vectorstore(texts, pdf):
    for single_pdf in pdf:
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

# Main function to run the Streamlit app


def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="streamlit")

    os.environ["OPENAI_API_KEY"] = settings.KEY
    st.write(css, unsafe_allow_html=True)

    st.session_state["conversation"] = None
    st.session_state["chat_history"] = None
    if "session_state" not in st.session_state:
        st.session_state["session_state"] = None

    if st.button("Reload page"):
        st.cache_resource.clear()
        st.session_state["conversation"] = None
        st.session_state["chat_history"] = None
        st.session_state["session_state"] = None
        st.experimental_rerun()

    st.title('Pdf Chat App')
    st.header('Chat with PDF')

    pdf = st.file_uploader("Upload your Pdf", type='pdf',
                           accept_multiple_files=True)
    raw_text = ''
    if pdf is not None:
        for single_pdf in pdf:
            # pdfreader = PdfReader(single_pdf)
            # for i, page in enumerate(pdfreader.pages):
            #     content = page.extract_text()
            #     if content:
            #         raw_text += content
            pdf_bytes = single_pdf.read()
            raw_text += pdf_to_text(pdf_bytes)

    if 'raw_text' in locals():
        # st.write(raw_text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        # st.write(texts)


    if len(texts) > 0:
        doc_search = get_vectorstore(texts, pdf)
        st.session_state["conversation"] = get_conversation_chain(doc_search)

    query = st.text_input("Ask questions about Pdf file:")
    if query:
        if len(texts) > 0:
            handle_userinput(query)
        else:
            st.write(
                'No data extracted from pdf uploaded. Please upload a correct pdf.')


if __name__ == '__main__':
    main()
