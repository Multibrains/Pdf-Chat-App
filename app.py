from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from typing_extensions import Concatenate
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
import pickle
from langchain.callbacks import get_openai_callback


# from dotenv import load_dotenv
from configEnv import settings
# load_dotenv()

os.environ["OPENAI_API_KEY"] = settings.KEY
st.title('Pdf Chat App')
st.header('Chat with PDF')
pdf = st.file_uploader("Upload your Pdf", type='pdf')
if pdf is not None:
    pdfreader = PdfReader(pdf)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    # st.write(raw_text)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,)
    texts = text_splitter.split_text(raw_text)

    # st.write(texts)

    # #vector support
    if len(texts) > 0:
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                doc_search = pickle.load(f)
            # st.write("Embeddings loaded from disk")
        else:
            with open(f"{store_name}.pkl", "wb") as f:
                embeddings = OpenAIEmbeddings()
                doc_search = FAISS.from_texts(texts, embeddings)
                pickle.dump(doc_search, f)
            # st.write("Embeddings completion completed")
    query = st.text_input("Ask questions about Pdf file:")
    if query:
        if len(texts) > 0:
            chain = load_qa_chain(OpenAI(model_name='gpt-3.5-turbo', temperature=0.3), chain_type='stuff')
            docs = doc_search.similarity_search(query,k=3)
            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
        else:
            st.write('No data extracted from pdf uploaded. Please upload correct pdf.')
