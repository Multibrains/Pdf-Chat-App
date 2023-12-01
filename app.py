from pdf_utils import download_pdf_from_url, pdf_to_text
from vectorstore_utils import get_vectorstore, get_conversation_chain, handle_userinput
from chat_app import run_streamlit_app

def main():
    run_streamlit_app()

if __name__ == '__main__':
    main()