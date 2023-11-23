import streamlit as st
import pickle
import requests
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import openai
import time

openai.api_key = st.secrets["OPENAI_API_KEY"]

def main():
    if 'text' not in st.session_state:
        st.session_state['text'] = ""

    #st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.header("Chat to a Document 💬👨🏻‍💻🤖")
    
    # Add a radio button for the user to select the input method
    input_method = st.radio("Choose your input method:", ("Paste text or web address", "Ask a question about the documents in the index"))

    if input_method == "Paste text or web address":
        # Paste text or URL
        text_or_url = st.text_area("Paste your text or URL here: URLS must be in format https://")
        process_button = st.button("Process Text")
        store_name = "pasted_text_or_url"
        
        if process_button:
            if text_or_url:
                # Check if it's a URL
                if text_or_url.startswith('http://') or text_or_url.startswith('https://'):
                    # It's a URL, fetch the content
                    response = requests.get(text_or_url)

                    # Check if it's a HTML page
                    if 'text/html' in response.headers['Content-Type']:
                        # Parse the HTML content with BeautifulSoup
                        soup = BeautifulSoup(response.content, 'html.parser')
                        # Extract all paragraph texts
                        text = ' '.join(p.get_text() for p in soup.find_all('p'))
                        st.text_area("**Fetched Information from site:  Note that some websites block content access so the fetched information may be limited**", text)  # Display the fetched information in a text box
                    else:
                        # It's not a HTML page, just use the raw content
                        text = response.text
                        st.text_area("Fetched Information:", text)  # Display the fetched information in a text box

                    store_name = "fetched_url_content"
                else:
                    # It's not a URL, just use the pasted text
                    text = text_or_url
                    store_name = "pasted_text"
                st.session_state['text'] = text  # Store the text in the session state

        # Check if text is provided
        if not st.session_state['text']:  # Use the text from the session state
            st.error("Please provide some text either by uploading a document or pasting text.")
            return

        # Process the pasted text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_text(text=st.session_state['text'])  # Use the text from the session state

        vector_index = pinecone.Index(index_name)
        vectors = embeddings.embed(chunks)  # Assuming `embeddings.embed` returns a list of vectors
        vector_index.upsert(vectors)

    # Ask a question about the documents in the index
    query = st.text_input("Ask question's about your document:")

    suggestions = ["", "What is the main topic of the document?", "Summarize the document in 200 words?", "Provide a bullet point list of the key points mentioned in the document?", "Create the headings and subheadings for Powerpoint slides", "Translate the first paragraph to French"]

    suggestion = st.selectbox("Or select a suggestion: (ENSURE QUESTION FIELD ABOVE IS BLANK)", suggestions, index=0)

    if query:
        query_vector = embeddings.embed([query])[0]  # Assuming `embeddings.embed` returns a list of vectors
        results = vector_index.query(queries=[query_vector], top_k=3)
        docs = [chunks[i] for i in results.ids[0]]  # Assuming `results.ids[0]` is a list of indices
        llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-3.5-turbo', max_tokens=2000, temperature=0.5)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb, st.spinner('Working on response...'):
            time.sleep(3)
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)

    elif suggestion:
        query = suggestion
        query_vector = embeddings.embed([query])[0]  # Assuming `embeddings.embed` returns a list of vectors
        results = vector_index.query(queries=[query_vector], top_k=3)
        docs = [chunks[i] for i in results.ids[0]]  # Assuming `results.ids[0]`
        llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-3.5-turbo', max_tokens=2000, temperature=0.5)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb, st.spinner('Working on response...'):
            time.sleep(3)
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        st.write(response)

if __name__ == '__main__':
    main()
