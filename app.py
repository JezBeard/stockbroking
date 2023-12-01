import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import openai
import time

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set the Pinecone API key
pinecone.api_key = st.secrets["PINECONE_API_KEY"]
pinecone.environment = st.secrets["PINECONE_ENVIRONMENT"]

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

index_name = 'stocks3'

# Create an instance of pinecone.Index
index = pinecone.Index(index_name)

embeddings = OpenAIEmbeddings()

docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Initialize the vector store object
vectorstore = Pinecone(index, embed_model, "text")

def parse_response(response):
    st.write(response['result'])
    st.write('\n\nSources:')
    for source_name in response["source_documents"]:
        st.write(source_name.metadata['source'], "page #:", source_name.metadata['page'])

def main():
    st.header("AI ASX Documents chat 💬 🤖")

    # Ask a question about the documents in the index
    query = st.text_input("Ask question's about ASX Quarterly Activites Reports : Note the current available test documents are Aurora Energy, Morella Corporation and Alpha HPA")

    suggestions = ["", "Provide an overview of the Morella ASX announcement.", "What are the highlights of the Aurora Energy Quarterly Report?", "Provide a bullet point list of the key points mentioned in the Alpha HPA report"]

    suggestion = st.selectbox("Or select a suggestion: (ENSURE QUESTION FIELD ABOVE IS BLANK)", suggestions, index=0)

    if query or suggestion:
        if suggestion:
            query = suggestion

        llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-3.5-turbo', max_tokens=2000, temperature=0.5)
        retriever = docsearch.as_retriever(include_metadata=True, metadata_key = 'source')
        qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                          chain_type="stuff", 
                                          retriever=retriever, 
                                          return_source_documents=True)
        response = qa_chain(query)
        parse_response(response)

if __name__ == '__main__':
    main()
