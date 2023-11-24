import streamlit as st
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import openai
import time

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set the Pinecone API key
pinecone.api_key = st.secrets["PINECONE_API_KEY"]
pinecone.environment = st.secrets["PINECONE_ENVIRONMENT"]

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

index_name = 'stocks6'

# Create an instance of pinecone.Index
index = pinecone.Index(index_name)

# Initialize the vector store object
vectorstore = Pinecone(index, embed_model.embed_documents, "text")

def main():
    st.header("Chat to a Document 💬👨🏻‍💻🤖")

    # Ask a question about the documents in the index
    query = st.text_input("Ask question's about your document:")

    suggestions = ["", "What is the main topic of the document?", "Summarize the document in 200 words?", "Provide a bullet point list of the key points mentioned in the document?", "Create the headings and subheadings for Powerpoint slides", "Translate the first paragraph to French"]

    suggestion = st.selectbox("Or select a suggestion: (ENSURE QUESTION FIELD ABOVE IS BLANK)", suggestions, index=0)

    if query or suggestion:
        if suggestion:
            query = suggestion

        # Get top 3 results from the vector store
        results = vectorstore.similarity_search(query, k=3)

        # Get the text from the results
        source_knowledge = "\n".join([x.page_content for x in results])

        # Feed into an augmented prompt
        augmented_prompt = f"""Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {query}"""

        llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], model_name='gpt-3.5-turbo', max_tokens=2000, temperature=0.5)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb, st.spinner('Working on response...'):
            time.sleep(3)
            response = chain.run(input_documents=[augmented_prompt], question=query)
            print(cb)
        st.write(response)

if __name__ == '__main__':
    main()
