import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

from bs4 import BeautifulSoup
import requests
from requests.exceptions import RequestException

st.title("AiraBot: News Research Tool ")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.0, max_tokens=500)

if process_url_clicked:
    # load data
    data = ""
    main_placeholder.text("Data Loading...")
    print(urls)
    for url in urls:
        if url:
            print("HJNKJKJBK")
            response = requests.get(url)
            print(response)
            if response.status_code == 200:
                print(response.content)
                soup = BeautifulSoup(response.content, "html.parser")
                print(soup)
                text = soup.get_text(separator="\n")
                data += text
        else:
            st.warning("Empty URL detected. Skipping...")

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter Started...")
    docs = []
    for chunk in text_splitter.split_text(data):
        doc = {"page_content": chunk}
        docs.append(doc)

    print(docs)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    print(embeddings)
    retry_attempts = 3
    retry_delay = 4  # seconds
    for attempt in range(retry_attempts):
        try:
            texts = [doc["page_content"] for doc in docs]
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            break  # Successful embedding generation, exit the retry loop
        except RequestException as e:
            if attempt < retry_attempts - 1:
                main_placeholder.text(f"Error communicating with OpenAI, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                main_placeholder.text("Failed to generate embeddings. Please try again later.")
                raise e

    main_placeholder.text("Embedding Vector Building...")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)


query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)