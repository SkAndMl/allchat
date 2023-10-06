import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import docx
import os
from dotenv import load_dotenv

load_dotenv()

def get_docx_text(word_doc):
    text = []
    doc = docx.Document(word_doc)
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)


def get_txt_text(txt_doc):
    st.write(txt_doc.getvalue().decode("utf-8"))
    return txt_doc.getvalue().decode("utf-8")

def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_from_files(files):
    texts = []
    for file in files:
        ext = file.name.split(".")[-1]
        if ext=="pdf":
            texts.append(get_pdf_text(file))
        elif ext=="docx":
            texts.append(get_docx_text(file))
        elif ext=="txt":
            texts.append(get_txt_text(file))
    
    return "\n".join(texts)


def get_text_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128
    )
    text_chunks = splitter.split_text(text=raw_text)
    return text_chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={"device" : "cpu"}
    )
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def load_chain(vector_store):

    custom_prompt = """Use the following pieces of context to answer the question given. If you don't know say you don't know. Do not hallucinate.
    Context: {context}
    Question: {question}
    Return helpful answer.
    Answer: 
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt
    )

    model_kwargs = {
        "temperature" : 0.3,
        "max_new_tokens" : 512,
        "top_p" : 0.95,
        "top_k" : 40,
        "repetition_penalty" : 1.15
    }

    llm = HuggingFaceHub(
        repo_id="tiiuae/falcon-7b-instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_kwargs=model_kwargs
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        chain_type_kwargs={"prompt" : prompt},
        retriever=vector_store.as_retriever(search_kwargs={"k" : 2}),
        verbose=True
    )

    return chain

def handle_input(user_query):
    response = st.session_state.chain({"query" : user_query})
    st.session_state["messages"].append({
        "role" : "user",
        "content" : user_query
    })
    st.session_state["messages"].append({
        "role" : "assistant",
        "content" : response["result"]
    })

    for message in st.session_state.messages:
        with st.chat_message(name=message["role"], ):
            st.markdown(message["content"])



def main():
    

    if "chain" not in st.session_state:
        st.session_state.chain = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.set_page_config(page_title="AllChat",
                       page_icon="🤖")

    st.header("AllChat :books:")
    st.caption("Made with :hearts: by [SkAndMl](https://www.linkedin.com/in/sathya-krishnan-suresh-914763217/)")

    user_query = st.chat_input(placeholder="Ask away...")
    if user_query:
        handle_input(user_query)

    with st.sidebar:
        files = st.file_uploader(label="Upload your documents here",
                                    accept_multiple_files=True)
        
        if st.button("Process"):
            
            with st.spinner("Processing..."):
                raw_text = get_text_from_files(files)
                text_chunks = get_text_chunks(raw_text=raw_text)
                vector_store = get_vector_store(text_chunks=text_chunks)
                for file in files:
                    os.remove(file)
                if st.session_state.chain is None:
                    st.session_state.chain = load_chain(vector_store=vector_store)




if __name__ == "__main__":
    main()