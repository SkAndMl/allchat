import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from streamlit_chat import message



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

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
        huggingfacehub_api_token="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",
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

    st.set_page_config(page_title="ChatPDF",
                       page_icon="🤖")

    st.header("ChatPDF")

    user_query = st.chat_input(placeholder="Ask away...")
    if user_query:
        handle_input(user_query)

    with st.sidebar:
        pdf_docs = st.file_uploader(label="Upload your PDF files here",
                                    accept_multiple_files=True)
        
        if st.button("Process"):
            
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs=pdf_docs)
                text_chunks = get_text_chunks(raw_text=raw_text)
                vector_store = get_vector_store(text_chunks=text_chunks)

                if st.session_state.chain is None:
                    st.session_state.chain = load_chain(vector_store=vector_store)




if __name__ == "__main__":
    main()