import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings

headers={
        "authorization": st.secrets["api_key"],
        "content-type": "application/json"
    }
os.environ["GOOGLE_API_KEY"]=st.secrets["api_key"]


def load_pdf(pdf_path):
    print("Loading PDF...")
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()


def build_vector_db(docs):
    print("Splitting PDF into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        # separator="\n",
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len
    )
    chunks = splitter.split_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    print("Generating embeddings using HuggingFace (FREE)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("🧠 Creating Chroma Vector DB...")
    db = Chroma.from_documents(documents=chunks, embedding=embeddings)

    return db


def ask_question(db, question):
    print("\n🤖 Thinking...")

    # 1) Retrieve relevant text chunks
    retriever = db.as_retriever()
    docs = retriever.invoke(question)

    context_text = "\n\n".join([d.page_content for d in docs])

    # 2) LLM Call
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=500,
    )

    prompt = f"""
            Answer the following question based only on the provided context.
            Never try to make up the answer.
            Think step by step before providing a detailed answer. 
            
            <context>
            {context_text}
            </context>

    Question: \n{question}
    """

    response = llm.invoke(prompt)
    return response.content


def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 🗨️")
    st.text("Please be with patience until the uploaded PDF loads 😊")
    st.text("Consider uploading PDF files less than 15MB")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_pdf.read())

        with st.spinner("Loading PDF..."):
            docs = load_pdf("temp.pdf")

        with st.spinner("Preparing Vector Database..."):
            db = build_vector_db(docs)

        st.success("PDF processed! You can now ask questions.")

    user_question = st.text_input("Ask something about the PDF:")

    if st.button("Ask") and user_question:
        with st.spinner("Thinking..."):
            answer = ask_question(db, user_question)

        st.subheader("Answer")
        st.write(answer)


    else:
        st.info("Upload a PDF to begin.")


if __name__ == "__main__":
    main()
