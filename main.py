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

# os.environ["GOOGLE_API_KEY"] = "AIzaSyBMSTBqYv74VqltxMj7G8eUtbuQg8tUROg"


def load_pdf(pdf_path):
    print("Loading PDF...")
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()


def build_vector_db(docs):
    print("Splitting PDF into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        separator="\n",
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
    You MUST answer ONLY using the context below.
    If the context does not contain the answer, say "I cannot find that in the PDF."

    --- CONTEXT START ---
    {context_text}
    --- CONTEXT END ---

    Question: \n{question}
    """

    response = llm.invoke(prompt)
    return response.content


def main():
    # pdf_path = r"E:\DesktopFolders\GenAI\Projects\chat-with-your-pdf\sample_ml.pdf"# input("📂 Enter PDF file path: ").strip()
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









# import streamlit as st
# import os
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma
# from langchain_core.prompts import ChatPromptTemplate

# def main():

#     headers={
#         "authorization": st.secrets["api_key"],
#         "content-type": "application/json"
#     }
#     os.environ["GOOGLE_API_KEY"]=st.secrets["api_key"]
    
#     st.set_page_config(page_title="Ask your PDF")
#     st.header("Ask your PDF 🗨️")
#     st.text("Please be with patience until the uploaded PDF loads 😊")
#     st.text("Consider uploading PDF files less than 15MB")
    
#     pdf = st.file_uploader("Upload your PDF", type="pdf")
    
#     if pdf is not None:
#         # Saving the uploaded file to a temporary location
#         with open("temp_uploaded_pdf.pdf", "wb") as f:
#             f.write(pdf.getbuffer())
        
#         loader = PyPDFLoader("temp_uploaded_pdf.pdf")
#         docs = loader.load()

#         text_splitter = CharacterTextSplitter(
#             separator="\n",
#             chunk_size=1500,
#             chunk_overlap=350,
#             length_function=len
#         )
#         chunks = text_splitter.split_documents(docs)
        
#         # g_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         g_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
#         db = Chroma.from_documents(documents=chunks, embedding=g_embeddings)
        
#         user_question = st.text_input("Ask a question about your PDF:")

#         if user_question:            
#             gemini_llm = ChatGoogleGenerativeAI(
#                 model="gemini-2.0-flash",
#                 max_output_tokens=450,
#                 temperature=0.3
#             )
            
#             # Designing the ChatPrompt Template
#             chat_prompt = ChatPromptTemplate.from_template("""
#             Answer the following question based only on the provided context.
#             Never try to make up the answer.
#             Think step by step before providing a detailed answer. 
#             <context>
#             {context}
#             </context>
#             Question: {input}""")
            
#             from langchain.chains.combine_documents import create_stuff_documents_chain
#             from langchain.chains import create_retrieval_chain

#             combined_chain=create_stuff_documents_chain(gemini_llm, chat_prompt)

#             retriever=db.as_retriever()

#             retriver_chain=create_retrieval_chain(retriever,combined_chain)

#             response_from_RAG=retriver_chain.invoke({"input":user_question})

#             st.write(response_from_RAG['answer'])

# if __name__ == '__main__':
#     main()



