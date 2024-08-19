import streamlit as st
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

def main():

    headers={
        "authorization": st.secrets["gemini_api_key"],
        "content-type": "application/json"
    }
    
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 🗨️")
    st.text("Please be with patience until the uploaded PDF loads 😊")
    st.text("Consider uploading PDF files less than 15MB")
    
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        # Saving the uploaded file to a temporary location
        with open("temp_uploaded_pdf.pdf", "wb") as f:
            f.write(pdf.getbuffer())
        
        loader = PyPDFLoader("temp_uploaded_pdf.pdf")
        docs = loader.load()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1500,
            chunk_overlap=350,
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)
        
        g_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma.from_documents(documents=chunks, embedding=g_embeddings)
        
        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:            
            gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                max_output_tokens=350,
                temperature=0
            )
            


            # prompt_template = PromptTemplate.from_template(
            #     "You are an assistant that helps with questions about PDFs. "
            #     "Don't try to make up the answer. "
            #     "Just answer to the point and in short (don't give long answers). "
            #     "Question: {question}"
            # )
            
            # prompt_template = ChatPromptTemplate.from_template(
            #     [{"role": "system", "content": '''You are an assistant that helps with questions about PDFs. 
            # Don't try to make up the answer.
            # Just answer to the point and in short (don't give long answers)'''},
            #      {"role": "user", "content": "{question}"}]
            # )
      

            # Designing the ChatPrompt Template
            chat_prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context.
            Never try to make up the answer.
            Think step by step before providing a detailed answer. 
            I will tip you $1000 if the user finds the answer helpful. 
            <context>
            {context}
            </context>
            Question: {input}""")

            
            from langchain.chains.combine_documents import create_stuff_documents_chain

            combined_chain=create_stuff_documents_chain(gemini_llm, chat_prompt)

            retriever=db.as_retriever()

            from langchain.chains import create_retrieval_chain

            retriver_chain=create_retrieval_chain(retriever,combined_chain)

            response_from_RAG=retriver_chain.invoke({"input":user_question})

            st.write(response_from_RAG['answer'])
            # print(response_from_RAG)


if __name__ == '__main__':
    main()
