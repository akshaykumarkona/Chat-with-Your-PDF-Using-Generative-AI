# Chat with Your PDF Using Generative AI (Langchain-RAG) and Vector DB (Chroma)

This project demonstrates a question-answering system built with Langchain libraries. The system leverages a pre-trained Generative AI model (Gemini-1.5-flash) and Chroma for document retrieval and response generation.

## Functionality

The code allows users to ask questions about a loaded PDF document. It then performs the following steps:

1. **Document Preprocessing**: Splits the PDF document into smaller chunks using a `CharacterTextSplitter`.
2. **Embedding Generation**: Generates vector representations (embeddings) for each document chunk using a `GoogleGenerativeAIEmbeddings` model.
3. **Document Indexing**: Creates a Chroma database by indexing the document chunks and their embeddings.
4. **Question Answering Chain**:
   - Defines a `ChatPromptTemplate` to format user questions and provide context from the document.
   - Combines the retrieval functionality of the Chroma database with the `ChatGoogleGenerativeAI` model to answer questions.

## Dependencies

This project requires the following Langchain libraries:

- `langchain_google_genai`
- `langchain.prompts`
- `langchain_community.document_loaders`
- `langchain_community.vectorstores`
- `langchain_text_splitters`

**Note**: Ensure these libraries are installed before running the script.

## Usage

### Set Up Your Environment

1. **Install the Required Libraries**:
   Follow the installation instructions on the [Langchain website](https://docs.langchain.com/docs/installation).

2. **Set Your API Key**:
   Replace `"your-api-key"` in `os.environ["GOOGLE_API_KEY"]` with your actual Google API Key (required for using the Generative AI model).

### Run the Script

1. Open the script in your preferred Python IDE or terminal.
2. Execute the script to test the question-answering functionality.
