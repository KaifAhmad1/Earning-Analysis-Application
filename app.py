# Import necessary packages for Streamlit app, PDF processing, and OpenAI integration.
import pinecone
import streamlit as st
import pdfplumber
import os
from langchain.vectorstores import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load and verify environment variables, specifically the OpenAI API key.
# Load .env file that is in the same directory as your script.
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found. Make sure you have an .env file with the key defined.")

# Initializing the Embedding Model and OpenAI Model 
embeddings = OpenAIEmbeddings(model_name="ada")
# Initialize the AnnoyIndex for vector storage and retrieval
vectorstore = pinecone(dim=len(embeddings.embed_query("Hello world")), metric="euclidean")

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.1, model_name='gpt-4'),
    retriever=vectorstore.as_retriever()
)

# Process an uploaded PDF document and extract its text content.
def process_document(uploaded_file):
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            document_text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return document_text
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None

# Summarize the extracted text from the document using the OpenAI language model.
def summarize_document(llm, document_text):
    try:
        text_splitter = CharacterTextSplitter(max_length=1000, chunk_overlap=20)
        texts = text_splitter.split_text(document_text)
        docs = [Document(content=t) for t in texts]
        summarize_chain = load_summarize_chain(llm, chain_type='map_reduce')
        return summarize_chain.run(docs)
    except Exception as e:
        st.error(f"Error summarizing document: {e}")
        return None

# Initialize a conversation chain with memory capabilities for the chatbot.
def initialize_conversation_chain(llm):
    return ConversationalRetrievalChain(
        llm=llm,
        memory=ConversationBufferWindowMemory(k=5)  # Stores the last 5 interactions.
    )

# Define the main function to run the Streamlit application.
def run_app():
    try:
        llm = initialize_conversation_chain(OPENAI_API_KEY)

        st.title("Earnings Call Analysis App")

        # UI for document upload and processing.
        uploaded_file = st.file_uploader("Upload your earnings call transcript", type=["pdf"])
        process_button = st.button("Process Document")

        # Process document and generate summaries
        if process_button and uploaded_file:
            with st.spinner('Processing Document...'):
                document_text = process_document(uploaded_file)
                if document_text:
                    summaries = summarize_document(llm, document_text)
                    if summaries:
                        display_summaries(summaries)
                        st.success("Document processed!")

        # UI for interactive chatbot with memory feature.
        conversation_chain = initialize_conversation_chain(llm)
        user_input = st.text_input("Ask a question about the earnings call:")
        if st.button('Get Response'):
            with st.spinner('Generating response...'):
                response = generate_chat_response(conversation_chain, user_input, document_text)
                st.write(response)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Display summaries on the app interface and provide download option for each.
def display_summaries(summaries):
    if summaries:
        for i, summary in enumerate(summaries):
            st.subheader(f"Topic {i+1}")
            st.write("One-line topic descriptor: ", summary.get("one_line_summary", ""))
            st.write("Detailed bulleted topic summaries: ", summary.get("bulleted_summary", ""))
            download_summary(summary.get("bulleted_summary", ""), i+1)

# Create a downloadable summary file.
def download_summary(summary, topic_number):
    summary_filename = f"topic_{topic_number}_summary.txt"
    st.download_button(
        label=f"Download Topic {topic_number} Summary",
        data=summary,
        file_name=summary_filename,
        mime="text/plain"
    )

# Generate a response from the chatbot based on the user's input and document's context.
def generate_chat_response(conversation_chain, user_input, document_text):
    response = conversation_chain.generate_response(
        prompt=user_input,
        context=document_text
    )
    return response.get('text', "Sorry, I couldn't generate a response.")

if __name__ == "__main__":
    run_app()