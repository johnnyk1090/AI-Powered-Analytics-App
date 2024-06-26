import os
import gc
import tempfile
import uuid

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

from pathlib import Path

from pandasai import SmartDataframe

from streamlit_pdf_viewer import pdf_viewer

from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

from langchain_chroma import Chroma

from huggingface_hub import login
import streamlit as st

# Session state initialization
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}    

session_id = st.session_state.id
client = None

# Hugging Face login
hf_token = os.getenv("HF_TOKEN", "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")  # Use environment variable for token
login(token=hf_token, add_to_git_credential=False)    


# ------------------------    CODING SECTION  ----------------------------


# Function to reset chat
def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()
    
    
# ------------------------    PDF SECTION  ----------------------------
    
def perform_pdf():                 
    if file_key not in st.session_state.get('file_cache', {}):             
        try:                        
            loader = PyMuPDFLoader(
                file_path
            )
            docs = loader.load()
            st.success("Ready to Rumble!")
        except:    
            st.error('Could not find the file you uploaded, please check again...')
            st.stop()                                        
        
        # define the llm model                                        
        llm = ChatOllama(model='llama3', temperature=0)
        
        # instance of document transformer
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        # Adjust this based on the correct attribute of Document objects
        docs = text_splitter.split_documents(docs)
                            
        st.write("Loading the embedding model and the document in vector store...\nThis might take a while...")
        model_name = "thenlper/gte-base"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs            
        )
        st.write("Model loaded!")
        
        # load the docs and the embedding function to chroma db                                         
        db = Chroma.from_documents(docs, hf)
        st.write("Store loaded!")   
        
        # define the retriever
        retriever = db.as_retriever()      
                                        
        # define the prompt
        template = """You are a professional AI PDF assistant that reads the document given and replies to 
        the user's questions always with accurate answers. If there is a chat history take it under consideration
        and DO NOT repeat yourself. 
        The form of the answer should always be different for each question.                     

        {context}
        
        Question: {question}

        Helpful Answer:"""
                            
        custom_rag_prompt = PromptTemplate.from_template(template)                    
        
        retrieval_chain = (                        
            {"context": retriever, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm           
            | StrOutputParser()             
        )
                                
        st.session_state.file_cache[file_key] = retrieval_chain
    else:
        retrieval_chain = st.session_state.file_cache[file_key]        
    
    return retrieval_chain       
    
# ------------------------    CSV SECTION  ----------------------------                         
    
# display csv files    
def display_csv(file):
    # Opening file from file path
    st.markdown("### CSV Preview")    
    st.dataframe(pd.read_csv(file))
    
def perform_csv():                
    if file_key not in st.session_state.get('file_cache', {}):                                     
        
        # define the llm model
                                                
        llm = ChatOllama(model='llama3', temperature=0)
        # create the Smart DataFrame
        sdf = SmartDataframe(file_path, config={"llm": llm})       
        st.success("Ready for Data Analysis!") 
                                                           
        st.session_state.file_cache[file_key] = sdf

    else:                
        sdf = st.session_state.file_cache[file_key]                                                     
    
    return sdf
    
# ------------------------    READING THE UPLOADED FILES SECTION  ----------------------------    
        
# Sidebar for file upload
with st.sidebar:    
    st.header("Upload your documents!!")
    
    file = st.file_uploader("Choose your `.pdf` or `.csv` file", type=["pdf", "csv"])                    
    
    if file is not None:                    
        try:
            with tempfile.TemporaryDirectory() as temp_dir:            
                file_path = os.path.join(temp_dir, file.name)
                
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())
                
                file_key = f"{session_id}-{file.name}"
                st.write("Indexing your document...")
                
                # get the suffix    
                suffix = Path(file.name).suffix                
                st.write(f"File format provided : {suffix}")

                if suffix == ".pdf":                                        
                    pdf_viewer(file_path, height=1000)                                            
                    perform_pdf()
                elif suffix == ".csv":
                    display_csv(file)                                                    
                    perform_csv()
                else: 
                    st.write("Provide only .pdf or .csv files!") 
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()                                       
    else:
        st.info("Please upload a file.")            
        
# ------------------------    CSS SECTION  ----------------------------        
                    
# Main interface
col1, col2 = st.columns([6, 1])

with col1:
    st.header("AI Document/Data Analyst")

with col2:
    st.button("Clear ↺", on_click=reset_chat)
    
# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()    

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ready for document/data analysis?"):    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder_pdf = st.empty()
        message_placeholder_csv = st.empty()
        full_response = ""        
                
        # Simulate stream of response with milliseconds delay
        if suffix == ".pdf":
            streaming_response = perform_pdf().invoke(prompt)    
            for chunk in streaming_response:
                full_response += chunk
            message_placeholder_pdf.markdown(full_response)
        else:
            streaming_response = perform_csv().chat(prompt) 
            full_response = streaming_response           
            message_placeholder_csv.markdown(full_response)                    

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})