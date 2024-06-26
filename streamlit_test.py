import os
import gc
import tempfile
import uuid
import base64

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

from pathlib import Path

from pandasai import SmartDataframe

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
# client = None

# Hugging Face login
hf_token = os.getenv("HF_TOKEN", "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")  # Use environment variable for token
login(token=hf_token, add_to_git_credential=True)    

# ------------------------    CODING SECTION  ----------------------------

# Function to reset chat
def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()
    
    
# ------------------------    PDF SECTION  ----------------------------

# display pdf files     
def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)
    
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
        model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
        encode_kwargs = {'normalize_embeddings': True}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,                      
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
                    display_pdf(file)                                            
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

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(jpg_file):
    bin_str = get_base64_of_bin_file(jpg_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/jpeg;base64,%s");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('images/36ff87fa-f66f-4794-a0c7-f9e6ad826917.jpeg')        

# CSS to make the header white and bold
header_style = '''
<style>
.header {
    color: white;
    font-weight: bold;
}
</style>
'''

# Embed the CSS in the Streamlit app
st.markdown(header_style, unsafe_allow_html=True)
                    
# Main interface
col1, col2 = st.columns([6, 1])

with col1:    
    st.markdown('<h2 class="header">AI Document/Data \n\n\nAssistant</h2>', unsafe_allow_html=True)    

with col2:
    st.button("Clear ↺", on_click=reset_chat)
    
# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()    

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        lines = str(message["content"]).splitlines()
        for line in lines:
            st.markdown(f'<span style="color:white;">{line}</span>', unsafe_allow_html=True)

# Accept user input
if prompt := st.chat_input("Ready for document/data analysis?"):    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        # st.markdown(prompt)
        st.markdown(f'<span style="color:white;">{prompt}</span>', unsafe_allow_html=True)        

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
            message_placeholder_pdf.markdown(f'<span style="color:white;">{full_response}</span>', unsafe_allow_html=True)                
                
        else:
            streaming_response = perform_csv().chat(prompt) 
            full_response = streaming_response                                                                   
            message_placeholder_csv.markdown(f'<span style="color:white;">{full_response}</span>', unsafe_allow_html=True)            
            

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})