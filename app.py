from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os 
from dotenv import load_dotenv
import streamlit as st

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate


load_dotenv()
# Hugging Face embedding
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
st.secrets['HF_TOKEN'] = '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'
st.secrets['GROQ_API_KEY'] = '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'
groq_api = os.getenv('GROQ_API_KEY')

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name='Search')


uploaded_files = st.file_uploader('choose a PDF file', type = 'pdf', accept_multiple_files=True)

st.title("Chat With Search")


if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {'role': 'assisstant', 'content':"Hi, Im a chatbot who can search the web. How can i help you?"}
    ]


# write the message to user
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# 1 prompt = text input its place holder is 'What is machine learning?'
if prompt:=st.chat_input(placeholder='What is machine learning?'):
    st.session_state.messages.append({'role':'user', 'content':prompt})
    st.chat_message("user").write(prompt)

    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f"./temp.pdf"
        with open(temppdf, 'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
            
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        #add the loaded pdf to documents list
        documents.extend(docs)
        
    # Split and create embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore=Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()
    
    
    llm = ChatGroq(groq_api_key=groq_api, model_name='Llama3-8b-8192', streaming=True)
     # Define a valid PromptTemplate
    prompt_template = PromptTemplate(
        input_variables=["context", "input"],
        template="""You are a helpful assistant. Use the following context to answer the user's question:
        {context}

        Question: {input}
        Answer:"""
    )

     # Combine documents chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain) if uploaded_files else None
    
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handlig_parsing_errors=True)

    with st.chat_message('assistant'):
        # here we show how agent think before giving the last answer
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        if uploaded_files:
            query_result = retrieval_chain.invoke({'input': prompt})
            response = search_agent.run(query_result['answer']+prompt, callbacks=[st_cb])
        else:
            response = search_agent.run(prompt, callbacks=[st_cb])
        #response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant', 'content':response})
        st.write(response)