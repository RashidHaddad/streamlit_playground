__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun


# os.environ["APIFY_API_TOKEN"] = "apify_api_SAP0d1xFEFs7mylGNKajb3WIcXSxYg3PDKLA"
# os.environ["OPENAI_API_KEY"] = "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"


apify = ApifyWrapper()
# Call the Actor to obtain text from the crawled webpages
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={
        "startUrls": [{"url": "https://www.cit.tum.de/en/cit/studies/students/"}]
    },
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)

data = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=HuggingFaceEmbeddings())

llm = ChatOpenAI()
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)

retriever = vectorstore.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)



# def generate_response_pdf(uploaded_file, openai_api_key, query_text):
#     # Load document if file is uploaded
#     if uploaded_file is not None:
#         documents = []
#         pdf = pypdf.PdfReader(uploaded_file)
#         for p in range(len(pdf.pages)):
#             page = pdf.pages[p]
#             text = page.extract_text()
#             documents += [text]

#     # documents = loader.load()
#     # Split documents into chunks
#     text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.create_documents(documents)
#     # Select embeddings
#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     # Create a vectorstore from documents
#     db = Chroma.from_documents(texts, embeddings)
#     # Create retriever interface
#     retriever = db.as_retriever()
#     # Create QA chain
#     qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
#     return qa.run(query_text)



import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

st.title("🔎 LangChain - Chat with search")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)






# # Page title
# st.set_page_config(page_title='🦜🔗 Ask the Doc App')
# st.title('🦜🔗 Ask the Doc App')
# # Query text
# query_text = st.text_input('Enter your question:', placeholder = 'How many credits do I need to graduate?')
# # Form input and query
# result = []
# with st.form('myform', clear_on_submit=True):
#     submitted = st.form_submit_button('Submit', disabled=not(query_text))
#     if submitted and openai_api_key.startswith('sk-'):
#         with st.spinner('Calculating...'):
#             response = qa(query_text)
#             result.append(response)
#             # del openai_api_key
# if len(result):
#     st.info(response)
