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
from langchain.vectorstores import FAISS

# os.environ["APIFY_API_TOKEN"] = "apify_api_SAP0d1xFEFs7mylGNKajb3WIcXSxYg3PDKLA"
# os.environ["OPENAI_API_KEY"] = "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"

embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.load_local("faiss_index", embeddings)

llm = ChatOpenAI()
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)

retriever = vectorstore.as_retriever()
qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)


map_template = """You are a translator who only translates into English. Output only the translated query.
Query:{query}"""


from langchain.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_template(map_template)

chat = ChatOpenAI(temperature=0.0)

def translate(question):
  messages = prompt_template.format_messages(query=question)
  response = chat(messages)
  return response.content

def chat_rag(question):
  response = translate(question)
  # print(response)
  docs= retriever.get_relevant_documents(response)
  docers = set([i.metadata['source'] for i in docs])

  return qa(response)['answer'], docers

def beautify(answer, sources):
    return answer + "\n" + "\n".join(sources)

import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

# st.title("🔎 LangChain - Chat with search")
company_logo = 'https://de.wikipedia.org/wiki/Datei:Logo_of_the_Technical_University_of_Munich.svg'
# Configure Streamlit page
st.set_page_config(
    page_title="TUM GPT",
    page_icon=company_logo
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a TUM Bot who can help you with your TUM queries. How can I help you?"}
    ]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# if query := st.chat_input("Ask me anything"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": query})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(query)
#     with st.chat_message("assistant", avatar=company_logo):
#         message_placeholder = st.empty()
#         # Send user's question to our chain
#         result = chain({"question": query})
#         response = result['answer']
#         full_response = ""
#         # Simulate stream of response with milliseconds delay
#         for chunk in response.split():
#             full_response += chunk + " "
#             time.sleep(0.05)
#             # Add a blinking cursor to simulate typing
#             message_placeholder.markdown(full_response + "▌")
#         message_placeholder.markdown(full_response)
#     # Add assistant message to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})


if prompt := st.chat_input(placeholder="Ask anything about TUM?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant", avatar=company_logo):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        answer, sources = chat_rag(prompt)
        response = beautify(answer, sources)
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
