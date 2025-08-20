import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from vertexai import init
from google.oauth2 import service_account


# ========================
# 1. Preparar los datos
# ========================
@st.cache_resource
def load_vectorstore():
    df = pd.read_csv("data/products.csv")

    # Convertir filas en documentos
    docs = [
        Document(
            page_content=f"{row['name']}. {row['description']}. Precio: ${row['price']}",
            metadata={"id": row["id"]}
        )
        for _, row in df.iterrows()
    ]

    creds = service_account.Credentials.from_service_account_file(
        "crm-inteligenze-3ede20115ea1.json"
    )

    # Embeddings (puedes cambiar a VertexAIEmbeddings si ya tienes GCP activo)

    init(
        project="crm-inteligenze",  # 👈 pon aquí tu PROJECT_ID de GCP
        location="us-central1",   # o la región donde activaste Vertex AI
        credentials=creds
    )

    embeddings = VertexAIEmbeddings(model="text-embedding-004")

    # Vector store FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# ========================
# 2. Crear el chatbot
# ========================
@st.cache_resource
def create_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    creds = service_account.Credentials.from_service_account_file(
        "crm-inteligenze-3ede20115ea1.json"
    )

    llm = VertexAI(model="gemini-2.5-flash", temperature=0.3, credentials=creds)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return chain

# ========================
# 3. Interfaz Streamlit
# ========================
st.set_page_config(page_title="Ecommerce Chatbot RAG", page_icon="🛍️")

st.title("🛒 Ecommerce Chatbot con RAG")
st.write("Hazme preguntas sobre los productos disponibles.")

# Estado de sesión
if "chain" not in st.session_state:
    st.session_state.chain = create_chain()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    role, text = msg
    with st.chat_message(role):
        st.write(text)

# Entrada de usuario
if user_input := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    response = st.session_state.chain.invoke({"question": user_input})
    answer = response["answer"]

    st.session_state.messages.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.write(answer)
