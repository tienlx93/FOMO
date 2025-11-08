import os
import io
import json
import uuid
import time
import traceback
from datetime import datetime
import tiktoken
from typing import List, Dict, Any, Literal

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
# LangChain core
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters.base import TextSplitter

# Pinecone client for index management
from pinecone import Pinecone, ServerlessSpec

# LLM & Embeddings (OpenAI / Azure OpenAI)
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings

# Loaders & Splitters & VectorStores
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


# from langchain.chains import RetrievalQA

from sentence_transformers import SentenceTransformer

from tts import load_tts, speak
from audio_player import create_audio_player

# Optional TTS via OpenAI (speech synthesis)
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

load_dotenv()

APP_TITLE = "📄 FOMO LangChain Combo"
TICKETS_FILE = os.getenv("TICKETS_FILE", "./data/tickets.json")
os.makedirs(os.path.dirname(TICKETS_FILE), exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./.tmp", exist_ok=True)

LANG_OPTIONS = ["vie", "eng", "kor", "fra", "deu"]
Languages = Literal["vie", "eng", "kor", "fra", "deu"]
SUMMARY_STYLES = ["concise", "detailed", "action-focused"]
SummaryStyles = Literal["concise", "detailed", "action-focused"]

LANGUAGE_MAP = {
    "vie": "Vietnamese",
    "eng": "English",
    "kor": "Korean",
    "fra": "French",
    "deu": "German",
}

# Chat memory per session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# Initialize session state
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""
if 'guide_context' not in st.session_state:
    st.session_state.guide_context = ""
if 'previous_question' not in st.session_state:
    st.session_state.previous_question = ""
if "tts_playing" not in st.session_state:
    st.session_state.tts_playing = {}
if "tts_last_clicked" not in st.session_state:
    st.session_state.tts_last_clicked = None
if "first_load" not in st.session_state:
    st.session_state.first_load = True
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

def log_query(qtype: str, input_text: str, output_text: str, meta: Dict[str, Any] = None):
    row = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "type": qtype,
        "input_len": len(input_text or ""),
        "output_len": len(output_text or ""),
        "meta": json.dumps(meta or {}, ensure_ascii=False),
    }
    path = "./logs/queries.csv"
    header_needed = not os.path.exists(path)
    import csv
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if header_needed:
            w.writeheader()
        w.writerow(row)

# -------------------------
# LLM / Embedding builders
# -------------------------
def build_llm() -> BaseChatModel:
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            temperature=st.session_state.get("temperature", 0.2),
        )
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=st.session_state.get("temperature", 0.2),
    )

def build_embeddings():
    if os.getenv("AZURE_OPENAI_EMBED_API_KEY"):
        return AzureOpenAIEmbeddings(
            api_key=os.getenv("AZURE_OPENAI_EMBED_API_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        )
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

def initialize_tts():
    for lang in LANG_OPTIONS:
        try:
            if f"tts_{lang}_model" not in st.session_state:
                print(f"Loading TTS model for {lang}...")
                st.session_state[f"tts_{lang}_model"] = load_tts(lang)
        except Exception as e:
            st.error(f"Error loading TTS model for {lang}: {e}")

def build_transformers():
    print("Building SentenceTransformer model...")
    return SentenceTransformer("intfloat/e5-base-v2")

def create_pinecone_index_if_not_exists(api_key: str, index_name: str, dimension: int = 1536):
    """
    Check if Pinecone index exists, create it if it doesn't.
    Returns the Pinecone client and index.
    """
    try:
        pc = Pinecone(api_key=api_key)
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            # Create index if it doesn't exist
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            st.success(f"✅ Created new Pinecone index: {index_name}")
        else:
            print(f"📋 Using existing Pinecone index: {index_name}")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        return pc.Index(index_name)
        
    except Exception as e:
        st.error(f"❌ Error with Pinecone index: {str(e)}")
        return None

def initialize_vector_store(embeddings: Embeddings, pinecone_index_name: str, pinecone_api_key: str):
    # Check if Pinecone API key is provided
    if not pinecone_api_key:
        st.warning("⚠️ Please configure Pinecone API key in the sidebar")
        return None
    
    # Check if 'pinecone_index_name' exists, otherwise create new index
    try:
        # Create Pinecone client and ensure index exists
        pc = create_pinecone_index_if_not_exists(
            api_key=pinecone_api_key,
            index_name=pinecone_index_name,
            dimension=1536  # Standard dimension for text-embedding-3-small
        )
        
        if pc is None:
            st.error("❌ Failed to initialize Pinecone")
            return None
        
        # Create LangChain vector store from existing index
        vectordb = PineconeVectorStore(
            index=pc,
            embedding=embeddings,
            index_name=pinecone_index_name,
        )
    except Exception as e:
        st.error(f"❌ Error connecting to Pinecone: {str(e)}")
        return None
    return vectordb

# -------------------------
# Document loading helpers
# -------------------------
def load_documents(uploaded_file_path: str, file_name: str) -> List[Document]:
    print(f"Loading document: {uploaded_file_path}")
    ext = os.path.splitext(uploaded_file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(uploaded_file_path).load()
    if ext in [".docx", ".doc"]:
        return Docx2txtLoader(uploaded_file_path).load()
    docs = TextLoader(uploaded_file_path, encoding="utf-8").load()
    for d in docs:
        d.metadata["source"] = os.path.basename(uploaded_file_path)
    return docs

def join_docs(docs: List[Any]) -> str:
    return "\n\n".join([d.page_content for d in docs])

# -------------------------
# Tickets storage helpers
# -------------------------
def load_tickets() -> List[Dict[str, Any]]:
    if not os.path.exists(TICKETS_FILE):
        return []
    try:
        with open(TICKETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_tickets(tickets: List[Dict[str, Any]]):
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(tickets, f, ensure_ascii=False, indent=2)

@tool(response_format="content_and_artifact")
def add_ticket(title: str, description: str, priority: str = "medium") -> Dict[str, Any]:
    """Submit a support ticket for further assistance if user provided question if agent cannot answer user's question or the provided content is irrelevant, when the user provides enough contact details including name and email."""
    tickets = load_tickets()
    ticket = {
        "id": str(uuid.uuid4()),
        "title": title.strip(),
        "description": description.strip(),
        "priority": priority,
        "status": "open",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    tickets.append(ticket)
    save_tickets(tickets)
    return (json.dumps(ticket), ticket)

def update_ticket_status(ticket_id: str, status: str):
    tickets = load_tickets()
    for t in tickets:
        if t["id"] == ticket_id:
            t["status"] = status
            break
    save_tickets(tickets)
    
# -------------------------
# Prompt builders
# -------------------------

def build_summarize_user_guide(text: str, summary_style: SummaryStyles = 'concise', language: Languages = 'eng') -> str:
    """Generate user guide summary prompt"""
    # Customize prompt based on style
    style_prompts = {
        "concise": "Summarize the following user guide documentation into concise bullet points covering key features, instructions, and important information:",
        "detailed": "Provide a detailed summary of the following user guide documentation, including all major sections, procedures, and important details:",
        "action-focused": "Extract and organize the key procedures, step-by-step instructions, and important guidelines from the following user guide documentation:"
    }
    
    base_prompt = style_prompts.get(summary_style, style_prompts['concise'])
    language_instruction = f"Please response in {LANGUAGE_MAP.get(language, 'eng')}." if language != "eng" else ""
    
    if language_instruction:
        prompt = f"{base_prompt} {language_instruction}\n\n{text}"
    else:
        prompt = f"{base_prompt}\n\n{text}"
    
    return prompt

def build_qna_prompt(question: str, context: str, language: Languages = 'eng') -> tuple[str, Dict[str, any]]:
    """Build prompt guides with function calling with Chain of Thought reasoning"""
    # Get the previous question BEFORE updating it
    previous_question = st.session_state.get('previous_question', '')

    language_instruction = f"Please response in {LANGUAGE_MAP.get(language, 'eng')}." if language != "eng" else ""

    # System prompt with Chain of Thought reasoning (from merege-code.py)
    system_prompt = f"""You have access to a tool that retrieves context from User Guide documentation.
Use the tool to help answer user queries.{language_instruction}
Follow these steps for every answer:
1. Always use the tool to retrieve the context based on the user question. Use the results from the tool to reason about the best possible answer.
2. Do not make up answers, only use the retrieved context.
3. If you you cannot answer, or the information is not found, ask if user wants to contact support, and also ask their name and email if unknown, **do not make up an example user and email, ask user to provide if missing**.
4. If the user wants to contact support, ask for their name and email if unknown, then use the `create_ticket` function to create a support ticket.
"""

    # Few-shot examples (from merege-code.py)
    few_shot_examples = [
        # 1) Reset device (EN) – direct + steps
        {
            "context": (
                "User Guide Summary:\n"
                "- Soft reset: Settings > System > Reset.\n"
                "- Hard reset: giữ nút Reset ~10 giây đến khi LED đỏ nhấp nháy.\n\n"
                "Original Document (truncated):\n"
                "Hardware reset requires pressing and holding the recessed button for 8–12 seconds.\n"
            ),
            "question": "How do I perform a hard reset?",
            "final_answer": (
                "Use the physical reset button:\n"
                "- Power on the device.\n"
                "- Press and hold the recessed **Reset** button for ~10 seconds until the LED blinks red.\n"
                "- Release to complete the hard reset.\n"
                "If the LED never blinks, check the Hardware Reset section for model-specific notes."
            ),
        },

        # 2) Export PDF (EN) – feature gating + alternatives
        {
            "context": (
                "User Guide Summary:\n"
                "- Export: TXT/HTML available by default.\n"
                "- PDF export requires Advanced Preview to be enabled.\n"
            ),
            "question": "Can I export the summary to PDF?",
            "final_answer": (
                "Yes, if **Advanced Preview** is enabled:\n"
                "- Open **Advanced Preview** → **Export** → **PDF**.\n"
                "- If it's disabled, either enable Advanced Preview or export **TXT/HTML** instead."
            ),
        },

        # 3) Missing data (EN) – Unknown/Not in guide
        {
            "context": (
                "User Guide Summary:\n"
                "- The app supports English, Spanish, and Japanese UI.\n"
                "- Auto language detection applies to the chatbot only.\n"
            ),
            "question": "Does the UI support German?\n",
            "final_answer": (
                "Not in guide. The summary lists **English, Spanish, Japanese** only for the UI. "
                "German isn't mentioned—please check the Localization section or release notes."
            ),
        },

        # 4) Multi-language auto (ES) – respond in question language
        {
            "context": (
                "User Guide Summary:\n"
                "- Las copias de seguridad automáticas se ejecutan a las 02:00.\n"
                "- Se pueden cambiar desde Settings > Backup Schedule.\n"
            ),
            "question": "¿Cómo cambio la hora de la copia de seguridad?\n",
            "final_answer": (
                "Ve a **Settings → Backup Schedule** y cambia la hora predeterminada (02:00) a la que prefieras. "
                "Guarda los cambios para aplicarlos en la próxima ejecución."
            ),
        },

        # 5) Troubleshooting (VI) – actions + conditions
        {
            "context": (
                "User Guide Summary:\n"
                "- Đăng nhập yêu cầu email đã xác thực.\n"
                "- Sau 5 lần sai mật khẩu, tài khoản bị tạm khóa 15 phút.\n"
                "- Có mục Reset password qua email.\n"
            ),
            "question": "Không đăng nhập được thì làm sao?\n",
            "final_answer": (
                "Thử theo thứ tự:\n"
                "- Kiểm tra bạn đã xác thực email chưa.\n"
                "- Nếu quên mật khẩu: dùng **Reset password** để đặt lại.\n"
                "- Nếu nhập sai >5 lần: chờ 15 phút rồi thử lại.\n"
                "- Vẫn lỗi: xem mục **Troubleshooting → Login** để kiểm tra mã lỗi cụ thể."
            ),
        },
    ]

    
    # Create context with document information
    context = f"Relevant Information from User Guide:\n{context}"
    

    # Build messages with few-shot examples
    messages = []

    # Add few-shot examples
    for ex in few_shot_examples:
        ex_prompt = f"""Based on the following user guide information, please answer the user's question accurately and concisely.

{ex["context"]}

User Question: {ex["question"]}

Answer:"""
        ex_response = {
            "reasoning": "Example reasoning omitted.",
            "answer": ex["final_answer"]
        }
        messages.append({"role": "user", "content": ex_prompt})
        messages.append({"role": "assistant", "content": json.dumps(ex_response, ensure_ascii=False)})

    return system_prompt, messages


def display_assistant_response(answer, reasoning=None, tool_call=None, language='eng'):
    with st.chat_message("assistant"):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.markdown(answer)
            if reasoning:
                # show reasoning in an expander (toggle controlled by session state)
                with st.expander("Show reasoning", expanded=False):
                    st.markdown(reasoning)
            if tool_call:
                # show tool_call in an expander (toggle controlled by session state)
                with st.expander("Show tool_call", expanded=False):
                    st.code(tool_call)
        with col2:
            key=f"tts_{hash(answer)}"
            if key not in st.session_state.tts_playing:
                st.session_state.tts_playing[key] = False
            tts_btn = st.button("🔊", key=key)
            if not st.session_state.tts_playing[key]:
                if tts_btn:
                    st.session_state.tts_playing[key] = True
                    st.session_state.tts_last_clicked = key
                    st.rerun()  # ensure immediate refresh to show player
            else:
                with st.spinner(""):
                    audio_data, sr = speak(answer, lang=language)
                create_audio_player(audio_data, sr, autoplay=True)
                st.session_state.tts_last_clicked = None
                st.session_state.tts_playing[key] = False

            if tts_btn:
                with st.spinner(""):
                    audio_data, sample_rate = speak(answer, lang='eng')
                    create_audio_player(audio_data, sample_rate)

# -------------------------
# Chains
# -------------------------
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vectordb.similarity_search(query, k=global_top_k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context, add_ticket]

class ReasoningAnswer(BaseModel):
    """An answer with details of language and the tool call made."""
    answer: str = Field(..., description="The pretty markdown format of the answer")
    reasoning: str = Field(..., description="The reasoning or tool call leading to the answer")
    language: str = Field(..., description="The language of the answer, only 'vie', 'eng', 'kor', 'fra', or 'deu'")

def build_rag(retriever, llm: BaseChatModel):
    system_prompt, few_shot = build_qna_prompt(question='', language='eng', context='')
    # llm = llm.with_structured_output(ReasoningAnswer)
    agent = create_agent(
        llm, 
        tools, 
        system_prompt=system_prompt,
        response_format=ToolStrategy(ReasoningAnswer),
    )
    # chain = (
    #     {
    #         "context": retriever | (lambda docs: join_docs(docs)),
    #         "question": RunnablePassthrough(),
    #         "lang": RunnablePassthrough(),
    #     }
    #     | qa_prompt
    #     | llm
    #     | StrOutputParser()
    # )
    return agent, few_shot

def build_summarizer(llm: BaseChatModel):
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        ("human", "Summarize the document based on this content:\n\n{content}")
    ])
    return (summary_prompt | llm | StrOutputParser())


def get_text_splitter(mode: str) -> TextSplitter:
    chunk_size = 4000
    overlap = 400
    if mode == "character_text_splitter":
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    
    return SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=chunk_size,
        chunk_overlap=overlap,
        model_name='intfloat/e5-base-v2',
        add_start_index=True,
    )


# -------------------------
# Optional: OpenAI TTS
# -------------------------
def tts_synthesize(text: str, voice: str = "alloy") -> bytes:
    if _OpenAI is None:
        print("TTS Error: OpenAI library not available")
        return b""
    
    # Check if we have OpenAI API key for TTS
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("TTS Error: OPENAI_API_KEY not set. Azure OpenAI doesn't support TTS endpoint. You need a standard OpenAI API key.")
        return b""
    
    try:
        client = _OpenAI(api_key=openai_key)
        resp = client.audio.speech.create(
            model=os.getenv("OPENAI_TTS_MODEL", "tts-1"),
            voice=voice,
            input=text,
        )
        audio_bytes = resp if isinstance(resp, (bytes, bytearray)) else resp.read()
        return audio_bytes
    except Exception as e:
        print(f"TTS Error: {e}")
        import traceback
        traceback.print_exc()
        return b""
    
# ---------------------------
# CSS helpers for Streamlit
# ---------------------------
st.markdown(
    f'''
    <style>
        .block-container {{
            padding: {1}rem !important;
        }}
        .stChatMessageContainer {{
            overflow-y: auto;
            scroll-behavior: smooth;
        }}
        .stChatMessageContainer > div:last-child {{
            margin-bottom: 80px;
        }}
    </style>
    ''',unsafe_allow_html=True)

# Define available functions for OpenAI
function_definitions = [{
    "type": "function",
    "function": {
        "name": "create_support_ticket",
        "description": "Submit a support ticket for further assistance if user provided question cannot be answered, when the user provides enough contact details including name and email",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The customer's full name"
                },
                "email": {
                    "type": "string",
                    "description": "The customer's email address"
                },
                "issue_description": {
                    "type": "string",
                    "description": "The question or issue the person is experiencing, getting from the 'Conversation history' where the question you cannot answered, or the current user requested support question"
                }
            },
            "required": ["name", "email", "issue_description"]
        }
    }
}]


# -------------------------
# UI Sections (Tabs-based)
# -------------------------
@st.dialog("Please confirm")
def confirm_choose(message:str = "Are you sure?"):
    st.write(message)
    if st.button("Yes"):
        st.session_state.dialog_choose = True
        st.rerun()
    if st.button("No"):
        st.session_state.dialog_choose = False
        st.rerun()


def sidebar_config():
    st.header("⚙️ Settings")
    st.session_state["temperature"] = st.slider("Temperature", 0.0, 1.2, 0.2, 0.1)
    lang = st.selectbox("Response Language", LANG_OPTIONS, index=0)
    style = st.selectbox("Summary Style", SUMMARY_STYLES, index=0)
    top_k = st.slider("Top-K (retriever)", 2, 15, 5)
    voice = st.selectbox("TTS Voice (OpenAI)", ["alloy", "verse", "aria", "sage"], index=0)
    # Vector Database Configuration
    st.sidebar.header("🗄️ Vector Database Configuration")
    
    # Default values - Load from environment variables if available
    pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "fomo-db")
    
    # Pinecone configuration
    pinecone_api_key = st.sidebar.text_input(
        "Pinecone API Key",
        value=pinecone_api_key,
        type="password",
        help="Your Pinecone API key (or set PINECONE_API_KEY in .env)"
    )
    pinecone_index_name = st.sidebar.text_input(
        "Index Name",
        value=pinecone_index_name,
        help="Your Pinecone index name (or set PINECONE_INDEX_NAME in .env)"
    )
    update_btn = st.button("Update Config", key="save_config")
    
    if pinecone_api_key and pinecone_index_name:
        st.sidebar.info(f"🌲 Pinecone: {pinecone_index_name}")
    else:
        st.sidebar.warning("⚠️ Please enter Pinecone API key and index name")
    
    return lang, style, top_k, voice, pinecone_api_key, pinecone_index_name, update_btn

def tab_documents(vectordb: PineconeVectorStore):
    st.subheader("🗂️ Documents")
    col_up, col_clear = st.columns([3,1])
    
    with col_up:
        uploaded = st.file_uploader("Upload (multiple files supported)", type=["txt","md","docx","doc","pdf"], accept_multiple_files=True)
    with col_clear:
        if st.button("🧹 Clear Index"):
            # remove pinecone index
            confirm_choose("Are you sure you want to clear the entire Pinecone index?")
            if st.session_state.get("dialog_choose", True):
                vectordb.delete(delete_all=True)
                st.success("Pinecone index removed.")
    splitter = get_text_splitter(mode="character_text_splitter")
    if uploaded:
        if st.button("⬇️ Import and Index Documents"):
            with st.spinner("Indexing documents..."):
                for file in uploaded:
                    path = os.path.join("./.tmp", file.name)
                    with open(path, "wb") as f:
                        f.write(file.read())
                    try:
                        docs = load_documents(path, file.name)
                        chunks = splitter.split_documents(docs)
                        if chunks:
                            print(f"Indexing {len(chunks)} chunks from {file.name}")
                            vectordb.add_documents(chunks)
                        st.success(f"Indexed: {file.name} ({len(chunks)} chunks)")
                    except Exception as e:
                        # print stack trace
                        traceback.print_exc()
                        st.error(f"Exception processing {file.name}: {e}")
    
    # Display index statistics
    try:
        if vectordb and vectordb.index:
            with st.spinner("Getting index stats..."):
                # Get Pinecone index stats
                index_stats = vectordb.index.describe_index_stats()
                total_vectors = index_stats.get('total_vector_count', 0)
                st.info(f"🌲 Pinecone index: • Total vectors: {total_vectors}")
        else:
            st.warning("⚠️ Vector database not available")
    except Exception as e:
        traceback.print_exc()
        st.warning(f"⚠️ Could not retrieve index stats: {str(e)}")
    
    return vectordb

def tab_summary(llm: BaseChatModel, vectordb: PineconeVectorStore, lang: str, style: str, voice: str):
    st.subheader("✂️ Summarize")
    
    # Initialize session state for summary
    if "summary_output" not in st.session_state:
        st.session_state.summary_output = None
    if "summary_content" not in st.session_state:
        st.session_state.summary_content = None
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    if st.button("Generate Summary"):
        guideline = build_summarize_user_guide("llm", style, lang)
        ctx_docs = retriever.invoke("overview, executive summary, main points, table of contents")
        content = join_docs(ctx_docs)[:15000]
        chain = build_summarizer(llm)
        with st.spinner("Summarizing..."):
            out = chain.invoke({"content": content, "system_prompt": guideline})
        st.session_state.summary_output = out
        st.session_state.summary_content = content
        log_query("summary", content, out, {"lang": lang, "style": style})
    
    # Display summary if it exists
    if st.session_state.summary_output:
        st.markdown(st.session_state.summary_output)
        with st.expander("🔍 Reasoning / Context (prompt vars)"):
            st.code(json.dumps({"style": style, "lang": lang, "content_sample": st.session_state.summary_content[:600] if st.session_state.summary_content else ""}, ensure_ascii=False, indent=2))

        if st.toggle("🔊 Đọc to (TTS)", value=False, key="summary_tts_toggle"):
            with st.spinner("Generating audio..."):
                # Use local TTS model (same as Q&A tab)
                audio_data, sample_rate = speak(st.session_state.summary_output, lang=lang)
                if audio_data is not None and len(audio_data) > 0:
                    print('🔊🔊🔊🔊🔊 Audio generated successfully')
                    create_audio_player(audio_data, sample_rate, autoplay=True)
                    st.success("✅ Audio generated!")
                else:
                    print("❌ TTS not available or configuration error.")
                    st.error("TTS not available or configuration error. Check terminal logs.")

def tab_qa(llm: BaseChatModel, retriever: BaseRetriever, lang: str, voice: str, top_k: int):
    st.subheader("❓ RAG - Q&A")
    # Chat interface
    st.info("💡 **Smart Language Detection**: Ask questions in any supported language, and I'll respond in the same language! The configured language above is used for summaries only.")
    chat_container = st.container()

    # Display chat history
    with chat_container:
        if len(st.session_state.chat_history) > 0:
            st.markdown("### 💬 Conversation History")
            # Display chat messages from history on app rerun
            for i, message in enumerate(st.session_state.chat_history):
                if (message["role"] == "assistant"):
                    reasoning = message.get("reasoning")
                    tool_call = message.get("tool_call")
                    answer = message["content"]
                    language = message.get("language", 'eng')
                    display_assistant_response(answer, reasoning, tool_call, language)
                else:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

        else:
            # Suggested questions
            if len(st.session_state.chat_history) == 0:
                st.markdown("### 💡 Suggested Questions (Multi-Language)")
                suggested_questions = [
                    "What are the main contents described in this guide?",
                    "Nội dung chính của tài liệu này là gì?",
                    "이 가이드에 설명된 주요 내용은 무엇입니까?",
                    "Quels sont les principaux contenus décrits dans ce guide?",
                    "Welche wichtigen Konfigurationsschritte gibt es?"
                ]

                cols = st.columns(2)
                for i, suggestion in enumerate(suggested_questions):
                    with cols[i % 2]:
                        if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                            # Set the question and trigger ask
                            # with st.spinner("Querying..."):
                            #     response = agent.invoke({"messages": [{"role": "user", "content": question}]})
                            #     answer = response["messages"][-1].content
                            #     print(response)
                            # st.markdown(answer)
                            # st.session_state.chat_history.add_user_message(question)
                            # st.session_state.chat_history.add_ai_message(answer)
                            st.session_state.chat_history.append({"role": "user", "content": suggestion})
                            with st.chat_message("user"):
                                st.markdown(suggestion)
                            with st.spinner("🤔 Thinking..."):
                                agent, few_shot = build_rag(retriever, llm)
                                history = st.session_state.chat_history 
                                # Use auto-detection for suggested questions too
                                response = agent.invoke({"messages": [*history, {"role": "user", "content": suggestion}]})
                                structured = response["structured_response"]
                                print(structured)
                                print("="*80)
                                print(response["messages"][-2])
                                print("="*80)
                                print(response["messages"][-3])
                                print("="*80)
                                answer = structured.answer
                                reasoning = structured.reasoning
                                language= structured.language
                                tool_call = response["messages"][-3].content
                                display_assistant_response(answer, reasoning, tool_call, language)
                                st.session_state.chat_history.append({"role": "assistant", "content": answer, "reasoning": reasoning, "tool_call": tool_call, "language": language})
                                st.rerun()

    # Process question
    question = st.chat_input("e.g., How do I configure this feature? | ¿Cómo configuro esta característica? | Comment configurer cette fonctionnalité?")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(question)
            with st.spinner("🤔 Thinking..."):
                # Use auto-detection for chatbot, but keep manual language for summaries
                agent, few_shot = build_rag(retriever, llm)
                history = st.session_state.chat_history 
                # Use auto-detection for suggested questions too
                response = agent.invoke({"messages": [*history, {"role": "user", "content": question}]})
                structured = response["structured_response"]
                print(structured)
                print("="*80)
                print(response["messages"][-2])
                print("="*80)
                print(response["messages"][-3])
                print("="*80)
                answer = structured.answer
                reasoning = structured.reasoning
                language= structured.language
                tool_call = response["messages"][-3].content
                display_assistant_response(answer, reasoning, tool_call, language)
                st.session_state.chat_history.append({"role": "assistant", "content": answer, "reasoning": reasoning, "tool_call": tool_call, "language": language})
            # Clear input and rerun to show new message
            # st.rerun()
    
    # Chat controls
    col_clear, col_export = st.columns(2)
    
    with col_clear:
        if st.button("🗑️ Clear Chat") and st.session_state.chat_history:
            st.session_state.chat_history = []
            st.rerun()
    
    with col_export:
        if st.session_state.chat_history:
            # Export chat history
            chat_export = "User Guide Q&A Session\n" + "="*50 + "\n\n"
            chat_export += f"User Guide Summary:\n{st.session_state.summary}\n\n"
            chat_export += "Conversation:\n" + "-"*30 + "\n"
            
            for i, message in enumerate(st.session_state.chat_history, 1):
                if message['role'] == 'user':
                    chat_export += "\n" + "-"*80 + "\n"
                chat_export += f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}\n"

            st.download_button(
                label="💾 Export Chat",
                data=chat_export.encode('utf-8'),
                file_name="user_guide_qa_session.txt",
                mime="text/plain"
            )


def tab_tickets():
    st.subheader("🎫 Tickets")
    col_export, _ = st.columns([1,3])
    with col_export:
        if st.button("⬇️ Export JSON"):
            tickets = load_tickets()
            b = io.BytesIO(json.dumps(tickets, ensure_ascii=False, indent=2).encode("utf-8"))
            st.download_button("Download tickets.json", b, file_name="tickets.json", mime="application/json")

    st.markdown("---")
    st.markdown("**Customer Support Requests:**")
    tickets = load_tickets()
    if tickets:
        for t in sorted(tickets, key=lambda x: (x["status"] != "open", x["created_at"])):
            with st.expander(f"[{t['status'].upper()}] {t['title']}  •  {t['priority']}  •  {t['id'][:8]}"):
                st.write(t["description"])
                st.caption(f"Created: {t['created_at']}")
                new_status = st.selectbox("Update status", ["open","in_progress","done","closed"],
                                          index=["open","in_progress","done","closed"].index(t["status"]), key=f"status_{t['id']}")
                if st.button("Save", key=f"save_{t['id']}"):
                    update_ticket_status(t["id"], new_status)
                    st.success("Status updated.")
    else:
        st.info("Empty ticket list.")
    

# -------------------------
# Main app
# -------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Streamlit + LangChain • RAG • Summarize • Chat • TTS • Tickets • Logs • Multi-file Upload • Clear Index")
llm = None
embeddings = None
vectordb = None
global_top_k = 5

def initialize_app(pinecone_api_key=None, pinecone_index_name=None):
    global llm, embeddings, vectordb
    if st.session_state.first_load:
        print("Initializing models and components...")
        with st.spinner("🚀 Initializing models and components..."):
            llm = build_llm()
            embeddings = build_embeddings()
            initialize_tts()
            # build_transformers()
            if not pinecone_api_key or not pinecone_index_name:
                pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
                pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "fomo-db")
            vectordb = initialize_vector_store(embeddings, pinecone_index_name, pinecone_api_key)
            st.session_state["vectordb"] = vectordb
            st.session_state["llm"] = llm
            st.session_state["embeddings"] = embeddings
        st.session_state.first_load = False
    else:
        llm = st.session_state.get("llm")
        embeddings = st.session_state.get("embeddings")
        vectordb = st.session_state.get("vectordb")

def main():
    global global_top_k
    with st.sidebar:
        lang, style, top_k, voice, pinecone_api_key, pinecone_index_name, update_btn = sidebar_config()
        global_top_k = top_k
        st.markdown("---")
        st.markdown("**Model info**")
        st.write("Azure:" if os.getenv("AZURE_OPENAI_API_KEY") else "OpenAI:")
        st.write(os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        if update_btn:
            st.session_state.first_load = True
            initialize_app(pinecone_api_key, pinecone_index_name)

    initialize_app()
    vectordb = st.session_state.get("vectordb")
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k}) if vectordb else None
    st.session_state["retriever"] = retriever

    tabs = st.tabs(["📚 Document Indexing", "🔍 User Guide Summary", "💬 Q&A Chatbot", "🎫 Support Questions"])
    
    with tabs[0]:
        tab_documents(vectordb)

    with tabs[1]:
        llm = st.session_state.get("llm")
        vectordb = st.session_state.get("vectordb")
        tab_summary(llm, vectordb, lang, style, voice)

    with tabs[2]:
        llm = st.session_state.get("llm")
        vectordb = st.session_state.get("vectordb")
        retriever = st.session_state.get("retriever")
        tab_qa(llm, retriever, lang, voice, top_k)

    with tabs[3]:
        tab_tickets()


if __name__ == "__main__":
    main()