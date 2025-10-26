import streamlit as st
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
from datetime import datetime
import chromadb
import tiktoken

from audio_player import create_audio_player
from tts import load_tts, speak
from rag import ChromaVectorStore, ChromaConfig

# Load environment variables from .env file
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="User Guide Summarization",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def scroll_to_bottom():
    st.markdown(
        """
        <script>
        var chatContainer = window.parent.document.querySelector('.stChatMessageContainer');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        } else {
            window.scrollTo(0, document.body.scrollHeight);
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

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

# Language-specific instructions
language_instructions = {
    "English": "",
    "Spanish": " Please respond in Spanish.",
    "French": " Please respond in French.",
    "German": " Please respond in German.",
    "Italian": " Please respond in Italian.",
    "Portuguese": " Please respond in Portuguese.",
    "Japanese": " Please respond in Japanese.",
    "Chinese (Simplified)": " Please respond in Simplified Chinese.",
    "Korean": " Please respond in Korean.",
    "Arabic": " Please respond in Arabic."
}

# Initialize session state
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'guide_context' not in st.session_state:
    st.session_state.guide_context = ""
if 'previous_question' not in st.session_state:
    st.session_state.previous_question = ""
if 'support_tickets' not in st.session_state:
    st.session_state.support_tickets = []
if "tts_playing" not in st.session_state:
    st.session_state.tts_playing = {}
if "tts_last_clicked" not in st.session_state:
    st.session_state.tts_last_clicked = None
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'chroma_collection' not in st.session_state:
    st.session_state.chroma_collection = None
if 'chromadb_config' not in st.session_state:
    st.session_state.chromadb_config = None
if 'chroma_vector_store' not in st.session_state:
    st.session_state.chroma_vector_store = None

def initialize_client():
    """Initialize Azure OpenAI client with error handling"""
    try:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        if not api_key or not endpoint:
            st.error("⚠️ Azure OpenAI credentials not found. Please check your .env file.")
            st.info("Required environment variables: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT")
            return None
            
        client = AzureOpenAI(
            api_version="2024-07-01-preview",
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Azure OpenAI client: {str(e)}")
        return None

def initialize_chromadb(use_cloud=False, api_key="", tenant="", database=""):
    """Initialize ChromaDB client and collection for semantic search
    
    Args:
        use_cloud: If True, use CloudClient with authentication
        api_key: API key for CloudClient authentication
        tenant: Tenant ID for CloudClient
        database: Database name for CloudClient
    """
    try:
        if use_cloud:
            # Connect to ChromaDB Cloud with authentication
            if not api_key or not tenant or not database:
                print("⚠️ Cloud mode requires API key, tenant, and database")
                return None, None, None
            
            client = chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database=database
            )
            print(f"☁️ Connecting to ChromaDB Cloud (tenant: {tenant}, database: {database})...")
            
            # Test connection
            try:
                client.heartbeat()
            except Exception as e:
                print(f"⚠️ ChromaDB connection test failed: {str(e)}")
                print(f"💡 Check your cloud credentials and network connection")
                return None, None, None
            
            # Create or get collection with auto-embedding
            collection = client.get_or_create_collection(
                name="user_guide_docs",
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"✅ ChromaDB Cloud connected: {tenant}/{database}")
            return client, collection, None
            
        else:
            # Use local ChromaVectorStore from rag module
            config = ChromaConfig(
                persist_dir=".chroma",
                collection_name="user_guide_docs",
                metadata={"hnsw:space": "cosine"}
            )
            vector_store = ChromaVectorStore(config)
            print(f"✅ ChromaDB initialized locally with rag module: .chroma")
            return vector_store.client, vector_store.collection, vector_store
        
    except Exception as e:
        print(f"❌ ChromaDB initialization failed: {str(e)}")
        if use_cloud:
            print(f"💡 Check your cloud credentials (API key, tenant, database) and network connection")
        else:
            print(f"💡 Check that fastembed is installed: pip install fastembed")
        return None, None, None

def chunk_text(text, chunk_size=500, overlap=50, encoding_name="cl100k_base"):
    """Split text into token-aware overlapping chunks"""
    try:
        # Get tokenizer
        encoding = tiktoken.get_encoding(encoding_name)
        
        # Encode the text
        tokens = encoding.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Get chunk
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - overlap
            
            # Prevent infinite loop on last chunk
            if end >= len(tokens):
                break
        
        print(f"📄 Split document into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"⚠️ Error chunking text: {str(e)}")
        # Fallback: simple word-based chunking
        words = text.split()
        chunk_word_size = chunk_size * 3  # Rough approximation
        chunks = []
        for i in range(0, len(words), chunk_word_size - overlap * 3):
            chunk = ' '.join(words[i:i + chunk_word_size])
            chunks.append(chunk)
        return chunks

def embed_and_store_document(collection, document_text, document_id="user_guide", vector_store=None):
    """Chunk and store document in ChromaDB with auto-embeddings"""
    try:
        if not collection:
            print("⚠️ No ChromaDB collection available")
            return False
        
        # Clear existing documents for this ID
        try:
            collection.delete(where={"doc_id": document_id})
        except:
            pass  # Collection might be empty
        
        # Chunk the document
        chunks = chunk_text(document_text, chunk_size=500, overlap=50)
        
        # Use ChromaVectorStore if available (local mode), otherwise use direct collection (cloud mode)
        if vector_store:
            # Prepare documents for ChromaVectorStore.upsert_texts
            docs = [
                {
                    "id": f"{document_id}_chunk_{i}",
                    "content": chunk,
                    "metadata": {"doc_id": document_id, "chunk_index": i}
                }
                for i, chunk in enumerate(chunks)
            ]
            vector_store.upsert_texts(docs, source="user_guide")
        else:
            # Cloud mode: use direct collection API
            ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"doc_id": document_id, "chunk_index": i} for i in range(len(chunks))]
            collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
        
        print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
        return True
        
    except Exception as e:
        print(f"❌ Error storing document in ChromaDB: {str(e)}")
        return False

def query_chromadb(collection, query_text, n_results=3, vector_store=None):
    """Semantic search to retrieve relevant document chunks"""
    try:
        if not collection:
            print("⚠️ No ChromaDB collection available")
            return []
        
        # Use ChromaVectorStore if available (local mode), otherwise use direct collection (cloud mode)
        if vector_store:
            # Use ChromaVectorStore.similarity_search
            results = vector_store.similarity_search(query_text, k=n_results)
            documents = [r["content"] for r in results]
        else:
            # Cloud mode: use direct collection API
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            documents = results['documents'][0] if results['documents'] else []
        
        print(f"🔍 Retrieved {len(documents)} relevant chunks from ChromaDB")
        return documents
        
    except Exception as e:
        print(f"❌ Error querying ChromaDB: {str(e)}")
        return []

def create_support_ticket(name, email, question, previous_question=""):
    """Create a customer support ticket"""
    try:
        # Create ticket object
        ticket = {
            "id": f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "name": name,
            "email": email,
            "question": question,
            "previous_question": previous_question,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # Log for debugging
        print(f"📋 Support ticket created: {ticket['id']}")
        print(f"   Name: {name}")
        print(f"   Email: {email}")
        print(f"   Question: {question}")
        if previous_question:
            print(f"   Context: {previous_question}")
        
        return {
            "success": True,
            "ticket": ticket,
            "ticket_id": ticket["id"],
            "message": f"Support ticket {ticket['id']} has been created successfully. Our team will contact you at {email} within 24-48 hours."
        }
        
    except Exception as e:
        print(f"❌ Error creating support ticket: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to create support ticket: {str(e)}"
        }


def summarize_user_guide(client, text, summary_style="concise", max_tokens=300, temperature=0.3, language="English", model="gpt-4o-mini"):
    """Generate user guide summary using Azure OpenAI"""
    try:
        if not text.strip():
            return "⚠️ No content to summarize. Please provide a user guide document."
        
        # Language-specific instructions
        language_instructions = {
            "English": "",
            "Spanish": "Respond in Spanish.",
            "French": "Respond in French.",
            "German": "Respond in German.",
            "Italian": "Respond in Italian.",
            "Portuguese": "Respond in Portuguese.",
            "Japanese": "Respond in Japanese.",
            "Chinese (Simplified)": "Respond in Simplified Chinese.",
            "Korean": "Respond in Korean.",
            "Arabic": "Respond in Arabic."
        }
        
        # Customize prompt based on style
        style_prompts = {
            "concise": "Summarize the following user guide documentation into concise bullet points covering key features, instructions, and important information:",
            "detailed": "Provide a detailed summary of the following user guide documentation, including all major sections, procedures, and important details:",
            "action-focused": "Extract and organize the key procedures, step-by-step instructions, and important guidelines from the following user guide documentation:"
        }
        
        base_prompt = style_prompts.get(summary_style, style_prompts['concise'])
        language_instruction = language_instructions.get(language, "")
        
        if language_instruction:
            prompt = f"{base_prompt} {language_instruction}\n\n{text}"
        else:
            prompt = f"{base_prompt}\n\n{text}"
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        summary = response.choices[0].message.content
        
        # Store document in ChromaDB for semantic search
        if st.session_state.chroma_collection:
            print("📦 Storing document in ChromaDB...")
            embed_and_store_document(
                st.session_state.chroma_collection,
                text,
                document_id="user_guide",
                vector_store=st.session_state.chroma_vector_store
            )
        
        return summary
        
    except Exception as e:
        return f"❌ Error generating summary: {str(e)}"

def answer_question(client, question, guide_summary, guide_document="", language="English", model="gpt-4o-mini"):
    """Answer questions about the user guide using native OpenAI function calling with Chain of Thought reasoning"""
    try:
        if not question.strip():
            return "⚠️ Please ask a question about the user guide."

        if not guide_summary.strip():
            return "⚠️ No guide summary available. Please generate a summary first."

        # Get the previous question BEFORE updating it
        previous_question = st.session_state.get('previous_question', '')

        language_instruction = language_instructions.get(language, "")

        # System prompt with Chain of Thought reasoning (from merege-code.py)
        system_prompt = f"""You are a reasoning AI assistant.{language_instruction}
Follow these steps for every answer:
1. Analyze the question carefully, include subject, context, relationships, and any relevant details.
2. Think step-by-step.
3. Produce a clear, final answer if possible.

For each question, respond as JSON:
{{
  "reasoning": "step-by-step explanation based on the guideline above",
  "answer": "final concise answer",
  "confidence": "confidence level from 0.0 to 1.0",
}}

IMPORTANT: 
- If you you cannot answer, or the information is not found, ask if user wants to contact support, and also ask their name and email if unknown, **do not make up an example user and email, ask user to provide if missing**.
- If the user wants to contact support, ask for their name and email if unknown, then use the create_support_ticket function to create a support ticket.
- When you decide to call tool, also include the JSON response as specified above."""

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

        # Try to get relevant context from ChromaDB first
        relevant_chunks = []
        if st.session_state.chroma_collection and guide_document:
            relevant_chunks = query_chromadb(
                st.session_state.chroma_collection,
                question,
                n_results=3,
                vector_store=st.session_state.chroma_vector_store
            )
        
        # Create context with document information
        context = f"User Guide Summary:\n{guide_summary}"
        
        # Use ChromaDB results if available, otherwise fall back to full document
        if relevant_chunks:
            context += "\n\nRelevant Information from User Guide:\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                context += f"\n[Section {i}]\n{chunk}\n"
            print("Chroma context 🤓")
            print(context)
        elif guide_document:
            context += (
                f"\n\nOriginal Document:\n{guide_document[:2000]}..."
                if len(guide_document) > 2000
                else f"\n\nOriginal Document:\n{guide_document}"
            )

        # Build messages with few-shot examples
        messages = [{"role": "system", "content": system_prompt}]

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

        # User prompt with context and question
        history = st.session_state.chat_history
        previous_questions = [f"{content['role']}: {content['content']}" for content in history[:-1]]
        user_prompt = f"""Based on the following user guide information, please answer the user's question accurately and concisely.{language_instruction}

{context}

Conversation history: 
{','.join(previous_questions)}

Current User Question: {question}

Answer:"""

        messages.append({"role": "user", "content": user_prompt})

        # Call API with function calling support
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=function_definitions,
            tool_choice='auto',
            max_tokens=1000,
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        # Check if the model wants to call a function
        response_message = response.choices[0].message
        print(response_message)
        
        if response_message.tool_calls:
            # The model wants to call a function
            function_name = response_message.tool_calls[0].function.name
            function_args = json.loads(response_message.tool_calls[0].function.arguments)
            
            print(f"🔧 Function call detected: {function_name}")
            print(f"📝 Arguments: {function_args}")
            
            if function_name == "create_support_ticket":
                # Call the actual ticket creation function
                result = create_support_ticket(
                    name=function_args.get("name"),
                    email=function_args.get("email"),
                    question=function_args.get("issue_description"),
                    previous_question=previous_question
                )
                function_called = {
                    "function_called": function_name,
                    "name": function_args.get("name"),
                    "email": function_args.get("email"),
                    "issue_description": function_args.get("issue_description"),
                    "ticket_id": result.get('ticket_id')
                }
                
                if result['success']:
                    # Save ticket to session state
                    st.session_state.support_tickets.append(result['ticket'])
                    
                    # Update previous_question for next interaction
                    st.session_state.previous_question = question
                    return (
                        f"""✅ {result['message']}

📧 **Contact Information Recorded:**
- Name: {function_args.get('name')}
- Email: {function_args.get('email')}

📋 **Issue Description:**
{function_args.get('issue_description')}

Our support team will review your query and respond within 24-48 hours.""", 
                        None, 
                        function_called
                    )
                else:
                    # Update previous_question for next interaction
                    st.session_state.previous_question = question
                    return (
                        f"❌ {result['message']}",
                        None, 
                        function_called
                    )
        
        # No function call, parse the JSON response
        try:
            response_json = json.loads(response.choices[0].message.content)
        except Exception:
            print("⚠️ Could not parse JSON response")
            response_json = {
                "reasoning": None,
                "answer": response_message.content
            }

        try:
            # Parse the JSON response
            response_json = json.loads(response.choices[0].message.content)

            # Log the reasoning internally (visible in terminal/logs)
            reasoning = response_json.get('reasoning', None)
            if reasoning:
                print(f"🧠 Reasoning: {reasoning}")

            # Get the answer from the JSON - try both 'answer' and 'response' fields
            answer = response_json.get('answer', response_message.content)

            # Check if information was not found and suggest support contact
            fallback_keywords = [
                "not found", "not in guide", "not in the guide", "not mentioned",
                "not available", "no information", "not explicitly found",
                "does not provide", "doesn't provide", "not provide",
                "does not contain", "doesn't contain", "not contain",
                "cannot find", "can't find", "unable to find",
                "not covered", "not included", "not described",
                "không tìm thấy", "không có", "không được đề cập"  # Vietnamese
            ]
            answer_lower = answer.lower()
            info_not_found = any(keyword in answer_lower for keyword in fallback_keywords)

            # Add confidence indicator if low confidence
            confidence = response_json.get('confidence', 1.0)
            if float(confidence) < 0.5:
                answer = f"⚠️ *Note: Lower confidence answer*\n\n{answer}"

            # Add sources if available
            sources = response_json.get('sources', [])
            if sources:
                answer += f"\n\n📚 **Sources:** {', '.join(sources)}"

            # Add note if not found in guide with support contact suggestion
            if info_not_found:
                answer += "\n\n📌 *Note: This information was not explicitly found in the user guide.*"
                answer += "\n\n💡 **Need more help?** Contact our support team by providing:"
                answer += "\n- Your **name**"
                answer += "\n- Your **email address**"
                answer += "\n\nExample: *\"I need help with [your issue]. My name is John Doe and email is john@example.com\"*"

            # Update previous_question for next interaction
            st.session_state.previous_question = question
            return answer, reasoning, None
        except (json.JSONDecodeError, KeyError) as e:
            # If it's not JSON or has unexpected structure, return the content as-is (fallback)
            print(f"⚠️ Could not parse JSON: {e}")
            # Update previous_question for next interaction
            st.session_state.previous_question = question
            return response_message.content, None, None
        
    except Exception as e:
        # Still update previous_question even on error
        st.session_state.previous_question = question
        print(f"❌ Error answering question: {str(e)}")
        return f"❌ Error answering question: {str(e)}", None, None

def detect_question_language(client, question, model="gpt-4o-mini"):
    """Detect the language of the user's question using AI"""
    try:
        if not question.strip():
            return "English"  # Default fallback
        
        # Simple language detection prompt
        detection_prompt = f"""Identify the language of the following text and respond with ONLY the language name in English (e.g., "Spanish", "French", "German", etc.). If you're not sure or it's mixed languages, respond with "English".

Text: "{question}"

Language:"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": detection_prompt}],
            max_tokens=10,
            temperature=0.1
        )
        
        detected_language = response.choices[0].message.content.strip()
        
        # Validate detected language against supported languages
        supported_languages = [
            "English", "Spanish", "French", "German", "Italian", 
            "Portuguese", "Japanese", "Chinese", "Korean", "Arabic"
        ]
        
        # Handle variations and ensure we return a supported language
        language_mapping = {
            "chinese (simplified)": "Chinese (Simplified)",
            "chinese": "Chinese (Simplified)",
            "simplified chinese": "Chinese (Simplified)",
            "mandarin": "Chinese (Simplified)"
        }
        
        detected_lower = detected_language.lower()
        if detected_lower in language_mapping:
            return language_mapping[detected_lower]
        
        # Check if detected language is in supported list (case insensitive)
        for lang in supported_languages:
            if lang.lower() == detected_lower:
                return lang
        
        # If not found, default to English
        return "English"
        
    except Exception as e:
        print(f"Language detection error: {e}")
        return "English"  # Fallback to English on error

def answer_question_auto_lang(client, question, guide_summary, guide_document="", fallback_language="English", model="gpt-4o-mini"):
    """Answer questions with automatic language detection from the question"""
    try:
        if not question.strip():
            return "⚠️ Please ask a question about the user guide."
        
        if not guide_summary.strip():
            return "⚠️ No guide summary available. Please generate a summary first."
        
        # Detect the language of the question
        detected_language = detect_question_language(client, question, model)
        
        # Use the original answer_question function with detected language
        return answer_question(client, question, guide_summary, guide_document, detected_language, model)
        
    except Exception as e:
        return f"❌ Error answering question: {str(e)}"


def display_assistant_response(answer, reasoning=None, tool_call=None):
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

            if not st.session_state.tts_playing[key]:
                if st.button("🔊", key=key):
                    st.session_state.tts_playing[key] = True
                    st.session_state.tts_last_clicked = key
                    st.rerun()  # ensure immediate refresh to show player
            else:
                with st.spinner(""):
                    audio_data, sr = speak(answer, lang='eng')
                autoplay = (st.session_state.tts_last_clicked == key)
                create_audio_player(audio_data, sr, autoplay=True)
                st.session_state.tts_last_clicked = None
                st.session_state.tts_playing[key] = False

            # if st.button("🔊", key=key):
            #     with st.spinner(""):
            #         audio_data, sample_rate = speak(answer, lang='eng')
            #         # st.audio(audio_data, format="audio/wav", sample_rate=sample_rate, autoplay=True)
            #         create_audio_player(audio_data, sample_rate)

def main():
    # Header
    st.title("🔍 User Guide Summarization")
    st.markdown("Transform your user guide documentation into organized, digestible summaries using AI")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model = st.sidebar.selectbox(
        "🤖 AI Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-35-turbo", "gpt-35-turbo-16k"],
        index=0,
        help="Choose the Azure OpenAI model for processing"
    )
    
    # Language selection
    language = st.sidebar.selectbox(
        "🌐 Output Language",
        ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Japanese", "Chinese (Simplified)", "Korean", "Arabic"],
        help="Choose the language for the summary and Q&A responses"
    )
    
    # Summary style options
    summary_style = st.sidebar.selectbox(
        "📝 Summary Style",
        ["concise", "detailed", "action-focused"],
        help="Choose the style of summary you want"
    )
    
    # ChromaDB Configuration
    st.sidebar.header("🗄️ ChromaDB Configuration")
    
    chromadb_mode = st.sidebar.radio(
        "Connection Mode",
        ["Local Storage", "Cloud (Managed)"],
        index=1,  # Default to Cloud (Managed)
        help="Choose how to connect to ChromaDB"
    )
    
    # Default values - Load from environment variables if available
    use_chromadb_cloud = False
    chromadb_api_key = os.getenv("CHROMADB_API_KEY", "")
    chromadb_tenant = os.getenv("CHROMADB_TENANT", "")
    chromadb_database = os.getenv("CHROMADB_DATABASE", "")
    
    if chromadb_mode == "Cloud (Managed)":
        use_chromadb_cloud = True
        chromadb_api_key = st.sidebar.text_input(
            "API Key",
            value=chromadb_api_key,
            type="password",
            help="Your ChromaDB Cloud API key (or set CHROMADB_API_KEY in .env)"
        )
        chromadb_tenant = st.sidebar.text_input(
            "Tenant ID",
            value=chromadb_tenant,
            help="Your ChromaDB Cloud tenant ID (or set CHROMADB_TENANT in .env)"
        )
        chromadb_database = st.sidebar.text_input(
            "Database Name",
            value=chromadb_database,
            help="Your ChromaDB Cloud database name (or set CHROMADB_DATABASE in .env)"
        )
        
        if chromadb_api_key and chromadb_tenant and chromadb_database:
            st.sidebar.info(f"☁️ Cloud: {chromadb_tenant}/{chromadb_database}")
        else:
            st.sidebar.warning("⚠️ Please enter all cloud credentials")
    
    else:  # Local Storage
        st.sidebar.info("💾 Using local ChromaDB storage at .chroma")
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        max_tokens = st.slider("Max Output Length", 150, 1000, 300, 50)
        temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.3, 0.1)
        
        # Model information
        st.info(f"**Selected Model:** {model}")
        model_info = {
            "gpt-4o-mini": "Fast and cost-effective, great for most tasks",
            "gpt-4o": "Advanced reasoning and complex tasks",
            "gpt-4": "High-quality responses with deep understanding",
            "gpt-35-turbo": "Balanced performance and speed",
            "gpt-35-turbo-16k": "Extended context length support"
        }
        st.caption(model_info.get(model, "Azure OpenAI model"))
    
    # Initialize client
    client = initialize_client()
    tts = load_tts('eng')
    
    # Initialize ChromaDB if not already initialized or if settings changed
    current_chromadb_config = (use_chromadb_cloud, chromadb_api_key, chromadb_tenant, chromadb_database)
    previous_chromadb_config = st.session_state.get('chromadb_config', None)
    
    if st.session_state.chroma_client is None or current_chromadb_config != previous_chromadb_config:
        with st.spinner("🔄 Initializing ChromaDB..."):
            chroma_client, chroma_collection, chroma_vector_store = initialize_chromadb(
                use_cloud=use_chromadb_cloud,
                api_key=chromadb_api_key,
                tenant=chromadb_tenant,
                database=chromadb_database
            )
            st.session_state.chroma_client = chroma_client
            st.session_state.chroma_collection = chroma_collection
            st.session_state.chroma_vector_store = chroma_vector_store
            st.session_state.chromadb_config = current_chromadb_config
            
            # Show connection status
            if chroma_client is not None:
                if use_chromadb_cloud:
                    st.sidebar.success(f"✅ Connected to ChromaDB Cloud: {chromadb_tenant}/{chromadb_database}")
                else:
                    st.sidebar.success("✅ ChromaDB initialized locally")
            else:
                st.sidebar.error("❌ ChromaDB initialization failed")
                if use_chromadb_cloud:
                    st.sidebar.warning("⚠️ Could not connect to ChromaDB Cloud. Please check your credentials and network connection.")
    
    if client is None or tts is None:
        st.stop()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["🔍 User Guide Summary", "💬 Q&A Chatbot", "🎫 Support Questions"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📥 Input")
            
            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["Upload file", "Paste text", "Load sample"]
            )
            
            transcript_text = ""
            
            if input_method == "Upload file":
                uploaded_file = st.file_uploader(
                    "Upload user guide document",
                    type=['txt', 'md'],
                    help="Upload a text file containing your user guide documentation",
                )
                
                if uploaded_file is not None:
                    transcript_text = str(uploaded_file.read(), "utf-8")
                    st.success(f"✅ File uploaded successfully! ({len(transcript_text)} characters)")
            
            elif input_method == "Paste text":
                transcript_text = st.text_area(
                    "Paste your user guide document here:",
                    height=300,
                    placeholder="Enter or paste your user guide documentation here..."
                )
            
            else:  # Load sample
                if st.button("📄 Load Sample User Guide"):
                    try:
                        with open("data/user_guide_sample.txt", "r") as f:
                            transcript_text = f.read()
                        st.success("✅ Sample user guide loaded!")
                    except FileNotFoundError:
                        st.error("❌ Sample file not found. Please create data/user_guide_sample.txt")
                        transcript_text = ""
            
            # Display input preview
            if transcript_text:
                with st.expander("📖 Preview Input"):
                    st.text_area("Document preview:", transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text, height=150, disabled=True)
        
        with col2:
            st.header("📤 Output")
            
            # Generate summary button
            if st.button("🎯 Generate Summary", type="primary", disabled=not transcript_text):
                if transcript_text.strip():
                    with st.spinner("🤖 Generating summary..."):
                        summary = summarize_user_guide(
                            client, 
                            transcript_text, 
                            summary_style, 
                            max_tokens, 
                            temperature,
                            language,
                            model
                        )
                        st.session_state.summary = summary
                        st.session_state.last_input = transcript_text
                        st.session_state.guide_context = transcript_text
                        # Clear chat history when new summary is generated
                        st.session_state.chat_history = []
                else:
                    st.warning("⚠️ Please provide a user guide document first.")
            
            # Display summary
            if st.session_state.summary:
                st.subheader("📋 User Guide Summary")
                
                # Summary output
                summary_container = st.container()
                with summary_container:
                    st.markdown(st.session_state.summary)
                
                # Action buttons
                col_download, col_copy = st.columns(2)
                
                with col_download:
                    # Download as text file
                    summary_bytes = st.session_state.summary.encode('utf-8')
                    st.download_button(
                        label="💾 Download Summary",
                        data=summary_bytes,
                        file_name="user_guide_summary.txt",
                        mime="text/plain"
                    )
                
                with col_copy:
                    # Copy to clipboard (placeholder - requires JavaScript)
                    if st.button("📋 Copy to Clipboard"):
                        st.info("💡 Use Ctrl+A, Ctrl+C to copy the summary above")
            
            # Display statistics
            if st.session_state.summary and st.session_state.last_input:
                st.subheader("📊 Statistics")
                input_words = len(st.session_state.last_input.split())
                output_words = len(st.session_state.summary.split())
                compression_ratio = round((1 - output_words/input_words) * 100, 1) if input_words > 0 else 0
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                col_stat1.metric("Input Words", input_words)
                col_stat2.metric("Output Words", output_words)
                col_stat3.metric("Compression", f"{compression_ratio}%")
    
    with tab2:
        st.header("🤖 User Guide Q&A Chatbot")
        
        # Check if there's a summary to chat about
        if not st.session_state.summary:
            st.info("� Please generate a user guide summary first to start chatting about it!")
            st.markdown("Go to the **User Guide Summary** tab to upload a document and generate a summary.")
        else:
            # Display user guide summary context
            with st.expander("📋 User Guide Context", expanded=False):
                st.markdown("**Current User Guide Summary:**")
                st.text_area("Summary", st.session_state.summary, height=100, disabled=True)
            
            # Chat interface
            st.markdown("**Ask questions about the user guide:**")
            st.info("💡 **Smart Language Detection**: Ask questions in any supported language, and I'll respond in the same language! The configured language above is used for summaries only.")
            chat_container = st.container()

            # Display chat history
            with chat_container:
                if len(st.session_state.chat_history) > 0:
                    st.markdown("### 💬 Conversation History")
                    # Display chat messages from history on app rerun
                    for i, message in enumerate(st.session_state.chat_history):
                        if (message["role"] == "assistant"):
                            print(message)
                            reasoning = message.get("reasoning")
                            tool_call = message.get("tool_call")
                            answer = message["content"]
                            display_assistant_response(answer, reasoning, tool_call)
                        else:
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])

                else:
                    # Suggested questions
                    if st.session_state.summary and len(st.session_state.chat_history) == 0:
                        st.markdown("### 💡 Suggested Questions (Multi-Language)")
                        suggested_questions = [
                            "What are the main features described in this guide?",
                            "¿Cuáles son las características principales descritas en esta guía?",
                            "Quelles sont les procédures étape par étape?",
                            "Welche wichtigen Konfigurationsschritte gibt es?",
                            "このガイドの主要な機能は何ですか？"
                        ]

                        cols = st.columns(2)
                        for i, suggestion in enumerate(suggested_questions):
                            with cols[i % 2]:
                                if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                                    # Set the question and trigger ask
                                    if client:
                                        st.session_state.chat_history.append({"role": "user", "content": suggestion})
                                        with st.chat_message("user"):
                                            st.markdown(suggestion)
                                        with st.spinner("🤔 Thinking..."):
                                            # Use auto-detection for suggested questions too
                                            answer, reasoning, tool_call = answer_question_auto_lang(
                                                client,
                                                suggestion,
                                                st.session_state.summary,
                                                st.session_state.last_input,
                                                language,  # fallback language
                                                model
                                            )
                                            display_assistant_response(answer, reasoning, tool_call)
                                            st.session_state.chat_history.append({"role": "assistant", "content": answer, "reasoning": reasoning, "tool_call": tool_call})
                                            st.rerun()

            # Process question
            question = st.chat_input("e.g., How do I configure this feature? | ¿Cómo configuro esta característica? | Comment configurer cette fonctionnalité?")

            if question:
                if client:
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(question)
                        with st.spinner("🤔 Thinking..."):
                            # Use auto-detection for chatbot, but keep manual language for summaries
                            answer, reasoning, tool_call = answer_question_auto_lang(
                                client,
                                question,
                                st.session_state.summary,
                                st.session_state.last_input,
                                language,  # fallback language
                                model
                            )
                            # Add assistant response to chat history
                            display_assistant_response(answer, reasoning, tool_call)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer, "reasoning": reasoning, "tool_call": tool_call})
                            scroll_to_bottom()
                        # Clear input and rerun to show new message
                        # st.rerun()
                else:
                    st.error("❌ Unable to process question. Please check your API configuration.")
            
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

        with tab3:
            st.header("🎫 Support Questions")    
            # Support Tickets Viewer (Admin Section)
            if st.session_state.support_tickets:
                st.markdown("**Customer Support Requests:**")
                for ticket in st.session_state.support_tickets:
                    st.markdown(f"""
                    **Ticket ID:** {ticket['id']}  
                    **Name:** {ticket['name']}  
                    **Email:** {ticket['email']}  
                    **Question:** {ticket['question']}  
                    **Previous Context:** {ticket.get('previous_question', 'N/A')}  
                    **Status:** {ticket['status']}  
                    **Time:** {ticket['timestamp']}
                    """)
                    st.markdown("---")
            

if __name__ == "__main__":
    main()