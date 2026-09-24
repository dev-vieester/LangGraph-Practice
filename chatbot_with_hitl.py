import os
import tempfile
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agentic_chatbot_hitl_backend import chatbot, get_all_threads, ingest_rag_document


st.set_page_config(
    page_title="Agentic ChatGPT",
    page_icon="AI",
    layout="wide",
)


def generate_thread_id():
    return str(uuid.uuid4())


def get_config(thread_id):
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


def init_state():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = generate_thread_id()

    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    if "chat_threads" not in st.session_state:
        st.session_state.chat_threads = get_all_threads()

    if "pending_hitl" not in st.session_state:
        st.session_state.pending_hitl = None

    add_thread(st.session_state.thread_id)


def reset_chat():
    st.session_state.thread_id = generate_thread_id()
    st.session_state.message_history = []
    st.session_state.pending_hitl = None
    add_thread(st.session_state.thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config=get_config(thread_id)
    )

    messages = []

    for message in state.values.get("messages", []):
        if isinstance(message, HumanMessage):
            messages.append(
                {
                    "role": "user",
                    "content": message.content
                }
            )

        elif isinstance(message, AIMessage) and message.content:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content
                }
            )

    return messages


def select_thread(thread_id):
    st.session_state.thread_id = thread_id
    st.session_state.message_history = load_conversation(thread_id)
    st.session_state.pending_hitl = None


def first_user_message():
    for message in st.session_state.message_history:
        if message["role"] == "user":
            text = message["content"].strip().replace("\n", " ")
            return text[:30] + ("..." if len(text) > 30 else "")

    return "New chat"


def append_latest_ai_message(result):
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            st.session_state.message_history.append(
                {
                    "role": "assistant",
                    "content": message.content
                }
            )
            return

    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": "Done."
        }
    )


def handle_result(result):
    interrupts = result.get("__interrupt__", [])

    if interrupts:
        prompt = interrupts[0].value
        st.session_state.pending_hitl = {
            "thread_id": st.session_state.thread_id,
            "prompt": prompt,
        }
        st.session_state.message_history.append(
            {
                "role": "assistant",
                "content": f"Approval required: {prompt}"
            }
        )
        return

    st.session_state.pending_hitl = None
    append_latest_ai_message(result)


def send_message(user_input):
    st.session_state.message_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    result = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=get_config(st.session_state.thread_id),
    )

    handle_result(result)


def resume_hitl(decision):
    pending = st.session_state.pending_hitl

    if not pending:
        return

    result = chatbot.invoke(
        Command(resume=decision),
        config=get_config(pending["thread_id"]),
    )

    st.session_state.pending_hitl = None
    append_latest_ai_message(result)


def apply_styles():
    st.markdown(
        """
        <style>
            .stApp {
                background: #212121;
                color: #ececec;
            }

            header[data-testid="stHeader"] {
                display: none;
            }

            section[data-testid="stSidebar"] {
                background: #171717;
                border-right: 1px solid #2f2f2f;
            }

            section[data-testid="stSidebar"] * {
                color: #ececec;
            }

            section[data-testid="stSidebar"] button {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 8px;
                min-height: 40px;
                text-align: left;
                width: 100%;
            }

            section[data-testid="stSidebar"] button:hover {
                background: #2f2f2f;
                border-color: #3a3a3a;
            }

            div[data-testid="stSidebarUserContent"] {
                padding-top: 1rem;
            }

            .main .block-container {
                max-width: 900px;
                padding-bottom: 7rem;
                padding-top: 1.2rem;
            }

            .topbar {
                align-items: center;
                border-bottom: 1px solid #3a3a3a;
                display: flex;
                justify-content: space-between;
                margin-bottom: 1.5rem;
                padding-bottom: 0.9rem;
            }

            .brand {
                font-size: 1.05rem;
                font-weight: 650;
            }

            .thread {
                color: #b4b4b4;
                font-size: 0.82rem;
            }

            .welcome {
                margin: 18vh auto 2rem;
                max-width: 720px;
                text-align: center;
            }

            .welcome h1 {
                font-size: 2.1rem;
                font-weight: 650;
                letter-spacing: 0;
                margin-bottom: 0.5rem;
            }

            .welcome p {
                color: #b4b4b4;
            }

            .prompt-grid {
                display: grid;
                gap: 0.75rem;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                margin: 2rem auto 0;
                max-width: 720px;
            }

            .prompt-card {
                background: #2f2f2f;
                border: 1px solid #3f3f3f;
                border-radius: 8px;
                padding: 0.9rem 1rem;
                text-align: left;
            }

            .prompt-card strong {
                display: block;
                margin-bottom: 0.2rem;
            }

            .prompt-card span {
                color: #b4b4b4;
                font-size: 0.88rem;
            }

            .hitl {
                background: #332b16;
                border: 1px solid #8f6f22;
                border-radius: 8px;
                margin: 1rem 0;
                padding: 1rem;
            }

            .hitl strong {
                color: #f4d47c;
            }

            div[data-testid="stChatMessage"] {
                background: transparent;
                color: #ececec;
            }

            div[data-testid="stChatMessageContent"] {
                color: #ececec;
            }

            div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
                background: #2b2b2b;
                border-bottom: 1px solid #3a3a3a;
                border-top: 1px solid #3a3a3a;
                margin-left: calc(50% - 50vw);
                margin-right: calc(50% - 50vw);
                padding-left: calc(50vw - 450px);
                padding-right: calc(50vw - 450px);
            }

            div[data-testid="stChatInput"] textarea {
                background: #2f2f2f;
                border: 1px solid #4a4a4a;
                color: #ececec;
            }

            div[data-testid="stChatInput"] textarea::placeholder {
                color: #b4b4b4;
            }

            div[data-testid="stFileUploader"] {
                background: #202020;
                border-radius: 8px;
                padding: 0.4rem;
            }

            div[data-testid="stAlert"] {
                background: #2f2f2f;
                color: #ececec;
            }

            @media (max-width: 760px) {
                .prompt-grid {
                    grid-template-columns: 1fr;
                }

                div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
                    padding-left: 0.85rem;
                    padding-right: 0.85rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### Agentic ChatGPT")
        st.caption("LangGraph assistant with tools and approval.")

        if st.button("+ New chat"):
            reset_chat()
            st.rerun()

        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        if uploaded_file and st.button("Ingest PDF"):
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                with st.spinner("Indexing PDF..."):
                    result = ingest_rag_document(temp_path)

                st.success(result)
            except Exception as exc:
                st.error(f"Could not ingest PDF: {exc}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        st.divider()
        st.markdown("#### Conversations")

        for index, thread_id in enumerate(st.session_state.chat_threads[::-1]):
            label = first_user_message() if thread_id == st.session_state.thread_id else thread_id[:8]

            if st.button(label, key=f"thread_{thread_id}_{index}"):
                select_thread(thread_id)
                st.rerun()


def render_header():
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">Agentic ChatGPT</div>
            <div class="thread">Thread {st.session_state.thread_id[:8]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome():
    st.markdown(
        """
        <div class="welcome">
            <h1>How can I help?</h1>
            <p>Ask questions, search the web, calculate, query a PDF, or request a stock purchase that needs approval.</p>
        </div>
        <div class="prompt-grid">
            <div class="prompt-card">
                <strong>Use tools</strong>
                <span>What is the current price of AAPL?</span>
            </div>
            <div class="prompt-card">
                <strong>Query a PDF</strong>
                <span>Upload a document, then ask for a summary.</span>
            </div>
            <div class="prompt-card">
                <strong>Calculate</strong>
                <span>Calculate 18% of 2450 plus 75.</span>
            </div>
            <div class="prompt-card">
                <strong>Human approval</strong>
                <span>Buy 2 shares of TSLA.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_messages():
    for message in st.session_state.message_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_hitl():
    pending = st.session_state.pending_hitl

    if not pending:
        return

    st.markdown(
        f"""
        <div class="hitl">
            <strong>Approval needed</strong><br>
            {pending["prompt"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    approve_col, decline_col, spacer = st.columns([1, 1, 4])

    with approve_col:
        if st.button("Approve", type="primary", use_container_width=True):
            with st.spinner("Resuming workflow..."):
                resume_hitl("yes")
            st.rerun()

    with decline_col:
        if st.button("Decline", use_container_width=True):
            with st.spinner("Resuming workflow..."):
                resume_hitl("no")
            st.rerun()


init_state()
apply_styles()
render_sidebar()
render_header()

if st.session_state.message_history:
    render_messages()
else:
    render_welcome()

render_hitl()

user_input = st.chat_input(
    "Message Agentic ChatGPT",
    disabled=st.session_state.pending_hitl is not None,
)

if user_input:
    with st.spinner("Thinking..."):
        send_message(user_input)

    st.rerun()
