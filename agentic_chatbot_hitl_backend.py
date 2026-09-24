from langgraph.graph import StateGraph, START
from typing import Annotated, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_openrouter import ChatOpenRouter
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.types import Command, interrupt
import requests
import math

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

llm = ChatOpenRouter(
    model="inclusionai/ling-3.0-flash-vl",
    temperature=0
)


def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    if documents:
        chunks = splitter.split_documents(documents)

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        vector_store.save_local(DB_PATH)

        return "Document successfully ingested"

    else:
        return "No Document to process"


def get_retriever():
    DB_PATH = "faiss_db"

    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    return retriever


@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.
    """

    retriever = get_retriever()

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown page"
        )

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    """

    try:

        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(
            expression,
            {"__builtins__": {}},
            allowed
        )

        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol.
    """

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey=9MZO2JUBR7IFNTOI"
    )

    r = requests.get(url)

    return r.json()


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    Human-in-the-Loop:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision.
    """

    decision = interrupt(
        f"Approve buying {quantity} share of {symbol}? (yes/no)"
    )

    if isinstance(decision, str) and decision.lower() == "yes":

        return {
            "status": "success",
            "message": (
                f"Purchase order placed for "
                f"{quantity} shares of {symbol}."
            ),
            "symbol": symbol,
            "quantity": quantity,
        }

    else:

        return {
            "status": "cancelled",
            "message": (
                f"Purchase of {quantity} shares of "
                f"{symbol} was declined by human."
            ),
            "symbol": symbol,
            "quantity": quantity,
        }


tools = [
    search_tool,
    calculator,
    get_stock_price,
    rag_tool,
    purchase_stock
]

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


def chat_node(state: ChatState):

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access "
            "to several tools.\n\n"

            "Tool usage instructions:\n"

            "- Use `rag_tool` for questions about the uploaded "
            "PDF or document. Always retrieve relevant document "
            "content before answering PDF-related questions.\n"

            "- Use `search_tool` for current events, recent "
            "information, or information that requires an "
            "internet search.\n"

            "- Use `calculator` for mathematical calculations. "
            "Do not calculate complex expressions manually when "
            "the calculator is available.\n"

            "- Use `get_stock_price` when the user asks for the "
            "current price of a stock.\n"

            "- Use `purchase_stock` when the user wants to "
            "purchase a stock.\n\n"

            "Answer general questions directly when no tool "
            "is required.\n"

            "Do not invent information from the uploaded "
            "document.\n"

            "If the user asks about a PDF but no document is "
            "available, ask them to upload a PDF.\n"

            "After receiving a tool result, provide a clear "
            "and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)


conn = sqlite3.connect(
    database="new_chat.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)


graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge(
    "tools",
    "chat_node"
)


chatbot = graph.compile(
    checkpointer=checkpointer
)


def get_all_threads():

    all_threads = set()

    for ckpt in checkpointer.list(None):

        all_threads.add(
            ckpt.config["configurable"]["thread_id"]
        )

    return list(all_threads)


if __name__ == "__main__":

    print("🤖 Agentic Chatbot CLI\n")
    print("Type 'exit' to quit.\n")

    thread_id = "demo-thread"

    while True:

        user_input = input("You: ")

        if user_input.lower().strip() in {
            "exit",
            "quit"
        }:
            print("Goodbye")
            break

        state = {
            "messages": [
                HumanMessage(
                    content=user_input
                )
            ]
        }

        result = chatbot.invoke(
            state,
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        interrupts = result.get(
            "__interrupt__",
            []
        )

        if interrupts:

            prompt_to_human = interrupts[0].value

            print(
                f"HITL: {prompt_to_human}"
            )

            decision = input(
                "Your decision: "
            ).strip().lower()

            result = chatbot.invoke(
                Command(
                    resume=decision
                ),
                config={
                    "configurable": {
                        "thread_id": thread_id
                    }
                }
            )

        messages = result["messages"]

        last_msg = messages[-1]

        print(
            f"Bot: {last_msg.content}\n"
        )