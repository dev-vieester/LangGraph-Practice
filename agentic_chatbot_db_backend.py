import sqlite3

from langgraph.graph import StateGraph, START, END
from langchain_openrouter import ChatOpenRouter
from typing import Annotated, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv

load_dotenv()

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenRouter(
    model="inclusionai/ling-3.0-flash-vl:free",
    temperature=0
)

def chat_node(state: ChatState):
    message = state["messages"]

    response = llm.invoke(message)

    return {
        "messages": [response]
    }

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    all_threads = set()

    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)