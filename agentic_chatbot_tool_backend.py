import sqlite3

from langgraph.graph import StateGraph, START, END
from langchain_openrouter import ChatOpenRouter
from typing import Annotated, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langgraph.prebuilt import tools_condition, ToolNode
from langchain.tools import tool
import math
import requests

load_dotenv()

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOpenRouter(
    model="inclusionai/ling-3.0-flash-vl:free",
    temperature=0
)

