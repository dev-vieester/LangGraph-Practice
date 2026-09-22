from langgraph.graph import StateGraph, START, END
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

load_dotenv()

llm = ChatOpenRouter(
    model="inclusionai/ling-3.0-flash-vl:free",
    temperature=0
)

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    #take user query from state
    messages = state['messages']
    # send to llm
    response = llm.invoke(messages)
    # response store state
    return {'messages': [response]}

checkpoint = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

#add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpoint)