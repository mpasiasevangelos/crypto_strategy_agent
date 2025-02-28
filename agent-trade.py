import os
import warnings
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

load_dotenv()

from langchain_ollama import ChatOllama

model = "llama3.2:latest"
# model = "qwen2.5"
llm = ChatOllama(model=model, base_url="http://localhost:11434")
#print(llm.invoke("Hello, how are you?"))

from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url='http://localhost:11434'
)

db_name = "trade"
vector_store = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs = {'k': 5})

question = "tell me about the falkands"
#print(retriever.invoke(question))


from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "trade",
    "Search and return information about the trading strategies",
)

tools = [retriever_tool]

from typing import Annotated, Sequence, TypedDict, Literal 
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

from langchain import hub
from langchain_core.messages import  BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with tool and validation
    llm_with_structured_output = llm.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_structured_output

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]

    llm_with_tools = llm.bind_tools(tools, tool_choice="required")
    response = llm_with_tools.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    response = llm.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent)
retriever = ToolNode([retriever_tool])
graph_builder.add_node("retriever", retriever)
graph_builder.add_node("rewrite", rewrite)

graph_builder.add_node("generate", generate)


graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges( 
    "agent",

    tools_condition,
    {
        "tools": "retriever",
        END: END
    }
)

graph_builder.add_conditional_edges(
    "retriever",
    grade_documents
)

graph_builder.add_edge("generate", END)
graph_builder.add_edge("rewrite", "agent")

graph = graph_builder.compile()

from pprint import pprint


#query = {"messages": [HumanMessage("Here is the most recent OHLCV data for Bitcoin over the past 4 hours:{ohlcv_data} Based on strategies from the database, what trading decision should i make? ")]}

import pandas as pd
import pandas_ta as ta

# Load CSV file
file_path = "BTCUSDT_data.csv"  # Path to your CSV file
ohlcv_data = pd.read_csv(file_path)

# Ensure the DataFrame has the required columns
required_columns = ['open', 'high', 'low', 'close', 'volume']
if not all(col in ohlcv_data.columns for col in required_columns):
    raise ValueError(f"The input CSV must contain the following columns: {required_columns}")

# Add TA indicators
ohlcv_data['MACD'], ohlcv_data['MACD_Signal'], ohlcv_data['MACD_Hist'] = ta.macd(ohlcv_data['close'])
ohlcv_data['EMA_12'] = ta.ema(ohlcv_data['close'], length=12)
ohlcv_data['SMA_50'] = ta.sma(ohlcv_data['close'], length=50)

# Format OHLCV data as a string for the query
ohlcv_data_formatted = ohlcv_data.tail(10).to_string(index=False)  # Take the last 10 rows for brevity

# Define the query with the OHLCV data and TA indicators
query = {
    "messages": [
        HumanMessage(
            content=f"""Here is the most recent OHLCV data for Bitcoin over the past 4 hours, including technical indicators (MACD, EMA, SMA):\n\n{ohlcv_data_formatted}\n\nBased on strategies from the database, what trading decision should I make?"""
        )
    ]
}

# graph.invoke(query)

for output in graph.stream(query):
    for key, value in output.items():
        pprint(f"Output from node '{key}':")
        pprint("----")
        pprint(value, indent=4, width=120)

    pprint("\n------\n")