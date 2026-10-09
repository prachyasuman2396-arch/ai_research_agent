
import os
import ast
import operator

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from tavily import TavilyClient

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# 1. Initialize Groq
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

# 2. Calculator tool
@tool
def calculator(expression: str) -> str:
    """Calculate arithmetic expressions such as 25 * 4 + 10."""

    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large")

            return allowed[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.operand))

        raise ValueError("Unsupported expression")

    try:
        return str(evaluate(ast.parse(expression, mode="eval").body))
    except Exception as error:
        return f"Calculation error: {error}"


# 3. Web search tool
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    results = tavily.search(query=query, max_results=5)
    return str(results)


# 4. Third tool: illustrative weather tool
@tool
def get_weather(city: str) -> str:
    """Get weather for a city. This is a demo tool, not live weather data."""
    return (
        f"Weather data for {city} is not connected yet. "
        "Please connect a real weather API to retrieve current conditions."
    )


# 5. Register all tools in one list
tools = [calculator, web_search, get_weather]

# Give all three tools to Groq
llm_with_tools = llm.bind_tools(tools)


# 6. Assistant node
from langchain_core.messages import SystemMessage

def assistant_node(state: MessagesState):
    system_message = SystemMessage(
        content="""
        You are an AI Research Assistant.

        Your responsibilities:
        1. Understand the user's research question.
        2. Use web_search for questions requiring external or current information.
        3. Use calculator for numerical calculations.
        4. Base factual claims on the retrieved search results.
        5. Never invent sources, URLs, statistics, or benchmark results.
        6. If evidence is insufficient, clearly state the limitation.
        7. Give a clear, structured answer with source titles and URLs
           when available in the tool results.
        """
    )

    response = llm_with_tools.invoke(
        [system_message] + state["messages"]
    )

    return {"messages": [response]}

# 7. Build graph
builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    tools_condition,
)

builder.add_edge("tools", "assistant")

graph = builder.compile()


# 8. Run agent
if __name__ == "__main__":
    question = input("Ask a question: ")

    result = graph.invoke({
        "messages": [
            (
                "system",
                "You are a helpful assistant. "
                "Use calculator for arithmetic, web_search for current "
                "information, and get_weather when asked about weather. "
                "Never claim the demo weather tool returned live data. "
                "After tools return results, provide a clear answer."
            ),
            ("human", question),
        ]
    })

    print("\nFINAL ANSWER:")
    print(result["messages"][-1].content)
