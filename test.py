
import os

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from schemas import ResearchReport


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from .env")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY is missing from .env")


# ==========================================
# 2. INITIALIZE LLM AND TAVILY
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

tavily = TavilyClient(
    api_key=tavily_api_key
)


# ==========================================
# 3. CALCULATOR TOOL
# ==========================================

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""

    # Learning-only implementation.
    # Do not use unrestricted eval with untrusted
    # input in production.

    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {e}"


# ==========================================
# 4. WEB SEARCH TOOL
# ==========================================

@tool
def web_search(query: str) -> str:
    """Search the web for research information
    and recent developments."""

    try:
        results = tavily.search(
            query=query,
            max_results=5,
        )

        cleaned_results = []

        for item in results.get("results", []):
            cleaned_results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            })

        return str(cleaned_results)

    except Exception as e:
        return f"Web search failed: {e}"


# ==========================================
# 5. REGISTER TOOLS
# ==========================================

tools = [
    calculator,
    web_search,
]

tools_by_name = {
    tool.name: tool
    for tool in tools
}

llm_with_tools = llm.bind_tools(tools)


# ==========================================
# 6. DEFINE GRAPH STATE
# ==========================================

class ResearchState(MessagesState):
    report: ResearchReport | None


# ==========================================
# 7. AGENT NODE
# ==========================================

def agent(state: ResearchState):
    system_message = SystemMessage(
        content="""
        You are an AI Research Assistant.

        Your responsibilities:

        1. Understand the user's research question.
        2. Use web_search for current or external
           factual information.
        3. Use calculator for numerical calculations
           when appropriate.
        4. Use the tools available to you rather
           than inventing search results.
        5. Base factual claims on retrieved evidence.
        6. Never invent statistics, benchmark results,
           sources, or URLs.
        7. When you have enough evidence, provide
           a concise answer based on the findings.
        8. Clearly state when evidence is insufficient.
        """
    )

    response = llm_with_tools.invoke(
        [system_message] + state["messages"]
    )

    return {
        "messages": [response]
    }


# ==========================================
# 8. STRUCTURED REPORT NODE
# ==========================================

structured_llm = llm.with_structured_output(
    ResearchReport
)


def generate_report(state: ResearchState):
    prompt = f"""
    Generate a structured research report using
    the conversation and retrieved tool results.

    Conversation:
    {state["messages"]}

    Requirements:
    - Answer the original research question.
    - Summarize the relevant evidence.
    - Compare the requested alternatives when applicable.
    - Give a reasoned recommendation.
    - Include only source URLs actually present
      in the retrieved tool results.
    - Do not invent statistics, sources, or URLs.
    - If evidence is insufficient, explain the limitation.
    - Return the result using the ResearchReport schema.
    """

    report = structured_llm.invoke(prompt)

    return {
        "report": report
    }


# ==========================================
# 9. CONDITIONAL ROUTING
# ==========================================

def route_after_agent(state: ResearchState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return "generate_report"


# ==========================================
# 10. BUILD LANGGRAPH
# ==========================================

builder = StateGraph(ResearchState)

# Register nodes
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_node("generate_report", generate_report)

# Entry point
builder.add_edge(START, "agent")

# Conditional routing
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        "generate_report": "generate_report",
    },
)

# Tool results return to the agent
builder.add_edge("tools", "agent")

# Report generation finishes the graph
builder.add_edge("generate_report", END)

# Compile
graph = builder.compile()


# ==========================================
# 11. RUN RESEARCH ASSISTANT
# ==========================================

if __name__ == "__main__":

    question = input(
        "\nEnter your research question: "
    ).strip()

    if not question:
        raise ValueError("Research question cannot be empty.")

    result = graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": question,
            }
        ]
    })

    report = result["report"]

    print("\n" + "=" * 50)
    print("AI RESEARCH REPORT")
    print("=" * 50)

    print(report.model_dump_json(indent=2))

