import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()


# =========================
# LLM
# =========================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# =========================
# Calculator Tool
# =========================

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    
    try:
        return str(eval(expression))
    
    except Exception as e:
        return f"Error: {e}"


# =========================
# Web Search Tool
# =========================

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    
    results = tavily.search(
        query=query,
        max_results=5
    )

    return str(results)


# =========================
# Tools
# =========================

tools = {
    "calculator": calculator,
    "web_search": web_search
}


llm_with_tools = llm.bind_tools(
    list(tools.values())
)


# =========================
# User Question
# =========================

messages = [
    (
        "user",
        "What is the latest stable version of Python?"
    )
]


# =========================
# Agent Loop
# =========================

while True:

    response = llm_with_tools.invoke(messages)

    print("\n====================")
    print("LLM RESPONSE")
    print("====================")

    print(response)


    # Add LLM response
    messages.append(response)


    # =========================
    # No tool → Finish
    # =========================

    if not response.tool_calls:

        print("\n====================")
        print("FINAL ANSWER")
        print("====================")

        print(response.content)

        break


    # =========================
    # Execute tools
    # =========================

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        tool_args = tool_call["args"]


        print("\n====================")
        print("TOOL CALL")
        print("====================")

        print("Tool:", tool_name)
        print("Arguments:", tool_args)


        # Find tool
        selected_tool = tools[tool_name]


        # Execute
        tool_result = selected_tool.invoke(
            tool_args
        )


        print("\nTOOL RESULT:")
        print(tool_result)


        # Add tool result to messages
        messages.append(
            {
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call["id"]
            }
        )