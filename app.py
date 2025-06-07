from __future__ import annotations

import os
import operator

from pydantic import BaseModel, Field
from typing_extensions import Annotated


from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

import chainlit as cl
from tavily import TavilyClient

from config import Configuration
from prompt import query_generation_prompt, get_current_date


class MainState(MessagesState):
    search_query: Annotated[list, operator.add]
    initial_search_query_count: int


class SearchQueryList(BaseModel):
    query: list[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


def generate_chat_context(messages: list[AnyMessage]) -> str:
    """
    Combine message history into a single string.
    """
    if len(messages) == 1:
        research_topic = messages[-1].content
    else:
        research_topic = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                research_topic += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                research_topic += f"Assistant: {message.content}\n"
    return research_topic


def generate_query(state: MainState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    current_date = get_current_date()
    formatted_prompt = query_generation_prompt.format(
        current_date=current_date,
        research_topic=generate_chat_context(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    """
    This function will be called to authenticate users.
    """
    if username == os.getenv("CHAINLIT_USERNAME") and password == os.getenv(
        "CHAINLIT_PASSWORD"
    ):
        return cl.User(
            identifier=username, metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def start():
    if os.getenv("GOOGLE_API_KEY") is None:
        raise ValueError("GOOGLE_API_KEY is not set")
    if os.getenv("TAVILY_API_KEY") is None:
        raise ValueError("TAVILY_API_KEY is not set")

    commands = [
        {
            "id": "Search",
            "icon": "globe",
            "description": "Find information on the web",
        },
    ]
    await cl.context.emitter.set_commands(commands)

    cl.user_session.set("chat_messages", [])

    builder = StateGraph(MainState, config_schema=Configuration)

    builder.add_node("generate_query", generate_query)

    builder.add_edge(START, "generate_query")
    builder.add_edge("generate_query", END)

    graph = builder.compile(name="rag-search-agent")

    cl.user_session.set("graph", graph)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Aqueous Poly(butenolide) Dispersions",
            message="I’m exploring aqueous poly(butenolide) dispersions as sustainable binders for waterborne coatings. Could you help me understand the key steps in their synthesis, critical performance metrics (e.g., particle size, film formation), and how to optimize the formulation for maximal environmental benefit?",
        ),
        cl.Starter(
            label="In-Situ Lignin-CNF Dispersions",
            message="Let’s examine the development of bio-based aqueous dispersions formed by in-situ polymerizing lignin onto cellulose nanofibrils (CNF). Can you walk me through the enzymatic polymerization mechanism, materials characterization methods, and potential applications in sustainable packaging?",
        ),
        cl.Starter(
            label="Lignin-CWPU-LX Dispersions",
            message="I want to design a lignin-incorporated, castor oil-based cationic waterborne PU dispersion (CWPU-LX). Help me outline a green synthesis plan, solvent-minimization strategies, and approaches to characterize the dispersion’s stability and performance.",
        ),
    ]


@cl.step(type="tool")
async def tool():
    await cl.sleep(2)
    return "Response from the tool!"


@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time a user sends a message.
    It decides the execution path based on whether a command was used.
    """
    graph: StateGraph = cl.user_session.get("graph")
    chat_messages: list[AnyMessage] = cl.user_session.get("chat_messages")

    chat_messages.append(HumanMessage(content=message.content))

    async with cl.Step(name="Generating search queries ...", type="llm") as step:
        step.input = message.content
        step.status = "Thinking..."

        inputs = {"messages": chat_messages}

        final_state = await graph.ainvoke(inputs)

        queries = final_state.get("search_query", [])

        query_list_str = "\n".join(f"- `{q}`" for q in queries)
        response_message = f"I've generated the following search queries to research your topic:\n{query_list_str}"

        step.output = response_message

    await cl.Message(content=response_message).send()

    cl.user_session.set("chat_messages", chat_messages)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
