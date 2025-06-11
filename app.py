from __future__ import annotations

import os
import yaml
import time


from langchain_core.tools import tool
from langchain.schema.runnable.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage


from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.types import Send

import chainlit as cl
from chainlit.types import ThreadDict
from google.genai import Client

from config import Configuration
from prompt import (
    get_current_date,
    query_generation_prompt,
    web_search_prompt,
    reflection_prompt,
    answer_prompt,
    rag_prompt,
)
from utils import (
    generate_chat_context,
    resolve_urls,
    get_citations,
    insert_citation_markers,
)
from state import MainState, WebSearchState, QueryGenerationState, ReflectionState
from schema import SearchQueryList, Reflection

try:
    with open("config.yaml") as f:
        yaml_config = yaml.safe_load(f)
except FileNotFoundError:
    yaml_config = None

for key in ["GOOGLE_API_KEY", "PINECONE_API_KEY", "OPENAI_API_KEY"]:
    if os.getenv(key) is None:
        raise ValueError(f"{key} is not set")


def route(state: MainState) -> str:
    """Choose branch based on is_research flag."""
    return "generate_query" if state.get("is_research") else "quick_answer_rag"


async def quick_answer_rag(state: MainState, config: RunnableConfig) -> MainState:
    configurable = Configuration.from_runnable_config(config)

    embeddings = OpenAIEmbeddings(model=configurable.embedding_model)
    vectorstore = PineconeVectorStore.from_existing_index(
        configurable.pinecone_index, embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    llm = ChatGoogleGenerativeAI(model=configurable.answer_model, temperature=0.1)

    prompt = ChatPromptTemplate.from_template(rag_prompt)

    user_query = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    docs = retriever.invoke(user_query)

    formatted_prompt = prompt.invoke(
        {
            "context": docs,
            "question": user_query,
            "previous_messages": generate_chat_context(state["messages"]),
        }
    )

    stream = llm.astream(formatted_prompt)
    msg = cl.Message(content="")

    async for chunk in stream:
        if token := chunk.content:
            await msg.stream_token(token)

    await msg.update()
    full_response = msg.content

    sources = {d.metadata["source"] for d in docs}

    return {
        "messages": [AIMessage(content=full_response)],
        "rag_sources": list(sources),
    }


def generate_query(state: MainState, config: RunnableConfig) -> QueryGenerationState:
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
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> MainState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)

    genai_client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

    formatted_prompt = web_search_prompt.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: MainState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reflection_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_prompt.format(
        current_date=current_date,
        research_topic=generate_chat_context(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> MainState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


async def finalize_answer(state: MainState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reflection_model = state.get("reflection_model") or configurable.reflection_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_prompt.format(
        current_date=current_date,
        research_topic=generate_chat_context(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=reflection_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    stream = llm.astream(formatted_prompt)
    msg = cl.Message(content="")

    async for chunk in stream:
        if token := chunk.content:
            await msg.stream_token(token)

    for src in state["sources_gathered"]:
        if src["short_url"] in msg.content:
            msg.content = msg.content.replace(src["short_url"], src["value"])

    await msg.update()
    full_summary = msg.content

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in full_summary:
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=full_summary)],
        "sources_gathered": unique_sources,
    }


def get_graph():
    builder = StateGraph(MainState, config_schema=Configuration)
    # Nodes
    builder.add_node("generate_query", generate_query)
    builder.add_node("quick_answer_rag", quick_answer_rag)
    builder.add_node("web_research", web_research)
    builder.add_node("reflection", reflection)
    builder.add_node("finalize_answer", finalize_answer)

    # Edges
    # builder.add_edge(START, "generate_query")
    builder.add_conditional_edges(
        START,
        route,
        ["generate_query", "quick_answer_rag"],  # ← DISPATCH POINT
    )
    # Add conditional edge to continue with search queries in a parallel branch
    builder.add_conditional_edges(
        "generate_query", continue_to_web_research, ["web_research"]
    )
    # Reflect on the web research
    builder.add_edge("web_research", "reflection")
    # Evaluate the research
    builder.add_conditional_edges(
        "reflection", evaluate_research, ["web_research", "finalize_answer"]
    )
    # Finalize the answer
    builder.add_edge("finalize_answer", END)
    builder.add_edge("quick_answer_rag", END)

    graph = builder.compile(name="rag-search-agent")
    return graph


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
    commands = [
        {
            "id": "Research",
            "icon": "globe",
            "description": "Deep research on a subject.",
            "button": True,
        },
    ]
    await cl.context.emitter.set_commands(commands)

    cl.user_session.set("chat_messages", [])

    graph = get_graph()

    cl.user_session.set("graph", graph)


graph = get_graph()


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
            label="Reinforce biodegradable polymers",
            message="Which metal-oxide nanoparticles are commonly used to reinforce starch-based biodegradable polymers? Explain in details.",
        ),
        cl.Starter(
            label="Polymer Dispersion",
            message="Tell me about Polymer Dispersion with a focus on sustainability and biobased ingredients",
        ),
    ]


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("chat_history", [])

    graph = get_graph()
    cl.user_session.set("graph", graph)

    restored: list[HumanMessage | AIMessage] = []
    for step in thread.get("steps", []):
        content = step.get("output")
        if not content:
            continue

        if step["type"] == "user_message":
            restored.append(HumanMessage(content=content))
        elif step["type"] == "assistant_message":
            restored.append(AIMessage(content=content))
    cl.user_session.set("chat_messages", restored)


@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time a user sends a message.
    It decides the execution path based on whether a command was used.
    """
    graph: StateGraph = cl.user_session.get("graph")
    chat_messages: list[AnyMessage] = cl.user_session.get("chat_messages", [])

    chat_messages.append(HumanMessage(content=message.content))

    inputs = {"messages": chat_messages, "is_research": message.command == "Research"}

    if inputs["is_research"]:
        step_text = "Deep research ..."
        step_type = "llm"
    else:
        step_text = "Thinking ..."
        step_type = "tool"

    async with cl.Step(name=step_text, type=step_type) as step:
        runnable_config = {"configurable": yaml_config} if yaml_config else None

        start = time.time()
        final_state = await graph.ainvoke(inputs, runnable_config)
        elapsed = time.time() - start

        search_sources = final_state.get("sources_gathered", [])
        rag_sources = final_state.get("rag_sources", [])

        if inputs["is_research"]:
            step.name = "Research Done in %d mins and %d secs" % divmod(
                int(elapsed), 60
            )
            step.output = f"Analyzed {len(search_sources)} web pages."
        else:
            step.name = "Retrieval Complete!"
            if rag_sources:
                step.type = "tool"
                bullet_list = "\n".join(f"- {src}" for src in rag_sources)
                step.output = (
                    f"{len(rag_sources)} document(s) retrieved:\n{bullet_list}"
                )
            else:
                step.output = "No documents retrieved."

    cl.user_session.set("chat_messages", final_state["messages"])


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
