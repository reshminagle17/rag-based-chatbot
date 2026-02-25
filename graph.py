# graph.py
import os
from dotenv import load_dotenv
load_dotenv()  # ← THIS LINE IS CRITICAL - must be before ChatGroq init

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from rag_pipeline import load_vectorstore, retrieve

llm = ChatGroq(model="llama3-8b-8192", temperature=0.2)

# ── State Schema ──────────────────────────────────────────
class ChatState(TypedDict):
    query: str
    intent: str
    context: List[str]
    confidence: str
    response: str
    formatted_response: str

# ── Node 1: Input Processing ──────────────────────────────
def input_processing_node(state: ChatState) -> ChatState:
    state["query"] = state["query"].strip()
    return state

# ── Node 2: Intent Routing ────────────────────────────────
def intent_routing_node(state: ChatState) -> ChatState:
    q = state["query"].lower()
    if any(w in q for w in ["hello", "hi", "hey", "how are you"]):
        state["intent"] = "greeting"
    elif any(w in q for w in ["what", "how", "why", "explain", "tell me", "define"]):
        state["intent"] = "knowledge_query"
    else:
        state["intent"] = "general"
    return state

# ── Node 3: Retrieval ─────────────────────────────────────
def retrieval_node(state: ChatState) -> ChatState:
    vs = load_vectorstore()
    docs, confidence = retrieve(state["query"], vs)
    state["context"] = docs
    state["confidence"] = confidence
    return state

# ── Node 4: Context Validation ────────────────────────────
def context_validation_node(state: ChatState) -> ChatState:
    if not state["context"] or state["confidence"] == "low":
        state["confidence"] = "low"
    else:
        state["confidence"] = "high"
    return state

# ── Node 5: Response Generation ──────────────────────────
def response_generation_node(state: ChatState) -> ChatState:
    context_text = "\n\n".join(state["context"])
    messages = [
        SystemMessage(content=f"""You are a helpful assistant.
Answer ONLY using the context below. If the answer is not in the context, say you don't know.

CONTEXT:
{context_text}"""),
        HumanMessage(content=state["query"])
    ]
    resp = llm.invoke(messages)
    state["response"] = resp.content
    return state

# ── Node 6: Fallback ──────────────────────────────────────
def fallback_node(state: ChatState) -> ChatState:
    if state["intent"] == "greeting":
        state["response"] = "Hello! I'm your RAG assistant. Ask me anything about the knowledge base!"
    else:
        state["response"] = (
            "I couldn't find confident information to answer that. "
            "Please try rephrasing or ask something else."
        )
    return state

# ── Node 7: Response Formatter ────────────────────────────
def response_formatter_node(state: ChatState) -> ChatState:
    conf_label = "✅ High Confidence" if state["confidence"] == "high" else "⚠️ Low Confidence"
    state["formatted_response"] = f"{state['response']}\n\n_{conf_label}_"
    return state

# ── Routing Logic ─────────────────────────────────────────
def route_after_intent(state: ChatState) -> str:
    if state["intent"] == "greeting":
        return "fallback"
    return "retrieval"

def route_after_validation(state: ChatState) -> str:
    return "generation" if state["confidence"] == "high" else "fallback"

# ── Build Graph ───────────────────────────────────────────
def build_graph():
    g = StateGraph(ChatState)

    g.add_node("input",      input_processing_node)
    g.add_node("intent",     intent_routing_node)
    g.add_node("retrieval",  retrieval_node)
    g.add_node("validation", context_validation_node)
    g.add_node("generation", response_generation_node)
    g.add_node("fallback",   fallback_node)
    g.add_node("formatter",  response_formatter_node)

    g.set_entry_point("input")
    g.add_edge("input", "intent")

    g.add_conditional_edges("intent", route_after_intent,
                            {"retrieval": "retrieval", "fallback": "fallback"})

    g.add_edge("retrieval", "validation")

    g.add_conditional_edges("validation", route_after_validation,
                            {"generation": "generation", "fallback": "fallback"})

    g.add_edge("generation", "formatter")
    g.add_edge("fallback",   "formatter")
    g.add_edge("formatter",  END)

    return g.compile()

graph = build_graph()

def run_query(query: str) -> dict:
    result = graph.invoke({
        "query": query,
        "intent": "",
        "context": [],
        "confidence": "",
        "response": "",
        "formatted_response": ""
    })
    return result
