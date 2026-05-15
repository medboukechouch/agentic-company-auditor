from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field


class AuditState(TypedDict):
    company_name: str
    company_context: str
    proposed_use_cases: List[str]
    final_report: str
    retry_count: int


class UseCase(BaseModel):
    title: str = Field(..., min_length=3)
    description: str = Field(..., min_length=10)
    expected_value: str = Field(..., min_length=5)


class AuditReport(BaseModel):
    company_name: str
    company_context: str
    proposed_use_cases: List[UseCase]
    assumptions: List[str]
    next_steps: List[str]


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gemini-flash-latest"
    temperature: float = 0.2


def build_llm(config: LLMConfig) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=config.model, temperature=config.temperature)


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def research_node(state: AuditState, llm: ChatGoogleGenerativeAI) -> AuditState:
    search = DuckDuckGoSearchRun()
    results = search.run(f"{state['company_name']} company data challenges AI")
    prompt = (
        "You are a senior market researcher. Synthesize a concise company context based on the web search results. "
        "Include industry, size, challenges, and data landscape. If the results are sparse, make reasonable inferences."
        f"\n\nCompany: {state['company_name']}"
        f"\n\nSearch results:\n{results}"
    )
    response = llm.invoke(prompt)
    context = _to_text(response.content).strip()
    return {**state, "company_context": context}


def _extract_distinct_facts(text: str) -> List[str]:
    if not text:
        return []
    separators = ["\n", ".", ";", ":", "-"]
    normalized = text
    for sep in separators:
        normalized = normalized.replace(sep, "|")
    parts = [part.strip() for part in normalized.split("|") if part.strip()]
    distinct = []
    for part in parts:
        if part not in distinct:
            distinct.append(part)
    return distinct


def validation_node(state: AuditState) -> AuditState:
    facts = _extract_distinct_facts(state["company_context"])
    if len(facts) >= 3:
        return state
    return {**state, "retry_count": state["retry_count"] + 1}


def should_retry(state: AuditState) -> str:
    facts = _extract_distinct_facts(state["company_context"])
    if len(facts) >= 3:
        return "continue"
    if state["retry_count"] < 2:
        return "retry"
    return "continue"


def ideation_node(state: AuditState, llm: ChatGoogleGenerativeAI) -> AuditState:
    prompt = (
        "You are a data/AI consultant. Based on the company context, propose exactly two AI/Data use cases. "
        "Return them as a numbered list with title and short description."
        f"\n\nCompany context:\n{state['company_context']}"
    )
    response = llm.invoke(prompt)
    content = _to_text(response.content)
    use_cases = [line.strip() for line in content.splitlines() if line.strip()]
    return {**state, "proposed_use_cases": use_cases}


def reporting_node(state: AuditState, llm: ChatGoogleGenerativeAI) -> AuditState:
    prompt = (
        "You are a tech lead. Produce a JSON report that matches this schema:\n"
        "AuditReport(company_name: str, company_context: str, proposed_use_cases: "
        "List[UseCase(title, description, expected_value)], assumptions: List[str], next_steps: List[str]).\n"
        "Use the company context and proposed use cases. Return only valid JSON."
        f"\n\nCompany name: {state['company_name']}\n"
        f"Company context: {state['company_context']}\n"
        f"Proposed use cases: {state['proposed_use_cases']}"
    )
    response = llm.invoke(prompt)
    report = AuditReport.model_validate_json(_to_text(response.content))
    return {**state, "final_report": report.model_dump_json(ensure_ascii=False)}


def build_compiled_graph(config: LLMConfig | None = None):
    llm = build_llm(config or LLMConfig())
    graph = StateGraph(AuditState)

    graph.add_node("research", lambda s: research_node(s, llm))
    graph.add_node("validation", validation_node)
    graph.add_node("ideation", lambda s: ideation_node(s, llm))
    graph.add_node("reporting", lambda s: reporting_node(s, llm))

    graph.set_entry_point("research")
    graph.add_edge("research", "validation")
    graph.add_conditional_edges(
        "validation",
        should_retry,
        {
            "retry": "research",
            "continue": "ideation",
        },
    )
    graph.add_edge("ideation", "reporting")
    graph.add_edge("reporting", END)

    return graph.compile()


compiled_graph = build_compiled_graph()
