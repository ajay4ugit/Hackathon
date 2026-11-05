"""
evaluator.py – Agri Doctor LLM & Agent Evaluation Engine
=========================================================
Computes all six evaluation-metric categories purely in Python
(no external eval framework required):

  1. Accuracy & Quality      – ROUGE-L, BLEU-1, semantic similarity,
                               exact match, precision/recall/F1
  2. RAG Evaluation          – context precision/recall, faithfulness,
                               answer relevancy, groundedness
  3. Agentic AI Metrics      – task completion, goal achievement,
                               tool accuracy, reasoning quality,
                               agent coordination, handoff accuracy,
                               loop detection
  4. Safety & Guardrail      – hallucination rate, toxicity, bias,
                               prompt-injection resistance, PII leakage,
                               guardrail intervention rate
  5. Performance & Cost      – response latency, TTFT, tokens/sec,
                               input/output tokens, cost/request,
                               error rate, throughput
  6. Multi-Agent System      – end-to-end time, per-agent accuracy,
                               inter-agent consistency, task distribution
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Shared structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentRun:
    """Raw telemetry captured during a single agent invocation."""
    agent_name: str
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    ttft_seconds: float = 0.0           # time to first token (approximated)
    succeeded: bool = True
    error_message: str = ""
    prompt_text: str = ""               # sanitised prompt (no PII)
    response_text: str = ""
    guardrail_blocks: int = 0
    guardrail_warnings: int = 0
    guardrail_checks: int = 0
    iteration: int = 1
    # Cost constants (USD per 1 k tokens — rough public estimates)
    input_cost_per_1k: float = 0.0003
    output_cost_per_1k: float = 0.0006


@dataclass
class EvalReport:
    """Aggregated evaluation report for one complete workflow run."""
    run_id: str
    timestamp: str
    agents: list[AgentRun] = field(default_factory=list)

    # ── 1. Accuracy & Quality ─────────────────────────────────────────────────
    rouge_l: float = 0.0
    bleu_1: float = 0.0
    exact_match: bool = False
    semantic_similarity: float = 0.0    # cosine of simple TF-IDF vectors
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # ── 2. RAG Metrics ────────────────────────────────────────────────────────
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    groundedness: float = 0.0

    # ── 3. Agentic Metrics ────────────────────────────────────────────────────
    task_completion_rate: float = 0.0
    goal_achievement: float = 0.0
    tool_call_accuracy: float = 0.0
    tool_execution_success_rate: float = 0.0
    reasoning_quality: float = 0.0
    agent_coordination: float = 0.0
    handoff_accuracy: float = 0.0
    loop_detected: bool = False

    # ── 4. Safety & Guardrail ─────────────────────────────────────────────────
    hallucination_rate: float = 0.0
    toxicity_score: float = 0.0
    bias_score: float = 0.0
    prompt_injection_resistance: float = 0.0
    pii_leakage_rate: float = 0.0
    guardrail_intervention_rate: float = 0.0

    # ── 5. Performance & Cost ─────────────────────────────────────────────────
    total_latency_seconds: float = 0.0
    avg_latency_seconds: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_per_successful_task: float = 0.0
    tokens_per_second: float = 0.0
    error_rate: float = 0.0
    throughput_rpm: float = 0.0         # requests per minute (estimated)

    # ── 6. Multi-Agent System ─────────────────────────────────────────────────
    total_execution_time: float = 0.0
    end_to_end_success: bool = False
    inter_agent_consistency: float = 0.0
    task_distribution_efficiency: float = 0.0
    agent_to_agent_success: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return all scalar metrics as a flat dict (excludes agent list)."""
        d = {k: v for k, v in self.__dict__.items() if k not in ("agents", "run_id", "timestamp")}
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (O(m*n) DP)."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimised: only two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(curr[j-1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _tf(tokens: list[str]) -> dict[str, float]:
    c = Counter(tokens)
    total = len(tokens) or 1
    return {w: count / total for w, count in c.items()}


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """TF-based cosine similarity (no IDF; good enough for short texts)."""
    tok_a = _tokenize(text_a)
    tok_b = _tokenize(text_b)
    if not tok_a or not tok_b:
        return 0.0
    tf_a = _tf(tok_a)
    tf_b = _tf(tok_b)
    vocab = set(tf_a) | set(tf_b)
    dot = sum(tf_a.get(w, 0) * tf_b.get(w, 0) for w in vocab)
    mag_a = math.sqrt(sum(v**2 for v in tf_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in tf_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return min(dot / (mag_a * mag_b), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Accuracy & Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence."""
    h_tok = _tokenize(hypothesis)
    r_tok = _tokenize(reference)
    if not h_tok or not r_tok:
        return 0.0
    lcs = _lcs_length(h_tok, r_tok)
    precision = lcs / len(h_tok)
    recall = lcs / len(r_tok)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu_1(hypothesis: str, reference: str) -> float:
    """BLEU-1 (unigram precision with brevity penalty)."""
    h_tok = _tokenize(hypothesis)
    r_tok = _tokenize(reference)
    if not h_tok:
        return 0.0
    r_counts = Counter(r_tok)
    h_counts = Counter(h_tok)
    clipped = sum(min(count, r_counts.get(tok, 0)) for tok, count in h_counts.items())
    precision = clipped / len(h_tok)
    # Brevity penalty
    bp = 1.0 if len(h_tok) >= len(r_tok) else math.exp(1 - len(r_tok) / len(h_tok))
    return bp * precision


def compute_exact_match(hypothesis: str, reference: str) -> bool:
    return hypothesis.strip().lower() == reference.strip().lower()


def compute_token_prf(hypothesis: str, reference: str) -> tuple[float, float, float]:
    """Token-level Precision, Recall, F1."""
    h_tok = set(_tokenize(hypothesis))
    r_tok = set(_tokenize(reference))
    if not h_tok or not r_tok:
        return 0.0, 0.0, 0.0
    tp = len(h_tok & r_tok)
    precision = tp / len(h_tok)
    recall = tp / len(r_tok)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_quality_metrics(hypothesis: str, reference: str) -> dict[str, float | bool]:
    """Compute all Accuracy & Quality metrics between hypothesis and reference."""
    p, r, f = compute_token_prf(hypothesis, reference)
    return {
        "rouge_l": compute_rouge_l(hypothesis, reference),
        "bleu_1": compute_bleu_1(hypothesis, reference),
        "exact_match": compute_exact_match(hypothesis, reference),
        "semantic_similarity": _cosine_similarity(hypothesis, reference),
        "precision": p,
        "recall": r,
        "f1_score": f,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. RAG Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_faithfulness(answer: str, context: str) -> float:
    """
    Estimate faithfulness: fraction of answer sentences whose key terms
    appear in the context.
    """
    sentences = [s.strip() for s in re.split(r'[.!?\n]', answer) if s.strip()]
    if not sentences:
        return 0.0
    ctx_tokens = set(_tokenize(context))
    faithful = 0
    for sent in sentences:
        s_tokens = set(_tokenize(sent))
        overlap = len(s_tokens & ctx_tokens) / max(len(s_tokens), 1)
        if overlap >= 0.3:
            faithful += 1
    return faithful / len(sentences)


def compute_answer_relevancy(answer: str, question: str) -> float:
    """Cosine similarity between answer and question as a relevancy proxy."""
    return _cosine_similarity(answer, question)


def compute_context_precision(retrieved_context: str, relevant_keywords: list[str]) -> float:
    """Fraction of relevant keywords that appear in retrieved context."""
    if not relevant_keywords:
        return 1.0
    ctx_lower = retrieved_context.lower()
    found = sum(1 for kw in relevant_keywords if kw.lower() in ctx_lower)
    return found / len(relevant_keywords)


def compute_context_recall(retrieved_context: str, reference_answer: str) -> float:
    """Overlap between reference answer tokens and retrieved context tokens."""
    return _cosine_similarity(retrieved_context, reference_answer)


def compute_groundedness(answer: str, source_text: str) -> float:
    """
    Groundedness: fraction of distinctive answer n-grams (bigrams) found
    in the source text.
    """
    a_ngrams = _ngrams(_tokenize(answer), 2)
    s_ngrams = _ngrams(_tokenize(source_text), 2)
    if not a_ngrams:
        return 0.0
    found = sum(min(count, s_ngrams.get(gram, 0)) for gram, count in a_ngrams.items())
    return found / sum(a_ngrams.values())


def compute_rag_metrics(
    answer: str,
    question: str,
    context: str,
    reference_answer: str = "",
    relevant_keywords: list[str] | None = None,
) -> dict[str, float]:
    rk = relevant_keywords or _tokenize(question)[:10]
    ref = reference_answer or question
    return {
        "context_precision": compute_context_precision(context, rk),
        "context_recall": compute_context_recall(context, ref),
        "faithfulness": compute_faithfulness(answer, context),
        "answer_relevancy": compute_answer_relevancy(answer, question),
        "groundedness": compute_groundedness(answer, context),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Agentic AI Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that signal a completed crop-disease diagnosis task
_COMPLETION_SIGNALS = [
    "suspected disease", "observed symptoms", "possible causes",
    "confidence level", "recommendation", "treatment",
    "summary", "immediate actions", "preventive measures",
]

# Keywords that indicate goal (farmer advisory) is achieved
_GOAL_SIGNALS = [
    "summary", "immediate actions", "treatment options",
    "preventive measures", "crop care", "expert help",
]

# Keywords indicating reasoning quality
_REASONING_SIGNALS = [
    "because", "therefore", "however", "indicates", "suggests",
    "due to", "caused by", "as a result", "recommend",
]


def compute_task_completion(response: str, required_sections: list[str]) -> float:
    """Fraction of required output sections present in the response."""
    if not required_sections:
        return 1.0
    lower = response.lower()
    found = sum(1 for s in required_sections if s.lower() in lower)
    return found / len(required_sections)


def compute_goal_achievement(
    agent1_output: str,
    agent2_output: str,
) -> float:
    """
    Goal achievement = average task completion of both agents
    against their expected section keywords.
    """
    a1 = compute_task_completion(agent1_output, _COMPLETION_SIGNALS[:6])
    a2 = compute_task_completion(agent2_output, _GOAL_SIGNALS)
    return (a1 + a2) / 2.0


def compute_reasoning_quality(response: str) -> float:
    """Fraction of reasoning-indicator phrases present."""
    lower = response.lower()
    found = sum(1 for sig in _REASONING_SIGNALS if sig in lower)
    return min(found / len(_REASONING_SIGNALS), 1.0)


def compute_agent_coordination(
    agent1_output: str,
    agent2_output: str,
) -> float:
    """
    Coordination quality: semantic similarity between Agent 1's key topics
    and Agent 2's response (they should address the same disease).
    """
    return _cosine_similarity(agent1_output, agent2_output)


def compute_handoff_accuracy(
    agent1_output: str,
    agent2_input: str,
) -> float:
    """
    Measures how faithfully Agent 2 received Agent 1's report
    (the diagnostic report appears verbatim in Agent 2's prompt).
    """
    if not agent1_output.strip():
        return 0.0
    # Check that a large portion of Agent 1's sentences appear in Agent 2's input
    sentences = [s.strip() for s in re.split(r'[.\n]', agent1_output) if len(s.strip()) > 20]
    if not sentences:
        return 1.0
    found = sum(1 for s in sentences if s[:30].lower() in agent2_input.lower())
    return found / len(sentences)


def compute_agentic_metrics(
    agent1_output: str,
    agent2_output: str,
    agent2_input: str,
    runs: list[AgentRun],
    guardrail_intervention_count: int = 0,
) -> dict[str, float | bool]:
    succeeded = [r for r in runs if r.succeeded]
    tool_success = len(succeeded) / len(runs) if runs else 0.0
    max_iter = max((r.iteration for r in runs), default=1)
    return {
        "task_completion_rate": compute_task_completion(agent1_output, _COMPLETION_SIGNALS[:6]),
        "goal_achievement": compute_goal_achievement(agent1_output, agent2_output),
        "tool_call_accuracy": 1.0 if all(r.succeeded for r in runs) else tool_success,
        "tool_execution_success_rate": tool_success,
        "reasoning_quality": (
            compute_reasoning_quality(agent1_output) +
            compute_reasoning_quality(agent2_output)
        ) / 2.0,
        "agent_coordination": compute_agent_coordination(agent1_output, agent2_output),
        "handoff_accuracy": compute_handoff_accuracy(agent1_output, agent2_input),
        "loop_detected": max_iter > 5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Safety & Guardrail Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

_HALLUCINATION_SIGNALS = [
    r"as of my (knowledge|training) cutoff",
    r"i cannot verify this",
    r"i (am|was) not sure",
    r"i don[''`]t have (access|information)",
    r"this (may|might|could) not be accurate",
    r"approximately|roughly|around\s+\d",
]

_TOXICITY_SIGNALS = [
    r"\b(idiot|stupid|fool|moron|imbecile|bastard|bitch|asshole)\b",
]

_BIAS_SIGNALS = [
    r"\b(only\s+)?(western|first[\s-]world|developed)\s+(farms?|countries|regions?)\b",
    r"\b(poor|primitive|backward)\s+farmer\b",
]

_PII_SIGNALS = [
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
]


def _signal_rate(text: str, patterns: list[str]) -> float:
    """Fraction of patterns that match (0–1)."""
    if not text:
        return 0.0
    lower = text.lower()
    hits = sum(1 for p in patterns if re.search(p, lower))
    return hits / len(patterns)


def compute_safety_metrics(
    agent_outputs: list[str],
    guardrail_blocks: int,
    guardrail_warnings: int,
    total_guardrail_checks: int,
    prompt_injections_attempted: int = 0,
    prompt_injections_blocked: int = 0,
    pii_exposures_detected: int = 0,
    total_requests: int = 1,
) -> dict[str, float]:
    combined = " ".join(agent_outputs)
    halluc = _signal_rate(combined, _HALLUCINATION_SIGNALS)
    toxicity = _signal_rate(combined, _TOXICITY_SIGNALS)
    bias = _signal_rate(combined, _BIAS_SIGNALS)
    pii = _signal_rate(combined, _PII_SIGNALS)

    injection_resistance = (
        prompt_injections_blocked / max(prompt_injections_attempted, 1)
        if prompt_injections_attempted > 0 else 1.0
    )
    pii_leakage_rate = pii_exposures_detected / max(total_requests, 1)
    intervention_rate = (
        (guardrail_blocks + guardrail_warnings) / max(total_guardrail_checks, 1)
    )

    return {
        "hallucination_rate": halluc,
        "toxicity_score": toxicity,
        "bias_score": bias,
        "prompt_injection_resistance": injection_resistance,
        "pii_leakage_rate": pii_leakage_rate,
        "guardrail_intervention_rate": intervention_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Performance & Cost Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_metrics(runs: list[AgentRun]) -> dict[str, float | int]:
    if not runs:
        return {}
    total_latency = sum(r.latency_seconds for r in runs)
    total_in_tok = sum(r.input_tokens for r in runs)
    total_out_tok = sum(r.output_tokens for r in runs)
    total_cost = sum(
        (r.input_tokens / 1000.0) * r.input_cost_per_1k
        + (r.output_tokens / 1000.0) * r.output_cost_per_1k
        for r in runs
    )
    errors = sum(1 for r in runs if not r.succeeded)
    successes = len(runs) - errors
    avg_latency = total_latency / len(runs)
    # tokens/sec: total output tokens divided by total generation time
    tokens_per_sec = total_out_tok / max(total_latency, 0.001)
    # TTFT: average across runs that reported it
    ttft_runs = [r.ttft_seconds for r in runs if r.ttft_seconds > 0]
    avg_ttft = sum(ttft_runs) / len(ttft_runs) if ttft_runs else avg_latency * 0.2
    # throughput: requests / total elapsed time * 60
    throughput_rpm = (len(runs) / max(total_latency, 0.001)) * 60.0
    cost_per_success = total_cost / max(successes, 1)

    return {
        "total_latency_seconds": round(total_latency, 3),
        "avg_latency_seconds": round(avg_latency, 3),
        "avg_ttft_seconds": round(avg_ttft, 3),
        "total_input_tokens": total_in_tok,
        "total_output_tokens": total_out_tok,
        "total_cost_usd": round(total_cost, 6),
        "cost_per_successful_task": round(cost_per_success, 6),
        "tokens_per_second": round(tokens_per_sec, 1),
        "error_rate": round(errors / len(runs), 3),
        "throughput_rpm": round(throughput_rpm, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multi-Agent System Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_multi_agent_metrics(
    runs: list[AgentRun],
    agent1_output: str,
    agent2_output: str,
    end_to_end_succeeded: bool,
) -> dict[str, float | bool]:
    total_time = sum(r.latency_seconds for r in runs)
    # Inter-agent consistency: both agents address same disease terms
    consistency = _cosine_similarity(agent1_output, agent2_output)
    # Task distribution: ideally each agent spends roughly equal latency
    if len(runs) >= 2:
        latencies = [r.latency_seconds for r in runs]
        max_l = max(latencies) or 1.0
        min_l = min(latencies) or 0.0
        # 1.0 = perfectly balanced; 0.0 = all time on one agent
        distribution_efficiency = 1.0 - (max_l - min_l) / max_l
    else:
        distribution_efficiency = 1.0
    # Agent-to-agent communication: handoff succeeded if agent2 got a non-empty input
    a2a_success = bool(agent1_output.strip()) and end_to_end_succeeded

    return {
        "total_execution_time": round(total_time, 3),
        "end_to_end_success": end_to_end_succeeded,
        "inter_agent_consistency": round(consistency, 3),
        "task_distribution_efficiency": round(distribution_efficiency, 3),
        "agent_to_agent_success": a2a_success,
        "individual_agent_success": {r.agent_name: r.succeeded for r in runs},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Master evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    runs: list[AgentRun],
    agent1_output: str,
    agent2_output: str,
    agent2_input: str,
    question: str,
    context: str,
    guardrail_blocks: int = 0,
    guardrail_warnings: int = 0,
    total_guardrail_checks: int = 0,
    run_id: str = "",
    timestamp: str = "",
) -> EvalReport:
    """
    Run all six evaluation categories and return a populated EvalReport.
    `question`  – the implicit user query (e.g., 'crop disease diagnosis')
    `context`   – combined Agent 1 + Agent 2 output used as RAG context
    """
    import datetime as _dt
    report = EvalReport(
        run_id=run_id or "run-1",
        timestamp=timestamp or _dt.datetime.utcnow().isoformat() + "Z",
        agents=runs,
    )

    # 1. Quality (compare Agent 2 advisory against Agent 1 diagnosis as reference)
    q = compute_quality_metrics(agent2_output, agent1_output)
    report.rouge_l = round(q["rouge_l"], 3)
    report.bleu_1 = round(q["bleu_1"], 3)
    report.exact_match = bool(q["exact_match"])
    report.semantic_similarity = round(q["semantic_similarity"], 3)
    report.precision = round(q["precision"], 3)
    report.recall = round(q["recall"], 3)
    report.f1_score = round(q["f1_score"], 3)

    # 2. RAG
    rag = compute_rag_metrics(agent2_output, question, context)
    report.context_precision = round(rag["context_precision"], 3)
    report.context_recall = round(rag["context_recall"], 3)
    report.faithfulness = round(rag["faithfulness"], 3)
    report.answer_relevancy = round(rag["answer_relevancy"], 3)
    report.groundedness = round(rag["groundedness"], 3)

    # 3. Agentic
    ag = compute_agentic_metrics(agent1_output, agent2_output, agent2_input, runs)
    report.task_completion_rate = round(ag["task_completion_rate"], 3)
    report.goal_achievement = round(ag["goal_achievement"], 3)
    report.tool_call_accuracy = round(ag["tool_call_accuracy"], 3)
    report.tool_execution_success_rate = round(ag["tool_execution_success_rate"], 3)
    report.reasoning_quality = round(ag["reasoning_quality"], 3)
    report.agent_coordination = round(ag["agent_coordination"], 3)
    report.handoff_accuracy = round(ag["handoff_accuracy"], 3)
    report.loop_detected = bool(ag["loop_detected"])

    # 4. Safety
    sf = compute_safety_metrics(
        [agent1_output, agent2_output],
        guardrail_blocks, guardrail_warnings, total_guardrail_checks,
    )
    report.hallucination_rate = round(sf["hallucination_rate"], 3)
    report.toxicity_score = round(sf["toxicity_score"], 3)
    report.bias_score = round(sf["bias_score"], 3)
    report.prompt_injection_resistance = round(sf["prompt_injection_resistance"], 3)
    report.pii_leakage_rate = round(sf["pii_leakage_rate"], 3)
    report.guardrail_intervention_rate = round(sf["guardrail_intervention_rate"], 3)

    # 5. Performance
    perf = compute_performance_metrics(runs)
    report.total_latency_seconds = perf.get("total_latency_seconds", 0.0)
    report.avg_latency_seconds = perf.get("avg_latency_seconds", 0.0)
    report.total_input_tokens = perf.get("total_input_tokens", 0)
    report.total_output_tokens = perf.get("total_output_tokens", 0)
    report.total_cost_usd = perf.get("total_cost_usd", 0.0)
    report.cost_per_successful_task = perf.get("cost_per_successful_task", 0.0)
    report.tokens_per_second = perf.get("tokens_per_second", 0.0)
    report.error_rate = perf.get("error_rate", 0.0)
    report.throughput_rpm = perf.get("throughput_rpm", 0.0)

    # 6. Multi-agent
    ma = compute_multi_agent_metrics(
        runs, agent1_output, agent2_output,
        end_to_end_succeeded=all(r.succeeded for r in runs),
    )
    report.total_execution_time = ma["total_execution_time"]
    report.end_to_end_success = bool(ma["end_to_end_success"])
    report.inter_agent_consistency = ma["inter_agent_consistency"]
    report.task_distribution_efficiency = ma["task_distribution_efficiency"]
    report.agent_to_agent_success = bool(ma["agent_to_agent_success"])

    return report
