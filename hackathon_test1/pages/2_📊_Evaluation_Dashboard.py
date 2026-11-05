"""
dashboard.py – Agri Doctor LLM & Agent Performance Dashboard
=============================================================
Standalone Streamlit page. Run with:
    streamlit run dashboard.py

Or navigate to it from app.py via the sidebar link.
Reads the evaluation report stored in st.session_state by app.py.
Also provides a demo mode with synthetic data so the dashboard can be
explored without running a live diagnosis first.
"""

import math
import random

import streamlit as st

from evaluator import AgentRun, EvalReport, evaluate

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agri Doctor – Evaluation Dashboard 📊",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Colour helpers ───────────────────────────────────────────────────────────

def _colour(value: float, low: float = 0.4, high: float = 0.7) -> str:
    """Return green / orange / red based on thresholds."""
    if value >= high:
        return "🟢"
    if value >= low:
        return "🟡"
    return "🔴"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _delta_colour(value: float, good_high: bool = True) -> str:
    """Positive delta label for st.metric."""
    if good_high:
        return "normal" if value >= 0.6 else "inverse"
    return "inverse" if value >= 0.1 else "normal"


# ─── Demo data generator ──────────────────────────────────────────────────────

def _demo_report() -> EvalReport:
    """Generate a realistic synthetic evaluation report for demo mode."""
    random.seed(42)

    def jitter(base: float, spread: float = 0.05) -> float:
        return max(0.0, min(1.0, base + random.uniform(-spread, spread)))

    runs = [
        AgentRun(
            agent_name="Pathologist Agent",
            model_id="mistralai/mistral-small-3-1-24b-instruct-2503",
            input_tokens=520,
            output_tokens=410,
            latency_seconds=3.8,
            ttft_seconds=0.72,
            succeeded=True,
        ),
        AgentRun(
            agent_name="Extension Officer Agent",
            model_id="ibm/granite-4-h-small",
            input_tokens=780,
            output_tokens=620,
            latency_seconds=4.2,
            ttft_seconds=0.65,
            succeeded=True,
        ),
    ]

    agent1_output = (
        "1. SUSPECTED DISEASE: Late Blight (Phytophthora infestans)\n"
        "2. OBSERVED SYMPTOMS: Brown water-soaked lesions on leaf edges with white mould "
        "underside. Rapid spreading necrosis visible.\n"
        "3. POSSIBLE CAUSES: Fungal pathogen Phytophthora infestans in humid conditions.\n"
        "4. CONFIDENCE LEVEL: High\n"
        "5. ALTERNATIVE DIAGNOSES: Early Blight (Alternaria solani)\n"
        "6. RECOMMENDATION: Consult a plant pathologist for laboratory confirmation."
    )
    agent2_output = (
        "1. SUMMARY: Your crop appears to have Late Blight, a serious fungal disease.\n"
        "2. IMMEDIATE ACTIONS: Remove and destroy infected leaves. Improve drainage.\n"
        "3. TREATMENT OPTIONS: Apply copper-based fungicide per product label instructions.\n"
        "4. PREVENTIVE MEASURES: Rotate crops, avoid overhead irrigation.\n"
        "5. CROP CARE TIPS: Ensure adequate spacing for air circulation.\n"
        "6. WHEN TO SEEK EXPERT HELP: Contact extension officer if disease spreads."
    )

    report = evaluate(
        runs=runs,
        agent1_output=agent1_output,
        agent2_output=agent2_output,
        agent2_input="PATHOLOGIST REPORT: " + agent1_output,
        question="crop disease diagnosis late blight tomato",
        context=agent1_output + " " + agent2_output,
        guardrail_blocks=0,
        guardrail_warnings=2,
        total_guardrail_checks=34,
        run_id="demo-001",
    )
    return report


# ─── Load report ──────────────────────────────────────────────────────────────

st.title("📊 Agri Doctor – LLM & Agent Performance Dashboard")
st.markdown("Comprehensive evaluation of the two-agent crop disease advisor.")
st.divider()

# Check if a real report exists in session state
report: EvalReport | None = st.session_state.get("eval_report", None)
runs: list[AgentRun] = st.session_state.get("agent_runs", [])

if report is None:
    st.info(
        "ℹ️ No live evaluation data found. **Demo mode** is active — "
        "synthetic metrics are shown below. Run a diagnosis in the main app "
        "to see real metrics here."
    )
    report = _demo_report()
    runs = report.agents

# ─── Top KPI strip ────────────────────────────────────────────────────────────

st.subheader("🏆 Key Performance Indicators")

kpi_cols = st.columns(6)
kpi_cols[0].metric(
    "Goal Achievement", _pct(report.goal_achievement),
    delta=None, delta_color=_delta_colour(report.goal_achievement),
    help="How well both agents achieved their combined objective."
)
kpi_cols[1].metric(
    "Task Completion", _pct(report.task_completion_rate),
    help="Fraction of required report sections produced."
)
kpi_cols[2].metric(
    "Faithfulness", _pct(report.faithfulness),
    help="Agent 2 advisory supported by Agent 1's findings."
)
kpi_cols[3].metric(
    "Avg Latency", f"{report.avg_latency_seconds:.2f}s",
    help="Average response time per agent."
)
kpi_cols[4].metric(
    "Total Cost", f"${report.total_cost_usd:.5f}",
    help="Estimated USD cost for this diagnosis run."
)
kpi_cols[5].metric(
    "Guardrail Rate", _pct(report.guardrail_intervention_rate),
    delta=None,
    delta_color="inverse" if report.guardrail_intervention_rate > 0.2 else "normal",
    help="Fraction of guardrail checks that triggered a warning or block."
)

st.divider()

# ─── Six metric categories as tabs ───────────────────────────────────────────

tabs = st.tabs([
    "📐 Accuracy & Quality",
    "🔍 RAG Evaluation",
    "🤖 Agentic AI",
    "🛡️ Safety & Guardrails",
    "⚡ Performance & Cost",
    "🔗 Multi-Agent System",
])

# ── Tab 1: Accuracy & Quality ─────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Accuracy & Quality Metrics")
    st.caption(
        "Measures text quality of Agent 2's advisory relative to Agent 1's diagnosis."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROUGE-L", _pct(report.rouge_l),
              help="Longest common subsequence overlap F1.")
    c2.metric("BLEU-1", _pct(report.bleu_1),
              help="Unigram precision with brevity penalty.")
    c3.metric("Semantic Similarity", _pct(report.semantic_similarity),
              help="TF-based cosine similarity between agent outputs.")
    c4.metric("Exact Match", "✅ Yes" if report.exact_match else "❌ No",
              help="Whether outputs are character-identical.")

    st.markdown("---")
    c5, c6, c7 = st.columns(3)
    c5.metric("Precision", _pct(report.precision),
              help="Token-level: how much of Agent 2's vocabulary matches Agent 1's.")
    c6.metric("Recall", _pct(report.recall),
              help="Token-level: how much of Agent 1's vocabulary is covered.")
    c7.metric("F1 Score", _pct(report.f1_score),
              help="Harmonic mean of precision and recall.")

    st.markdown("#### Score Overview")
    metrics_q = {
        "ROUGE-L": report.rouge_l,
        "BLEU-1": report.bleu_1,
        "Semantic Similarity": report.semantic_similarity,
        "Precision": report.precision,
        "Recall": report.recall,
        "F1 Score": report.f1_score,
    }
    for name, val in metrics_q.items():
        col_a, col_b = st.columns([3, 7])
        col_a.markdown(f"{_colour(val)} **{name}**")
        col_b.progress(val, text=_pct(val))

# ── Tab 2: RAG Evaluation ─────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("RAG Evaluation Metrics")
    st.caption(
        "Evaluates how well the generated advisory is supported by and "
        "relevant to the retrieved diagnostic context."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Context Precision", _pct(report.context_precision),
              help="Fraction of disease keywords present in retrieved context.")
    c2.metric("Context Recall", _pct(report.context_recall),
              help="How much of the reference answer is captured in context.")
    c3.metric("Faithfulness", _pct(report.faithfulness),
              help="Fraction of advisory sentences supported by the diagnosis.")

    c4, c5 = st.columns(2)
    c4.metric("Answer Relevancy", _pct(report.answer_relevancy),
              help="Semantic match between advisory and the user's query.")
    c5.metric("Groundedness", _pct(report.groundedness),
              help="Bigram overlap between advisory and source diagnosis.")

    st.markdown("#### Score Overview")
    rag_metrics = {
        "Context Precision": report.context_precision,
        "Context Recall": report.context_recall,
        "Faithfulness": report.faithfulness,
        "Answer Relevancy": report.answer_relevancy,
        "Groundedness": report.groundedness,
    }
    for name, val in rag_metrics.items():
        col_a, col_b = st.columns([3, 7])
        col_a.markdown(f"{_colour(val)} **{name}**")
        col_b.progress(val, text=_pct(val))

# ── Tab 3: Agentic AI ─────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Agentic AI Evaluation Metrics")
    st.caption("Evaluates individual agent behaviour and multi-agent coordination.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Task Completion", _pct(report.task_completion_rate))
    c2.metric("Goal Achievement", _pct(report.goal_achievement))
    c3.metric("Tool Call Accuracy", _pct(report.tool_call_accuracy))
    c4.metric("Tool Exec Success", _pct(report.tool_execution_success_rate))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Reasoning Quality", _pct(report.reasoning_quality))
    c6.metric("Agent Coordination", _pct(report.agent_coordination))
    c7.metric("Handoff Accuracy", _pct(report.handoff_accuracy))
    c8.metric("Loop Detected", "⚠️ Yes" if report.loop_detected else "✅ No")

    st.markdown("#### Per-Agent Performance")
    if runs:
        for run in runs:
            with st.expander(f"🤖 {run.agent_name} — {run.model_id}", expanded=True):
                ac1, ac2, ac3, ac4, ac5 = st.columns(5)
                ac1.metric("Status", "✅ OK" if run.succeeded else "❌ Failed")
                ac2.metric("Latency", f"{run.latency_seconds:.2f}s")
                ac3.metric("TTFT", f"{run.ttft_seconds:.2f}s")
                ac4.metric("Input Tokens", run.input_tokens)
                ac5.metric("Output Tokens", run.output_tokens)
                if not run.succeeded:
                    st.error(f"Error: {run.error_message}")

    st.markdown("#### Score Overview")
    agentic_scores = {
        "Task Completion Rate": report.task_completion_rate,
        "Goal Achievement": report.goal_achievement,
        "Tool Call Accuracy": report.tool_call_accuracy,
        "Tool Exec Success Rate": report.tool_execution_success_rate,
        "Reasoning Quality": report.reasoning_quality,
        "Agent Coordination": report.agent_coordination,
        "Handoff Accuracy": report.handoff_accuracy,
    }
    for name, val in agentic_scores.items():
        col_a, col_b = st.columns([3, 7])
        col_a.markdown(f"{_colour(val)} **{name}**")
        col_b.progress(val, text=_pct(val))

# ── Tab 4: Safety & Guardrails ────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Safety & Guardrail Evaluation Metrics")
    st.caption(
        "Measures safety of AI outputs. Lower is better for most metrics here "
        "(except injection resistance)."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Hallucination Rate", _pct(report.hallucination_rate),
        delta_color="inverse",
        help="Fraction of known hallucination-signal phrases detected."
    )
    c2.metric("Toxicity Score", _pct(report.toxicity_score),
              delta_color="inverse")
    c3.metric("Bias Score", _pct(report.bias_score),
              delta_color="inverse")

    c4, c5, c6 = st.columns(3)
    c4.metric(
        "Injection Resistance", _pct(report.prompt_injection_resistance),
        help="1.0 = all injections blocked."
    )
    c5.metric("PII Leakage Rate", _pct(report.pii_leakage_rate),
              delta_color="inverse")
    c6.metric(
        "Guardrail Intervention Rate", _pct(report.guardrail_intervention_rate),
        help="Fraction of checks that resulted in a warning or block."
    )

    st.markdown("#### Safety Scorecard")
    safety_items = [
        ("Injection Resistance", report.prompt_injection_resistance, True),
        ("Hallucination Rate", 1.0 - report.hallucination_rate, True),  # invert
        ("Toxicity Score", 1.0 - report.toxicity_score, True),
        ("Bias Score", 1.0 - report.bias_score, True),
        ("PII Leakage Rate", 1.0 - report.pii_leakage_rate, True),
    ]
    for name, val, _good in safety_items:
        col_a, col_b = st.columns([3, 7])
        col_a.markdown(f"{_colour(val)} **{name}**")
        col_b.progress(val, text=_pct(val) + " safe")

# ── Tab 5: Performance & Cost ─────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Performance & Cost Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Latency", f"{report.total_latency_seconds:.2f}s")
    c2.metric("Avg Latency", f"{report.avg_latency_seconds:.2f}s")
    c3.metric("Tokens / Second", f"{report.tokens_per_second:.0f}")
    c4.metric("Error Rate", _pct(report.error_rate),
              delta_color="inverse")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Input Tokens", f"{report.total_input_tokens:,}")
    c6.metric("Output Tokens", f"{report.total_output_tokens:,}")
    c7.metric("Est. Cost (USD)", f"${report.total_cost_usd:.5f}")
    c8.metric("Throughput (RPM)", f"{report.throughput_rpm:.1f}")

    st.markdown("#### Per-Agent Token Usage")
    if runs:
        for run in runs:
            st.markdown(f"**{run.agent_name}**")
            tok_cols = st.columns(4)
            tok_cols[0].metric("Input Tokens", run.input_tokens)
            tok_cols[1].metric("Output Tokens", run.output_tokens)
            tok_cols[2].metric("Latency", f"{run.latency_seconds:.2f}s")
            tok_cols[3].metric(
                "Est. Cost",
                f"${(run.input_tokens/1000*run.input_cost_per_1k + run.output_tokens/1000*run.output_cost_per_1k):.5f}"
            )

    st.markdown("#### Token Distribution")
    if runs:
        total_in = sum(r.input_tokens for r in runs)
        total_out = sum(r.output_tokens for r in runs)
        if total_in + total_out > 0:
            in_frac = total_in / (total_in + total_out)
            out_frac = total_out / (total_in + total_out)
            st.markdown(
                f"📥 **Input tokens** ({total_in:,}) — "
                f"{_pct(in_frac)} of total  |  "
                f"📤 **Output tokens** ({total_out:,}) — {_pct(out_frac)} of total"
            )

# ── Tab 6: Multi-Agent System ─────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Multi-Agent System Evaluation")
    st.caption("End-to-end workflow health and agent collaboration metrics.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Exec Time", f"{report.total_execution_time:.2f}s")
    c2.metric("E2E Success", "✅ Yes" if report.end_to_end_success else "❌ No")
    c3.metric("A2A Success", "✅ Yes" if report.agent_to_agent_success else "❌ No")

    c4, c5 = st.columns(2)
    c4.metric("Inter-Agent Consistency", _pct(report.inter_agent_consistency),
              help="Semantic similarity between agent outputs – both should address same disease.")
    c5.metric("Task Distribution Efficiency", _pct(report.task_distribution_efficiency),
              help="How evenly workload is split between agents (1.0 = perfectly balanced).")

    st.markdown("#### Multi-Agent Score Overview")
    ma_scores = {
        "Inter-Agent Consistency": report.inter_agent_consistency,
        "Task Distribution Efficiency": report.task_distribution_efficiency,
    }
    for name, val in ma_scores.items():
        col_a, col_b = st.columns([3, 7])
        col_a.markdown(f"{_colour(val)} **{name}**")
        col_b.progress(val, text=_pct(val))

    st.markdown("#### Workflow Timeline")
    if runs:
        total_t = sum(r.latency_seconds for r in runs) or 1.0
        for i, run in enumerate(runs):
            frac = run.latency_seconds / total_t
            st.markdown(f"**Agent {i+1}: {run.agent_name}**")
            st.progress(frac, text=f"{run.latency_seconds:.2f}s ({_pct(frac)} of total)")

    st.markdown("#### Agent Communication Flow")
    st.markdown(
        "```\n"
        "Farmer uploads image\n"
        "        │\n"
        "        ▼\n"
        "┌─────────────────────┐\n"
        "│  Pathologist Agent  │  ← Vision LLM (image_chat)\n"
        "│  (Agent 1)          │\n"
        "└────────┬────────────┘\n"
        "         │ Diagnostic Report (text handoff)\n"
        "         ▼\n"
        "┌──────────────────────────┐\n"
        "│  Extension Officer Agent │  ← Text LLM (text_chat)\n"
        "│  (Agent 2)               │\n"
        "└──────────────────────────┘\n"
        "         │\n"
        "         ▼\n"
        "  Farmer-Friendly Advisory\n"
        "```"
    )

st.divider()

# ─── Full metrics table ───────────────────────────────────────────────────────

with st.expander("📋 Full Metrics Table", expanded=False):
    all_metrics = report.as_dict()
    rows = []
    for key, val in all_metrics.items():
        if isinstance(val, float):
            display = f"{val:.3f}"
            if 0 <= val <= 1:
                display += f" ({_pct(val)})"
        elif isinstance(val, bool):
            display = "Yes" if val else "No"
        elif isinstance(val, int):
            display = f"{val:,}"
        else:
            display = str(val)
        rows.append({"Metric": key.replace("_", " ").title(), "Value": display})

    # Render as a simple markdown table
    header = "| Metric | Value |\n|---|---|\n"
    body = "\n".join(f"| {r['Metric']} | {r['Value']} |" for r in rows)
    st.markdown(header + body)

# ─── Download ─────────────────────────────────────────────────────────────────

import datetime as _dt, json

report_dict = {
    "run_id": report.run_id,
    "timestamp": report.timestamp,
    "metrics": report.as_dict(),
    "agents": [
        {
            "name": r.agent_name,
            "model": r.model_id,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "latency_s": r.latency_seconds,
            "succeeded": r.succeeded,
        }
        for r in runs
    ],
}

st.download_button(
    label="⬇️ Download Evaluation Report (JSON)",
    data=json.dumps(report_dict, indent=2),
    file_name=f"agri_doctor_eval_{_dt.datetime.now().strftime('%Y%m%d_%H%M')}.json",
    mime="application/json",
)

st.divider()
st.caption(
    "Agri Doctor v2.0 · Evaluation Dashboard · "
    "Powered by IBM watsonx Granite · For educational purposes only."
)
