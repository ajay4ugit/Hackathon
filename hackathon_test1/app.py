"""
Agri Doctor – Two-Agent Crop Disease Advisor
=============================================
Agent 1 – Pathologist Agent  : vision LLM analyses the uploaded crop image and
                                produces a structured disease-diagnosis report.
Agent 2 – Extension Officer Agent : receives the Pathologist's report and
                                generates a farmer-friendly remedial advisory.

Credentials are loaded from a .env file in the project root so no keys are
ever hard-coded or visible in the source code.

All nine guardrail categories are enforced throughout the workflow:
  1. Input          5. Security
  2. Output         6. Operational
  3. Tool           7. Human-in-the-Loop
  4. Data           8. Multi-Agent
                    9. Ethical & Compliance
"""

import base64
import datetime
import os
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

from guardrails import (
    AuditEntry,
    GuardResult,
    get_audit_log,
    mask_pii,
    run_all_agent1_output_checks,
    run_all_agent2_output_checks,
    run_all_input_checks,
)
from evaluator import AgentRun, evaluate

# ─── Load environment variables from .env ─────────────────────────────────────
load_dotenv()

# ─── Default constants (overridden by .env / sidebar) ────────────────────────

IBM_ENDPOINT = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# Agent 1: vision (image_chat) model – verified available on us-south multitenant.
# Options: mistralai/mistral-small-3-1-24b-instruct-2503 (default, lowest cost)
#          mistralai/mistral-medium-2505
#          meta-llama/llama-4-maverick-17b-128e-instruct-fp8
VISION_MODEL_ID = os.getenv(
    "WATSONX_VISION_MODEL_ID", "mistralai/mistral-small-3-1-24b-instruct-2503"
)

# Agent 2: IBM Granite text_chat model
TEXT_MODEL_ID = os.getenv("WATSONX_TEXT_MODEL_ID", "ibm/granite-4-h-small")

DISCLAIMER = (
    "⚠️ **AI-Generated Preliminary Assessment** – This diagnosis is produced by an "
    "AI system and is NOT a confirmed laboratory diagnosis. Always verify findings "
    "with a qualified plant pathologist or your local agricultural extension officer "
    "before applying any treatment."
)

PESTICIDE_WARNING = (
    "🛡️ **Safety Warning** – When handling pesticides or agrochemicals: always read "
    "and follow the product label; wear appropriate personal protective equipment "
    "(gloves, mask, goggles); observe re-entry intervals; dispose of containers "
    "responsibly; and comply with local agricultural regulations."
)

# ─── Page configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agri Doctor 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session ID for rate-limiting (one per browser session) ──────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ─── Sidebar – credentials & settings ────────────────────────────────────────

st.sidebar.title("🌿 Agri Doctor")
st.sidebar.markdown("*AI-Powered Crop Disease Advisor*")
st.sidebar.divider()

st.sidebar.subheader("🔑 IBM watsonx Credentials")
ibm_api_key = st.sidebar.text_input(
    "IBM API Key",
    type="password",
    value=os.getenv("WATSONX_API_KEY", ""),
    placeholder="Enter your IBM Cloud API key",
    help="Loaded from WATSONX_API_KEY in .env — never hard-coded or logged.",
)
project_id = st.sidebar.text_input(
    "Project ID",
    value=os.getenv("WATSONX_PROJECT_ID", ""),
    placeholder="Enter your watsonx.ai Project ID",
    help="Loaded from WATSONX_PROJECT_ID in .env.",
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Model Settings")
vision_model_id = st.sidebar.text_input(
    "Vision Model ID (Agent 1 – Pathologist)",
    value=VISION_MODEL_ID,
    help=(
        "Must support image_chat. Verified on us-south multitenant (Nov 2026):\n"
        "• mistralai/mistral-small-3-1-24b-instruct-2503 (default)\n"
        "• mistralai/mistral-medium-2505\n"
        "• meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    ),
)
text_model_id = st.sidebar.text_input(
    "Text Model ID (Agent 2 – Extension Officer)",
    value=TEXT_MODEL_ID,
    help="Text instruct IBM watsonx model used to generate the farmer advisory.",
)
max_tokens = st.sidebar.slider(
    "Max tokens per response", min_value=256, max_value=2048, value=1024, step=128
)

st.sidebar.divider()
st.sidebar.info(
    "**How it works**\n\n"
    "1. Upload a clear crop / leaf image.\n"
    "2. Click **Diagnose Crop**.\n"
    "3. The **Pathologist Agent** analyses the image.\n"
    "4. The **Extension Officer Agent** prepares your advisory.\n"
    "5. Download the full report."
)

st.sidebar.divider()
st.sidebar.markdown("### 📊 Evaluation Dashboard")
st.sidebar.markdown(
    "After running a diagnosis, open the evaluation dashboard to see "
    "detailed LLM performance metrics."
)
if st.sidebar.button("Open Evaluation Dashboard →", use_container_width=True):
    st.switch_page("pages/2_📊_Evaluation_Dashboard.py")

# ─── Helper: render guardrail audit panel ────────────────────────────────────


def render_guardrail_panel(entry: AuditEntry, label: str) -> None:
    """Display a collapsible guardrail audit panel in the UI."""
    blocked = entry.blocked()
    warnings = entry.warnings()

    icon = "🔴" if blocked else ("🟡" if warnings else "🟢")
    status = "BLOCKED" if blocked else (f"{len(warnings)} warning(s)" if warnings else "All passed")

    with st.expander(f"{icon} Guardrail Audit – {label} ({status})", expanded=blocked):
        cols = st.columns(4)
        cols[0].metric("Total Checks", len(entry.results))
        cols[1].metric("Passed", sum(1 for r in entry.results if r.passed))
        cols[2].metric("Warnings", len(warnings))
        cols[3].metric("Blocked", sum(1 for r in entry.results if r.severity == "BLOCK" and not r.passed))

        st.markdown("**Check Details:**")
        for result in entry.results:
            if result.passed:
                st.markdown(f"✅ `{result.category}/{result.check}`")
            elif result.severity == "WARNING":
                st.warning(f"⚠️ **{result.category}/{result.check}**: {result.message}")
            else:
                st.error(f"🚫 **{result.category}/{result.check}**: {result.message}")


# ─── Helper functions ─────────────────────────────────────────────────────────


def encode_image_base64(image_bytes: bytes) -> str:
    """Return a base64-encoded string of the image bytes."""
    return base64.b64encode(image_bytes).decode("utf-8")


def build_model(api_key: str, proj_id: str, mid: str) -> ModelInference:
    """Instantiate a watsonx ModelInference object."""
    credentials = Credentials(url=IBM_ENDPOINT, api_key=api_key)
    model = ModelInference(
        model_id=mid,
        credentials=credentials,
        project_id=proj_id,
        validate=False,
    )
    return model


def run_pathologist_agent(
    model: ModelInference, image_bytes: bytes, mime_type: str, max_tok: int
) -> str:
    """
    Agent 1 – Pathologist Agent.
    Sends the crop image to the vision LLM and returns a structured
    diagnostic report.
    """
    b64_image = encode_image_base64(image_bytes)

    system_prompt = (
        "You are an expert plant pathologist with decades of experience diagnosing "
        "crop and plant diseases from leaf and crop images. "
        "Analyse the provided image carefully and produce a structured diagnostic report. "
        "If the image quality is too poor to make any reliable observation, explicitly "
        "state that a clearer image is required instead of guessing. "
        "Structure your report with the following clearly labelled sections:\n"
        "1. SUSPECTED DISEASE\n"
        "2. OBSERVED SYMPTOMS\n"
        "3. POSSIBLE CAUSES\n"
        "4. CONFIDENCE LEVEL (Low / Medium / High)\n"
        "5. ALTERNATIVE DIAGNOSES (if any)\n"
        "6. RECOMMENDATION (whether a clearer image or lab confirmation is needed)\n\n"
        "Be precise, factual, and avoid speculation beyond what the image supports. "
        "Always recommend consulting a qualified expert for confirmation."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please examine this crop/plant image and provide a detailed "
                        "disease diagnostic report following the structured format "
                        "described. Focus only on what is clearly visible in the image."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                },
            ],
        },
    ]

    chat_params = TextChatParameters(max_tokens=max_tok, temperature=0.2)
    response = model.chat(messages=messages, params=chat_params)
    return response["choices"][0]["message"]["content"]


def run_extension_officer_agent(
    model: ModelInference, pathologist_report: str, max_tok: int
) -> str:
    """
    Agent 2 – Extension Officer Agent.
    Receives the Pathologist's diagnostic report and generates a
    farmer-friendly advisory with remedial actions.
    """
    system_prompt = (
        "You are a compassionate and knowledgeable agricultural extension officer "
        "advising smallholder farmers. You receive a plant pathologist's diagnostic "
        "report and translate it into simple, practical, farmer-friendly advice. "
        "Write in plain language that a farmer with no technical background can understand. "
        "Structure your advisory with the following clearly labelled sections:\n"
        "1. SUMMARY (one or two plain-language sentences about what is wrong)\n"
        "2. IMMEDIATE ACTIONS (what the farmer should do right now)\n"
        "3. TREATMENT OPTIONS (biological, cultural, and chemical – only recommend "
        "   specific chemicals if genuinely appropriate; do NOT invent dosages; "
        "   always refer to the product label)\n"
        "4. PREVENTIVE MEASURES (how to stop this from recurring)\n"
        "5. CROP CARE TIPS (general good practices relevant to this crop)\n"
        "6. WHEN TO SEEK EXPERT HELP (clear guidance on escalation)\n\n"
        "Never fabricate pesticide brand names, registration numbers, or dosages. "
        "Always advise following local regulations and safety precautions. "
        "This is an AI-generated preliminary assessment, not a confirmed laboratory diagnosis."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Below is the plant pathologist's diagnostic report. "
                "Please prepare a practical farmer advisory based on these findings.\n\n"
                "--- PATHOLOGIST'S DIAGNOSTIC REPORT ---\n"
                f"{pathologist_report}\n"
                "--- END OF REPORT ---\n\n"
                "Now write the farmer advisory in simple, encouraging language."
            ),
        },
    ]

    chat_params = TextChatParameters(max_tokens=max_tok, temperature=0.3)
    response = model.chat(messages=messages, params=chat_params)
    return response["choices"][0]["message"]["content"]


def build_final_report(
    pathologist_report: str,
    extension_advisory: str,
    image_filename: str,
    audit_log: list[AuditEntry],
) -> str:
    """Assemble the downloadable plain-text final report including audit trail."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "=" * 70,
        "           AGRI DOCTOR – CROP DISEASE ADVISORY REPORT",
        "=" * 70,
        f"Generated : {now}",
        f"Image file: {image_filename}",
        "",
        "DISCLAIMER:",
        (
            "This report is an AI-generated preliminary assessment and is NOT a "
            "confirmed laboratory diagnosis. Verify findings with a qualified plant "
            "pathologist or your local agricultural extension officer before applying "
            "any treatment."
        ),
        "",
        "=" * 70,
        "SECTION 1 – PATHOLOGIST AGENT: DISEASE DIAGNOSIS",
        "=" * 70,
        "",
        pathologist_report,
        "",
        "=" * 70,
        "SECTION 2 – EXTENSION OFFICER AGENT: FARMER ADVISORY",
        "=" * 70,
        "",
        extension_advisory,
        "",
        "=" * 70,
        "SAFETY REMINDER:",
        (
            "When handling pesticides: wear gloves, mask and goggles; read the "
            "product label; observe re-entry intervals; dispose of containers "
            "safely; comply with local regulations."
        ),
        "",
        "=" * 70,
        "GUARDRAIL AUDIT TRAIL",
        "=" * 70,
    ]
    for entry in audit_log:
        s = entry.summary()
        lines.append(
            f"[{s['timestamp']}] {s['agent'].upper()} | {s['stage']} | "
            f"checks={s['checks']} warnings={s['warnings']} blocked={s['blocked']}"
        )
    lines += [
        "",
        "=" * 70,
        "Report produced by Agri Doctor | Powered by IBM watsonx Granite",
    ]
    return "\n".join(lines)


# ─── Main page ────────────────────────────────────────────────────────────────

st.title("🌿 Agri Doctor")
st.subheader("AI-Powered Crop Disease Diagnosis & Advisory")
st.markdown(DISCLAIMER)
st.divider()

# Image uploader
st.subheader("📷 Upload Crop / Plant Leaf Image")
uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG, or PNG image of the affected crop or plant leaf",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, well-lit image for the most accurate diagnosis.",
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
    with col2:
        st.info(
            f"**File name:** {uploaded_file.name}\n\n"
            f"**File type:** {uploaded_file.type}\n\n"
            f"**File size:** {uploaded_file.size / 1024:.1f} KB"
        )

st.divider()

# Diagnose button
diagnose_btn = st.button(
    "🔬 Diagnose Crop",
    type="primary",
    disabled=(uploaded_file is None),
    help="Upload an image first, then click to start the diagnosis.",
)

# ─── Diagnosis workflow ───────────────────────────────────────────────────────

if diagnose_btn:
    # ── Credential validation ──────────────────────────────────────────────────
    if not ibm_api_key.strip():
        st.error("❌ Please enter your IBM API key in the sidebar before diagnosing.")
        st.stop()
    if not project_id.strip():
        st.error("❌ Please enter your watsonx.ai Project ID in the sidebar.")
        st.stop()
    if not vision_model_id.strip():
        st.error("❌ Vision Model ID cannot be empty.")
        st.stop()
    if not text_model_id.strip():
        st.error("❌ Text Model ID cannot be empty.")
        st.stop()

    image_bytes = uploaded_file.read()
    mime_type = uploaded_file.type

    # ── GUARDRAIL STAGE 1: Input guardrails ───────────────────────────────────
    st.subheader("🛡️ Guardrail Checks")

    with st.spinner("Running input guardrails…"):
        input_audit = run_all_input_checks(
            session_id=st.session_state.session_id,
            image_bytes=image_bytes,
            mime_type=mime_type,
            requested_tokens=max_tokens,
            vision_model_id=vision_model_id.strip(),
            endpoint_url=IBM_ENDPOINT,
        )

    render_guardrail_panel(input_audit, "Input Validation")

    if input_audit.blocked():
        st.error(
            "🚫 **Input blocked by guardrails.** Please review the issues above "
            "and upload a valid crop image."
        )
        st.stop()

    # Mask any PII in image filename (edge case)
    safe_filename = mask_pii(uploaded_file.name)

    st.divider()

    # ── Step 1: Build model clients ───────────────────────────────────────────
    try:
        with st.spinner("Connecting to IBM watsonx…"):
            vision_model = build_model(
                ibm_api_key.strip(), project_id.strip(), vision_model_id.strip()
            )
            text_model = build_model(
                ibm_api_key.strip(), project_id.strip(), text_model_id.strip()
            )
    except Exception as exc:
        st.error(f"❌ Failed to connect to IBM watsonx: {exc}")
        st.stop()

    # ── Step 2: Pathologist Agent ─────────────────────────────────────────────
    st.subheader("🔬 Agent 1 – Pathologist Agent")
    st.markdown("*Analysing the crop image for disease indicators…*")

    pathologist_report = ""
    agent1_run = AgentRun(
        agent_name="Pathologist Agent",
        model_id=vision_model_id.strip(),
        input_cost_per_1k=0.0003,
        output_cost_per_1k=0.0006,
    )
    with st.spinner("Pathologist Agent is examining the image…"):
        _t0 = time.time()
        try:
            pathologist_report = run_pathologist_agent(
                vision_model, image_bytes, mime_type, max_tokens
            )
            agent1_run.latency_seconds = time.time() - _t0
            agent1_run.ttft_seconds = agent1_run.latency_seconds * 0.18
            # Estimate tokens: ~0.75 tokens/char for English text
            agent1_run.input_tokens = max(100, int(len(str(image_bytes)) * 0.01))
            agent1_run.output_tokens = max(50, int(len(pathologist_report) * 0.75))
            agent1_run.response_text = pathologist_report
            agent1_run.succeeded = True
        except Exception as exc:
            agent1_run.latency_seconds = time.time() - _t0
            agent1_run.succeeded = False
            agent1_run.error_message = str(exc)
            err_msg = str(exc)
            if "not supported" in err_msg.lower() or "model" in err_msg.lower():
                st.error(
                    f"❌ The vision model `{vision_model_id}` may not support vision / chat. "
                    f"Try a vision-capable model.\n\nDetails: {err_msg}"
                )
            else:
                st.error(f"❌ Pathologist Agent error: {err_msg}")
            st.stop()

    # ── GUARDRAIL STAGE 2: Agent 1 output guardrails ──────────────────────────
    with st.spinner("Validating Pathologist Agent output…"):
        agent1_audit = run_all_agent1_output_checks(pathologist_report)

    render_guardrail_panel(agent1_audit, "Pathologist Agent Output")

    if agent1_audit.blocked():
        st.error(
            "🚫 **Pathologist Agent output blocked by guardrails.** "
            "The response contained unsafe or invalid content."
        )
        st.stop()

    # Show HITL escalation banner if triggered
    hitl_warnings = [
        r for r in agent1_audit.results
        if r.check == "EscalationTrigger" and not r.passed
    ]
    if hitl_warnings:
        st.warning(
            "👨‍⚕️ **Human Expert Recommended** – The diagnosis indicates uncertainty "
            "or a serious condition. Please consult a qualified plant pathologist "
            "or your local agricultural extension officer before taking action."
        )

    with st.expander("📋 Pathologist Agent – Diagnostic Report", expanded=True):
        st.markdown(pathologist_report)

    st.success("✅ Pathologist Agent completed the diagnosis.")
    st.divider()

    # ── Step 3: Extension Officer Agent ──────────────────────────────────────
    st.subheader("👨‍🌾 Agent 2 – Extension Officer Agent")
    st.markdown(
        "*Preparing farmer-friendly advisory based on the Pathologist's findings…*"
    )

    extension_advisory = ""
    agent2_input_text = (
        "--- PATHOLOGIST'S DIAGNOSTIC REPORT ---\n"
        + pathologist_report
        + "\n--- END OF REPORT ---"
    )
    agent2_run = AgentRun(
        agent_name="Extension Officer Agent",
        model_id=text_model_id.strip(),
        input_cost_per_1k=0.0003,
        output_cost_per_1k=0.0006,
    )
    with st.spinner("Extension Officer Agent is preparing your advisory…"):
        _t1 = time.time()
        try:
            extension_advisory = run_extension_officer_agent(
                text_model, pathologist_report, max_tokens
            )
            agent2_run.latency_seconds = time.time() - _t1
            agent2_run.ttft_seconds = agent2_run.latency_seconds * 0.15
            agent2_run.input_tokens = max(100, int(len(agent2_input_text) * 0.75))
            agent2_run.output_tokens = max(50, int(len(extension_advisory) * 0.75))
            agent2_run.response_text = extension_advisory
            agent2_run.succeeded = True
        except Exception as exc:
            agent2_run.latency_seconds = time.time() - _t1
            agent2_run.succeeded = False
            agent2_run.error_message = str(exc)
            st.error(f"❌ Extension Officer Agent error: {exc}")
            st.stop()

    # ── GUARDRAIL STAGE 3: Agent 2 output guardrails ──────────────────────────
    with st.spinner("Validating Extension Officer Agent output…"):
        agent2_audit = run_all_agent2_output_checks(extension_advisory)

    render_guardrail_panel(agent2_audit, "Extension Officer Agent Output")

    if agent2_audit.blocked():
        st.error(
            "🚫 **Extension Officer Agent output blocked by guardrails.** "
            "The response contained unsafe or invalid content."
        )
        st.stop()

    # HITL for Agent 2 as well
    hitl2_warnings = [
        r for r in agent2_audit.results
        if r.check == "EscalationTrigger" and not r.passed
    ]
    if hitl2_warnings:
        st.warning(
            "👨‍⚕️ **Expert Escalation Suggested** – The advisory recommends professional "
            "consultation. Contact your nearest agricultural extension office."
        )

    with st.expander("📝 Extension Officer Agent – Farmer Advisory", expanded=True):
        st.markdown(extension_advisory)

    st.success("✅ Extension Officer Agent completed the advisory.")
    st.divider()

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    all_audit = get_audit_log()
    total_blocks = sum(1 for e in all_audit for r in e.results if r.severity == "BLOCK" and not r.passed)
    total_warnings = sum(1 for e in all_audit for r in e.results if r.severity == "WARNING" and not r.passed)
    total_checks = sum(len(e.results) for e in all_audit)

    eval_report = evaluate(
        runs=[agent1_run, agent2_run],
        agent1_output=pathologist_report,
        agent2_output=extension_advisory,
        agent2_input=agent2_input_text,
        question="crop disease diagnosis",
        context=pathologist_report + " " + extension_advisory,
        guardrail_blocks=total_blocks,
        guardrail_warnings=total_warnings,
        total_guardrail_checks=total_checks,
        run_id=st.session_state.session_id,
    )
    # Store in session state for dashboard.py
    st.session_state["eval_report"] = eval_report
    st.session_state["agent_runs"] = [agent1_run, agent2_run]

    # Quick eval summary strip
    st.subheader("📊 Evaluation Snapshot")
    ec1, ec2, ec3, ec4, ec5 = st.columns(5)
    ec1.metric("Goal Achievement", f"{eval_report.goal_achievement * 100:.0f}%")
    ec2.metric("Faithfulness", f"{eval_report.faithfulness * 100:.0f}%")
    ec3.metric("Total Latency", f"{eval_report.total_latency_seconds:.1f}s")
    ec4.metric("Est. Cost", f"${eval_report.total_cost_usd:.5f}")
    ec5.metric("Guardrail Rate", f"{eval_report.guardrail_intervention_rate * 100:.0f}%")
    if st.button("📊 View Full Evaluation Dashboard"):
        st.switch_page("pages/2_📊_Evaluation_Dashboard.py")
    st.divider()

    # ── Step 5: Combined final report ─────────────────────────────────────────
    st.subheader("📄 Combined Final Report")
    st.markdown(PESTICIDE_WARNING)

    tab1, tab2, tab3 = st.tabs(["🔬 Diagnosis", "👨‍🌾 Advisory", "🛡️ Audit Trail"])
    with tab1:
        st.markdown(pathologist_report)
    with tab2:
        st.markdown(extension_advisory)
    with tab3:
        st.markdown("**Full Guardrail Audit Log for this session:**")
        for entry in get_audit_log():
            s = entry.summary()
            status_icon = "🔴" if s["blocked"] else ("🟡" if s["warnings"] else "🟢")
            st.markdown(
                f"{status_icon} `{s['timestamp']}` · **{s['agent'].upper()}** · "
                f"{s['stage']} · {s['checks']} checks · "
                f"{s['warnings']} warnings · blocked={s['blocked']}"
            )

    # Download button
    final_report_text = build_final_report(
        pathologist_report,
        extension_advisory,
        safe_filename,
        get_audit_log(),
    )
    st.download_button(
        label="⬇️ Download Full Advisory Report (.txt)",
        data=final_report_text.encode("utf-8"),
        file_name=f"agri_doctor_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
    )

    st.info(
        "💡 **Tip:** If the diagnosis seems uncertain or the confidence is Low, "
        "please take a clearer, well-lit photo of the affected area and try again, "
        "or contact your local agricultural extension officer for on-site assessment."
    )

# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Agri Doctor v2.0 · Powered by IBM watsonx Granite · "
    "For educational and advisory purposes only. "
    "Not a substitute for professional plant pathology services."
)
