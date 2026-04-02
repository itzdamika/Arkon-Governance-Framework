import streamlit as st
import json
from pathlib import Path
import datetime

st.set_page_config(page_title="Human Validation - Architect & Auditor", layout="wide")
st.title("🎯 Human Validation Dashboard")
st.markdown("**Architect & Auditor Phase-Locked System** — 50 samples each")

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    auditor_path = Path("gemma_auditor_50_human_eval.jsonl")
    architect_path = Path("llama_architect_50_human_eval.jsonl")

    auditor_data = [json.loads(line) for line in open(auditor_path, "r", encoding="utf-8") if line.strip()]
    architect_data = [json.loads(line) for line in open(architect_path, "r", encoding="utf-8") if line.strip()]

    return auditor_data, architect_data

auditor_samples, architect_samples = load_data()

# ====================== SESSION STATE ======================
if "evaluator_name" not in st.session_state:
    st.session_state.evaluator_name = ""
if "current_auditor_idx" not in st.session_state:
    st.session_state.current_auditor_idx = 0
if "current_architect_idx" not in st.session_state:
    st.session_state.current_architect_idx = 0

# ====================== SIDEBAR ======================
st.sidebar.title("Navigation")

role = st.sidebar.radio("Select Role", ["Auditor (Gemma-2-2B)", "Architect (Llama-3.1-8B)"])

name = st.sidebar.text_input("Your Full Name (required)", value=st.session_state.evaluator_name)
if name:
    st.session_state.evaluator_name = name.strip().title()

if not st.session_state.evaluator_name:
    st.sidebar.warning("⚠️ Please enter your name to continue")
    st.stop()

# Progress
if role == "Auditor (Gemma-2-2B)":
    samples = auditor_samples
    current_idx = st.session_state.current_auditor_idx
    role_key = "auditor"
else:
    samples = architect_samples
    current_idx = st.session_state.current_architect_idx
    role_key = "architect"

st.sidebar.progress((current_idx + 1) / len(samples))
st.sidebar.caption(f"Progress: {current_idx + 1} / {len(samples)}")

# ====================== MAIN PAGE ======================
st.subheader(f"Sample {current_idx + 1:02d} / {len(samples)} — {samples[current_idx].get('sample_id', 'N/A')}")

# Input Payload (separate section)
with st.expander("📥 Input Payload (what was given to the model)", expanded=False):
    st.json(samples[current_idx].get("input_payload", {}))

# Side-by-side comparison
col_gt, col_pred = st.columns(2)

with col_gt:
    st.markdown("**✅ Ground Truth (Actual Output)**")
    st.json(samples[current_idx].get("actual_output", {}))

with col_pred:
    st.markdown("**🤖 Model Generated Output**")
    st.json(samples[current_idx].get("predicted_parsed", {}) or {})

if samples[current_idx].get("predicted_raw_text"):
    with st.expander("Show full raw generated text"):
        st.text(samples[current_idx]["predicted_raw_text"])

# ====================== SCORING ======================
st.markdown("### Your Scoring")

if role == "Auditor (Gemma-2-2B)":
    dims = ["requirements_alignment", "architecture_quality", "security", "operability", "internal_consistency"]
else:
    dims = ["contract_alignment", "fix_report_accuracy", "architecture_quality", "security_coverage", "plan_completeness"]

scores = {}
for dim in dims:
    scores[dim] = st.slider(dim.replace("_", " ").title(), 0, 10, 5, key=f"score_{dim}")

justification = st.text_area("Brief Justification (be critical, 2-3 sentences)",  placeholder="e.g. The model ignored the private VPC constraint...")

blocking = st.checkbox("Blocking Agreement? (Would you approve this to move to the next phase?)", value=False)

# ====================== BUTTONS ======================
col_back, col_submit = st.columns([1, 3])

with col_back:
    if st.button("⬅️ Back", use_container_width=True):
        if role == "Auditor (Gemma-2-2B)":
            st.session_state.current_auditor_idx = max(0, current_idx - 1)
        else:
            st.session_state.current_architect_idx = max(0, current_idx - 1)
        st.rerun()

with col_submit:
    if st.button("✅ Submit Score & Go to Next Sample", type="primary", use_container_width=True):
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "evaluator_name": st.session_state.evaluator_name,
            "role": role_key,
            "sample_id": samples[current_idx].get("sample_id"),
            "scores": scores,
            "justification": justification.strip(),
            "blocking_agreement": blocking
        }

        # Create per-user folder
        user_folder = Path(f"human_scores/{st.session_state.evaluator_name}")
        user_folder.mkdir(parents=True, exist_ok=True)

        # Save to per-user per-role file
        user_file = user_folder / f"{role_key}_scores.jsonl"
        with open(user_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Also append to master file
        with open("human_scores_final.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        st.success(f"Score saved for {samples[current_idx].get('sample_id')}")

        # Move to next sample
        if role == "Auditor (Gemma-2-2B)":
            st.session_state.current_auditor_idx = min(current_idx + 1, len(samples) - 1)
        else:
            st.session_state.current_architect_idx = min(current_idx + 1, len(samples) - 1)

        # === AUTO SCROLL TO TOP ===
        st.components.v1.html(
            """
            <script>
                window.scrollTo({ top: 0, behavior: 'smooth' });
            </script>
            """,
            height=0
        )

        st.rerun()

st.caption(f"Evaluator: **{st.session_state.evaluator_name}** | Role: **{role}**")