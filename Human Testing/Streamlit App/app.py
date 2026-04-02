import streamlit as st
import json
from pathlib import Path
import datetime

st.set_page_config(page_title="Human Validation - Architect & Auditor", layout="wide")
st.title("🎯 Human Validation Dashboard")
st.markdown("**Architect & Auditor Phase-Locked System** — 50 samples each")

@st.cache_data
def load_data():
    auditor_path = Path("gemma_auditor_50_human_eval.jsonl")
    architect_path = Path("llama_architect_50_human_eval.jsonl")
    auditor_data = [json.loads(line) for line in open(auditor_path, "r", encoding="utf-8") if line.strip()]
    architect_data = [json.loads(line) for line in open(architect_path, "r", encoding="utf-8") if line.strip()]
    return auditor_data, architect_data

auditor_samples, architect_samples = load_data()

if "evaluator_name" not in st.session_state:
    st.session_state.evaluator_name = ""
if "current_auditor_idx" not in st.session_state:
    st.session_state.current_auditor_idx = 0
if "current_architect_idx" not in st.session_state:
    st.session_state.current_architect_idx = 0
if "progress_loaded_auditor" not in st.session_state:
    st.session_state.progress_loaded_auditor = False
if "progress_loaded_architect" not in st.session_state:
    st.session_state.progress_loaded_architect = False

st.sidebar.title("Navigation")

role = st.sidebar.radio("Select Role", ["Auditor (Gemma-2-2B)", "Architect (Llama-3.1-8B)"])

name_input = st.sidebar.text_input(
    "Your Full Name (required)",
    value=st.session_state.evaluator_name,
    key="name_input"
)

if name_input and name_input.strip().title() != st.session_state.evaluator_name:
    st.session_state.evaluator_name = name_input.strip().title()
    st.session_state.current_auditor_idx = 0
    st.session_state.current_architect_idx = 0
    st.session_state.progress_loaded_auditor = False
    st.session_state.progress_loaded_architect = False
    st.rerun()

if not st.session_state.evaluator_name:
    st.sidebar.warning("⚠️ Please enter your name to continue")
    st.stop()

user_folder = Path(f"human_scores/{st.session_state.evaluator_name}")
user_folder.mkdir(parents=True, exist_ok=True)

if role == "Auditor (Gemma-2-2B)":
    samples = auditor_samples
    current_idx = st.session_state.current_auditor_idx
    role_key = "auditor"
    progress_loaded_key = "progress_loaded_auditor"
else:
    samples = architect_samples
    current_idx = st.session_state.current_architect_idx
    role_key = "architect"
    progress_loaded_key = "progress_loaded_architect"

progress_file = user_folder / f"{role_key}_progress.json"
scores_file = user_folder / f"{role_key}_scores.jsonl"

if not st.session_state[progress_loaded_key]:
    last_idx = -1

    if progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                saved_progress = json.load(f)
                last_idx = saved_progress.get("last_completed_idx", -1)
        except:
            pass
    elif scores_file.exists():
        try:
            with open(scores_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
                if lines:
                    last_sample_id = lines[-1]["sample_id"]
                    last_idx = next(
                        (i for i, s in enumerate(samples) if s.get("sample_id") == last_sample_id),
                        -1
                    )
        except:
            pass

    if role_key == "auditor":
        st.session_state.current_auditor_idx = min(last_idx + 1, len(samples) - 1)
    else:
        st.session_state.current_architect_idx = min(last_idx + 1, len(samples) - 1)

    st.session_state[progress_loaded_key] = True
    st.rerun()

if role_key == "auditor":
    current_idx = st.session_state.current_auditor_idx
else:
    current_idx = st.session_state.current_architect_idx

st.sidebar.progress((current_idx + 1) / len(samples))
st.sidebar.caption(f"Progress: {current_idx + 1} / {len(samples)}")

sample = samples[current_idx]

st.subheader(f"Sample {current_idx + 1:02d} / {len(samples)} — {sample.get('sample_id', 'N/A')}")

with st.expander("📥 Input Payload (what was given to the model)", expanded=False):
    st.json(sample.get("input_payload", {}))

col_gt, col_pred = st.columns(2)

with col_gt:
    st.markdown("**✅ Ground Truth (Actual Output)**")
    gt_json = json.dumps(sample.get("actual_output", {}), indent=2, ensure_ascii=False)

    if st.button("📋 Copy Ground Truth", key=f"copy_gt_{current_idx}"):
        full_text = f"Ground Truth - \n{gt_json}"
        st.components.v1.html(
            f"""
            <script>
                navigator.clipboard.writeText(`{full_text.replace('`', '\\`')}`);
                alert("✅ Ground Truth copied to clipboard!");
            </script>
            """,
            height=0
        )
    st.json(sample.get("actual_output", {}))

with col_pred:
    st.markdown("**🤖 Model Generated Output**")
    pred_json = json.dumps(sample.get("predicted_parsed", {}) or {}, indent=2, ensure_ascii=False)

    if st.button("📋 Copy Model Output", key=f"copy_pred_{current_idx}"):
        full_text = f"Model Generated Output - \n{pred_json}"
        st.components.v1.html(
            f"""
            <script>
                navigator.clipboard.writeText(`{full_text.replace('`', '\\`')}`);
                alert("✅ Model Output copied to clipboard!");
            </script>
            """,
            height=0
        )
    st.json(sample.get("predicted_parsed", {}) or {})

st.markdown("### Your Scoring (same as LLM-as-a-Judge)")

if role_key == "auditor":
    dims = [
        "issue_identification",
        "reasoning_quality",
        "recommendation_actionability",
        "rubric_calibration",
        "overall_audit_quality"
    ]
else:
    dims = [
        "contract_alignment",
        "fix_report_accuracy",
        "architecture_quality",
        "security_coverage",
        "plan_completeness"
    ]

scores = {}
for dim in dims:
    scores[dim] = st.slider(dim.replace("_", " ").title(), 0, 10, 5, key=f"{role_key}_{dim}_{current_idx}")

justification = st.text_area(
    "Brief Justification (be critical, 2-3 sentences)",
    key=f"justification_{role_key}_{current_idx}"
)

blocking = st.checkbox(
    "Blocking Agreement? (Would you approve this to move to the next phase?)",
    value=False,
    key=f"blocking_{role_key}_{current_idx}"
)

col_back, col_submit = st.columns([1, 3])

with col_back:
    if st.button("⬅️ Back", use_container_width=True):
        if role_key == "auditor":
            st.session_state.current_auditor_idx = max(0, current_idx - 1)
        else:
            st.session_state.current_architect_idx = max(0, current_idx - 1)
        st.rerun()

with col_submit:
    if st.button("✅ Submit Score & Next", type="primary", use_container_width=True):
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "evaluator_name": st.session_state.evaluator_name,
            "role": role_key,
            "sample_id": sample.get("sample_id"),
            "scores": scores,
            "justification": justification.strip(),
            "blocking_agreement": blocking
        }

        user_file = user_folder / f"{role_key}_scores.jsonl"
        with open(user_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open("human_scores_final.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump({"last_completed_idx": current_idx}, f)

        st.success(f"Saved: {sample.get('sample_id')}")

        if role_key == "auditor":
            st.session_state.current_auditor_idx = min(current_idx + 1, len(samples) - 1)
        else:
            st.session_state.current_architect_idx = min(current_idx + 1, len(samples) - 1)

        st.rerun()

st.markdown("---")

if st.button("📥 Download My Complete Results as JSON"):
    auditor_file = user_folder / "auditor_scores.jsonl"
    architect_file = user_folder / "architect_scores.jsonl"

    all_scores = []

    if auditor_file.exists():
        with open(auditor_file, "r", encoding="utf-8") as f:
            all_scores.extend([json.loads(line) for line in f if line.strip()])

    if architect_file.exists():
        with open(architect_file, "r", encoding="utf-8") as f:
            all_scores.extend([json.loads(line) for line in f if line.strip()])

    if all_scores:
        st.download_button(
            label="Download JSON",
            data=json.dumps(all_scores, indent=2, ensure_ascii=False),
            file_name=f"{st.session_state.evaluator_name}_results.json",
            mime="application/json"
        )

st.caption(f"Evaluator: **{st.session_state.evaluator_name}** | Role: **{role}**")