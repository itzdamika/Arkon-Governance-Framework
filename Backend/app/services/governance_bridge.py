from __future__ import annotations

import copy
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

_REPO = Path(__file__).resolve().parents[2]
_PKG = _REPO / "packages"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from governance import (
    GovernanceEngine,
    GovernanceUIBridge,
    rerun_last_turn,
    run_handle_turn_only,
    run_user_turn,
)
from governance.persistence import state_from_blob, state_to_blob
from governance.state import ChatTurn

from ..core.config import settings
from ..db.models import ArtifactRecord, Branch, Message, Thread, UserSettingsRow

_engines: dict[str, GovernanceEngine] = {}


def _cache_key(thread_id: str, branch_id: str) -> str:
    return f"{thread_id}:{branch_id}"


def _apply_user_settings_to_engine(eng: GovernanceEngine, row: UserSettingsRow | None) -> None:
    if not row:
        return
    eng.state.pass_threshold = row.pass_threshold
    eng.state.max_planning_rounds = row.max_planning_rounds
    eng.state.max_requirement_hops = row.max_requirement_hops
    eng.state.report_depth = row.report_depth
    eng.state.show_internal_panels = row.show_internal_panels
    eng.state.debug_mode = bool(row.thinking_enabled)


def new_engine() -> GovernanceEngine:
    ui = GovernanceUIBridge()
    base = settings.artifacts_dir
    return GovernanceEngine(artifacts_base=base, ui=ui)


def _get_branch(db: Session, thread_id: str, branch_id: str) -> Branch | None:
    return (
        db.query(Branch)
        .filter(Branch.id == branch_id, Branch.thread_id == thread_id)
        .first()
    )


def get_engine(
    db: Session,
    thread: Thread,
    branch_id: str,
    user_settings: UserSettingsRow | None,
) -> GovernanceEngine:
    key = _cache_key(thread.id, branch_id)
    if key in _engines:
        eng = _engines[key]
        _apply_user_settings_to_engine(eng, user_settings)
        return eng
    eng = new_engine()
    loaded = False
    br = _get_branch(db, thread.id, branch_id)
    blob = br.engine_state_blob if br and br.engine_state_blob else None
    if not blob and thread.engine_state_blob:
        blob = thread.engine_state_blob
    if blob:
        try:
            eng.state = state_from_blob(blob)
            loaded = True
        except Exception:
            loaded = False
    if not loaded:
        msgs = (
            db.query(Message)
            .filter(Message.thread_id == thread.id, Message.branch_id == branch_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        for m in msgs:
            ex = m.extra if isinstance(m.extra, dict) else None
            eng.state.dialogue.append(
                ChatTurn(role=m.role, content=m.content or "", agent=m.agent, extra=ex)
            )
        if thread.phase:
            eng.state.phase = thread.phase
    _apply_user_settings_to_engine(eng, user_settings)
    _engines[key] = eng
    return eng


def drop_engine_cache(thread_id: str, branch_id: str | None = None) -> None:
    if branch_id:
        _engines.pop(_cache_key(thread_id, branch_id), None)
        return
    prefix = f"{thread_id}:"
    for k in list(_engines.keys()):
        if k.startswith(prefix):
            del _engines[k]


def persist_engine(
    db: Session, thread: Thread, branch_id: str, eng: GovernanceEngine
) -> None:
    blob = state_to_blob(eng.state)
    br = _get_branch(db, thread.id, branch_id)
    if not br:
        br = Branch(
            id=branch_id,
            thread_id=thread.id,
            label="main",
            engine_state_blob=blob,
        )
        db.add(br)
    else:
        br.engine_state_blob = blob
        db.add(br)
    thread.engine_state_blob = blob
    thread.phase = eng.state.phase
    thread.updated_at = datetime.now(timezone.utc)
    db.add(thread)
    _sync_messages(db, thread.id, branch_id, eng)
    _sync_branch_planning_extra(db, thread.id, branch_id, eng)
    _sync_artifacts(db, thread, eng)
    db.commit()


def _sync_branch_planning_extra(
    db: Session, thread_id: str, branch_id: str, eng: GovernanceEngine
) -> None:
    """Keep a durable copy of the richest planning UI transcript on the branch row."""
    br = _get_branch(db, thread_id, branch_id)
    if not br:
        return
    best: list[Any] | None = None
    best_len = 0
    for turn in eng.state.dialogue:
        if turn.role != "assistant":
            continue
        ex = getattr(turn, "extra", None)
        if not isinstance(ex, dict):
            continue
        pe = ex.get("planning_ui_events")
        if isinstance(pe, list) and len(pe) > best_len:
            best = pe
            best_len = len(pe)
    if not best:
        return
    exb = dict(br.extra) if isinstance(br.extra, dict) else {}
    exb["planning_ui_events"] = best
    br.extra = exb
    db.add(br)


def _sync_messages(db: Session, thread_id: str, branch_id: str, eng: GovernanceEngine) -> None:
    db.query(Message).filter(
        Message.thread_id == thread_id, Message.branch_id == branch_id
    ).delete()
    for turn in eng.state.dialogue:
        ex = getattr(turn, "extra", None)
        m = Message(
            thread_id=thread_id,
            branch_id=branch_id,
            role=turn.role,
            content=turn.content or "",
            agent=turn.agent,
            extra=ex if isinstance(ex, dict) else None,
        )
        db.add(m)


def _sync_artifacts(db: Session, thread: Thread, eng: GovernanceEngine) -> None:
    path = eng.state.final_pdf_path
    if not path:
        return
    p = Path(path)
    if not p.is_file():
        return
    exists = (
        db.query(ArtifactRecord)
        .filter(ArtifactRecord.thread_id == thread.id, ArtifactRecord.path == str(p.resolve()))
        .first()
    )
    if exists:
        return
    rec = ArtifactRecord(
        thread_id=thread.id,
        kind="pdf",
        filename=p.name,
        path=str(p.resolve()),
        mime_type="application/pdf",
        size_bytes=p.stat().st_size,
    )
    db.add(rec)


def _resolve_branch_id(thread: Thread, branch_id: str | None) -> str:
    return branch_id or thread.active_branch_id


def execute_user_turn(
    db: Session,
    thread: Thread,
    user_settings: UserSettingsRow | None,
    text: str,
    branch_id: str | None = None,
    stream_job_id: str | None = None,
) -> dict[str, Any]:
    bid = _resolve_branch_id(thread, branch_id)
    if branch_id and branch_id != thread.active_branch_id:
        thread.active_branch_id = bid
        db.add(thread)
        db.commit()
    eng = get_engine(db, thread, bid, user_settings)
    saved_ui: GovernanceUIBridge | None = None
    if stream_job_id:

        def _emit(ui: GovernanceUIBridge) -> None:
            from . import jobs as job_svc

            job_svc.sync_live_ui_events(stream_job_id, list(ui.events))

        saved_ui = eng._ui
        eng._ui = GovernanceUIBridge(on_emit=_emit)
    try:
        before = len(eng.state.dialogue)
        result = run_user_turn(eng, text)
        result["messages"] = result["dialogue"][before:]
        persist_engine(db, thread, bid, eng)
        result["thread_id"] = thread.id
        result["branch_id"] = bid
        result["planning_summary"] = _planning_summary_from_events(
            result.get("ui_events") or []
        )
        return result
    finally:
        if saved_ui is not None:
            eng._ui = saved_ui


def execute_regenerate(
    db: Session,
    thread: Thread,
    user_settings: UserSettingsRow | None,
    branch_id: str | None = None,
) -> dict[str, Any]:
    bid = _resolve_branch_id(thread, branch_id)
    eng = get_engine(db, thread, bid, user_settings)
    before = max(0, len(eng.state.dialogue) - 1)
    result = rerun_last_turn(eng)
    if result.get("error"):
        return result
    result["messages"] = result["dialogue"][before:]
    persist_engine(db, thread, bid, eng)
    result["thread_id"] = thread.id
    result["branch_id"] = bid
    result["planning_summary"] = _planning_summary_from_events(result.get("ui_events") or [])
    return result


def fork_from_user_edit(
    db: Session,
    thread: Thread,
    user_settings: UserSettingsRow | None,
    parent_branch_id: str,
    message_id: str,
    new_content: str,
) -> dict[str, Any]:
    msgs = (
        db.query(Message)
        .filter(Message.thread_id == thread.id, Message.branch_id == parent_branch_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    idx = next((i for i, m in enumerate(msgs) if m.id == message_id), -1)
    if idx < 0:
        return {"error": "message_not_found"}
    if msgs[idx].role != "user":
        return {"error": "not_user_message"}
    parent_eng = get_engine(db, thread, parent_branch_id, user_settings)
    st = copy.deepcopy(parent_eng.state)
    st.dialogue = [
        ChatTurn(
            role=m.role,
            content=m.content or "",
            agent=m.agent,
            extra=m.extra if isinstance(m.extra, dict) else None,
        )
        for m in msgs[:idx]
    ]
    st.dialogue.append(ChatTurn(role="user", content=new_content.strip(), agent=None))
    new_bid = str(uuid.uuid4())
    db.add(
        Branch(
            id=new_bid,
            thread_id=thread.id,
            parent_branch_id=parent_branch_id,
            forked_from_message_id=message_id,
            label="Fork",
            engine_state_blob=None,
        )
    )
    thread.active_branch_id = new_bid
    db.add(thread)
    db.commit()
    new_eng = new_engine()
    new_eng.state = st
    _apply_user_settings_to_engine(new_eng, user_settings)
    before = max(0, len(new_eng.state.dialogue) - 1)
    result = run_handle_turn_only(new_eng, new_content.strip())
    if result.get("error"):
        return result
    result["messages"] = result["dialogue"][before:]
    _engines[_cache_key(thread.id, new_bid)] = new_eng
    persist_engine(db, thread, new_bid, new_eng)
    _tag_fork_anchor_on_last_user_message(db, thread.id, new_bid, message_id)
    result["thread_id"] = thread.id
    result["branch_id"] = new_bid
    result["planning_summary"] = _planning_summary_from_events(result.get("ui_events") or [])
    assistant_turn = {
        "thread_id": thread.id,
        "branch_id": new_bid,
        "phase": result["phase"],
        "messages": result["messages"],
        "quick_replies": result.get("quick_replies", []),
        "planning_summary": result.get("planning_summary"),
        "artifacts": [],
        "ui_events": result.get("ui_events", []),
        "final_pdf_path": result.get("final_pdf_path"),
        "stream_planning_live": bool(result.get("stream_planning_live", False)),
    }
    return {"branch_id": new_bid, "assistant_turn": assistant_turn}


def _tag_fork_anchor_on_last_user_message(
    db: Session, thread_id: str, branch_id: str, anchor_message_id: str
) -> None:
    """So the UI can load /fork-versions for the parent message while on a fork branch."""
    last_u = (
        db.query(Message)
        .filter(
            Message.thread_id == thread_id,
            Message.branch_id == branch_id,
            Message.role == "user",
        )
        .order_by(Message.created_at.desc())
        .first()
    )
    if not last_u:
        return
    ex = dict(last_u.extra or {})
    ex["fork_anchor_message_id"] = anchor_message_id
    last_u.extra = ex
    db.add(last_u)
    db.commit()


def _planning_summary_from_events(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    rounds: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("type") != "round_tables":
            continue
        rounds.append(
            {
                "round": ev.get("round"),
                "plan_rows": ev.get("plan_rows"),
                "audit_rows": ev.get("audit_rows"),
            }
        )
    if not rounds:
        return None
    return {"status": "completed", "rounds": rounds, "finalization_reason": None}


def development_answer(
    db: Session,
    thread: Thread,
    user_settings: UserSettingsRow | None,
    text: str,
    branch_id: str | None = None,
) -> dict[str, Any]:
    from governance.constants import PHASE_DEVELOPMENT

    bid = _resolve_branch_id(thread, branch_id)
    eng = get_engine(db, thread, bid, user_settings)
    if eng.state.phase != PHASE_DEVELOPMENT:
        return {"error": "forbidden", "detail": "Development chat is only available after planning completes."}
    rp = eng.state.report_package
    if not rp:
        return {"error": "forbidden", "detail": "No report package yet."}
    import json

    ctx = json.dumps(rp, ensure_ascii=False)[:100000]
    system = """You are a senior engineer helping the user implement the project.
Use the architecture report JSON as the primary source of truth.
If something is not in the report, say so and suggest updating requirements or docs.
Be concise and actionable."""
    eng.append_dialogue("user", text.strip())
    try:
        answer = eng.llm.complete_text(
            system,
            f"Report JSON (truncated if needed):\n{ctx}\n\nUser question:\n{text.strip()}",
            max_tokens=4096,
            temperature=0.2,
        )
    except Exception as e:
        return {"error": "llm_error", "detail": str(e)}
    eng.append_dialogue("assistant", answer, "DevelopmentTutor")
    persist_engine(db, thread, bid, eng)
    return {
        "message": {
            "role": "assistant",
            "content": answer,
            "agent": "DevelopmentTutor",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    }


def run_governance_only(
    db: Session,
    thread: Thread,
    user_settings: UserSettingsRow | None,
    branch_id: str | None = None,
    stream_job_id: str | None = None,
) -> dict[str, Any]:
    """Resume PLANNING phase (e.g. after advance_phase without cycle)."""
    bid = _resolve_branch_id(thread, branch_id)
    eng = get_engine(db, thread, bid, user_settings)
    from governance.constants import PHASE_PLANNING

    if eng.state.phase != PHASE_PLANNING:
        return {"error": "not_planning", "detail": f"Phase is {eng.state.phase}"}
    saved_ui: GovernanceUIBridge | None = None
    if stream_job_id:

        def _emit(ui: GovernanceUIBridge) -> None:
            from . import jobs as job_svc

            job_svc.sync_live_ui_events(stream_job_id, list(ui.events))

        saved_ui = eng._ui
        eng._ui = GovernanceUIBridge(on_emit=_emit)
    try:
        eng._ui.clear()
        try:
            eng.run_governance_cycle()
        except Exception as e:
            import traceback

            return {"error": str(e), "trace": traceback.format_exc()}
        persist_engine(db, thread, bid, eng)
        return {
            "phase": eng.state.phase,
            "ui_events": list(eng._ui.events),
            "final_pdf_path": eng.state.final_pdf_path,
        }
    finally:
        if saved_ui is not None:
            eng._ui = saved_ui
