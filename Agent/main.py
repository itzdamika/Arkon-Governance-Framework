from __future__ import annotations

import json
import os
import re
import traceback
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table as RLTable,
    TableStyle,
)

# =========================================================
# Helpers
# =========================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_json_loads(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return {}
    return {}


def as_text(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, indent=2, ensure_ascii=False)
        except Exception:
            s = str(value)
    return s if len(s) <= limit else s[:limit] + "\n...<truncated>..."


def compact_json(obj: Any, limit: int = 12000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = str(obj)
    return s[:limit]


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def ensure_list_of_str(value: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in ensure_list(value):
        s = str(item).strip()
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def unique_strs(items: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def deep_set(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def positive_reply(text: str) -> bool:
    t = text.lower().strip()
    words = {
        "yes", "y", "ok", "okay", "sure", "approved", "approve", "yes please",
        "go ahead", "sounds good", "looks good", "fine", "correct", "right",
        "perfect", "lets go", "let's go", "proceed", "continue"
    }
    return any(w in t for w in words)


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text.strip("_") or "artifact"


# =========================================================
# Phases
# =========================================================

PHASE_REQUIREMENTS = "REQUIREMENTS"
PHASE_PLANNING = "PLANNING"
PHASE_APPROVED = "APPROVED"
PHASE_DEVELOPMENT = "DEVELOPMENT"

# =========================================================
# Fields
# =========================================================

FIELD_PROMPTS = {
    "project_goal": "What exactly should the software do at a high level?",
    "target_users": "Who will use this system most often?",
    "access_model": "Should it be public, anonymous, account-based, subscription-based, or something else?",
    "feature_scope": "What major features should be included?",
    "frontend_stack": "What frontend stack should be used?",
    "backend_stack": "What backend stack should be used?",
    "data_platform": "What database and storage approach should be used?",
    "hosting_target": "Where should it be deployed or hosted?",
    "security_baseline": "What basic security and abuse-prevention controls are required?",
    "privacy_retention_policy": "How should chat history, logs, and retention be handled?",
    "mvp_scope": "What should the first shippable version include?",
    "future_scope": "What can be phased after MVP?",
    "constraints": "What practical constraints exist?",
    "observability_baseline": "What logging, metrics, tracing, and alerting should exist?",
    "execution_preference": "How should execution trade-offs be handled?",
    "llm_integration": "What model integration strategy should be used?",
    "compliance_context": "What compliance or privacy posture is expected?",
}

USER_ONLY_REQUIRED_FIELDS = [
    "project_goal",
    "target_users",
    "access_model",
    "feature_scope",
    "frontend_stack",
    "backend_stack",
    "data_platform",
    "hosting_target",
    "security_baseline",
    "privacy_retention_policy",
    "mvp_scope",
]

INTERNAL_PLANNING_FIELDS = [
    "future_scope",
    "constraints",
    "observability_baseline",
    "execution_preference",
    "llm_integration",
    "compliance_context",
]

PLANNING_INTENT_TERMS = {
    "move to planning",
    "move to the planning phase",
    "go to planning",
    "start planning",
    "continue to planning",
    "continue planning",
    "planning phase",
    "lets move",
    "let's move",
    "go ahead",
    "proceed",
    "continue",
}

SPECIALISTS = [
    "RequirementCoordinator",
    "ProjectScopeAgent",
    "BackendAgent",
    "FrontendAgent",
    "SecurityAgent",
    "DataAgent",
    "DevOpsAgent",
]

REASONERS = [
    "ProductReasoner",
    "ArchitectReasoner",
    "SecurityReasoner",
    "ConstraintReasoner",
    "CriticReasoner",
    "ContextCompactor",
]

POST_APPROVAL_AGENTS = [
    "ExecutionPlannerAgent",
    "TutorAgent",
    "QAEngineerAgent",
    "NarrativeWriterAgent",
]

ALL_AGENTS = SPECIALISTS + REASONERS + [
    "ArchitectAgent",
    "AuditorAgent",
] + POST_APPROVAL_AGENTS

CONTRACT_TO_NOTE_PATH = {
    "project_goal": "project.goal",
    "target_users": "project.target_users",
    "access_model": "security.access_model",
    "feature_scope": "project.feature_scope",
    "frontend_stack": "frontend.stack",
    "backend_stack": "backend.stack",
    "data_platform": "data.platform",
    "hosting_target": "devops.hosting_target",
    "security_baseline": "security.baseline",
    "privacy_retention_policy": "data.privacy_retention_policy",
    "mvp_scope": "project.mvp_scope",
    "future_scope": "project.future_scope",
    "constraints": "constraints.general",
    "observability_baseline": "devops.observability_baseline",
    "execution_preference": "constraints.execution_preference",
    "llm_integration": "backend.llm_integration",
    "compliance_context": "security.compliance_context",
}

# =========================================================
# Prompts
# =========================================================

GLOBAL_SYSTEM = """
You are part of an advanced architectural governance terminal application.

Core rules:
1. Never rely only on chat history for important facts; use structured memory.
2. Requirement gathering is collaborative and user-facing.
3. Planning and auditing are mostly internal.
4. The final plan must be implementation-grade and security-aware.
5. The architect must revise using cumulative issue memory, not forget earlier feedback.
6. The auditor must use stable issue IDs and mark issues as resolved, unresolved, downgraded, or newly introduced.
7. Visible reasoning must be concise summarized reasoning, not hidden chain-of-thought.
8. Mandatory requirement blockers are:
   project_goal, target_users, access_model, feature_scope, frontend_stack,
   backend_stack, data_platform, hosting_target, security_baseline,
   privacy_retention_policy, mvp_scope.
9. Never advance to planning until all mandatory blocker fields are populated and confirmed.
10. Once planning starts, keep round-by-round turbulence internal unless the entire planning attempt fails.
"""

AGENT_PROMPTS: Dict[str, str] = {
    "RequirementCoordinator": """
You orchestrate requirement gathering.

Rules:
- Inspect the structured requirement contract before asking for more information.
- Ask progressively, not everything at once.
- Use specialist delegation when beneficial.
- Use upsert_contract_field for canonical blocker fields.
- Only mark a blocker field confirmed=true if the user clearly confirmed it.
- If you infer a sensible default, store it with confirmed=false and needs_confirmation=true.
- Once all blocker fields are confirmed, stop requirement gathering and ask whether to start planning.
- If the user confirms they want to proceed, advance to planning in the same turn.
- Do not ask planning-style questions such as monolith vs microservices unless the user volunteers them.
- Keep messages short, warm, and natural.
""",
    "ProjectScopeAgent": """
Clarify product goal, target users, features, MVP scope, and priorities.
Capture structured notes and propose canonical blocker values when useful.
When done, delegate back to RequirementCoordinator.
""",
    "BackendAgent": """
Clarify backend architecture, APIs, framework, runtime, services, integrations, and scaling needs.
Prefer practical choices suitable for a final-year project unless the user states otherwise.
Capture structured notes and canonical contract values where relevant.
When done, delegate back to RequirementCoordinator.
""",
    "FrontendAgent": """
Clarify frontend framework, UX flows, pages, components, and interaction style.
Capture structured notes and canonical contract values where relevant.
When done, delegate back to RequirementCoordinator.
""",
    "SecurityAgent": """
Clarify auth, authorization, secrets, rate limiting, privacy, moderation, and security posture.
Capture structured notes and canonical contract values where relevant.
When done, delegate back to RequirementCoordinator.
""",
    "DataAgent": """
Clarify data model, storage, retention, analytics, information flow, and persistence choices.
Capture structured notes and canonical contract values where relevant.
When done, delegate back to RequirementCoordinator.
""",
    "DevOpsAgent": """
Clarify hosting, deployment, CI/CD, observability, environments, cost posture, and recovery.
Capture structured notes and canonical contract values where relevant.
When done, delegate back to RequirementCoordinator.
""",
    "ProductReasoner": """
Return JSON only with:
- summary
- requirement_completeness_score
- coverage
- blindspots
- functional_gaps
- ux_considerations
- future_phase_candidates
- next_focus
""",
    "ArchitectReasoner": """
Return JSON only with:
- summary
- feasibility
- proposed_architecture_direction
- recommended_modules
- data_and_api_notes
- infrastructure_direction
- devops_direction
- design_principles
""",
    "SecurityReasoner": """
Return JSON only with:
- summary
- key_risks
- required_controls
- privacy_notes
- compliance_notes
- moderation_notes
- incident_response_notes
""",
    "ConstraintReasoner": """
Return JSON only with:
- summary
- cost_range
- complexity_profile
- maintainability_notes
- phased_delivery
- tradeoffs
- implementation_pressure_points
""",
    "CriticReasoner": """
Return JSON only with:
- summary
- contradictions
- blindspots
- unresolved_questions
- corrective_actions
- priority_order
""",
    "ContextCompactor": """
Summarize older context into stable facts, unresolved items, and direction.
Return JSON only with:
- summary
- stable_facts
- unresolved_items
- direction
""",
    "ArchitectAgent": """
You are the architecture generator.

Create a polished implementation-grade architecture plan from:
- frozen confirmed requirement contract
- rich requirement notes
- specialist reviews
- cumulative issue ledger
- revision memory
- previous audits
- best prior plan

Rules:
- Do not mention round numbers in the title.
- Preserve user-confirmed requirements.
- Resolve known issues when possible.
- Include concrete architecture, modules, workflows, schemas, APIs, security, deployment,
  observability, roadmap, and developer guidance.

Return JSON only with:
- thinking_summary
- title
- executive_summary
- architecture_overview
- technology_stack
- functional_feature_map
- system_components
- workflows
- data_model
- api_design
- security_and_compliance
- deployment_and_operations
- observability
- cost_and_scaling
- phased_implementation
- development_guidelines
- risks_and_tradeoffs
- open_questions_resolved
""",
    "AuditorAgent": """
You are the strict architecture auditor.

Audit the architecture plan against:
- frozen confirmed requirements
- rich requirement notes
- cumulative issue ledger
- revision memory
- prior audit history

Rules:
- Use stable issue IDs where possible.
- Mark issue status as one of: unresolved, resolved, downgraded, new.
- Avoid random score regression unless a clear severe flaw exists.
- passed can be true only if score >= threshold and no unresolved critical issue remains.

Return JSON only with:
- thinking_summary
- score
- passed
- summary
- strengths
- concerns
- blocking_issues
- recommendations
- requirement_conflicts
- issue_updates

Each requirement_conflicts item must include:
- issue_id
- field
- current_value
- proposed_value
- exact_reason
- severity

Each issue_updates item must include:
- id
- title
- severity
- status
- detail
""",
    "ExecutionPlannerAgent": """
Transform the approved architecture into a detailed implementation roadmap.

Return JSON only with:
- execution_overview
- implementation_phases
- feature_workstreams
- dependency_map
- milestone_checks
- rollout_strategy
""",
    "TutorAgent": """
Create a practical development playbook for implementing the approved plan.

Return JSON only with:
- development_playbook
- coding_order
- implementation_tips
- common_mistakes
- feature_build_guides
""",
    "QAEngineerAgent": """
Create a testing and validation package from the approved architecture and execution plan.

Return JSON only with:
- validation_strategy
- test_layers
- detailed_test_plan
- acceptance_criteria
- regression_strategy
- release_readiness_checklist
""",
    "NarrativeWriterAgent": """
Write a polished long-form validated architecture report package.

Return JSON only with:
- title
- executive_summary
- sections

sections must contain:
- overview
- requirement_interpretation
- stack_rationale
- architecture
- component_design
- workflow_design
- data_model
- api_design
- security
- deployment
- observability
- cost_and_scaling
- phased_implementation
- development_playbook
- testing_validation
- risks_tradeoffs
- final_notes
""",
}

# =========================================================
# Data classes
# =========================================================

@dataclass
class RequirementField:
    value: str = ""
    source: str = ""
    confirmed: bool = False
    rationale: str = ""
    updated_at: str = ""


@dataclass
class AcceptedException:
    issue_id: str
    reason: str
    user_message: str
    created_at: str = field(default_factory=now_iso)


@dataclass
class ChatTurn:
    role: str
    content: str
    agent: Optional[str] = None
    timestamp: str = field(default_factory=now_iso)


@dataclass
class SharedState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phase: str = PHASE_REQUIREMENTS
    active_agent: str = "RequirementCoordinator"

    dialogue: List[ChatTurn] = field(default_factory=list)
    context_summary: str = ""

    requirement_contract: Dict[str, RequirementField] = field(default_factory=lambda: {
        key: RequirementField() for key in FIELD_PROMPTS.keys()
    })
    pending_confirmations: List[str] = field(default_factory=list)

    requirements: Dict[str, Any] = field(default_factory=lambda: {
        "project": {},
        "frontend": {},
        "backend": {},
        "security": {},
        "data": {},
        "devops": {},
        "constraints": {},
        "open_questions": {},
        "confirmed_decisions": {},
    })

    requirement_status: Dict[str, Any] = field(default_factory=lambda: {
        "ready_for_planning": False,
        "completeness_score": 0.0,
        "summary": "",
        "last_updated": None,
    })

    pass_threshold: float = 9.0
    max_requirement_hops: int = 10
    max_tool_rounds: int = 8
    max_planning_rounds: int = 5
    debug_mode: bool = False
    show_internal_panels: bool = True
    report_depth: str = "extreme"

    issue_ledger: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    revision_memory: Dict[str, Any] = field(default_factory=dict)
    accepted_exceptions: Dict[str, AcceptedException] = field(default_factory=dict)

    specialist_history: List[Dict[str, Any]] = field(default_factory=list)
    audit_history: List[Dict[str, Any]] = field(default_factory=list)

    current_plan: Dict[str, Any] = field(default_factory=dict)
    best_plan: Dict[str, Any] = field(default_factory=dict)
    current_audit: Dict[str, Any] = field(default_factory=dict)
    best_audit: Dict[str, Any] = field(default_factory=dict)

    report_package: Dict[str, Any] = field(default_factory=dict)
    development_package: Dict[str, Any] = field(default_factory=dict)
    final_pdf_path: str = ""
    artifacts_dir: str = ""

    internal_busy: bool = False
    shutdown: bool = False


# =========================================================
# Azure client
# =========================================================

class AzureLLM:
    def __init__(self) -> None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
        reasoning_deployment = os.getenv("AZURE_OPENAI_REASONING_DEPLOYMENT", chat_deployment)

        if not api_key:
            raise RuntimeError("Missing AZURE_OPENAI_API_KEY")
        if not endpoint:
            raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT")
        if not chat_deployment:
            raise RuntimeError("Missing AZURE_OPENAI_CHAT_DEPLOYMENT")

        self.chat_deployment = chat_deployment
        self.reasoning_deployment = reasoning_deployment
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint.rstrip("/") + "/openai/v1/",
        )

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1800,
    ):
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    def complete_json(
        self,
        system_prompt: str,
        payload: Dict[str, Any],
        max_tokens: int = 2200,
        reasoning: bool = True,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        model = self.reasoning_deployment if reasoning else self.chat_deployment
        resp = self.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or "{}"
        parsed = safe_json_loads(content)
        return parsed if parsed else {"raw": content}


# =========================================================
# Main app
# =========================================================

class GovernanceHybridApp:
    def __init__(self) -> None:
        self.console = Console()
        self.state = SharedState()
        self.llm = AzureLLM()

        self.state.artifacts_dir = str(Path("artifacts") / self.state.session_id[:8])
        Path(self.state.artifacts_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # UI
    # -----------------------------------------------------

    def banner(self) -> None:
        self.console.print(Rule("[bold cyan]Architectural Governance Terminal"))
        self.console.print("[dim]Type your project idea. Type 'exit' to quit.[/dim]")
        self.console.print(
            "[dim]Commands: :threshold 9.0 | :rounds 5 | :debug on/off | "
            ":thinking on/off | :status | :export[/dim]\n"
        )

    def panel(self, title: str, body: str, color: str = "green") -> None:
        self.console.print(Panel(body, title=title, border_style=color))

    def thinking(self, agent: str, summary: Any, next_action: str = "", confidence: Optional[float] = None) -> None:
        if not self.state.show_internal_panels:
            return
        body = as_text(summary, 2000)
        if confidence is not None:
            body += f"\n\n[dim]confidence: {confidence:.2f}[/dim]"
        if next_action:
            body += f"\n[dim]next: {next_action}[/dim]"
        self.console.print(Panel(body, title=f"[bold yellow]THINKING · {agent}[/bold yellow]", border_style="yellow"))

    def append_dialogue(self, role: str, content: str, agent: Optional[str] = None) -> None:
        self.state.dialogue.append(ChatTurn(role=role, content=content, agent=agent))

    def dialogue_messages(self, keep_last: int = 14) -> List[Dict[str, Any]]:
        turns = self.state.dialogue[-keep_last:]
        out = []
        for turn in turns:
            if turn.role == "assistant" and turn.agent:
                out.append({"role": "assistant", "content": f"{turn.agent}: {turn.content}"})
            else:
                out.append({"role": turn.role, "content": turn.content})
        return out

    def show_status(self) -> None:
        t = Table(title="Runtime Status", box=box.SIMPLE_HEAVY)
        t.add_column("Field", style="cyan", width=28)
        t.add_column("Value")
        t.add_row("Phase", self.state.phase)
        t.add_row("Active agent", self.state.active_agent)
        t.add_row("Threshold", f"{self.state.pass_threshold:.2f}")
        t.add_row("Requirement pending", ", ".join(self.state.pending_confirmations) or "None")
        t.add_row("Missing required", ", ".join(self.missing_required_fields()) or "None")
        t.add_row("Planning rounds", str(self.state.max_planning_rounds))
        t.add_row("Known issues", str(len(self.state.issue_ledger)))
        t.add_row("Best score", f"{self.state.best_audit.get('score', 0):.2f}" if self.state.best_audit else "N/A")
        t.add_row("Approved PDF", self.state.final_pdf_path or "None")
        self.console.print(t)

    # -----------------------------------------------------
    # Contract + structured memory
    # -----------------------------------------------------

    def set_contract_field(self, field_name: str, value: str, source: str, confirmed: bool, rationale: str) -> None:
        if field_name not in self.state.requirement_contract:
            return
        self.state.requirement_contract[field_name] = RequirementField(
            value=str(value).strip(),
            source=str(source).strip(),
            confirmed=bool(confirmed),
            rationale=str(rationale).strip(),
            updated_at=now_iso(),
        )
        note_path = CONTRACT_TO_NOTE_PATH.get(field_name)
        if note_path:
            deep_set(self.state.requirements, note_path, str(value).strip())

    def confirm_fields(self, fields: List[str]) -> None:
        for f in fields:
            if f in self.state.requirement_contract:
                self.state.requirement_contract[f].confirmed = True
                self.state.requirement_contract[f].updated_at = now_iso()

    def missing_required_fields(self) -> List[str]:
        out = []
        for f in USER_ONLY_REQUIRED_FIELDS:
            item = self.state.requirement_contract[f]
            if not item.value.strip() or not item.confirmed:
                out.append(f)
        return out

    def all_required_locked(self) -> bool:
        return len(self.missing_required_fields()) == 0

    def contract_snapshot(self) -> Dict[str, Any]:
        return {k: asdict(v) for k, v in self.state.requirement_contract.items()}

    def frozen_contract(self) -> Dict[str, Any]:
        return {k: asdict(v) for k, v in self.state.requirement_contract.items() if v.value.strip()}

    def requirement_summary_paragraphs(self) -> List[str]:
        out: List[str] = []
        for section_name in ["project", "frontend", "backend", "security", "data", "devops", "constraints"]:
            section = self.state.requirements.get(section_name, {})
            if not isinstance(section, dict) or not section:
                continue
            facts = []
            for k, v in section.items():
                if isinstance(v, dict):
                    continue
                facts.append(f"{k.replace('_', ' ').title()}: {v}")
            if facts:
                out.append(f"<b>{section_name.title()}</b> — " + "; ".join(facts))
        if not out:
            out.append("The validated plan was generated from the confirmed requirement contract.")
        return out

    def fill_internal_defaults(self) -> None:
        defaults = {
            "future_scope": "Derive post-MVP enhancements internally from the requested product direction, including richer capabilities after the first stable release.",
            "constraints": "Assume a final-year-project context with strong emphasis on quality, maintainability, and implementation readiness under limited resources.",
            "observability_baseline": "Structured application logs, error tracking, request metrics, latency monitoring, traces, uptime alerts, audit events, and an admin-visible operations dashboard.",
            "execution_preference": "Prioritize correctness, completeness, maintainability, and secure implementation readiness over brevity.",
            "llm_integration": "Use secure backend-managed GPT-compatible integration through an adapter layer, never direct browser-side secret exposure.",
            "compliance_context": "Adopt privacy-by-design baseline with deletion support, retention enforcement, secure secret storage, and explicit handling of user data and logs.",
        }
        for field_name, value in defaults.items():
            current = self.state.requirement_contract[field_name]
            if not current.value.strip():
                self.set_contract_field(
                    field_name=field_name,
                    value=value,
                    source="system_default_for_planning",
                    confirmed=True,
                    rationale="Internal planning default; not a user-blocking requirement.",
                )

    def wants_planning_transition(self, text: str) -> bool:
        t = text.lower().strip()
        if positive_reply(t):
            return True
        return any(term in t for term in PLANNING_INTENT_TERMS)

    def maybe_compact_context(self) -> None:
        if len(self.state.dialogue) < 24:
            return
        payload = {
            "current_summary": self.state.context_summary,
            "older_dialogue": [asdict(x) for x in self.state.dialogue[:-10]],
            "requirement_contract": self.contract_snapshot(),
            "requirements": self.state.requirements,
        }
        result = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ContextCompactor"],
            payload,
            max_tokens=1000,
            reasoning=True,
        )
        summary = result.get("summary")
        if summary:
            self.state.context_summary = str(summary)
            self.state.dialogue = self.state.dialogue[-10:]

    # -----------------------------------------------------
    # Tool calling for requirement phase
    # -----------------------------------------------------

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "phase": self.state.phase,
            "active_agent": self.state.active_agent,
            "context_summary": self.state.context_summary,
            "requirement_contract": self.contract_snapshot(),
            "pending_confirmations": self.state.pending_confirmations,
            "requirements": self.state.requirements,
            "requirement_status": self.state.requirement_status,
            "issue_ledger": self.state.issue_ledger,
            "revision_memory": self.state.revision_memory,
            "pass_threshold": self.state.pass_threshold,
            "debug_mode": self.state.debug_mode,
        }

    def build_agent_messages(self, agent_name: str) -> List[Dict[str, Any]]:
        snapshot = compact_json(self.state_snapshot(), 14000)
        messages = [
            {"role": "system", "content": GLOBAL_SYSTEM},
            {"role": "system", "content": f"Current agent: {agent_name}\n\n{AGENT_PROMPTS[agent_name]}"},
            {"role": "system", "content": f"Shared state snapshot:\n{snapshot}"},
        ]
        messages.extend(self.dialogue_messages())
        return messages

    def schema(self, name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def tool_schemas(self, agent_name: str) -> List[Dict[str, Any]]:
        valid_targets = [a for a in ALL_AGENTS if a != agent_name]

        return [
            self.schema(
                "inspect_contract",
                "Inspect the canonical requirement contract or one field.",
                {"field": {"type": "string"}},
                [],
            ),
            self.schema(
                "inspect_requirement_notes",
                "Inspect the rich structured requirement notes or one section.",
                {"section": {"type": "string"}},
                [],
            ),
            self.schema(
                "upsert_contract_field",
                "Write or update a canonical requirement contract field.",
                {
                    "field": {"type": "string"},
                    "value": {"type": "string"},
                    "rationale": {"type": "string"},
                    "confirmed": {"type": "boolean"},
                    "needs_confirmation": {"type": "boolean"},
                },
                ["field", "value", "rationale", "confirmed", "needs_confirmation"],
            ),
            self.schema(
                "confirm_contract_fields",
                "Confirm one or more contract fields after explicit user confirmation.",
                {
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                ["fields"],
            ),
            self.schema(
                "upsert_requirement_note",
                "Write or update a richer structured requirement note.",
                {
                    "path": {"type": "string"},
                    "value": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                ["path", "value", "rationale"],
            ),
            self.schema(
                "log_thinking",
                "Show concise summarized reasoning in the terminal.",
                {
                    "summary": {"type": "string"},
                    "confidence": {"type": "number"},
                    "next_action": {"type": "string"},
                },
                ["summary", "confidence", "next_action"],
            ),
            self.schema(
                "consult_reasoner",
                "Consult a reasoner or specialist for deeper analysis.",
                {
                    "agent": {"type": "string", "enum": valid_targets},
                    "task": {"type": "string"},
                    "deliverable": {"type": "string"},
                },
                ["agent", "task", "deliverable"],
            ),
            self.schema(
                "delegate_to",
                "Transfer control to another specialist agent.",
                {
                    "agent": {"type": "string", "enum": valid_targets},
                    "objective": {"type": "string"},
                    "reason": {"type": "string"},
                },
                ["agent", "objective", "reason"],
            ),
            self.schema(
                "set_readiness",
                "Update requirement completeness and readiness.",
                {
                    "ready_for_planning": {"type": "boolean"},
                    "completeness_score": {"type": "number"},
                    "summary": {"type": "string"},
                },
                ["ready_for_planning", "completeness_score", "summary"],
            ),
            self.schema(
                "advance_phase",
                "Advance to planning when the mandatory requirement contract is ready.",
                {
                    "target_phase": {"type": "string", "enum": [PHASE_PLANNING]},
                    "reason": {"type": "string"},
                },
                ["target_phase", "reason"],
            ),
        ]

    def execute_tool(self, caller: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "inspect_contract":
            field_name = str(args.get("field", "")).strip()
            if field_name:
                return {"ok": True, "field": field_name, "data": asdict(self.state.requirement_contract.get(field_name, RequirementField()))}
            return {"ok": True, "contract": self.contract_snapshot()}

        if tool_name == "inspect_requirement_notes":
            section = str(args.get("section", "")).strip()
            if section:
                return {"ok": True, "section": section, "data": self.state.requirements.get(section, {})}
            return {"ok": True, "requirements": self.state.requirements}

        if tool_name == "upsert_contract_field":
            field_name = str(args.get("field", "")).strip()
            value = str(args.get("value", "")).strip()
            rationale = str(args.get("rationale", "")).strip()
            confirmed = bool(args.get("confirmed", False))
            needs_confirmation = bool(args.get("needs_confirmation", False))

            if field_name not in self.state.requirement_contract or not value:
                return {"ok": False, "error": "Invalid field or empty value."}

            self.set_contract_field(field_name, value, caller, confirmed, rationale)

            if needs_confirmation or not confirmed:
                if field_name in USER_ONLY_REQUIRED_FIELDS and field_name not in self.state.pending_confirmations:
                    self.state.pending_confirmations.append(field_name)
            else:
                self.state.pending_confirmations = [f for f in self.state.pending_confirmations if f != field_name]

            return {"ok": True, "field": field_name}

        if tool_name == "confirm_contract_fields":
            fields = [f for f in ensure_list_of_str(args.get("fields")) if f in self.state.requirement_contract]
            self.confirm_fields(fields)
            self.state.pending_confirmations = [f for f in self.state.pending_confirmations if f not in fields]
            return {"ok": True, "confirmed": fields}

        if tool_name == "upsert_requirement_note":
            path = str(args.get("path", "")).strip()
            value = str(args.get("value", "")).strip()
            rationale = str(args.get("rationale", "")).strip()
            if not path or not value:
                return {"ok": False, "error": "Path and value are required."}
            deep_set(self.state.requirements, path, value)
            return {"ok": True, "path": path, "rationale": rationale}

        if tool_name == "log_thinking":
            summary = str(args.get("summary", "")).strip()
            confidence = args.get("confidence")
            next_action = str(args.get("next_action", "")).strip()
            if self.state.debug_mode:
                try:
                    conf = float(confidence) if confidence is not None else None
                except Exception:
                    conf = None
                self.thinking(caller, summary, next_action, conf)
            return {"ok": True}

        if tool_name == "consult_reasoner":
            target = str(args.get("agent", "")).strip()
            task = str(args.get("task", "")).strip()
            deliverable = str(args.get("deliverable", "")).strip()
            if target not in ALL_AGENTS:
                return {"ok": False, "error": "Invalid target agent."}
            result = self.consult_direct(target, task, deliverable)
            return {"ok": True, "agent": target, "result": result}

        if tool_name == "delegate_to":
            target = str(args.get("agent", "")).strip()
            if target not in ALL_AGENTS:
                return {"ok": False, "error": "Invalid delegate target."}
            self.state.active_agent = target
            return {
                "ok": True,
                "delegated_to": target,
                "objective": str(args.get("objective", "")),
                "reason": str(args.get("reason", "")),
            }

        if tool_name == "set_readiness":
            self.state.requirement_status = {
                "ready_for_planning": bool(args.get("ready_for_planning", False)),
                "completeness_score": float(args.get("completeness_score", 0.0)),
                "summary": str(args.get("summary", "")).strip(),
                "last_updated": now_iso(),
            }
            return {"ok": True}

        if tool_name == "advance_phase":
            target_phase = str(args.get("target_phase", "")).strip()
            if target_phase != PHASE_PLANNING:
                return {"ok": False, "error": "Unsupported phase transition."}
            if not self.all_required_locked():
                return {"ok": False, "error": "Mandatory blocker fields are not all confirmed."}
            self.fill_internal_defaults()
            self.state.phase = PHASE_PLANNING
            self.state.active_agent = "ArchitectAgent"
            self.state.pending_confirmations = []
            return {"ok": True, "phase": self.state.phase}

        return {"ok": False, "error": f"Unknown tool: {tool_name}"}

    def single_requirement_step(self, agent_name: str) -> bool:
        messages = self.build_agent_messages(agent_name)

        for _ in range(self.state.max_tool_rounds):
            resp = self.llm.completion(
                model=self.llm.chat_deployment,
                messages=messages,
                tools=self.tool_schemas(agent_name),
                temperature=0.2,
                max_tokens=1500,
            )

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            assistant_message = {"role": "assistant", "content": msg.content or ""}

            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            messages.append(assistant_message)

            if tool_calls:
                active_before = self.state.active_agent
                phase_before = self.state.phase

                for tc in tool_calls:
                    result = self.execute_tool(
                        agent_name,
                        tc.function.name,
                        safe_json_loads(tc.function.arguments),
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

                if self.state.phase != phase_before:
                    return False
                if self.state.active_agent != active_before:
                    return False
                continue

            content = (msg.content or "").strip()
            if content:
                self.append_dialogue("assistant", content, agent_name)
                self.panel(agent_name, content, "green")
                return True

        return False

    def consult_direct(self, agent: str, task: str, deliverable: str = "") -> Dict[str, Any]:
        payload = {
            "phase": self.state.phase,
            "task": task,
            "deliverable": deliverable,
            "context_summary": self.state.context_summary,
            "requirement_contract": self.contract_snapshot(),
            "requirements": self.state.requirements,
            "requirement_status": self.state.requirement_status,
            "issue_ledger": self.state.issue_ledger,
            "revision_memory": self.state.revision_memory,
            "current_plan": self.state.current_plan,
            "best_plan": self.state.best_plan,
            "audit_history": self.state.audit_history[-3:],
        }
        result = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS[agent],
            payload,
            max_tokens=1200,
            reasoning=True,
        )
        short = result.get("summary") or result.get("next_focus") or result
        if self.state.debug_mode:
            self.thinking(agent, short, "consultation complete")
        return result

    # -----------------------------------------------------
    # Commands and run loop
    # -----------------------------------------------------

    def run(self) -> None:
        self.banner()
        welcome = (
            "Hi — I’ll help you define the project step by step. "
            "I’ll lock the mandatory requirement contract first, then move into internal planning and validation."
        )
        self.append_dialogue("assistant", welcome, "RequirementCoordinator")
        self.panel("RequirementCoordinator", welcome)

        while not self.state.shutdown:
            if self.state.internal_busy:
                continue

            user_text = input("> ").strip()
            if not user_text:
                continue

            if user_text.lower() in {"exit", "quit"}:
                self.state.shutdown = True
                self.panel("System", "Session ended.", "red")
                break

            if self.handle_command(user_text):
                continue

            self.append_dialogue("user", user_text)
            self.maybe_compact_context()

            try:
                self.handle_turn(user_text)
            except Exception as e:
                self.panel("ERROR", f"{e}\n\n{traceback.format_exc()}", "red")

    def handle_command(self, text: str) -> bool:
        if not text.startswith(":"):
            return False

        parts = text.split()
        cmd = parts[0].lower()

        if cmd == ":threshold" and len(parts) == 2:
            try:
                value = float(parts[1])
                self.state.pass_threshold = max(7.0, min(10.0, value))
                self.panel("System", f"Pass threshold set to {self.state.pass_threshold:.2f}.", "cyan")
            except Exception:
                self.panel("System", "Invalid threshold value.", "red")
            return True

        if cmd == ":rounds" and len(parts) == 2:
            try:
                value = int(parts[1])
                self.state.max_planning_rounds = max(1, min(10, value))
                self.panel("System", f"Planning rounds set to {self.state.max_planning_rounds}.", "cyan")
            except Exception:
                self.panel("System", "Invalid round count.", "red")
            return True

        if cmd == ":debug" and len(parts) == 2:
            self.state.debug_mode = parts[1].lower() in {"on", "true", "1"}
            self.panel("System", f"Debug mode set to {self.state.debug_mode}.", "cyan")
            return True

        if cmd == ":thinking" and len(parts) == 2:
            self.state.show_internal_panels = parts[1].lower() in {"on", "true", "1"}
            self.panel("System", f"Internal thinking panels set to {self.state.show_internal_panels}.", "cyan")
            return True

        if cmd == ":status":
            self.show_status()
            return True

        if cmd == ":export":
            if not self.state.best_plan or not self.state.best_audit:
                self.panel("System", "There is no approved plan to export yet.", "red")
            else:
                self.generate_report_and_export()
                self.panel("System", f"PDF exported:\n{self.state.final_pdf_path}", "cyan")
            return True

        self.panel("System", "Unknown command.", "red")
        return True

    def handle_turn(self, user_text: str) -> None:
        if self.state.phase == PHASE_REQUIREMENTS:
            self.handle_requirement_turn(user_text)
            return

        if self.state.phase in {PHASE_APPROVED, PHASE_DEVELOPMENT}:
            msg = (
                f"The validated plan is already approved.\n"
                f"PDF: {self.state.final_pdf_path or 'not exported yet'}\n\n"
                "Use :export to regenerate the report, or start a new session for a new project."
            )
            self.panel("System", msg, "cyan")
            return

    # -----------------------------------------------------
    # Requirement phase
    # -----------------------------------------------------

    def handle_requirement_turn(self, user_text: str) -> None:
        normalized = user_text.lower().strip()

        if normalized in {"approve requirements", "approve contract", "finalize requirements"}:
            if self.all_required_locked():
                self.fill_internal_defaults()
                self.state.requirement_status = {
                    "ready_for_planning": True,
                    "completeness_score": 1.0,
                    "summary": "Mandatory requirement contract fully locked.",
                    "last_updated": now_iso(),
                }
                self.state.phase = PHASE_PLANNING
                self.state.active_agent = "ArchitectAgent"
                self.panel(
                    "RequirementCoordinator",
                    "Perfect — the mandatory requirement contract is locked. I’m moving to the internal planning and validation phase now.",
                    "green",
                )
                self.run_governance_cycle()
            else:
                self.panel(
                    "RequirementCoordinator",
                    f"We’re close, but these mandatory fields still need confirmation: {', '.join(self.missing_required_fields())}.",
                    "yellow",
                )
            return

        hops = 0
        while hops < self.state.max_requirement_hops and self.state.phase == PHASE_REQUIREMENTS:
            emitted = self.single_requirement_step(self.state.active_agent)
            if emitted:
                break
            hops += 1

        if self.all_required_locked():
            self.state.requirement_status = {
                "ready_for_planning": True,
                "completeness_score": 1.0,
                "summary": "Mandatory requirement contract fully locked.",
                "last_updated": now_iso(),
            }

            if self.wants_planning_transition(user_text):
                self.fill_internal_defaults()
                self.state.phase = PHASE_PLANNING
                self.state.active_agent = "ArchitectAgent"
                self.panel(
                    "RequirementCoordinator",
                    "Perfect — the required project requirements are now locked. I’m moving to the internal planning and validation phase now.",
                    "green",
                )
                self.run_governance_cycle()
                return

            self.panel(
                "RequirementCoordinator",
                "Perfect — the essential requirements are fully locked. Reply yes when you want me to move straight into the internal planning and validation phase.",
                "green",
            )
            return

        if self.state.pending_confirmations:
            self.panel(
                "RequirementCoordinator",
                "I have a few proposed values that still need your confirmation before planning can begin: "
                + ", ".join(self.state.pending_confirmations),
                "cyan",
            )

    # -----------------------------------------------------
    # Planning swarm
    # -----------------------------------------------------

    def token_budget(self, purpose: str) -> int:
        budgets = {
            "medium": {"analysis": 1800, "plan": 3200, "report": 3800},
            "long": {"analysis": 2600, "plan": 4400, "report": 5200},
            "extreme": {"analysis": 3400, "plan": 5800, "report": 6800},
        }
        return budgets.get(self.state.report_depth, budgets["long"]).get(purpose, 3000)

    def run_governance_cycle(self) -> None:
        self.state.internal_busy = True
        self.console.print(Rule("[bold magenta]Internal Planning & Audit Started"))

        try:
            for round_no in range(1, self.state.max_planning_rounds + 1):
                self.console.print(Rule(f"[bold cyan]Architecture Round {round_no}"))

                specialist_reviews = self.run_specialist_reasoners(round_no)
                plan = self.architect_generate(round_no, specialist_reviews)
                audit = self.auditor_validate(round_no, plan, specialist_reviews)

                self.state.specialist_history.append(
                    {"round": round_no, "reviews": specialist_reviews, "timestamp": now_iso()}
                )
                self.state.audit_history.append(deepcopy(audit))
                self.state.current_plan = deepcopy(plan)
                self.state.current_audit = deepcopy(audit)

                write_json(Path(self.state.artifacts_dir) / f"specialists_round_{round_no}.json", specialist_reviews)
                write_json(Path(self.state.artifacts_dir) / f"plan_round_{round_no}.json", plan)
                write_json(Path(self.state.artifacts_dir) / f"audit_round_{round_no}.json", audit)

                self.update_issue_ledger(audit)
                self.update_revision_memory(plan, audit)
                self.update_best_artifact(plan, audit)

                self.show_round_tables(round_no, plan, audit)

                if audit["passed"]:
                    self.state.phase = PHASE_APPROVED
                    self.generate_report_and_export()
                    self.panel(
                        "APPROVED",
                        f"Validated plan approved with score {audit['score']:.2f}\n\nPDF: {self.state.final_pdf_path}",
                        "green",
                    )
                    self.state.phase = PHASE_DEVELOPMENT
                    self.present_development_handoff()
                    return

                self.panel(
                    "Revision In Progress",
                    "The planning swarm is revising the architecture internally based on cumulative audit feedback.",
                    "yellow",
                )

            self.state.phase = PHASE_REQUIREMENTS
            self.panel(
                "Planning",
                "The architecture did not reach approval within the current round limit. Refine the requirements, increase rounds, or lower the threshold.",
                "red",
            )
        finally:
            self.state.internal_busy = False

    def run_specialist_reasoners(self, round_no: int) -> Dict[str, Any]:
        base_payload = {
            "round": round_no,
            "frozen_requirement_contract": self.frozen_contract(),
            "requirements": self.state.requirements,
            "issue_ledger": self.state.issue_ledger,
            "revision_memory": self.state.revision_memory,
            "previous_audits": self.state.audit_history[-3:],
            "best_audit": self.state.best_audit,
        }

        product = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ProductReasoner"],
            base_payload,
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        if self.state.debug_mode:
            self.thinking("ProductReasoner", product.get("summary", "Product review complete."), "handoff to swarm")

        architect = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ArchitectReasoner"],
            {**base_payload, "product_review": product},
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        if self.state.debug_mode:
            self.thinking("ArchitectReasoner", architect.get("summary", "Architecture review complete."), "handoff to swarm")

        security = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["SecurityReasoner"],
            {**base_payload, "product_review": product, "architect_review": architect},
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        if self.state.debug_mode:
            self.thinking("SecurityReasoner", security.get("summary", "Security review complete."), "handoff to swarm")

        constraints = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ConstraintReasoner"],
            {
                **base_payload,
                "product_review": product,
                "architect_review": architect,
                "security_review": security,
            },
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        if self.state.debug_mode:
            self.thinking("ConstraintReasoner", constraints.get("summary", "Constraint review complete."), "handoff to swarm")

        critic = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["CriticReasoner"],
            {
                **base_payload,
                "product_review": product,
                "architect_review": architect,
                "security_review": security,
                "constraint_review": constraints,
            },
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        if self.state.debug_mode:
            self.thinking("CriticReasoner", critic.get("summary", "Critic review complete."), "guide ArchitectAgent")

        return {
            "product": product,
            "architect_reasoner": architect,
            "security": security,
            "constraints": constraints,
            "critic": critic,
        }

    def architect_generate(self, round_no: int, specialist_reviews: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "round": round_no,
            "frozen_requirement_contract": self.frozen_contract(),
            "requirements": self.state.requirements,
            "specialist_reviews": specialist_reviews,
            "issue_ledger": self.state.issue_ledger,
            "revision_memory": self.state.revision_memory,
            "accepted_exceptions": {k: asdict(v) for k, v in self.state.accepted_exceptions.items()},
            "previous_audits": self.state.audit_history[-3:],
            "previous_plan": self.state.current_plan,
            "best_plan": self.state.best_plan,
        }

        result = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ArchitectAgent"],
            payload,
            max_tokens=self.token_budget("plan"),
            reasoning=True,
        )

        summary = result.get("executive_summary") or result.get("thinking_summary") or "Architecture draft generated."
        if self.state.debug_mode:
            self.thinking("ArchitectAgent", summary, "submit to AuditorAgent")

        return self.normalize_plan(result, specialist_reviews)

    def normalize_plan(self, raw: Dict[str, Any], specialist_reviews: Dict[str, Any]) -> Dict[str, Any]:
        contract = self.frozen_contract()

        def c(field: str, fallback: str = "Derived from confirmed requirements.") -> str:
            item = contract.get(field, {})
            return str(item.get("value") or fallback)

        title = str(raw.get("title") or "Validated Architecture Plan")
        title = title.replace("Round 1", "").replace("Round 2", "").replace("Round 3", "").replace("Round 4", "").strip(" -")
        if not title:
            title = "Validated Architecture Plan"

        plan = {
            "title": title,
            "executive_summary": raw.get("executive_summary") or "Detailed validated architecture plan generated from the confirmed requirement contract.",
            "architecture_overview": raw.get("architecture_overview") or {
                "system_style": "Modular cloud-native application",
                "primary_goal": c("project_goal"),
                "target_users": c("target_users"),
                "access_model": c("access_model"),
            },
            "technology_stack": raw.get("technology_stack") or {
                "frontend": c("frontend_stack"),
                "backend": c("backend_stack"),
                "data_platform": c("data_platform"),
                "hosting_target": c("hosting_target"),
                "llm_integration": c("llm_integration", "Secure backend-managed GPT-compatible integration"),
            },
            "functional_feature_map": raw.get("functional_feature_map") or {
                "mvp_scope": c("mvp_scope"),
                "expanded_scope": c("feature_scope"),
                "future_scope": c("future_scope", "Additional advanced features after MVP stabilization"),
            },
            "system_components": raw.get("system_components") or [
                {"name": "Web Client", "responsibility": "Interactive UI, authentication UI, settings, and feature access"},
                {"name": "API Gateway", "responsibility": "Authentication, validation, routing, throttling, and request governance"},
                {"name": "Application Service Layer", "responsibility": "Business logic, orchestration, and use-case execution"},
                {"name": "LLM Adapter", "responsibility": "Provider abstraction, retries, safety checks, and token accounting"},
                {"name": "Data Layer", "responsibility": "Persistence for users, sessions, messages, metadata, and audit events"},
                {"name": "Observability Layer", "responsibility": "Logs, metrics, traces, alerts, and audit monitoring"},
            ],
            "workflows": raw.get("workflows") or {
                "primary_flows": [
                    "User authentication or approved guest access",
                    "Feature request submission and validation",
                    "Prompt or request assembly with policy checks",
                    "Core service processing and optional LLM inference",
                    "Persistence, monitoring, and operational oversight",
                ]
            },
            "data_model": raw.get("data_model") or {
                "entities": ["User", "Session", "Message", "FeatureArtifact", "Feedback", "UsageEvent", "AuditEvent"],
                "storage_strategy": c("data_platform"),
                "retention_policy": c("privacy_retention_policy"),
            },
            "api_design": raw.get("api_design") or {
                "style": "REST plus streaming where needed",
                "endpoints": ["/api/auth", "/api/users", "/api/sessions", "/api/messages", "/api/stream", "/api/feedback"],
            },
            "security_and_compliance": raw.get("security_and_compliance") or {
                "baseline": c("security_baseline"),
                "privacy": c("privacy_retention_policy"),
                "compliance_context": c("compliance_context", "Privacy-by-design baseline"),
            },
            "deployment_and_operations": raw.get("deployment_and_operations") or {
                "hosting_target": c("hosting_target"),
                "observability_baseline": c("observability_baseline"),
                "ops_model": "Phased deployment with monitoring, rollback, and cost tracking",
            },
            "observability": raw.get("observability") or {
                "baseline": c("observability_baseline"),
            },
            "cost_and_scaling": raw.get("cost_and_scaling") or {
                "cost_position": "Usage-driven, especially if external model APIs are used",
                "scaling_direction": "Horizontal application scaling with managed services, quotas, and caching",
            },
            "phased_implementation": raw.get("phased_implementation") or {
                "phase_1": c("mvp_scope"),
                "phase_2": c("future_scope", "Expanded features after MVP stabilization"),
            },
            "development_guidelines": raw.get("development_guidelines") or [
                "Keep service boundaries explicit",
                "Design data contracts before implementation",
                "Automate tests early",
                "Never expose model secrets in the frontend",
            ],
            "risks_and_tradeoffs": raw.get("risks_and_tradeoffs") or {
                "risks": ["API cost growth", "Public abuse pressure", "Latency variability", "Feature complexity"],
                "tradeoffs": "A faster MVP may reduce governance depth, while stronger governance increases implementation overhead.",
            },
            "open_questions_resolved": raw.get("open_questions_resolved") or specialist_reviews.get("critic", {}),
            "generated_at": now_iso(),
        }
        return plan

    def auditor_validate(self, round_no: int, plan: Dict[str, Any], specialist_reviews: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "round": round_no,
            "frozen_requirement_contract": self.frozen_contract(),
            "requirements": self.state.requirements,
            "accepted_exceptions": {k: asdict(v) for k, v in self.state.accepted_exceptions.items()},
            "issue_ledger": self.state.issue_ledger,
            "revision_memory": self.state.revision_memory,
            "specialist_reviews": specialist_reviews,
            "plan": plan,
            "pass_threshold": self.state.pass_threshold,
            "best_audit": self.state.best_audit,
        }

        result = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["AuditorAgent"],
            payload,
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )

        if self.state.debug_mode:
            self.thinking("AuditorAgent", result.get("summary", "Audit complete."), "approve or request revision")

        score = self.normalize_score(result.get("score", 6.5))
        strengths = ensure_list_of_str(result.get("strengths"))
        concerns = ensure_list_of_str(result.get("concerns"))
        blocking_issues = ensure_list_of_str(result.get("blocking_issues"))
        recommendations = ensure_list_of_str(result.get("recommendations"))
        issue_updates = ensure_list(result.get("issue_updates"))
        requirement_conflicts = [
            item for item in ensure_list(result.get("requirement_conflicts"))
            if isinstance(item, dict)
        ]

        unresolved_critical = any(
            str(item.get("severity", "")).lower() == "critical"
            and str(item.get("status", "")).lower() != "resolved"
            for item in issue_updates if isinstance(item, dict)
        )

        raw_passed = bool(result.get("passed", False))
        passed = raw_passed and score >= self.state.pass_threshold and not unresolved_critical

        previous_best = float(self.state.best_audit.get("score", 0.0)) if self.state.best_audit else 0.0
        if previous_best > 0 and score + 0.7 < previous_best:
            recommendations.append(
                "Score regression detected relative to the prior best result; retain the stronger artifact unless a new severe flaw clearly justifies the drop."
            )

        return {
            "round": round_no,
            "score": score,
            "passed": passed,
            "summary": str(result.get("summary") or "Audit completed."),
            "strengths": strengths,
            "concerns": concerns,
            "blocking_issues": blocking_issues,
            "recommendations": unique_strs(recommendations),
            "issue_updates": issue_updates,
            "requirement_conflicts": requirement_conflicts,
            "timestamp": now_iso(),
            "raw": result,
        }

    def normalize_score(self, value: Any) -> float:
        try:
            score = float(value)
        except Exception:
            score = 6.5
        return max(0.0, min(10.0, score))

    def update_issue_ledger(self, audit: Dict[str, Any]) -> None:
        for item in ensure_list(audit.get("issue_updates")):
            if not isinstance(item, dict):
                continue
            issue_id = str(item.get("id") or item.get("issue_id") or "").strip()
            if not issue_id:
                continue

            existing = self.state.issue_ledger.get(issue_id, {})
            history = ensure_list(existing.get("history"))
            history.append({
                "round": audit.get("round"),
                "status": item.get("status", ""),
                "severity": item.get("severity", ""),
                "detail": item.get("detail", ""),
                "timestamp": now_iso(),
            })

            self.state.issue_ledger[issue_id] = {
                "id": issue_id,
                "title": str(item.get("title") or existing.get("title") or issue_id),
                "severity": str(item.get("severity") or existing.get("severity") or "medium").lower(),
                "status": str(item.get("status") or existing.get("status") or "unresolved").lower(),
                "detail": str(item.get("detail") or existing.get("detail") or ""),
                "last_seen_round": audit.get("round"),
                "history": history,
                "updated_at": now_iso(),
            }

    def update_revision_memory(self, plan: Dict[str, Any], audit: Dict[str, Any]) -> None:
        resolved: List[str] = []
        unresolved: List[str] = []
        for issue_id, issue in self.state.issue_ledger.items():
            if str(issue.get("status", "")).lower() == "resolved":
                resolved.append(issue_id)
            else:
                unresolved.append(issue_id)

        self.state.revision_memory = {
            "last_round": audit.get("round"),
            "last_score": audit.get("score"),
            "resolved_issue_ids": sorted(resolved),
            "unresolved_issue_ids": sorted(unresolved),
            "latest_recommendations": audit.get("recommendations", []),
            "latest_plan_title": plan.get("title"),
        }

    def update_best_artifact(self, plan: Dict[str, Any], audit: Dict[str, Any]) -> None:
        candidate_score = float(audit.get("score", 0.0))
        current_best = float(self.state.best_audit.get("score", 0.0)) if self.state.best_audit else 0.0

        better = False
        if not self.state.best_plan:
            better = True
        elif candidate_score > current_best:
            better = True
        elif candidate_score == current_best:
            current_best_blocking = len(self.state.best_audit.get("blocking_issues", [])) if self.state.best_audit else 999
            candidate_blocking = len(audit.get("blocking_issues", []))
            if candidate_blocking < current_best_blocking:
                better = True

        if better:
            self.state.best_plan = deepcopy(plan)
            self.state.best_audit = deepcopy(audit)

    def show_round_tables(self, round_no: int, plan: Dict[str, Any], audit: Dict[str, Any]) -> None:
        pt = Table(title=f"Architecture Draft Round {round_no}", box=box.SIMPLE_HEAVY)
        pt.add_column("Field", style="cyan", width=22)
        pt.add_column("Value")
        pt.add_row("Title", str(plan.get("title")))
        pt.add_row("Summary", as_text(plan.get("executive_summary"), 320))
        pt.add_row(
            "Top-level sections",
            ", ".join(k for k in plan.keys() if k not in {"title", "executive_summary", "generated_at"}),
        )
        self.console.print(pt)

        at = Table(title=f"Audit Result Round {round_no}", box=box.SIMPLE_HEAVY)
        at.add_column("Metric", style="magenta", width=20)
        at.add_column("Value")
        at.add_row("Score", f"{audit['score']:.2f}")
        at.add_row("Passed", str(audit["passed"]))
        at.add_row("Threshold", f"{self.state.pass_threshold:.2f}")
        at.add_row("Summary", as_text(audit.get("summary"), 320))
        at.add_row("Recommendations", str(len(audit.get("recommendations", []))))
        self.console.print(at)

        if self.state.debug_mode and audit.get("blocking_issues"):
            self.panel("Internal Blocking Issues", "\n".join(f"- {x}" for x in audit["blocking_issues"]), "red")

    # -----------------------------------------------------
    # Final package
    # -----------------------------------------------------

    def generate_report_and_export(self) -> None:
        plan = self.state.best_plan or self.state.current_plan
        audit = self.state.best_audit or self.state.current_audit
        report = self.build_final_package(plan, audit)
        self.state.report_package = report
        self.state.final_pdf_path = self.export_pdf(report, plan, audit)
        write_json(Path(self.state.artifacts_dir) / "approved_report_package.json", report)

    def build_final_package(self, plan: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
        execution = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ExecutionPlannerAgent"],
            {
                "plan": plan,
                "audit": audit,
                "locked_contract": self.frozen_contract(),
                "requirements": self.state.requirements,
                "specialist_history": self.state.specialist_history,
            },
            max_tokens=self.token_budget("report"),
            reasoning=True,
        )

        tutor = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["TutorAgent"],
            {
                "plan": plan,
                "audit": audit,
                "execution": execution,
                "locked_contract": self.frozen_contract(),
            },
            max_tokens=self.token_budget("report"),
            reasoning=True,
        )

        qa = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["QAEngineerAgent"],
            {
                "plan": plan,
                "audit": audit,
                "execution": execution,
                "locked_contract": self.frozen_contract(),
            },
            max_tokens=self.token_budget("report"),
            reasoning=True,
        )

        narrative = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["NarrativeWriterAgent"],
            {
                "plan": plan,
                "audit": audit,
                "execution": execution,
                "tutor": tutor,
                "qa": qa,
                "locked_contract": self.frozen_contract(),
                "requirements": self.state.requirements,
            },
            max_tokens=self.token_budget("report"),
            reasoning=True,
        )

        development = {
            "development_summary": "The project now moves into implementation using the approved validated architecture package.",
            "first_week_plan": execution.get("implementation_phases", [])[:1] if isinstance(execution.get("implementation_phases"), list) else [],
            "coding_sequence": tutor.get("coding_order", []),
            "practical_starting_point": "Start by scaffolding the repository, baseline config, auth flow, and core data contracts before feature implementation.",
        }

        package = self.normalize_report_package(
            {
                "title": narrative.get("title") or plan.get("title") or "Validated Architecture Plan",
                "executive_summary": narrative.get("executive_summary") or plan.get("executive_summary") or audit.get("summary"),
                "sections": narrative.get("sections") or {},
                "execution": execution,
                "tutor": tutor,
                "qa": qa,
                "development": development,
            },
            plan,
            audit,
        )
        self.state.development_package = development
        return package

    def normalize_report_package(self, package: Dict[str, Any], plan: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
        sections = package.get("sections") or {}
        if not isinstance(sections, dict):
            sections = {}

        execution = package.get("execution") or {}
        tutor = package.get("tutor") or {}
        qa = package.get("qa") or {}

        defaults = {
            "overview": as_text(plan.get("architecture_overview"), 50000),
            "requirement_interpretation": self.contract_summary_text(),
            "stack_rationale": as_text(plan.get("technology_stack"), 50000),
            "architecture": as_text(plan.get("architecture_overview"), 50000),
            "component_design": as_text(plan.get("system_components"), 50000),
            "workflow_design": as_text(plan.get("workflows"), 50000),
            "data_model": as_text(plan.get("data_model"), 50000),
            "api_design": as_text(plan.get("api_design"), 50000),
            "security": as_text(plan.get("security_and_compliance"), 50000),
            "deployment": as_text(plan.get("deployment_and_operations"), 50000),
            "observability": as_text(plan.get("observability"), 50000),
            "cost_and_scaling": as_text(plan.get("cost_and_scaling"), 50000),
            "phased_implementation": as_text(execution, 50000),
            "development_playbook": as_text(tutor, 50000),
            "testing_validation": as_text(qa, 50000),
            "risks_tradeoffs": as_text(plan.get("risks_and_tradeoffs"), 50000),
            "final_notes": "This validated package is ready to guide implementation, testing, and phased rollout.",
        }

        for key, value in defaults.items():
            if not sections.get(key):
                sections[key] = value

        package["sections"] = sections
        package["title"] = str(package.get("title") or "Validated Architecture Plan")
        package["executive_summary"] = str(package.get("executive_summary") or "Validated architecture report.")
        return package

    def contract_summary_text(self) -> str:
        lines = ["Confirmed requirement contract:"]
        for k, v in self.state.requirement_contract.items():
            if v.value.strip():
                suffix = "confirmed" if v.confirmed else "pending"
                lines.append(f"- {k}: {v.value} ({suffix})")
        return "\n".join(lines)

    # -----------------------------------------------------
    # PDF export helpers
    # -----------------------------------------------------

    def export_pdf(self, report: Dict[str, Any], plan: Dict[str, Any], audit: Dict[str, Any]) -> str:
        out = Path(self.state.artifacts_dir) / f"validated_architecture_plan_{self.state.session_id[:8]}.pdf"

        doc = SimpleDocTemplate(
            str(out),
            pagesize=A4,
            rightMargin=15 * mm,
            leftMargin=15 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "titlex",
            parent=styles["Title"],
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=10,
        )
        h1 = ParagraphStyle(
            "h1x",
            parent=styles["Heading1"],
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#0B3B66"),
            spaceBefore=10,
            spaceAfter=6,
        )
        h2 = ParagraphStyle(
            "h2x",
            parent=styles["Heading2"],
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#1D4ED8"),
            spaceBefore=8,
            spaceAfter=4,
        )
        body = ParagraphStyle(
            "bodyx",
            parent=styles["BodyText"],
            fontSize=9.2,
            leading=13.5,
            alignment=TA_LEFT,
            textColor=colors.black,
            spaceAfter=5,
        )
        small = ParagraphStyle(
            "smallx",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=10,
            textColor=colors.HexColor("#475569"),
            spaceAfter=4,
        )

        story: List[Any] = []

        story.append(Paragraph(self.pdf_escape(report.get("title", "Validated Architecture Plan")), title_style))
        story.append(Paragraph("Final Validated Project Architecture Package", small))
        story.append(Spacer(1, 5))

        meta = RLTable(
            [
                ["Generated", now_iso()],
                ["Validation score", f"{audit.get('score', 0.0):.2f}"],
                ["Approval threshold", f"{self.state.pass_threshold:.2f}"],
                ["Planning rounds used", str(audit.get("round", 0))],
                ["Accepted exceptions", str(len(self.state.accepted_exceptions))],
            ],
            colWidths=[48 * mm, 128 * mm],
        )
        meta.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(meta)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Executive Summary", h1))
        for p in self.split_paragraphs(as_text(report.get("executive_summary", ""), 50000)):
            story.append(Paragraph(self.pdf_escape(p), body))

        story.append(Paragraph("Locked Requirements", h1))
        rows = [["Field", "Value", "Source", "Confirmed"]]
        for k, v in self.state.requirement_contract.items():
            if v.value.strip():
                rows.append([k, v.value[:120], v.source or "unknown", "Yes" if v.confirmed else "No"])
        req_table = RLTable(rows, colWidths=[38 * mm, 90 * mm, 35 * mm, 18 * mm])
        req_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(req_table)
        story.append(Spacer(1, 10))

        ordered_sections = [
            ("overview", "System Overview"),
            ("requirement_interpretation", "Requirements Interpretation"),
            ("stack_rationale", "Technology Stack and Rationale"),
            ("architecture", "Architecture Design"),
            ("component_design", "Component Design"),
            ("workflow_design", "Workflow Design"),
            ("data_model", "Data Model"),
            ("api_design", "API Design"),
            ("security", "Security and Compliance"),
            ("deployment", "Deployment and Operations"),
            ("observability", "Observability"),
            ("cost_and_scaling", "Cost and Scaling Strategy"),
            ("phased_implementation", "Detailed Execution Plan"),
            ("development_playbook", "Development Playbook"),
            ("testing_validation", "Testing and Validation"),
            ("risks_tradeoffs", "Risks and Trade-offs"),
            ("final_notes", "Final Notes"),
        ]

        report_sections = report.get("sections", {})
        for key, title in ordered_sections:
            story.append(Paragraph(title, h1))
            content = report_sections.get(key, "")
            for p in self.split_paragraphs(as_text(content, 80000)):
                story.append(Paragraph(self.pdf_escape(p), body))

            if key == "phased_implementation":
                self.append_execution_breakdown(story, report.get("execution", {}), h2, body)
            elif key == "development_playbook":
                self.append_feature_build_guides(story, report.get("tutor", {}), h2, body)
            elif key == "testing_validation":
                self.append_qa_sections(story, report.get("qa", {}), h2, body)

        if self.state.accepted_exceptions:
            story.append(PageBreak())
            story.append(Paragraph("Accepted Exceptions", h1))
            items = [f"{ex.issue_id}: {ex.reason}" for ex in self.state.accepted_exceptions.values()]
            story.append(self.bullet_list(items, body))

        if audit.get("recommendations"):
            story.append(Paragraph("Residual Recommendations", h1))
            story.append(self.bullet_list(audit["recommendations"], body))

        doc.build(story)
        return str(out.resolve())

    def split_paragraphs(self, text: str) -> List[str]:
        text = (text or "").replace("\r", "").strip()
        if not text:
            return []
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return parts if parts else [text]

    def append_execution_breakdown(self, story: List[Any], execution: Dict[str, Any], h2, body) -> None:
        if not execution:
            return

        overview = execution.get("execution_overview")
        if overview:
            story.append(Paragraph("Execution Overview", h2))
            for p in self.split_paragraphs(as_text(overview, 30000)):
                story.append(Paragraph(self.pdf_escape(p), body))

        phases = ensure_list(execution.get("implementation_phases"))
        if phases:
            story.append(Paragraph("Implementation Phases", h2))
            for idx, phase in enumerate(phases, start=1):
                if isinstance(phase, dict):
                    title = str(phase.get("phase") or phase.get("name") or f"Phase {idx}")
                    story.append(Paragraph(f"{idx}. {self.pdf_escape(title)}", body))
                    details = []
                    for key in ["objective", "deliverables", "tasks", "frontend", "backend", "data", "infra", "security", "qa", "done_criteria"]:
                        if phase.get(key):
                            details.append(f"{key}: {as_text(phase.get(key), 2000)}")
                    if details:
                        story.append(self.bullet_list(details, body))

        workstreams = ensure_list(execution.get("feature_workstreams"))
        if workstreams:
            story.append(Paragraph("Feature Workstreams", h2))
            for item in workstreams:
                if isinstance(item, dict):
                    name = str(item.get("feature") or item.get("name") or "Feature")
                    story.append(Paragraph(self.pdf_escape(name), body))
                    story.append(self.bullet_list([as_text(item, 2500)], body))

    def append_feature_build_guides(self, story: List[Any], tutor: Dict[str, Any], h2, body) -> None:
        if not tutor:
            return

        for key, title in [
            ("development_playbook", "Development Playbook"),
            ("coding_order", "Coding Order"),
            ("implementation_tips", "Implementation Tips"),
            ("common_mistakes", "Common Mistakes"),
        ]:
            value = tutor.get(key)
            if not value:
                continue
            story.append(Paragraph(title, h2))
            if isinstance(value, list):
                story.append(self.bullet_list([as_text(v, 2000) for v in value], body))
            else:
                for p in self.split_paragraphs(as_text(value, 40000)):
                    story.append(Paragraph(self.pdf_escape(p), body))

        guides = ensure_list(tutor.get("feature_build_guides"))
        if guides:
            story.append(Paragraph("Feature Build Guides", h2))
            for guide in guides:
                if isinstance(guide, dict):
                    name = str(guide.get("feature") or guide.get("name") or "Feature Guide")
                    story.append(Paragraph(self.pdf_escape(name), body))
                    story.append(self.bullet_list([as_text(guide, 2500)], body))

    def append_qa_sections(self, story: List[Any], qa: Dict[str, Any], h2, body) -> None:
        if not qa:
            return

        for key, title in [
            ("validation_strategy", "Validation Strategy"),
            ("test_layers", "Test Layers"),
            ("detailed_test_plan", "Detailed Test Plan"),
            ("acceptance_criteria", "Acceptance Criteria"),
            ("regression_strategy", "Regression Strategy"),
            ("release_readiness_checklist", "Release Readiness Checklist"),
        ]:
            value = qa.get(key)
            if not value:
                continue
            story.append(Paragraph(title, h2))
            if isinstance(value, list):
                story.append(self.bullet_list([as_text(v, 2000) for v in value], body))
            else:
                for p in self.split_paragraphs(as_text(value, 40000)):
                    story.append(Paragraph(self.pdf_escape(p), body))

    def bullet_list(self, items: List[str], body_style) -> ListFlowable:
        flow = [ListItem(Paragraph(self.pdf_escape(item), body_style)) for item in ensure_list_of_str(items)]
        return ListFlowable(flow, bulletType="bullet", leftIndent=14)

    def pdf_escape(self, text: str) -> str:
        return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")

    # -----------------------------------------------------
    # Development handoff
    # -----------------------------------------------------

    def present_development_handoff(self) -> None:
        dev = self.state.development_package or {}
        body = as_text(
            dev.get("development_summary", "The project now moves into the development phase."),
            2000,
        )
        body += "\n\nA full tutor-style implementation guide, QA strategy, and execution roadmap have been added to the approved PDF."
        self.panel("Development Phase", body, "cyan")


if __name__ == "__main__":
    app = GovernanceHybridApp()
    app.run()
