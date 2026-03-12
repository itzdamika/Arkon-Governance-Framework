from __future__ import annotations

import json
import math
import os
import re
import textwrap
import traceback
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table as RLTable,
    TableStyle,
)

try:
    from graphviz import Digraph
except Exception:
    Digraph = None

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
except Exception:
    PILImage = None
    ImageDraw = None
    ImageFont = None


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


def compact_json(obj: Any, limit: int = 12000) -> str:
    try:
        s = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "\n...<truncated>..."


def as_text(value: Any, limit: int = 5000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value
    else:
        s = compact_json(value, limit)
    return s if len(s) <= limit else s[:limit] + "..."


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


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text.strip("_") or "artifact"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_markdownish(text: str) -> str:
    text = text.replace("\r", "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_paragraphs(text: str) -> List[str]:
    text = clean_markdownish(text)
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts if parts else [text]


def json_to_lines(obj: Any, indent: int = 0) -> List[str]:
    pad = "  " * indent
    lines: List[str] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.extend(json_to_lines(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(json_to_lines(item, indent + 1))
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}{obj}")

    return lines


def obj_to_text(obj: Any, limit: int = 40000) -> str:
    if isinstance(obj, str):
        return obj
    return "\n".join(json_to_lines(obj))[:limit]


def positive_reply(text: str) -> bool:
    t = text.lower().strip()
    terms = [
        "yes", "y", "ok", "okay", "sure", "approved", "approve", "yes please",
        "go ahead", "looks good", "sounds good", "thats good", "that's good",
        "fine", "correct", "right", "perfect"
    ]
    return any(x in t for x in terms)


def negative_reply(text: str) -> bool:
    t = text.lower().strip()
    terms = ["no", "reject", "change", "not good", "dont", "don't", "not this", "wrong"]
    return any(x in t for x in terms)


# =========================================================
# Phases and fields
# =========================================================

PHASE_REQUIREMENTS = "REQUIREMENTS"
PHASE_REQUIREMENT_CONFIRMATION = "REQUIREMENT_CONFIRMATION"
PHASE_PLANNING = "PLANNING"
PHASE_CHANGE_REVIEW = "CHANGE_REVIEW"
PHASE_APPROVED = "APPROVED"
PHASE_DEVELOPMENT = "DEVELOPMENT"

MANDATORY_FIELDS = [
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
    "future_scope": "What features can wait for later phases?",
    "constraints": "Are there budget, time, team, complexity, or platform constraints?",
    "observability_baseline": "What monitoring, logging, tracing, and alerting should exist?",
    "execution_preference": "Should the plan optimize for speed, simplicity, low cost, scale, or compliance first?",
    "llm_integration": "What LLM or model access strategy should be used?",
    "compliance_context": "Are there compliance expectations such as GDPR, HIPAA, or internal policy controls?",
}


# =========================================================
# Prompts
# =========================================================

FRIENDLY_REQ_SYSTEM = """
You are RequirementCoordinator for a phase-based SDLC planning system.

Your visible behavior:
- be highly intelligent, warm, natural, and friendly
- produce exactly ONE assistant message for this turn
- do not sound robotic or bureaucratic
- do not dump all mandatory fields at once
- ask 1-3 natural clarifying questions at a time
- if the user does not know technical details, gently suggest sensible defaults
- if you suggest defaults, ask for confirmation in the SAME message
- do not mention hidden gates or mandatory tracking
- never create a second separate summary block

Your hidden behavior:
- planning must not start until all mandatory fields are captured and confirmed
- only propose a small cluster of technical decisions at a time
- preserve already confirmed user choices

Return JSON only with:
- thinking_summary
- assistant_message
- updates: list of {field, value, source, confirmed, rationale, needs_confirmation}
- pending_confirmations: list of field names that need confirmation after this turn
- ready_for_requirement_approval: boolean
"""

REQUIREMENT_CONFIRM_SYSTEM = """
You are RequirementCoordinator for a phase-based SDLC planning system.

You are handling the user's reply to previously proposed requirement choices.

Visible behavior:
- produce exactly ONE assistant message for this turn
- be warm, smart, and conversational
- if the user accepts the proposed choices, confirm them naturally and either ask the next missing question or tell them requirements are ready for approval
- if the user edits or rejects the choices, respond naturally and refine them
- do not output robotic confirmation panels

Return JSON only with:
- thinking_summary
- assistant_message
- action: one of ["approve_proposals", "revise_proposals", "treat_as_edit"]
- updates: list of {field, value, source, confirmed, rationale, needs_confirmation}
- pending_confirmations: list of field names that need confirmation after this turn
- ready_for_requirement_approval: boolean
"""

PRODUCT_REASONER_SYSTEM = """
You are ProductReasoner in a multi-agent SDLC planning swarm.

Analyze the confirmed requirement contract from a product and scope perspective.
Focus on value fit, completeness, blind spots, user experience needs, and product risks.
Be concrete and specific.

Return JSON only with:
- summary
- requirement_completeness_score
- coverage
- blindspots
- functional_gaps
- ux_considerations
- future_phase_candidates
- next_focus
"""

ARCHITECT_REASONER_SYSTEM = """
You are ArchitectReasoner in a multi-agent SDLC planning swarm.

Analyze the confirmed requirement contract and prior reviews from an architecture perspective.
Focus on feasibility, modularity, coherence, scalability, API direction, data flow, and deployment approach.

Return JSON only with:
- summary
- feasibility
- proposed_architecture_direction
- recommended_modules
- data_and_api_notes
- infrastructure_direction
- devops_direction
- design_principles
"""

SECURITY_REASONER_SYSTEM = """
You are SecurityReasoner in a multi-agent SDLC planning swarm.

Analyze the confirmed requirements and prior reviews from a security, privacy, and abuse-prevention perspective.
Be concrete and practical.

Return JSON only with:
- summary
- key_risks
- required_controls
- privacy_notes
- compliance_notes
- moderation_notes
- incident_response_notes
"""

CONSTRAINT_REASONER_SYSTEM = """
You are ConstraintReasoner in a multi-agent SDLC planning swarm.

Analyze cost, complexity, maintainability, phased delivery, team burden, and operational trade-offs.

Return JSON only with:
- summary
- cost_range
- complexity_profile
- maintainability_notes
- phased_delivery
- tradeoffs
- implementation_pressure_points
"""

CRITIC_REASONER_SYSTEM = """
You are CriticReasoner in a multi-agent SDLC planning swarm.

Synthesize all specialist reviews.
Identify contradictions, blind spots, unresolved assumptions, and corrective directions before the architecture draft is generated.

Return JSON only with:
- summary
- contradictions
- blindspots
- unresolved_questions
- corrective_actions
- priority_order
"""

ARCHITECT_GENERATOR_SYSTEM = """
You are ArchitectAgent in a phase-based multi-agent SDLC system.

Create a very detailed architecture plan from:
- the frozen confirmed requirements
- the specialist review outputs
- the issue ledger
- the last audit feedback if any

Rules:
- this is NOT a short summary
- create a long, implementation-grade plan
- include concrete technical details, boundaries, schemas, APIs, workflows, deployment, security, observability, and roadmap
- include enough detail that the final report can exceed 10 pages if needed
- preserve user-confirmed requirements
- if a user choice creates trade-offs, design around it rather than silently changing it

Return JSON only with:
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
"""

AUDITOR_SYSTEM = """
You are AuditorAgent in a phase-based multi-agent SDLC system.

Audit the architecture plan against the frozen confirmed requirements.
Review coherence, security, privacy, maintainability, completeness, phase readiness, and execution realism.
You may approve only if the plan is genuinely strong.

Return JSON only with:
- score
- summary
- strengths
- concerns
- blocking_issues
- recommendations
- issue_updates
- requirement_conflicts

Each requirement_conflict item must include:
- issue_id
- field
- current_value
- proposed_value
- exact_reason
- severity
"""

EXECUTION_PLANNER_SYSTEM = """
You are ExecutionPlannerAgent.

Transform the approved architecture into a very detailed implementation roadmap.
This must be long, specific, and step-by-step.

Requirements:
- organize by phases
- include frontend, backend, data, infra, security, QA, and rollout work
- include dependencies, deliverables, and done criteria
- include feature-by-feature implementation guidance
- include suggested task breakdowns that a student or small team can follow

Return JSON only with:
- execution_overview
- implementation_phases
- feature_workstreams
- dependency_map
- milestone_checks
- rollout_strategy
"""

TUTOR_SYSTEM = """
You are TutorAgent.

Create a detailed development playbook for implementing the approved plan.
Teach in a practical, step-by-step way for a developer.

Return JSON only with:
- development_playbook
- coding_order
- implementation_tips
- common_mistakes
- feature_build_guides
"""

QA_SYSTEM = """
You are QAEngineerAgent.

Create a detailed testing and validation package from the approved architecture and execution plan.

Return JSON only with:
- validation_strategy
- test_layers
- detailed_test_plan
- acceptance_criteria
- regression_strategy
- release_readiness_checklist
"""

NARRATIVE_SYSTEM = """
You are NarrativeWriterAgent.

Write a long-form validated architecture report package.
Never leave sections blank.
The report should be detailed enough to support a long PDF.

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
"""

DIAGRAM_SYSTEM = """
You are DiagramSpecAgent.

Create real structured diagram specs for:
- architecture
- component_interaction
- data_er
- deployment
- security_boundary
- implementation_roadmap

Rules:
- provide actual nodes and edges
- do not return placeholders
- for ER, include entities with fields and relationships
- for roadmap, model phases and dependencies

Return JSON only with:
- diagrams
"""

DEVELOPMENT_SUMMARY_SYSTEM = """
You are DevelopmentPhaseAgent.

Create the final development phase handoff after approval.
This should bridge the validated plan into implementation support.

Return JSON only with:
- development_summary
- first_week_plan
- coding_sequence
- practical_starting_point
"""


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
class SharedState:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    phase: str = PHASE_REQUIREMENTS
    dialogue: List[Dict[str, Any]] = field(default_factory=list)

    requirement_contract: Dict[str, RequirementField] = field(default_factory=lambda: {
        key: RequirementField() for key in FIELD_PROMPTS.keys()
    })
    pending_confirmations: List[str] = field(default_factory=list)
    pending_change_requests: List[Dict[str, Any]] = field(default_factory=list)

    pass_threshold: float = 9.0
    max_planning_rounds: int = 4
    debug_mode: bool = False
    show_internal_panels: bool = True
    report_depth: str = "extreme"

    issue_ledger: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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

    shutdown: bool = False


# =========================================================
# Azure client
# =========================================================

class AzureLLM:
    def __init__(self) -> None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY","F79rr24XOyTKAprSSVMiQuo8j99MQM9gzJD3oEIAmlfn4vrsj0TVJQQJ99CBACHYHv6XJ3w3AAABACOGX5Md")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT","https://cmg-ai-poc-eu2.openai.azure.com/")
        chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT","gpt-5-chat")
        reasoning_deployment = os.getenv("AZURE_OPENAI_REASONING_DEPLOYMENT", chat_deployment)

        if not api_key or not endpoint or not chat_deployment:
            raise RuntimeError(
                "Missing Azure OpenAI configuration. Set AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_CHAT_DEPLOYMENT."
            )

        self.chat_deployment = chat_deployment
        self.reasoning_deployment = reasoning_deployment
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint.rstrip("/") + "/openai/v1/",
        )

    def complete_json(
        self,
        system_prompt: str,
        payload: Dict[str, Any],
        max_tokens: int = 3000,
        reasoning: bool = True,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        model = self.reasoning_deployment if reasoning else self.chat_deployment
        resp = self.client.chat.completions.create(
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
# Application
# =========================================================

class GovernanceTerminal:
    def __init__(self) -> None:
        self.console = Console()
        self.state = SharedState()
        self.llm = AzureLLM()

        self.state.artifacts_dir = str(Path("artifacts") / self.state.session_id[:8])
        Path(self.state.artifacts_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # UI helpers
    # -----------------------------------------------------

    def banner(self) -> None:
        self.console.print(Rule("[bold cyan]Architectural Governance Terminal"))
        self.console.print("[dim]Type your project idea. Type 'exit' to quit.[/dim]")
        self.console.print("[dim]Commands: :threshold 9.0 | :rounds 5 | :depth extreme | :debug on/off | :thinking on/off | :status | :export | :exceptions[/dim]\n")

    def panel(self, title: str, body: str, color: str = "green") -> None:
        self.console.print(Panel(body, title=title, border_style=color))

    def thinking(self, agent: str, summary: Any, next_action: str = "") -> None:
        if not self.state.show_internal_panels:
            return
        body = as_text(summary, 3200)
        if next_action:
            body += f"\n\n[dim]next: {next_action}[/dim]"
        self.console.print(Panel(body, title=f"[bold yellow]THINKING · {agent}[/bold yellow]", border_style="yellow"))

    def append_dialogue(self, role: str, content: str, agent: Optional[str] = None) -> None:
        self.state.dialogue.append({
            "role": role,
            "content": content,
            "agent": agent,
            "timestamp": now_iso(),
        })

    def dialogue_tail(self, n: int = 14) -> List[Dict[str, Any]]:
        return self.state.dialogue[-n:]

    def show_status(self) -> None:
        t = Table(title="Runtime Status", box=box.SIMPLE_HEAVY)
        t.add_column("Field", style="cyan", width=28)
        t.add_column("Value")
        t.add_row("Phase", self.state.phase)
        t.add_row("Threshold", f"{self.state.pass_threshold:.2f}")
        t.add_row("Planning rounds", str(self.state.max_planning_rounds))
        t.add_row("Report depth", self.state.report_depth)
        t.add_row("Internal panels", str(self.state.show_internal_panels))
        t.add_row("Debug", str(self.state.debug_mode))
        t.add_row("Missing required fields", ", ".join(self.missing_required_fields()) or "None")
        t.add_row("Best audit score", f"{self.state.best_audit.get('score', 0):.2f}" if self.state.best_audit else "N/A")
        t.add_row("Approved PDF", self.state.final_pdf_path or "None")
        self.console.print(t)

    # -----------------------------------------------------
    # Contract helpers
    # -----------------------------------------------------

    def set_field(self, field_name: str, value: str, source: str, confirmed: bool, rationale: str) -> None:
        if field_name not in self.state.requirement_contract:
            return
        self.state.requirement_contract[field_name] = RequirementField(
            value=str(value).strip(),
            source=str(source).strip(),
            confirmed=bool(confirmed),
            rationale=str(rationale).strip(),
            updated_at=now_iso(),
        )

    def confirm_fields(self, fields: List[str]) -> None:
        for f in fields:
            if f in self.state.requirement_contract:
                self.state.requirement_contract[f].confirmed = True
                self.state.requirement_contract[f].updated_at = now_iso()

    def missing_required_fields(self) -> List[str]:
        out = []
        for f in MANDATORY_FIELDS:
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

    def next_missing_field(self) -> Optional[str]:
        missing = self.missing_required_fields()
        return missing[0] if missing else None

    def contract_summary_text(self) -> str:
        lines = ["Confirmed requirement contract:"]
        for k, v in self.state.requirement_contract.items():
            if v.value.strip():
                suffix = "confirmed" if v.confirmed else "pending"
                lines.append(f"- {k}: {v.value} ({suffix})")
        return "\n".join(lines)

    # -----------------------------------------------------
    # Run loop
    # -----------------------------------------------------

    def run(self) -> None:
        self.banner()
        welcome = (
            "Hi — I’ll help you define the project step by step in a natural way, and once the important requirements are clear, "
            "I’ll move the project into the internal multi-agent planning and validation phase. Start by telling me what you want to build."
        )
        self.append_dialogue("assistant", welcome, "RequirementCoordinator")
        self.panel("RequirementCoordinator", welcome)

        while not self.state.shutdown:
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
                self.state.pass_threshold = max(7.0, min(10.0, float(parts[1])))
                self.panel("System", f"Pass threshold set to {self.state.pass_threshold:.2f}.", "cyan")
            except Exception:
                self.panel("System", "Invalid threshold value.", "red")
            return True

        if cmd == ":rounds" and len(parts) == 2:
            try:
                self.state.max_planning_rounds = max(1, min(10, int(parts[1])))
                self.panel("System", f"Planning rounds set to {self.state.max_planning_rounds}.", "cyan")
            except Exception:
                self.panel("System", "Invalid round count.", "red")
            return True

        if cmd == ":depth" and len(parts) == 2:
            value = parts[1].lower()
            if value in {"medium", "long", "extreme"}:
                self.state.report_depth = value
                self.panel("System", f"Report depth set to {value}.", "cyan")
            else:
                self.panel("System", "Use :depth medium | long | extreme", "red")
            return True

        if cmd == ":debug" and len(parts) == 2:
            self.state.debug_mode = parts[1].lower() in {"on", "true", "1"}
            self.panel("System", f"Debug mode set to {self.state.debug_mode}.", "cyan")
            return True

        if cmd == ":thinking" and len(parts) == 2:
            self.state.show_internal_panels = parts[1].lower() in {"on", "true", "1"}
            self.panel("System", f"Internal planning panels set to {self.state.show_internal_panels}.", "cyan")
            return True

        if cmd == ":status":
            self.show_status()
            return True

        if cmd == ":exceptions":
            if not self.state.accepted_exceptions:
                self.panel("Accepted Exceptions", "No accepted exceptions.")
            else:
                body = "\n\n".join(
                    f"{ex.issue_id}\nReason: {ex.reason}\nUser message: {ex.user_message}"
                    for ex in self.state.accepted_exceptions.values()
                )
                self.panel("Accepted Exceptions", body, "cyan")
            return True

        if cmd == ":export":
            if not self.state.best_plan or not self.state.best_audit:
                self.panel("System", "There is no approved plan to export yet.", "red")
            else:
                self.generate_report_and_export()
                self.panel("System", f"PDF exported:\n{self.state.final_pdf_path}", "cyan")
            return True

        return False

    def handle_turn(self, user_text: str) -> None:
        if self.state.phase in {PHASE_REQUIREMENTS, PHASE_REQUIREMENT_CONFIRMATION}:
            self.handle_requirement_turn(user_text)
            return

        if self.state.phase == PHASE_CHANGE_REVIEW:
            self.handle_change_review(user_text)
            return

        if self.state.phase in {PHASE_APPROVED, PHASE_DEVELOPMENT}:
            msg = (
                f"The validated plan is already approved.\nPDF: {self.state.final_pdf_path or 'not exported yet'}\n\n"
                f"You can use :export to regenerate the report, or start a new session for a new project."
            )
            self.panel("System", msg, "cyan")
            return

    # -----------------------------------------------------
    # Requirement flow
    # -----------------------------------------------------

    def requirement_coordinator_consult(self, user_text: str) -> Dict[str, Any]:
        payload = {
            "current_contract": self.contract_snapshot(),
            "mandatory_fields": MANDATORY_FIELDS,
            "missing_required_fields": self.missing_required_fields(),
            "field_prompts": FIELD_PROMPTS,
            "user_message": user_text,
            "dialogue_tail": self.dialogue_tail(),
        }
        result = self.llm.complete_json(FRIENDLY_REQ_SYSTEM, payload, max_tokens=2400, reasoning=True)
        if self.state.debug_mode:
            self.thinking("RequirementCoordinator", result.get("thinking_summary", "Continuing requirement gathering."), "continue natural requirement gathering")
        return result

    def requirement_confirmation_consult(self, user_text: str) -> Dict[str, Any]:
        pending_fields = deepcopy(self.state.pending_confirmations)
        pending_snapshot = {
            f: asdict(self.state.requirement_contract[f])
            for f in pending_fields
            if f in self.state.requirement_contract
        }
        payload = {
            "current_contract": self.contract_snapshot(),
            "mandatory_fields": MANDATORY_FIELDS,
            "missing_required_fields": self.missing_required_fields(),
            "pending_confirmation_fields": pending_fields,
            "pending_confirmation_snapshot": pending_snapshot,
            "user_message": user_text,
            "dialogue_tail": self.dialogue_tail(),
        }
        result = self.llm.complete_json(REQUIREMENT_CONFIRM_SYSTEM, payload, max_tokens=2200, reasoning=True)
        if self.state.debug_mode:
            self.thinking("RequirementCoordinator", result.get("thinking_summary", "Resolving requirement confirmations."), "produce one friendly reply")
        return result

    def apply_requirement_updates(self, result: Dict[str, Any]) -> List[str]:
        updates = result.get("updates", [])
        pending = []

        if not isinstance(updates, list):
            return pending

        for item in updates:
            if not isinstance(item, dict):
                continue
            field_name = item.get("field")
            value = str(item.get("value", "")).strip()
            if not field_name or field_name not in self.state.requirement_contract or not value:
                continue

            self.set_field(
                field_name=field_name,
                value=value,
                source=str(item.get("source", "agent")),
                confirmed=bool(item.get("confirmed", False)),
                rationale=str(item.get("rationale", "")),
            )

            if bool(item.get("needs_confirmation", False)) and not self.state.requirement_contract[field_name].confirmed:
                pending.append(field_name)

        explicit_pending = ensure_list_of_str(result.get("pending_confirmations"))
        if explicit_pending:
            pending = explicit_pending

        return pending

    def handle_requirement_turn(self, user_text: str) -> None:
        normalized = user_text.lower().strip()

        if normalized in {"approve requirements", "approve contract", "finalize requirements"}:
            if self.all_required_locked():
                msg = (
                    "Perfect — the requirement contract is now complete, so I’m moving into the internal multi-agent planning and validation stage. "
                    "You’ll now see the architecture swarm reason through the plan phase by phase."
                )
                self.panel("RequirementCoordinator", msg, "green")
                self.start_governance_cycle()
            else:
                next_field = self.next_missing_field()
                self.panel(
                    "RequirementCoordinator",
                    f"We’re close, but I still need to lock one important area before planning: {next_field}.",
                    "yellow",
                )
            return

        if self.state.pending_confirmations:
            result = self.requirement_confirmation_consult(user_text)
            action = str(result.get("action", "treat_as_edit")).lower().strip()
            old_pending = deepcopy(self.state.pending_confirmations)

            if action == "approve_proposals":
                self.confirm_fields(old_pending)
                self.state.pending_confirmations = []
                fresh_pending = self.apply_requirement_updates(result)
                self.state.pending_confirmations = fresh_pending
            else:
                self.state.pending_confirmations = []
                fresh_pending = self.apply_requirement_updates(result)
                self.state.pending_confirmations = fresh_pending

            self.state.phase = PHASE_REQUIREMENT_CONFIRMATION if self.state.pending_confirmations else PHASE_REQUIREMENTS
            self.panel("RequirementCoordinator", result.get("assistant_message", "Got it — I’ve updated the requirement notes."), "green")
            return

        result = self.requirement_coordinator_consult(user_text)
        self.state.pending_confirmations = self.apply_requirement_updates(result)
        self.state.phase = PHASE_REQUIREMENT_CONFIRMATION if self.state.pending_confirmations else PHASE_REQUIREMENTS
        self.panel("RequirementCoordinator", result.get("assistant_message", "Got it — I’ve updated the requirement notes."), "green")

    # -----------------------------------------------------
    # Planning swarm
    # -----------------------------------------------------

    def token_budget(self, purpose: str) -> int:
        depth = self.state.report_depth
        budgets = {
            "medium": {"analysis": 1800, "plan": 3000, "report": 3500},
            "long": {"analysis": 2600, "plan": 4200, "report": 5000},
            "extreme": {"analysis": 3400, "plan": 5600, "report": 6500},
        }
        return budgets.get(depth, budgets["long"]).get(purpose, 3000)

    def start_governance_cycle(self) -> None:
        self.state.phase = PHASE_PLANNING
        self.console.print(Rule("[bold magenta]Internal Planning & Audit Started"))

        for round_no in range(1, self.state.max_planning_rounds + 1):
            self.console.print(Rule(f"[bold cyan]Architecture Round {round_no}"))

            specialist_reviews = self.run_specialist_reasoners(round_no)
            plan = self.architect_generate(round_no, specialist_reviews)
            audit = self.auditor_validate(round_no, plan, specialist_reviews)

            self.state.specialist_history.append({
                "round": round_no,
                "reviews": specialist_reviews,
                "timestamp": now_iso(),
            })
            self.state.audit_history.append(deepcopy(audit))
            self.state.current_plan = deepcopy(plan)
            self.state.current_audit = deepcopy(audit)

            write_json(Path(self.state.artifacts_dir) / f"specialists_round_{round_no}.json", specialist_reviews)
            write_json(Path(self.state.artifacts_dir) / f"plan_round_{round_no}.json", plan)
            write_json(Path(self.state.artifacts_dir) / f"audit_round_{round_no}.json", audit)

            self.merge_issue_ledger(audit)
            self.update_best(plan, audit)
            self.show_round_tables(round_no, plan, audit)

            unresolved_conflicts = [
                c for c in ensure_list(audit.get("requirement_conflicts"))
                if isinstance(c, dict) and c.get("issue_id") not in self.state.accepted_exceptions
            ]
            if unresolved_conflicts:
                self.state.pending_change_requests = unresolved_conflicts
                self.state.phase = PHASE_CHANGE_REVIEW
                self.present_change_requests()
                return

            if audit["passed"]:
                self.state.phase = PHASE_APPROVED
                self.generate_report_and_export()
                self.panel(
                    "APPROVED",
                    f"Validated plan approved with score {audit['score']:.2f}\nPDF: {self.state.final_pdf_path}",
                    "green",
                )
                self.state.phase = PHASE_DEVELOPMENT
                self.present_development_handoff()
                return

            self.panel(
                "Revision In Progress",
                "The planning swarm is revising the architecture based on auditor feedback before the next round.",
                "yellow",
            )

        self.state.phase = PHASE_REQUIREMENTS
        self.panel(
            "Planning",
            "The architecture did not reach approval within the current round limit. You can refine the requirements, raise planning rounds, or lower the threshold.",
            "red",
        )

    def run_specialist_reasoners(self, round_no: int) -> Dict[str, Any]:
        base_payload = {
            "round": round_no,
            "frozen_requirement_contract": self.frozen_contract(),
            "issue_ledger": self.state.issue_ledger,
            "accepted_exceptions": {k: asdict(v) for k, v in self.state.accepted_exceptions.items()},
            "previous_audits": self.state.audit_history[-2:],
            "best_audit": self.state.best_audit,
        }

        product = self.llm.complete_json(
            PRODUCT_REASONER_SYSTEM,
            base_payload,
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        self.thinking("ProductReasoner", product.get("summary", "Product review complete."), "handoff to swarm")

        architect_reason = self.llm.complete_json(
            ARCHITECT_REASONER_SYSTEM,
            {**base_payload, "product_review": product},
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        self.thinking("ArchitectReasoner", architect_reason.get("summary", "Architecture review complete."), "handoff to swarm")

        security = self.llm.complete_json(
            SECURITY_REASONER_SYSTEM,
            {**base_payload, "product_review": product, "architect_review": architect_reason},
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        self.thinking("SecurityReasoner", security.get("summary", "Security review complete."), "handoff to swarm")

        constraints = self.llm.complete_json(
            CONSTRAINT_REASONER_SYSTEM,
            {**base_payload, "product_review": product, "architect_review": architect_reason, "security_review": security},
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        self.thinking("ConstraintReasoner", constraints.get("summary", "Constraint review complete."), "handoff to swarm")

        critic = self.llm.complete_json(
            CRITIC_REASONER_SYSTEM,
            {
                **base_payload,
                "product_review": product,
                "architect_review": architect_reason,
                "security_review": security,
                "constraint_review": constraints,
            },
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        self.thinking("CriticReasoner", critic.get("summary", "Critic review complete."), "guide ArchitectAgent")

        return {
            "product": product,
            "architect_reasoner": architect_reason,
            "security": security,
            "constraints": constraints,
            "critic": critic,
        }

    def architect_generate(self, round_no: int, specialist_reviews: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "round": round_no,
            "frozen_requirement_contract": self.frozen_contract(),
            "specialist_reviews": specialist_reviews,
            "issue_ledger": self.state.issue_ledger,
            "accepted_exceptions": {k: asdict(v) for k, v in self.state.accepted_exceptions.items()},
            "previous_audits": self.state.audit_history[-3:],
            "previous_plan": self.state.current_plan,
            "best_plan": self.state.best_plan,
        }
        result = self.llm.complete_json(
            ARCHITECT_GENERATOR_SYSTEM,
            payload,
            max_tokens=self.token_budget("plan"),
            reasoning=True,
        )
        summary = result.get("executive_summary") or result.get("summary") or "Architecture draft generated."
        self.thinking("ArchitectAgent", summary, "submit to AuditorAgent")
        return self.normalize_plan(result, specialist_reviews)

    def normalize_plan(self, raw: Dict[str, Any], specialist_reviews: Dict[str, Any]) -> Dict[str, Any]:
        contract = self.frozen_contract()

        def c(field: str, fallback: str = "Derived from confirmed requirements.") -> str:
            item = contract.get(field, {})
            return str(item.get("value") or fallback)

        plan = {
            "title": str(raw.get("title") or "Validated Architecture Plan"),
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
                "llm_integration": c("llm_integration", "Backend-managed OpenAI-compatible API integration"),
            },
            "functional_feature_map": raw.get("functional_feature_map") or {
                "mvp_scope": c("mvp_scope"),
                "expanded_scope": c("feature_scope"),
                "future_scope": c("future_scope", "Additional advanced features after MVP"),
            },
            "system_components": raw.get("system_components") or {
                "components": [
                    {"name": "Web Client", "responsibility": "User interface and interactive chat experience"},
                    {"name": "API Layer", "responsibility": "Authentication, request validation, and routing"},
                    {"name": "Conversation Service", "responsibility": "Session and prompt orchestration"},
                    {"name": "LLM Gateway", "responsibility": "Model calls, retries, guardrails, and usage accounting"},
                    {"name": "Persistence Layer", "responsibility": "Users, sessions, messages, and metadata"},
                    {"name": "Monitoring Layer", "responsibility": "Logs, metrics, alerts, tracing, and auditing"},
                ]
            },
            "workflows": raw.get("workflows") or {
                "primary_flows": [
                    "Account registration and login",
                    "Session creation and chat submission",
                    "Prompt assembly and guardrail checks",
                    "LLM inference and streaming response delivery",
                    "Message persistence and observability event capture",
                ]
            },
            "data_model": raw.get("data_model") or {
                "entities": ["User", "ChatSession", "Message", "Attachment", "Feedback", "UsageEvent", "AuditEvent"],
                "storage_strategy": c("data_platform"),
                "retention_policy": c("privacy_retention_policy"),
            },
            "api_design": raw.get("api_design") or {
                "style": "REST plus streaming endpoints",
                "endpoints": [
                    "/api/auth/signup",
                    "/api/auth/login",
                    "/api/chat/sessions",
                    "/api/chat/messages",
                    "/api/chat/stream",
                    "/api/user/profile",
                ],
            },
            "security_and_compliance": raw.get("security_and_compliance") or {
                "baseline": c("security_baseline"),
                "privacy": c("privacy_retention_policy"),
                "compliance_context": c("compliance_context", "Basic privacy-by-design and deletion support"),
            },
            "deployment_and_operations": raw.get("deployment_and_operations") or {
                "hosting_target": c("hosting_target"),
                "observability_baseline": c("observability_baseline", "Logs, metrics, traces, alerts"),
                "ops_model": "Phased deployment with rollback and monitoring",
            },
            "observability": raw.get("observability") or {
                "baseline": c("observability_baseline", "Centralized logs, metrics, tracing, and alerting"),
            },
            "cost_and_scaling": raw.get("cost_and_scaling") or {
                "cost_position": "Usage-driven due to external LLM calls",
                "scaling_direction": "Horizontal application scaling with managed services and caching",
            },
            "phased_implementation": raw.get("phased_implementation") or {
                "phase_1": c("mvp_scope"),
                "phase_2": c("future_scope", "Expanded features after MVP stabilization"),
            },
            "development_guidelines": raw.get("development_guidelines") or {
                "principles": ["Keep services modular", "Preserve clear contracts", "Automate testing early"],
            },
            "risks_and_tradeoffs": raw.get("risks_and_tradeoffs") or {
                "risks": ["API cost growth", "Abuse pressure", "Latency variability"],
                "tradeoffs": ["Faster MVP simplicity versus later governance and scale hardening"],
            },
            "open_questions_resolved": raw.get("open_questions_resolved") or specialist_reviews.get("critic", {}),
            "generated_at": now_iso(),
        }

        plan["title"] = re.sub(r"\bRound\s+\d+\b", "", plan["title"], flags=re.IGNORECASE).strip(" -–") or "Validated Architecture Plan"
        return plan

    def auditor_validate(self, round_no: int, plan: Dict[str, Any], specialist_reviews: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "round": round_no,
            "frozen_requirement_contract": self.frozen_contract(),
            "accepted_exceptions": {k: asdict(v) for k, v in self.state.accepted_exceptions.items()},
            "issue_ledger": self.state.issue_ledger,
            "specialist_reviews": specialist_reviews,
            "plan": plan,
            "pass_threshold": self.state.pass_threshold,
        }
        result = self.llm.complete_json(
            AUDITOR_SYSTEM,
            payload,
            max_tokens=self.token_budget("analysis"),
            reasoning=True,
        )
        self.thinking("AuditorAgent", result.get("summary", "Audit complete."), "approve or request revision")

        score = float(result.get("score", 6.5))
        score = max(0.0, min(10.0, score))
        blocking_issues = ensure_list_of_str(result.get("blocking_issues"))
        recommendations = ensure_list_of_str(result.get("recommendations"))
        conflicts = ensure_list(result.get("requirement_conflicts"))
        conflicts = [c for c in conflicts if isinstance(c, dict)]
        unresolved_conflicts = [c for c in conflicts if c.get("issue_id") not in self.state.accepted_exceptions]

        passed = score >= self.state.pass_threshold and len(blocking_issues) == 0 and len(unresolved_conflicts) == 0

        return {
            "round": round_no,
            "score": score,
            "passed": passed,
            "summary": str(result.get("summary", "Audit completed.")),
            "strengths": ensure_list_of_str(result.get("strengths")),
            "concerns": ensure_list_of_str(result.get("concerns")),
            "blocking_issues": blocking_issues,
            "recommendations": recommendations,
            "issue_updates": ensure_list(result.get("issue_updates")),
            "requirement_conflicts": conflicts,
            "timestamp": now_iso(),
            "raw": result,
        }

    def merge_issue_ledger(self, audit: Dict[str, Any]) -> None:
        for item in ensure_list(audit.get("issue_updates")):
            if not isinstance(item, dict):
                continue
            issue_id = str(item.get("id") or item.get("issue_id") or "").strip()
            if not issue_id:
                continue
            existing = self.state.issue_ledger.get(issue_id, {})
            self.state.issue_ledger[issue_id] = {
                "id": issue_id,
                "title": str(item.get("title") or existing.get("title") or issue_id),
                "severity": str(item.get("severity") or existing.get("severity") or "medium"),
                "status": str(item.get("status") or existing.get("status") or "open"),
                "detail": str(item.get("detail") or existing.get("detail") or ""),
                "history": existing.get("history", []) + [{
                    "round": audit["round"],
                    "status": str(item.get("status") or ""),
                    "detail": str(item.get("detail") or ""),
                }],
                "updated_at": now_iso(),
            }

    def update_best(self, plan: Dict[str, Any], audit: Dict[str, Any]) -> None:
        if not self.state.best_audit:
            self.state.best_plan = deepcopy(plan)
            self.state.best_audit = deepcopy(audit)
            return
        prev_score = float(self.state.best_audit.get("score", 0.0))
        new_score = float(audit.get("score", 0.0))
        prev_blockers = len(self.state.best_audit.get("blocking_issues", []))
        new_blockers = len(audit.get("blocking_issues", []))
        if new_score > prev_score or (new_score == prev_score and new_blockers < prev_blockers):
            self.state.best_plan = deepcopy(plan)
            self.state.best_audit = deepcopy(audit)

    def show_round_tables(self, round_no: int, plan: Dict[str, Any], audit: Dict[str, Any]) -> None:
        pt = Table(title=f"Architecture Draft · Round {round_no}", box=box.SIMPLE_HEAVY)
        pt.add_column("Field", style="cyan", width=22)
        pt.add_column("Value")
        pt.add_row("Title", plan.get("title", ""))
        pt.add_row("Summary", as_text(plan.get("executive_summary", ""), 320))
        pt.add_row("Top-level sections", ", ".join([k for k in plan.keys() if k not in {"title", "executive_summary", "generated_at"}]))
        self.console.print(pt)

        at = Table(title=f"Audit Result · Round {round_no}", box=box.SIMPLE_HEAVY)
        at.add_column("Metric", style="magenta", width=20)
        at.add_column("Value")
        at.add_row("Score", f"{audit['score']:.2f}")
        at.add_row("Passed", str(audit["passed"]))
        at.add_row("Threshold", f"{self.state.pass_threshold:.2f}")
        at.add_row("Summary", as_text(audit["summary"], 320))
        at.add_row("Recommendations", str(len(audit.get("recommendations", []))))
        self.console.print(at)

        if audit.get("blocking_issues"):
            body = "\n".join(f"- {x}" for x in audit["blocking_issues"])
            self.panel("Blocking Issues", body, "red")

    # -----------------------------------------------------
    # Change review
    # -----------------------------------------------------

    def present_change_requests(self) -> None:
        lines = ["The auditor found requirement-level conflicts that need your decision:"]
        for item in self.state.pending_change_requests:
            lines.append("")
            lines.append(f"{item.get('issue_id')}")
            lines.append(f"Field: {item.get('field')}")
            lines.append(f"Current: {item.get('current_value')}")
            lines.append(f"Suggested: {item.get('proposed_value')}")
            lines.append(f"Reason: {item.get('exact_reason')}")
            lines.append(f"Severity: {item.get('severity')}")
        lines.append("")
        lines.append("Reply yes to accept the suggested change, or no to keep your original choice as an accepted exception.")
        self.panel("Change Review", "\n".join(lines), "cyan")

    def handle_change_review(self, user_text: str) -> None:
        if positive_reply(user_text):
            for item in self.state.pending_change_requests:
                field_name = item.get("field")
                proposed_value = item.get("proposed_value")
                reason = str(item.get("exact_reason", "Auditor-guided correction"))
                if field_name in self.state.requirement_contract and proposed_value:
                    self.set_field(field_name, str(proposed_value), "auditor_change_approved", True, reason)
            self.panel("Change Review", "Approved — the requirement contract has been updated with the suggested changes.", "green")
        else:
            for item in self.state.pending_change_requests:
                issue_id = str(item.get("issue_id"))
                self.state.accepted_exceptions[issue_id] = AcceptedException(
                    issue_id=issue_id,
                    reason=str(item.get("exact_reason", "User accepted this trade-off.")),
                    user_message=user_text,
                )
                if issue_id in self.state.issue_ledger:
                    self.state.issue_ledger[issue_id]["status"] = "accepted_exception"
            self.panel("Change Review", "Understood — your original choices have been kept and recorded as accepted exceptions.", "green")

        self.state.pending_change_requests = []
        self.start_governance_cycle()

    # -----------------------------------------------------
    # Final package generation
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
            EXECUTION_PLANNER_SYSTEM,
            {
                "plan": plan,
                "audit": audit,
                "locked_contract": self.frozen_contract(),
                "specialist_history": self.state.specialist_history,
            },
            max_tokens=self.token_budget("report"),
            reasoning=True,
        )

        tutor = self.llm.complete_json(
            TUTOR_SYSTEM,
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
            QA_SYSTEM,
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
            NARRATIVE_SYSTEM,
            {
                "plan": plan,
                "audit": audit,
                "execution": execution,
                "tutor": tutor,
                "qa": qa,
                "locked_contract": self.frozen_contract(),
            },
            max_tokens=self.token_budget("report"),
            reasoning=True,
        )

        diagrams = self.llm.complete_json(
            DIAGRAM_SYSTEM,
            {
                "plan": plan,
                "audit": audit,
                "execution": execution,
                "locked_contract": self.frozen_contract(),
            },
            max_tokens=3000,
            reasoning=True,
        )

        development = self.llm.complete_json(
            DEVELOPMENT_SUMMARY_SYSTEM,
            {
                "plan": plan,
                "audit": audit,
                "execution": execution,
                "tutor": tutor,
                "qa": qa,
            },
            max_tokens=self.token_budget("report"),
            reasoning=True,
        )

        package = self.normalize_report_package(
            {
                "title": narrative.get("title") or plan.get("title") or "Validated Architecture Plan",
                "executive_summary": narrative.get("executive_summary") or plan.get("executive_summary") or audit.get("summary"),
                "sections": narrative.get("sections") or {},
                "execution": execution,
                "tutor": tutor,
                "qa": qa,
                "diagrams": diagrams.get("diagrams") or {},
                "development": development,
            },
            plan,
            audit,
        )

        package["diagram_files"] = self.render_diagrams(package["diagrams"])
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
            "overview": obj_to_text(plan.get("architecture_overview")),
            "requirement_interpretation": self.contract_summary_text(),
            "stack_rationale": obj_to_text(plan.get("technology_stack")),
            "architecture": obj_to_text(plan.get("architecture_overview")),
            "component_design": obj_to_text(plan.get("system_components")),
            "workflow_design": obj_to_text(plan.get("workflows")),
            "data_model": obj_to_text(plan.get("data_model")),
            "api_design": obj_to_text(plan.get("api_design")),
            "security": obj_to_text(plan.get("security_and_compliance")),
            "deployment": obj_to_text(plan.get("deployment_and_operations")),
            "observability": obj_to_text(plan.get("observability")),
            "cost_and_scaling": obj_to_text(plan.get("cost_and_scaling")),
            "phased_implementation": obj_to_text(execution),
            "development_playbook": obj_to_text(tutor),
            "testing_validation": obj_to_text(qa),
            "risks_tradeoffs": obj_to_text(plan.get("risks_and_tradeoffs")),
            "final_notes": "This approved package is ready to guide implementation, testing, and phased rollout.",
        }

        for key, value in defaults.items():
            if not sections.get(key):
                sections[key] = value

        package["sections"] = sections
        package["title"] = re.sub(r"\bRound\s+\d+\b", "", str(package.get("title") or "Validated Architecture Plan"), flags=re.IGNORECASE).strip(" -–") or "Validated Architecture Plan"
        package["executive_summary"] = package.get("executive_summary") or "Validated architecture report."
        return package

    def present_development_handoff(self) -> None:
        dev = self.state.development_package or {}
        body = (
            as_text(dev.get("development_summary", "The project now moves into the development phase."), 2000)
            + "\n\n"
            + "A full tutor-style implementation guide, QA strategy, execution roadmap, and diagrams have been added to the approved PDF."
        )
        self.panel("Development Phase", body, "cyan")

    # -----------------------------------------------------
    # Diagram rendering
    # -----------------------------------------------------

    def render_diagrams(self, diagrams: Dict[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name, spec in diagrams.items():
            file_path = Path(self.state.artifacts_dir) / f"{slugify(name)}.png"
            try:
                if Digraph is not None:
                    if name == "data_er":
                        self.render_er_graphviz(spec, file_path, name)
                    else:
                        self.render_graphviz(spec, file_path, name)
                else:
                    self.render_pil_diagram(spec, file_path, name)
            except Exception:
                self.render_pil_diagram(spec, file_path, name)
            out[name] = str(file_path.resolve())
        return out

    def render_graphviz(self, spec: Dict[str, Any], out_path: Path, title: str) -> None:
        dot = Digraph(comment=title, format="png")
        dot.attr(rankdir="LR", bgcolor="white", fontsize="11", fontname="Helvetica")
        dot.attr("node", shape="box", style="rounded,filled", fillcolor="#EFF6FF", color="#2563EB", fontname="Helvetica")
        dot.attr("edge", color="#475569", fontname="Helvetica")

        nodes = ensure_list(spec.get("nodes"))
        edges = ensure_list(spec.get("edges"))

        seen = set()
        for node in nodes:
            if isinstance(node, dict):
                node_id = str(node.get("id") or node.get("name") or uuid.uuid4().hex[:6])
                label = str(node.get("label") or node.get("name") or node_id)
            else:
                node_id = str(node)
                label = node_id
            if node_id in seen:
                continue
            seen.add(node_id)
            dot.node(node_id, label)

        for edge in edges:
            if isinstance(edge, dict):
                src = str(edge.get("from") or edge.get("source") or "")
                dst = str(edge.get("to") or edge.get("target") or "")
                lbl = str(edge.get("label") or "")
                if src and dst:
                    dot.edge(src, dst, lbl)
            elif isinstance(edge, list) and len(edge) >= 2:
                dot.edge(str(edge[0]), str(edge[1]))

        rendered = dot.render(filename=out_path.with_suffix("").as_posix(), cleanup=True)
        src = Path(rendered)
        if src.resolve() != out_path.resolve():
            out_path.write_bytes(src.read_bytes())

    def render_er_graphviz(self, spec: Dict[str, Any], out_path: Path, title: str) -> None:
        dot = Digraph(comment=title, format="png")
        dot.attr(rankdir="LR", bgcolor="white", fontname="Helvetica")

        entities = ensure_list(spec.get("entities"))
        relationships = ensure_list(spec.get("relationships"))

        for entity in entities:
            if isinstance(entity, dict):
                name = str(entity.get("name") or "Entity")
                fields = ensure_list_of_str(entity.get("fields"))
            else:
                name = str(entity)
                fields = []
            label = "<<TABLE BORDER='1' CELLBORDER='0' CELLSPACING='0' CELLPADDING='5'>"
            label += f"<TR><TD BGCOLOR='#DBEAFE'><B>{name}</B></TD></TR>"
            for f in fields:
                label += f"<TR><TD ALIGN='LEFT'>{f}</TD></TR>"
            label += "</TABLE>>"
            dot.node(name, label=label, shape="plain")

        for rel in relationships:
            if isinstance(rel, dict):
                a = str(rel.get("from") or rel.get("left") or "")
                b = str(rel.get("to") or rel.get("right") or "")
                lbl = str(rel.get("label") or rel.get("cardinality") or "")
                if a and b:
                    dot.edge(a, b, label=lbl)

        rendered = dot.render(filename=out_path.with_suffix("").as_posix(), cleanup=True)
        src = Path(rendered)
        if src.resolve() != out_path.resolve():
            out_path.write_bytes(src.read_bytes())

    def render_pil_diagram(self, spec: Dict[str, Any], out_path: Path, title: str) -> None:
        if PILImage is None or ImageDraw is None:
            out_path.write_bytes(b"")
            return

        width, height = 1800, 1100
        img = PILImage.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()
        except Exception:
            font_title = None
            font_body = None

        draw.rectangle((20, 20, width - 20, height - 20), outline="#1E3A8A", width=3)
        draw.text((40, 30), title, fill="black", font=font_title)

        if title == "data_er":
            entities = ensure_list(spec.get("entities"))
            relationships = ensure_list(spec.get("relationships"))
            boxes = {}
            cols = max(1, math.ceil(math.sqrt(max(1, len(entities)))))
            box_w = 320
            box_h = 190
            gap_x = 80
            gap_y = 70
            start_x = 50
            start_y = 100

            for idx, entity in enumerate(entities):
                row = idx // cols
                col = idx % cols
                x1 = start_x + col * (box_w + gap_x)
                y1 = start_y + row * (box_h + gap_y)
                x2 = x1 + box_w
                y2 = y1 + box_h
                name = str(entity.get("name") if isinstance(entity, dict) else entity)
                fields = ensure_list_of_str(entity.get("fields")) if isinstance(entity, dict) else []
                draw.rounded_rectangle((x1, y1, x2, y2), radius=16, outline="#2563EB", width=3, fill="#EFF6FF")
                draw.text((x1 + 12, y1 + 10), name, fill="black", font=font_title)
                body = "\n".join(fields[:8])
                draw.text((x1 + 12, y1 + 40), body, fill="#334155", font=font_body)
                boxes[name] = ((x1 + x2) // 2, (y1 + y2) // 2)

            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                a = str(rel.get("from") or rel.get("left") or "")
                b = str(rel.get("to") or rel.get("right") or "")
                lbl = str(rel.get("label") or rel.get("cardinality") or "")
                if a in boxes and b in boxes:
                    ax, ay = boxes[a]
                    bx, by = boxes[b]
                    draw.line((ax, ay, bx, by), fill="#64748B", width=3)
                    mx, my = (ax + bx) // 2, (ay + by) // 2
                    draw.text((mx + 8, my + 4), lbl, fill="black", font=font_body)
        else:
            nodes = ensure_list(spec.get("nodes"))
            edges = ensure_list(spec.get("edges"))
            if not nodes:
                nodes = [{"id": "A", "label": "Diagram content unavailable"}]

            cols = max(1, min(4, math.ceil(math.sqrt(len(nodes)))))
            box_w = 300
            box_h = 120
            gap_x = 90
            gap_y = 90
            start_x = 60
            start_y = 120

            centers: Dict[str, Tuple[int, int]] = {}
            for idx, node in enumerate(nodes):
                row = idx // cols
                col = idx % cols
                x1 = start_x + col * (box_w + gap_x)
                y1 = start_y + row * (box_h + gap_y)
                x2 = x1 + box_w
                y2 = y1 + box_h
                if isinstance(node, dict):
                    node_id = str(node.get("id") or node.get("name") or f"N{idx}")
                    label = str(node.get("label") or node.get("name") or node_id)
                else:
                    node_id = str(node)
                    label = node_id
                draw.rounded_rectangle((x1, y1, x2, y2), radius=18, outline="#2563EB", width=3, fill="#EFF6FF")
                draw.text((x1 + 16, y1 + 18), textwrap.fill(label, width=26), fill="black", font=font_body)
                centers[node_id] = ((x1 + x2) // 2, (y1 + y2) // 2)

            for edge in edges:
                if isinstance(edge, dict):
                    src = str(edge.get("from") or edge.get("source") or "")
                    dst = str(edge.get("to") or edge.get("target") or "")
                    lbl = str(edge.get("label") or "")
                elif isinstance(edge, list) and len(edge) >= 2:
                    src, dst = str(edge[0]), str(edge[1])
                    lbl = ""
                else:
                    continue
                if src in centers and dst in centers:
                    ax, ay = centers[src]
                    bx, by = centers[dst]
                    draw.line((ax, ay, bx, by), fill="#64748B", width=3)
                    mx, my = (ax + bx) // 2, (ay + by) // 2
                    if lbl:
                        draw.text((mx + 8, my + 4), lbl, fill="black", font=font_body)

        img.save(out_path)

    # -----------------------------------------------------
    # PDF export
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
                ["Validation score", f"{audit.get('score', 0):.2f}"],
                ["Approval threshold", f"{self.state.pass_threshold:.2f}"],
                ["Planning rounds used", str(audit.get("round", 0))],
                ["Accepted exceptions", str(len(self.state.accepted_exceptions))],
                ["Report depth", self.state.report_depth],
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
        for p in split_paragraphs(as_text(report.get("executive_summary", ""), 50000)):
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

        for key, title in ordered_sections:
            story.append(Paragraph(title, h1))
            content = report["sections"].get(key, "")
            for p in split_paragraphs(obj_to_text(content, 80000)):
                story.append(Paragraph(self.pdf_escape(p), body))

            if key == "architecture":
                self.append_diagram(story, report, "architecture", "High-Level Architecture Diagram", h2, body)
                self.append_diagram(story, report, "component_interaction", "Component Interaction Diagram", h2, body)
            elif key == "data_model":
                self.append_diagram(story, report, "data_er", "Database ER Diagram", h2, body)
            elif key == "deployment":
                self.append_diagram(story, report, "deployment", "Deployment Diagram", h2, body)
                self.append_diagram(story, report, "security_boundary", "Security Boundary Diagram", h2, body)
            elif key == "phased_implementation":
                self.append_diagram(story, report, "implementation_roadmap", "Implementation Roadmap Diagram", h2, body)
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

    def append_diagram(self, story: List[Any], report: Dict[str, Any], key: str, title: str, h2, body) -> None:
        file_path = report.get("diagram_files", {}).get(key)
        if not file_path:
            return
        p = Path(file_path)
        if not p.exists() or p.stat().st_size == 0:
            return
        story.append(Paragraph(title, h2))
        story.append(Paragraph("The following diagram is generated from the approved architecture package and is intended to support implementation and review.", body))
        story.append(Image(str(p), width=170 * mm, height=95 * mm))
        story.append(Spacer(1, 8))

    def append_execution_breakdown(self, story: List[Any], execution: Dict[str, Any], h2, body) -> None:
        if not execution:
            return

        overview = execution.get("execution_overview")
        if overview:
            story.append(Paragraph("Execution Overview", h2))
            for p in split_paragraphs(obj_to_text(overview, 30000)):
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
                            details.append(f"{key}:")
                            details.extend(json_to_lines(phase.get(key), 1))
                    story.append(self.bullet_list(details, body))

        feature_workstreams = ensure_list(execution.get("feature_workstreams"))
        if feature_workstreams:
            story.append(Paragraph("Feature Workstreams", h2))
            for item in feature_workstreams:
                if isinstance(item, dict):
                    name = str(item.get("feature") or item.get("name") or "Feature")
                    story.append(Paragraph(self.pdf_escape(name), body))
                    details = json_to_lines(item, 1)
                    story.append(self.bullet_list(details, body))

        milestone_checks = ensure_list_of_str(execution.get("milestone_checks"))
        if milestone_checks:
            story.append(Paragraph("Milestone Checks", h2))
            story.append(self.bullet_list(milestone_checks, body))

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
            if value:
                story.append(Paragraph(title, h2))
                if isinstance(value, list):
                    story.append(self.bullet_list([obj_to_text(v) for v in value], body))
                else:
                    for p in split_paragraphs(obj_to_text(value, 40000)):
                        story.append(Paragraph(self.pdf_escape(p), body))

        guides = ensure_list(tutor.get("feature_build_guides"))
        if guides:
            story.append(Paragraph("Feature Build Guides", h2))
            for guide in guides:
                if isinstance(guide, dict):
                    name = str(guide.get("feature") or guide.get("name") or "Feature Guide")
                    story.append(Paragraph(self.pdf_escape(name), body))
                    story.append(self.bullet_list(json_to_lines(guide, 1), body))

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
                story.append(self.bullet_list([obj_to_text(v) for v in value], body))
            else:
                for p in split_paragraphs(obj_to_text(value, 40000)):
                    story.append(Paragraph(self.pdf_escape(p), body))

    def bullet_list(self, items: List[str], body_style) -> ListFlowable:
        flow = [ListItem(Paragraph(self.pdf_escape(item), body_style)) for item in ensure_list_of_str(items)]
        return ListFlowable(flow, bulletType="bullet", leftIndent=14)

    def pdf_escape(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )


# =========================================================
# Entrypoint
# =========================================================

if __name__ == "__main__":
    app = GovernanceTerminal()
    app.run()
