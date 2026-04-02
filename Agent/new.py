from __future__ import annotations

import io
import json
import os
import re
import textwrap
import time
import traceback
import urllib.request
import urllib.parse
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table as RLTable,
    TableStyle,
    KeepTogether,
)

load_dotenv()

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


def as_text(value: Any, limit: int = 200000) -> str:
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


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text.strip("_") or "artifact"


def wrap_text(text: str, width: int = 90) -> str:
    """Wrap long text for readable display."""
    return "\n".join(textwrap.fill(line, width) for line in text.splitlines())


# =========================================================
# Diagram generation via mermaid.ink
# =========================================================

def render_mermaid_to_image(mermaid_code: str, output_path: Path, timeout: int = 20) -> bool:
    """Render a mermaid diagram to PNG via mermaid.ink API."""
    try:
        encoded = urllib.parse.quote(mermaid_code, safe="")
        url = f"https://mermaid.ink/img/{urllib.parse.quote(mermaid_code.encode('utf-8').hex()[:], safe='')}"
        # Use the base64 approach which is more reliable
        import base64
        b64 = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("utf-8")
        url = f"https://mermaid.ink/img/{b64}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if data and len(data) > 500:
            output_path.write_bytes(data)
            return True
        return False
    except Exception:
        return False


def render_mermaid_via_kroki(mermaid_code: str, output_path: Path, timeout: int = 20) -> bool:
    """Fallback: render via kroki.io"""
    try:
        import base64
        import zlib
        compressed = zlib.compress(mermaid_code.encode("utf-8"), 9)
        b64 = base64.urlsafe_b64encode(compressed).decode("utf-8")
        url = f"https://kroki.io/mermaid/png/{b64}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if data and len(data) > 500:
            output_path.write_bytes(data)
            return True
        return False
    except Exception:
        return False


def get_diagram_image(mermaid_code: str, output_path: Path) -> Optional[Path]:
    """Try to render a mermaid diagram, return path if successful."""
    if render_mermaid_via_kroki(mermaid_code, output_path):
        return output_path
    if render_mermaid_to_image(mermaid_code, output_path):
        return output_path
    return None


# =========================================================
# Phases / Constants
# =========================================================

PHASE_REQUIREMENTS = "REQUIREMENTS"
PHASE_PLANNING = "PLANNING"
PHASE_APPROVED = "APPROVED"
PHASE_DEVELOPMENT = "DEVELOPMENT"

FIELD_PROMPTS = {
    "project_goal": "What exactly should the software do at a high level?",
    "target_users": "Who will use this system most often?",
    "project_class": "What is the main project type? Use one label such as web_app, fullstack_app, mobile_app, desktop_app, api_service, static_website, cli_tool, library_sdk, automation_tool, data_pipeline, ai_system, research_prototype, or infrastructure_project.",
    "capabilities": "Which capabilities are needed? Use comma-separated labels such as frontend, backend, data, auth, ai_llm, integrations, analytics, realtime, payments, admin_panel, public_api, batch_jobs, or devops.",
    "complexity_level": "Use one label such as simple, moderate, advanced, or high_scale.",
    "risk_level": "Use one label such as low, medium, or high.",
    "data_sensitivity": "Use one label such as none, internal, personal, financial, health, or confidential.",
    "external_exposure": "Use one label such as local_only, internal_only, private_authenticated, partner_facing, or public_internet.",
    "access_model": "Should it be public, anonymous, account-based, subscription-based, or something else?",
    "feature_scope": "What major features should be included?",
    "frontend_stack": "What frontend stack should be used?",
    "backend_stack": "What backend stack should be used?",
    "data_platform": "What database and storage approach should be used?",
    "hosting_target": "Where should it be deployed or hosted?",
    "security_baseline": "What basic security and abuse-prevention controls are required?",
    "privacy_retention_policy": "How should logs, stored data, and retention be handled?",
    "mvp_scope": "What should the first shippable version include?",
    "future_scope": "What can be phased after MVP?",
    "constraints": "What practical constraints exist?",
    "observability_baseline": "What logging, metrics, tracing, and alerting should exist?",
    "execution_preference": "How should execution trade-offs be handled?",
    "llm_integration": "What model integration strategy should be used?",
    "compliance_context": "What compliance or privacy posture is expected?",
}

CORE_REQUIRED_FIELDS = [
    "project_goal", "target_users", "project_class", "capabilities",
    "access_model", "feature_scope", "risk_level",
    "data_sensitivity", "external_exposure", "security_baseline",
]

INTERNAL_PLANNING_FIELDS = [
    "future_scope", "constraints", "observability_baseline",
    "execution_preference", "llm_integration", "compliance_context",
]

PROJECT_CLASS_DEFAULT_CAPABILITIES = {
    "static_website": ["frontend"],
    "landing_page": ["frontend"],
    "dashboard": ["frontend", "backend", "data"],
    "web_app": ["frontend", "backend", "data", "devops"],
    "fullstack_app": ["frontend", "backend", "data", "devops"],
    "mobile_app": ["frontend", "backend", "data", "devops"],
    "desktop_app": ["frontend", "backend", "data"],
    "api_service": ["backend", "data", "devops"],
    "cli_tool": [],
    "library_sdk": [],
    "automation_tool": ["backend", "batch_jobs"],
    "data_pipeline": ["backend", "data", "batch_jobs", "devops"],
    "ai_system": ["backend", "data", "ai_llm", "devops"],
    "research_prototype": [],
    "infrastructure_project": ["devops"],
}

SPECIALISTS = [
    "RequirementCoordinator", "ProjectScopeAgent", "BackendAgent",
    "FrontendAgent", "SecurityAgent", "DataAgent", "DevOpsAgent",
]
REASONERS = [
    "ProductReasoner", "ArchitectReasoner", "SecurityReasoner",
    "ConstraintReasoner", "CriticReasoner", "ContextCompactor",
]
POST_APPROVAL_AGENTS = [
    "ExecutionPlannerAgent", "TutorAgent", "QAEngineerAgent", "NarrativeWriterAgent",
    "DeepSectionWriterAgent", "DiagramAgent",
]
ALL_AGENTS = SPECIALISTS + REASONERS + ["ArchitectAgent", "AuditorAgent"] + POST_APPROVAL_AGENTS

CONTRACT_TO_NOTE_PATH = {
    "project_goal": "project.goal",
    "target_users": "project.target_users",
    "project_class": "project.project_class",
    "capabilities": "project.capabilities",
    "complexity_level": "constraints.complexity_level",
    "risk_level": "security.risk_level",
    "data_sensitivity": "data.sensitivity",
    "external_exposure": "security.external_exposure",
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
# System prompts
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
8. Mandatory requirement blockers are dynamic.
9. Never advance to planning until all active required fields are populated and confirmed.
10. Once planning starts, keep round-by-round turbulence internal.
"""

AGENT_PROMPTS: Dict[str, str] = {
    "RequirementCoordinator": """
You orchestrate requirement gathering.

Rules:
- Inspect the structured requirement contract before asking for more information.
- Ask progressively, not everything at once.
- Use specialist delegation when beneficial.
- Use upsert_contract_field for canonical blocker fields.
- If the user directly answers a blocker-field question, store that field immediately with confirmed=true and needs_confirmation=false.
- Use confirmed=false and needs_confirmation=true only when you are proposing, inferring, or rewording a value the user has not explicitly accepted yet.
- If the user says yes/ok/correct to a proposed value, confirm the relevant pending field(s).
- Never say a field is confirmed unless the canonical contract has been updated for that field.
- If the user confirms they want to proceed, advance to planning in the same turn.
- Do not ask planning-style questions such as monolith vs microservices unless the user volunteers them.
- Keep messages short, warm, and natural.
- Ask for project_class, capabilities, risk_level, data_sensitivity, and external_exposure early.
- Only ask for fields activated by the current project profile.
- Once all active required fields are confirmed, stop requirement gathering and ask whether to start planning.
- Never show raw field names or enum labels directly to the user. Always phrase questions naturally.
- Infer values intelligently from context. If the user describes a ChatGPT-like app, infer ai_system and ai_llm capability without asking explicitly.
""",
    "ProjectScopeAgent": """
Clarify product goal, target users, features, MVP scope, and priorities.
Capture structured notes and propose canonical blocker values when useful.
When done, delegate back to RequirementCoordinator.
""",
    "BackendAgent": """
You are a backend planning specialist during the internal planning phase.
Create a backend architecture sub-plan from the locked requirements, rich requirement notes,
reasoner reviews, issue ledger, and revision memory.

Return JSON only with:
- service_design
- api_patterns
- business_modules
- llm_integration_design
- background_jobs
- failure_handling
- scaling_notes
- backend_risks
""",
    "FrontendAgent": """
You are a frontend planning specialist during the internal planning phase.
Create a frontend architecture sub-plan from the locked requirements, rich requirement notes,
reasoner reviews, issue ledger, and revision memory.

Return JSON only with:
- app_structure
- pages_and_flows
- state_management
- ui_modules
- accessibility_notes
- frontend_security_notes
- performance_notes
- frontend_risks
""",
    "SecurityAgent": """
You are a security planning specialist during the internal planning phase.
Create a security architecture sub-plan from the locked requirements, rich requirement notes,
reasoner reviews, issue ledger, and revision memory.

Return JSON only with:
- auth_design
- authorization_model
- secrets_management
- abuse_prevention
- privacy_controls
- audit_and_logging_controls
- incident_response_notes
- security_risks
""",
    "DataAgent": """
You are a data planning specialist during the internal planning phase.
Create a data architecture sub-plan from the locked requirements, rich requirement notes,
reasoner reviews, issue ledger, and revision memory.

Return JSON only with:
- entities
- storage_design
- schema_notes
- retention_and_deletion
- analytics_events
- consistency_notes
- migration_notes
- data_risks
""",
    "DevOpsAgent": """
You are a DevOps planning specialist during the internal planning phase.
Create an infrastructure and operations sub-plan from the locked requirements, rich requirement notes,
reasoner reviews, issue ledger, and revision memory.

Return JSON only with:
- deployment_topology
- environments
- ci_cd_design
- observability_stack
- rollback_strategy
- backup_and_recovery
- cost_controls
- devops_risks
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
- focus issues
- revision memory
- previous audits
- best prior plan

Main goal:
- First, address the current focus issues.
- Second, preserve all user-confirmed requirements.
- Third, improve the architecture without introducing regressions.

Rules:
- Do not mention round numbers in the title.
- Preserve user-confirmed requirements.
- Prioritize unresolved critical and high-severity focus issues first.
- For each focus issue, either fix it in the plan or clearly explain why it remains unresolved.
- Do not ignore recurring unresolved issues from previous rounds.
- Try to improve weak areas identified by the auditor before adding extra design complexity.
- Include concrete architecture, modules, workflows, schemas, APIs, security, deployment, observability, roadmap, and developer guidance.
- Keep the plan implementation-grade and specific, not generic.

Return JSON only with:
- thinking_summary
- fix_report
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

fix_report must be a list of items with:
- issue_id
- action_taken
- changed_sections
- expected_outcome

For each fix_report item:
- issue_id must match the issue being addressed
- action_taken must say what was changed
- changed_sections must name the plan sections updated
- expected_outcome must explain what the auditor should now find improved
""",
    "AuditorAgent": """
You are the strict architecture auditor.

Audit the architecture plan against:
- frozen confirmed requirements
- rich requirement notes
- cumulative issue ledger
- revision memory
- prior audit history

Main goal:
- First, verify whether previously reported issues were actually fixed.
- Second, identify the most important remaining weaknesses.
- Third, explain clearly why the score stayed the same, improved, or dropped.

Rules:
- Use stable issue IDs whenever the same issue still exists.
- Mark each issue status as one of: unresolved, resolved, downgraded, new.
- Re-check prior unresolved issues before creating new ones.
- If an earlier issue was fixed, keep the same issue ID and mark it resolved.
- If an earlier issue still exists, keep the same issue ID and explain what is still missing.
- Only create a new issue ID if the problem is materially different from previous issues.
- Score the plan against an absolute rubric, not against any approval threshold.
- Do not try to make the plan pass or fail a gate.
- Be willing to score below 9 if the plan has real weaknesses.
- If the score drops, explain the exact reason for the drop.
- If the score does not improve, explain what blocked improvement.
- Prefer the most important unresolved issues over minor nitpicks.
- passed is advisory only; the runtime decides approval.

Return JSON only with:
- thinking_summary
- rubric_scores
- summary
- strengths
- concerns
- blocking_issues
- recommendations
- requirement_conflicts
- issue_updates

rubric_scores must include numeric values from 0 to 10 for:
- requirements_alignment
- architecture_quality
- security
- operability
- internal_consistency

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

For each issue_updates.detail:
- State whether the issue was fixed, partially fixed, unchanged, or newly introduced.
- Explain exactly what in the plan caused this judgment.
- If the issue affected the score, explain how.
- If the architect improved one part but created another problem, say that clearly.

recommendations should:
- focus on the next highest-impact fixes
- be specific enough for the architect to act on in the next round
- avoid vague advice like "improve architecture quality"

summary should:
- briefly explain overall quality
- say whether the round meaningfully improved over the prior round
- mention the main reason the score changed or stayed flat
""",
    "ExecutionPlannerAgent": """
Transform the approved architecture into a detailed implementation roadmap.

Return JSON only with:
- execution_overview
- implementation_phases (each phase: phase_name, duration_estimate, objectives, deliverables, tasks, dependencies, team_roles, done_criteria, risks)
- feature_workstreams (each: feature, tasks, estimated_effort, dependencies)
- dependency_map
- milestone_checks (each: milestone, criteria, verification_method)
- rollout_strategy
- infrastructure_checklist
- go_live_checklist
""",
    "TutorAgent": """
Create a practical development playbook for implementing the approved plan.

Return JSON only with:
- development_playbook
- coding_order
- implementation_tips
- common_mistakes (each: mistake, why_it_happens, how_to_avoid)
- feature_build_guides (each: feature, step_by_step_guide, code_patterns, testing_approach)
- environment_setup_guide
- branching_strategy
- code_review_checklist
- performance_considerations
""",
    "QAEngineerAgent": """
Create a testing and validation package from the approved architecture and execution plan.

Return JSON only with:
- validation_strategy
- test_layers (object with: unit, integration, e2e, performance, security, accessibility)
- detailed_test_plan (list of test suites, each with: suite_name, scope, test_cases list with: id, description, preconditions, steps, expected_result, severity)
- acceptance_criteria
- regression_strategy
- release_readiness_checklist (grouped: functionality, performance, security, ops)
- test_data_strategy
- continuous_testing_plan
- defect_management_process
""",
    "DiagramAgent": """
Generate Mermaid diagram code for the approved architecture.
Return JSON only with:
- system_architecture (Mermaid graph LR code for the complete system)
- sequence_diagram (Mermaid sequenceDiagram for the main user flow)
- data_model_erd (Mermaid erDiagram for the database schema)
- deployment_diagram (Mermaid graph TB for deployment topology)
- component_diagram (Mermaid graph LR for internal components)
- cicd_pipeline (Mermaid graph LR for CI/CD flow)
- user_journey (Mermaid journey for the primary user flow)

Each diagram must be complete, valid Mermaid syntax that renders correctly.
Do not include markdown code fences - just the raw Mermaid code.
""",
    "DeepSectionWriterAgent": """
You write an extremely detailed, comprehensive section of a technical architecture document.
You write like a senior staff engineer writing for other engineers.
Your output should be thorough, specific, and deeply technical.
Use the provided plan data and context to write a rich, detailed section.
Minimum 800 words. Include specific technology choices, code patterns, configuration details, and rationale.

Return JSON only with:
- section_content (the full written section as a string, using \\n for newlines)
- key_decisions (list of architectural decisions made in this section)
- implementation_notes (list of specific implementation guidance points)
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
        "project": {}, "frontend": {}, "backend": {},
        "security": {}, "data": {}, "devops": {},
        "constraints": {}, "open_questions": {}, "confirmed_decisions": {},
    })

    requirement_status: Dict[str, Any] = field(default_factory=lambda: {
        "ready_for_planning": False, "completeness_score": 0.0,
        "summary": "", "last_updated": None,
    })
    # Tracks exactly what field was last asked about to avoid repetition
    last_requested_fields: List[str] = field(default_factory=list)
    # Tracks which fields have been asked about this session to prevent looping
    fields_asked_this_session: List[str] = field(default_factory=list)
    planning_confirmation_requested: bool = False

    pass_threshold: float = 9.0
    max_requirement_hops: int = 12
    max_tool_rounds: int = 10
    max_planning_rounds: int = 10
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

    focus_issues: List[Dict[str, Any]] = field(default_factory=list)
    convergence_state: Dict[str, Any] = field(default_factory=dict)
    finalization_reason: str = ""

    # Diagram cache
    generated_diagrams: Dict[str, str] = field(default_factory=dict)


# =========================================================
# Azure LLM client
# =========================================================

class AzureLLM:
    def __init__(self) -> None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
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

    def complete_text(
        self,
        system_prompt: str,
        user_content: str,
        max_tokens: int = 4000,
        temperature: float = 0.3,
    ) -> str:
        """Generate free-form text (not JSON)."""
        model = self.reasoning_deployment
        resp = self.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


# =========================================================
# Main application
# =========================================================

class GovernanceHybridApp:
    def __init__(self) -> None:
        self.console = Console()
        self.state = SharedState()
        self.llm = AzureLLM()

        self.state.artifacts_dir = str(Path("artifacts") / self.state.session_id[:8])
        Path(self.state.artifacts_dir).mkdir(parents=True, exist_ok=True)
        Path(self.state.artifacts_dir, "diagrams").mkdir(exist_ok=True)

    # ----------------------------------------------------------
    # UI helpers
    # ----------------------------------------------------------

    def banner(self) -> None:
        self.console.print(Rule("[bold cyan]Architectural Governance Terminal"))
        self.console.print("[dim]Type your project idea. Type 'exit' to quit.[/dim]")
        self.console.print(
            "[dim]Commands: :threshold 9.0 | :rounds 10 | :debug on/off | "
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
            content = (turn.content or "").strip()
            if turn.role == "assistant" and turn.agent:
                content = self.clean_assistant_text(content, turn.agent)
                if content:
                    out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": turn.role, "content": content})
        return out

    def show_status(self) -> None:
        t = Table(title="Runtime Status", box=box.SIMPLE_HEAVY)
        t.add_column("Field", style="cyan", width=28)
        t.add_column("Value")
        t.add_row("Phase", self.state.phase)
        t.add_row("Active agent", self.state.active_agent)
        t.add_row("Threshold", f"{self.state.pass_threshold:.2f}")
        t.add_row("Pending confirmations", ", ".join(self.state.pending_confirmations) or "None")
        t.add_row("Missing required", ", ".join(self.missing_required_fields()) or "None")
        t.add_row("Planning rounds", str(self.state.max_planning_rounds))
        t.add_row("Known issues", str(len(self.state.issue_ledger)))
        t.add_row("Best score", f"{self.state.best_audit.get('score', 0):.2f}" if self.state.best_audit else "N/A")
        t.add_row("Approved PDF", self.state.final_pdf_path or "None")
        self.console.print(t)

    # ----------------------------------------------------------
    # Contract + structured memory
    # ----------------------------------------------------------

    def set_contract_field(self, field_name: str, value: str, source: str, confirmed: bool, rationale: str) -> None:
        if field_name not in self.state.requirement_contract:
            return
        existing = self.state.requirement_contract[field_name]
        clean_value = self.canonicalize_contract_value(field_name, str(value).strip())
        if not clean_value:
            return
        same_value = clean_value == str(existing.value or "").strip()
        confirmed_state = bool(confirmed) or (existing.confirmed and same_value)
        self.state.requirement_contract[field_name] = RequirementField(
            value=clean_value,
            source=str(source).strip() or existing.source,
            confirmed=confirmed_state,
            rationale=str(rationale).strip() or existing.rationale,
            updated_at=now_iso(),
        )
        note_path = CONTRACT_TO_NOTE_PATH.get(field_name)
        if note_path:
            deep_set(self.state.requirements, note_path, clean_value)

    def confirm_fields(self, fields: List[str]) -> None:
        for f in fields:
            if f not in self.state.requirement_contract:
                continue
            item = self.state.requirement_contract[f]
            if not item.value.strip():
                continue
            item.confirmed = True
            item.updated_at = now_iso()

    def get_contract_value(self, field_name: str) -> str:
        item = self.state.requirement_contract.get(field_name)
        if not item:
            return ""
        return str(item.value or "").strip().lower()

    def contract_tokens(self, field_name: str) -> List[str]:
        raw = self.get_contract_value(field_name)
        if not raw:
            return []
        parts = re.split(r"[,;\n|]+", raw)
        tokens = []
        for part in parts:
            token = re.sub(r"[^a-z0-9_ -]+", "", part.strip().lower())
            token = token.replace("-", "_").replace(" ", "_").strip("_")
            if token:
                tokens.append(token)
        return unique_strs(tokens)

    def normalized_project_class(self) -> str:
        value = self.get_contract_value("project_class")
        value = re.sub(r"[^a-z0-9_ -]+", "", value)
        value = value.replace("-", "_").replace(" ", "_").strip("_")
        return value

    def inferred_capabilities(self) -> List[str]:
        project_class = self.normalized_project_class()
        caps = set(self.contract_tokens("capabilities"))
        for cap in PROJECT_CLASS_DEFAULT_CAPABILITIES.get(project_class, []):
            caps.add(cap)
        return sorted(caps)

    def active_required_fields(self) -> List[str]:
        required = list(CORE_REQUIRED_FIELDS)
        caps = set(self.inferred_capabilities())
        project_class = self.normalized_project_class()
        risk = self.get_contract_value("risk_level")
        sensitivity = self.get_contract_value("data_sensitivity")
        exposure = self.get_contract_value("external_exposure")

        frontend_classes = {"static_website", "landing_page", "dashboard", "web_app", "fullstack_app", "mobile_app", "desktop_app"}
        backend_classes = {"web_app", "fullstack_app", "api_service", "automation_tool", "data_pipeline", "ai_system"}
        data_classes = {"dashboard", "web_app", "fullstack_app", "mobile_app", "desktop_app", "api_service", "data_pipeline", "ai_system"}

        if "frontend" in caps or "admin_panel" in caps or project_class in frontend_classes:
            required.append("frontend_stack")
        if "backend" in caps or project_class in backend_classes or {"public_api", "ai_llm", "batch_jobs"} & caps:
            required.append("backend_stack")
        if "data" in caps or project_class in data_classes or sensitivity not in {"", "none"}:
            required.append("data_platform")
        if ("frontend" in caps or "backend" in caps or "data" in caps or "devops" in caps
                or exposure in {"internal_only", "private_authenticated", "partner_facing", "public_internet"}):
            required.append("hosting_target")
        if (sensitivity not in {"", "none"} or risk in {"medium", "high"}
                or exposure in {"private_authenticated", "partner_facing", "public_internet"}
                or {"auth", "analytics", "payments", "ai_llm"} & caps):
            required.append("privacy_retention_policy")
        if "ai_llm" in caps:
            required.append("llm_integration")
        if (risk == "high" or sensitivity in {"personal", "financial", "health", "confidential"} or "payments" in caps):
            required.append("compliance_context")

        # mvp_scope is only meaningful for complex projects where MVP differs from full scope.
        # For simple static sites and landing pages the MVP is trivially the full page — don't require it.
        simple_classes = {"static_website", "landing_page", "cli_tool", "library_sdk", "research_prototype"}
        if project_class not in simple_classes:
            required.append("mvp_scope")

        return unique_strs(required)

    def missing_required_fields(self) -> List[str]:
        out = []
        for f in self.active_required_fields():
            item = self.state.requirement_contract[f]
            if not item.value.strip() or not item.confirmed:
                out.append(f)
        return out

    def next_missing_field(self) -> Optional[str]:
        missing = self.missing_required_fields()
        return missing[0] if missing else None

    def all_required_locked(self) -> bool:
        return len(self.missing_required_fields()) == 0

    def contract_snapshot(self) -> Dict[str, Any]:
        return {k: asdict(v) for k, v in self.state.requirement_contract.items()}

    def frozen_contract(self) -> Dict[str, Any]:
        return {k: asdict(v) for k, v in self.state.requirement_contract.items() if v.value.strip()}

    def fill_internal_defaults(self) -> None:
        defaults = {
            "future_scope": "Derive post-MVP enhancements internally from the requested product direction.",
            "constraints": "Assume limited resources with emphasis on quality, maintainability, and implementation readiness.",
            "execution_preference": "Prioritize correctness, maintainability, and secure implementation readiness over brevity.",
        }
        caps = set(self.inferred_capabilities())
        risk = self.get_contract_value("risk_level")
        sensitivity = self.get_contract_value("data_sensitivity")
        exposure = self.get_contract_value("external_exposure")

        if "frontend" in caps or "backend" in caps or "data" in caps or "devops" in caps:
            defaults["observability_baseline"] = (
                "Structured logs, error tracking, request metrics, latency monitoring, traces, uptime alerts, and audit events."
            )
        if "ai_llm" in caps:
            defaults["llm_integration"] = (
                "Use secure backend-managed GPT-compatible integration through an adapter layer, never direct browser-side secret exposure."
            )
        if (risk == "high" or sensitivity in {"personal", "financial", "health", "confidential"}
                or exposure in {"private_authenticated", "partner_facing", "public_internet"}):
            defaults["compliance_context"] = (
                "Adopt privacy-by-design with deletion support, retention enforcement, secret storage, auditability, and explicit handling of user data and logs."
            )

        # Auto-derive mvp_scope for simple project types where MVP == full scope.
        project_class = self.normalized_project_class()
        simple_classes = {"static_website", "landing_page", "cli_tool", "library_sdk", "research_prototype"}
        if project_class in simple_classes:
            feature_scope = self.get_contract_value("feature_scope")
            defaults["mvp_scope"] = (
                f"Full feature set delivered at initial launch: {feature_scope}."
                if feature_scope else "Complete project delivered at initial launch with all requested features."
            )

        for field_name, value in defaults.items():
            current = self.state.requirement_contract[field_name]
            if not current.value.strip():
                self.set_contract_field(
                    field_name=field_name, value=value,
                    source="system_default_for_planning", confirmed=True,
                    rationale="Internal planning default; not a user-blocking requirement.",
                )

    def ai_json(self, system_prompt: str, payload: Dict[str, Any], max_tokens: int = 450) -> Dict[str, Any]:
        try:
            result = self.llm.complete_json(
                system_prompt=system_prompt, payload=payload,
                max_tokens=max_tokens, reasoning=True, temperature=0.0,
            )
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}

    def allowed_values_for_field(self, field_name: str) -> List[str]:
        controlled = {
            "project_class": [
                "web_app", "fullstack_app", "mobile_app", "desktop_app", "api_service",
                "static_website", "landing_page", "cli_tool", "library_sdk", "automation_tool",
                "data_pipeline", "ai_system", "research_prototype", "infrastructure_project",
            ],
            "capabilities": [
                "frontend", "backend", "data", "auth", "ai_llm", "integrations", "analytics",
                "realtime", "payments", "admin_panel", "public_api", "batch_jobs", "devops",
            ],
            "complexity_level": ["simple", "moderate", "advanced", "high_scale"],
            "risk_level": ["low", "medium", "high"],
            "data_sensitivity": ["none", "internal", "personal", "financial", "health", "confidential"],
            "external_exposure": ["local_only", "internal_only", "private_authenticated", "partner_facing", "public_internet"],
        }
        return controlled.get(field_name, [])

    def last_assistant_text(self) -> str:
        for turn in reversed(self.state.dialogue):
            if turn.role == "assistant":
                if turn.agent:
                    return self.clean_assistant_text(turn.content, turn.agent)
                return str(turn.content or "").strip()
        return ""

    def canonicalize_contract_value(self, field_name: str, value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        allowed = self.allowed_values_for_field(field_name)
        payload = {
            "field_name": field_name, "raw_value": raw,
            "allowed_values": allowed, "field_prompt": FIELD_PROMPTS.get(field_name, ""),
        }
        prompt = """
        Normalize one requirement value.
        Return ONLY valid JSON with:
        - canonical_value: string
        - canonical_list: array of strings
        Rules:
        - If allowed_values is empty, return a concise cleaned rewrite in canonical_value.
        - If field_name is capabilities, put only exact allowed_values in canonical_list.
        - If field_name has controlled allowed_values, return exactly one from allowed_values in canonical_value.
        - If unclear for a controlled field, return the original trimmed value in canonical_value.
        """
        result = self.ai_json(prompt, payload, max_tokens=250)
        if field_name == "capabilities":
            values = [v for v in ensure_list_of_str(result.get("canonical_list")) if v in allowed]
            return ", ".join(unique_strs(values)) if values else raw
        canonical = str(result.get("canonical_value") or raw).strip()
        if allowed and canonical not in allowed:
            return raw
        return canonical

    def sync_pending_confirmations(self) -> None:
        self.state.pending_confirmations = [
            f for f in unique_strs(self.state.pending_confirmations)
            if f in self.state.requirement_contract
            and self.state.requirement_contract[f].value.strip()
            and not self.state.requirement_contract[f].confirmed
        ]

    def interpret_user_message(self, text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            return {
                "is_affirmation": False, "is_clarification": False,
                "explicitly_requests_planning": False,
                "answered_fields": [], "answer_value": "",
            }

        payload = {
            "phase": self.state.phase,
            "user_text": raw,
            "last_assistant_message": self.last_assistant_text(),
            "last_requested_fields": list(self.state.last_requested_fields),
            "pending_confirmations": list(self.state.pending_confirmations),
            "missing_required_fields": self.missing_required_fields(),
            "all_required_locked": self.all_required_locked(),
            "field_prompts": FIELD_PROMPTS,
            "requirement_contract": self.contract_snapshot(),
        }

        prompt = """
        You are a smart intent classifier for a requirement-gathering conversation.
        Return ONLY valid JSON with:
        - is_affirmation: boolean
        - is_clarification: boolean
        - explicitly_requests_planning: boolean
        - answered_fields: array of exact field names from payload.field_prompts
        - answer_value: string

        Classification rules:
        - is_affirmation: true when the user is confirming, agreeing, or saying yes to something proposed. Words like yes, yeah, yep, ok, okay, sure, correct, right, sounds good, that works, all of them, everything.
        - is_clarification: true ONLY when the user is genuinely asking for an explanation or clarification, not answering. Do NOT set this to true when the user is giving a vague but valid answer like "everything" or "you decide".
        - explicitly_requests_planning: true when the user clearly wants to move into planning now. This includes: "yes" as a response to a planning question, "start planning", "go ahead", "let's go", "proceed", "begin", etc.
        - answered_fields: use last_requested_fields as the primary signal. If the last assistant message asked about a specific field and the user's response (even vague like "everything") is an answer to that field, include it.
        - answer_value: the user's answer text. For vague answers like "all of them", "everything", "you decide", preserve the original text — these are valid answers.
        - IMPORTANT: "everything", "all of them", "all", "you decide", "whatever works" are VALID ANSWERS to most fields. Do not mark them as clarifications.
        - IMPORTANT: When the context shows a planning confirmation question was just asked and the user says "yes" or similar, set explicitly_requests_planning=true.
        """

        result = self.ai_json(prompt, payload, max_tokens=400)
        fields = [f for f in ensure_list_of_str(result.get("answered_fields")) if f in FIELD_PROMPTS]
        return {
            "is_affirmation": bool(result.get("is_affirmation", False)),
            "is_clarification": bool(result.get("is_clarification", False)),
            "explicitly_requests_planning": bool(result.get("explicitly_requests_planning", False)),
            "answered_fields": unique_strs(fields),
            "answer_value": str(result.get("answer_value") or "").strip(),
        }

    def wants_planning_transition(self, text: str) -> bool:
        return bool(self.interpret_user_message(text).get("explicitly_requests_planning", False))

    def infer_requested_fields_from_text(self, text: str) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        payload = {
            "assistant_text": raw,
            "field_prompts": FIELD_PROMPTS,
            "missing_required_fields": self.missing_required_fields(),
            "pending_confirmations": list(self.state.pending_confirmations),
        }
        prompt = """
        Read one assistant message and infer which requirement field it is asking about.
        Return ONLY valid JSON with:
        - fields: array of exact field names from payload.field_prompts

        Rules:
        - Return only exact field names from payload.field_prompts.
        - Prefer the single main field being asked for.
        - Return [] if the assistant text is not asking for requirement data.
        """
        result = self.ai_json(prompt, payload, max_tokens=200)
        fields = [f for f in ensure_list_of_str(result.get("fields")) if f in FIELD_PROMPTS]
        unresolved = set(self.missing_required_fields()) | set(self.state.pending_confirmations)
        ranked = [f for f in fields if f in unresolved]
        return unique_strs(ranked or fields)

    def remember_requirement_prompt(self, assistant_text: str) -> None:
        if self.state.phase != PHASE_REQUIREMENTS:
            return
        fields = self.infer_requested_fields_from_text(assistant_text)
        if fields:
            self.state.last_requested_fields = fields[:1]
            # Track what we've asked about to prevent repetition
            for f in fields[:1]:
                if f not in self.state.fields_asked_this_session:
                    self.state.fields_asked_this_session.append(f)

    def capture_direct_user_answer(self, user_text: str) -> bool:
        """
        FIXED: More robust answer capture with multiple fallback strategies.
        Returns True if the answer was successfully captured and stored.
        """
        text = str(user_text or "").strip()
        if not text:
            return False

        self.sync_pending_confirmations()
        analysis = self.interpret_user_message(text)

        # Strategy 1: User is confirming a pending proposed value
        if self.state.pending_confirmations and analysis["is_affirmation"]:
            self.confirm_fields(self.state.pending_confirmations)
            self.sync_pending_confirmations()
            return True

        # Strategy 2: Not a clarification question
        if analysis["is_clarification"]:
            return False

        # Strategy 3: Determine target field
        targets = list(analysis.get("answered_fields") or [])

        # Fallback: use the field we last asked about
        if not targets and self.state.last_requested_fields:
            targets = list(self.state.last_requested_fields)

        # Fallback: if project_goal is empty and this is the first message, capture it
        if not targets and not self.state.requirement_contract["project_goal"].value.strip():
            targets = ["project_goal"]

        # Fallback: use the first missing required field if we have a clear direct answer
        if not targets and not analysis["is_clarification"] and analysis["answer_value"]:
            missing = self.missing_required_fields()
            if missing:
                # Only if what we'd set is not obviously wrong
                targets = [missing[0]]

        if not targets:
            return False

        # Strategy 4: Affirmation — confirm the primary target field or all pending.
        field_name = targets[0]
        if analysis["is_affirmation"]:
            current = self.state.requirement_contract[field_name]
            if current.value.strip() and not current.confirmed:
                self.confirm_fields([field_name])
                self.sync_pending_confirmations()
                return True
            if self.state.pending_confirmations:
                self.confirm_fields(self.state.pending_confirmations)
                self.sync_pending_confirmations()
                return True
            return False

        # Strategy 5: Store the direct answer for ALL identified target fields.
        # This handles compound answers like "features are X and Y; MVP is the full release".
        value = str(analysis.get("answer_value") or "").strip()
        if not value:
            # Fall back to raw text for free-form fields.
            value = text

        stored_any = False
        for target_field in targets:
            if not value:
                continue
            self.set_contract_field(
                field_name=target_field,
                value=value,
                source="user_direct_answer",
                confirmed=True,
                rationale="Captured from user's direct answer.",
            )
            stored_any = True

        if stored_any:
            self.sync_pending_confirmations()
        return stored_any

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
            payload, max_tokens=1000, reasoning=True,
        )
        summary = result.get("summary")
        if summary:
            self.state.context_summary = str(summary)
            self.state.dialogue = self.state.dialogue[-10:]

    # ----------------------------------------------------------
    # Tool calling for requirement phase
    # ----------------------------------------------------------

    def state_snapshot(self) -> Dict[str, Any]:
        return {
            "phase": self.state.phase,
            "active_agent": self.state.active_agent,
            "context_summary": self.state.context_summary,
            "requirement_contract": self.contract_snapshot(),
            "pending_confirmations": self.state.pending_confirmations,
            "missing_required_fields": self.missing_required_fields(),
            "next_missing_field": self.next_missing_field(),
            "all_required_locked": self.all_required_locked(),
            "requirements": self.state.requirements,
            "requirement_status": self.state.requirement_status,
            "issue_ledger": self.state.issue_ledger,
            "revision_memory": self.state.revision_memory,
            "pass_threshold": self.state.pass_threshold,
            "debug_mode": self.state.debug_mode,
            "last_requested_fields": self.state.last_requested_fields,
            "fields_asked_this_session": self.state.fields_asked_this_session,
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
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    def tool_schemas(self, agent_name: str) -> List[Dict[str, Any]]:
        valid_targets = [a for a in ALL_AGENTS if a != agent_name]
        return [
            self.schema("inspect_contract", "Inspect the canonical requirement contract or one field.",
                        {"field": {"type": "string"}}, []),
            self.schema("inspect_requirement_notes", "Inspect the rich structured requirement notes or one section.",
                        {"section": {"type": "string"}}, []),
            self.schema("upsert_contract_field", "Write or update a canonical requirement contract field.",
                        {
                            "field": {"type": "string"},
                            "value": {"type": "string"},
                            "rationale": {"type": "string"},
                            "confirmed": {"type": "boolean"},
                            "needs_confirmation": {"type": "boolean"},
                        },
                        ["field", "value", "rationale", "confirmed", "needs_confirmation"]),
            self.schema("confirm_contract_fields", "Confirm one or more contract fields after explicit user confirmation.",
                        {"fields": {"type": "array", "items": {"type": "string"}}}, ["fields"]),
            self.schema("upsert_requirement_note", "Write or update a richer structured requirement note.",
                        {"path": {"type": "string"}, "value": {"type": "string"}, "rationale": {"type": "string"}},
                        ["path", "value", "rationale"]),
            self.schema("log_thinking", "Show concise summarized reasoning in the terminal.",
                        {"summary": {"type": "string"}, "confidence": {"type": "number"}, "next_action": {"type": "string"}},
                        ["summary", "confidence", "next_action"]),
            self.schema("consult_reasoner", "Consult a reasoner or specialist for deeper analysis.",
                        {"agent": {"type": "string", "enum": valid_targets}, "task": {"type": "string"}, "deliverable": {"type": "string"}},
                        ["agent", "task", "deliverable"]),
            self.schema("delegate_to", "Transfer control to another specialist agent.",
                        {"agent": {"type": "string", "enum": valid_targets}, "objective": {"type": "string"}, "reason": {"type": "string"}},
                        ["agent", "objective", "reason"]),
            self.schema("set_readiness", "Update requirement completeness and readiness.",
                        {"ready_for_planning": {"type": "boolean"}, "completeness_score": {"type": "number"}, "summary": {"type": "string"}},
                        ["ready_for_planning", "completeness_score", "summary"]),
            self.schema("advance_phase", "Advance to planning when the mandatory requirement contract is ready.",
                        {"target_phase": {"type": "string", "enum": [PHASE_PLANNING]}, "reason": {"type": "string"}},
                        ["target_phase", "reason"]),
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

            user_facing_fields = [f for f in FIELD_PROMPTS.keys() if f not in INTERNAL_PLANNING_FIELDS]
            if needs_confirmation or not confirmed:
                if field_name in user_facing_fields and field_name not in self.state.pending_confirmations:
                    self.state.pending_confirmations.append(field_name)
            else:
                self.state.pending_confirmations = [f for f in self.state.pending_confirmations if f != field_name]

            return {"ok": True, "field": field_name, "value_stored": value}

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
            return {"ok": True, "delegated_to": target}

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
                missing = self.missing_required_fields()
                return {"ok": False, "error": f"Mandatory blocker fields not confirmed: {missing}"}
            self.fill_internal_defaults()
            self.state.phase = PHASE_PLANNING
            self.state.active_agent = "ArchitectAgent"
            self.state.pending_confirmations = []
            return {"ok": True, "phase": self.state.phase}

        return {"ok": False, "error": f"Unknown tool: {tool_name}"}

    def normalize_tool_name(self, name: str) -> str:
        n = (name or "").strip()
        if n.startswith("functions."):
            n = n.split(".", 1)[1]
        return n

    def extract_pseudo_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        text = (content or "").strip()
        if not text:
            return []
        data = safe_json_loads(text)
        if not isinstance(data, dict):
            return []
        calls: List[Dict[str, Any]] = []
        tool_uses = data.get("tool_uses")
        if isinstance(tool_uses, list):
            for item in tool_uses:
                if not isinstance(item, dict):
                    continue
                raw_name = item.get("recipient_name") or item.get("name") or item.get("tool")
                args = item.get("parameters") or item.get("arguments") or {}
                tool_name = self.normalize_tool_name(str(raw_name or ""))
                if tool_name:
                    calls.append({"name": tool_name, "args": args if isinstance(args, dict) else {}})
        if not calls and data.get("target_phase") == PHASE_PLANNING:
            calls.append({"name": "advance_phase", "args": {"target_phase": PHASE_PLANNING, "reason": str(data.get("reason", "")).strip()}})
        return calls

    def clean_assistant_text(self, content: str, agent_name: str) -> str:
        text = (content or "").strip()
        if not text:
            return text
        prefixes = [f"{agent_name}:", "RequirementCoordinator:", "ProjectScopeAgent:", "BackendAgent:", "FrontendAgent:", "SecurityAgent:", "DataAgent:", "DevOpsAgent:"]
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
                    changed = True
        if safe_json_loads(text):
            parsed = safe_json_loads(text)
            if isinstance(parsed, dict) and ("tool_uses" in parsed or "target_phase" in parsed):
                return ""
        return text

    def single_requirement_step(self, agent_name: str) -> bool:
        messages = self.build_agent_messages(agent_name)
        # Reserve the last 2 rounds as a forced-text round if the agent
        # has been burning tool calls without producing a visible reply.
        force_text_after = max(1, self.state.max_tool_rounds - 2)
        for round_idx in range(self.state.max_tool_rounds):
            # On the penultimate round, stop providing tools so the model
            # MUST produce a text reply rather than another tool call.
            use_tools = round_idx < force_text_after
            resp = self.llm.completion(
                model=self.llm.chat_deployment,
                messages=messages,
                tools=self.tool_schemas(agent_name) if use_tools else None,
                temperature=0.2,
                max_tokens=1500,
            )
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            raw_content = msg.content or ""

            assistant_message: Dict[str, Any] = {"role": "assistant", "content": raw_content}
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ]
            messages.append(assistant_message)

            if tool_calls:
                active_before = self.state.active_agent
                phase_before = self.state.phase
                for tc in tool_calls:
                    result = self.execute_tool(agent_name, tc.function.name, safe_json_loads(tc.function.arguments))
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "name": tc.function.name, "content": json.dumps(result, ensure_ascii=False),
                    })
                if self.state.phase != phase_before:
                    return False
                if self.state.active_agent != active_before:
                    return False
                continue

            pseudo_calls = self.extract_pseudo_tool_calls(raw_content)
            if pseudo_calls:
                active_before = self.state.active_agent
                phase_before = self.state.phase
                for idx, call in enumerate(pseudo_calls, start=1):
                    result = self.execute_tool(agent_name, call["name"], call["args"])
                    messages.append({
                        "role": "tool", "tool_call_id": f"pseudo_{idx}",
                        "name": call["name"], "content": json.dumps(result, ensure_ascii=False),
                    })
                if self.state.phase != phase_before:
                    return False
                if self.state.active_agent != active_before:
                    return False
                continue

            content = self.clean_assistant_text(raw_content, agent_name)
            if content:
                self.remember_requirement_prompt(content)
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
            payload, max_tokens=1200, reasoning=True,
        )
        if self.state.debug_mode:
            short = result.get("summary") or result.get("next_focus") or "consultation complete"
            self.thinking(agent, short, "consultation complete")
        return result

    # ----------------------------------------------------------
    # Run loop + commands
    # ----------------------------------------------------------

    def run(self) -> None:
        self.banner()
        welcome = (
            "Hi — I'll help you define the project step by step. "
            "I'll lock the mandatory requirement contract first, then move into internal planning and validation."
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
                self.state.pass_threshold = max(7.0, min(10.0, float(parts[1])))
                self.panel("System", f"Pass threshold set to {self.state.pass_threshold:.2f}.", "cyan")
            except Exception:
                self.panel("System", "Invalid threshold value.", "red")
            return True
        if cmd == ":rounds" and len(parts) == 2:
            try:
                self.state.max_planning_rounds = max(1, min(50, int(parts[1])))
                self.panel("System", f"Planning rounds set to {self.state.max_planning_rounds}.", "cyan")
            except Exception:
                self.panel("System", "Invalid round count.", "red")
            return True
        if cmd == ":debug" and len(parts) == 2:
            self.state.debug_mode = parts[1].lower() in {"on", "true", "1"}
            self.panel("System", f"Debug mode: {self.state.debug_mode}.", "cyan")
            return True
        if cmd == ":thinking" and len(parts) == 2:
            self.state.show_internal_panels = parts[1].lower() in {"on", "true", "1"}
            self.panel("System", f"Internal panels: {self.state.show_internal_panels}.", "cyan")
            return True
        if cmd == ":status":
            self.show_status()
            return True
        if cmd == ":export":
            if not self.state.best_plan or not self.state.best_audit:
                self.panel("System", "No approved plan to export yet.", "red")
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

    # ----------------------------------------------------------
    # FIXED Requirement phase handler
    # ----------------------------------------------------------

    def handle_requirement_turn(self, user_text: str) -> None:
        self.sync_pending_confirmations()

        # -------------------------------------------------------
        # CASE 1: We already asked the user if they want to plan
        # -------------------------------------------------------
        if self.state.planning_confirmation_requested:
            wants_to_start = self._user_wants_to_start(user_text)

            # Bug fix: when the user clearly wants to start planning, auto-confirm any
            # pending fields that already have a stored value. These are fields the agent
            # proposed and the user never explicitly rejected — treating a "yes, let's go"
            # as implicit confirmation prevents the bot from blocking on its own pending queue.
            if wants_to_start and self.state.pending_confirmations:
                self.confirm_fields(self.state.pending_confirmations)
                self.sync_pending_confirmations()

            if self.all_required_locked() and wants_to_start:
                self._start_planning()
                return

            # Not starting yet — try to capture any additional requirements in what they said
            self.capture_direct_user_answer(user_text)

            # Re-check after capture
            if self.all_required_locked() and wants_to_start:
                self._start_planning()
                return

            if not self.all_required_locked():
                self.state.planning_confirmation_requested = False

            # Let the agent respond naturally (it knows it already asked about planning)
            self._run_agent_step()
            return

        # -------------------------------------------------------
        # CASE 2: Normal requirement gathering
        # -------------------------------------------------------
        # Try to capture what the user just said into the contract
        captured = self.capture_direct_user_answer(user_text)

        # After capture, check if we are now fully locked
        if self.all_required_locked():
            self.state.requirement_status = {
                "ready_for_planning": True,
                "completeness_score": 1.0,
                "summary": "Mandatory requirement contract fully locked.",
                "last_updated": now_iso(),
            }
            self.state.planning_confirmation_requested = True
            # Let the agent naturally ask whether to start planning
            self._run_agent_step()
            return

        # Whether or not we captured something, always run an agent step.
        # The agent will naturally acknowledge what was said and ask the next thing.
        self._run_agent_step()

        # Final check: did the agent step fill the last required fields?
        if self.all_required_locked():
            self.state.requirement_status = {
                "ready_for_planning": True,
                "completeness_score": 1.0,
                "summary": "Mandatory requirement contract fully locked.",
                "last_updated": now_iso(),
            }
            self.state.planning_confirmation_requested = True
            self._run_agent_step()

    def _run_agent_step(self) -> None:
        hops = 0
        emitted = False
        while hops < self.state.max_requirement_hops and self.state.phase == PHASE_REQUIREMENTS:
            emitted = self.single_requirement_step(self.state.active_agent)
            if emitted:
                break
            hops += 1

        # Bug fix: if the agent exhausted all hops without producing a visible reply,
        # show a safe recovery message so the conversation never goes silent.
        if not emitted and self.state.phase == PHASE_REQUIREMENTS:
            missing = self.missing_required_fields()
            if missing:
                field_label = missing[0].replace("_", " ")
                recovery = (
                    f"I still need a bit more information before we can start planning. "
                    f"Could you tell me about the **{field_label}**?"
                )
            else:
                recovery = "Everything looks good! Would you like me to start the planning phase now?"
            self.append_dialogue("assistant", recovery, self.state.active_agent)
            self.panel(self.state.active_agent, recovery, "yellow")

    def _user_wants_to_start(self, text: str) -> bool:
        """
        Smart detection of user intent to start planning.
        Uses both simple keyword matching AND LLM interpretation for reliability.
        """
        raw = str(text or "").strip().lower()

        # Direct keyword match - handles common affirmations without LLM call
        affirmations = {
            "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "start", "go",
            "start planning", "begin", "proceed", "let's go", "lets go",
            "go ahead", "do it", "let's start", "lets start", "start now",
            "yes please", "yes start", "yep start", "ready", "i'm ready",
            "im ready", "please start", "go for it", "start the planning",
        }
        if raw in affirmations:
            return True
        for aff in affirmations:
            if raw.startswith(aff) or raw.endswith(aff):
                return True

        # Fall back to LLM for ambiguous cases
        return self.wants_planning_transition(text)

    def _start_planning(self) -> None:
        self.fill_internal_defaults()
        self.state.requirement_status = {
            "ready_for_planning": True,
            "completeness_score": 1.0,
            "summary": "Mandatory requirement contract fully locked.",
            "last_updated": now_iso(),
        }
        self.state.phase = PHASE_PLANNING
        self.state.active_agent = "ArchitectAgent"
        self.state.pending_confirmations = []
        self.state.planning_confirmation_requested = False
        self.panel(
            "RequirementCoordinator",
            "Starting internal planning and validation phase now. This may take several minutes.",
            "green",
        )
        self.run_governance_cycle()

    # ----------------------------------------------------------
    # Planning cycle
    # ----------------------------------------------------------

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
                reasoner_reviews = self.run_specialist_reasoners(round_no)
                specialist_subplans = self.run_planning_specialists(round_no, reasoner_reviews)
                specialist_reviews = {"reasoner_reviews": reasoner_reviews, "specialist_subplans": specialist_subplans}

                plan = self.architect_generate(round_no, reasoner_reviews, specialist_subplans)
                audit = self.auditor_validate(round_no, plan, reasoner_reviews, specialist_subplans)

                self.state.specialist_history.append({"round": round_no, "reviews": deepcopy(specialist_reviews), "timestamp": now_iso()})
                self.state.audit_history.append(deepcopy(audit))
                self.state.current_plan = deepcopy(plan)
                self.state.current_audit = deepcopy(audit)

                write_json(Path(self.state.artifacts_dir) / f"specialists_round_{round_no}.json", specialist_reviews)
                write_json(Path(self.state.artifacts_dir) / f"plan_round_{round_no}.json", plan)
                write_json(Path(self.state.artifacts_dir) / f"audit_round_{round_no}.json", audit)

                self.update_issue_ledger(audit)
                self.state.focus_issues = self.build_focus_issues()
                self.update_revision_memory(plan, audit)
                self.update_best_artifact(plan, audit)
                self.show_round_tables(round_no, plan, audit)

                if audit.get("passed"):
                    self.state.phase = PHASE_APPROVED
                    self.console.print(Rule("[bold green]Plan Approved - Generating Comprehensive Report"))
                    self.generate_report_and_export()
                    self.panel("APPROVED", f"Validated plan approved with score {audit['score']:.2f}\n\nPDF: {self.state.final_pdf_path}", "green")
                    self.state.phase = PHASE_DEVELOPMENT
                    self.present_development_handoff()
                    return

                conv = self.detect_convergence(window=3, epsilon=0.10)
                self.state.convergence_state = conv
                if conv.get("converged"):
                    self.finish_as_best_draft("converged_without_meaningful_improvement")
                    return

                if round_no < self.state.max_planning_rounds:
                    self.panel("Revision In Progress", "Revising architecture based on cumulative audit feedback.", "yellow")

            self.finish_as_best_draft("round_limit_reached_without_approval")
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

        product = self.llm.complete_json(GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ProductReasoner"], base_payload, max_tokens=self.token_budget("analysis"), reasoning=True)
        architect = self.llm.complete_json(GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ArchitectReasoner"], {**base_payload, "product_review": product}, max_tokens=self.token_budget("analysis"), reasoning=True)
        security = self.llm.complete_json(GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["SecurityReasoner"], {**base_payload, "product_review": product, "architect_review": architect}, max_tokens=self.token_budget("analysis"), reasoning=True)
        constraints = self.llm.complete_json(GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ConstraintReasoner"], {**base_payload, "product_review": product, "architect_review": architect, "security_review": security}, max_tokens=self.token_budget("analysis"), reasoning=True)
        critic = self.llm.complete_json(GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["CriticReasoner"], {**base_payload, "product_review": product, "architect_review": architect, "security_review": security, "constraint_review": constraints}, max_tokens=self.token_budget("analysis"), reasoning=True)

        if self.state.debug_mode:
            self.thinking("Reasoners", "All 5 reasoners complete", "feed to planning specialists")

        return {"product": product, "architect_reasoner": architect, "security": security, "constraints": constraints, "critic": critic}

    def should_run_specialist(self, agent_name: str) -> bool:
        caps = set(self.inferred_capabilities())
        project_class = self.normalized_project_class()
        exposure = self.get_contract_value("external_exposure")
        sensitivity = self.get_contract_value("data_sensitivity")
        if agent_name == "SecurityAgent":
            return True
        if agent_name == "FrontendAgent":
            return ("frontend" in caps or "admin_panel" in caps or project_class in {"static_website", "landing_page", "dashboard", "web_app", "fullstack_app", "mobile_app", "desktop_app"})
        if agent_name == "BackendAgent":
            return ("backend" in caps or {"public_api", "ai_llm", "batch_jobs"} & caps or project_class in {"web_app", "fullstack_app", "api_service", "automation_tool", "data_pipeline", "ai_system"})
        if agent_name == "DataAgent":
            return ("data" in caps or self.get_contract_value("data_platform") != "" or sensitivity not in {"", "none"})
        if agent_name == "DevOpsAgent":
            return ("devops" in caps or self.get_contract_value("hosting_target") != "" or exposure in {"internal_only", "private_authenticated", "partner_facing", "public_internet"})
        return False

    def call_planning_specialist(self, agent_name: str, round_no: int, reasoner_reviews: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "round": round_no,
            "frozen_requirement_contract": self.frozen_contract(),
            "requirements": self.state.requirements,
            "reasoner_reviews": reasoner_reviews,
            "issue_ledger": self.state.issue_ledger,
            "revision_memory": self.state.revision_memory,
            "previous_audits": self.state.audit_history[-3:],
            "best_plan": self.state.best_plan,
        }
        return self.llm.complete_json(GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS[agent_name], payload, max_tokens=self.token_budget("analysis"), reasoning=True)

    def run_planning_specialists(self, round_no: int, reasoner_reviews: Dict[str, Any]) -> Dict[str, Any]:
        subplans: Dict[str, Any] = {"backend": {}, "frontend": {}, "security": {}, "data": {}, "devops": {}}
        if self.should_run_specialist("BackendAgent"):
            subplans["backend"] = self.call_planning_specialist("BackendAgent", round_no, reasoner_reviews)
        if self.should_run_specialist("FrontendAgent"):
            subplans["frontend"] = self.call_planning_specialist("FrontendAgent", round_no, reasoner_reviews)
        subplans["security"] = self.call_planning_specialist("SecurityAgent", round_no, reasoner_reviews)
        if self.should_run_specialist("DataAgent"):
            subplans["data"] = self.call_planning_specialist("DataAgent", round_no, reasoner_reviews)
        if self.should_run_specialist("DevOpsAgent"):
            subplans["devops"] = self.call_planning_specialist("DevOpsAgent", round_no, reasoner_reviews)
        return subplans

    def architect_generate(self, round_no: int, reasoner_reviews: Dict[str, Any], specialist_subplans: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "round": round_no,
            "frozen_requirement_contract": self.frozen_contract(),
            "requirements": self.state.requirements,
            "reasoner_reviews": reasoner_reviews,
            "specialist_subplans": specialist_subplans,
            "issue_ledger": self.state.issue_ledger,
            "focus_issues": self.state.focus_issues,
            "revision_memory": self.state.revision_memory,
            "accepted_exceptions": {k: asdict(v) for k, v in self.state.accepted_exceptions.items()},
            "previous_audits": self.state.audit_history[-3:],
            "previous_plan": self.state.current_plan,
            "best_plan": self.state.best_plan,
        }
        result = self.llm.complete_json(GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ArchitectAgent"], payload, max_tokens=self.token_budget("plan"), reasoning=True)
        if self.state.debug_mode:
            self.thinking("ArchitectAgent", result.get("thinking_summary", "Plan generated."), "submit to AuditorAgent")
        return self.normalize_plan(result, reasoner_reviews, specialist_subplans)

    def generic_plan_defaults(self) -> Dict[str, Any]:
        project_class = self.normalized_project_class()
        caps = set(self.inferred_capabilities())
        default_components = []
        if "frontend" in caps:
            default_components.append("Frontend application")
        if "backend" in caps:
            default_components.append("Backend service")
        if "data" in caps:
            default_components.append("Persistence layer")
        if "ai_llm" in caps:
            default_components.append("LLM integration layer")
        if "devops" in caps:
            default_components.append("Deployment and operations layer")
        if not default_components:
            default_components = ["Core application modules"]
        return {
            "title": f"Validated Architecture Plan",
            "architecture_overview": {"system_style": "Requirement-driven modular system", "primary_components": default_components},
            "technology_stack": {
                "frontend": self.get_contract_value("frontend_stack") or "Only if confirmed",
                "backend": self.get_contract_value("backend_stack") or "Only if confirmed",
                "data": self.get_contract_value("data_platform") or "Only if confirmed",
                "hosting": self.get_contract_value("hosting_target") or "Only if confirmed",
            },
            "system_components": default_components,
            "workflows": ["Execute core project flows", "Operate and monitor"],
            "data_model": {"primary_entities": ["domain_entities"], "storage_notes": "Derived from confirmed requirements."},
            "api_design": {"style": "Requirement-driven", "notes": "Only include interfaces matching the confirmed profile."},
            "deployment_and_operations": {
                "topology": "Derived from confirmed hosting requirements",
                "environments": ["dev", "staging", "production"] if "devops" in caps else ["local_dev", "release"],
            },
        }

    def merge_plan_section(self, raw_value: Any, fallback_value: Any) -> Any:
        if raw_value is None:
            return fallback_value
        if isinstance(raw_value, str) and not raw_value.strip():
            return fallback_value
        if isinstance(raw_value, list) and not raw_value:
            return fallback_value
        if isinstance(raw_value, dict) and not raw_value:
            return fallback_value
        return raw_value

    def normalize_plan(self, raw: Dict[str, Any], reasoner_reviews: Dict[str, Any], specialist_subplans: Dict[str, Any]) -> Dict[str, Any]:
        contract = self.frozen_contract()
        def c(f: str, fallback: str = "Derived from confirmed requirements.") -> str:
            item = contract.get(f, {})
            return str(item.get("value") or fallback)

        defaults = self.generic_plan_defaults()
        title = str(raw.get("title") or defaults.get("title") or "Validated Architecture Plan")
        for token in ["Round 1", "Round 2", "Round 3", "Round 4", "Round 5", "(Round 1)", "(Round 2)"]:
            title = title.replace(token, "")
        title = title.strip(" -") or "Validated Architecture Plan"

        return {
            "title": title,
            "executive_summary": raw.get("executive_summary") or f"Implementation-grade architecture for {c('project_goal')}",
            "architecture_overview": self.merge_plan_section(raw.get("architecture_overview") or raw.get("architectureoverview"), {**defaults.get("architecture_overview", {}), "primary_goal": c("project_goal"), "target_users": c("target_users")}),
            "technology_stack": self.merge_plan_section(raw.get("technology_stack") or raw.get("technologystack"), defaults.get("technology_stack", {})),
            "functional_feature_map": self.merge_plan_section(raw.get("functional_feature_map") or raw.get("functionalfeaturemap"), {"feature_scope": c("feature_scope"), "mvp_scope": c("mvp_scope"), "future_scope": c("future_scope", "Planned after MVP.")}),
            "system_components": self.merge_plan_section(raw.get("system_components") or raw.get("systemcomponents"), defaults.get("system_components", [])),
            "workflows": self.merge_plan_section(raw.get("workflows"), defaults.get("workflows", [])),
            "data_model": self.merge_plan_section(raw.get("data_model") or raw.get("datamodel"), defaults.get("data_model", {})),
            "api_design": self.merge_plan_section(raw.get("api_design") or raw.get("apidesign"), defaults.get("api_design", {})),
            "security_and_compliance": self.merge_plan_section(raw.get("security_and_compliance") or raw.get("securityandcompliance"), {"risk_level": c("risk_level"), "data_sensitivity": c("data_sensitivity"), "external_exposure": c("external_exposure"), "security_baseline": c("security_baseline"), "privacy_retention_policy": c("privacy_retention_policy", "Apply privacy controls per confirmed requirements."), "compliance_context": c("compliance_context", "Apply compliance controls if activated.")}),
            "deployment_and_operations": self.merge_plan_section(raw.get("deployment_and_operations") or raw.get("deploymentandoperations"), defaults.get("deployment_and_operations", {})),
            "observability": self.merge_plan_section(raw.get("observability"), {"baseline": c("observability_baseline", "Use per project profile.")}),
            "cost_and_scaling": self.merge_plan_section(raw.get("cost_and_scaling") or raw.get("costandscaling"), {"complexity_level": c("complexity_level", "moderate"), "execution_preference": c("execution_preference", "Prioritize maintainability.")}),
            "phased_implementation": self.merge_plan_section(raw.get("phased_implementation") or raw.get("phasedimplementation"), {"mvp_first": c("mvp_scope"), "future_later": c("future_scope", "Phase future enhancements after MVP.")}),
            "development_guidelines": self.merge_plan_section(raw.get("development_guidelines") or raw.get("developmentguidelines"), {"constraints": c("constraints", "Keep implementation practical."), "specialist_inputs_used": list(specialist_subplans.keys()), "reasoner_inputs_used": list(reasoner_reviews.keys())}),
            "risks_and_tradeoffs": self.merge_plan_section(raw.get("risks_and_tradeoffs") or raw.get("risksandtradeoffs"), {"known_risks": ["Avoid components not activated by confirmed requirements."], "tradeoffs": ["Favor requirement fit over generic templates."]}),
            "open_questions_resolved": self.merge_plan_section(raw.get("open_questions_resolved") or raw.get("openquestionsresolved"), []),
            "fix_report": ensure_list(raw.get("fix_report") or []),
            "thinking_summary": str(raw.get("thinking_summary") or ""),
            "generated_at": now_iso(),
        }

    def auditor_validate(self, round_no: int, plan: Dict[str, Any], reasoner_reviews: Dict[str, Any], specialist_subplans: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "round": round_no,
            "frozen_requirement_contract": self.frozen_contract(),
            "requirements": self.state.requirements,
            "accepted_exceptions": {k: asdict(v) for k, v in self.state.accepted_exceptions.items()},
            "issue_ledger": self.state.issue_ledger,
            "revision_memory": self.state.revision_memory,
            "previous_audits": self.state.audit_history[-3:],
            "reasoner_reviews": reasoner_reviews,
            "specialist_subplans": specialist_subplans,
            "plan": plan,
            "best_audit": self.state.best_audit,
        }
        result = self.llm.complete_json(GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["AuditorAgent"], payload, max_tokens=self.token_budget("analysis"), reasoning=True)

        strengths = ensure_list_of_str(result.get("strengths"))
        concerns = ensure_list_of_str(result.get("concerns"))
        blocking_issues = ensure_list_of_str(result.get("blocking_issues"))
        recommendations = ensure_list_of_str(result.get("recommendations"))
        issue_updates = ensure_list(result.get("issue_updates"))
        requirement_conflicts = [item for item in ensure_list(result.get("requirement_conflicts")) if isinstance(item, dict)]

        rubric = result.get("rubric_scores", {}) or {}
        req_align = self.normalize_score(rubric.get("requirements_alignment", 0))
        arch_qual = self.normalize_score(rubric.get("architecture_quality", 0))
        security = self.normalize_score(rubric.get("security", 0))
        operability = self.normalize_score(rubric.get("operability", 0))
        consistency = self.normalize_score(rubric.get("internal_consistency", 0))

        base_score = req_align * 0.30 + arch_qual * 0.25 + security * 0.20 + operability * 0.15 + consistency * 0.10
        penalty = 0.0
        unresolved_critical = False

        for item in issue_updates:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).lower()
            severity = str(item.get("severity", "")).lower()
            if status == "resolved":
                continue
            if severity == "critical":
                penalty += 1.50
                unresolved_critical = True
            elif severity == "high":
                penalty += 0.60
            elif severity == "medium":
                penalty += 0.20
            elif severity == "low":
                penalty += 0.05

        score = max(0.0, min(10.0, base_score - penalty))
        passed = score >= self.state.pass_threshold and not unresolved_critical

        prev_best = float(self.state.best_audit.get("score", 0.0)) if self.state.best_audit else 0.0
        if prev_best > 0 and score + 0.7 < prev_best:
            recommendations.append("Score regression detected; retain the stronger prior artifact.")

        return {
            "round": round_no, "score": score, "passed": passed,
            "summary": str(result.get("summary") or "Audit completed."),
            "strengths": strengths, "concerns": concerns,
            "blocking_issues": blocking_issues,
            "recommendations": unique_strs(recommendations),
            "issue_updates": issue_updates,
            "requirement_conflicts": requirement_conflicts,
            "rubric_scores": {"requirements_alignment": req_align, "architecture_quality": arch_qual, "security": security, "operability": operability, "internal_consistency": consistency},
            "base_score": round(base_score, 2), "penalty": round(penalty, 2),
            "timestamp": now_iso(), "raw": result,
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
            history.append({"round": audit.get("round"), "status": item.get("status", ""), "severity": item.get("severity", ""), "detail": item.get("detail", ""), "timestamp": now_iso()})
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
        resolved = [id for id, issue in self.state.issue_ledger.items() if str(issue.get("status", "")).lower() == "resolved"]
        unresolved = [id for id, issue in self.state.issue_ledger.items() if str(issue.get("status", "")).lower() != "resolved"]
        self.state.revision_memory = {
            "last_round": audit.get("round"), "last_score": audit.get("score"),
            "resolved_issue_ids": sorted(resolved), "unresolved_issue_ids": sorted(unresolved),
            "latest_recommendations": audit.get("recommendations", []),
            "latest_plan_title": plan.get("title"),
            "focus_issue_ids": [x.get("id") for x in self.state.focus_issues if isinstance(x, dict)],
            "convergence_state": self.state.convergence_state,
        }

    def build_focus_issues(self, limit: int = 6) -> List[Dict[str, Any]]:
        severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        items = []
        for issue_id, issue in self.state.issue_ledger.items():
            status = str(issue.get("status", "")).lower()
            if status == "resolved":
                continue
            items.append({"id": issue_id, "title": str(issue.get("title", issue_id)), "severity": str(issue.get("severity", "medium")).lower(), "status": status or "unresolved", "detail": str(issue.get("detail", "")), "last_seen_round": int(issue.get("last_seen_round", 0) or 0)})
        items.sort(key=lambda x: (severity_rank.get(x["severity"], 9), x["last_seen_round"], x["id"]))
        return items[:limit]

    def detect_convergence(self, window: int = 3, epsilon: float = 0.10) -> Dict[str, Any]:
        history = self.state.audit_history
        if len(history) < window:
            return {"converged": False, "reason": "", "detail": {}}
        recent = history[-window:]
        scores = [float(x.get("score", 0.0)) for x in recent]
        score_span = max(scores) - min(scores)
        recent_unresolved_sets = []
        for audit in recent:
            unresolved = set()
            for item in ensure_list(audit.get("issue_updates")):
                if not isinstance(item, dict):
                    continue
                if str(item.get("status", "")).lower() == "resolved":
                    continue
                if str(item.get("severity", "")).lower() in {"critical", "high", "medium"}:
                    unresolved.add(str(item.get("id") or "").strip())
            recent_unresolved_sets.append(unresolved)
        unresolved_stable = all(s == recent_unresolved_sets[0] for s in recent_unresolved_sets[1:])
        resolved_counts = [sum(1 for item in ensure_list(audit.get("issue_updates")) if isinstance(item, dict) and str(item.get("status", "")).lower() == "resolved") for audit in recent]
        no_resolution_growth = max(resolved_counts) == min(resolved_counts)
        converged = score_span <= epsilon and unresolved_stable and no_resolution_growth
        return {"converged": converged, "reason": "plateau" if converged else "", "detail": {"scores": scores, "score_span": round(score_span, 3), "unresolved_stable": unresolved_stable, "no_resolution_growth": no_resolution_growth}}

    def update_best_artifact(self, plan: Dict[str, Any], audit: Dict[str, Any]) -> None:
        candidate_score = float(audit.get("score", 0.0))
        current_best = float(self.state.best_audit.get("score", 0.0)) if self.state.best_audit else 0.0
        better = not self.state.best_plan or candidate_score > current_best or (candidate_score == current_best and len(audit.get("blocking_issues", [])) < len(self.state.best_audit.get("blocking_issues", [])))
        if better:
            self.state.best_plan = deepcopy(plan)
            self.state.best_audit = deepcopy(audit)

    def finish_as_best_draft(self, reason: str) -> None:
        best_plan = self.state.best_plan or self.state.current_plan
        best_audit = self.state.best_audit or self.state.current_audit
        if best_plan and best_audit:
            self.generate_report_and_export()
        self.state.phase = PHASE_DEVELOPMENT
        self.state.finalization_reason = reason
        message = (
            f"I've reached the strongest validated draft after {best_audit.get('round', 0)} rounds.\n\n"
            f"Best score: {float(best_audit.get('score', 0.0)):.2f}\n"
            f"PDF: {self.state.final_pdf_path or 'not exported yet'}\n\n"
            "Options:\n1. Use this validated draft as the implementation baseline.\n2. Reopen requirements to target improvements."
        )
        self.panel("Best Validated Draft", message, "cyan")
        self.present_development_handoff()

    def show_round_tables(self, round_no: int, plan: Dict[str, Any], audit: Dict[str, Any]) -> None:
        pt = Table(title=f"Architecture Draft Round {round_no}", box=box.SIMPLE_HEAVY)
        pt.add_column("Field", style="cyan", width=22)
        pt.add_column("Value")
        pt.add_row("Title", str(plan.get("title")))
        pt.add_row("Summary", as_text(plan.get("executive_summary"), 320))
        self.console.print(pt)

        at = Table(title=f"Audit Result Round {round_no}", box=box.SIMPLE_HEAVY)
        at.add_column("Metric", style="magenta", width=24)
        at.add_column("Value")
        rubric = audit.get("rubric_scores", {}) or {}
        at.add_row("Requirements", f"{float(rubric.get('requirements_alignment', 0.0)):.2f}")
        at.add_row("Architecture", f"{float(rubric.get('architecture_quality', 0.0)):.2f}")
        at.add_row("Security", f"{float(rubric.get('security', 0.0)):.2f}")
        at.add_row("Operability", f"{float(rubric.get('operability', 0.0)):.2f}")
        at.add_row("Consistency", f"{float(rubric.get('internal_consistency', 0.0)):.2f}")
        at.add_row("Final score", f"{audit['score']:.2f}")
        at.add_row("Passed", str(audit["passed"]))
        at.add_row("Summary", as_text(audit.get("summary"), 320))
        self.console.print(at)

    # ----------------------------------------------------------
    # COMPLETELY REDESIGNED REPORT GENERATION
    # ----------------------------------------------------------

    def generate_report_and_export(self) -> None:
        plan = self.state.best_plan or self.state.current_plan
        audit = self.state.best_audit or self.state.current_audit

        self.console.print(Rule("[bold cyan]Building Comprehensive Architecture Report"))

        # Phase 1: Generate specialist post-approval agents
        self.console.print("[cyan]Step 1/5: Generating execution plan...[/cyan]")
        execution = self._generate_execution_plan(plan, audit)

        self.console.print("[cyan]Step 2/5: Generating development playbook...[/cyan]")
        tutor = self._generate_tutor_guide(plan, audit, execution)

        self.console.print("[cyan]Step 3/5: Generating QA and testing package...[/cyan]")
        qa = self._generate_qa_package(plan, audit, execution)

        # Phase 2: Generate Mermaid diagrams
        self.console.print("[cyan]Step 4/5: Generating architecture diagrams...[/cyan]")
        diagrams = self._generate_diagrams(plan)

        # Phase 3: Write deep section content for each section
        self.console.print("[cyan]Step 5/5: Writing comprehensive report sections...[/cyan]")
        deep_sections = self._write_all_deep_sections(plan, audit, execution, tutor, qa)

        # Build the full report package
        report = {
            "title": plan.get("title", "Validated Architecture Plan"),
            "executive_summary": self._write_executive_summary(plan, audit),
            "plan": plan,
            "audit": audit,
            "execution": execution,
            "tutor": tutor,
            "qa": qa,
            "diagrams": diagrams,
            "deep_sections": deep_sections,
            "generated_at": now_iso(),
        }

        self.state.report_package = report
        write_json(Path(self.state.artifacts_dir) / "approved_report_package.json", report)

        # Build and export PDF
        self.console.print("[cyan]Building PDF report...[/cyan]")
        self.state.final_pdf_path = self._export_comprehensive_pdf(report, plan, audit, diagrams)
        self.console.print(f"[green]Report complete: {self.state.final_pdf_path}[/green]")

    def _generate_execution_plan(self, plan: Dict[str, Any], audit: Dict[str, Any]) -> Dict[str, Any]:
        return self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["ExecutionPlannerAgent"],
            {"plan": plan, "audit": audit, "locked_contract": self.frozen_contract(), "requirements": self.state.requirements, "specialist_history": self.state.specialist_history},
            max_tokens=6000, reasoning=True,
        )

    def _generate_tutor_guide(self, plan: Dict[str, Any], audit: Dict[str, Any], execution: Dict[str, Any]) -> Dict[str, Any]:
        return self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["TutorAgent"],
            {"plan": plan, "audit": audit, "execution": execution, "locked_contract": self.frozen_contract()},
            max_tokens=6000, reasoning=True,
        )

    def _generate_qa_package(self, plan: Dict[str, Any], audit: Dict[str, Any], execution: Dict[str, Any]) -> Dict[str, Any]:
        return self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["QAEngineerAgent"],
            {"plan": plan, "audit": audit, "execution": execution, "locked_contract": self.frozen_contract()},
            max_tokens=6000, reasoning=True,
        )

    def _generate_diagrams(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Mermaid diagrams and render them to images."""
        diagrams_dir = Path(self.state.artifacts_dir) / "diagrams"
        diagrams_dir.mkdir(exist_ok=True)

        # Generate diagram code via DiagramAgent
        mermaid_result = self.llm.complete_json(
            GLOBAL_SYSTEM + "\n" + AGENT_PROMPTS["DiagramAgent"],
            {"plan": plan, "frozen_contract": self.frozen_contract(), "requirements": self.state.requirements},
            max_tokens=4000, reasoning=True,
        )

        diagram_keys = [
            ("system_architecture", "System Architecture"),
            ("sequence_diagram", "Main User Flow Sequence"),
            ("data_model_erd", "Data Model ERD"),
            ("deployment_diagram", "Deployment Topology"),
            ("component_diagram", "Component Structure"),
            ("cicd_pipeline", "CI/CD Pipeline"),
            ("user_journey", "User Journey"),
        ]

        rendered: Dict[str, Any] = {}

        for key, title in diagram_keys:
            mermaid_code = mermaid_result.get(key, "")
            if not mermaid_code or len(mermaid_code) < 20:
                # Generate a fallback diagram
                mermaid_code = self._fallback_mermaid(key, plan)

            if mermaid_code:
                img_path = diagrams_dir / f"{key}.png"
                success = get_diagram_image(mermaid_code, img_path)
                rendered[key] = {
                    "title": title,
                    "mermaid_code": mermaid_code,
                    "image_path": str(img_path) if success else None,
                    "rendered": success is not None,
                }

        self.state.generated_diagrams = {k: v.get("mermaid_code", "") for k, v in rendered.items()}
        return rendered

    def _fallback_mermaid(self, diagram_type: str, plan: Dict[str, Any]) -> str:
        """Generate simple fallback Mermaid diagrams when AI generation fails."""
        caps = set(self.inferred_capabilities())
        project_class = self.normalized_project_class()

        if diagram_type == "system_architecture":
            nodes = []
            if "frontend" in caps:
                nodes.append("    Client[Client Browser/App]")
            if "backend" in caps:
                nodes.append("    API[API Service]")
            if "data" in caps:
                nodes.append("    DB[(Database)]")
            if "devops" in caps:
                nodes.append("    CDN[CDN/Load Balancer]")
            if "auth" in caps:
                nodes.append("    Auth[Auth Service]")
            code = "graph LR\n" + "\n".join(nodes)
            if "frontend" in caps and "backend" in caps:
                code += "\n    Client --> API"
            if "backend" in caps and "data" in caps:
                code += "\n    API --> DB"
            if "backend" in caps and "auth" in caps:
                code += "\n    API --> Auth"
            return code

        if diagram_type == "deployment_diagram":
            hosting = self.get_contract_value("hosting_target") or "Cloud Platform"
            return f"""graph TB
    Internet[Internet]
    LB[Load Balancer]
    App1[App Server 1]
    App2[App Server 2]
    DB[(Primary Database)]
    Cache[Cache Layer]
    Internet --> LB
    LB --> App1
    LB --> App2
    App1 --> DB
    App2 --> DB
    App1 --> Cache
    App2 --> Cache"""

        if diagram_type == "data_model_erd":
            return """erDiagram
    USER {
        int id PK
        string email
        string name
        datetime created_at
    }
    SESSION {
        int id PK
        int user_id FK
        string token
        datetime expires_at
    }
    AUDIT_LOG {
        int id PK
        int user_id FK
        string action
        datetime timestamp
    }
    USER ||--o{ SESSION : has
    USER ||--o{ AUDIT_LOG : generates"""

        if diagram_type == "cicd_pipeline":
            return """graph LR
    Push[Code Push] --> Test[Run Tests]
    Test --> Build[Build Artifacts]
    Build --> Staging[Deploy Staging]
    Staging --> Review[Review & Approve]
    Review --> Prod[Deploy Production]
    Prod --> Monitor[Monitor & Alert]"""

        if diagram_type == "sequence_diagram":
            return """sequenceDiagram
    actor User
    participant Frontend
    participant API
    participant Auth
    participant Database

    User->>Frontend: Request page
    Frontend->>API: API call with token
    API->>Auth: Validate token
    Auth-->>API: Token valid
    API->>Database: Query data
    Database-->>API: Return data
    API-->>Frontend: JSON response
    Frontend-->>User: Render page"""

        if diagram_type == "component_diagram":
            return f"""graph LR
    subgraph Frontend
        UI[UI Components]
        State[State Management]
        Router[Router]
    end
    subgraph Backend
        Controllers[Controllers]
        Services[Business Services]
        Models[Data Models]
        Middleware[Auth Middleware]
    end
    subgraph Data
        PrimaryDB[(Primary DB)]
        Cache[(Cache)]
        Storage[(File Storage)]
    end
    UI --> State
    State --> Router
    Router --> Controllers
    Controllers --> Middleware
    Middleware --> Services
    Services --> Models
    Models --> PrimaryDB
    Services --> Cache
    Services --> Storage"""

        if diagram_type == "user_journey":
            return """journey
    title User Journey
    section Onboarding
      Visit Landing Page: 5: User
      Sign Up: 4: User
      Email Verification: 3: User
    section Core Usage
      Login: 5: User
      Dashboard: 5: User
      Use Main Feature: 4: User
    section Advanced
      Settings: 3: User
      Export Data: 4: User"""

        return ""

    def _write_executive_summary(self, plan: Dict[str, Any], audit: Dict[str, Any]) -> str:
        """Write a comprehensive executive summary using the LLM."""
        system_prompt = """You are a senior principal architect writing the executive summary of a comprehensive architecture report.
Write a detailed, professional executive summary that covers:
1. What this system is and what it does
2. The key architectural decisions and why they were made
3. The technology stack and its rationale
4. Security and compliance posture
5. How the system will scale
6. Key risks and mitigations
7. The implementation approach and timeline
8. Expected outcomes

Write in flowing, authoritative prose. Minimum 600 words. Be specific, not generic."""

        user_content = f"""Plan title: {plan.get('title', '')}
Architecture overview: {json.dumps(plan.get('architecture_overview', {}), indent=2)[:3000]}
Technology stack: {json.dumps(plan.get('technology_stack', {}), indent=2)[:2000]}
Security: {json.dumps(plan.get('security_and_compliance', {}), indent=2)[:1500]}
Audit score: {audit.get('score', 0):.2f}
Audit summary: {audit.get('summary', '')}
Strengths: {'; '.join(audit.get('strengths', [])[:5])}"""

        return self.llm.complete_text(system_prompt, user_content, max_tokens=2000, temperature=0.3)

    def _write_all_deep_sections(self, plan: Dict[str, Any], audit: Dict[str, Any], execution: Dict[str, Any], tutor: Dict[str, Any], qa: Dict[str, Any]) -> Dict[str, str]:
        """Write deeply detailed content for every section of the report."""
        sections_to_write = [
            ("requirements_analysis", "Requirements Analysis and Interpretation", {"contract": self.frozen_contract(), "requirements": self.state.requirements}),
            ("architecture_overview", "System Architecture Overview", {"architecture": plan.get("architecture_overview"), "components": plan.get("system_components"), "feature_map": plan.get("functional_feature_map")}),
            ("technology_stack", "Technology Stack: Selection Rationale and Configuration", {"stack": plan.get("technology_stack"), "guidelines": plan.get("development_guidelines")}),
            ("component_design", "Detailed Component Design", {"components": plan.get("system_components"), "workflows": plan.get("workflows")}),
            ("data_model", "Data Architecture and Schema Design", {"data_model": plan.get("data_model")}),
            ("api_design", "API Design and Interface Contracts", {"api_design": plan.get("api_design")}),
            ("security_compliance", "Security Architecture and Compliance", {"security": plan.get("security_and_compliance")}),
            ("deployment_ops", "Deployment Architecture and Operations", {"deployment": plan.get("deployment_and_operations"), "observability": plan.get("observability")}),
            ("scalability", "Scalability, Performance, and Cost", {"cost_scaling": plan.get("cost_and_scaling")}),
            ("implementation_roadmap", "Phased Implementation Roadmap", {"execution": execution, "phases": plan.get("phased_implementation")}),
            ("dev_playbook", "Development Playbook and Engineering Standards", {"tutor": tutor, "guidelines": plan.get("development_guidelines")}),
            ("testing_strategy", "Testing Strategy and Quality Assurance", {"qa": qa}),
            ("risks_mitigations", "Risks, Trade-offs, and Mitigations", {"risks": plan.get("risks_and_tradeoffs"), "audit_concerns": audit.get("concerns")}),
            ("operational_runbook", "Operational Runbook and Incident Response", {"observability": plan.get("observability"), "deployment": plan.get("deployment_and_operations")}),
        ]

        deep_sections: Dict[str, str] = {}

        for section_key, section_title, section_data in sections_to_write:
            self.console.print(f"  [dim]Writing: {section_title}...[/dim]")
            content = self._write_deep_section(section_title, section_data, plan)
            deep_sections[section_key] = content

        return deep_sections

    def _write_deep_section(self, section_title: str, section_data: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """Write a deeply detailed individual section."""
        system_prompt = f"""You are a principal staff engineer writing a comprehensive technical architecture document.

You are writing the section titled: "{section_title}"

Write with extreme technical depth and specificity. Include:
- Detailed explanations with technical rationale
- Specific implementation guidance
- Code patterns, configuration examples where relevant
- Named technologies with version recommendations
- Specific metrics, thresholds, and success criteria
- Potential pitfalls and how to avoid them
- Dependencies and integration points

This section must be at least 800 words. Do not be generic. Be specific to the project.
Write in flowing prose with clear subheadings using markdown format (## for subheadings).
Do not use bullet points for everything - use them selectively for lists, not for explanations."""

        user_content = f"""Project title: {plan.get('title', '')}
Project goal: {self.get_contract_value('project_goal')}
Target users: {self.get_contract_value('target_users')}
Technology stack: {json.dumps(plan.get('technology_stack', {}), indent=2)[:1500]}

Section-specific data:
{json.dumps(section_data, indent=2, ensure_ascii=False)[:4000]}"""

        try:
            return self.llm.complete_text(system_prompt, user_content, max_tokens=3000, temperature=0.3)
        except Exception as e:
            return f"Section content for {section_title}:\n\n{as_text(section_data, 10000)}"

    # ----------------------------------------------------------
    # Comprehensive PDF Export
    # ----------------------------------------------------------

    def _export_comprehensive_pdf(self, report: Dict[str, Any], plan: Dict[str, Any], audit: Dict[str, Any], diagrams: Dict[str, Any]) -> str:
        out = Path(self.state.artifacts_dir) / f"validated_architecture_plan_{self.state.session_id[:8]}.pdf"

        doc = SimpleDocTemplate(
            str(out), pagesize=A4,
            rightMargin=18 * mm, leftMargin=18 * mm,
            topMargin=20 * mm, bottomMargin=18 * mm,
        )

        # --- Styles ---
        styles = getSampleStyleSheet()

        style_cover_title = ParagraphStyle("cover_title", fontSize=26, leading=32, alignment=TA_CENTER, textColor=colors.HexColor("#0F172A"), spaceAfter=8, fontName="Helvetica-Bold")
        style_cover_sub = ParagraphStyle("cover_sub", fontSize=13, leading=17, alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceAfter=6)
        style_h1 = ParagraphStyle("h1", fontSize=18, leading=22, textColor=colors.HexColor("#0B3B66"), spaceBefore=16, spaceAfter=8, fontName="Helvetica-Bold", keepWithNext=True)
        style_h2 = ParagraphStyle("h2", fontSize=13, leading=16, textColor=colors.HexColor("#1D4ED8"), spaceBefore=12, spaceAfter=5, fontName="Helvetica-Bold", keepWithNext=True)
        style_h3 = ParagraphStyle("h3", fontSize=11, leading=14, textColor=colors.HexColor("#374151"), spaceBefore=8, spaceAfter=4, fontName="Helvetica-Bold", keepWithNext=True)
        style_body = ParagraphStyle("body", fontSize=9.5, leading=14.5, alignment=TA_JUSTIFY, textColor=colors.HexColor("#1F2937"), spaceAfter=5)
        style_body_mono = ParagraphStyle("body_mono", fontSize=8.5, leading=12, textColor=colors.HexColor("#374151"), spaceAfter=4, fontName="Courier", backColor=colors.HexColor("#F8FAFC"), leftIndent=8)
        style_code_block = ParagraphStyle("code_block", fontSize=8, leading=11.5, textColor=colors.HexColor("#1E293B"), fontName="Courier", backColor=colors.HexColor("#F1F5F9"), leftIndent=10, rightIndent=10, spaceAfter=0, spaceBefore=0, borderPad=4)
        style_caption = ParagraphStyle("caption", fontSize=8, leading=10, alignment=TA_CENTER, textColor=colors.HexColor("#6B7280"), spaceAfter=8, fontName="Helvetica-Oblique")
        style_small = ParagraphStyle("small", fontSize=8.5, leading=11, textColor=colors.HexColor("#475569"), spaceAfter=3)
        style_toc = ParagraphStyle("toc", fontSize=10, leading=15, textColor=colors.HexColor("#1D4ED8"), spaceAfter=2)
        style_score_good = ParagraphStyle("score_good", fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#059669"))
        style_score_warn = ParagraphStyle("score_warn", fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#D97706"))

        story: List[Any] = []

        def hr(color: str = "#CBD5E1", thickness: float = 0.5) -> HRFlowable:
            return HRFlowable(width="100%", thickness=thickness, color=colors.HexColor(color), spaceAfter=6)

        def add_page_break() -> None:
            story.append(PageBreak())

        def add_h1(text: str) -> None:
            story.append(Paragraph(self._pdf_escape(text), style_h1))
            story.append(hr("#93C5FD", 1.0))

        def add_h2(text: str) -> None:
            story.append(Paragraph(self._pdf_escape(text), style_h2))

        def add_h3(text: str) -> None:
            story.append(Paragraph(self._pdf_escape(text), style_h3))

        def add_body(text: str) -> None:
            for para in self._split_paragraphs(text):
                if para.strip():
                    story.append(Paragraph(md_inline_to_xml(para), style_body))

        def add_bullet_list(items: List[str], indent: int = 14) -> None:
            flow = [ListItem(Paragraph(md_inline_to_xml(str(item)), style_body)) for item in items if str(item).strip()]
            if flow:
                story.append(ListFlowable(flow, bulletType="bullet", leftIndent=indent))

        def add_numbered_list(items: List[str]) -> None:
            flow = [ListItem(Paragraph(md_inline_to_xml(str(item)), style_body)) for item in items if str(item).strip()]
            if flow:
                story.append(ListFlowable(flow, bulletType="1", leftIndent=18))

        def add_json_block(data: Any, max_len: int = 8000) -> None:
            text = as_text(data, max_len)
            for line in text.splitlines()[:100]:
                story.append(Paragraph(self._pdf_escape(line), style_body_mono))

        def add_kv_table(rows: List[Tuple[str, str]], col_widths=None) -> None:
            if not rows:
                return
            if col_widths is None:
                col_widths = [55 * mm, 110 * mm]
            tbl = RLTable([[r[0], r[1]] for r in rows], colWidths=col_widths)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("BACKGROUND", (1, 0), (1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1E3A5F")),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 6))

        def add_score_table(rubric_scores: Dict[str, float]) -> None:
            rows = [["Dimension", "Score", "Weight", "Weighted"]]
            weights = {"requirements_alignment": 0.30, "architecture_quality": 0.25, "security": 0.20, "operability": 0.15, "internal_consistency": 0.10}
            labels = {"requirements_alignment": "Requirements Alignment", "architecture_quality": "Architecture Quality", "security": "Security", "operability": "Operability", "internal_consistency": "Internal Consistency"}
            total_weighted = 0.0
            for key, weight in weights.items():
                score = float(rubric_scores.get(key, 0.0))
                weighted = score * weight
                total_weighted += weighted
                color_hex = "#059669" if score >= 8.5 else "#D97706" if score >= 6.5 else "#DC2626"
                rows.append([labels.get(key, key), f"{score:.2f}/10", f"{weight:.0%}", f"{weighted:.2f}"])
            rows.append(["COMPOSITE SCORE", f"{total_weighted:.2f}/10", "100%", ""])

            tbl = RLTable(rows, colWidths=[70 * mm, 35 * mm, 25 * mm, 35 * mm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#F0FDF4")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#94A3B8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#F8FAFC")]),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 8))

        def add_diagram(diagram_info: Dict[str, Any]) -> None:
            title = diagram_info.get("title", "Diagram")
            img_path = diagram_info.get("image_path")
            mermaid_code = diagram_info.get("mermaid_code", "")

            add_h3(title)
            if img_path and Path(img_path).exists():
                try:
                    img_file = Path(img_path)
                    if img_file.stat().st_size > 500:
                        # Constrain image to fit within page safely
                        max_w = 155 * mm
                        max_h = 100 * mm
                        from reportlab.platypus import Image as RLImage
                        tmp_img = RLImage(str(img_file))
                        orig_w = tmp_img.imageWidth
                        orig_h = tmp_img.imageHeight
                        # Scale to fit within max dimensions maintaining aspect ratio
                        if orig_w > 0 and orig_h > 0:
                            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
                            draw_w = orig_w * scale
                            draw_h = orig_h * scale
                        else:
                            draw_w = max_w
                            draw_h = max_h
                        img = Image(str(img_file), width=draw_w, height=draw_h)
                        img.hAlign = "CENTER"
                        story.append(img)
                        story.append(Paragraph(f"Figure: {title}", style_caption))
                        return
                except Exception:
                    pass
            # Fall back to showing the Mermaid code
            if mermaid_code:
                story.append(Paragraph("Diagram code (Mermaid):", style_small))
                for line in mermaid_code.splitlines()[:40]:
                    story.append(Paragraph(self._pdf_escape(line), style_body_mono))
            story.append(Spacer(1, 4))

        def md_inline_to_xml(text: str) -> str:
            """Convert inline markdown to ReportLab-safe XML markup (bold, italic, code)."""
            result = []
            i = 0
            n = len(text)
            while i < n:
                # Inline code: `code`
                if text[i] == '`' and i + 1 < n:
                    j = text.find('`', i + 1)
                    if j != -1:
                        code_inner = text[i+1:j]
                        code_inner = code_inner.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        result.append(f'<font name="Courier" size="8" color="#C0392B">{code_inner}</font>')
                        i = j + 1
                        continue
                # Bold+Italic: ***text***
                if text[i:i+3] == '***':
                    j = text.find('***', i + 3)
                    if j != -1:
                        inner = md_inline_to_xml(text[i+3:j])
                        result.append(f'<b><i>{inner}</i></b>')
                        i = j + 3
                        continue
                # Bold: **text**
                if text[i:i+2] == '**':
                    j = text.find('**', i + 2)
                    if j != -1:
                        inner = md_inline_to_xml(text[i+2:j])
                        result.append(f'<b>{inner}</b>')
                        i = j + 2
                        continue
                # Bold: __text__
                if text[i:i+2] == '__':
                    j = text.find('__', i + 2)
                    if j != -1:
                        inner = md_inline_to_xml(text[i+2:j])
                        result.append(f'<b>{inner}</b>')
                        i = j + 2
                        continue
                # Italic: *text* (single asterisk, not double)
                if text[i] == '*' and (i == 0 or text[i-1] != '*') and i + 1 < n and text[i+1] != '*':
                    j = i + 1
                    while j < n and text[j] != '*':
                        j += 1
                    if j < n and (j + 1 >= n or text[j+1] != '*'):
                        inner = md_inline_to_xml(text[i+1:j])
                        result.append(f'<i>{inner}</i>')
                        i = j + 1
                        continue
                # XML-special characters
                if text[i] == '&':
                    result.append('&amp;')
                elif text[i] == '<':
                    result.append('&lt;')
                elif text[i] == '>':
                    result.append('&gt;')
                else:
                    result.append(text[i])
                i += 1
            return ''.join(result)

        def add_md_body(text: str) -> None:
            """Add a body paragraph with inline markdown rendering."""
            converted = md_inline_to_xml(text)
            story.append(Paragraph(converted, style_body))

        def add_deep_section_content(content: str) -> None:
            """Render deep section content with full markdown support:
            headings, bold, italic, inline code, code blocks, tables, bullets, numbered lists."""
            if not content:
                return

            lines = content.splitlines()
            i = 0
            para_buffer: List[str] = []

            def flush_para_buffer() -> None:
                if para_buffer:
                    combined = " ".join(para_buffer).strip()
                    if combined:
                        add_md_body(combined)
                    para_buffer.clear()

            while i < len(lines):
                raw = lines[i]
                stripped = raw.strip()

                # ── Fenced code block ───────────────────────────────────────
                if stripped.startswith("```"):
                    flush_para_buffer()
                    lang = stripped[3:].strip()
                    i += 1
                    code_lines: List[str] = []
                    while i < len(lines) and not lines[i].strip().startswith("```"):
                        code_lines.append(lines[i])
                        i += 1
                    i += 1  # skip closing ```
                    if code_lines:
                        # Trim common leading whitespace
                        min_indent = min((len(l) - len(l.lstrip()) for l in code_lines if l.strip()), default=0)
                        code_lines = [l[min_indent:] for l in code_lines]
                        story.append(Spacer(1, 4))
                        for cl in code_lines:
                            safe = cl.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                            display = safe if safe.strip() else "&nbsp;"
                            story.append(Paragraph(display, style_code_block))
                        story.append(Spacer(1, 4))
                    continue

                # ── Horizontal rule ─────────────────────────────────────────
                if re.match(r'^[-*_]{3,}$', stripped):
                    flush_para_buffer()
                    story.append(hr())
                    i += 1
                    continue

                # ── ATX Headings ────────────────────────────────────────────
                if stripped.startswith("#### "):
                    flush_para_buffer()
                    story.append(Paragraph(f"<b>{md_inline_to_xml(stripped[5:])}</b>", style_body))
                    i += 1
                    continue
                if stripped.startswith("### "):
                    flush_para_buffer()
                    add_h3(stripped[4:])
                    i += 1
                    continue
                if stripped.startswith("## "):
                    flush_para_buffer()
                    add_h2(stripped[3:])
                    i += 1
                    continue
                if stripped.startswith("# ") and not stripped.startswith("## "):
                    flush_para_buffer()
                    add_h2(stripped[2:])
                    i += 1
                    continue

                # ── Bullet list ─────────────────────────────────────────────
                if re.match(r'^[-*•]\s', stripped):
                    flush_para_buffer()
                    bullet_items: List[str] = []
                    while i < len(lines):
                        ls = lines[i].strip()
                        if re.match(r'^[-*•]\s', ls):
                            bullet_items.append(ls[2:])
                            i += 1
                        elif lines[i].startswith("  ") and bullet_items:
                            bullet_items[-1] += " " + ls
                            i += 1
                        else:
                            break
                    flow = [ListItem(Paragraph(md_inline_to_xml(item), style_body)) for item in bullet_items if item.strip()]
                    if flow:
                        story.append(ListFlowable(flow, bulletType="bullet", leftIndent=14))
                    continue

                # ── Numbered list ───────────────────────────────────────────
                if re.match(r'^\d+[.)]\s', stripped):
                    flush_para_buffer()
                    num_items: List[str] = []
                    while i < len(lines):
                        ls = lines[i].strip()
                        m = re.match(r'^\d+[.)]\s+(.*)', ls)
                        if m:
                            num_items.append(m.group(1))
                            i += 1
                        elif lines[i].startswith("   ") and num_items:
                            num_items[-1] += " " + ls
                            i += 1
                        else:
                            break
                    flow = [ListItem(Paragraph(md_inline_to_xml(item), style_body)) for item in num_items if item.strip()]
                    if flow:
                        story.append(ListFlowable(flow, bulletType="1", leftIndent=18))
                    continue

                # ── Markdown table ──────────────────────────────────────────
                if stripped.startswith("|") and "|" in stripped[1:]:
                    flush_para_buffer()
                    tbl_lines: List[str] = []
                    while i < len(lines) and lines[i].strip().startswith("|"):
                        tbl_lines.append(lines[i].strip())
                        i += 1
                    rows: List[List[str]] = []
                    for tl in tbl_lines:
                        if re.match(r'^\|[\s|:-]+\|$', tl):
                            continue  # separator row
                        cells = [c.strip() for c in tl.split("|")][1:-1]
                        if cells:
                            rows.append(cells)
                    if len(rows) >= 1:
                        max_cols = max(len(r) for r in rows)
                        rows = [r + [""] * (max_cols - len(r)) for r in rows]
                        col_w = (155 * mm) / max_cols
                        # Convert inline markdown in cells
                        pdf_rows = [[Paragraph(md_inline_to_xml(cell), style_body if ri > 0 else ParagraphStyle("th", fontSize=8.5, fontName="Helvetica-Bold", textColor=colors.white)) for cell in row] for ri, row in enumerate(rows)]
                        rt = RLTable(pdf_rows, colWidths=[col_w] * max_cols, repeatRows=1)
                        rt.setStyle(TableStyle([
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
                            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("PADDING", (0, 0), (-1, -1), 5),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                        ]))
                        story.append(rt)
                        story.append(Spacer(1, 6))
                    continue

                # ── Blank line → flush paragraph buffer ─────────────────────
                if not stripped:
                    flush_para_buffer()
                    i += 1
                    continue

                # ── Regular text → accumulate into paragraph ─────────────────
                para_buffer.append(stripped)
                i += 1

            flush_para_buffer()

        # ===================================================================
        # COVER PAGE
        # ===================================================================
        story.append(Spacer(1, 30 * mm))
        story.append(Paragraph(self._pdf_escape(report.get("title", "Validated Architecture Plan")), style_cover_title))
        story.append(Spacer(1, 6))
        story.append(Paragraph("Comprehensive Architecture & Implementation Report", style_cover_sub))
        story.append(Spacer(1, 4))
        story.append(Paragraph("Governance-Validated | Implementation-Grade | Security-Reviewed", style_cover_sub))
        story.append(Spacer(1, 16 * mm))
        story.append(hr("#3B82F6", 2.0))
        story.append(Spacer(1, 8))

        # Cover metadata table
        score = float(audit.get("score", 0.0))
        score_color = "#059669" if score >= 9.0 else "#D97706" if score >= 7.0 else "#DC2626"
        cover_rows = [
            ["Generated", now_iso()[:19].replace("T", " ") + " UTC"],
            ["Validation Score", f"{score:.2f} / 10.00"],
            ["Approval Threshold", f"{self.state.pass_threshold:.2f}"],
            ["Planning Rounds", str(audit.get("round", 0))],
            ["Project Class", self.normalized_project_class()],
            ["Risk Level", self.get_contract_value("risk_level").upper()],
            ["Data Sensitivity", self.get_contract_value("data_sensitivity").upper()],
        ]
        add_kv_table(cover_rows)
        story.append(Spacer(1, 8))
        story.append(hr("#3B82F6", 2.0))
        add_page_break()

        # ===================================================================
        # TABLE OF CONTENTS
        # ===================================================================
        add_h1("Table of Contents")
        toc_entries = [
            ("1", "Executive Summary"),
            ("2", "Validated Requirements Contract"),
            ("3", "Architecture Validation Report"),
            ("4", "Requirements Analysis and Interpretation"),
            ("5", "System Architecture Overview"),
            ("6", "Technology Stack: Selection Rationale and Configuration"),
            ("7", "Detailed Component Design"),
            ("8", "Data Architecture and Schema Design"),
            ("9", "API Design and Interface Contracts"),
            ("10", "Security Architecture and Compliance"),
            ("11", "Deployment Architecture and Operations"),
            ("12", "Scalability, Performance, and Cost"),
            ("13", "Architecture Diagrams"),
            ("14", "Phased Implementation Roadmap"),
            ("15", "Development Playbook and Engineering Standards"),
            ("16", "Testing Strategy and Quality Assurance"),
            ("17", "Risks, Trade-offs, and Mitigations"),
            ("18", "Operational Runbook and Incident Response"),
        ]
        for num, title in toc_entries:
            story.append(Paragraph(f"{num}.  {self._pdf_escape(title)}", style_toc))
        add_page_break()

        # ===================================================================
        # 1. EXECUTIVE SUMMARY
        # ===================================================================
        add_h1("1. Executive Summary")
        exec_summary = report.get("executive_summary", "")
        if exec_summary:
            add_deep_section_content(exec_summary)
        else:
            add_body(as_text(plan.get("executive_summary", ""), 10000))
        add_page_break()

        # ===================================================================
        # 2. VALIDATED REQUIREMENTS CONTRACT
        # ===================================================================
        add_h1("2. Validated Requirements Contract")
        add_body("The following requirement contract was locked before architecture planning commenced. Every architectural decision in this document traces directly to one or more confirmed fields in this contract.")
        story.append(Spacer(1, 5))

        req_header = [["Field", "Value", "Confirmed", "Source"]]
        req_rows = req_header[:]
        for k, v in self.state.requirement_contract.items():
            if v.value.strip():
                req_rows.append([
                    k.replace("_", " ").title(),
                    v.value[:200],
                    "Yes" if v.confirmed else "No",
                    v.source or "user",
                ])

        req_tbl = RLTable(req_rows, colWidths=[42 * mm, 90 * mm, 18 * mm, 15 * mm])
        req_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
            ("FONTSIZE", (0, 0), (-1, -1), 8.2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]))
        story.append(req_tbl)
        add_page_break()

        # ===================================================================
        # 3. ARCHITECTURE VALIDATION REPORT
        # ===================================================================
        add_h1("3. Architecture Validation Report")
        add_h2("3.1 Validation Score Breakdown")
        add_score_table(audit.get("rubric_scores", {}))

        add_h2("3.2 Audit Summary")
        add_body(audit.get("summary", ""))

        if audit.get("strengths"):
            add_h2("3.3 Architectural Strengths")
            add_bullet_list(audit.get("strengths", []))

        if audit.get("concerns"):
            add_h2("3.4 Identified Concerns")
            add_bullet_list(audit.get("concerns", []))

        if audit.get("recommendations"):
            add_h2("3.5 Recommendations")
            add_numbered_list(audit.get("recommendations", []))

        if self.state.issue_ledger:
            add_h2("3.6 Issue Resolution History")
            issue_rows = [["Issue ID", "Title", "Severity", "Final Status"]]
            for issue_id, issue in self.state.issue_ledger.items():
                issue_rows.append([issue_id, str(issue.get("title", ""))[:60], str(issue.get("severity", "")).upper(), str(issue.get("status", "")).upper()])
            issue_tbl = RLTable(issue_rows, colWidths=[25 * mm, 85 * mm, 25 * mm, 30 * mm])
            issue_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]))
            story.append(issue_tbl)

        add_page_break()

        # ===================================================================
        # 4-12. DEEP SECTION CONTENT
        # ===================================================================
        deep_sections = report.get("deep_sections", {})
        section_map = [
            (4, "requirements_analysis", "4. Requirements Analysis and Interpretation"),
            (5, "architecture_overview", "5. System Architecture Overview"),
            (6, "technology_stack", "6. Technology Stack: Selection Rationale and Configuration"),
            (7, "component_design", "7. Detailed Component Design"),
            (8, "data_model", "8. Data Architecture and Schema Design"),
            (9, "api_design", "9. API Design and Interface Contracts"),
            (10, "security_compliance", "10. Security Architecture and Compliance"),
            (11, "deployment_ops", "11. Deployment Architecture and Operations"),
            (12, "scalability", "12. Scalability, Performance, and Cost"),
        ]

        for section_num, section_key, section_title in section_map:
            add_h1(section_title)
            content = deep_sections.get(section_key, "")
            if content:
                add_deep_section_content(content)
            else:
                # Fall back to plan data
                plan_data_map = {
                    "requirements_analysis": {"contract": self.frozen_contract()},
                    "architecture_overview": plan.get("architecture_overview"),
                    "technology_stack": plan.get("technology_stack"),
                    "component_design": plan.get("system_components"),
                    "data_model": plan.get("data_model"),
                    "api_design": plan.get("api_design"),
                    "security_compliance": plan.get("security_and_compliance"),
                    "deployment_ops": plan.get("deployment_and_operations"),
                    "scalability": plan.get("cost_and_scaling"),
                }
                fallback = plan_data_map.get(section_key)
                if fallback:
                    add_json_block(fallback)
            add_page_break()

        # ===================================================================
        # 13. ARCHITECTURE DIAGRAMS
        # ===================================================================
        add_h1("13. Architecture Diagrams")
        add_body("The following diagrams were automatically generated from the validated architecture plan. Each diagram represents a different architectural view of the system.")
        story.append(Spacer(1, 5))

        diagram_order = [
            "system_architecture", "component_diagram", "sequence_diagram",
            "data_model_erd", "deployment_diagram", "cicd_pipeline", "user_journey",
        ]

        for diagram_key in diagram_order:
            if diagram_key in diagrams:
                add_diagram(diagrams[diagram_key])
                story.append(Spacer(1, 8))

        add_page_break()

        # ===================================================================
        # 14. IMPLEMENTATION ROADMAP
        # ===================================================================
        add_h1("14. Phased Implementation Roadmap")
        implementation_content = deep_sections.get("implementation_roadmap", "")
        if implementation_content:
            add_deep_section_content(implementation_content)
        else:
            execution = report.get("execution", {})
            if execution.get("execution_overview"):
                add_h2("Execution Overview")
                add_body(as_text(execution.get("execution_overview"), 5000))

            phases = ensure_list(execution.get("implementation_phases"))
            if phases:
                add_h2("Implementation Phases")
                for idx, phase in enumerate(phases, start=1):
                    if isinstance(phase, dict):
                        phase_name = str(phase.get("phase_name") or phase.get("name") or phase.get("phase") or f"Phase {idx}")
                        add_h3(f"Phase {idx}: {phase_name}")
                        if phase.get("duration_estimate"):
                            story.append(Paragraph(f"Duration: {self._pdf_escape(str(phase.get('duration_estimate')))}", style_small))
                        if phase.get("objectives"):
                            add_h3("Objectives")
                            add_bullet_list(ensure_list_of_str(phase.get("objectives", [])))
                        if phase.get("deliverables"):
                            add_h3("Deliverables")
                            add_bullet_list(ensure_list_of_str(phase.get("deliverables", [])))
                        if phase.get("tasks"):
                            add_h3("Tasks")
                            add_bullet_list(ensure_list_of_str(phase.get("tasks", [])))
                        if phase.get("done_criteria"):
                            add_h3("Done Criteria")
                            add_bullet_list(ensure_list_of_str(phase.get("done_criteria", [])))
                        story.append(Spacer(1, 5))

            milestones = ensure_list(execution.get("milestone_checks"))
            if milestones:
                add_h2("Milestones and Go/No-Go Checks")
                milestone_rows = [["Milestone", "Criteria", "Verification Method"]]
                for m in milestones:
                    if isinstance(m, dict):
                        milestone_rows.append([str(m.get("milestone", ""))[:40], str(m.get("criteria", ""))[:80], str(m.get("verification_method", ""))[:40]])
                if len(milestone_rows) > 1:
                    m_tbl = RLTable(milestone_rows, colWidths=[45 * mm, 90 * mm, 30 * mm])
                    m_tbl.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]))
                    story.append(m_tbl)

        add_page_break()

        # ===================================================================
        # 15. DEVELOPMENT PLAYBOOK
        # ===================================================================
        add_h1("15. Development Playbook and Engineering Standards")
        playbook_content = deep_sections.get("dev_playbook", "")
        if playbook_content:
            add_deep_section_content(playbook_content)
        else:
            tutor = report.get("tutor", {})
            for key, title in [("development_playbook", "Development Playbook"), ("coding_order", "Coding Order"), ("implementation_tips", "Implementation Tips"), ("environment_setup_guide", "Environment Setup"), ("branching_strategy", "Branching Strategy"), ("code_review_checklist", "Code Review Checklist"), ("common_mistakes", "Common Mistakes to Avoid"), ("feature_build_guides", "Feature Build Guides")]:
                value = tutor.get(key)
                if not value:
                    continue
                add_h2(title)
                if isinstance(value, list):
                    add_bullet_list([as_text(v, 3000) for v in value])
                else:
                    add_body(as_text(value, 10000))

        add_page_break()

        # ===================================================================
        # 16. TESTING STRATEGY
        # ===================================================================
        add_h1("16. Testing Strategy and Quality Assurance")
        testing_content = deep_sections.get("testing_strategy", "")
        if testing_content:
            add_deep_section_content(testing_content)
        else:
            qa_data = report.get("qa", {})
            if qa_data.get("validation_strategy"):
                add_h2("Validation Strategy")
                add_body(as_text(qa_data.get("validation_strategy"), 5000))

            test_layers = qa_data.get("test_layers", {})
            if isinstance(test_layers, dict) and test_layers:
                add_h2("Test Layers")
                for layer_name, layer_data in test_layers.items():
                    add_h3(layer_name.replace("_", " ").title())
                    add_body(as_text(layer_data, 3000))

            test_plan = ensure_list(qa_data.get("detailed_test_plan"))
            if test_plan:
                add_h2("Detailed Test Plan")
                for suite in test_plan:
                    if isinstance(suite, dict):
                        suite_name = str(suite.get("suite_name") or suite.get("name") or "Test Suite")
                        add_h3(suite_name)
                        cases = ensure_list(suite.get("test_cases"))
                        if cases:
                            case_rows = [["Test ID", "Description", "Expected Result", "Severity"]]
                            for case in cases[:30]:
                                if isinstance(case, dict):
                                    case_rows.append([str(case.get("id", ""))[:15], str(case.get("description", ""))[:70], str(case.get("expected_result", ""))[:60], str(case.get("severity", "medium"))])
                            if len(case_rows) > 1:
                                c_tbl = RLTable(case_rows, colWidths=[20 * mm, 70 * mm, 60 * mm, 15 * mm])
                                c_tbl.setStyle(TableStyle([
                                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")),
                                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94A3B8")),
                                    ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
                                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                    ("PADDING", (0, 0), (-1, -1), 3),
                                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                                ]))
                                story.append(c_tbl)
                                story.append(Spacer(1, 5))

            checklist = qa_data.get("release_readiness_checklist")
            if checklist:
                add_h2("Release Readiness Checklist")
                if isinstance(checklist, dict):
                    for category, items in checklist.items():
                        add_h3(category.replace("_", " ").title())
                        add_bullet_list(ensure_list_of_str(items))
                else:
                    add_bullet_list(ensure_list_of_str(checklist))

        add_page_break()

        # ===================================================================
        # 17. RISKS AND TRADE-OFFS
        # ===================================================================
        add_h1("17. Risks, Trade-offs, and Mitigations")
        risks_content = deep_sections.get("risks_mitigations", "")
        if risks_content:
            add_deep_section_content(risks_content)
        else:
            risks_data = plan.get("risks_and_tradeoffs", {})
            if risks_data:
                add_json_block(risks_data)
            if audit.get("concerns"):
                add_h2("Auditor-Identified Concerns")
                add_bullet_list(audit.get("concerns", []))

        add_page_break()

        # ===================================================================
        # 18. OPERATIONAL RUNBOOK
        # ===================================================================
        add_h1("18. Operational Runbook and Incident Response")
        ops_content = deep_sections.get("operational_runbook", "")
        if ops_content:
            add_deep_section_content(ops_content)
        else:
            observability = plan.get("observability", {})
            if observability:
                add_h2("Observability and Monitoring")
                add_json_block(observability, 5000)
            deployment = plan.get("deployment_and_operations", {})
            if deployment:
                add_h2("Operations Reference")
                add_json_block(deployment, 5000)

        # ===================================================================
        # BUILD PDF
        # ===================================================================
        doc.build(story)
        return str(out.resolve())

    # ----------------------------------------------------------
    # PDF helper methods
    # ----------------------------------------------------------

    def _pdf_escape(self, text: str) -> str:
        return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")

    def _split_paragraphs(self, text: str) -> List[str]:
        text = (text or "").replace("\r", "").strip()
        if not text:
            return []
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return parts if parts else [text]

    def contract_summary_text(self) -> str:
        lines = ["Confirmed requirement contract:"]
        for k, v in self.state.requirement_contract.items():
            if v.value.strip():
                suffix = "confirmed" if v.confirmed else "pending"
                lines.append(f"- {k}: {v.value} ({suffix})")
        return "\n".join(lines)

    def present_development_handoff(self) -> None:
        dev = self.state.development_package or {}
        body = "The project now moves into the development phase.\n\nA comprehensive implementation guide, QA strategy, execution roadmap, and architecture diagrams have been included in the approved PDF report."
        self.panel("Development Phase", body, "cyan")


if __name__ == "__main__":
    app = GovernanceHybridApp()
    app.run()