"""
Architect Training Dataset Generator — Async Concurrent Edition
================================================================
Speed vs. original: ~20-40x faster via:
  - asyncio + AsyncAzureOpenAI (true async, no thread blocking)
  - MAX_CONCURRENT parallel API calls at all times
  - Larger BATCH_SIZE (5 rows per request, fewer round-trips)
  - Lock-free atomic file writes
  - Schema validation before writing (no silent corruption)
  - Automatic resume from existing output file
  - Rich live progress dashboard

Usage:
    pip install openai tenacity rich python-dotenv aiofiles
    python generate_dataset.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import aiofiles
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

load_dotenv()

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — tune these for your rate limits
# ══════════════════════════════════════════════════════════════

AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY          = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION      = "2025-01-01-preview"
DEPLOYMENT_NAME  = "gpt-5-chat"

TOTAL_ROWS       = 1000      # Target row count
BATCH_SIZE       = 5         # Rows per API call  (5 = fewer calls, richer context)
MAX_CONCURRENT   = 12        # Parallel in-flight API calls (tune to your TPM limit)
MAX_TOKENS       = 16384     # Per call — 5 rows need more room than 3
TEMPERATURE      = 0.75
OUTPUT_FILE      = Path("architect_training_dataset.jsonl")
ERRORS_FILE      = Path("architect_training_errors.jsonl")

# Retry settings
MAX_RETRIES      = 5
RETRY_BASE_DELAY = 2.0       # seconds, doubles on each retry

# ══════════════════════════════════════════════════════════════
# VALIDATION — fast schema check before any row hits disk
# ══════════════════════════════════════════════════════════════

REQUIRED_TOP_KEYS = {
    "sample_id", "dataset", "agent", "split",
    "profile", "input_payload", "target_output", "metadata",
}
REQUIRED_CONTRACT_FIELDS = {
    "project_goal", "target_users", "project_class", "capabilities",
    "complexity_level", "risk_level", "data_sensitivity", "external_exposure",
    "access_model", "feature_scope", "mvp_scope", "security_baseline",
    "frontend_stack", "backend_stack", "data_platform", "hosting_target",
    "privacy_retention_policy", "future_scope", "constraints",
    "observability_baseline", "execution_preference",
}
REQUIRED_OUTPUT_KEYS = {
    "title", "executive_summary", "architecture_overview",
    "technology_stack", "data_model", "workflows", "api_design",
    "security_and_compliance", "deployment_and_operations",
    "phased_implementation", "risks_and_tradeoffs",
}
VALID_PLAN_QUALITY   = {"strong", "moderate", "weak", "flawed"}
VALID_CASE_TYPE      = {"first_pass", "revision_round"}
FORBIDDEN_ENUMS      = {"phi", "pci", "pii", "secret",
                        "public_internet", "partner_api", "private_authenticated"}

def _unwrap(obj: dict, key: str) -> dict:
    """Unwrap a value that the model may have double-encoded as a JSON string."""
    val = obj.get(key, {})
    if isinstance(val, str):
        try:
            val = json.loads(val)
            obj[key] = val
        except Exception:
            return {}
    return val if isinstance(val, dict) else {}


def validate_row(obj: dict) -> list[str]:
    """Returns a list of validation error strings (empty = valid)."""
    errors: list[str] = []

    # Top-level keys
    missing = REQUIRED_TOP_KEYS - obj.keys()
    if missing:
        errors.append(f"Missing top-level keys: {missing}")
        return errors  # can't proceed without these

    # Unwrap any double-encoded nested dicts the model may produce
    input_payload = _unwrap(obj, "input_payload")
    contract = _unwrap(input_payload, "frozen_requirement_contract")

    missing_contract = REQUIRED_CONTRACT_FIELDS - contract.keys()
    if missing_contract:
        errors.append(f"Missing contract fields: {missing_contract}")

    # Every contract field must be an object with 'value'
    for field, val in contract.items():
        if not isinstance(val, dict) or "value" not in val:
            errors.append(f"Contract field '{field}' is not a proper object with 'value'")

    # Output sections — unwrap if double-encoded
    output = _unwrap(obj, "target_output")
    if not output and obj.get("target_output") is not None:
        errors.append(f"target_output is not a dict (got {type(obj.get('target_output')).__name__})")
        return errors
    missing_output = REQUIRED_OUTPUT_KEYS - output.keys()
    if missing_output:
        errors.append(f"Missing target_output sections: {missing_output}")

    # Minimum richness (word count)
    min_words = {
        "architecture_overview": 50,
        "workflows": 40,
        "data_model": 40,
        "deployment_and_operations": 50,
        "risks_and_tradeoffs": 30,
    }
    for section, min_wc in min_words.items():
        text = output.get(section, "")
        if isinstance(text, str):
            wc = len(text.split())
            if wc < min_wc:
                errors.append(f"'{section}' too short: {wc} words (need {min_wc})")

    # Forbidden enum values — coerce to str so lists/None never cause TypeError
    profile = obj.get("profile", {})
    if not isinstance(profile, dict):
        profile = {}
    ds = str(profile.get("datasensitivity", ""))
    ee = str(profile.get("externalexposure", ""))
    if ds in FORBIDDEN_ENUMS:
        errors.append(f"Forbidden datasensitivity value: '{ds}'")
    if ee in FORBIDDEN_ENUMS:
        errors.append(f"Forbidden externalexposure value: '{ee}'")

    # metadata — same coercion guard
    meta = obj.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    pq = str(meta.get("plan_quality", ""))
    ct = str(meta.get("case_type", ""))
    if pq not in VALID_PLAN_QUALITY:
        errors.append(f"Invalid plan_quality: '{pq}'")
    if ct not in VALID_CASE_TYPE:
        errors.append(f"Invalid case_type: '{ct}'")

    return errors


# ══════════════════════════════════════════════════════════════
# PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert synthetic data generator for enterprise architecture systems. "
    "Return ONLY valid JSONL. No markdown, no fences, no commentary."
)

BASE_PROMPT = """\
You are generating a premium synthetic training dataset for an internal architectural governance runtime.
I have generated {current_count} rows so far. Generate the next {batch_size} rows starting from {start_id}.

Return ONLY valid JSONL (JSON Lines format).
Each line must be one complete JSON object.
Do NOT wrap output in a JSON array.
Do NOT use markdown or code fences.
Do NOT include comments or explanations.

═══════════════════════════════════════════════════════════════
CRITICAL JSON ESCAPING RULE
═══════════════════════════════════════════════════════════════
Because you are generating JSON, you MUST NOT use unescaped double quotes (") inside your text strings. 
If you need to write dialogue, examples, or nested JSON payloads inside a plan section, STRICTLY use SINGLE QUOTES (').

BAD (Breaks JSON):
"workflows": "The user asks "What is my balance?" and the bot replies "You have $5"."

GOOD (Valid JSON):
"workflows": "The user asks 'What is my balance?' and the bot replies 'You have $5'."

CORRECT OUTPUT FORMAT (3 separate lines, no array brackets):
{"sample_id":"ARC-PLAN26-001","dataset":"architect",...}
{"sample_id":"ARC-PLAN26-002","dataset":"architect",...}
{"sample_id":"ARC-PLAN26-003","dataset":"architect",...}

WRONG OUTPUT FORMAT (do NOT do this):
[
  {"sample_id":"ARC-PLAN26-001",...},
  {"sample_id":"ARC-PLAN26-002",...}
]

TASK
Generate EXACTLY 3 rows as JSONL (one JSON object per line).

DATASET PURPOSE
These rows train the ArchitectAgent of a governance system that generates implementation-grade architecture plans from locked requirement contracts, specialist sub-plans, reasoner reviews, issue history, and revision memory.

═══════════════════════════════════════════════════════════════
ANTI-BOILERPLATE RULES (MOST IMPORTANT SECTION)
═══════════════════════════════════════════════════════════════

Every plan section MUST be UNIQUE to the specific project domain, technology stack, and requirements described in the frozen_requirement_contract. Generic text that could apply to any project is FORBIDDEN.

The #1 quality metric is: could you read a plan section and correctly guess what project it belongs to WITHOUT seeing the title? If the answer is no, the section is generic boilerplate and must be rewritten.

FORBIDDEN PATTERNS (never write these):
- "The target system utilizes a robust and highly scalable..."
- "...connecting various isolated components seamlessly"
- "Users have a standard one to many relationship with Orders"
- "Phase one rapidly delivers the minimum viable product focusing heavily on essential core functionality"
- "All active software engineers must strictly adhere to comprehensive code linting rules"
- "System autoscaling policies dynamically adjust active compute resources based strictly on CPU utilization metrics"
- "Relying heavily on a single primary cloud vendor introduces long term architectural lock in risks"
- Any sentence that uses filler words like "robust", "comprehensive", "seamlessly", "effectively", "strictly" without specific technical content

DOMAIN-SPECIFIC EXAMPLES — WHAT GOOD vs BAD LOOKS LIKE:

For a HEALTHCARE EMR system:

BAD data_model (generic boilerplate):
"The primary relational data entities include Users, Orders, and Products."

GOOD data_model (domain-specific):
"patients (id, mrn, name_encrypted, dob, blood_type, allergies_json, primary_physician_id, created_at), encounters (id, patient_id, encounter_type ENUM consultation/emergency/followup, chief_complaint, vitals_json, diagnosis_codes ICD10[], attending_id, encounter_date), prescriptions (id, encounter_id, medication_ndc, dosage, frequency, duration_days, prescriber_id, pharmacy_id, status ENUM pending/filled/cancelled), lab_results (id, encounter_id, test_code_loinc, result_value, unit, reference_range, abnormal_flag, collected_at, reported_at)."

BAD architecture_overview (generic):
"The target system utilizes a robust architecture connecting various isolated components seamlessly."

GOOD architecture_overview (domain-specific):
"HL7 FHIR-compliant architecture with a Vue 3 clinical dashboard communicating via REST to a FastAPI backend on GCP Cloud Run. Patient data flows through an HL7 integration engine for interoperability with existing hospital EHR systems. All PHI is encrypted at rest via Cloud KMS and in transit via TLS 1.3. A dedicated audit service captures every patient record access event for HIPAA compliance."

For a FINTECH PAYMENTS system:

BAD workflows (generic):
"The secure user registration workflow involves collecting user input, validating data integrity, hashing passwords securely."

GOOD workflows (domain-specific):
"Payment Initiation: Merchant sends POST /v1/payments with amount, currency, and card token. Gateway validates idempotency key against Redis, performs fraud score check via ML service, routes to optimal processor (Stripe or Adyen) based on card BIN and geography. Processor returns auth code. Settlement: Nightly batch job reconciles authorized transactions against processor settlement files, computes net amounts after interchange fees, and triggers payout transfers to merchant bank accounts via ACH."

For an IoT DATA PIPELINE:

BAD system_components (generic):
"The core components include a scalable API Gateway, an Identity Provider, a core data processing engine."

GOOD system_components (domain-specific):
"MQTT Broker (EMQX cluster handling 50K concurrent device connections), Kafka Cluster (3-broker MSK with 64 partitions for sensor-data topic), Stream Processor (Flink jobs for windowed aggregation — 5-min tumbling windows for soil moisture averaging), Time-Series DB (InfluxDB for hot data, last 30 days), Cold Storage (S3 with Apache Iceberg tables for historical analysis), Alert Engine (custom Go service evaluating threshold rules against sliding windows)."

RULE: Every row's plan sections must contain terminology, entities, workflows, and components SPECIFIC to that row's domain. A healthcare plan must mention patients, encounters, diagnoses, HL7, HIPAA. A fintech plan must mention transactions, settlements, card tokens, PCI. An IoT plan must mention sensors, telemetry, MQTT, time-series. A CLI tool must mention commands, flags, parsers, stdout.

═══════════════════════════════════════════════════════════════
CROSS-ROW UNIQUENESS RULES
═══════════════════════════════════════════════════════════════

No two rows in any batch may share:
- The same architecture_overview wording
- The same data_model entities
- The same workflow descriptions
- The same phased_implementation text
- The same risks_and_tradeoffs text
- The same development_guidelines text

If you find yourself writing similar sentences across rows, STOP and rewrite with project-specific content.

═══════════════════════════════════════════════════════════════
TOP-LEVEL SHAPE
═══════════════════════════════════════════════════════════════

Each JSONL line must be:
{"sample_id":"ARC-PLAN26-001","dataset":"architect","agent":"ArchitectAgent","split":"train","profile":{...},"input_payload":{...},"target_output":{...},"metadata":{...}}

ALLOWED ENUMS

profile.projectclass:
webapp, fullstackapp, mobileapp, desktopapp, apiservice, staticwebsite, landingpage, clitool, librarysdk, automationtool, datapipeline, aisystem, researchprototype, infrastructureproject

capabilities labels:
frontend, backend, data, auth, aillm, integrations, analytics, realtime, payments, adminpanel, publicapi, batchjobs, devops

complexitylevel: simple, moderate, advanced, highscale
risklevel: low, medium, high
datasensitivity: none, internal, personal, financial, health, confidential
externalexposure: localonly, internalonly, privateauthenticated, partnerfacing, publicinternet

Issue severity: critical, high, medium, low
Issue status: unresolved, resolved, downgraded, new

STRICT ENUM RULES
Never use: phi, pci, pii, secret, public_internet, partner_api, private_authenticated
For healthcare: datasensitivity="health"
For PCI/card: datasensitivity="financial"
For B2B partner APIs: externalexposure="partnerfacing"

═══════════════════════════════════════════════════════════════
CRITICAL: CONTRACT FIELD OBJECT SHAPE
═══════════════════════════════════════════════════════════════

EVERY field inside frozen_requirement_contract must be an OBJECT with exactly these 5 keys:
{"value":"<string>","source":"<short label>","confirmed":true,"rationale":"<brief reason>","updated_at":"2026-03-15T00:00:00Z"}

NEVER write a contract field as a plain string.
WRONG: "target_users": "Clinic staff"
RIGHT: "target_users": {"value":"Clinic staff and patients","source":"PRD","confirmed":true,"rationale":"Primary users from discovery","updated_at":"2026-03-15T00:00:00Z"}

═══════════════════════════════════════════════════════════════
MANDATORY CONTRACT FIELDS (all 21+ must be present as objects)
═══════════════════════════════════════════════════════════════

1. project_goal
2. target_users
3. project_class
4. capabilities
5. complexity_level
6. risk_level
7. data_sensitivity
8. external_exposure
9. access_model
10. feature_scope
11. mvp_scope
12. security_baseline
13. frontend_stack
14. backend_stack
15. data_platform
16. hosting_target
17. privacy_retention_policy
18. future_scope
19. constraints
20. observability_baseline
21. execution_preference

Conditional:
22. llm_integration — required when capabilities includes "aillm"
23. compliance_context — required when risk_level="high" OR data_sensitivity is personal/financial/health/confidential OR capabilities includes "payments"

PROFILE SHAPE
"profile":{"domain":"<realistic domain>","projectclass":"<enum>","capabilities":["<enum>",...],"complexitylevel":"<enum>","risklevel":"<enum>","datasensitivity":"<enum>","externalexposure":"<enum>"}

INPUT PAYLOAD SHAPE
"input_payload":{"round":<int>,"frozen_requirement_contract":{...},"requirements":{...},"reasoner_reviews":{...},"specialist_subplans":{...},"issue_ledger":{...},"focus_issues":[...],"revision_memory":{...},"accepted_exceptions":{...},"previous_audits":[...],"previous_plan":{...},"best_plan":{...}}

REQUIREMENTS SHAPE (dict of dicts, not arrays)
"requirements":{"project":{"goal":"...","target_users":"...","timeline":"..."},"frontend":{"framework":"...","approach":"..."},"backend":{"runtime":"...","framework":"..."},"security":{"auth":"...","transport":"..."},"data":{"db":"...","cache":"..."},"devops":{"ci":"...","hosting":"..."},"constraints":{...},"open_questions":{...},"confirmed_decisions":{...}}

REASONER REVIEWS: 3-5 keys, each 2-5 sentences of domain-specific analysis. Keys: product, architect_reasoner, security, constraints, critic

SPECIALIST SUBPLANS: 2-5 keys, each a dict with 2-4 domain-specific keys. Keys: backend, frontend, security, data, devops

ISSUE LEDGER: Round 1 = {}. Round > 1 = real issues keyed by ID.
FOCUS ISSUES: Round 1 = []. Round > 1 = 1-3 priority issues.
REVISION MEMORY: Round 1 = {}. Round > 1 = {"round_1":"...","changes_for_round_2":"..."}
PREVIOUS AUDITS: Round 1 = []. Round > 1 = 1-3 summary strings.
PREVIOUS PLAN / BEST PLAN: Round 1 = {}. Round > 1 = {"title":"...","summary":"..."}

═══════════════════════════════════════════════════════════════
TARGET OUTPUT SHAPE
═══════════════════════════════════════════════════════════════

"target_output":{"thinking_summary":"...","fix_report":[...],"title":"...","executive_summary":"...","architecture_overview":"...","technology_stack":"...","functional_feature_map":"...","system_components":"...","workflows":"...","data_model":"...","api_design":"...","security_and_compliance":"...","deployment_and_operations":"...","observability":"...","cost_and_scaling":"...","phased_implementation":"...","development_guidelines":"...","risks_and_tradeoffs":"...","open_questions_resolved":"..."}

FIX REPORT: Round 1 = []. Round > 1 = [{"issue_id":"...","action_taken":"...","changed_sections":[...],"expected_outcome":"..."}]

PLAN SECTION MINIMUM RICHNESS (word counts):
- architecture_overview: 50+ words
- technology_stack: 40+ words
- functional_feature_map: 35+ words
- system_components: 30+ words
- workflows: 40+ words with at least 2 distinct domain workflows
- data_model: 40+ words with named entities specific to the domain
- api_design: 30+ words with actual endpoint examples
- security_and_compliance: 35+ words with named controls
- deployment_and_operations: 50+ words with topology and CI/CD specifics
- observability: 30+ words with named tools and domain metrics
- cost_and_scaling: 25+ words
- phased_implementation: 35+ words with domain-specific phase content
- development_guidelines: 25+ words
- risks_and_tradeoffs: 30+ words with domain-specific risks
- open_questions_resolved: 20+ words

PLAN QUALITY VARIATION (across the full dataset):
- 40% STRONG: All sections rich and domain-specific
- 35% MODERATE: Good but 1-2 sections slightly thin
- 15% WEAK: Several thin sections, less detail
- 10% INTENTIONALLY FLAWED: Contains contract violations

FIRST-PASS / REVISION BALANCE:
- 40% first_pass (round = 1, fix_report = [], issue_ledger = {}, focus_issues = [])
- 60% revision_round (round > 1, fix_report addresses focus_issues)

PLAN CONSISTENCY RULES:
- staticwebsite/landingpage: no server-side runtime on static hosting
- clitool: CLI architecture (commands, parsers, flags), not browser UI
- librarysdk: package architecture (exports, API surface), not hosted platform
- infrastructureproject: platform/environment architecture (Terraform, networking)
- aisystem with local-model constraint: no public model APIs
- partnerfacing with mandatory mTLS: include mTLS in security sections
- Plan content MUST match the contract's technology choices

METADATA SHAPE
"metadata":{"schema_version":"v2_architect_aligned","case_type":"first_pass"|"revision_round","plan_quality":"strong"|"moderate"|"weak"|"flawed","primary_theme":"<label>","generation_source":"synthetic","quality_flags":[],"notes":"<brief note>"}

primary_theme: greenfield, migration, scalability, security_hardening, compliance, performance_optimization, cost_reduction, microservices_decomposition, api_first, data_intensive, realtime, ml_pipeline, devops_modernization, legacy_modernization

PROJECT-CLASS COVERAGE: Spread across all 14 classes. Each batch of 3 should use 3 different classes.

DIVERSITY: Vary domains, tech stacks, clouds, databases, architectures. Every row must describe a genuinely different project.

═══════════════════════════════════════════════════════════════
COMPLETE EXAMPLE ROW (follow this format exactly)
═══════════════════════════════════════════════════════════════

{"sample_id":"ARC-PLAN26-EXAMPLE","dataset":"architect","agent":"ArchitectAgent","split":"train","profile":{"domain":"Healthcare Operations","projectclass":"fullstackapp","capabilities":["frontend","backend","data","auth"],"complexitylevel":"moderate","risklevel":"high","datasensitivity":"health","externalexposure":"privateauthenticated"},"input_payload":{"round":1,"frozen_requirement_contract":{"project_goal":{"value":"Patient intake management system for clinics","source":"PRD","confirmed":true,"rationale":"Core business need replacing paper forms","updated_at":"2026-03-15T00:00:00Z"},"target_users":{"value":"Clinic staff and patients","source":"PRD","confirmed":true,"rationale":"Primary end users identified in discovery","updated_at":"2026-03-15T00:00:00Z"},"project_class":{"value":"fullstackapp","source":"Architect","confirmed":true,"rationale":"Requires both UI and backend API","updated_at":"2026-03-15T00:00:00Z"},"capabilities":{"value":"frontend, backend, data, auth","source":"Architect","confirmed":true,"rationale":"Standard fullstack capabilities","updated_at":"2026-03-15T00:00:00Z"},"complexity_level":{"value":"moderate","source":"Architect","confirmed":true,"rationale":"Standard CRUD with HIPAA overlay","updated_at":"2026-03-15T00:00:00Z"},"risk_level":{"value":"high","source":"Security","confirmed":true,"rationale":"Handles protected health information","updated_at":"2026-03-15T00:00:00Z"},"data_sensitivity":{"value":"health","source":"Compliance","confirmed":true,"rationale":"Stores PHI including medical history","updated_at":"2026-03-15T00:00:00Z"},"external_exposure":{"value":"privateauthenticated","source":"Security","confirmed":true,"rationale":"Login required for all access","updated_at":"2026-03-15T00:00:00Z"},"access_model":{"value":"RBAC with MFA for all roles","source":"Security","confirmed":true,"rationale":"HIPAA mandates strong authentication","updated_at":"2026-03-15T00:00:00Z"},"feature_scope":{"value":"Patient registration, intake forms, medical history, staff review dashboard","source":"PRD","confirmed":true,"rationale":"Core workflow features","updated_at":"2026-03-15T00:00:00Z"},"mvp_scope":{"value":"Basic intake forms with staff review and patient self-service","source":"Product","confirmed":true,"rationale":"Minimum viable for Q3 pilot","updated_at":"2026-03-15T00:00:00Z"},"security_baseline":{"value":"Encryption at rest AES-256, TLS 1.3 in transit, MFA required","source":"Security","confirmed":true,"rationale":"HIPAA minimum controls","updated_at":"2026-03-15T00:00:00Z"},"frontend_stack":{"value":"React 18 with TypeScript","source":"Engineering","confirmed":true,"rationale":"Team expertise and ecosystem","updated_at":"2026-03-15T00:00:00Z"},"backend_stack":{"value":"Node 20 with NestJS","source":"Engineering","confirmed":true,"rationale":"TypeScript end-to-end","updated_at":"2026-03-15T00:00:00Z"},"data_platform":{"value":"Aurora PostgreSQL 15","source":"Engineering","confirmed":true,"rationale":"Relational data with managed HA","updated_at":"2026-03-15T00:00:00Z"},"hosting_target":{"value":"AWS ECS Fargate in us-east-1","source":"DevOps","confirmed":true,"rationale":"Managed containers with data residency","updated_at":"2026-03-15T00:00:00Z"},"privacy_retention_policy":{"value":"7-year retention for medical records, 90-day log retention","source":"Legal","confirmed":true,"rationale":"State healthcare data retention law","updated_at":"2026-03-15T00:00:00Z"},"future_scope":{"value":"Telehealth integration and patient portal mobile app","source":"Product","confirmed":true,"rationale":"Year 2 roadmap items","updated_at":"2026-03-15T00:00:00Z"},"constraints":{"value":"Must deploy to us-east-1 only, budget under $1500/month","source":"Ops","confirmed":true,"rationale":"Data residency and budget limits","updated_at":"2026-03-15T00:00:00Z"},"observability_baseline":{"value":"Datadog APM, structured JSON logging, PagerDuty alerts","source":"Ops","confirmed":true,"rationale":"Corporate monitoring standard","updated_at":"2026-03-15T00:00:00Z"},"execution_preference":{"value":"Agile sprints with 2-week iterations","source":"Engineering","confirmed":true,"rationale":"Team workflow preference","updated_at":"2026-03-15T00:00:00Z"},"compliance_context":{"value":"HIPAA and HITECH compliance required, BAA with AWS","source":"Legal","confirmed":true,"rationale":"Healthcare data regulations","updated_at":"2026-03-15T00:00:00Z"}},"requirements":{"project":{"goal":"Manage patient intake forms and medical history","target_users":"Clinic staff and patients","timeline":"Q3 delivery"},"frontend":{"framework":"React 18 with TypeScript","styling":"Tailwind CSS","state":"React Query"},"backend":{"runtime":"Node 20","framework":"NestJS with TypeORM","api":"RESTful with OpenAPI 3.0"},"security":{"auth":"Cognito with MFA","encryption":"AES-256 at rest, TLS 1.3 in transit","compliance":"HIPAA"},"data":{"db":"Aurora PostgreSQL 15","cache":"ElastiCache Redis for sessions"},"devops":{"ci":"GitHub Actions","hosting":"AWS ECS Fargate","environments":"dev, staging, production"},"constraints":{"region":"us-east-1","budget":"$1500/month"},"open_questions":{},"confirmed_decisions":{"cloud":"AWS","auth_provider":"Cognito"}},"reasoner_reviews":{"product":"Scope is well-defined for MVP. Intake forms and patient history are the core features. Staff review dashboard is essential for clinical workflow adoption.","architect_reasoner":"NestJS with TypeORM on Fargate is feasible for this scale. Recommend read replica for reporting queries to avoid impacting transactional performance.","security":"HIPAA compliance requires encryption at rest with KMS, comprehensive audit logging for all PHI access, and a BAA with AWS. MFA must be enforced for every user role.","constraints":"Budget of $1500/month supports Fargate with auto-scaling. Q3 delivery is achievable with the defined MVP scope."},"specialist_subplans":{"backend":{"service_design":"Modular NestJS with domain modules for intake, patients, users, and audit","api_patterns":"RESTful with OpenAPI 3.0 and versioned endpoints","failure_handling":"Circuit breaker for external calls, global exception filter"},"frontend":{"app_structure":"React SPA with route-based code splitting","pages_and_flows":"Login, Patient Registration, Intake Form, Staff Dashboard, Admin Panel","state_management":"React Query for server state, Zustand for local UI state"},"security":{"auth_design":"Cognito user pools with MFA in HIPAA-eligible configuration","authorization_model":"RBAC with patient/staff/physician/admin roles","audit_and_logging_controls":"CloudTrail for AWS actions, custom audit_log table for PHI access"},"data":{"entities":"patients, intake_forms, form_submissions, documents, audit_log","storage_design":"Aurora PostgreSQL with KMS encryption, S3 for document uploads","retention_and_deletion":"7-year retention with automated archival to S3 Glacier"}},"issue_ledger":{},"focus_issues":[],"revision_memory":{},"accepted_exceptions":{},"previous_audits":[],"previous_plan":{},"best_plan":{}},"target_output":{"thinking_summary":"Designing a HIPAA-compliant patient intake system on AWS. Key decisions: NestJS modular backend for domain separation, Aurora PostgreSQL with KMS for encryption at rest, Cognito with MFA for authentication. Prioritizing security and compliance given health data sensitivity while staying within $1500/month budget.","fix_report":[],"title":"Patient Intake Portal Architecture","executive_summary":"A HIPAA-compliant fullstack application for clinic patient intake management built with React 18 and NestJS on AWS ECS Fargate with Aurora PostgreSQL. Handles patient registration, intake form submission, medical history capture, and staff review workflows with end-to-end encryption and comprehensive audit logging.","architecture_overview":"Three-tier architecture with React 18 SPA served via CloudFront communicating over HTTPS to a NestJS REST API on ECS Fargate behind an ALB with WAF. Backend connects to Aurora PostgreSQL 15 for persistence with KMS encryption at rest and TLS 1.3 in transit. ElastiCache Redis handles session caching. Authentication flows through AWS Cognito with mandatory MFA. All PHI access logged to a dedicated audit table and forwarded to CloudTrail.","technology_stack":"Frontend: React 18, TypeScript, Tailwind CSS, React Query, React Router v6. Backend: NestJS on Node 20, TypeORM, class-validator, Passport.js. Database: Aurora PostgreSQL 15 with KMS and read replica. Cache: ElastiCache Redis 7. Infrastructure: ECS Fargate, ALB with WAF, CloudFront, Route53, ACM, S3 for documents.","functional_feature_map":"Patient Registration module for self-service and staff-assisted signup with Cognito MFA enrollment. Intake Form Engine with configurable templates and conditional fields. Form Submission Workflow routing from patient entry through staff review to physician sign-off. Medical History Viewer with timeline display and PDF export. Staff Dashboard showing pending reviews, overdue alerts, and patient search.","system_components":"CloudFront CDN for static assets, ALB with WAF rules for API traffic, ECS Fargate cluster running API and background worker containers, Aurora PostgreSQL primary with read replica, ElastiCache Redis for sessions, S3 with server-side encryption for documents, Cognito user pool and identity pool, CloudWatch for logs, CloudTrail for audit trail.","workflows":"Patient Registration: Patient creates account via Cognito, completes MFA enrollment, fills profile form, data saved to patients table, welcome email via SES. Intake Submission: Patient selects form template, fills required sections with real-time validation, uploads documents to S3, submits triggering staff notification via SNS. Staff Review: Staff views pending queue, opens submission to review PHI, approves or requests corrections, physician performs final sign-off, record status updated to finalized.","data_model":"patients (id UUID PK, cognito_sub, first_name, last_name, dob, email_encrypted, phone_encrypted, created_at). intake_forms (id UUID PK, template_id FK, patient_id FK, status ENUM pending/in_review/approved/rejected, assigned_staff_id, submitted_at). form_fields (id UUID PK, form_id FK, field_key, value_encrypted TEXT). documents (id UUID PK, form_id FK, s3_key, content_type). audit_log (id UUID PK, user_id, action, resource_type, resource_id, ip_address, timestamp). All PHI encrypted with AES-256 before storage.","api_design":"RESTful API with OpenAPI 3.0 at /api/v1. POST /patients (register), GET /patients/:id (profile), POST /intakes (submit form), GET /intakes?status=pending (staff queue), PATCH /intakes/:id/review (review action), POST /intakes/:id/documents (upload), GET /audit-log (compliance report). Auth: Cognito JWT at ALB and NestJS guard. Rate limit: 100 req/min per user.","security_and_compliance":"HIPAA controls: BAA with AWS, KMS encryption for PHI at rest, TLS 1.3 in transit, Cognito MFA for all roles, RBAC with patient/staff/physician/admin, PHI access audit logging to append-only table plus CloudTrail, 15-minute session timeout, WAF blocking SQL injection and XSS, optional IP allowlisting for staff portal.","deployment_and_operations":"GitHub Actions CI: lint, test (80% coverage gate), build Docker, push ECR, deploy ECS rolling update. Environments: dev (single AZ), staging (multi-AZ mirror), production (multi-AZ us-east-1a/1c, auto-scaling 2-6 tasks, Aurora multi-AZ failover). Rollback: automatic on failed health checks. Migrations: TypeORM via ECS one-off tasks. Secrets: AWS Secrets Manager with 30-day rotation.","observability":"Datadog APM sidecar on each ECS task. CloudWatch Logs with structured JSON and correlation IDs. Custom metrics: intake submission rate, review queue depth, avg review time, auth failure rate. PagerDuty escalation for: error rate >1%, p99 latency >2s, queue depth >50, DB connections >80%.","cost_and_scaling":"~$1200/month for 500 daily intakes. ECS auto-scaling on CPU 60% target. Aurora read replica on-demand for reporting. CloudFront caching for static assets. Cost optimization: reserved Fargate baseline, S3 Intelligent-Tiering, Glacier lifecycle for documents >1 year.","phased_implementation":"Phase 1 MVP (8 weeks): Patient registration with MFA, 3 standard intake templates, staff review dashboard, core API with OpenAPI spec, HIPAA security controls, CI/CD pipeline. Phase 2 (4 weeks): Custom form builder, document upload with S3, medical history timeline, PDF export. Phase 3 (4 weeks): Analytics dashboard, SES notifications, SAML federation for hospital SSO, advanced audit reporting.","development_guidelines":"TypeScript strict mode. ESLint + Prettier in CI. 80% test coverage for API endpoints via Jest. Cypress E2E for registration, intake submission, and review flows. PR requires 1 senior approval. Trunk-based development with feature flags. ADR documents for major decisions.","risks_and_tradeoffs":"Risk: Cognito vendor lock-in. Mitigation: abstract auth behind service interface for Auth0/Keycloak portability. Risk: Aurora cost at scale beyond 2000 daily submissions. Mitigation: read replica only when reporting load exceeds threshold. Tradeoff: server-side KMS encryption over client-side for simpler key management. Tradeoff: monolithic NestJS for MVP speed with module boundaries for future service extraction.","open_questions_resolved":"Selected Aurora over RDS PostgreSQL for automatic failover. Confirmed S3 for documents over EFS. Chose React Query over Redux for simpler server state. MFA enforced for all roles including patients."},"metadata":{"schema_version":"v2_architect_aligned","case_type":"first_pass","plan_quality":"strong","primary_theme":"compliance","generation_source":"synthetic","quality_flags":[],"notes":"HIPAA-compliant healthcare intake with comprehensive security."}}

═══════════════════════════════════════════════════════════════
FINAL CHECKLIST (verify for EVERY row)
═══════════════════════════════════════════════════════════════

1. Output is JSONL — one JSON object per line, no array wrapper
2. Every frozen_requirement_contract field is a FULL OBJECT (never a plain string)
3. All 21 mandatory contract fields present
4. Conditional fields included when conditions met
5. requirements is dict-of-dicts
6. Plan sections are DOMAIN-SPECIFIC — not generic boilerplate
7. data_model entities match the actual project domain (not Users/Orders/Products for healthcare)
8. workflows describe actual domain workflows (not generic registration/checkout for all projects)
9. No two rows share the same plan section text
10. fix_report is [] for round 1, addresses focus_issues for round > 1
11. revision_round rows have non-empty issue_ledger, focus_issues, previous_audits
12. plan_quality matches actual content quality
13. 3 different project classes per batch
14. Domain, technology, and architecture diversity is high
15. No forbidden enum values

Now generate EXACTLY {batch_size} rows as JSONL.
"""

# ══════════════════════════════════════════════════════════════
# ASYNC ENGINE
# ══════════════════════════════════════════════════════════════

class DatasetGenerator:
    def __init__(self) -> None:
        self.client = AsyncAzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=API_KEY,
            api_version=API_VERSION,
        )
        self.semaphore   = asyncio.Semaphore(MAX_CONCURRENT)
        self.write_lock  = asyncio.Lock()
        self.error_lock  = asyncio.Lock()

        # Shared counters (protected by write_lock)
        self.written       = 0
        self.dropped       = 0
        self.api_errors    = 0
        self.total_tokens  = 0
        self.batch_times: list[float] = []

        # Rich
        self.console = Console()

    # ----------------------------------------------------------
    # File resume: count existing rows
    # ----------------------------------------------------------
    def count_existing(self) -> int:
        if not OUTPUT_FILE.exists():
            return 0
        with OUTPUT_FILE.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    # ----------------------------------------------------------
    # Build prompt
    # ----------------------------------------------------------
    def build_prompt(self, current_count: int, start_index: int) -> str:
        start_id = f"ARC-PLAN26-{start_index:03d}"
        return (
            BASE_PROMPT
            .replace("{current_count}", str(current_count))
            .replace("{batch_size}", str(BATCH_SIZE))
            .replace("{start_id}", start_id)
        )

    # ----------------------------------------------------------
    # Strip markdown fences the model might still emit
    # ----------------------------------------------------------
    @staticmethod
    def strip_fences(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:jsonl?|json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*$", "", text)
        return text.strip()

    # ----------------------------------------------------------
    # Attempt light JSON repair for common model mistakes
    # ----------------------------------------------------------
    @staticmethod
    def try_repair(line: str) -> str | None:
        """Very light repair: strip trailing commas before } or ]."""
        line = re.sub(r",\s*([}\]])", r"\1", line)
        return line

    # ----------------------------------------------------------
    # Parse raw output into validated JSON objects
    # ----------------------------------------------------------
    def parse_batch(self, raw: str) -> tuple[list[dict], list[dict]]:
        """
        Returns (valid_rows, error_rows).
        error_rows = [{"line": ..., "errors": [...]}]
        """
        valid, errors = [], []
        raw = self.strip_fences(raw)
        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            obj = None
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                repaired = self.try_repair(line)
                try:
                    obj = json.loads(repaired)
                except json.JSONDecodeError as e:
                    errors.append({"line": line[:300], "errors": [f"JSONDecodeError: {e}"]})
                    continue

            if not isinstance(obj, dict):
                errors.append({"line": line[:300], "errors": ["Not a JSON object"]})
                continue

            validation_errors = validate_row(obj)
            if validation_errors:
                errors.append({"line": line[:300], "errors": validation_errors, "sample_id": obj.get("sample_id", "?")})
            else:
                valid.append(obj)

        return valid, errors

    # ----------------------------------------------------------
    # Single API call with exponential back-off
    # ----------------------------------------------------------
    async def call_api(self, prompt: str, attempt: int = 0) -> tuple[str, int]:
        """Returns (raw_text, total_tokens). Raises on final failure."""
        try:
            resp = await self.client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_p=0.95,
                frequency_penalty=0.2,
                presence_penalty=0.1,
            )
            tokens = resp.usage.total_tokens if resp.usage else 0
            return resp.choices[0].message.content.strip(), tokens
        except Exception as exc:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                await asyncio.sleep(delay)
                return await self.call_api(prompt, attempt + 1)
            raise exc

    # ----------------------------------------------------------
    # Process one batch (acquire semaphore → call → validate → write)
    # ----------------------------------------------------------
    async def process_batch(
        self,
        batch_index: int,
        current_count_at_dispatch: int,
        progress,
        task_id,
    ) -> None:
        start_index = current_count_at_dispatch + 1
        prompt = self.build_prompt(current_count_at_dispatch, start_index)
        t0 = time.monotonic()

        async with self.semaphore:
            try:
                raw, tokens = await self.call_api(prompt)
            except Exception as exc:
                async with self.write_lock:
                    self.api_errors += 1
                self.console.print(f"[red]Batch {batch_index} API error:[/] {exc}")
                return

        elapsed = time.monotonic() - t0
        valid_rows, error_rows = self.parse_batch(raw)

        # Write valid rows
        if valid_rows:
            lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in valid_rows) + "\n"
            async with self.write_lock:
                async with aiofiles.open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    await f.write(lines)
                self.written      += len(valid_rows)
                self.total_tokens += tokens
                self.batch_times.append(elapsed)
                progress.advance(task_id, advance=len(valid_rows))

        # Log bad rows for debugging
        if error_rows:
            bad_lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in error_rows) + "\n"
            async with self.error_lock:
                async with aiofiles.open(ERRORS_FILE, "a", encoding="utf-8") as f:
                    await f.write(bad_lines)
            async with self.write_lock:
                self.dropped += len(error_rows)

    # ----------------------------------------------------------
    # Main orchestration loop
    # ----------------------------------------------------------
    async def run(self) -> None:
        already_written = self.count_existing()
        remaining = TOTAL_ROWS - already_written

        if remaining <= 0:
            self.console.print(f"[green]Dataset already complete ({already_written} rows).[/]")
            return

        self.written = already_written
        self.console.print(
            Panel(
                f"[bold cyan]Architect Dataset Generator[/]\n"
                f"Target: [yellow]{TOTAL_ROWS}[/] rows  |  "
                f"Existing: [yellow]{already_written}[/]  |  "
                f"To generate: [yellow]{remaining}[/]\n"
                f"Concurrency: [yellow]{MAX_CONCURRENT}[/] workers  |  "
                f"Batch size: [yellow]{BATCH_SIZE}[/]  |  "
                f"Max tokens/call: [yellow]{MAX_TOKENS}[/]",
                title="Config",
                border_style="cyan",
            )
        )

        # Build all batch tasks up front
        # We over-schedule slightly (extra batches) to absorb dropped rows
        n_batches = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
        # +20% extra to cover validation drops
        n_batches = int(n_batches * 1.2) + MAX_CONCURRENT

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            refresh_per_second=4,
        )

        with progress:
            task_id = progress.add_task(
                "Generating rows…",
                total=remaining,
            )

            # Dispatch batches in chunks of MAX_CONCURRENT to avoid over-scheduling
            # while still keeping the pipeline full
            dispatched_count = 0
            current_count = already_written

            while self.written < TOTAL_ROWS:
                # How many more do we need?
                still_needed = TOTAL_ROWS - self.written
                if still_needed <= 0:
                    break

                # Dispatch up to MAX_CONCURRENT batches in parallel
                wave_size = min(MAX_CONCURRENT, (still_needed + BATCH_SIZE - 1) // BATCH_SIZE + 2)

                coros = []
                for i in range(wave_size):
                    batch_idx = dispatched_count + i
                    # Estimate the count at dispatch time
                    count_estimate = current_count + i * BATCH_SIZE
                    coros.append(
                        self.process_batch(batch_idx, count_estimate, progress, task_id)
                    )

                await asyncio.gather(*coros)
                dispatched_count += wave_size
                current_count = self.written  # sync after wave

                # Safety: if we somehow can't make progress, break
                if wave_size == 0:
                    break

        # Final stats
        avg_time = (
            sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        )
        stats = Table(title="Generation Complete", show_header=False, box=None)
        stats.add_column(style="cyan")
        stats.add_column(style="green")
        stats.add_row("Rows written",    str(self.written))
        stats.add_row("Rows dropped",    str(self.dropped))
        stats.add_row("API errors",      str(self.api_errors))
        stats.add_row("Total tokens",    f"{self.total_tokens:,}")
        stats.add_row("Avg batch time",  f"{avg_time:.1f}s")
        stats.add_row("Output file",     str(OUTPUT_FILE))
        if self.dropped:
            stats.add_row("Error log",   str(ERRORS_FILE))
        self.console.print(stats)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

async def main() -> None:
    gen = DatasetGenerator()
    await gen.run()


if __name__ == "__main__":
    asyncio.run(main())