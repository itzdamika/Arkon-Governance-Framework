from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

GENERATOR_PROMPT = r"""
You are generating a premium-quality synthetic fine-tuning batch for the AuditorAgent of an architectural governance framework.

GOAL
Create EXACTLY 10 final AuditorAgent training samples in one response.
Each sample must be a complete final dataset row containing:
1. a realistic audit input_payload with a flawed architecture plan and its surrounding governance state
2. a gold-standard target_output representing the correct AuditorAgent response

IMPORTANT
This is a one-stage generation task.
Do NOT split the work.
Generate the full final training rows directly.

OUTPUT FORMAT
Return EXACTLY one valid JSON array containing EXACTLY 10 objects.
Do not use markdown.
Do not add commentary before or after the JSON.
Do not wrap the JSON in backticks.

TOP-LEVEL SHAPE OF EACH ARRAY ITEM
{
  "sample_id": "",
  "dataset": "auditor",
  "agent": "AuditorAgent",
  "split": "train",
  "profile": {
    "projectclass": "",
    "capabilities": [],
    "complexitylevel": "",
    "risklevel": "",
    "datasensitivity": "",
    "externalexposure": ""
  },
  "input_payload": {
    "round": 1,
    "frozenrequirementcontract": {},
    "requirements": {},
    "acceptedexceptions": {},
    "issueledger": {},
    "revisionmemory": {},
    "previousaudits": [],
    "plan": {},
    "bestaudit": {}
  },
  "target_output": {
    "thinkingsummary": "",
    "rubricscores": {
      "requirementsalignment": 0,
      "architecturequality": 0,
      "security": 0,
      "operability": 0,
      "internalconsistency": 0
    },
    "summary": "",
    "strengths": [],
    "concerns": [],
    "blockingissues": [],
    "recommendations": [],
    "requirementconflicts": [],
    "issueupdates": []
  },
  "metadata": {
    "schema_version": "v1",
    "case_type": "",
    "primary_flaw_family": "",
    "secondary_flaw_families": [],
    "intended_severity_mix": [],
    "generation_source": "synthetic",
    "quality_flags": [],
    "notes": ""
  }
}

SAMPLE ID RULE
Use this exact sample_id pattern:
"AUD-{{BATCH_ID}}-{{NNN}}"
Where:
- {{BATCH_ID}} is supplied externally
- {{NNN}} is a zero-padded index from 001 to 010

SYSTEM CONTEXT
This AuditorAgent belongs to a stateful, phase-based SDLC governance system where:
- requirements are locked before planning
- planning and auditing are internal
- the Auditor audits architecture plans against frozen confirmed requirements, rich requirement notes, cumulative issue ledger, revision memory, and prior audit history
- acceptedexceptions and bestaudit may appear as supporting context
- the Auditor must use stable issue IDs where possible
- the Auditor must mark issue status as one of: unresolved, resolved, downgraded, new
- the Auditor must score against an absolute rubric, not against any approval threshold
- the runtime computes final pass/fail separately
- the goal is to train a rigorous, non-appeasing, technically precise architecture critic

ALIGNMENT RULE
Prioritize fidelity to the real Auditor contract over unnecessary extra context.
The most important input fields are:
- frozenrequirementcontract
- requirements
- issueledger
- revisionmemory
- previousaudits
- plan

acceptedexceptions and bestaudit may be included when relevant, but do not make the sample depend on them.

QUALITY STANDARD
Each sample must be:
- technically realistic
- internally coherent
- implementation-grade
- detailed enough for a real audit
- high-signal for fine-tuning
- non-repetitive
- useful for training strict architecture critique

Avoid:
- toy projects
- filler text
- vague plans
- generic criticisms
- repeated templates with renamed technologies
- shallow governance history
- obviously broken nonsense plans

BATCH DIVERSITY RULES
Across the 10 samples:
- use at least 6 distinct projectclass values
- use at least 4 distinct domains
- use at least 3 datasensitivity levels
- use at least 3 externalexposure levels
- use at least 3 case_type values
- use at least 4 primary flaw families
- do not let one tech stack dominate
- do not overuse chatbots or simple CRUD apps

PROJECTCLASS OPTIONS
Use realistic values such as:
- webapp
- fullstackapp
- mobileapp
- desktopapp
- apiservice
- staticwebsite
- clitool
- automationtool
- datapipeline
- aisystem
- researchprototype
- infrastructureproject

CAPABILITIES OPTIONS
Use coherent combinations from:
- frontend
- backend
- data
- auth
- aillm
- integrations
- analytics
- realtime
- payments
- adminpanel
- publicapi
- batchjobs
- devops

CASE TYPES
Choose one per sample:
- first_pass
- revision_round
- regression_case
- requirement_conflict_case

FLAW FAMILIES
Use one primary flaw family and optionally one or two secondary flaw families:
- requirementsalignment
- architecturequality
- security
- operability
- internalconsistency
- requirementconflict

DOMAIN DIVERSITY
Vary realistic domains such as:
- fintech
- healthcare
- e-commerce
- logistics
- education
- internal enterprise tooling
- IoT monitoring
- public sector services
- developer platforms
- research workflows
- media/content systems
- document processing
- AI-assisted productivity
- security operations

FROZEN REQUIREMENT CONTRACT RULES
Create a realistic frozenrequirementcontract using relevant fields such as:
- projectgoal
- targetusers
- projectclass
- capabilities
- complexitylevel
- risklevel
- datasensitivity
- externalexposure
- accessmodel
- featurescope
- frontendstack
- backendstack
- dataplatform
- hostingtarget
- securitybaseline
- privacyretentionpolicy
- mvpscope
- futurescope
- constraints
- observabilitybaseline
- executionpreference
- llmintegration
- compliancecontext

Each populated field must use this exact object shape:
{
  "value": "...",
  "source": "...",
  "confirmed": true,
  "rationale": "...",
  "updatedat": "2026-03-15T00:00:00Z"
}

Contract depth rules:
- populate at least 12 fields
- always include:
  - projectgoal
  - targetusers
  - projectclass
  - capabilities
  - accessmodel
  - featurescope
  - mvpscope
  - risklevel
  - datasensitivity
  - externalexposure
  - securitybaseline
- include llmintegration whenever aillm exists in capabilities
- include compliancecontext whenever risk is high or data is personal, financial, health, or confidential
- make privacy and retention stricter for sensitive systems
- make the stack, database, and hosting choices fit the project type and exposure

RICH REQUIREMENTS RULES
Populate "requirements" as structured notes with sections such as:
- project
- frontend
- backend
- security
- data
- devops
- constraints
- openquestions
- confirmeddecisions

Depth rules:
- populate at least 5 sections
- each populated section must contain at least 2 concrete facts or design notes
- do not merely repeat the contract
- confirmeddecisions must contain real locked choices
- openquestions may be empty in some cases, but not all

ISSUE HISTORY RULES
For first_pass:
- previousaudits may be empty or minimal
- issueledger may be empty or lightly seeded
- revisionmemory may be empty or minimal
- round should usually be 1

For revision_round:
- round should usually be 2 or 3
- include 1 or 2 prior audit objects
- include 2 to 5 prior issues in issueledger
- revisionmemory must explain what changed
- at least one prior issue should still matter now
- stable issue IDs must be reused correctly

For regression_case:
- include believable prior history
- at least one earlier issue should appear improved
- at least one new serious issue must now exist
- the audit summary must later be able to say this is a regression

For requirement_conflict_case:
- the locked contract must clearly require one thing
- the plan must materially violate it
- the conflict must be auditable from the sample itself

ISSUE ID RULES
Use readable stable IDs such as:
- SEC-001
- ARCH-002
- OPS-003
- REQ-004
- CONS-005

If prior history exists:
- reuse IDs consistently
- do not fabricate continuity
- make prior titles, severities, and statuses plausible

PLAN RULES
The "plan" must look like a realistic ArchitectAgent output and may include keys such as:
- thinkingsummary
- fixreport
- title
- executivesummary
- architectureoverview
- technologystack
- functionalfeaturemap
- systemcomponents
- workflows
- datamodel
- apidesign
- securityandcompliance
- deploymentandoperations
- observability
- costandscaling
- phasedimplementation
- developmentguidelines
- risksandtradeoffs
- openquestionsresolved

Plan depth rules:
- include at least 10 substantial plan sections
- each major section must contain concrete implementation details
- the plan must be mostly plausible and professionally written
- the plan must not openly admit its own flaws
- most of the plan should be credible so the audit must be precise

FLAW DESIGN RULES
Each sample must contain 1 to 3 deliberate high-value flaws.

Allowed flaw families:
- requirementsalignment
- architecturequality
- security
- operability
- internalconsistency
- requirementconflict

Flaw rules:
- the flaws must be auditable from actual plan content
- at least one flaw must be medium, high, or critical severity
- not all flaws should be security-only
- some flaws may be omissions
- some flaws may be contradictions across sections
- some flaws may be requirement mismatches
- some flaws may be regressions after partial fixes
- do not make the whole plan garbage
- do not make the flaw so hidden that the audit becomes vague
- do not make the flaw read like an answer key

Useful flaw examples include:
- weak auth design for sensitive systems
- missing RBAC where role separation is required
- absent rate limiting on exposed APIs
- poor secrets management
- missing deletion or retention enforcement
- wrong database choice for workload or scale
- contradictory API and schema assumptions
- weak observability for high-risk systems
- impossible deployment claims for chosen stack
- frontend-side LLM secret exposure
- architecture violating confirmed requirements

TARGET_OUTPUT RULES
Generate the correct gold-standard AuditorAgent output for each sample.

The target_output must obey all of these rules:
- thinkingsummary must be brief, precise, and professional
- rubricscores must contain numeric values from 0 to 10 for:
  - requirementsalignment
  - architecturequality
  - security
  - operability
  - internalconsistency
- do NOT include any overall score field
- do NOT include any pass/fail field
- do NOT mention any approval threshold
- summary must explain overall quality and, when prior history exists, whether the plan improved, stayed flat, or regressed
- strengths must mention real positive qualities still present in the flawed plan
- concerns must mention the major weaknesses
- blockingissues must contain only the most important blockers
- recommendations must be concrete and action-ready
- requirementconflicts must contain only real conflicts grounded in the contract
- issueupdates must be the main audit memory object

MINIMUM TARGET DEPTH
For each sample:
- strengths: 2 to 4 items
- concerns: 2 to 5 items
- recommendations: 2 to 5 items
- issueupdates: 2 to 6 items
- blockingissues: 0 to 3 items
- requirementconflicts: 0 to 3 items

REQUIREMENT CONFLICT ITEM SHAPE
Each requirementconflicts item must be:
{
  "issueid": "",
  "field": "",
  "currentvalue": "",
  "proposedvalue": "",
  "exactreason": "",
  "severity": "low|medium|high|critical"
}

Requirement conflict rules:
- field must exist in frozenrequirementcontract
- currentvalue must reflect the conflicting plan choice
- proposedvalue must reflect the locked requirement or required correction
- exactreason must clearly explain the mismatch
- severity must reflect real impact

ISSUEUPDATE ITEM SHAPE
Each issueupdates item must be:
{
  "id": "",
  "title": "",
  "severity": "low|medium|high|critical",
  "status": "unresolved|resolved|downgraded|new",
  "detail": ""
}

ISSUEUPDATE RULES
- reuse stable IDs when the same issue still exists
- only create a new ID when the issue is materially different
- if a prior issue was fixed, keep the same ID and mark it resolved
- if a prior issue still exists, keep the same ID and explain what remains wrong
- if the issue is less severe now, mark it downgraded
- if a new serious flaw appears, mark it new
- do not invent continuity without evidence in previousaudits or issueledger

DETAIL FIELD RULES
For every issueupdates.detail:
- state whether the issue was fixed, partially fixed, unchanged, or newly introduced
- mention the exact plan section name or names supporting the judgment, such as:
  - architectureoverview
  - systemcomponents
  - workflows
  - datamodel
  - apidesign
  - securityandcompliance
  - deploymentandoperations
  - observability
  - costandscaling
  - phasedimplementation
  - developmentguidelines
  - risksandtradeoffs
- explain how the issue affected one or more rubric dimensions
- if one area improved but another problem was introduced, say that clearly
- keep the detail concise but evidence-bearing

RUBRIC CALIBRATION RULES
The rubric must be honest and severity-sensitive.
- do not inflate scores
- do not cluster everything around 8 to 9
- weak plans should get visibly weak subscores
- strong-but-flawed plans may get mixed score profiles
- the most severe unresolved flaw must noticeably reduce the relevant rubric dimension

Rubric meaning:
- requirementsalignment = fit to locked requirements
- architecturequality = structural soundness and fit for purpose
- security = auth, authorization, secrets, abuse prevention, privacy, exposure handling
- operability = deployment realism, observability, rollback, backup, runtime manageability
- internalconsistency = whether the plan sections agree with each other

CROSS-CHECK RULES
Silently enforce all of the following for every sample:
1. Every important concern maps to at least one issueupdate.
2. Every blocking issue is supported by issueupdates and plan evidence.
3. Every requirement conflict refers to a real contract field and real plan mismatch.
4. Every reused issue ID already exists in previousaudits or issueledger.
5. Every resolved or downgraded issue explains what changed from prior history.
6. If summary says improved, stayed flat, or regressed, issueupdates and rubricscores must support it.
7. If a severe unresolved flaw exists, the relevant rubric dimension must show a meaningful penalty.
8. Recommendations must target unresolved or highest-impact issues.
9. The audit must not contradict the plan, the history, or its own scores.

BIAS-REDUCTION RULES
Keep the audit fair and evidence-based.
- do not favor one stack by default
- do not always punish monoliths
- do not always reward microservices
- do not assume Python/React is always best
- do not assume cloud-first is always best
- judge by requirement fit, risk profile, and engineering evidence

METADATA RULES
Populate metadata with:
- schema_version = "v1"
- case_type
- primary_flaw_family
- secondary_flaw_families
- intended_severity_mix
- generation_source = "synthetic"
- quality_flags = []
- notes = a short explanation of why the sample is useful for Auditor training

FINAL SELF-CHECK
Before returning the JSON array, silently verify:
- there are exactly 10 items
- all sample_id values are unique
- every item follows the schema exactly
- every plan is detailed enough to support the audit
- every target_output is grounded in its own plan and history
- issue IDs are reused correctly only when justified
- there is no threshold language
- there is no overall score field
- there are no duplicate or near-duplicate samples
- the output is valid JSON

FINAL OUTPUT RULE
Return EXACTLY one valid JSON array of EXACTLY 10 final dataset rows and nothing else.

"""


LLM_AUDIT_PROMPT = r"""
You are a premium-quality dataset validator for the AuditorAgent of an architectural governance framework.

GOAL
Validate EXACTLY 10 candidate AuditorAgent training rows.
Your job is to determine whether each row is truly suitable for premium fine-tuning, not merely whether it looks structurally valid.

You must check:
1. exact schema compliance
2. alignment to the real AuditorAgent contract
3. technical realism of the flawed plan
4. internal consistency between requirements, history, plan, and audit output
5. correctness of issue continuity and status tracking
6. calibration and honesty of rubric scores
7. evidence-grounded requirement conflicts and issue updates
8. duplication or near-duplication across the batch
9. whether each row is safe to keep for premium training

STRICT PRINCIPLE
Do NOT trust a row just because it is well-formatted.
A row must be rejected if it is shallow, inconsistent, weakly grounded, repetitive, or likely to teach the wrong behavior.

INPUT
You will receive EXACTLY one JSON array of EXACTLY 10 candidate final dataset rows.

Each candidate row is expected to follow this shape:
{
  "sample_id": "",
  "dataset": "auditor",
  "agent": "AuditorAgent",
  "split": "train",
  "profile": {
    "projectclass": "",
    "capabilities": [],
    "complexitylevel": "",
    "risklevel": "",
    "datasensitivity": "",
    "externalexposure": ""
  },
  "input_payload": {
    "round": 1,
    "frozenrequirementcontract": {},
    "requirements": {},
    "acceptedexceptions": {},
    "issueledger": {},
    "revisionmemory": {},
    "previousaudits": [],
    "plan": {},
    "bestaudit": {}
  },
  "target_output": {
    "thinkingsummary": "",
    "rubricscores": {
      "requirementsalignment": 0,
      "architecturequality": 0,
      "security": 0,
      "operability": 0,
      "internalconsistency": 0
    },
    "summary": "",
    "strengths": [],
    "concerns": [],
    "blockingissues": [],
    "recommendations": [],
    "requirementconflicts": [],
    "issueupdates": []
  },
  "metadata": {
    "schema_version": "v1",
    "case_type": "",
    "primary_flaw_family": "",
    "secondary_flaw_families": [],
    "intended_severity_mix": [],
    "generation_source": "synthetic",
    "quality_flags": [],
    "notes": ""
  }
}

REAL AUDITOR CONTRACT TO VALIDATE AGAINST
The real AuditorAgent is a strict architecture critic.
Its core audit basis is:
- frozen confirmed requirements
- rich requirement notes
- cumulative issue ledger
- revision memory
- prior audit history
- current architecture plan

Supporting context may also appear:
- acceptedexceptions
- bestaudit

The validator must ensure that the candidate rows train the model to:
- use stable issue IDs when the same issue persists
- mark issue status as unresolved, resolved, downgraded, or new
- score against an absolute rubric, not against any threshold
- explain whether quality improved, stayed flat, or regressed when history exists
- ground every important criticism in the plan and governance context
- avoid appeasement, score inflation, and threshold chasing

RESPONSE FORMAT
Return EXACTLY one valid JSON object and nothing else.
Do not use markdown.
Do not include commentary before or after the JSON.
Do not wrap the JSON in backticks.

RETURN THIS EXACT TOP-LEVEL SHAPE
{
  "validator_version": "v1",
  "batch_verdict": "accept|accept_with_minor_repairs|reject",
  "batch_summary": {
    "total_rows": 10,
    "accepted_count": 0,
    "accepted_with_minor_repairs_count": 0,
    "rejected_count": 0,
    "overall_quality_assessment": "",
    "most_common_failure_modes": [],
    "notes": ""
  },
  "diversity_report": {
    "duplicate_or_near_duplicate_pairs": [],
    "projectclass_distribution_ok": true,
    "domain_diversity_ok": true,
    "flaw_family_diversity_ok": true,
    "stack_bias_warnings": [],
    "repetition_warnings": []
  },
  "sample_results": [
    {
      "sample_id": "",
      "verdict": "accept|accept_with_minor_repairs|reject",
      "quality_tier": "premium|usable_with_review|reject",
      "critical_failures": [],
      "warnings": [],
      "minor_repairs_applied": [],
      "repairable": true,
      "repaired_row": null,
      "validation_notes": ""
    }
  ],
  "approved_rows": []
}

BATCH VERDICT RULES
Set batch_verdict as:
- "accept" only if all 10 rows are premium and need no changes
- "accept_with_minor_repairs" if all non-rejected rows are acceptable after only minor repairs, and no more than 2 rows are rejected
- "reject" if 3 or more rows are rejected, or if the batch shows serious duplication, shallow generation, systemic inconsistency, or obvious quality collapse

SAMPLE VERDICT RULES
For each sample:
- verdict = "accept" when the row is premium and should be kept unchanged
- verdict = "accept_with_minor_repairs" when only small non-semantic fixes are needed
- verdict = "reject" when the row has substantive quality problems and must not be used for training

QUALITY_TIER RULES
- "premium" = strong enough for final fine-tuning data
- "usable_with_review" = structurally useful but only acceptable after limited repair
- "reject" = too inconsistent, shallow, noisy, repetitive, or misleading for training

APPROVED_ROWS RULES
Populate approved_rows with only:
- rows whose verdict is "accept"
- repaired versions of rows whose verdict is "accept_with_minor_repairs"

Do NOT include rejected rows in approved_rows.

REPAIR POLICY
You may apply ONLY minor repairs.
Minor repairs are allowed only when they are purely mechanical and do NOT alter substantive meaning.

Allowed minor repairs:
- remove accidental extra keys
- normalize obvious whitespace or formatting defects
- convert numeric rubric values from strings to numbers when meaning is unambiguous
- remove duplicated list items when duplicates are clearly accidental
- normalize empty arrays or empty objects where the intended type is obvious
- fix trivial enum casing only when the intended value is obvious and already supported by context

Forbidden repairs:
- inventing new architectural facts
- rewriting the plan
- changing the core meaning of the audit
- changing rubric scores for quality reasons
- inventing issue history continuity
- creating new requirement conflicts not clearly supported by the row
- replacing weak issue details with stronger ones
- adding missing depth that was never present
- repairing major inconsistencies by making up evidence

If a row needs forbidden repair, REJECT it.

VALIDATION DIMENSIONS
Validate every row against all of the following dimensions.

A. SCHEMA EXACTNESS
Reject if:
- any required top-level key is missing
- dataset is not "auditor"
- agent is not "AuditorAgent"
- split is not "train"
- target_output contains extra unsupported keys
- rubricscores is missing any of the five required dimensions
- requirementconflicts items or issueupdates items do not follow required shapes

B. CORE AUDITOR ALIGNMENT
Check that the row actually trains the intended Auditor behavior.
The most important inputs must be:
- frozenrequirementcontract
- requirements
- issueledger
- revisionmemory
- previousaudits
- plan

acceptedexceptions and bestaudit may appear, but the row should not depend on them as the main audit basis.

Reject if:
- the audit ignores the core input basis
- the audit appears driven mainly by secondary fields
- the row teaches threshold-based judging
- the row teaches pass/fail behavior inside target_output
- the row teaches generic code review instead of architecture audit

C. CONTRACT QUALITY
Check frozenrequirementcontract for realism and usefulness.

Reject if:
- fewer than 12 populated fields exist without strong justification
- core fields are missing, especially projectgoal, targetusers, projectclass, capabilities, accessmodel, featurescope, mvpscope, risklevel, datasensitivity, externalexposure, securitybaseline
- fields are mutually contradictory without that contradiction being intentionally part of the case
- llmintegration is missing even though capabilities includes aillm
- compliancecontext is missing even though risk is high or data is personal, financial, health, or confidential
- privacy expectations are implausibly weak for sensitive systems

Warn if:
- the contract is technically valid but thin
- some optional fields are underdeveloped

D. REQUIREMENT NOTES QUALITY
Check requirements sections for depth and usefulness.

Reject if:
- fewer than 5 meaningful sections are populated
- sections are generic filler
- notes simply repeat contract values without adding detail
- confirmeddecisions is empty or meaningless
- the notes do not materially help explain later audit reasoning

Warn if:
- one section is shallow but the rest are strong

E. PLAN QUALITY
The plan must be implementation-grade and auditable.

Reject if:
- the plan is too shallow to support a real audit
- fewer than 10 substantial plan sections are meaningfully populated
- sections contain generic filler rather than architecture detail
- the plan is obviously unrealistic or nonsensical
- the plan openly reveals its own flaws like an answer key
- the entire plan is bad everywhere, making the audit trivial and low-signal
- the flaws are too hidden to support evidence-bearing issueupdates

Warn if:
- the plan is acceptable but one or two sections are weak

F. FLAW TRAINING VALUE
Check whether the flaws are high-value and teach useful Auditor behavior.

Reject if:
- no meaningful flaw exists
- all flaws are trivial
- all flaws are repetitive security clichés without architectural value
- the sample contains only obvious textbook mistakes and no realistic trade-off reasoning
- the supposed primary flaw family in metadata does not actually match the case

Warn if:
- the case is usable but less interesting than the rest of the batch

G. TARGET_OUTPUT SCHEMA AND DEPTH
Check target_output carefully.

Reject if:
- thinkingsummary is missing or overly long and rambling
- strengths has fewer than 2 items or is generic
- concerns has fewer than 2 items or is generic
- recommendations has fewer than 2 items or is vague
- issueupdates has fewer than 2 items unless the case is exceptionally strong and simple
- issueupdates items lack usable detail
- blockingissues includes minor issues that are not truly blockers
- requirementconflicts is populated when no real contract conflict exists
- requirementconflicts is empty even though the case_type and plan clearly create a real requirement conflict

Warn if:
- one list is slightly thin but still usable

H. ISSUE ID CONTINUITY
Check issue ID logic very strictly.

Reject if:
- a reused issue ID does not appear in previousaudits or issueledger when prior history exists
- the same underlying issue is given different IDs without justification
- a materially different issue reuses an old ID incorrectly
- status labels are inconsistent with the evidence
- resolved or downgraded items do not explain what changed
- regression cases fail to distinguish improved issues from newly introduced issues

Warn if:
- issue continuity exists but one status explanation is weaker than ideal

I. ISSUE DETAIL EVIDENCE
Every important issue must be grounded.

Reject if:
- issueupdates.detail does not reference concrete plan section names or clearly identifiable evidence
- detail does not state whether the issue was fixed, partially fixed, unchanged, or newly introduced
- detail does not explain rubric impact when relevant
- detail is vague, generic, or disconnected from the plan
- blockingissues are unsupported by issueupdates
- concerns are unsupported by issueupdates

Warn if:
- one issue detail is acceptable but not strong

J. SUMMARY AND HISTORY CONSISTENCY
Check that the summary matches history and scores.

Reject if:
- summary says improved, stayed flat, or regressed but issueupdates and rubricscores do not support that claim
- a first_pass case talks like a revision round
- a revision_round case looks like a fresh unrelated design
- a regression_case does not contain a real regression
- a requirement_conflict_case does not contain a real contract violation

Warn if:
- the trend language is correct but slightly weak

K. SCORE CALIBRATION
The rubrics must be honest and severity-sensitive.

Reject if:
- scores are obviously inflated relative to the flaws
- scores cluster unrealistically near 8 to 9 despite major weaknesses
- severe unresolved flaws do not noticeably reduce relevant dimensions
- summary and scores tell different stories
- a clearly weak plan receives a strong score profile
- a strong-but-flawed plan is scored unrealistically low without evidence

Warn if:
- one score could reasonably be 0.5 to 1 point different, but the overall profile still makes sense

L. THRESHOLD LEAKAGE
Reject if:
- target_output mentions pass/fail thresholds
- target_output includes any overall score field
- the language suggests the audit is trying to clear a gate rather than judge the plan on its own merits
- the row implicitly teaches threshold chasing

M. REQUIREMENT CONFLICT VALIDITY
If requirementconflicts is non-empty, verify all items.

Reject if:
- the field is not a real field from frozenrequirementcontract
- currentvalue does not match the conflicting plan decision
- proposedvalue does not reflect the locked requirement or necessary correction
- exactreason is vague or not evidence-based
- severity is clearly mismatched

N. BIAS AND STACK FAIRNESS
Reject if:
- the audit criticizes the stack mainly because of stack preference rather than engineering fit
- the sample clearly teaches bias such as “microservices are always better” or “Python/React is always best”
- the criticism is not grounded in risk, requirements, exposure, or design evidence

Warn if:
- the row has mild stack bias but remains mostly evidence-driven

O. DUPLICATION AND BATCH REPETITION
Compare all 10 rows against each other.

Reject rows or the whole batch if:
- two or more rows are near-duplicates in domain, stack, flaw pattern, and critique structure
- the same issue titles, recommendation phrasing, and score patterns are repeated with only renamed technologies
- the batch shows obvious template collapse

Warn if:
- repetition is moderate but not severe

SCORING THE VALIDATION
Use this interpretation when deciding verdicts:
- premium: technically rich, well-grounded, coherent, and safe for final fine-tuning
- usable_with_review: mostly good but needs minor cleanup
- reject: teaches wrong behavior, is too noisy, or is too weak for premium data

CRITICAL FAILURE RULE
A row must be rejected if it has any critical failure from these categories:
- substantive schema break
- unsupported or inconsistent issue continuity
- score-summary-history mismatch
- fake or shallow plan
- fake or unsupported requirement conflict
- threshold leakage
- near-duplicate low-signal repetition
- unsupported blocking issues
- vague or non-evidence-based issue details
- severe stack bias not grounded in engineering fit

VALIDATION OUTPUT RULES
For each sample_results item:
- critical_failures must list only the most important rejection reasons
- warnings must list smaller quality concerns
- minor_repairs_applied must list exactly what was mechanically repaired, if any
- repairable = true only if the row needed only allowed minor repairs
- repaired_row must be:
  - null for accept rows
  - the repaired full row for accept_with_minor_repairs rows
  - null for reject rows
- validation_notes must be concise but specific

BATCH SUMMARY RULES
In batch_summary:
- total_rows must be 10
- accepted_count must equal the number of accept rows
- accepted_with_minor_repairs_count must equal the number of accept_with_minor_repairs rows
- rejected_count must equal the number of reject rows
- overall_quality_assessment must honestly describe whether this batch is premium, mixed, or weak
- most_common_failure_modes must name repeated problems across the batch
- notes should summarize whether the batch is safe to keep

DIVERSITY REPORT RULES
In diversity_report:
- duplicate_or_near_duplicate_pairs should list sample_id pairs when applicable
- projectclass_distribution_ok should be false if variety is poor
- domain_diversity_ok should be false if domains are too repetitive
- flaw_family_diversity_ok should be false if flaws are too repetitive
- stack_bias_warnings should list any batch-level technology favoritism
- repetition_warnings should list repeated phrasing or repeated critique templates

SELF-CHECK BEFORE RETURNING
Before returning the final JSON object, silently verify:
- the response is one JSON object only
- all 10 input rows were evaluated
- approved_rows contains only accepted or minor-repaired rows
- no rejected row appears in approved_rows
- repaired_row is included only for accept_with_minor_repairs
- no forbidden repair was applied
- every verdict is supported by the stated failures or warnings
- the batch verdict matches the actual sample verdict distribution

FINAL OUTPUT RULE
Return EXACTLY one valid JSON object and nothing else.
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_block(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    try:
        return json.loads(text)
    except Exception:
        pass

    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        raise ValueError("No JSON start token found")

    start = min(starts)
    end = max(text.rfind("]"), text.rfind("}"))
    if end <= start:
        raise ValueError("No JSON end token found")

    return json.loads(text[start:end + 1])


def deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def norm_text(x: Any) -> str:
    return " ".join(str(x or "").lower().split())


def stable_fingerprint(row: Dict[str, Any]) -> str:
    parts = {
        "projectclass": deep_get(row, "profile.projectclass", ""),
        "capabilities": deep_get(row, "profile.capabilities", []),
        "risklevel": deep_get(row, "profile.risklevel", ""),
        "datasensitivity": deep_get(row, "profile.datasensitivity", ""),
        "externalexposure": deep_get(row, "profile.externalexposure", ""),
        "case_type": deep_get(row, "metadata.casetype", deep_get(row, "metadata.case_type", "")),
        "primary_flaw_family": deep_get(
            row,
            "metadata.primaryflawfamily",
            deep_get(row, "metadata.primary_flaw_family", ""),
        ),
        "title": deep_get(row, "inputpayload.plan.title", deep_get(row, "input_payload.plan.title", "")),
        "summary": deep_get(row, "targetoutput.summary", deep_get(row, "target_output.summary", "")),
    }
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


@dataclass
class PipelineConfig:
    target_rows: int = 1000
    batch_size: int = 10
    model: str = "gpt-5-mini"
    output_dir: str = "auditor_dataset"
    generation_max_completion_tokens: int = 12000
    audit_max_completion_tokens: int = 3500
    max_api_retries: int = 4
    max_generation_attempts_per_batch: int = 3
    sleep_between_calls_sec: float = 0.2
    resume: bool = True
    seed: int = 42
    llm_audit_every_n_batches: int = 5
    llm_audit_sample_size: int = 10
    heartbeat_seconds: int = 20


class AzureOpenAIJsonClient:
    def __init__(self, model: str, max_api_retries: int):
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZUREOPENAIAPIKEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZUREOPENAIENDPOINT")
        deployment = (
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or os.getenv("AZUREOPENAICHATDEPLOYMENT")
            or model
        )

        if not api_key:
            raise RuntimeError("Missing AZURE_OPENAI_API_KEY or AZUREOPENAIAPIKEY")
        if not endpoint:
            raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT or AZUREOPENAIENDPOINT")

        self.model = deployment
        self.max_api_retries = max_api_retries
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint.rstrip("/") + "/openai/v1",
        )

    def complete_json(
        self,
        system_prompt: str,
        user_payload: Any,
        max_completion_tokens: int,
        response_dir: Path,
        prefix: str,
        logger,
    ) -> Any:
        user_text = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=False)
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_api_retries + 1):
            try:
                logger(f"{prefix}: api attempt {attempt} started")
                start = time.time()

                request_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt + "\n\nReturn only valid JSON."},
                        {"role": "user", "content": user_text},
                    ],
                    "max_completion_tokens": max_completion_tokens,
                    "reasoning_effort": "minimal",
                }

                try:
                    resp = self.client.chat.completions.create(**request_kwargs)
                except Exception as e:
                    msg = str(e).lower()
                    if "reasoning_effort" in msg or "unrecognized request argument" in msg:
                        request_kwargs.pop("reasoning_effort", None)
                        resp = self.client.chat.completions.create(**request_kwargs)
                    else:
                        raise

                elapsed = time.time() - start
                choice = resp.choices[0]
                content = choice.message.content or ""
                finish_reason = getattr(choice, "finish_reason", None)

                usage = None
                usage_obj = getattr(resp, "usage", None)
                if usage_obj:
                    usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else str(usage_obj)

                write_json(
                    response_dir / f"{prefix}_meta_attempt_{attempt}.json",
                    {
                        "finished_at": now_iso(),
                        "elapsed_seconds": round(elapsed, 2),
                        "max_completion_tokens": max_completion_tokens,
                        "finish_reason": finish_reason,
                        "usage": usage,
                    },
                )

                (response_dir / f"{prefix}_raw_attempt_{attempt}.txt").write_text(content, encoding="utf-8")

                if not content.strip():
                    raise RuntimeError(
                        f"Empty model response: finish_reason={finish_reason}, usage={usage}"
                    )

                parsed = extract_json_block(content)
                write_json(response_dir / f"{prefix}_parsed_attempt_{attempt}.json", parsed)

                logger(f"{prefix}: api attempt {attempt} finished in {elapsed:.1f}s")
                return parsed

            except Exception as e:
                last_err = e
                msg = str(e)
                logger(f"{prefix}: api attempt {attempt} failed: {msg}")

                lowered = msg.lower()
                if (
                    "unsupported_parameter" in lowered
                    or "invalid_request_error" in lowered
                    or "unsupported_value" in lowered
                ):
                    raise RuntimeError(f"Permanent API request error: {e}") from e

                time.sleep(min(20, 2 ** attempt + random.random()))

        raise RuntimeError(f"API call failed after retries: {last_err}")



class Heartbeat:
    def __init__(self, interval: int, log_func):
        self.interval = interval
        self.log = log_func
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, label: str):
        def run():
            start = time.time()
            while not self._stop.wait(self.interval):
                self.log(f"{label}: still running after {time.time() - start:.0f}s")

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


class LocalValidator:
    REQUIRED_TOP = [
        "sampleid",
        "dataset",
        "agent",
        "split",
        "profile",
        "inputpayload",
        "targetoutput",
        "metadata",
    ]
    RUBRIC_KEYS = [
        "requirementsalignment",
        "architecturequality",
        "security",
        "operability",
        "internalconsistency",
    ]
    ISSUE_STATUSES = {"unresolved", "resolved", "downgraded", "new"}
    SEVERITIES = {"low", "medium", "high", "critical"}
    CORE_FIELDS = {
        "projectgoal",
        "targetusers",
        "projectclass",
        "capabilities",
        "accessmodel",
        "featurescope",
        "mvpscope",
        "risklevel",
        "datasensitivity",
        "externalexposure",
        "securitybaseline",
    }

    def validate_row(self, row: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []

        for k in self.REQUIRED_TOP:
            if k not in row:
                errors.append(f"missing top-level key {k}")

        if row.get("dataset") != "auditor":
            errors.append("dataset must be auditor")
        if row.get("agent") != "AuditorAgent":
            errors.append("agent must be AuditorAgent")
        if row.get("split") != "train":
            errors.append("split must be train")

        rubric = deep_get(row, "targetoutput.rubricscores", {})
        if not isinstance(rubric, dict):
            errors.append("rubricscores must be object")
        else:
            for key in self.RUBRIC_KEYS:
                value = rubric.get(key)
                if value is None:
                    errors.append(f"missing rubricscores.{key}")
                elif not isinstance(value, (int, float)):
                    errors.append(f"rubricscores.{key} must be numeric")
                elif value < 0 or value > 10:
                    errors.append(f"rubricscores.{key} out of range")

        contract = deep_get(row, "inputpayload.frozenrequirementcontract", {})
        if not isinstance(contract, dict) or len(contract) < 10:
            errors.append("frozenrequirementcontract too thin")
        else:
            missing_core = [k for k in self.CORE_FIELDS if k not in contract]
            if len(missing_core) > 3:
                errors.append("too many core requirement fields missing")

        requirements = deep_get(row, "inputpayload.requirements", {})
        if not isinstance(requirements, dict):
            errors.append("requirements must be object")
        else:
            meaningful_sections = 0
            for section in [
                "project",
                "frontend",
                "backend",
                "security",
                "data",
                "devops",
                "constraints",
                "confirmeddecisions",
            ]:
                value = requirements.get(section)
                if isinstance(value, dict) and len(value) >= 2:
                    meaningful_sections += 1
                elif isinstance(value, list) and len(value) >= 2:
                    meaningful_sections += 1
            if meaningful_sections < 5:
                errors.append("requirements too thin")

        plan = deep_get(row, "inputpayload.plan", {})
        if not isinstance(plan, dict):
            errors.append("plan must be object")
        else:
            substantial_sections = 0
            for section in [
                "title",
                "executivesummary",
                "architectureoverview",
                "technologystack",
                "systemcomponents",
                "workflows",
                "datamodel",
                "apidesign",
                "securityandcompliance",
                "deploymentandoperations",
                "observability",
                "costandscaling",
                "phasedimplementation",
                "developmentguidelines",
                "risksandtradeoffs",
            ]:
                v = plan.get(section)
                if isinstance(v, str) and len(v.strip()) > 60:
                    substantial_sections += 1
                elif isinstance(v, (dict, list)) and len(v) > 0:
                    substantial_sections += 1
            if substantial_sections < 8:
                errors.append("plan too thin")

        out = deep_get(row, "targetoutput", {})
        if not isinstance(out, dict):
            errors.append("targetoutput must be object")
        else:
            if len(out.get("strengths", [])) < 2:
                errors.append("strengths too thin")
            if len(out.get("concerns", [])) < 2:
                errors.append("concerns too thin")
            if len(out.get("recommendations", [])) < 2:
                errors.append("recommendations too thin")

            issueupdates = out.get("issueupdates", [])
            if not isinstance(issueupdates, list) or len(issueupdates) < 2:
                errors.append("issueupdates too thin")
            else:
                known_ids = set(deep_get(row, "inputpayload.issueledger", {}).keys())

                for pa in deep_get(row, "inputpayload.previousaudits", []):
                    if isinstance(pa, dict):
                        for iu in pa.get("issueupdates", []):
                            if isinstance(iu, dict) and iu.get("id"):
                                known_ids.add(str(iu["id"]))

                for iu in issueupdates:
                    if not isinstance(iu, dict):
                        errors.append("issueupdates item must be object")
                        continue
                    if iu.get("status") not in self.ISSUE_STATUSES:
                        errors.append("invalid issue status")
                    if iu.get("severity") not in self.SEVERITIES:
                        errors.append("invalid issue severity")
                    detail = norm_text(iu.get("detail", ""))
                    if len(detail) < 30:
                        errors.append("issue detail too short")
                    if iu.get("status") in {"resolved", "downgraded"} and str(iu.get("id")) not in known_ids:
                        errors.append("resolved/downgraded issue missing prior history")

            text_blob = " ".join(
                [
                    norm_text(out.get("summary", "")),
                    norm_text(out.get("thinkingsummary", "")),
                    norm_text(json.dumps(out.get("recommendations", []), ensure_ascii=False)),
                ]
            )
            if (
                "pass threshold" in text_blob
                or "overall score" in text_blob
                or "approval threshold" in text_blob
            ):
                errors.append("threshold leakage detected")

        req_conflicts = deep_get(row, "targetoutput.requirementconflicts", [])
        if isinstance(req_conflicts, list):
            for rc in req_conflicts:
                if not isinstance(rc, dict):
                    errors.append("requirementconflict item must be object")
                    continue
                if rc.get("field") and rc.get("field") not in contract:
                    errors.append("requirement conflict references non-contract field")
                if rc.get("severity") not in self.SEVERITIES:
                    errors.append("invalid requirement conflict severity")

        return (len(errors) == 0, errors)


class AuditorPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

        if self.config.batch_size != 10:
            raise ValueError(
                "This script must use batch_size=10 because the unchanged generator prompt requires EXACTLY 10 rows."
            )
        if self.config.llm_audit_sample_size != 10:
            raise ValueError(
                "This script must use llm_audit_sample_size=10 because the unchanged validator prompt expects EXACTLY 10 rows."
            )

        random.seed(config.seed)

        self.root = ensure_dir(Path(config.output_dir))
        self.raw_dir = ensure_dir(self.root / "raw_batches")
        self.audit_dir = ensure_dir(self.root / "llm_audits")
        self.logs_dir = ensure_dir(self.root / "logs")
        self.manifest_dir = ensure_dir(self.root / "batch_manifests")

        self.state_path = self.root / "pipeline_state.json"
        self.dataset_json_path = self.root / "auditor_dataset.json"
        self.dataset_jsonl_path = self.root / "auditor_dataset.jsonl"

        self.client = AzureOpenAIJsonClient(config.model, config.max_api_retries)
        self.validator = LocalValidator()

        self.rows: List[Dict[str, Any]] = []
        self.sample_ids: set[str] = set()
        self.fingerprints: set[str] = set()

        self.next_batch_num = 1
        self.total_generated_candidate_rows = 0
        self.total_locally_rejected_rows = 0
        self.total_llm_rejected_rows = 0

        if config.resume and self.state_path.exists():
            self._load_state()

    def _log(self, message: str) -> None:
        line = f"[{now_iso()}] {message}"
        print(line, flush=True)
        with (self.logs_dir / "pipeline.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))

        if self.dataset_json_path.exists():
            self.rows = json.loads(self.dataset_json_path.read_text(encoding="utf-8"))

        self.sample_ids = {str(row.get("sampleid", "")).strip() for row in self.rows}
        self.fingerprints = {stable_fingerprint(row) for row in self.rows}
        self.next_batch_num = int(
            state.get("next_batch_num", len(self.rows) // max(1, self.config.batch_size) + 1)
        )
        self.total_generated_candidate_rows = int(state.get("total_generated_candidate_rows", 0))
        self.total_locally_rejected_rows = int(state.get("total_locally_rejected_rows", 0))
        self.total_llm_rejected_rows = int(state.get("total_llm_rejected_rows", 0))

        self._log(f"Resumed with {len(self.rows)} approved rows already saved")

    def _save_state(self) -> None:
        write_json(self.dataset_json_path, self.rows)
        write_json(
            self.state_path,
            {
                "next_batch_num": self.next_batch_num,
                "total_generated_candidate_rows": self.total_generated_candidate_rows,
                "total_approved_rows": len(self.rows),
                "total_locally_rejected_rows": self.total_locally_rejected_rows,
                "total_llm_rejected_rows": self.total_llm_rejected_rows,
                "updated_at": now_iso(),
                "config": asdict(self.config),
            },
        )

    def _batch_id(self) -> str:
        return f"{self.next_batch_num:03d}"

    def _generate_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        batch_dir = ensure_dir(self.raw_dir / f"batch_{batch_id}")

        # ONLY BATCHID EXISTS IN THE UNCHANGED PROMPT
        prompt = GENERATOR_PROMPT.replace("BATCHID", batch_id)

        payload = {
            "batchid": batch_id,
            "requestedcount": 10,
        }

        hb = Heartbeat(self.config.heartbeat_seconds, self._log)
        hb.start(f"batch {batch_id} generation")
        start = time.time()

        try:
            data = self.client.complete_json(
                prompt,
                payload,
                self.config.generation_max_completion_tokens,
                batch_dir,
                "generation",
                self._log,
            )
        finally:
            hb.stop()

        self._log(f"Batch {batch_id}: generation stage took {time.time() - start:.1f}s")

        if not isinstance(data, list):
            raise ValueError("Generator did not return a JSON array")

        if len(data) != 10:
            raise ValueError(f"Generator returned {len(data)} rows, expected exactly 10")

        write_json(batch_dir / "candidates.json", data)
        return data

    def _apply_local_filters(
        self, batch_id: str, rows: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        approved: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        seen_fp: set[str] = set()
        reject_log: List[Dict[str, Any]] = []

        for row in rows:
            sid = str(row.get("sampleid", "")).strip()
            ok, errors = self.validator.validate_row(row)
            fp = stable_fingerprint(row) if isinstance(row, dict) else ""

            duplicate = (
                sid in self.sample_ids
                or sid in seen_ids
                or fp in self.fingerprints
                or fp in seen_fp
            )

            if not sid:
                ok = False
                errors.append("empty sampleid")

            if duplicate:
                ok = False
                errors.append("duplicate or near-duplicate row")

            if ok:
                approved.append(row)
                seen_ids.add(sid)
                seen_fp.add(fp)
            else:
                rejected.append(row)
                reject_log.append({"sampleid": sid, "errors": errors})

        if reject_log:
            write_json(self.raw_dir / f"batch_{batch_id}" / "local_rejections.json", reject_log)

        return approved, rejected

    def _needs_llm_audit(self, batch_num: int, approved_rows: List[Dict[str, Any]]) -> bool:
        if not approved_rows:
            return False
        return batch_num % self.config.llm_audit_every_n_batches == 0

    def _run_llm_audit(
        self, batch_id: str, rows: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        if len(rows) != 10:
            raise ValueError(
                f"LLM audit requires exactly 10 rows because the unchanged validator prompt expects 10, got {len(rows)}"
            )

        audit_batch_dir = ensure_dir(self.audit_dir / f"batch_{batch_id}")
        hb = Heartbeat(self.config.heartbeat_seconds, self._log)
        hb.start(f"batch {batch_id} llm-audit")
        start = time.time()

        try:
            result = self.client.complete_json(
                LLM_AUDIT_PROMPT,
                rows,
                self.config.audit_max_completion_tokens,
                audit_batch_dir,
                "spot_audit",
                self._log,
            )
        finally:
            hb.stop()

        self._log(f"Batch {batch_id}: llm spot-audit took {time.time() - start:.1f}s")

        if not isinstance(result, dict):
            raise ValueError("LLM audit did not return JSON object")

        write_json(audit_batch_dir / "spot_audit_result.json", result)

        sample_results = result.get("sampleresults", [])
        approved_rows = result.get("approvedrows", [])

        if not isinstance(sample_results, list):
            raise ValueError("LLM audit response missing valid sampleresults list")
        if not isinstance(approved_rows, list):
            raise ValueError("LLM audit response missing valid approvedrows list")

        rejected_ids = [
            str(item.get("sampleid"))
            for item in sample_results
            if isinstance(item, dict) and str(item.get("verdict", "")).lower() == "reject"
        ]

        kept = approved_rows
        return kept, rejected_ids

    def _save_rows(self, batch_id: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        remaining = self.config.target_rows - len(self.rows)
        rows = rows[:remaining]

        self.rows.extend(rows)
        self.sample_ids.update(str(r.get("sampleid", "")).strip() for r in rows)
        self.fingerprints.update(stable_fingerprint(r) for r in rows)

        append_jsonl(self.dataset_jsonl_path, rows)
        write_json(self.raw_dir / f"batch_{batch_id}" / "saved_rows.json", rows)
        self._save_state()

    def _write_manifest(
        self,
        batch_id: str,
        generated: int,
        local_ok: int,
        local_rejected: int,
        llm_rejected_ids: List[str],
        saved: int,
        elapsed: float,
    ) -> None:
        write_json(
            self.manifest_dir / f"batch_{batch_id}.json",
            {
                "batch_id": batch_id,
                "generated_count": generated,
                "local_pass_count": local_ok,
                "local_rejected_count": local_rejected,
                "llm_rejected_ids": llm_rejected_ids,
                "saved_count": saved,
                "cumulative_saved": len(self.rows),
                "elapsed_seconds": round(elapsed, 2),
                "finished_at": now_iso(),
            },
        )

    def run(self) -> None:
        self._log(f"Starting auditor pipeline. Target approved rows: {self.config.target_rows}")

        while len(self.rows) < self.config.target_rows:
            batch_id = self._batch_id()
            batch_num = self.next_batch_num

            self._log(f"Preparing batch {batch_id}. Current approved total: {len(self.rows)}")

            start = time.time()
            saved_rows: List[Dict[str, Any]] = []
            local_rejected = 0
            llm_rejected_ids: List[str] = []
            generated_count = 0

            for gen_attempt in range(1, self.config.max_generation_attempts_per_batch + 1):
                self._log(f"Batch {batch_id}: generation attempt {gen_attempt}")

                try:
                    candidates = self._generate_batch(batch_id)
                    generated_count = len(candidates)
                    self.total_generated_candidate_rows += generated_count

                    approved_rows, rejected_rows = self._apply_local_filters(batch_id, candidates)
                    local_rejected = len(rejected_rows)
                    self.total_locally_rejected_rows += local_rejected

                    self._log(
                        f"Batch {batch_id}: local validator approved {len(approved_rows)} of {generated_count}"
                    )

                    if self._needs_llm_audit(batch_num, approved_rows):
                        if len(approved_rows) != 10:
                            self._log(
                                f"Batch {batch_id}: skipped llm audit because local validator kept {len(approved_rows)} rows, not 10"
                            )
                        else:
                            approved_rows, llm_rejected_ids = self._run_llm_audit(batch_id, approved_rows)
                            self.total_llm_rejected_rows += len(llm_rejected_ids)
                            self._log(
                                f"Batch {batch_id}: llm spot-audit rejected {len(llm_rejected_ids)} rows"
                            )

                    saved_rows = approved_rows
                    break

                except Exception as e:
                    self._log(f"Batch {batch_id}: attempt {gen_attempt} failed: {e}")
                    time.sleep(min(15, 2 ** gen_attempt))

            self._save_rows(batch_id, saved_rows)

            elapsed = time.time() - start
            self._write_manifest(
                batch_id,
                generated_count,
                len(saved_rows),
                local_rejected,
                llm_rejected_ids,
                len(saved_rows),
                elapsed,
            )

            self._log(
                f"Batch {batch_id}: saved {len(saved_rows)} rows in {elapsed:.1f}s; cumulative approved {len(self.rows)}"
            )

            self.next_batch_num += 1
            self._save_state()

            if len(self.rows) >= self.config.target_rows:
                break

            time.sleep(self.config.sleep_between_calls_sec)

        self._log(f"Finished. Final approved rows: {len(self.rows)}")
        self._log(f"JSON file: {self.dataset_json_path.resolve()}")
        self._log(f"JSONL file: {self.dataset_jsonl_path.resolve()}")


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(
        description="Auditor dataset pipeline for Azure OpenAI GPT-5 mini"
    )
    parser.add_argument("--target-rows", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--output-dir", type=str, default="auditor_dataset")
    parser.add_argument("--generation-max-completion-tokens", type=int, default=100000)
    parser.add_argument("--audit-max-completion-tokens", type=int, default=100000)
    parser.add_argument("--max-api-retries", type=int, default=4)
    parser.add_argument("--max-generation-attempts-per-batch", type=int, default=3)
    parser.add_argument("--sleep-between-calls-sec", type=float, default=0.2)
    parser.add_argument("--llm-audit-every-n-batches", type=int, default=5)
    parser.add_argument("--llm-audit-sample-size", type=int, default=10)
    parser.add_argument("--heartbeat-seconds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    if args.batch_size != 10:
        print("[WARN] Overriding --batch-size to 10 to match the unchanged generator prompt.")
        args.batch_size = 10

    if args.llm_audit_sample_size != 10:
        print("[WARN] Overriding --llm-audit-sample-size to 10 to match the unchanged validator prompt.")
        args.llm_audit_sample_size = 10

    return PipelineConfig(
        target_rows=args.target_rows,
        batch_size=args.batch_size,
        model=args.model,
        output_dir=args.output_dir,
        generation_max_completion_tokens=args.generation_max_completion_tokens,
        audit_max_completion_tokens=args.audit_max_completion_tokens,
        max_api_retries=args.max_api_retries,
        max_generation_attempts_per_batch=args.max_generation_attempts_per_batch,
        sleep_between_calls_sec=args.sleep_between_calls_sec,
        resume=not args.no_resume,
        seed=args.seed,
        llm_audit_every_n_batches=args.llm_audit_every_n_batches,
        llm_audit_sample_size=args.llm_audit_sample_size,
        heartbeat_seconds=args.heartbeat_seconds,
    )


def main() -> None:
    cfg = parse_args()
    pipeline = AuditorPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
