# Arkon-Governance-Framework

> **A Validated, Phase-Locked Multi-Agent System for Enforcing Engineering Discipline in AI-Assisted Software Development**
> 
> *By Damika Udantha Saputhanthri — BSc (Hons) Artificial Intelligence and Data Science, Robert Gordon University (April 2026)*

## Overview
This repository contains the official implementation of **Architect & Auditor**, a phase-locked multi-agent AI system designed to enforce software engineering discipline at the architectural planning stage of AI-assisted development. 

The core problem this system addresses is the **Validation Gap**: the widening divide between how easily Large Language Models generate code and how difficult it is to ensure that code belongs to a structurally sound, secure, and maintainable system. 

By enforcing the Software Development Lifecycle (SDLC) through hard phase-locking constraints, the system structurally blocks code generation until a dedicated Critic agent validates and approves the architectural plan produced by a Generator agent.

## Key Features
- **Phase-Locked Governance:** Hard phase transitions managed by a bespoke Python orchestrator to structurally limit agent progression.
- **Generator-Critic Feedback Loop:** An iterative validation cycle between the **Architect Agent** (Llama-3.1-8B) and **Auditor Agent** (Gemma-2-2B).
- **Five-Agent Collaboration:** The orchestrator manages five specialized roles: Architect, Auditor, Tutor, QA Engineer, and Narrative Writer to produce a cohesive development handoff package.
- **Frozen Requirement Contract:** A mechanism to ensure that generated plans remain strictly aligned to immutable baseline requirements.
- **Enterprise-Grade Focus:** Purpose-trained Small Language Models (SLMs) achieved 100% structural compliance, full data privacy, and superior on-premise execution capability compared to zero-shot frontier models.

## Repository Structure

The project is structured into distinct, modular components:

- 📂 **`Agent/`**
  Contains the core Python system sequence orchestrator. It manages the multi-agent logic, tool usage, schema validation, and hard phase-transitions.
- 📂 **`Backend/`**
  The FastAPI application providing endpoints, database integration (SQLAlchemy), and state management for the user interface.
- 📂 **`Frontend/`**
  The Next.js and React 19 web interface utilizing Tailwind CSS and Framer Motion for a robust, interactive user experience.
- 📂 **`Datasets/`**
  Houses the two standalone datasets created using a strict 16-stage validation pipeline:
  - *AuditorAgent Dataset*: 799 annotated (plan, structured-critique) pairs containing 14 project classes.
  - *ArchitectAgent Dataset*: 891 (requirement-contract, implementation-grade-plan) pairs.
- 📂 **`ModelTraining/`**
  Scripts and configurations for the parameter-efficient fine-tuning (QLoRA) of the Small Language Models.
- 📂 **`Models/`**
  Stores the finalized, task-specific tuned SLM weights and adapters.
- 📂 **`AllModelsResults/` & `BaseModelTesting/`**
  Contains the benchmarking data, logs, and comparative evaluation points against external frontier LLMs (like GPT-4.1).
- 📂 **`Expert Approvals/` & `Human Testing/`**
  Qualitative evaluation rubrics, human review scores, and validation approvals supporting the effectiveness of the fine-tuned agents.

## Evaluation Findings
The research and evaluation phase extensively tested eight QLoRA fine-tuned SLMs against zero-shot frontier models. The results revealed that for structured, domain-specific governance pipelines, custom-trained SLMs outperform broad frontier models on critical enterprise dimensions (calibration, structure, and privacy). 
* **Auditor Role:** Gemma-2-2B
* **Architect Role:** Llama-3.1-8B

## Tech Stack Highlights
* **Core Agent Runtime:** Python, OpenAI Integration, Rich, ReportLab
* **Orchestration Backend:** FastAPI, Uvicorn, SQLAlchemy, BCrypt, PyJWT
* **Web Client Frontend:** Next.js (App Router), React, Framer Motion, Tailwind CSS
* **AI Training:** HuggingFace Transformers, PEFT / QLoRA
