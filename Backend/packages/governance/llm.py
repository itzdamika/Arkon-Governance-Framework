"""Azure OpenAI client — same behavior as new.py AzureLLM."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .helpers import safe_json_loads

_DEFAULT_DEPLOYMENT = "gpt-5-chat"


def _azure_openai_error_hint(exc: Exception) -> str:
    msg = str(exc)
    if "404" in msg or "Resource not found" in msg:
        dep = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", _DEFAULT_DEPLOYMENT)
        ep = os.getenv("AZURE_OPENAI_ENDPOINT", "https://palceholder.openai.azure.com/s")
        return (
            f"{msg}\n\n"
            "Azure returned 404 — the deployment name usually does not exist on this resource.\n"
            f"- Set AZURE_OPENAI_CHAT_DEPLOYMENT to the exact name from Azure AI Studio / Foundry (currently {dep!r}).\n"
            f"- Set AZURE_OPENAI_ENDPOINT to https://<resource>.openai.azure.com with no extra path (currently {ep!r}).\n"
            "- Default in code is 'gpt-4o' if unset; override in .env if you use another deployment."
        )
    return msg


def _normalize_endpoint(raw: str) -> str:
    ep = (raw or "").strip().rstrip("/")
    if ep.endswith("/openai"):
        ep = ep[: -len("/openai")].rstrip("/")
    # common typo: ...azure.com/s
    if ep.endswith("/s") and ".azure.com" in ep:
        ep = ep[:-2].rstrip("/")
    return ep


class AzureLLM:
    def __init__(self) -> None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "AZURE_OPEN_AI_KEY")
        endpoint = _normalize_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT", "https://palceholder.openai.azure.com/s"))
        chat_deployment = os.getenv(
            "AZURE_OPENAI_CHAT_DEPLOYMENT", _DEFAULT_DEPLOYMENT
        )
        reasoning_deployment = os.getenv(
            "AZURE_OPENAI_REASONING_DEPLOYMENT", chat_deployment
        )

        if not api_key:
            raise RuntimeError("Missing AZURE_OPENAI_API_KEY")
        if not endpoint:
            raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT")
        if not chat_deployment:
            raise RuntimeError("Missing AZURE_OPENAI_CHAT_DEPLOYMENT")

        self.chat_deployment = chat_deployment
        self.reasoning_deployment = reasoning_deployment
        base = endpoint + "/openai/v1/"
        self.client = OpenAI(
            api_key=api_key,
            base_url=base,
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
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(_azure_openai_error_hint(e)) from e

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
