from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open('a', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def extract_json_block(text: str) -> Any:
    text = (text or '').strip()
    if not text:
        raise ValueError('Empty model response')
    try:
        return json.loads(text)
    except Exception:
        pass
    start_candidates = [i for i in [text.find('['), text.find('{')] if i != -1]
    if not start_candidates:
        raise ValueError('No JSON start token found')
    start = min(start_candidates)
    end_array = text.rfind(']')
    end_obj = text.rfind('}')
    end = max(end_array, end_obj)
    if end == -1 or end <= start:
        raise ValueError('No JSON end token found')
    candidate = text[start:end + 1]
    return json.loads(candidate)


def deep_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def stable_fingerprint(row: Dict[str, Any]) -> str:
    parts = {
        'projectclass': deep_get(row, 'profile.projectclass', ''),
        'capabilities': deep_get(row, 'profile.capabilities', []),
        'risklevel': deep_get(row, 'profile.risklevel', ''),
        'datasensitivity': deep_get(row, 'profile.datasensitivity', ''),
        'externalexposure': deep_get(row, 'profile.externalexposure', ''),
        'case_type': deep_get(row, 'metadata.case_type', ''),
        'primary_flaw_family': deep_get(row, 'metadata.primary_flaw_family', ''),
        'title': deep_get(row, 'input_payload.plan.title', ''),
        'summary': deep_get(row, 'target_output.summary', ''),
    }
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def validate_top_level_row_shape(row: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required = ['sample_id', 'dataset', 'agent', 'split', 'profile', 'input_payload', 'target_output', 'metadata']
    for key in required:
        if key not in row:
            errors.append(f'missing top-level key: {key}')
    if row.get('dataset') != 'auditor':
        errors.append('dataset must be auditor')
    if row.get('agent') != 'AuditorAgent':
        errors.append('agent must be AuditorAgent')
    if row.get('split') != 'train':
        errors.append('split must be train')
    rubric = deep_get(row, 'target_output.rubricscores', {})
    for k in ['requirementsalignment', 'architecturequality', 'security', 'operability', 'internalconsistency']:
        if k not in rubric:
            errors.append(f'missing rubricscores.{k}')
    return errors


@dataclass
class PipelineConfig:
    target_rows: int = 1000
    batch_size: int = 10
    model: str = 'gpt-5-mini'
    output_dir: str = 'auditor_dataset_run'
    max_api_retries: int = 6
    max_generation_attempts_per_batch: int = 4
    temperature_generation: float = 0.4
    temperature_validation: float = 0.0
    generation_max_tokens: int = 12000
    validation_max_tokens: int = 8000
    sleep_between_calls_sec: float = 1.5
    resume: bool = True
    seed: int = 42


class AzureOpenAIJsonClient:
    def __init__(self, model: str):
        api_key = os.getenv('AZURE_OPENAI_API_KEY') or os.getenv('AZUREOPENAIAPIKEY')
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT') or os.getenv('AZUREOPENAIENDPOINT')
        deployment = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT') or os.getenv('AZUREOPENAICHATDEPLOYMENT') or model
        if not api_key:
            raise RuntimeError('Missing AZURE_OPENAI_API_KEY or AZUREOPENAIAPIKEY')
        if not endpoint:
            raise RuntimeError('Missing AZURE_OPENAI_ENDPOINT or AZUREOPENAIENDPOINT')
        self.model = deployment
        self.client = OpenAI(api_key=api_key, base_url=endpoint.rstrip('/') + '/openai/v1')

    def complete_json(self, system_prompt: str, user_payload: Any, temperature: float, max_tokens: int, response_dir: Path, prefix: str) -> Any:
        user_text = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=False)
        last_err: Optional[Exception] = None

        for attempt in range(1, 7):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': system_prompt + '\n\nReturn only valid JSON.'},
                        {'role': 'user', 'content': user_text},
                    ],
                    max_completion_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ''
                raw_path = response_dir / f'{prefix}_raw_attempt_{attempt}.txt'
                raw_path.write_text(content, encoding='utf-8')
                parsed = extract_json_block(content)
                parsed_path = response_dir / f'{prefix}_parsed_attempt_{attempt}.json'
                write_json(parsed_path, parsed)
                return parsed

            except Exception as e:
                last_err = e
                msg = str(e)

                if 'unsupported_parameter' in msg or 'invalid_request_error' in msg:
                    raise RuntimeError(f'Permanent API request error: {e}') from e

                wait = min(60, 2 ** attempt + random.random())
                time.sleep(wait)

        raise RuntimeError(f'API call failed after retries: {last_err}')



class AuditorDatasetPipeline:
    def __init__(self, config: PipelineConfig, generator_prompt_path: Path, validator_prompt_path: Path):
        self.config = config
        random.seed(config.seed)
        self.root = ensure_dir(Path(config.output_dir))
        self.prompts_dir = ensure_dir(self.root / 'prompts')
        self.raw_dir = ensure_dir(self.root / 'raw_batches')
        self.validated_dir = ensure_dir(self.root / 'validated_batches')
        self.logs_dir = ensure_dir(self.root / 'logs')
        self.state_path = self.root / 'pipeline_state.json'
        self.dataset_json_path = self.root / 'auditor_dataset.json'
        self.dataset_jsonl_path = self.root / 'auditor_dataset.jsonl'
        self.generator_prompt_template = load_text(generator_prompt_path)
        self.validator_prompt = load_text(validator_prompt_path)
        (self.prompts_dir / 'generator_prompt.txt').write_text(self.generator_prompt_template, encoding='utf-8')
        (self.prompts_dir / 'validator_prompt.txt').write_text(self.validator_prompt, encoding='utf-8')
        self.client = AzureOpenAIJsonClient(config.model)
        self.rows: List[Dict[str, Any]] = []
        self.sample_ids: set[str] = set()
        self.fingerprints: set[str] = set()
        self.next_batch_num = 1
        self.total_generated_candidate_rows = 0
        self.total_approved_rows = 0
        self.total_rejected_rows = 0
        if config.resume and self.state_path.exists():
            self._load_state()

    def _log(self, message: str) -> None:
        line = f'[{now_iso()}] {message}'
        print(line, flush=True)
        with (self.logs_dir / 'pipeline.log').open('a', encoding='utf-8') as f:
            f.write(line + '\n')

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text(encoding='utf-8'))
        self.rows = json.loads(self.dataset_json_path.read_text(encoding='utf-8')) if self.dataset_json_path.exists() else []
        self.sample_ids = {row.get('sample_id', '') for row in self.rows}
        self.fingerprints = {stable_fingerprint(row) for row in self.rows}
        self.next_batch_num = int(state.get('next_batch_num', len(self.rows) // max(1, self.config.batch_size) + 1))
        self.total_generated_candidate_rows = int(state.get('total_generated_candidate_rows', 0))
        self.total_approved_rows = len(self.rows)
        self.total_rejected_rows = int(state.get('total_rejected_rows', 0))
        self._log(f'Resumed with {self.total_approved_rows} approved rows already saved.')

    def _save_state(self) -> None:
        write_json(self.dataset_json_path, self.rows)
        state = {
            'next_batch_num': self.next_batch_num,
            'total_generated_candidate_rows': self.total_generated_candidate_rows,
            'total_approved_rows': len(self.rows),
            'total_rejected_rows': self.total_rejected_rows,
            'updated_at': now_iso(),
            'config': asdict(self.config),
        }
        write_json(self.state_path, state)

    def _batch_id(self) -> str:
        return f'{self.next_batch_num:03d}'

    def _generator_prompt_for_batch(self, batch_id: str) -> str:
        prompt = self.generator_prompt_template
        prompt = prompt.replace('{{BATCH_ID}}', batch_id)
        return prompt

    def _generate_candidate_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        batch_dir = ensure_dir(self.raw_dir / f'batch_{batch_id}')
        system_prompt = self._generator_prompt_for_batch(batch_id)
        user_payload = {
            'batch_id': batch_id,
            'requested_count': self.config.batch_size,
            'note': 'Generate the next candidate batch now.'
        }
        data = self.client.complete_json(
            system_prompt=system_prompt,
            user_payload=user_payload,
            temperature=self.config.temperature_generation,
            max_tokens=self.config.generation_max_tokens,
            response_dir=batch_dir,
            prefix='generation'
        )
        if not isinstance(data, list):
            raise ValueError('Generator did not return a JSON array')
        return data

    def _validate_batch(self, batch_id: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_dir = ensure_dir(self.validated_dir / f'batch_{batch_id}')
        payload_text = json.dumps(candidates, ensure_ascii=False, indent=2)
        result = self.client.complete_json(
            system_prompt=self.validator_prompt,
            user_payload=payload_text,
            temperature=self.config.temperature_validation,
            max_tokens=self.config.validation_max_tokens,
            response_dir=batch_dir,
            prefix='validation'
        )
        if not isinstance(result, dict):
            raise ValueError('Validator did not return a JSON object')
        write_json(batch_dir / 'validator_result.json', result)
        return result

    def _apply_local_filters(self, rows: List[Dict[str, Any]], batch_id: str) -> List[Dict[str, Any]]:
        approved: List[Dict[str, Any]] = []
        batch_seen_ids: set[str] = set()
        batch_seen_fps: set[str] = set()
        for row in rows:
            sid = str(row.get('sample_id', '')).strip()
            if not sid:
                self._log(f'Batch {batch_id}: dropped row with empty sample_id')
                continue
            errors = validate_top_level_row_shape(row)
            if errors:
                self._log(f'Batch {batch_id}: dropped {sid} after local schema check: {errors}')
                continue
            fp = stable_fingerprint(row)
            if sid in self.sample_ids or sid in batch_seen_ids:
                self._log(f'Batch {batch_id}: dropped duplicate sample_id {sid}')
                continue
            if fp in self.fingerprints or fp in batch_seen_fps:
                self._log(f'Batch {batch_id}: dropped near-duplicate row {sid}')
                continue
            approved.append(row)
            batch_seen_ids.add(sid)
            batch_seen_fps.add(fp)
        return approved

    def _extract_approved_rows(self, validator_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = validator_result.get('approved_rows', [])
        return rows if isinstance(rows, list) else []

    def _save_approved_rows(self, approved_rows: List[Dict[str, Any]], batch_id: str) -> None:
        if not approved_rows:
            return
        remaining = self.config.target_rows - len(self.rows)
        approved_rows = approved_rows[:remaining]
        self.rows.extend(approved_rows)
        self.sample_ids.update(str(r.get('sample_id', '')).strip() for r in approved_rows)
        self.fingerprints.update(stable_fingerprint(r) for r in approved_rows)
        append_jsonl(self.dataset_jsonl_path, approved_rows)
        write_json(self.validated_dir / f'batch_{batch_id}' / 'approved_rows.json', approved_rows)
        self._save_state()

    def run(self) -> None:
        self._log(f'Starting pipeline. Target approved rows: {self.config.target_rows}')
        while len(self.rows) < self.config.target_rows:
            batch_id = self._batch_id()
            self._log(f'Preparing batch {batch_id}. Current approved total: {len(self.rows)}')
            approved_this_round: List[Dict[str, Any]] = []
            for gen_attempt in range(1, self.config.max_generation_attempts_per_batch + 1):
                self._log(f'Batch {batch_id}: generation attempt {gen_attempt}')
                try:
                    candidates = self._generate_candidate_batch(batch_id)
                    self.total_generated_candidate_rows += len(candidates)
                    write_json(self.raw_dir / f'batch_{batch_id}' / 'candidates.json', candidates)
                    validator_result = self._validate_batch(batch_id, candidates)
                    approved_rows = self._extract_approved_rows(validator_result)
                    locally_filtered = self._apply_local_filters(approved_rows, batch_id)
                    rejected_here = max(0, len(candidates) - len(locally_filtered))
                    self.total_rejected_rows += rejected_here
                    approved_this_round = locally_filtered
                    self._log(
                        f'Batch {batch_id}: generated={len(candidates)} validator_approved={len(approved_rows)} '
                        f'locally_saved={len(locally_filtered)}'
                    )
                    break
                except Exception as e:
                    self._log(f'Batch {batch_id}: attempt {gen_attempt} failed: {e}')
                    time.sleep(min(20, 2 ** gen_attempt))
            if approved_this_round:
                self._save_approved_rows(approved_this_round, batch_id)
                self._log(f'Batch {batch_id}: cumulative approved rows now {len(self.rows)}')
            else:
                self._log(f'Batch {batch_id}: no approved rows saved')
                self._save_state()
            self.next_batch_num += 1
            self._save_state()
            if len(self.rows) >= self.config.target_rows:
                break
            time.sleep(self.config.sleep_between_calls_sec)
        self._log(f'Finished. Final approved rows: {len(self.rows)}')
        self._log(f'JSON file: {self.dataset_json_path.resolve()}')
        self._log(f'JSONL file: {self.dataset_jsonl_path.resolve()}')


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description='Auditor dataset generation pipeline for Azure OpenAI GPT-5 mini.')
    parser.add_argument('--target-rows', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--model', type=str, default='gpt-5-mini')
    parser.add_argument('--output-dir', type=str, default='auditor_dataset_run')
    parser.add_argument('--max-api-retries', type=int, default=6)
    parser.add_argument('--max-generation-attempts-per-batch', type=int, default=4)
    parser.add_argument('--temperature-generation', type=float, default=0.4)
    parser.add_argument('--temperature-validation', type=float, default=0.0)
    parser.add_argument('--generation-max-tokens', type=int, default=12000)
    parser.add_argument('--validation-max-tokens', type=int, default=8000)
    parser.add_argument('--sleep-between-calls-sec', type=float, default=1.5)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return PipelineConfig(
        target_rows=args.target_rows,
        batch_size=args.batch_size,
        model=args.model,
        output_dir=args.output_dir,
        max_api_retries=args.max_api_retries,
        max_generation_attempts_per_batch=args.max_generation_attempts_per_batch,
        temperature_generation=args.temperature_generation,
        temperature_validation=args.temperature_validation,
        generation_max_tokens=args.generation_max_tokens,
        validation_max_tokens=args.validation_max_tokens,
        sleep_between_calls_sec=args.sleep_between_calls_sec,
        resume=not args.no_resume,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    root = Path(__file__).resolve().parent
    pipeline = AuditorDatasetPipeline(
        config=cfg,
        generator_prompt_path=root / 'generator_prompt.txt',
        validator_prompt_path=root / 'validator_prompt.txt',
    )
    pipeline.run()


if __name__ == '__main__':
    main()
