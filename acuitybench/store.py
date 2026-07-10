"""Crash-safe SQLite storage for inference and judging runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationStore:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EvaluationStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                manifest_fingerprint TEXT NOT NULL,
                model_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                api_model TEXT NOT NULL,
                model_config_json TEXT NOT NULL,
                benchmark_path TEXT NOT NULL,
                benchmark_sha256 TEXT NOT NULL,
                tasks_json TEXT NOT NULL,
                samples INTEGER NOT NULL,
                selected_cases INTEGER NOT NULL,
                expected_generations INTEGER NOT NULL,
                selection_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS generations (
                run_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                source_id TEXT NOT NULL,
                task_type TEXT NOT NULL CHECK(task_type IN ('qa', 'conv')),
                sample_idx INTEGER NOT NULL,
                normalized_label TEXT,
                split TEXT,
                mapping_method TEXT,
                is_edge_case INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                prompt_sha256 TEXT NOT NULL,
                model_config_sha256 TEXT NOT NULL,
                response TEXT,
                response_sha256 TEXT,
                parsed_label TEXT,
                parse_ok INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error TEXT,
                attempts INTEGER NOT NULL,
                finish_reason TEXT,
                response_id TEXT,
                returned_model TEXT,
                system_fingerprint TEXT,
                request_id TEXT,
                input_tokens INTEGER,
                cached_input_tokens INTEGER,
                cache_write_tokens INTEGER,
                output_tokens INTEGER,
                reasoning_tokens INTEGER,
                total_tokens INTEGER,
                latency_ms REAL,
                rate_limit_json TEXT,
                provider_metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, dataset, source_id, task_type, sample_idx),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_generations_run_status
                ON generations(run_id, task_type, status);

            CREATE TABLE IF NOT EXISTS judgments (
                run_id TEXT NOT NULL,
                case_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                source_id TEXT NOT NULL,
                sample_idx INTEGER NOT NULL,
                judge_id TEXT NOT NULL,
                judge_config_sha256 TEXT NOT NULL,
                generation_response_sha256 TEXT NOT NULL,
                judge_prompt TEXT NOT NULL,
                judge_prompt_sha256 TEXT NOT NULL,
                response TEXT,
                response_sha256 TEXT,
                judge_label TEXT,
                judge_reasoning TEXT,
                parse_ok INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error TEXT,
                attempts INTEGER NOT NULL,
                finish_reason TEXT,
                response_id TEXT,
                returned_model TEXT,
                system_fingerprint TEXT,
                request_id TEXT,
                input_tokens INTEGER,
                cached_input_tokens INTEGER,
                cache_write_tokens INTEGER,
                output_tokens INTEGER,
                reasoning_tokens INTEGER,
                total_tokens INTEGER,
                latency_ms REAL,
                rate_limit_json TEXT,
                provider_metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, dataset, source_id, sample_idx, judge_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_judgments_run_status
                ON judgments(run_id, judge_id, status);
            """
        )
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )
        elif int(row["value"]) != self.SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported evaluation DB schema {row['value']} at {self.path}"
            )
        self.connection.commit()

    def ensure_run(self, manifest: dict[str, Any]) -> bool:
        """Create a run or validate that an existing run has identical identity."""
        existing = self.connection.execute(
            "SELECT manifest_fingerprint FROM runs WHERE run_id=?",
            (manifest["run_id"],),
        ).fetchone()
        if existing:
            if existing["manifest_fingerprint"] != manifest["manifest_fingerprint"]:
                raise ValueError(
                    f"Run ID {manifest['run_id']!r} already exists with a different "
                    "benchmark, model configuration, or selection. Choose another --run-id."
                )
            return False
        now = utc_now()
        self.connection.execute(
            """
            INSERT INTO runs(
                run_id, manifest_fingerprint, model_id, provider, api_model,
                model_config_json, benchmark_path, benchmark_sha256, tasks_json,
                samples, selected_cases, expected_generations, selection_json,
                status, created_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                manifest["run_id"],
                manifest["manifest_fingerprint"],
                manifest["model_id"],
                manifest["provider"],
                manifest["api_model"],
                json.dumps(manifest["model_config"], sort_keys=True),
                manifest["benchmark_path"],
                manifest["benchmark_sha256"],
                json.dumps(manifest["tasks"]),
                manifest["samples"],
                manifest["selected_cases"],
                manifest["expected_generations"],
                json.dumps(manifest["selection"], sort_keys=True),
                "created",
                now,
                now,
            ),
        )
        self.connection.commit()
        return True

    def set_run_status(self, run_id: str, status: str) -> None:
        self.connection.execute(
            "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
            (status, utc_now(), run_id),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Run {run_id!r} does not exist in {self.path}")
        result = dict(row)
        for key in ("model_config_json", "tasks_json", "selection_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        return result

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT run_id, model_id, api_model, status, selected_cases,
                      expected_generations, created_at, updated_at
               FROM runs ORDER BY created_at DESC"""
        ).fetchall()
        return [dict(row) for row in rows]

    def successful_generation_keys(self, run_id: str) -> set[tuple[str, str, str, int]]:
        rows = self.connection.execute(
            """SELECT dataset, source_id, task_type, sample_idx
               FROM generations WHERE run_id=? AND status='ok'""",
            (run_id,),
        ).fetchall()
        return {
            (row["dataset"], row["source_id"], row["task_type"], row["sample_idx"])
            for row in rows
        }

    def upsert_generation(self, row: dict[str, Any]) -> None:
        now = utc_now()
        values = dict(row)
        values.setdefault("created_at", now)
        values["updated_at"] = now
        columns = (
            "run_id", "case_id", "dataset", "source_id", "task_type", "sample_idx",
            "normalized_label", "split", "mapping_method", "is_edge_case", "prompt",
            "prompt_sha256", "model_config_sha256", "response", "response_sha256",
            "parsed_label", "parse_ok", "status", "error", "attempts", "finish_reason",
            "response_id", "returned_model", "system_fingerprint", "request_id",
            "input_tokens", "cached_input_tokens", "cache_write_tokens", "output_tokens",
            "reasoning_tokens", "total_tokens", "latency_ms", "rate_limit_json",
            "provider_metadata_json", "created_at", "updated_at"
        )
        placeholders = ",".join("?" for _ in columns)
        assignments = ",".join(
            f"{name}=excluded.{name}" for name in columns if name not in {"run_id", "dataset", "source_id", "task_type", "sample_idx", "created_at"}
        )
        self.connection.execute(
            f"""INSERT INTO generations({','.join(columns)}) VALUES({placeholders})
                ON CONFLICT(run_id,dataset,source_id,task_type,sample_idx)
                DO UPDATE SET {assignments}""",
            tuple(values.get(name) for name in columns),
        )
        self.connection.commit()

    def completed_conv_generations(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT * FROM generations
               WHERE run_id=? AND task_type='conv' AND status='ok'
               ORDER BY dataset, source_id, sample_idx""",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def judgment_records(self, run_id: str, judge_id: str) -> dict[tuple[str, str, int], dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM judgments WHERE run_id=? AND judge_id=?",
            (run_id, judge_id),
        ).fetchall()
        return {
            (row["dataset"], row["source_id"], row["sample_idx"]): dict(row)
            for row in rows
        }

    def upsert_judgment(self, row: dict[str, Any]) -> None:
        now = utc_now()
        values = dict(row)
        values.setdefault("created_at", now)
        values["updated_at"] = now
        columns = (
            "run_id", "case_id", "dataset", "source_id", "sample_idx", "judge_id",
            "judge_config_sha256", "generation_response_sha256", "judge_prompt",
            "judge_prompt_sha256", "response", "response_sha256", "judge_label",
            "judge_reasoning", "parse_ok", "status", "error", "attempts",
            "finish_reason", "response_id", "returned_model", "system_fingerprint",
            "request_id", "input_tokens", "cached_input_tokens", "cache_write_tokens",
            "output_tokens", "reasoning_tokens", "total_tokens", "latency_ms",
            "rate_limit_json", "provider_metadata_json", "created_at", "updated_at"
        )
        placeholders = ",".join("?" for _ in columns)
        key_columns = {"run_id", "dataset", "source_id", "sample_idx", "judge_id", "created_at"}
        assignments = ",".join(
            f"{name}=excluded.{name}" for name in columns if name not in key_columns
        )
        self.connection.execute(
            f"""INSERT INTO judgments({','.join(columns)}) VALUES({placeholders})
                ON CONFLICT(run_id,dataset,source_id,sample_idx,judge_id)
                DO UPDATE SET {assignments}""",
            tuple(values.get(name) for name in columns),
        )
        self.connection.commit()

    def count_statuses(self, table: str, run_id: str) -> dict[str, int]:
        if table not in {"generations", "judgments"}:
            raise ValueError(table)
        rows = self.connection.execute(
            f"SELECT status, COUNT(*) AS n FROM {table} WHERE run_id=? GROUP BY status",
            (run_id,),
        ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    def dataframe_rows(self, query: str, parameters: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(query, tuple(parameters))]
