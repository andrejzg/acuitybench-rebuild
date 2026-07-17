"""Crash-safe SQLite storage for inference, timing, and judging runs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_TIMING_COLUMNS: tuple[tuple[str, str], ...] = (
    ("latest_execution_id", "TEXT"),
    ("timing_version", "INTEGER"),
    ("timing_source", "TEXT"),
    ("queued_at", "TEXT"),
    ("request_started_at", "TEXT"),
    ("first_token_at", "TEXT"),
    ("response_completed_at", "TEXT"),
    ("queue_wait_ms", "REAL"),
    ("request_wall_ms", "REAL"),
    ("request_wall_total_ms", "REAL"),
    ("service_latency_ms", "REAL"),
    ("retry_backoff_ms", "REAL"),
    ("first_event_ms", "REAL"),
    ("ttft_ms", "REAL"),
    ("time_after_first_token_ms", "REAL"),
    ("total_duration_ms", "REAL"),
    ("server_processing_ms", "REAL"),
)

_REQUIRED_V2_COLUMNS: dict[str, set[str]] = {
    "runs": set(
        """run_id manifest_fingerprint model_id provider api_model
        model_config_json benchmark_path benchmark_sha256 tasks_json samples
        selected_cases expected_generations selection_json execution_config_json
        experiment_contract_json
        status created_at updated_at""".split()
    ),
    "run_executions": set(
        """execution_id run_id phase profile_id judge_id provider api_model
        endpoint config_sha256 configured_concurrency streaming timing_version
        sdk_retries max_attempts retry_policy_json runner_metadata_json task_count
        cache_hit_count pending_count success_count failure_count cancelled_count
        unpersisted_count status started_at ended_at""".split()
    ),
    "generations": set(
        """run_id case_id dataset source_id task_type sample_idx normalized_label
        split mapping_method is_edge_case prompt prompt_sha256 model_config_sha256
        response response_sha256 parsed_label parse_ok status error attempts
        finish_reason response_id returned_model system_fingerprint request_id
        input_tokens cached_input_tokens cache_write_tokens output_tokens
        reasoning_tokens total_tokens latency_ms rate_limit_json
        provider_metadata_json created_at updated_at""".split()
    )
    | {name for name, _ in _TIMING_COLUMNS},
    "judgments": set(
        """run_id case_id dataset source_id sample_idx judge_id
        judge_config_sha256 generation_response_sha256 judge_prompt
        judge_prompt_sha256 response response_sha256 judge_label judge_reasoning
        parse_ok status error attempts finish_reason response_id returned_model
        system_fingerprint request_id input_tokens cached_input_tokens
        cache_write_tokens output_tokens reasoning_tokens total_tokens latency_ms
        rate_limit_json provider_metadata_json created_at updated_at""".split()
    )
    | {name for name, _ in _TIMING_COLUMNS},
    "request_attempts": set(
        """attempt_id execution_id request_key attempt_index dataset source_id
        task_type sample_idx judge_id max_output_tokens eligible_at
        request_started_at first_token_at request_finished_at queue_wait_ms
        request_wall_ms attempt_wall_ms first_event_ms ttft_ms
        time_after_first_token_ms server_processing_ms backoff_planned_ms
        backoff_actual_ms backoff_source outcome retryable http_status error_type
        error partial_response finish_reason request_id response_id returned_model input_tokens
        cached_input_tokens cache_write_tokens output_tokens reasoning_tokens
        total_tokens rate_limit_json provider_metadata_json created_at
        updated_at""".split()
    ),
}


class EvaluationStore:
    SCHEMA_VERSION = 2
    TIMING_VERSION = 2

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self._create_or_migrate_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EvaluationStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _table_exists(self, table: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None

    def _column_names(self, table: str) -> set[str]:
        return {
            str(row["name"])
            for row in self.connection.execute(f"PRAGMA table_info({table})")
        }

    @staticmethod
    def _timing_ddl() -> str:
        return ",\n                ".join(
            f"{name} {declaration}" for name, declaration in _TIMING_COLUMNS
        )

    def _create_or_migrate_schema(self) -> None:
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            core_tables = {"runs", "generations", "judgments"}
            existing_core = {
                table for table in core_tables if self._table_exists(table)
            }
            looks_like_v1 = existing_core == core_tables and (
                "execution_config_json" not in self._column_names("runs")
            )
            if looks_like_v1:
                self.connection.execute(
                    "INSERT INTO metadata(key,value) VALUES('schema_version','1')"
                )
                self.connection.commit()
                self._migrate_v1_to_v2()
            elif not existing_core or (
                "runs" in existing_core
                and "execution_config_json" in self._column_names("runs")
            ):
                # CREATE IF NOT EXISTS safely resumes an interrupted fresh-v2
                # initialization when the version marker was not yet written.
                self._create_v2_schema()
            else:
                raise RuntimeError(
                    f"Incomplete unversioned evaluation schema at {self.path}"
                )
            return

        version = int(row["value"])
        if version == 1:
            self._migrate_v1_to_v2()
        elif version == self.SCHEMA_VERSION:
            # CREATE IF NOT EXISTS also repairs missing indexes after an interrupted
            # manual copy without mutating existing rows.
            self._create_v2_schema(set_version=False)
        else:
            raise RuntimeError(
                f"Unsupported evaluation DB schema {version} at {self.path}"
            )

    def _repair_v2_additive_columns(self) -> None:
        if self._table_exists("runs"):
            run_columns = self._column_names("runs")
            if "execution_config_json" not in run_columns:
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN execution_config_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )
            if "experiment_contract_json" not in run_columns:
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN experiment_contract_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )
        for table in ("generations", "judgments"):
            if not self._table_exists(table):
                continue
            columns = self._column_names(table)
            for name, declaration in _TIMING_COLUMNS:
                if name not in columns:
                    self.connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                    )
        if self._table_exists("run_executions"):
            execution_columns = self._column_names("run_executions")
            for name in ("cancelled_count", "unpersisted_count"):
                if name not in execution_columns:
                    self.connection.execute(
                        f"ALTER TABLE run_executions ADD COLUMN {name} "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
        if self._table_exists("request_attempts"):
            attempt_columns = self._column_names("request_attempts")
            if "time_after_first_token_ms" not in attempt_columns:
                self.connection.execute(
                    "ALTER TABLE request_attempts "
                    "ADD COLUMN time_after_first_token_ms REAL"
                )
            if "partial_response" not in attempt_columns:
                self.connection.execute(
                    "ALTER TABLE request_attempts ADD COLUMN partial_response TEXT"
                )

    def _validate_v2_schema(self) -> None:
        for table, required in _REQUIRED_V2_COLUMNS.items():
            if not self._table_exists(table):
                raise RuntimeError(
                    f"Schema v2 is missing required table {table!r} at {self.path}"
                )
            missing = sorted(required - self._column_names(table))
            if missing:
                raise RuntimeError(
                    f"Schema v2 table {table!r} is missing required columns "
                    f"at {self.path}: {missing}"
                )

    def _create_v2_schema(self, *, set_version: bool = True) -> None:
        # Repair additive columns before index creation, since an interrupted
        # prior migration may have left an existing table without an indexed
        # timing column such as latest_execution_id.
        self._repair_v2_additive_columns()
        timing = self._timing_ddl()
        self.connection.executescript(
            f"""
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
                execution_config_json TEXT NOT NULL DEFAULT '{{}}',
                experiment_contract_json TEXT NOT NULL DEFAULT '{{}}',
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_executions (
                execution_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                phase TEXT NOT NULL CHECK(phase IN ('generation','judge')),
                profile_id TEXT NOT NULL,
                judge_id TEXT,
                provider TEXT NOT NULL,
                api_model TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                config_sha256 TEXT NOT NULL,
                configured_concurrency INTEGER NOT NULL CHECK(configured_concurrency > 0),
                streaming INTEGER NOT NULL,
                timing_version INTEGER NOT NULL,
                sdk_retries INTEGER NOT NULL,
                max_attempts INTEGER NOT NULL,
                retry_policy_json TEXT NOT NULL,
                runner_metadata_json TEXT NOT NULL,
                task_count INTEGER NOT NULL,
                cache_hit_count INTEGER NOT NULL,
                pending_count INTEGER NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                cancelled_count INTEGER NOT NULL DEFAULT 0,
                unpersisted_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_run_executions_run_phase
                ON run_executions(run_id, phase, started_at);

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
                {timing},
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, dataset, source_id, task_type, sample_idx),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_generations_run_status
                ON generations(run_id, task_type, status);
            CREATE INDEX IF NOT EXISTS idx_generations_execution
                ON generations(latest_execution_id);

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
                {timing},
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, dataset, source_id, sample_idx, judge_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_judgments_run_status
                ON judgments(run_id, judge_id, status);
            CREATE INDEX IF NOT EXISTS idx_judgments_execution
                ON judgments(latest_execution_id);

            CREATE TABLE IF NOT EXISTS request_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL,
                request_key TEXT NOT NULL,
                attempt_index INTEGER NOT NULL CHECK(attempt_index >= 1),
                dataset TEXT NOT NULL,
                source_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                sample_idx INTEGER NOT NULL,
                judge_id TEXT,
                max_output_tokens INTEGER NOT NULL,
                eligible_at TEXT NOT NULL,
                request_started_at TEXT,
                first_token_at TEXT,
                request_finished_at TEXT,
                queue_wait_ms REAL,
                request_wall_ms REAL,
                attempt_wall_ms REAL,
                first_event_ms REAL,
                ttft_ms REAL,
                time_after_first_token_ms REAL,
                server_processing_ms REAL,
                backoff_planned_ms REAL,
                backoff_actual_ms REAL,
                backoff_source TEXT,
                outcome TEXT NOT NULL,
                retryable INTEGER NOT NULL,
                http_status INTEGER,
                error_type TEXT,
                error TEXT,
                partial_response TEXT,
                finish_reason TEXT,
                request_id TEXT,
                response_id TEXT,
                returned_model TEXT,
                input_tokens INTEGER,
                cached_input_tokens INTEGER,
                cache_write_tokens INTEGER,
                output_tokens INTEGER,
                reasoning_tokens INTEGER,
                total_tokens INTEGER,
                rate_limit_json TEXT,
                provider_metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (execution_id, request_key, attempt_index),
                FOREIGN KEY (execution_id)
                    REFERENCES run_executions(execution_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_request_attempts_execution
                ON request_attempts(execution_id, request_key, attempt_index);
            """
        )
        self._repair_v2_additive_columns()
        self._validate_v2_schema()
        if set_version:
            self.connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)",
                (str(self.SCHEMA_VERSION),),
            )
            self.connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
        self.connection.commit()

    def _migrate_v1_to_v2(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            run_columns = self._column_names("runs")
            if "execution_config_json" not in run_columns:
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN execution_config_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "experiment_contract_json" not in run_columns:
                self.connection.execute(
                    "ALTER TABLE runs ADD COLUMN experiment_contract_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )
            for table in ("generations", "judgments"):
                existing = self._column_names(table)
                for name, declaration in _TIMING_COLUMNS:
                    if name not in existing:
                        self.connection.execute(
                            f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                        )
                self.connection.execute(
                    f"""UPDATE {table}
                        SET timing_version=COALESCE(timing_version,1),
                            timing_source=COALESCE(timing_source,'legacy_aggregate'),
                            total_duration_ms=COALESCE(total_duration_ms,latency_ms)"""
                )

            self._create_execution_tables()
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_generations_execution ON generations(latest_execution_id)"
            )
            self.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_judgments_execution ON judgments(latest_execution_id)"
            )
            for table in ("generations", "judgments"):
                rows = self.connection.execute(
                    f"""SELECT rowid, provider_metadata_json FROM {table}
                        WHERE server_processing_ms IS NULL
                          AND provider_metadata_json IS NOT NULL"""
                ).fetchall()
                for row in rows:
                    try:
                        metadata = json.loads(row["provider_metadata_json"])
                        value = metadata.get("openai_processing_ms")
                        if value is None:
                            value = metadata.get("server_headers", {}).get(
                                "openai-processing-ms"
                            )
                        processing_ms = float(value)
                    except (TypeError, ValueError, json.JSONDecodeError, AttributeError):
                        continue
                    self.connection.execute(
                        f"UPDATE {table} SET server_processing_ms=? WHERE rowid=?",
                        (processing_ms, row["rowid"]),
                    )
            self.connection.execute(
                "UPDATE metadata SET value=? WHERE key='schema_version'",
                (str(self.SCHEMA_VERSION),),
            )
            self.connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _create_execution_tables(self) -> None:
        """Create v2 audit tables inside the caller's current transaction."""
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS run_executions (
                execution_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                phase TEXT NOT NULL CHECK(phase IN ('generation','judge')),
                profile_id TEXT NOT NULL,
                judge_id TEXT,
                provider TEXT NOT NULL,
                api_model TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                config_sha256 TEXT NOT NULL,
                configured_concurrency INTEGER NOT NULL CHECK(configured_concurrency > 0),
                streaming INTEGER NOT NULL,
                timing_version INTEGER NOT NULL,
                sdk_retries INTEGER NOT NULL,
                max_attempts INTEGER NOT NULL,
                retry_policy_json TEXT NOT NULL,
                runner_metadata_json TEXT NOT NULL,
                task_count INTEGER NOT NULL,
                cache_hit_count INTEGER NOT NULL,
                pending_count INTEGER NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                cancelled_count INTEGER NOT NULL DEFAULT 0,
                unpersisted_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            )"""
        )
        self.connection.execute(
            """CREATE INDEX IF NOT EXISTS idx_run_executions_run_phase
               ON run_executions(run_id, phase, started_at)"""
        )
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS request_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT NOT NULL,
                request_key TEXT NOT NULL,
                attempt_index INTEGER NOT NULL CHECK(attempt_index >= 1),
                dataset TEXT NOT NULL,
                source_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                sample_idx INTEGER NOT NULL,
                judge_id TEXT,
                max_output_tokens INTEGER NOT NULL,
                eligible_at TEXT NOT NULL,
                request_started_at TEXT,
                first_token_at TEXT,
                request_finished_at TEXT,
                queue_wait_ms REAL,
                request_wall_ms REAL,
                attempt_wall_ms REAL,
                first_event_ms REAL,
                ttft_ms REAL,
                time_after_first_token_ms REAL,
                server_processing_ms REAL,
                backoff_planned_ms REAL,
                backoff_actual_ms REAL,
                backoff_source TEXT,
                outcome TEXT NOT NULL,
                retryable INTEGER NOT NULL,
                http_status INTEGER,
                error_type TEXT,
                error TEXT,
                partial_response TEXT,
                finish_reason TEXT,
                request_id TEXT,
                response_id TEXT,
                returned_model TEXT,
                input_tokens INTEGER,
                cached_input_tokens INTEGER,
                cache_write_tokens INTEGER,
                output_tokens INTEGER,
                reasoning_tokens INTEGER,
                total_tokens INTEGER,
                rate_limit_json TEXT,
                provider_metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (execution_id, request_key, attempt_index),
                FOREIGN KEY (execution_id)
                    REFERENCES run_executions(execution_id) ON DELETE CASCADE
            )"""
        )
        self.connection.execute(
            """CREATE INDEX IF NOT EXISTS idx_request_attempts_execution
               ON request_attempts(execution_id, request_key, attempt_index)"""
        )

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
                    "benchmark, model configuration, selection, or transport. "
                    "Choose another --run-id."
                )
            return False
        now = utc_now()
        self.connection.execute(
            """
            INSERT INTO runs(
                run_id, manifest_fingerprint, model_id, provider, api_model,
                model_config_json, benchmark_path, benchmark_sha256, tasks_json,
                samples, selected_cases, expected_generations, selection_json,
                execution_config_json, experiment_contract_json, status,
                created_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                json.dumps(manifest.get("execution_config", {}), sort_keys=True),
                json.dumps(manifest.get("experiment_contract", {}), sort_keys=True),
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
        for key in (
            "model_config_json",
            "tasks_json",
            "selection_json",
            "execution_config_json",
            "experiment_contract_json",
        ):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        return result

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """SELECT run_id, model_id, api_model, status, selected_cases,
                      expected_generations, created_at, updated_at
               FROM runs ORDER BY created_at DESC"""
        ).fetchall()
        return [dict(row) for row in rows]

    def start_execution(
        self,
        *,
        run_id: str,
        phase: str,
        profile_id: str,
        judge_id: str | None,
        provider: str,
        api_model: str,
        endpoint: str,
        config_sha256: str,
        concurrency: int,
        streaming: bool,
        max_attempts: int,
        retry_policy: dict[str, Any],
        runner_metadata: dict[str, Any],
        task_count: int,
        cache_hit_count: int,
        pending_count: int,
    ) -> str:
        execution_id = uuid.uuid4().hex
        self.connection.execute(
            """INSERT INTO run_executions(
                   execution_id,run_id,phase,profile_id,judge_id,provider,api_model,
                   endpoint,config_sha256,configured_concurrency,streaming,
                   timing_version,sdk_retries,max_attempts,retry_policy_json,
                   runner_metadata_json,task_count,cache_hit_count,pending_count,
                   status,started_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                execution_id,
                run_id,
                phase,
                profile_id,
                judge_id,
                provider,
                api_model,
                endpoint,
                config_sha256,
                concurrency,
                int(streaming),
                self.TIMING_VERSION,
                0,
                max_attempts,
                json.dumps(retry_policy, sort_keys=True),
                json.dumps(runner_metadata, sort_keys=True),
                task_count,
                cache_hit_count,
                pending_count,
                "running",
                utc_now(),
            ),
        )
        self.connection.commit()
        return execution_id

    def finish_execution(
        self,
        execution_id: str,
        *,
        status: str,
        success_count: int,
        failure_count: int,
        cancelled_count: int = 0,
        unpersisted_count: int = 0,
    ) -> None:
        self.connection.execute(
            """UPDATE run_executions
               SET status=?,success_count=?,failure_count=?,cancelled_count=?,
                   unpersisted_count=?,ended_at=?
               WHERE execution_id=?""",
            (
                status,
                success_count,
                failure_count,
                cancelled_count,
                unpersisted_count,
                utc_now(),
                execution_id,
            ),
        )
        self.connection.commit()

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

    @staticmethod
    def _timing_column_names() -> tuple[str, ...]:
        return tuple(name for name, _ in _TIMING_COLUMNS)

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
            "provider_metadata_json", *self._timing_column_names(), "created_at", "updated_at"
        )
        placeholders = ",".join("?" for _ in columns)
        key_columns = {
            "run_id", "dataset", "source_id", "task_type", "sample_idx", "created_at"
        }
        timing_columns = set(self._timing_column_names())
        assignments = ",".join(
            (
                f"{name}=COALESCE(excluded.{name},generations.{name})"
                if name in timing_columns
                else f"{name}=excluded.{name}"
            )
            for name in columns
            if name not in key_columns
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
            "rate_limit_json", "provider_metadata_json", *self._timing_column_names(),
            "created_at", "updated_at"
        )
        placeholders = ",".join("?" for _ in columns)
        key_columns = {
            "run_id", "dataset", "source_id", "sample_idx", "judge_id", "created_at"
        }
        timing_columns = set(self._timing_column_names())
        assignments = ",".join(
            (
                f"{name}=COALESCE(excluded.{name},judgments.{name})"
                if name in timing_columns
                else f"{name}=excluded.{name}"
            )
            for name in columns
            if name not in key_columns
        )
        self.connection.execute(
            f"""INSERT INTO judgments({','.join(columns)}) VALUES({placeholders})
                ON CONFLICT(run_id,dataset,source_id,sample_idx,judge_id)
                DO UPDATE SET {assignments}""",
            tuple(values.get(name) for name in columns),
        )
        self.connection.commit()

    def upsert_attempt(self, row: dict[str, Any]) -> None:
        now = utc_now()
        values = dict(row)
        values.setdefault("created_at", now)
        values["updated_at"] = now
        columns = (
            "execution_id", "request_key", "attempt_index", "dataset", "source_id",
            "task_type", "sample_idx", "judge_id", "max_output_tokens", "eligible_at",
            "request_started_at", "first_token_at", "request_finished_at",
            "queue_wait_ms", "request_wall_ms", "attempt_wall_ms", "first_event_ms",
            "ttft_ms", "time_after_first_token_ms", "server_processing_ms",
            "backoff_planned_ms",
            "backoff_actual_ms", "backoff_source", "outcome", "retryable",
            "http_status", "error_type", "error", "partial_response",
            "finish_reason", "request_id",
            "response_id", "returned_model", "input_tokens", "cached_input_tokens",
            "cache_write_tokens", "output_tokens", "reasoning_tokens", "total_tokens",
            "rate_limit_json", "provider_metadata_json", "created_at", "updated_at"
        )
        placeholders = ",".join("?" for _ in columns)
        key_columns = {"execution_id", "request_key", "attempt_index", "created_at"}
        assignments = ",".join(
            f"{name}=excluded.{name}" for name in columns if name not in key_columns
        )
        self.connection.execute(
            f"""INSERT INTO request_attempts({','.join(columns)}) VALUES({placeholders})
                ON CONFLICT(execution_id,request_key,attempt_index)
                DO UPDATE SET {assignments}""",
            tuple(values.get(name) for name in columns),
        )
        self.connection.commit()

    def update_attempt_backoff(
        self,
        *,
        execution_id: str,
        request_key: str,
        attempt_index: int,
        planned_ms: float,
        actual_ms: float,
        source: str,
    ) -> None:
        self.connection.execute(
            """UPDATE request_attempts
               SET backoff_planned_ms=?,backoff_actual_ms=?,backoff_source=?,updated_at=?
               WHERE execution_id=? AND request_key=? AND attempt_index=?""",
            (
                planned_ms,
                actual_ms,
                source,
                utc_now(),
                execution_id,
                request_key,
                attempt_index,
            ),
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
