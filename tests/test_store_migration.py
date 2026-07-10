from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from acuitybench.store import EvaluationStore


def _make_minimal_v1(path: Path, *, version: int = 1) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        f"""
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO metadata VALUES('schema_version','{version}');
        CREATE TABLE runs (
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
        INSERT INTO runs VALUES(
            'legacy','fingerprint','model','openai','model','{{}}','benchmark.csv',
            'sha','["qa","conv"]',1,1,2,'{{}}','complete','created','updated'
        );
        CREATE TABLE generations (
            run_id TEXT NOT NULL,
            case_id TEXT NOT NULL,
            dataset TEXT NOT NULL,
            source_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
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
            parse_ok INTEGER NOT NULL,
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
            PRIMARY KEY (run_id,dataset,source_id,task_type,sample_idx)
        );
        INSERT INTO generations(
            run_id,case_id,dataset,source_id,task_type,sample_idx,
            normalized_label,split,mapping_method,is_edge_case,prompt,prompt_sha256,
            model_config_sha256,response,response_sha256,parsed_label,parse_ok,status,
            error,attempts,finish_reason,response_id,returned_model,system_fingerprint,
            request_id,input_tokens,cached_input_tokens,cache_write_tokens,output_tokens,
            reasoning_tokens,total_tokens,latency_ms,rate_limit_json,
            provider_metadata_json,created_at,updated_at
        ) VALUES(
            'legacy','synthetic:1','synthetic','1','qa',0,'A','primary','fixture',0,
            'prompt','prompt-sha','model-sha','ACUITY: A','response-sha','A',1,'ok',
            NULL,1,'stop','response','model',NULL,'request',10,0,NULL,2,0,12,123.0,
            '{{}}','{{"openai_processing_ms":"45.5"}}','created','updated'
        );
        CREATE TABLE judgments (
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
            parse_ok INTEGER NOT NULL,
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
            PRIMARY KEY (run_id,dataset,source_id,sample_idx,judge_id)
        );
        INSERT INTO judgments(
            run_id,case_id,dataset,source_id,sample_idx,judge_id,
            judge_config_sha256,generation_response_sha256,judge_prompt,
            judge_prompt_sha256,response,response_sha256,judge_label,judge_reasoning,
            parse_ok,status,error,attempts,finish_reason,response_id,returned_model,
            system_fingerprint,request_id,input_tokens,cached_input_tokens,
            cache_write_tokens,output_tokens,reasoning_tokens,total_tokens,latency_ms,
            rate_limit_json,provider_metadata_json,created_at,updated_at
        ) VALUES(
            'legacy','synthetic:1','synthetic','1',0,'judge','judge-sha',
            'response-sha','judge prompt','judge-prompt-sha','ACUITY: A','judge-response-sha',
            'A','reason',1,'ok',NULL,1,'stop','judge-response','judge-model',NULL,
            'judge-request',12,0,NULL,3,0,15,234.0,'{{}}',
            '{{"server_headers":{{"openai-processing-ms":"12"}}}}','created','updated'
        );
        """
    )
    connection.commit()
    connection.close()


def test_v1_database_migrates_without_reclassifying_legacy_latency(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy.sqlite3"
    _make_minimal_v1(database)

    with EvaluationStore(database) as store:
        version = store.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0]
        generation = store.connection.execute(
            """SELECT latency_ms,total_duration_ms,timing_version,timing_source,
                      request_wall_ms,queue_wait_ms,ttft_ms,server_processing_ms
               FROM generations"""
        ).fetchone()
        judgment = store.connection.execute(
            "SELECT total_duration_ms,server_processing_ms FROM judgments"
        ).fetchone()
        columns = {
            row[1]
            for row in store.connection.execute("PRAGMA table_info(generations)")
        }

    assert version == "2"
    assert generation["latency_ms"] == 123
    assert generation["total_duration_ms"] == 123
    assert generation["timing_version"] == 1
    assert generation["timing_source"] == "legacy_aggregate"
    assert generation["request_wall_ms"] is None
    assert generation["queue_wait_ms"] is None
    assert generation["ttft_ms"] is None
    assert generation["server_processing_ms"] == 45.5
    assert judgment["total_duration_ms"] == 234
    assert judgment["server_processing_ms"] == 12
    assert "service_latency_ms" in columns
    assert "first_token_at" in columns

    # Reopening is idempotent and leaves the single legacy row intact.
    with EvaluationStore(database) as reopened:
        assert reopened.connection.execute(
            "SELECT COUNT(*) FROM generations"
        ).fetchone()[0] == 1
        assert reopened.connection.execute(
            "SELECT COUNT(*) FROM request_attempts"
        ).fetchone()[0] == 0


def test_fresh_database_starts_at_v2(tmp_path: Path) -> None:
    database = tmp_path / "fresh.sqlite3"
    with EvaluationStore(database) as store:
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert store.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0] == "2"
        tables = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        execution_columns = {
            row[1]
            for row in store.connection.execute("PRAGMA table_info(run_executions)")
        }
        attempt_columns = {
            row[1]
            for row in store.connection.execute("PRAGMA table_info(request_attempts)")
        }
    assert {"run_executions", "request_attempts"} <= tables
    assert {"cancelled_count", "unpersisted_count"} <= execution_columns
    assert "time_after_first_token_ms" in attempt_columns
    assert "partial_response" in attempt_columns


def test_v1_null_latency_rows_are_still_classified_as_legacy(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy-null.sqlite3"
    _make_minimal_v1(database)
    connection = sqlite3.connect(database)
    connection.execute(
        """INSERT INTO generations(
               run_id,case_id,dataset,source_id,task_type,sample_idx,is_edge_case,
               prompt,prompt_sha256,model_config_sha256,parse_ok,status,attempts,
               latency_ms,provider_metadata_json,created_at,updated_at
           ) VALUES(
               'legacy','synthetic:2','synthetic','2','qa',0,0,'prompt','prompt-sha',
               'model-sha',0,'failed',1,NULL,NULL,'created','updated'
           )"""
    )
    connection.commit()
    connection.close()

    with EvaluationStore(database) as store:
        row = store.connection.execute(
            """SELECT timing_version,timing_source,total_duration_ms
               FROM generations WHERE source_id='2'"""
        ).fetchone()
    assert row["timing_version"] == 1
    assert row["timing_source"] == "legacy_aggregate"
    assert row["total_duration_ms"] is None


def test_interrupted_fresh_v2_creation_resumes_idempotently(tmp_path: Path) -> None:
    database = tmp_path / "interrupted.sqlite3"
    with EvaluationStore(database):
        pass
    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM metadata WHERE key='schema_version'")
    connection.execute("DROP TABLE request_attempts")
    connection.execute("DROP TABLE judgments")
    connection.commit()
    connection.close()

    with EvaluationStore(database) as store:
        tables = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        version = store.connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0]
    assert {"runs", "generations", "judgments", "request_attempts"} <= tables
    assert version == "2"


def test_v2_reopen_repairs_missing_additive_timing_column(tmp_path: Path) -> None:
    database = tmp_path / "repair.sqlite3"
    with EvaluationStore(database):
        pass
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE generations DROP COLUMN service_latency_ms")
    connection.commit()
    connection.close()

    with EvaluationStore(database) as store:
        columns = {
            row[1]
            for row in store.connection.execute("PRAGMA table_info(generations)")
        }
    assert "service_latency_ms" in columns


def test_v2_reopen_rejects_missing_nonadditive_core_column(tmp_path: Path) -> None:
    database = tmp_path / "corrupt.sqlite3"
    with EvaluationStore(database):
        pass
    connection = sqlite3.connect(database)
    connection.execute("ALTER TABLE run_executions DROP COLUMN endpoint")
    connection.commit()
    connection.close()

    with pytest.raises(
        RuntimeError,
        match=r"run_executions.*missing required columns.*endpoint",
    ):
        EvaluationStore(database)


def test_unknown_future_schema_is_rejected_without_version_change(
    tmp_path: Path,
) -> None:
    database = tmp_path / "future.sqlite3"
    _make_minimal_v1(database, version=999)
    with pytest.raises(RuntimeError, match="Unsupported evaluation DB schema 999"):
        EvaluationStore(database)
    connection = sqlite3.connect(database)
    try:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()[0] == "999"
    finally:
        connection.close()
