from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import pytest

from acuitybench.plotting import (
    PLOT_LEFT,
    PLOT_RIGHT,
    PLOT_TOP,
    SVG_HEIGHT,
    SVG_WIDTH,
    FrontierPoint,
    _pareto_ids,
    write_frontier_charts,
)


SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


def _write_frontier(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _point_groups(root: ET.Element) -> list[ET.Element]:
    return [
        element
        for element in root.findall(f".//{SVG_NAMESPACE}g")
        if "point" in element.attrib.get("class", "").split()
    ]


def test_single_point_charts_are_valid_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    frontier = _write_frontier(
        tmp_path / "frontier.csv",
        [
            {
                "run_id": "single-run",
                "model": "single-model",
                "average_exact": 0.75,
                "target_cost_per_1000_successful_calls_usd": 1.25,
                "target_cost_completeness": "complete",
                "latency_plot_p95_ms": 420,
                "latency_plot_source": "client_service_latency",
                "reasoning_effort": "medium",
                "latency_profile_streaming": "true",
                "latency_profile_concurrency": 20,
                "latency_profile_service_tier": "default",
            }
        ],
    )
    first = write_frontier_charts(frontier, tmp_path / "first")
    second = write_frontier_charts(frontier, tmp_path / "second")

    for first_path, second_path in zip(first, second, strict=True):
        first_bytes = first_path.read_bytes()
        assert first_bytes == second_path.read_bytes()
        text = first_bytes.decode("utf-8")
        assert "nan" not in text.lower()
        assert "infinity" not in text.lower()
        root = ET.fromstring(text)
        assert root.tag == f"{SVG_NAMESPACE}svg"
        assert root.attrib["viewBox"] == f"0 0 {SVG_WIDTH} {SVG_HEIGHT}"
        assert root.find(f"{SVG_NAMESPACE}title") is not None
        assert root.find(f"{SVG_NAMESPACE}desc") is not None
        points = _point_groups(root)
        assert len(points) == 1
        assert "pareto-point" in points[0].attrib["class"]
        circle = points[0].find(f"{SVG_NAMESPACE}circle")
        assert circle is not None
        x = float(circle.attrib["cx"])
        y = float(circle.attrib["cy"])
        assert math.isfinite(x) and math.isfinite(y)
        assert PLOT_LEFT <= x <= SVG_WIDTH - PLOT_RIGHT
        assert PLOT_TOP <= y <= SVG_HEIGHT

    cost_text = first[0].read_text(encoding="utf-8")
    latency_text = first[1].read_text(encoding="utf-8")
    assert "Average accuracy" in cost_text
    assert "Target-model cost per 1,000 calls (USD)" in cost_text
    assert "p95 service latency (seconds)" in latency_text
    assert "effort medium" in latency_text
    assert "Profile: streaming, concurrency 20, default tier" in latency_text
    assert "Our trained model?" in cost_text
    assert "Frontier models" in cost_text
    assert 'data-kind="aspirational"' in cost_text
    assert "not a measured result" in cost_text
    assert "paint-order:stroke fill" in cost_text
    assert "pareto-frontier" not in cost_text
    latency_root = ET.fromstring(latency_text)
    latency_desc = latency_root.find(f"{SVG_NAMESPACE}desc")
    assert latency_desc is not None
    assert "single-model (single-run)" in (latency_desc.text or "")
    assert "0.42s p95" in (latency_desc.text or "")
    duration_ticks = [
        element.text
        for element in latency_root.findall(f".//{SVG_NAMESPACE}text")
        if element.attrib.get("class") == "tick"
        and (element.text or "").endswith("s")
    ]
    assert len(duration_ticks) == len(set(duration_ticks))
    assert "0.2s" in duration_ticks
    accuracy_ticks = [
        element.text
        for element in latency_root.findall(f".//{SVG_NAMESPACE}text")
        if element.attrib.get("class") == "tick"
        and (element.text or "").endswith("%")
    ]
    assert accuracy_ticks == ["72%", "74%", "76%", "78%", "80%", "82%", "84%", "86%"]
    grid_lines = [
        element
        for element in latency_root.findall(f".//{SVG_NAMESPACE}line")
        if element.attrib.get("class") == "grid"
    ]
    assert grid_lines
    assert all(line.attrib["y1"] == line.attrib["y2"] for line in grid_lines)


def test_legacy_provider_processing_is_explicit_latency_proxy(
    tmp_path: Path,
) -> None:
    frontier = _write_frontier(
        tmp_path / "frontier.csv",
        [
            {
                "run_id": "legacy",
                "model": "legacy-model",
                "average_exact": 0.74,
                "latency_plot_p95_ms": 1234,
                "latency_plot_source": "provider_processing_legacy_proxy",
            },
            {
                "run_id": "instrumented",
                "model": "instrumented-model",
                "average_exact": 0.76,
                "latency_plot_p95_ms": 500,
                "latency_plot_source": "client_service_latency",
                "server_processing_p95_macro_ms": 300,
            },
            {
                "run_id": "missing",
                "model": "missing-model",
                "average_exact": 0.80,
            },
        ],
    )
    latency_path = write_frontier_charts(frontier, tmp_path)[1]
    root = ET.parse(latency_path).getroot()
    points = {point.attrib["data-run-id"]: point for point in _point_groups(root)}

    assert set(points) == {"legacy", "instrumented"}
    assert "proxy-point" in points["legacy"].attrib["class"]
    assert "proxy-point" not in points["instrumented"].attrib["class"]
    legacy_title = points["legacy"].find(f"{SVG_NAMESPACE}title")
    assert legacy_title is not None
    assert "legacy proxy, not client service latency" in (legacy_title.text or "")
    assert points["legacy"].find(f"{SVG_NAMESPACE}polygon") is not None
    assert points["instrumented"].find(f"{SVG_NAMESPACE}circle") is not None
    root_desc = root.find(f"{SVG_NAMESPACE}desc")
    assert root_desc is not None
    assert "Legacy proxy points are not compared" in (root_desc.text or "")
    assert "legacy-model (legacy)" in (root_desc.text or "")


def test_svg_escapes_model_and_run_labels(tmp_path: Path) -> None:
    model = 'A&B <fast> "quoted"'
    run_id = "run'1 & more"
    frontier = _write_frontier(
        tmp_path / "frontier.csv",
        [
            {
                "run_id": run_id,
                "model": model,
                "average_exact": 0.7,
                "target_cost_per_1000_successful_calls_usd": 2,
                "target_cost_completeness": "complete",
            }
        ],
    )
    cost_path = write_frontier_charts(frontier, tmp_path)[0]
    raw = cost_path.read_text(encoding="utf-8")
    assert "A&amp;B" in raw
    assert "&lt;fast&gt;" in raw
    assert "<fast>" not in raw

    root = ET.fromstring(raw)
    point = _point_groups(root)[0]
    assert point.attrib["data-run-id"] == run_id
    labels = [
        element.text
        for element in point.findall(f"{SVG_NAMESPACE}text")
        if "point-label" in element.attrib.get("class", "")
    ]
    assert labels == [model]


def test_unplottable_rows_still_write_valid_placeholder_svgs(
    tmp_path: Path,
) -> None:
    frontier = _write_frontier(
        tmp_path / "frontier.csv",
        [
            {
                "run_id": "bad",
                "model": "bad",
                "average_exact": "not-a-number",
                "target_cost_per_1000_successful_calls_usd": float("inf"),
            }
        ],
    )
    outputs = write_frontier_charts(frontier, tmp_path)
    for output in outputs:
        raw = output.read_text(encoding="utf-8")
        root = ET.fromstring(raw)
        assert not _point_groups(root)
        assert "No plottable data" in raw
        assert 'data-kind="aspirational"' not in raw
        assert "infinity" not in raw.lower()


def test_pareto_frontier_excludes_dominated_points() -> None:
    points = [
        FrontierPoint("a", "A", 0.80, 1.0, "measured"),
        FrontierPoint("b", "B", 0.70, 2.0, "measured"),
        FrontierPoint("c", "C", 0.90, 2.0, "measured"),
        FrontierPoint("d", "D", 0.90, 3.0, "measured"),
    ]
    assert _pareto_ids(points) == {"a", "c"}


def test_committed_charts_use_latest_full_results_and_separate_aspiration() -> None:
    root_path = Path(__file__).resolve().parents[1]
    frontier = pd.read_csv(root_path / "results/model-comparison/frontier.csv")
    expected = {
        "gpt-5-mini-paper-stream-medium-20260711": {
            "accuracy": 0.7371916508538899,
            "cost": 2.092634146608315,
            "latency_ms": 17233.86545202302,
        },
        "gpt-5.4-paper-stream-none-20260711": {
            "accuracy": 0.7732447817836812,
            "cost": 5.085045404814005,
            "latency_ms": 6982.850375002092,
        },
    }
    assert set(frontier["run_id"]) == set(expected)
    indexed = frontier.set_index("run_id")
    for run_id, values in expected.items():
        assert indexed.loc[run_id, "average_exact"] == pytest.approx(
            values["accuracy"]
        )
        assert indexed.loc[
            run_id, "target_cost_per_1000_successful_calls_usd"
        ] == pytest.approx(values["cost"])
        assert indexed.loc[run_id, "latency_plot_p95_ms"] == pytest.approx(
            values["latency_ms"]
        )

    for filename in ("accuracy-vs-cost.svg", "accuracy-vs-latency.svg"):
        chart = ET.parse(root_path / "results/model-comparison" / filename).getroot()
        measured = _point_groups(chart)
        assert {point.attrib["data-run-id"] for point in measured} == set(expected)
        aspirations = [
            element
            for element in chart.findall(f".//{SVG_NAMESPACE}g")
            if element.attrib.get("data-kind") == "aspirational"
        ]
        assert len(aspirations) == 1
        aspiration = aspirations[0]
        assert "data-run-id" not in aspiration.attrib
        title = aspiration.find(f"{SVG_NAMESPACE}title")
        assert title is not None and "not a measured result" in (title.text or "")
        labels = [
            element.text
            for element in aspiration.findall(f"{SVG_NAMESPACE}text")
        ]
        assert labels == ["Our trained model?"]

        point_titles = " ".join(
            (point.find(f"{SVG_NAMESPACE}title").text or "")
            for point in measured
        )
        assert "73.72%" in point_titles
        assert "77.32%" in point_titles
        if filename == "accuracy-vs-cost.svg":
            assert "$2.09 / 1k calls" in point_titles
            assert "$5.09 / 1k calls" in point_titles
        else:
            assert "17.23s p95" in point_titles
            assert "6.98s p95" in point_titles
