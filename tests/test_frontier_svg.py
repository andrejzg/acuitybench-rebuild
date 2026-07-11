from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

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
    assert "Average exact accuracy" in cost_text
    assert "Cost per 1,000 successful target-model calls (USD)" in cost_text
    assert "p95 duration macro-average" in latency_text
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
        assert "infinity" not in raw.lower()


def test_pareto_frontier_excludes_dominated_points() -> None:
    points = [
        FrontierPoint("a", "A", 0.80, 1.0, "measured"),
        FrontierPoint("b", "B", 0.70, 2.0, "measured"),
        FrontierPoint("c", "C", 0.90, 2.0, "measured"),
        FrontierPoint("d", "D", 0.90, 3.0, "measured"),
    ]
    assert _pareto_ids(points) == {"a", "c"}
