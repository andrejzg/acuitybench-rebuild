"""Deterministic, dependency-free SVG charts for model frontier comparisons."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Iterable


SVG_WIDTH = 1200
SVG_HEIGHT = 760
PLOT_LEFT = 118
PLOT_RIGHT = 72
PLOT_TOP = 126
PLOT_BOTTOM = 112


@dataclass(frozen=True)
class FrontierPoint:
    run_id: str
    model: str
    accuracy: float
    x: float
    source: str
    is_proxy: bool = False
    is_partial: bool = False


@dataclass(frozen=True)
class ChartSpec:
    filename: str
    title: str
    subtitle: str
    x_label: str
    empty_message: str
    x_formatter: Callable[[float], str]
    value_formatter: Callable[[float], str]


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nice_step(span: float, target_ticks: int = 6) -> float:
    if not math.isfinite(span) or span <= 0:
        return 1.0
    raw = span / max(target_ticks, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        nice = 1.0
    elif normalized <= 2:
        nice = 2.0
    elif normalized <= 2.5:
        nice = 2.5
    elif normalized <= 5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * magnitude


def _format_cost_tick(value: float) -> str:
    if math.isclose(value, round(value)):
        return f"${value:,.0f}"
    return f"${value:,.1f}"


def _format_duration_tick(value: float) -> str:
    if math.isclose(value, round(value)):
        return f"{value:,.0f}s"
    decimals = 2 if abs(value) < 1 else 1
    formatted = f"{value:,.{decimals}f}".rstrip("0").rstrip(".")
    return f"{formatted}s"


def _x_domain(values: list[float]) -> tuple[float, float, list[float]]:
    maximum = max(values, default=1.0)
    padded = max(maximum * 1.22, 1.0)
    step = _nice_step(padded)
    upper = max(step, math.ceil(padded / step) * step)
    ticks = [index * step for index in range(int(round(upper / step)) + 1)]
    return 0.0, upper, ticks


def _y_domain(values: list[float]) -> tuple[float, float, list[float]]:
    if values:
        low = max(0.0, math.floor((min(values) - 0.08) / 0.05) * 0.05)
        high = min(1.0, math.ceil((max(values) + 0.08) / 0.05) * 0.05)
    else:
        low, high = 0.5, 0.9
    if high - low < 0.20:
        midpoint = (high + low) / 2
        low = max(0.0, math.floor((midpoint - 0.10) / 0.05) * 0.05)
        high = min(1.0, math.ceil((midpoint + 0.10) / 0.05) * 0.05)
    if high <= low:
        low, high = max(0.0, low - 0.1), min(1.0, high + 0.1)
    step = 0.05 if high - low <= 0.35 else 0.10
    count = int(round((high - low) / step))
    ticks = [round(low + index * step, 10) for index in range(count + 1)]
    if not math.isclose(ticks[-1], high):
        ticks.append(high)
    return low, high, ticks


def _pareto_ids(points: Iterable[FrontierPoint]) -> set[str]:
    """Return the minimize-x/maximize-accuracy nondominated point IDs."""
    candidates = sorted(points, key=lambda point: (point.x, -point.accuracy, point.run_id))
    best_accuracy = -math.inf
    frontier: set[str] = set()
    for point in candidates:
        if point.accuracy > best_accuracy:
            frontier.add(point.run_id)
            best_accuracy = point.accuracy
    return frontier


def _frontier_path(
    points: list[FrontierPoint],
    *,
    x_position: Callable[[float], float],
    y_position: Callable[[float], float],
) -> str | None:
    frontier_ids = _pareto_ids(points)
    frontier = sorted(
        (point for point in points if point.run_id in frontier_ids),
        key=lambda point: (point.x, -point.accuracy, point.run_id),
    )
    if len(frontier) < 2:
        return None
    coordinates = " ".join(
        f"{x_position(point.x):.2f},{y_position(point.accuracy):.2f}"
        for point in frontier
    )
    return coordinates


def _chart_svg(points: list[FrontierPoint], spec: ChartSpec) -> str:
    points = sorted(points, key=lambda point: (point.x, -point.accuracy, point.run_id))
    plot_width = SVG_WIDTH - PLOT_LEFT - PLOT_RIGHT
    plot_height = SVG_HEIGHT - PLOT_TOP - PLOT_BOTTOM
    x_low, x_high, x_ticks = _x_domain([point.x for point in points])
    y_low, y_high, y_ticks = _y_domain([point.accuracy for point in points])

    def x_position(value: float) -> float:
        return PLOT_LEFT + (value - x_low) / (x_high - x_low) * plot_width

    def y_position(value: float) -> float:
        return PLOT_TOP + (y_high - value) / (y_high - y_low) * plot_height

    title_id = f"{spec.filename.removesuffix('.svg')}-title"
    desc_id = f"{spec.filename.removesuffix('.svg')}-desc"
    point_summaries = " ".join(
        (
            f"{point.model} ({point.run_id}): {point.accuracy:.2%} average exact, "
            f"{spec.value_formatter(point.x)}, {point.source}."
        )
        for point in points
    )
    comparison_guidance = (
        "Within each measurement type, lower x and higher accuracy are better. "
        "Legacy proxy points are not compared with client service-latency points."
        if any(point.is_proxy for point in points)
        else "Lower x and higher accuracy are better."
    )
    description = (
        f"{spec.title}. {spec.subtitle} {len(points)} plotted model run"
        f"{'s' if len(points) != 1 else ''}. {comparison_guidance} "
        f"{point_summaries}"
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" '
            f'height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" '
            f'role="img" aria-labelledby="{title_id} {desc_id}">'
        ),
        f"  <title id=\"{title_id}\">{escape(spec.title)}</title>",
        f"  <desc id=\"{desc_id}\">{escape(description)}</desc>",
        "  <style>",
        "    :root { --bg:#ffffff; --fg:#44403c; --muted:#78716c; --grid:#e7e5e4; --axis:#a8a29e; --series:#0f9f79; --proxy:#78716c; --partial:#b7791f; --frontier:#0f9f79; }",
        "    @media (prefers-color-scheme: dark) { :root { --bg:#1c1917; --fg:#f5f5f4; --muted:#d6d3d1; --grid:#44403c; --axis:#78716c; --series:#34d399; --proxy:#d6d3d1; --partial:#fbbf24; --frontier:#34d399; } }",
        "    .background{fill:var(--bg)} .title{fill:var(--fg);font:500 30px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif} .subtitle{fill:var(--muted);font:400 17px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
        "    .grid{stroke:var(--grid);stroke-width:1} .axis{stroke:var(--axis);stroke-width:1.4} .tick{fill:var(--muted);font:400 15px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif} .axis-label{fill:var(--fg);font:500 18px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
        "    .point{stroke-width:2.5} .measured-point{fill:var(--series);stroke:var(--series)} .proxy-point{fill:var(--bg);stroke:var(--proxy)} .partial-point{fill:var(--bg);stroke:var(--partial)} .pareto-frontier{fill:none;stroke:var(--frontier);stroke-width:2;stroke-opacity:.45}",
        "    .point-label{fill:var(--fg);font:500 17px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif} .point-value{fill:var(--muted);font:400 15px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif} .better{fill:var(--muted);font:500 14px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;letter-spacing:.08em} .empty{fill:var(--muted);font:400 20px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
        "  </style>",
        f'  <rect class="background" width="{SVG_WIDTH}" height="{SVG_HEIGHT}"/>',
        f'  <text class="title" x="{PLOT_LEFT}" y="52">{escape(spec.title)}</text>',
        f'  <text class="subtitle" x="{PLOT_LEFT}" y="82">{escape(spec.subtitle)}</text>',
    ]

    for tick in y_ticks:
        y = y_position(tick)
        lines.extend(
            [
                f'  <line class="grid" x1="{PLOT_LEFT}" y1="{y:.2f}" x2="{PLOT_LEFT + plot_width}" y2="{y:.2f}"/>',
                f'  <text class="tick" x="{PLOT_LEFT - 16}" y="{y + 5:.2f}" text-anchor="end">{tick:.0%}</text>',
            ]
        )
    for tick in x_ticks:
        x = x_position(tick)
        lines.extend(
            [
                f'  <line class="grid" x1="{x:.2f}" y1="{PLOT_TOP}" x2="{x:.2f}" y2="{PLOT_TOP + plot_height}"/>',
                f'  <text class="tick" x="{x:.2f}" y="{PLOT_TOP + plot_height + 32}" text-anchor="middle">{escape(spec.x_formatter(tick))}</text>',
            ]
        )
    lines.extend(
        [
            f'  <line class="axis" x1="{PLOT_LEFT}" y1="{PLOT_TOP + plot_height}" x2="{PLOT_LEFT + plot_width}" y2="{PLOT_TOP + plot_height}"/>',
            f'  <line class="axis" x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{PLOT_TOP + plot_height}"/>',
            f'  <text class="better" x="{PLOT_LEFT + 14}" y="{PLOT_TOP + 24}">BETTER ↖</text>',
            f'  <text class="axis-label" x="{PLOT_LEFT + plot_width / 2:.2f}" y="{SVG_HEIGHT - 28}" text-anchor="middle">{escape(spec.x_label)}</text>',
            f'  <text class="axis-label" transform="translate(35 {PLOT_TOP + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">Average exact accuracy</text>',
        ]
    )

    comparable = [point for point in points if not point.is_proxy and not point.is_partial]
    path = _frontier_path(
        comparable,
        x_position=x_position,
        y_position=y_position,
    )
    if path:
        lines.append(f'  <polyline class="pareto-frontier" points="{path}"/>')

    frontier_ids = _pareto_ids(comparable)
    if not points:
        lines.append(
            f'  <text class="empty" x="{PLOT_LEFT + plot_width / 2:.2f}" y="{PLOT_TOP + plot_height / 2:.2f}" text-anchor="middle">{escape(spec.empty_message)}</text>'
        )
    for index, point in enumerate(points):
        x = x_position(point.x)
        y = y_position(point.accuracy)
        classes = ["point"]
        if point.is_proxy:
            classes.append("proxy-point")
        elif point.is_partial:
            classes.append("partial-point")
        else:
            classes.append("measured-point")
        if point.run_id in frontier_ids:
            classes.append("pareto-point")
        value = f"{point.accuracy:.1%} · {spec.value_formatter(point.x)}"
        if point.is_proxy:
            value += " · legacy proxy"
        elif point.is_partial:
            value += " · partial cost telemetry"
        mark_title = (
            f"{point.model} ({point.run_id}): {point.accuracy:.2%} average exact; "
            f"{spec.value_formatter(point.x)}; {point.source}."
        )
        escaped_run = escape(point.run_id, quote=True)
        lines.append(
            f'  <g class="{" ".join(classes)}" data-run-id="{escaped_run}"><title>{escape(mark_title)}</title>'
        )
        if point.is_proxy:
            size = 8
            polygon = " ".join(
                [
                    f"{x:.2f},{y - size:.2f}",
                    f"{x + size:.2f},{y:.2f}",
                    f"{x:.2f},{y + size:.2f}",
                    f"{x - size:.2f},{y:.2f}",
                ]
            )
            lines.append(f'    <polygon points="{polygon}"/>')
        else:
            lines.append(f'    <circle cx="{x:.2f}" cy="{y:.2f}" r="8"/>')
        label_anchor = "end" if x > PLOT_LEFT + plot_width * 0.73 else "start"
        label_x = x - 14 if label_anchor == "end" else x + 14
        label_y = y - 15 if index % 2 == 0 else y + 28
        if label_y < PLOT_TOP + 20:
            label_y = y + 28
        if label_y > PLOT_TOP + plot_height - 20:
            label_y = y - 28
        lines.extend(
            [
                f'    <text class="point-label" x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{label_anchor}">{escape(point.model)}</text>',
                f'    <text class="point-value" x="{label_x:.2f}" y="{label_y + 20:.2f}" text-anchor="{label_anchor}">{escape(value)}</text>',
                "  </g>",
            ]
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _frontier_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _point_identity(row: dict[str, str]) -> tuple[str, str, float] | None:
    accuracy = _number(row.get("average_exact"))
    if accuracy is None or not 0 <= accuracy <= 1:
        return None
    return (
        row.get("run_id") or "unknown-run",
        row.get("model") or row.get("run_id") or "unknown model",
        accuracy,
    )


def _cost_points(rows: list[dict[str, str]]) -> list[FrontierPoint]:
    points: list[FrontierPoint] = []
    for row in rows:
        identity = _point_identity(row)
        cost = _number(
            row.get("target_cost_per_1000_successful_calls_usd")
            or row.get("target_cost_per_1000_tasks_usd")
        )
        if identity is None or cost is None or cost < 0:
            continue
        run_id, model, accuracy = identity
        completeness = row.get("target_cost_completeness") or "unknown"
        points.append(
            FrontierPoint(
                run_id=run_id,
                model=model,
                accuracy=accuracy,
                x=cost,
                source=f"cost telemetry: {completeness}",
                is_partial=completeness != "complete",
            )
        )
    return points


def _latency_points(rows: list[dict[str, str]]) -> list[FrontierPoint]:
    points: list[FrontierPoint] = []
    for row in rows:
        identity = _point_identity(row)
        if identity is None:
            continue
        run_id, model, accuracy = identity
        plot_ms = _number(row.get("latency_plot_p95_ms"))
        plot_source = row.get("latency_plot_source") or ""
        if plot_ms is not None and plot_ms >= 0:
            is_proxy = plot_source == "provider_processing_legacy_proxy"
            points.append(
                FrontierPoint(
                    run_id=run_id,
                    model=model,
                    accuracy=accuracy,
                    x=plot_ms / 1000,
                    source=(
                        "provider processing p95 macro-average; legacy proxy, "
                        "not client service latency"
                        if is_proxy
                        else "client p95 service-latency macro-average"
                    ),
                    is_proxy=is_proxy,
                )
            )
            continue
        service_ms = _number(row.get("service_latency_p95_macro_ms"))
        provider_ms = _number(row.get("server_processing_p95_macro_ms"))
        if service_ms is not None and service_ms >= 0:
            points.append(
                FrontierPoint(
                    run_id=run_id,
                    model=model,
                    accuracy=accuracy,
                    x=service_ms / 1000,
                    source="client p95 service-latency macro-average",
                )
            )
        elif provider_ms is not None and provider_ms >= 0:
            points.append(
                FrontierPoint(
                    run_id=run_id,
                    model=model,
                    accuracy=accuracy,
                    x=provider_ms / 1000,
                    source=(
                        "provider processing p95 macro-average; legacy proxy, "
                        "not client service latency"
                    ),
                    is_proxy=True,
                )
            )
    return points


def write_frontier_charts(frontier_path: Path, destination: Path) -> list[Path]:
    """Write accuracy-vs-cost and accuracy-vs-latency SVGs from frontier.csv."""
    rows = _frontier_rows(frontier_path)
    destination.mkdir(parents=True, exist_ok=True)
    specifications = (
        (
            _cost_points(rows),
            ChartSpec(
                filename="accuracy-vs-cost.svg",
                title="Average accuracy vs cost",
                subtitle=(
                    "Target inference only; judge cost excluded. Top-left is better."
                ),
                x_label="Cost per 1,000 successful target-model calls (USD)",
                empty_message="No plottable data: target cost is unavailable",
                x_formatter=_format_cost_tick,
                value_formatter=lambda value: f"${value:,.2f} / 1k calls",
            ),
        ),
        (
            _latency_points(rows),
            ChartSpec(
                filename="accuracy-vs-latency.svg",
                title="Average accuracy vs latency",
                subtitle=(
                    "Filled: client p95 service latency. Hollow diamond: legacy "
                    "provider-header proxy."
                ),
                x_label="p95 duration macro-average across QA + conversation (seconds)",
                empty_message=(
                    "No plottable data: service latency and legacy proxy are unavailable"
                ),
                x_formatter=_format_duration_tick,
                value_formatter=lambda value: f"{value:,.2f}s p95",
            ),
        ),
    )
    outputs: list[Path] = []
    for points, specification in specifications:
        path = destination / specification.filename
        path.write_text(_chart_svg(points, specification), encoding="utf-8")
        outputs.append(path)
    return outputs
