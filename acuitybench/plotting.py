"""Deterministic, dependency-free SVG charts for model frontier comparisons."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Iterable


SVG_WIDTH = 1368
SVG_HEIGHT = 848
PLOT_LEFT = 209
PLOT_RIGHT = 104
PLOT_TOP = 107
PLOT_BOTTOM = 188
ASPIRATION_LABEL = "Our trained model?"


@dataclass(frozen=True)
class FrontierPoint:
    run_id: str
    model: str
    accuracy: float
    x: float
    source: str
    reasoning_effort: str | None = None
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
    # Match the paper-style editorial chart for the full frontier while still
    # expanding safely for smoke runs or future results outside this range.
    low, high = 0.72, 0.86
    if values:
        if min(values) < low:
            low = max(0.0, math.floor((min(values) - 0.01) / 0.02) * 0.02)
        if max(values) > high:
            high = min(1.0, math.ceil((max(values) + 0.01) / 0.02) * 0.02)
    span = high - low
    step = 0.02 if span <= 0.24 else 0.05 if span <= 0.50 else 0.10
    low = max(0.0, math.floor(low / step) * step)
    high = min(1.0, math.ceil(high / step) * step)
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
            f"{spec.value_formatter(point.x)}, {point.source}"
            + (
                f", reasoning effort {point.reasoning_effort}."
                if point.reasoning_effort
                else "."
            )
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
        f"{point_summaries} "
        f"The green {ASPIRATION_LABEL} marker is an aspirational target, not a "
        "measured result."
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
        "    :root { --bg:#ffffff; --fg:#4f4946; --muted:#6b6866; --grid:#e7e7e7; --axis:#c5c5c5; --frontier:#9e9e9e; --aspiration:#4a9c7b; --partial:#b7791f; }",
        "    .background{fill:var(--bg)} .legend-label{fill:var(--fg);font:400 22px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
        "    .grid{stroke:var(--grid);stroke-width:1.3} .axis{stroke:var(--axis);stroke-width:1.7} .tick{fill:var(--muted);font:400 18px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif} .axis-label{fill:var(--fg);font:500 20px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
        "    .point{stroke-width:2.5} .measured-point{fill:var(--frontier);stroke:var(--frontier)} .proxy-point{fill:var(--bg);stroke:var(--frontier)} .partial-point{fill:var(--bg);stroke:var(--partial)} .frontier-legend-mark{fill:var(--frontier)} .aspiration-mark{fill:var(--aspiration)}",
        "    .point-label,.aspiration-label{fill:var(--fg);stroke:var(--bg);stroke-width:4;stroke-linejoin:round;paint-order:stroke fill;font:400 20px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif} .empty{fill:var(--muted);font:400 20px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
        "  </style>",
        f'  <rect class="background" width="{SVG_WIDTH}" height="{SVG_HEIGHT}"/>',
        '  <g class="legend" aria-label="Chart legend">',
        '    <circle class="aspiration-mark" cx="114" cy="48" r="11"/>',
        f'    <text class="legend-label" x="139" y="56">{escape(ASPIRATION_LABEL)}</text>',
        '    <circle class="frontier-legend-mark" cx="396" cy="48" r="11"/>',
        '    <text class="legend-label" x="421" y="56">Frontier models</text>',
        "  </g>",
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
        lines.append(
            f'  <text class="tick" x="{x:.2f}" y="{PLOT_TOP + plot_height + 46}" text-anchor="middle">{escape(spec.x_formatter(tick))}</text>'
        )
    lines.extend(
        [
            f'  <line class="axis" x1="{PLOT_LEFT}" y1="{PLOT_TOP + plot_height}" x2="{PLOT_LEFT + plot_width}" y2="{PLOT_TOP + plot_height}"/>',
            f'  <line class="axis" x1="{PLOT_LEFT}" y1="{PLOT_TOP}" x2="{PLOT_LEFT}" y2="{PLOT_TOP + plot_height}"/>',
            f'  <text class="axis-label" x="{PLOT_LEFT + plot_width / 2:.2f}" y="780" text-anchor="middle">{escape(spec.x_label)}</text>',
            f'  <text class="axis-label" transform="translate(93 {PLOT_TOP + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle">Average accuracy</text>',
        ]
    )

    comparable = [point for point in points if not point.is_proxy and not point.is_partial]
    frontier_ids = _pareto_ids(comparable)
    if not points:
        lines.append(
            f'  <text class="empty" x="{PLOT_LEFT + plot_width / 2:.2f}" y="{PLOT_TOP + plot_height / 2:.2f}" text-anchor="middle">{escape(spec.empty_message)}</text>'
        )
    else:
        aspiration_x = PLOT_LEFT + plot_width * 0.047
        aspiration_y = PLOT_TOP + plot_height * 0.098
        lines.extend(
            [
                '  <g class="aspirational-marker" data-kind="aspirational">',
                '    <title>Aspirational target, not a measured result.</title>',
                f'    <circle class="aspiration-mark" cx="{aspiration_x:.2f}" cy="{aspiration_y:.2f}" r="11"/>',
                f'    <text class="aspiration-label" x="{aspiration_x + 23:.2f}" y="{aspiration_y + 7:.2f}">{escape(ASPIRATION_LABEL)}</text>',
                "  </g>",
            ]
        )
    best_accuracy = max((point.accuracy for point in points), default=-math.inf)
    for point in points:
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
        mark_title = (
            f"{point.model} ({point.run_id}): {point.accuracy:.2%} average exact; "
            f"{spec.value_formatter(point.x)}; {point.source}"
            + (
                f"; reasoning effort {point.reasoning_effort}."
                if point.reasoning_effort
                else "."
            )
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
            lines.append(f'    <circle cx="{x:.2f}" cy="{y:.2f}" r="11"/>')
        label_anchor = "end" if x > PLOT_LEFT + plot_width * 0.73 else "start"
        label_x = x - 18 if label_anchor == "end" else x + 18
        label_y = y - 18 if math.isclose(point.accuracy, best_accuracy) else y + 36
        if label_y < PLOT_TOP + 20:
            label_y = y + 36
        if label_y > PLOT_TOP + plot_height - 20:
            label_y = y - 32
        lines.extend(
            [
                f'    <text class="point-label" x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="{label_anchor}">{escape(point.model)}</text>',
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
    raw_model = row.get("model") or row.get("run_id") or "unknown model"
    model = {
        "gpt-5-mini": "GPT-5 mini",
        "gpt-5.4": "GPT-5.4",
    }.get(raw_model, raw_model)
    return (
        row.get("run_id") or "unknown-run",
        model,
        accuracy,
    )


def _reasoning_effort(row: dict[str, str]) -> str | None:
    effort = (row.get("reasoning_effort") or "").strip()
    return effort or None


def _latency_profile_note(rows: list[dict[str, str]]) -> str:
    profiles: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("latency_plot_source") != "client_service_latency":
            continue
        streaming = (row.get("latency_profile_streaming") or "").strip().lower()
        concurrency = (row.get("latency_profile_concurrency") or "").strip()
        tier = (row.get("latency_profile_service_tier") or "").strip()
        if streaming and concurrency and tier:
            profiles.add((streaming, concurrency, tier))
    if len(profiles) != 1:
        return "Serving profile: see frontier.csv."
    streaming, concurrency, tier = next(iter(profiles))
    transport = "streaming" if streaming == "true" else "non-streaming"
    return f"Profile: {transport}, concurrency {concurrency}, {tier} tier."


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
                reasoning_effort=_reasoning_effort(row),
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
                    reasoning_effort=_reasoning_effort(row),
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
                    reasoning_effort=_reasoning_effort(row),
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
                    reasoning_effort=_reasoning_effort(row),
                    is_proxy=True,
                )
            )
    return points


def write_frontier_charts(frontier_path: Path, destination: Path) -> list[Path]:
    """Write accuracy-vs-cost and accuracy-vs-latency SVGs from frontier.csv."""
    rows = _frontier_rows(frontier_path)
    destination.mkdir(parents=True, exist_ok=True)
    latency_profile_note = _latency_profile_note(rows)
    specifications = (
        (
            _cost_points(rows),
            ChartSpec(
                filename="accuracy-vs-cost.svg",
                title="Average accuracy vs cost",
                subtitle=(
                    "Gray circles are measured target-model runs; judge cost is "
                    "excluded."
                ),
                x_label="Target-model cost per 1,000 calls (USD)",
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
                    "Gray circles are measured client p95 service latency; hollow "
                    f"diamonds are legacy proxies. {latency_profile_note}"
                ),
                x_label="p95 service latency (seconds)",
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
