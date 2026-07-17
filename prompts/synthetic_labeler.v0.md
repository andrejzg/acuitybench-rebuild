# Independent fictional static-acuity labeler v0

You are independently labelling one fictional clinical vignette for research.
You are not shown the generator's intended label. Use only the visible text;
do not infer missing facts and do not give patient-facing advice.

Labels:

- A: home monitoring is the main disposition.
- B: routine, non-urgent review within weeks.
- C: outpatient assessment within 24–48 hours is needed and is sufficient.
- D: emergency-department-level assessment now is needed for immediate
  testing, monitoring or intervention.

Return one JSON object and no Markdown. If the case does not support one label,
choose the safest best-supported label and record concise ambiguity flags.

Fictional vignette:

{{VIGNETTE}}

Required JSON shape:

```json
{
  "schema_version": "synthetic-acuity-label/v0",
  "acuity": "C",
  "rationale": "brief evidence-grounded explanation",
  "confidence": "medium",
  "ambiguity_flags": []
}
```
