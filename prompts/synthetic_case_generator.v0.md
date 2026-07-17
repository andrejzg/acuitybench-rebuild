# Fictional static-acuity case generator v0

You are creating one entirely fictional clinical-acuity research vignette.
This is synthetic data construction, not patient care.

Hard constraints:

1. Use no real person, real medical record, source vignette, benchmark case,
   quotation, URL, username, date of birth, address, institution or other
   identifying detail.
2. Do not recall, adapt, paraphrase or imitate a known benchmark or published
   clinical vignette. Construct the case from generic clinical knowledge only.
3. The vignette must contain enough information to support the requested
   disposition while retaining realistic uncertainty. It must not name the
   acuity letter, disposition category or this instruction.
4. A means home monitoring; B means routine review within weeks; C means an
   outpatient assessment within 24–48 hours; D means emergency-department-level
   assessment now because immediate testing, monitoring or intervention may be
   needed.
5. Include only facts established inside this fictional case. Avoid a final
   diagnosis and do not add a clinician answer.
6. Return one JSON object and no Markdown.

Generation slot:

{{SLOT_JSON}}

Required JSON shape:

```json
{
  "schema_version": "synthetic-acuity-generation/v0",
  "vignette": "patient-facing presentation only",
  "intended_acuity": "A",
  "presentation_group": "infection_systemic",
  "age_years": 40,
  "sex_context": "not clinically relevant",
  "relevant_facts": ["fact supporting the intended disposition"],
  "distractor_facts": ["plausible but non-decisive fact"],
  "intended_rationale": "why the requested disposition is supported",
  "fictional_attestation": "Entirely fictional; no real person or source case was used."
}
```
