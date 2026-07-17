"""Build the source-grounded, evaluation-only interactive seed set.

The transformation is deliberately deterministic and non-generative.  It
keeps only user messages, routes source text fragments to a fixed question
catalog, and uses ``I'm not sure.`` whenever a fact was not present.  The
result is useful for exercising an interactive triage pipeline, but the
automatically transformed case content is not a clinician-authored simulation.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

from acuitybench import __version__
from acuitybench.interactive.schema_validation import (
    load_json_schema,
    validate_case_card,
)
from acuitybench.sources import project_root


CASE_SCHEMA_VERSION = "interactive-case-card/v1"
ACTION_SCHEMA_VERSION = "interactive-action/v1"
MANIFEST_SCHEMA_VERSION = "interactive-seed-manifest/v1"
TRANSFORMATION_VERSION = "user-messages-source-grounded-routing/v1"
LABELS = ("A", "B", "C", "D")
SEVERITY = {label: index for index, label in enumerate(LABELS, 1)}


@dataclass(frozen=True)
class SeedBuildResult:
    output_dir: Path
    case_cards_path: Path
    manifest_path: Path
    case_count: int
    manifest: dict[str, Any]


def default_seed_config_path() -> Path:
    return project_root() / "configs/interactive/seed_set.v1.yaml"


def default_seed_output_dir() -> Path:
    return project_root() / "data/interactive/seed_v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path, root: Path) -> str:
    return str(path.relative_to(root) if path.is_relative_to(root) else path)


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a YAML mapping: {path}")
    return raw


def load_question_catalog(path: Path | None = None) -> dict[str, Any]:
    path = path or project_root() / "configs/interactive/question_catalog.v1.yaml"
    catalog = _load_yaml_mapping(path, "question catalog")
    if catalog.get("schema_version") != "interactive-question-catalog/v1":
        raise ValueError("Unsupported interactive question catalog version")
    questions = catalog.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Question catalog must contain a non-empty questions list")
    ids = [question.get("id") for question in questions if isinstance(question, dict)]
    if len(ids) != len(questions) or len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be present and unique")
    if not all(isinstance(value, str) and value for value in ids):
        raise ValueError("Every question ID must be a non-empty string")
    if not str(catalog.get("unknown_response", "")).strip():
        raise ValueError("Question catalog must define unknown_response")
    return catalog


def _validate_contract_alignment(
    *,
    question_catalog: Mapping[str, Any],
    action_catalog: Mapping[str, Any],
    case_schema: Mapping[str, Any],
    action_schema: Mapping[str, Any],
) -> None:
    question_ids = [str(question["id"]) for question in question_catalog["questions"]]
    action_questions = action_catalog.get("questions")
    if action_catalog.get("schema_version") != ACTION_SCHEMA_VERSION:
        raise ValueError("Action catalog schema version differs from action v1")
    if not isinstance(action_questions, list):
        raise ValueError("Action catalog must contain questions")
    action_ids = [
        str(question.get("id"))
        for question in action_questions
        if isinstance(question, Mapping)
    ]
    case_ids = list(case_schema.get("$defs", {}).get("questionId", {}).get("enum", []))
    schema_action_ids = list(
        action_schema.get("$defs", {}).get("questionId", {}).get("enum", [])
    )
    if not question_ids == action_ids == case_ids == schema_action_ids:
        raise ValueError("Question IDs differ across catalogs and schemas")
    if case_schema.get("properties", {}).get("schema_version", {}).get("const") != CASE_SCHEMA_VERSION:
        raise ValueError("Case schema version constant differs from builder")
    action_versions = {
        branch.get("properties", {}).get("schema_version", {}).get("const")
        for branch in action_schema.get("oneOf", [])
        if isinstance(branch, Mapping)
    }
    if action_versions != {ACTION_SCHEMA_VERSION}:
        raise ValueError("Action schema version constants differ from builder")


def _normalise_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value.replace("\x00", ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_user_text(dataset: str, normalized_prompt_text: str) -> str:
    """Return source-authored user text, excluding any assistant messages."""

    if dataset != "healthbench":
        return _normalise_text(normalized_prompt_text)
    try:
        messages = json.loads(normalized_prompt_text)
    except json.JSONDecodeError as exc:
        raise ValueError("HealthBench prompt is not valid JSON") from exc
    if not isinstance(messages, list):
        raise ValueError("HealthBench prompt must be a list of messages")
    user_messages = [
        str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
        and str(message.get("role", "")).lower() == "user"
        and str(message.get("content", "")).strip()
    ]
    if not user_messages:
        raise ValueError("HealthBench prompt contains no user message")
    return _normalise_text("\n\n".join(user_messages))


def _clean_fragment(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", value)
    return value.strip()


def _segments(text: str) -> list[str]:
    parts: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", paragraph)
        for sentence in sentences:
            cleaned = _clean_fragment(sentence)
            if cleaned:
                parts.append(cleaned)
    return parts or [text]


QUESTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "demographic.age": (
        r"\b\d{1,3}\s*(?:years?[- ]old|yo\b|y/o\b)",
        r"\bage\s*[:=-]",
        r"\b\d{1,3}[mMfF]\b",
    ),
    "demographic.sex": (
        r"\b(?:male|female|man|woman|boy|girl)\b",
        r"\b\d{1,3}[mMfF]\b",
        r"\bsex\s*[:=-]",
    ),
    "demographic.pregnancy": (
        r"\bpregnan",
        r"\bpostpartum\b",
        r"\bgave birth\b",
        r"\bdelivered\b",
    ),
    "symptom.onset": (
        r"\b(?:started|began|onset|sudden|suddenly|since|ago|today|yesterday|last night)\b",
        r"\b(?:minute|hour|day|week|month|year)s?\b",
    ),
    "symptom.course": (
        r"\b(?:worsen|worse|better|improv|progress|persist|recurr|resolved?|still|increas|decreas)\w*\b",
        r"\bcomes? and goes?\b",
    ),
    "symptom.severity": (
        r"\b(?:mild|moderate|severe|extreme|intense|major|worst|unbearable)\b",
        r"\b\d{1,2}\s*/\s*10\b",
    ),
    "symptom.location": (
        r"\b(?:head|face|eye|ear|mouth|throat|neck|chest|breast|back|abdomen|abdominal|stomach|pelvis|groin|arm|hand|leg|foot|skin|gum|tooth|teeth|nose|rectal|vaginal|penis|testicle|urinary)\w*\b",
    ),
    "symptom.character": (
        r"\b(?:sharp|dull|burning|throbbing|aching|ache|pressure|tight|itch|tingl|numb|cramp|stabbing|sore|swollen)\w*\b",
    ),
    "symptom.triggers": (
        r"\b(?:after|during|when|trigger|reliev|aggravat|movement|exercise|eating|lying|standing|walking|breathing)\w*\b",
        r"\bmakes? (?:it|this) (?:better|worse)\b",
    ),
    "symptom.associated": (
        r"\b(?:nausea|vomit|diarrh|rash|cough|dizz|headache|fatigue|chills|sweat|discharge|palpitation|constipation)\w*\b",
    ),
    "context.fever": (r"\b(?:fever|febrile|temperature|chills|rigors?)\b",),
    "context.trauma": (
        r"\b(?:fall|fell|injur|accident|hit|struck|cut|wound|crash|trauma|bite|burn)\w*\b",
    ),
    "context.exposure": (
        r"\b(?:travel|contact|expos|coworker|colleague|family member|contagious|infected|sick)\w*\b",
    ),
    "function.impact": (
        r"\b(?:unable|cannot|can't|difficulty|struggl|work|walk|sleep|eat|drink|drive|daily activit|care for)\w*\b",
    ),
    "treatment.tried": (
        r"\b(?:took|taken|tried|used|applied|helped|treatment|remedy|ice|rest|paracetamol|acetaminophen|ibuprofen)\w*\b",
    ),
    "history.conditions": (
        r"\b(?:history of|diagnos|condition|disease|diabetes|asthma|cancer|hypertension|migraine|previously|before)\w*\b",
    ),
    "history.medications": (
        r"\b(?:medication|medicine|prescription|tablet|pill|dose|supplement|chemo|insulin|antibiotic|paracetamol|acetaminophen|ibuprofen)\w*\b",
    ),
    "history.allergies": (r"\b(?:allerg|anaphyl)\w*\b",),
    "history.procedures": (
        r"\b(?:surgery|operation|procedure|scan|biopsy|injection|hospitali[sz]|chemotherapy|radiotherapy)\w*\b",
    ),
    "history.substance_use": (
        r"\b(?:alcohol|drink|smok|cigarette|vape|cannabis|marijuana|cocaine|coke|heroin|recreational drug)\w*\b",
    ),
    "red_flag.breathing": (
        r"\b(?:shortness of breath|short of breath|difficulty breathing|trouble breathing|"
        r"struggl(?:e|ing) to breathe|can't breathe|cannot breathe|breathless|wheez)\w*\b",
        r"\b(?:needs?|requires?|on) (?:supplemental )?oxygen\b",
        r"\boxygen (?:saturation|level)\b.{0,25}\b(?:low|below|under|[0-8]\d|9[0-1])\b",
    ),
    "red_flag.chest_pain": (r"\b(?:chest pain|chest pressure|chest tightness)\b",),
    "red_flag.consciousness": (
        r"\b(?:fainted|fainting|passed out|unconscious|somnolent|difficult to wake|"
        r"very drowsy|barely conscious|can't stay conscious|cannot stay conscious|"
        r"losing consciousness|slipping in and out of (?:consciousness|it)|"
        r"barely keep (?:my|their|his|her) eyes open|struggling to stay awake|"
        r"black(?:ing|ed) out)\b",
    ),
    "red_flag.neurologic": (
        r"\b(?:confus|seizure|new weakness|one-sided|slurred speech|trouble speaking|"
        r"slurr(?:ing|ed) (?:her|his|their|my|the) (?:words|speech)|"
        r"speech (?:is|sounds) slurred|can't (?:really )?get (?:the )?words out|"
        r"facial droop|vision loss|"
        r"blurred vision|paralys|sudden severe headache)\w*\b",
    ),
    "red_flag.bleeding": (
        r"\b(?:bleeding|bloody|blood in|blood from|blood when|blood loss|nosebleed|"
        r"nose bleed|haemorrhag(?:e|ed|ing)|hemorrhag(?:e|ed|ing)|"
        r"black stool|vomit(?:ing)? blood|"
        r"puk(?:ed|ing) up blood|ha?ematemesis)\w*\b",
    ),
    "red_flag.sepsis": (
        r"\b(?:high fever|shaking chills|rigors|sepsis|immunocompromised|immune system is shot)\b",
    ),
    "red_flag.dehydration": (
        r"\b(?:can't keep|cannot keep|unable to keep).{0,20}\bfluids?\b",
        r"\b(?:not urinating|no urine|very little urine|dehydrat)\w*\b",
        r"\b(?:persistent|repeated|nonstop)\s+vomit\w*\b",
    ),
    "red_flag.anaphylaxis": (
        r"\b(?:anaphyl|tongue swelling|lip swelling|swollen tongue|swollen lips|throat closing|epipen)\w*\b",
    ),
    "red_flag.pregnancy": (
        r"\b(?:pregnan|postpartum)\w*\b.{0,80}\b(?:bleeding|severe|headache|vision|pain)\w*\b",
    ),
    "red_flag.mental_health": (
        r"\b(?:suicid|self[- ]harm|harm myself|harm someone|kill myself|kill someone|unsafe)\w*\b",
        r"\b(?:attempted|attenpted)\b.{0,40}\b(?:hang(?:ing)? myself|"
        r"hanging|my life|own life)\b",
        r"\battempt at (?:my|their|own) life\b",
    ),
    "logistics.support": (
        r"\b(?:alone|husband|wife|partner|friend|family|support|someone with me|ride|transport)\w*\b",
    ),
}


PRESENTATION_GROUPS: dict[str, tuple[str, ...]] = {
    "cardiopulmonary": (
        r"\bchest\b",
        r"\bheart\b",
        r"\bbreath",
        r"\boxygen\b",
        r"\bpalpitation",
    ),
    "neurologic": (
        r"\bheadache\b",
        r"\bseizure\b",
        r"\bconfus",
        r"\bnumb",
        r"\bweakness\b",
        r"\bdizz",
        r"\bfaint",
    ),
    "gastrointestinal": (
        r"\babdom",
        r"\bstomach\b",
        r"\bvomit",
        r"\bdiarrh",
        r"\bbowel\b",
        r"\bnausea\b",
    ),
    "infection_systemic": (
        r"\bfever\b",
        r"\binfect",
        r"\bchills\b",
        r"\btemperature\b",
    ),
    "injury_bleeding": (
        r"\binjur",
        r"\bfall\b",
        r"\bcut\b",
        r"\bbleed",
        r"\bblood\b",
        r"\bbruise",
    ),
    "skin_allergy": (
        r"\brash\b",
        r"\bskin\b",
        r"\ballerg",
        r"\bhives\b",
        r"\bitch",
        r"\bswelling\b",
    ),
    "reproductive_urinary": (
        r"\bpregnan",
        r"\bpostpartum\b",
        r"\bvaginal\b",
        r"\burin",
        r"\btestic",
        r"\bpenis\b",
        r"\bpelvi",
    ),
    "mental_health": (
        r"\banxiety\b",
        r"\bdepress",
        r"\bsuicid",
        r"\bmental health\b",
        r"\bpanic\b",
    ),
    "eye_ent_dental": (
        r"\beye\b",
        r"\bear\b",
        r"\bthroat\b",
        r"\bnose\b",
        r"\btooth\b",
        r"\bteeth\b",
        r"\bgum\b",
    ),
    "medication_toxicology": (
        r"\boverdose\b",
        r"\bpoison",
        r"\bmedication\b",
        r"\bsupplement\b",
        r"\bcocaine\b",
        r"\bcoke\b",
    ),
    "musculoskeletal": (
        r"\bjoint\b",
        r"\bmuscle\b",
        r"\bback pain\b",
        r"\barm pain\b",
        r"\bleg pain\b",
        r"\bsprain\b",
    ),
}


RED_FLAG_METADATA: dict[str, str] = {
    "red_flag.breathing": "respiratory_compromise",
    "red_flag.chest_pain": "acute_chest_symptom",
    "red_flag.consciousness": "altered_consciousness",
    "red_flag.neurologic": "acute_neurologic_symptom",
    "red_flag.bleeding": "significant_bleeding",
    "red_flag.sepsis": "systemic_infection",
    "red_flag.dehydration": "fluid_intolerance",
    "red_flag.anaphylaxis": "severe_allergic_reaction",
    "red_flag.pregnancy": "pregnancy_related_risk",
    "red_flag.mental_health": "immediate_mental_health_risk",
}


def _matches(fragment: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, fragment, flags=re.IGNORECASE) for pattern in patterns)


def _has_positive_match(fragment: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        for match in re.finditer(pattern, fragment, flags=re.IGNORECASE):
            prefix = fragment[max(0, match.start() - 90) : match.start()]
            negation_scope = re.split(
                r"\b(?:but|however|yet|although)\b|[.!?;]",
                prefix,
                flags=re.IGNORECASE,
            )[-1]
            negation = re.search(
                r"(?:\b(?:no|not|without|denies|denied|never|cannot|can't|"
                r"don['’]t|doesn['’]t|didn['’]t|isn['’]t|wasn['’]t)\b|"
                r"\b(?:negative for|absence of)\b)"
                r"(?:\W+\w+){0,8}\W*$",
                negation_scope,
                flags=re.IGNORECASE,
            )
            suffix = fragment[match.end() : match.end() + 50]
            postposed_negation = re.match(
                r"\W{0,12}(?:(?:is|are|was|were)\W+)?"
                r"(?:absent|negative|denied|not present)\b",
                suffix,
                flags=re.IGNORECASE,
            )
            uncertainty = re.search(
                r"\b(?:don['’]t|doesn['’]t|didn['’]t|do not|does not|did not)\b"
                r".{0,35}\b(?:know|say|mention|tell)\b.{0,25}\b(?:if|whether|any)\b"
                r"|\b(?:not sure|unclear)\b.{0,25}\b(?:if|whether)\b",
                fragment[max(0, match.start() - 100) : match.end() + 40],
                flags=re.IGNORECASE,
            )
            informational = re.search(
                r"\b(?:from what i(?:'ve| have) learned\b.{0,80}|"
                r"symptoms? (?:are|include)|signs? (?:are|include)|"
                r"worried about|concerned about|afraid of|could this be|"
                r"is this (?:a |an )?)\W*$",
                negation_scope,
                flags=re.IGNORECASE,
            )
            if (
                not negation
                and not postposed_negation
                and not uncertainty
                and not informational
            ):
                return True
    return False


def _positive_red_flag(question_id: str, fragment: str) -> bool:
    lowered = fragment.lower().replace("’", "'")
    if re.match(
        r"^\s*from what i(?:'ve| have) learned\b.{0,160}\bsymptoms?\b",
        lowered,
    ):
        return False
    patterns = QUESTION_PATTERNS[question_id]
    if not _has_positive_match(fragment, patterns):
        return False
    if question_id == "red_flag.bleeding":
        if re.search(
            r"\b(?:family history|runs? in (?:the|my) family|hereditary)\b",
            lowered,
        ):
            return False
        return _has_positive_match(
            fragment,
            (
                r"\b(?:heav(?:y|ier|iest)|uncontrolled|profuse|nonstop|severe|"
                r"significant|a lot of)\b"
                r".{0,30}\b(?:bleed|blood)\w*\b",
                r"\b(?:haemorrhag(?:e|ed|ing)|hemorrhag(?:e|ed|ing)|"
                r"black stool|vomit(?:ing)? blood|"
                r"puk(?:ed|ing) up blood|ha?ematemesis|blood loss|"
                r"soaking (?:a )?pad|won't stop bleeding)\w*\b",
                r"\b(?:bleed|blood)\w*\b(?:(?!\b(?:but|however|not)\b).){0,45}"
                r"\b(?:heav(?:y|ier|iest)|"
                r"uncontrolled|profuse|nonstop|severe|significant|large|big)\b",
            ),
        )
    if question_id == "red_flag.consciousness":
        return bool(
            re.search(
                r"\b(?:fainted|fainting|passed out|unconscious|somnolent|"
                r"difficult to wake|very drowsy|barely conscious|"
                r"can't stay conscious|cannot stay conscious|losing consciousness|"
                r"slipping in and out of (?:consciousness|it)|"
                r"barely keep (?:my|their|his|her) eyes open|"
                r"struggling to stay awake|black(?:ing|ed) out)\b",
                lowered,
            )
        )
    if question_id == "red_flag.neurologic" and re.search(
        r"\bconfused (?:about|as to|why)\b", lowered
    ):
        return _has_positive_match(
            fragment,
            (
                r"\b(?:seizure|new weakness|one-sided|slurred speech|"
                r"slurr(?:ing|ed) (?:her|his|their|my|the) (?:words|speech)|"
                r"speech (?:is|sounds) slurred|trouble speaking|facial droop|"
                r"vision loss|blurred vision|"
                r"paralys|sudden severe headache)\w*\b",
            ),
        )
    if question_id == "red_flag.sepsis" and re.search(
        r"\b(?:worried|concerned|question|wondering).{0,35}\bsepsis\b", lowered
    ):
        return False
    if question_id == "red_flag.mental_health":
        warning = re.search(
            r"\b(?:content warning|trigger warning|history or fear of suicide|"
            r"fear of suicide|if anyone has a history)\b|\btw\s*:",
            lowered,
        )
        first_person_signal = re.search(
            r"\b(?:i(?:'ve| have)? (?:attempted|attenpted|tried)|"
            r"i (?:am|feel) suicidal|i (?:want|plan) to (?:kill|harm) myself)\b",
            lowered,
        )
        if warning and not first_person_signal:
            return False
    return True


def _opening_fragments(fragments: list[str]) -> list[str]:
    selected: list[str] = []
    characters = 0
    for fragment in fragments:
        if selected and characters >= 140:
            break
        if len(selected) >= 3:
            break
        selected.append(fragment)
        characters += len(fragment)
    return selected


def _speaker_role(opening: str) -> str:
    lowered = opening.lower().replace("’", "'")
    if re.search(
        r"\b(?:my patient|my [^.\n]{1,40}\bpatient|one of my patients|"
        r"caring for a patient|"
        r"i have (?:an? |the |this )?[^.\n]{0,60}\bpatient|"
        r"i am (?:a|an|the) (?:doctor|physician|nurse|fellow|obstetrician|resident)|"
        r"i'm (?:a|an|the) (?:doctor|physician|nurse|fellow|obstetrician|resident))\b",
        lowered,
    ):
        return "clinician"
    caregiver = re.search(
        r"(?:^|[.!?]\s+)(?:okay,?\s*)?(?:so\s*)?"
        r"(?:my|the) (?:husband|wife|partner|boyfriend|girlfriend|gf|bf|"
        r"child|kid|son|daughter|mother|father|mom|dad|sister|brother|"
        r"sibling|friend|roommate)\b"
        r"|\bpatient is my (?:husband|wife|partner|boyfriend|girlfriend|"
        r"child|kid|son|daughter|mother|father|mom|dad|sister|brother|"
        r"sibling|friend|roommate)\b"
        r"|\bmy (?:husband|wife|partner|boyfriend|girlfriend|child|kid|"
        r"son|daughter|mother|father|mom|dad|sister|brother|sibling|"
        r"friend|roommate)\s+(?:is|has|was|who|aged|age\b|"
        r"\d{1,3}\s*(?:years?[- ]old|yo\b|y/o\b|[mMfF]\b))"
        r"|\bi (?:have|am asking about|am worried about) (?:a|my) "
        r"(?:roommate|friend|sibling|child|kid) (?:who|with)\b"
        r"|\binside my (?:girlfriend|gf|boyfriend|bf)\b"
        r"|^(?:bf|gf|boyfriend|girlfriend|husband|wife|partner|mother|"
        r"father|mom|dad|son|daughter|sister|brother)\s+(?:with|has|is|was)\b",
        lowered,
    )
    third_person_name = re.search(
        r"^(?!(?:What|This|That|There|When|Where|Why|How|Who|Which|It)\b)"
        r"[A-Z][a-z]{1,30}\s+(?:has|had|is|was)\b",
        opening,
    )
    self_symptom = re.search(
        r"\b(?:i have|i've had|i feel|i'm feeling|i am feeling|"
        r"i started|i was diagnosed|my (?:pain|head|chest|stomach|symptoms?))\b",
        lowered,
    )
    caregiver_start = (
        caregiver.start()
        if caregiver
        else third_person_name.start()
        if third_person_name
        else None
    )
    if caregiver_start is not None:
        if self_symptom is None or caregiver_start <= self_symptom.start():
            return "caregiver"
    return "patient"


def _profile(text: str, opening: str) -> dict[str, Any]:
    speaker_role = _speaker_role(opening)
    if speaker_role != "patient":
        return {"age": None, "sex": "unknown", "speaker_role": speaker_role}

    scope = text[:800]
    candidates: list[tuple[int, int, str]] = []
    paired_patterns = (
        r"\b(?P<age>\d{1,3})\s*(?:[- /]?\s*(?:yo|y/o|years?[- ]old))?"
        r"\s*(?P<sex>male|female|man|woman|m|f)\b",
        r"\b(?P<sex>male|female|man|woman|m|f)\s*[- /]?\s*(?P<age>\d{1,3})\b",
    )
    for pattern in paired_patterns:
        for match in re.finditer(pattern, scope, flags=re.IGNORECASE):
            age = int(match.group("age"))
            if 0 < age <= 120:
                candidates.append((match.start(), age, match.group("sex").lower()))
    if candidates:
        _, age, raw_sex = min(candidates, key=lambda value: value[0])
        sex = "female" if raw_sex in {"f", "female", "woman"} else "male"
        return {"age": age, "sex": sex, "speaker_role": speaker_role}

    age_match = re.search(r"\bage\s*[:=-]\s*(\d{1,3})\b", scope, re.IGNORECASE)
    sex_match = re.search(
        r"\bsex\s*[:=-]\s*(male|female)\b"
        r"|\b(?:i am|i'm)\s+(?:a\s+)?(man|woman|male|female)\b",
        scope,
        re.IGNORECASE,
    )
    age = int(age_match.group(1)) if age_match and 0 < int(age_match.group(1)) <= 120 else None
    raw_sex = next((value for value in sex_match.groups() if value), None) if sex_match else None
    sex = (
        "female"
        if raw_sex and raw_sex.lower() in {"female", "woman"}
        else "male"
        if raw_sex
        else "unknown"
    )
    return {"age": age, "sex": sex, "speaker_role": speaker_role}


def _presentation_group(text: str) -> str:
    scores = {
        group: sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)
        for group, patterns in PRESENTATION_GROUPS.items()
    }
    best_score = max(scores.values(), default=0)
    if best_score == 0:
        return "general_other"
    return sorted(group for group, score in scores.items() if score == best_score)[0]


def _queryable_facts(
    text: str,
    catalog: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    fragments = _segments(text)
    opening_parts = _opening_fragments(fragments)
    opening = " ".join(opening_parts)
    routed: dict[str, list[str]] = defaultdict(list)
    routed["concern.summary"].extend(opening_parts)

    for fragment in fragments:
        matched = False
        for question_id, patterns in QUESTION_PATTERNS.items():
            if _matches(fragment, patterns) and fragment not in routed[question_id]:
                routed[question_id].append(fragment)
                matched = True
        if not matched and fragment not in opening_parts:
            routed["context.additional"].append(fragment)

    question_ids = [str(question["id"]) for question in catalog["questions"]]
    unknown_response = str(catalog["unknown_response"])
    facts: list[dict[str, Any]] = []
    for question_id in question_ids:
        evidence = routed.get(question_id, [])
        facts.append(
            {
                "question_id": question_id,
                "status": "known" if evidence else "unknown",
                "answer": " ".join(evidence) if evidence else unknown_response,
                "evidence_text": evidence,
            }
        )

    red_flags: list[dict[str, Any]] = []
    for question_id, flag_id in RED_FLAG_METADATA.items():
        positive = [
            fragment
            for fragment in routed.get(question_id, [])
            if _positive_red_flag(question_id, fragment)
        ]
        if positive:
            red_flags.append(
                {
                    "id": flag_id,
                    "question_id": question_id,
                    "evidence_text": [positive[0]],
                    "derivation": "unreviewed_lexical_rule_v1",
                }
            )
    return opening, facts, red_flags


def _label_basis(row: Mapping[str, Any]) -> str:
    if str(row["mapping_method"]) == "median":
        return "released_five_physician_panel_median"
    if str(row["dataset"]) == "healthbench" and str(row["mapping_method"]) == "direct":
        return "healthbench_physician_agreed_emergent_category"
    return "released_direct_source_mapping"


def _unsafe_dispositions(gold: str) -> list[str]:
    return [label for label in LABELS if SEVERITY[label] < SEVERITY[gold]]


def _candidate(row: Mapping[str, Any], catalog: Mapping[str, Any]) -> dict[str, Any]:
    source_text = str(row["normalized_prompt_text"])
    user_text = extract_user_text(str(row["dataset"]), source_text)
    opening, facts, red_flags = _queryable_facts(user_text, catalog)
    known_count = sum(fact["status"] == "known" for fact in facts)
    fragment_count = len(_segments(user_text))
    information_score = (
        known_count * 5
        + min(fragment_count, 25)
        + min(len(user_text) // 200, 12)
        - max(0, (len(user_text) - 3000) // 250)
    )
    return {
        "row": dict(row),
        "user_text": user_text,
        "opening": opening,
        "facts": facts,
        "red_flags": red_flags,
        "presentation_group": _presentation_group(user_text),
        "information_score": information_score,
    }


def _diverse_take(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_group[str(candidate["presentation_group"])].append(candidate)
    for rows in by_group.values():
        rows.sort(
            key=lambda item: (
                -int(item["information_score"]),
                str(item["row"]["case_id"]),
            )
        )

    selected: list[dict[str, Any]] = []
    while len(selected) < count and by_group:
        ordered_groups = sorted(
            by_group,
            key=lambda group: (
                -int(by_group[group][0]["information_score"]),
                group,
            ),
        )
        for group in ordered_groups:
            selected.append(by_group[group].pop(0))
            if not by_group[group]:
                del by_group[group]
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} eligible candidates for quota {count}")
    return selected


def _make_card(candidate: Mapping[str, Any], case_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    row = candidate["row"]
    gold = str(row["normalized_label"])
    red_flags = list(candidate["red_flags"])
    opening = str(candidate["opening"])
    required = sorted(
        {
            str(flag["question_id"])
            for flag in red_flags
            if not any(
                str(evidence) in opening for evidence in flag["evidence_text"]
            )
        }
    )
    source_text = str(row["normalized_prompt_text"])
    user_text = str(candidate["user_text"])
    maximum_questions = int(config["maximum_questions"])
    unknown_response = str(config["unknown_response"])
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "intended_use": "evaluation_only",
        "training_allowed": False,
        "provenance": {
            "benchmark_case_id": str(row["case_id"]),
            "source_dataset": str(row["dataset"]),
            "source_id": str(row["source_id"]),
            "source_text_sha256": _sha256_text(source_text),
            "benchmark_artifact": str(config["benchmark_path"]),
            "label_basis": _label_basis(row),
            "mapping_method": str(row["mapping_method"]),
            "transformation": TRANSFORMATION_VERSION,
        },
        "opening_utterance": opening,
        "queryable_facts": list(candidate["facts"]),
        "gold_acuity": gold,
        "red_flags": red_flags,
        "required_question_ids": required,
        "unsafe_dispositions": _unsafe_dispositions(gold),
        "stratum": {
            "acuity": gold,
            "source_dataset": str(row["dataset"]),
            "presentation_group": str(candidate["presentation_group"]),
        },
        "patient_profile": _profile(user_text, str(candidate["opening"])),
        "simulator": {
            "maximum_questions": maximum_questions,
            "unknown_response": unknown_response,
            "response_policy": "question_id_lookup",
        },
        "review_status": {
            "reference_label": "source_derived",
            "case_content": "automated_checked",
            "requires_clinician_review": True,
        },
    }


def _counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def load_case_cards(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or default_seed_output_dir() / "case_cards.jsonl"
    cards: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Case card at {path}:{line_number} is not an object")
            cards.append(value)
    return cards


def build_seed_set(
    *,
    config_path: Path | None = None,
    benchmark_path: Path | None = None,
    output_dir: Path | None = None,
) -> SeedBuildResult:
    root = project_root()
    config_path = config_path or default_seed_config_path()
    config = _load_yaml_mapping(config_path, "seed config")
    if config.get("schema_version") != "interactive-seed-config/v1":
        raise ValueError("Unsupported interactive seed config version")
    if config.get("intended_use") != "evaluation_only" or config.get("training_allowed") is not False:
        raise ValueError("Seed v1 must remain evaluation-only and training-disallowed")

    question_path = root / str(config["question_catalog_path"])
    catalog = load_question_catalog(question_path)
    action_catalog_path = root / str(config["action_catalog_path"])
    action_catalog = _load_yaml_mapping(action_catalog_path, "action catalog")
    case_schema_path = root / str(config["case_schema_path"])
    action_schema_path = root / str(config["action_schema_path"])
    case_schema = load_json_schema(case_schema_path)
    action_schema = load_json_schema(action_schema_path)
    _validate_contract_alignment(
        question_catalog=catalog,
        action_catalog=action_catalog,
        case_schema=case_schema,
        action_schema=action_schema,
    )
    configured_unknown = str(config["unknown_response"])
    if configured_unknown != str(catalog["unknown_response"]):
        raise ValueError("Seed config and question catalog unknown responses differ")

    benchmark_path = benchmark_path or root / str(config["benchmark_path"])
    output_dir = output_dir or default_seed_output_dir()
    frame = pd.read_parquet(benchmark_path).fillna("")
    required_columns = {
        "case_id",
        "dataset",
        "source_id",
        "normalized_prompt_text",
        "normalized_label",
        "original_label",
        "mapping_method",
        "split",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Benchmark is missing seed columns: {sorted(missing)}")

    selection = config["selection"]
    labels = tuple(str(value) for value in selection["labels"])
    if labels != LABELS or int(selection["cases_per_label"]) != 25:
        raise ValueError("Seed v1 selection contract is exactly 25 cases per A/B/C/D")
    allowed = {str(value) for value in selection["allowed_datasets"]}
    minimum = int(selection["minimum_user_text_characters"])
    maximum = int(selection["maximum_user_text_characters"])
    exclusion_patterns = [str(value) for value in selection.get("exclusion_patterns", [])]

    eligible = frame[
        (frame["split"] == str(selection["split"]))
        & frame["normalized_label"].isin(labels)
        & frame["dataset"].isin(allowed)
    ]
    candidates: list[dict[str, Any]] = []
    for row in eligible.to_dict(orient="records"):
        candidate = _candidate(row, catalog)
        length = len(str(candidate["user_text"]))
        excluded = any(
            re.search(pattern, str(candidate["user_text"]))
            for pattern in exclusion_patterns
        )
        if minimum <= length <= maximum and not excluded:
            candidates.append(candidate)

    candidate_counts = _counter_dict(
        f"{item['row']['normalized_label']}:{item['row']['dataset']}" for item in candidates
    )
    selected: list[dict[str, Any]] = []
    source_quotas = selection["source_quotas"]
    for label in LABELS:
        quota_total = 0
        for dataset, quota_value in source_quotas[label].items():
            quota = int(quota_value)
            quota_total += quota
            pool = [
                item
                for item in candidates
                if str(item["row"]["normalized_label"]) == label
                and str(item["row"]["dataset"]) == str(dataset)
            ]
            selected.extend(_diverse_take(pool, quota))
        if quota_total != 25:
            raise ValueError(f"Source quotas for {label} sum to {quota_total}, not 25")

    if len(selected) != 100:
        raise ValueError(f"Selected {len(selected)} cases, expected exactly 100")
    benchmark_ids = [str(item["row"]["case_id"]) for item in selected]
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("Seed selection contains duplicate benchmark cases")

    seed_id = str(config["seed_id"])
    cards: list[dict[str, Any]] = []
    for index, candidate in enumerate(
        sorted(
            selected,
            key=lambda item: _sha256_text(
                f"{seed_id}|{item['row']['case_id']}"
            ),
        ),
        1,
    ):
        case_id = f"acuity-interactive-v1-{index:03d}"
        cards.append(_make_card(candidate, case_id, config))

    output_dir.mkdir(parents=True, exist_ok=True)
    case_cards_path = output_dir / "case_cards.jsonl"
    manifest_path = output_dir / "manifest.json"
    _write_jsonl(case_cards_path, cards)

    by_label_source = _counter_dict(
        f"{card['gold_acuity']}:{card['provenance']['source_dataset']}" for card in cards
    )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "builder_version": __version__,
        "seed_id": str(config["seed_id"]),
        "case_schema_version": CASE_SCHEMA_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "selection_version": str(config["selection_version"]),
        "intended_use": "evaluation_only",
        "training_allowed": False,
        "case_count": len(cards),
        "label_counts": _counter_dict(str(card["gold_acuity"]) for card in cards),
        "source_counts": _counter_dict(str(card["provenance"]["source_dataset"]) for card in cards),
        "label_source_counts": by_label_source,
        "presentation_group_counts": _counter_dict(str(card["stratum"]["presentation_group"]) for card in cards),
        "candidate_counts_by_label_source": candidate_counts,
        "reference_label_basis_counts": _counter_dict(str(card["provenance"]["label_basis"]) for card in cards),
        "clinician_content_reviewed_cases": sum(
            card["review_status"]["case_content"] == "clinician_reviewed"
            for card in cards
        ),
        "config": {
            "path": _portable_path(config_path, root),
            "sha256": _sha256_file(config_path),
        },
        "question_catalog": {
            "path": _portable_path(question_path, root),
            "sha256": _sha256_file(question_path),
            "question_count": len(catalog["questions"]),
        },
        "action_catalog": {
            "path": _portable_path(action_catalog_path, root),
            "sha256": _sha256_file(action_catalog_path),
        },
        "schemas": {
            "case_card": {
                "path": _portable_path(case_schema_path, root),
                "sha256": _sha256_file(case_schema_path),
            },
            "action": {
                "path": _portable_path(action_schema_path, root),
                "sha256": _sha256_file(action_schema_path),
            },
        },
        "benchmark": {
            "path": _portable_path(benchmark_path, root),
            "sha256": _sha256_file(benchmark_path),
        },
        "sources_lock": {
            "path": "sources.lock.json",
            "sha256": _sha256_file(root / "sources.lock.json"),
            "dataset_to_source_name": {
                "healthbench": "healthbench_consensus",
                "pmr_reddit": "pmr_reddit_test",
                "reference_labels": "physician_labels",
            },
        },
        "files": {
            "case_cards.jsonl": {
                "bytes": case_cards_path.stat().st_size,
                "sha256": _sha256_file(case_cards_path),
            }
        },
        "limitations": [
            "Reference acuity is inherited from the released benchmark; transformed dialogue content has not been clinician reviewed.",
            "Queryable facts are deterministic text routing, not newly inferred or generated patient facts.",
            "A whole source fragment may be reused under multiple question IDs; turn-efficiency and consultation-latency conclusions require clinician fact splitting first.",
            "The set is held-out evaluation data and must not be used for training.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return SeedBuildResult(
        output_dir=output_dir,
        case_cards_path=case_cards_path,
        manifest_path=manifest_path,
        case_count=len(cards),
        manifest=manifest,
    )


def validate_seed_set(
    *,
    case_cards_path: Path | None = None,
    manifest_path: Path | None = None,
    benchmark_path: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    root = project_root()
    config_path = config_path or default_seed_config_path()
    config = _load_yaml_mapping(config_path, "seed config")
    question_path = root / str(config["question_catalog_path"])
    action_catalog_path = root / str(config["action_catalog_path"])
    case_schema_path = root / str(config["case_schema_path"])
    action_schema_path = root / str(config["action_schema_path"])
    catalog = load_question_catalog(question_path)
    action_catalog = _load_yaml_mapping(action_catalog_path, "action catalog")
    case_schema = load_json_schema(case_schema_path)
    action_schema = load_json_schema(action_schema_path)
    _validate_contract_alignment(
        question_catalog=catalog,
        action_catalog=action_catalog,
        case_schema=case_schema,
        action_schema=action_schema,
    )
    question_ids = [str(question["id"]) for question in catalog["questions"]]
    output_dir = default_seed_output_dir()
    case_cards_path = case_cards_path or output_dir / "case_cards.jsonl"
    manifest_path = manifest_path or output_dir / "manifest.json"
    benchmark_path = benchmark_path or root / str(config["benchmark_path"])
    cards = load_case_cards(case_cards_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark = pd.read_parquet(benchmark_path).fillna("").set_index("case_id")

    if len(cards) != 100 or int(manifest.get("case_count", -1)) != 100:
        raise ValueError("Interactive seed must contain exactly 100 cases")
    case_ids = [str(card.get("case_id")) for card in cards]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Interactive seed case IDs are not unique")
    benchmark_ids = [str(card["provenance"]["benchmark_case_id"]) for card in cards]
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("Interactive seed benchmark case IDs are not unique")

    clinician_reviewed_cases = 0
    for card in cards:
        validate_case_card(card, schema=case_schema)
        if card.get("schema_version") != CASE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported case schema for {card.get('case_id')}")
        if card.get("intended_use") != "evaluation_only" or card.get("training_allowed") is not False:
            raise ValueError(f"Training-data guard failed for {card['case_id']}")
        expected_review_status = {
            "reference_label": "source_derived",
            "case_content": "automated_checked",
            "requires_clinician_review": True,
        }
        if card.get("review_status") != expected_review_status:
            raise ValueError(f"Review-status drift for {card['case_id']}")
        clinician_reviewed_cases += int(
            card["review_status"]["case_content"] == "clinician_reviewed"
        )
        fact_ids = [str(fact["question_id"]) for fact in card["queryable_facts"]]
        if fact_ids != question_ids:
            raise ValueError(f"Question catalog mismatch for {card['case_id']}")
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError(f"Duplicate queryable facts for {card['case_id']}")
        known_ids = {fact["question_id"] for fact in card["queryable_facts"] if fact["status"] == "known"}
        if not set(card["required_question_ids"]).issubset(known_ids):
            raise ValueError(f"Required question lacks a known fact for {card['case_id']}")

        benchmark_id = str(card["provenance"]["benchmark_case_id"])
        if benchmark_id not in benchmark.index:
            raise ValueError(f"Missing benchmark source row {benchmark_id}")
        row = benchmark.loc[benchmark_id]
        if str(row["normalized_label"]) != str(card["gold_acuity"]):
            raise ValueError(f"Gold label drift for {card['case_id']}")
        if str(row["dataset"]) != str(card["provenance"]["source_dataset"]):
            raise ValueError(f"Source dataset drift for {card['case_id']}")
        if str(row["source_id"]) != str(card["provenance"]["source_id"]):
            raise ValueError(f"Source ID drift for {card['case_id']}")
        if str(row["mapping_method"]) != str(card["provenance"]["mapping_method"]):
            raise ValueError(f"Mapping-method drift for {card['case_id']}")
        expected_basis = _label_basis(row)
        if str(card["provenance"]["label_basis"]) != expected_basis:
            raise ValueError(f"Reference-label basis drift for {card['case_id']}")
        if expected_basis == "released_five_physician_panel_median":
            panel = [str(row.get(f"anon_label_{index}", "")) for index in range(1, 6)]
            if str(row["mapping_method"]) != "median" or any(not vote for vote in panel):
                raise ValueError(f"Incomplete physician panel for {card['case_id']}")
        elif expected_basis == "healthbench_physician_agreed_emergent_category":
            if str(row["original_label"]) != "emergent":
                raise ValueError(f"Direct HealthBench label drift for {card['case_id']}")
        source_text = str(row["normalized_prompt_text"])
        user_text = extract_user_text(str(row["dataset"]), source_text)
        if _sha256_text(source_text) != str(card["provenance"]["source_text_sha256"]):
            raise ValueError(f"Source digest drift for {card['case_id']}")
        valid_fragments = set(_segments(user_text))
        revealed_fragments: set[str] = set()
        for fact in card["queryable_facts"]:
            evidence = fact["evidence_text"]
            if fact["status"] == "known":
                if not evidence or any(fragment not in valid_fragments for fragment in evidence):
                    raise ValueError(f"Ungrounded evidence for {card['case_id']}:{fact['question_id']}")
                if fact["answer"] != " ".join(evidence):
                    raise ValueError(f"Fact answer drift for {card['case_id']}:{fact['question_id']}")
                revealed_fragments.update(str(fragment) for fragment in evidence)
            elif fact["status"] == "unknown":
                if evidence or fact["answer"] != catalog["unknown_response"]:
                    raise ValueError(f"Unknown response drift for {card['case_id']}:{fact['question_id']}")
            else:
                raise ValueError(f"Invalid fact status for {card['case_id']}")
        omitted_fragments = sorted(
            fragment
            for fragment in valid_fragments
            if fragment not in revealed_fragments
            and fragment not in str(card["opening_utterance"])
        )
        if omitted_fragments:
            raise ValueError(
                f"Source fragments are unreachable for {card['case_id']}: "
                f"{omitted_fragments[:2]}"
            )
        expected_unsafe = _unsafe_dispositions(str(card["gold_acuity"]))
        if card["unsafe_dispositions"] != expected_unsafe:
            raise ValueError(f"Unsafe-disposition drift for {card['case_id']}")

    label_counts = _counter_dict(str(card["gold_acuity"]) for card in cards)
    if label_counts != {label: 25 for label in LABELS}:
        raise ValueError(f"Seed label balance drift: {label_counts}")
    expected_sources = {
        f"{label}:{dataset}": int(count)
        for label, datasets in config["selection"]["source_quotas"].items()
        for dataset, count in datasets.items()
    }
    observed_sources = _counter_dict(
        f"{card['gold_acuity']}:{card['provenance']['source_dataset']}" for card in cards
    )
    if observed_sources != dict(sorted(expected_sources.items())):
        raise ValueError(f"Seed source quotas drift: {observed_sources}")
    source_counts = _counter_dict(
        str(card["provenance"]["source_dataset"]) for card in cards
    )
    presentation_counts = _counter_dict(
        str(card["stratum"]["presentation_group"]) for card in cards
    )
    basis_counts = _counter_dict(
        str(card["provenance"]["label_basis"]) for card in cards
    )
    manifest_expectations = {
        "label_counts": label_counts,
        "label_source_counts": observed_sources,
        "source_counts": source_counts,
        "presentation_group_counts": presentation_counts,
        "reference_label_basis_counts": basis_counts,
    }
    for field, expected in manifest_expectations.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Manifest {field} differs from case cards")
    if manifest.get("intended_use") != "evaluation_only" or manifest.get("training_allowed") is not False:
        raise ValueError("Manifest training-data guard failed")
    if manifest.get("clinician_content_reviewed_cases") != clinician_reviewed_cases:
        raise ValueError("Manifest clinician-review count differs from case cards")
    artifact_checks = [
        (config_path, manifest["config"]["sha256"], "seed config"),
        (question_path, manifest["question_catalog"]["sha256"], "question catalog"),
        (action_catalog_path, manifest["action_catalog"]["sha256"], "action catalog"),
        (case_schema_path, manifest["schemas"]["case_card"]["sha256"], "case schema"),
        (action_schema_path, manifest["schemas"]["action"]["sha256"], "action schema"),
        (benchmark_path, manifest["benchmark"]["sha256"], "benchmark"),
        (root / "sources.lock.json", manifest["sources_lock"]["sha256"], "sources lock"),
    ]
    for artifact_path, expected_digest, label in artifact_checks:
        if _sha256_file(artifact_path) != str(expected_digest):
            raise ValueError(f"Manifest {label} digest differs from artifact")
    expected_file_digest = str(manifest["files"]["case_cards.jsonl"]["sha256"])
    if _sha256_file(case_cards_path) != expected_file_digest:
        raise ValueError("Case-card artifact digest differs from manifest")
    with tempfile.TemporaryDirectory(prefix="acuitybench-seed-validate-") as temporary:
        rebuilt = build_seed_set(
            config_path=config_path,
            benchmark_path=benchmark_path,
            output_dir=Path(temporary),
        )
        if rebuilt.case_cards_path.read_bytes() != case_cards_path.read_bytes():
            raise ValueError("Case cards differ from a deterministic rebuild")
        if rebuilt.manifest_path.read_bytes() != manifest_path.read_bytes():
            raise ValueError("Seed manifest differs from a deterministic rebuild")
    return {
        "case_count": len(cards),
        "label_counts": label_counts,
        "label_source_counts": observed_sources,
        "question_count": len(question_ids),
        "training_allowed": False,
        "clinician_content_reviewed_cases": clinician_reviewed_cases,
        "case_cards_sha256": expected_file_digest,
    }
