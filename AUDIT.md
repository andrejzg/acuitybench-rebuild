# Upstream reproducibility audit

This rebuild treats the checksum-pinned `normalized_survey_labels.csv` as the
canonical released annotation artifact. It independently recomputes every
case's Remove count, endorsed ordinal median, and average pairwise set-overlap
distance from `anon_label_1` through `anon_label_5` during each build.

## Annotation threshold inconsistency

The released label CSV is internally consistent with the manuscript's strict
ambiguity rule, `avg_sd > 0.75`. After 76 pre-screening exclusions and eight
majority-Remove exclusions, 667 accepted cases split into 450
physician-consensus and 217 ambiguous cases.

Some upstream documentation and regeneration utilities instead specify a
threshold of `1.0`, as does the committed upstream agreement report. That
threshold produces 477 consensus and 190 ambiguous cases. Twenty-seven cases
lie in `(0.75, 1.0]`. This project therefore does not regenerate physician
labels with the upstream CLI default; it consumes the published label CSV by
exact SHA-256 and verifies the paper's 450/217 split from the individual votes.

## Anonymized rater identifier discrepancy

The released CSV contains 22 distinct anonymized rater identifiers
(`rater_01` through `rater_22`), while the manuscript reports 20 unique
physicians. Every one of the 675 surveyed cases nevertheless contains five
labels from five distinct identifiers, and every survey subset contains five
identifiers. Across eight subsets there are 40 rater-survey assignments: 18
identifiers occur in two subsets and four occur in one (`rater_03`,
`rater_12`, `rater_15`, and `rater_16`).

The public data cannot establish whether these are identity aliases,
replacement physicians, or a manuscript-count discrepancy because the private
name-to-ID mapping and raw Qualtrics exports are not released. This does not
alter per-case aggregation. Claims about the number of unique physicians or
longitudinal per-rater behaviour require clarification from the authors.

## Agreement statistic

The upstream implementation computes observed disagreement as an unweighted
mean of per-case pairwise means. Canonical Krippendorff coincidence weighting
weights units by usable rating count, which differs when a case has one or two
Remove votes. The upstream custom implementation reports alpha values of
0.545005 overall, 0.863404 for consensus, and 0.040948 for ambiguous cases;
canonical coincidence weighting on the released votes gives approximately
0.550121, 0.864972, and 0.049129. This project preserves the released splits
but refers to those paper values as the reported custom implementation.

## Threshold interpretation

The manuscript says `avg_sd > 0.75` requires more than seven of ten rater pairs
to disagree. That is not generally true because the distance is squared and
can take values 0, 1, 4, or 9; one distance-9 pair can cross the threshold.
The released labels and this rebuild apply the numeric average-distance rule
correctly, so this wording issue does not change the reconstructed benchmark.

## Reproduction boundary

The public release is sufficient to verify all five votes, exclusions,
aggregates, splits, texts, and reference case IDs. It is not sufficient to
replay Qualtrics decoding, independently establish physician identities, or
reproduce the original recruitment process. Those require private inputs.

## Legacy evaluation latency

The original evaluation runner started `latency_ms` before acquiring its local
concurrency semaphore. In the completed GPT-5-mini run that field therefore
contains scheduler queueing, every provider attempt, and retry backoff; it is
not API request latency. The OpenAI `openai-processing-ms` header remains
available for every successful legacy call, but it is provider-defined and
cannot reconstruct request wall time or TTFT.

Schema-v2 runs stream responses and separately record local queue wait,
terminal and cumulative request wall time, retry sleep, first visible-token
time, stream tail, total logical duration, and provider processing. Reports
retain legacy totals under `total_duration_ms` and leave unrecoverable fields
null rather than estimating them. Billing summaries sum every tracked attempt,
including length retries, and expose usage coverage whenever a terminated
stream did not deliver final token accounting.

## Paired July 2026 latency run

The paired GPT-5-mini and GPT-5.4 runs deliberately diverge from the paper's
transport so TTFT can be measured: both stream at concurrency 20 on the
`default` service tier. The paper reports no concurrency; it does not report
streaming or service tier, while its public adapter is non-streaming. The
manifests keep those reconstruction choices separate from paper-reported
temperature 1, five samples per case and format, and the 4,096 completion-token
maximum.

The paper does not report reasoning effort or a separate reasoning-token
budget. The reruns explicitly pin GPT-5-mini to `medium` and GPT-5.4 to `none`,
record the documentation source and access date, and treat 4,096 as the shared
hidden-reasoning-plus-visible-output cap. Observed reasoning-token telemetry is
complete: 5,429,450 billed target tokens for mini and verified zero for GPT-5.4.

Three mini conversational calls in the first invocation ended at the 4,096
cap while an older in-memory runner path still treated non-empty truncations as
failures. The corrected resume retained capped non-empty output as evaluable
without increasing the cap. The final run contains 9,140 successful logical
targets and 9,143 billed target attempts; the three superseded attempts added
$0.0248265. Attempt exports preserve the full history, while latency summaries
use each successful parent row's linked execution.
