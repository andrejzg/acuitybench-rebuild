# Data and licence notice

This repository contains original reconstruction code together with pinned
source snapshots, physician annotations, reconstructed benchmark files, and
model-evaluation artifacts. Inclusion in this private repository does not grant
a licence to any third-party dataset, annotation, or generated derivative.

The build downloads immutable, checksum-verified snapshots from these sources:

- OpenAI HealthBench: MIT.
- PortalPal PMR-Reddit and PMR-Synth: CC BY-NC 4.0.
- Ramaswamy et al. structured triage release: MIT repository; consult the
  source release for data terms.
- Semigran vignettes: the convenience JSONL repository is MIT, while the
  underlying BMJ article and supplement are CC BY-NC 4.0.
- AcuityBench physician annotations: released through an anonymous review
  repository whose README currently says that a licence is to be added.

Accordingly, generated data is intended for local, non-commercial research
until the AcuityBench authors publish final terms. Do not redistribute the
downloaded annotations or benchmark without reviewing every upstream licence.
The benchmark is an evaluation artifact, not a clinical decision system.

Keep the GitHub repository private while upstream annotation terms remain
pending. `sources.lock.json` identifies the origin, revision, checksum, and
known licence status of every source snapshot.
