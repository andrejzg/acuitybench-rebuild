# Static-student data

This directory is reserved for training/development pools that are completely
separate from the 914-case AcuityBench benchmark and the interactive evaluation
seed. Nothing under `data/processed/` or `data/interactive/seed_v1/`, including
paraphrases or derived teacher outputs, may be used to populate it.

`synthetic_pilot_v0/` contains only the deterministic planning scaffold for a
20-case fictional pipeline check. No vignette has been generated, no provider
call has been made, and no record is ready for training.

`synthetic_pilot_v1/` contains the completed 200-case Fable generation and
dual-model blinded-label run. Its 131 machine-screened candidates are still
`training_allowed: false`; they are not a clinically reviewed training set.
