# Experiments log

Record of what each experiment did, the config, the numbers, and the conclusion. One markdown
file per experiment. Large binary outputs (models, caches) are not stored here — only the code
(on git branches) and the results.

| experiment | branch | outcome |
|---|---|---|
| [On-policy GKD](onpolicy_gkd.md) | `onpolicy-gkd` | no gain (~38/228); teacher endorses student's own tokens 96.6%, nothing to correct |
