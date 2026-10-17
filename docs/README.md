# Documentation index

| File | Content |
|---|---|
| `01_coverage_audit.md` | Paper 3.12 ledger vs paper 3.23 docx: coverage matrix, numerical cross-check, version drift, remaining last-digit issue |
| `02_existing_repos.md` | Existing GitHub repositories (Supplementary, private deployment) and their structure |
| `03_dataset_protocol.md` | Dataset A/B source, cleaning, grouping thresholds, fixed split, augmentation, restoration |
| `04_dataset_group_summary.md` | Dataset names, classes, batches/sequence families, group IDs, group sizes, split totals |
| `05_weight_inventory.md` | 12 best.pt + pretrained: parameters, best epoch, metrics, SHA256, verification |
| `06_reproduction.md` | Environment, data placement, training/evaluation commands, expected values, pitfalls |
| `07_known_issues.md` | B.2 last digit, annotation split field, 3.23 vs 3.13, archival paths, metric caveats |
| `08_source_inventory.md` | FSDConv/SHEB code locations, registration, configs, train/eval/data scripts, run mapping |
| `09_results_index.md` | Paper table <-> result file <-> checkpoint |
| `10_filename_manifests.md` | Complete filename lists for both datasets, with augmentation category/operation/parameters parsed from names |
| `11_manuscript_fix_B2.md` | Word steps and comment text for the Appendix B.2 SD correction (0.334 -> 0.335) |
| `12_data_availability.md` | Ready-to-paste Data availability and Supplementary statements, plus separate-vs-merged analysis |

Recommended reading:

1. `01` to see what the 3.12 ledger covers and what it does not;
2. `03`/`04` for the fixed splits and groups;
3. `05`/`09` to find checkpoints and results;
4. `06` to reproduce;
5. `07` for differences and caveats.