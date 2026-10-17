# 12. Data availability statement: separate or merged with Supplementary materials?

## 0. Recommendation

**Keep them separate, with explicit cross-references.**

Use a concise **Data availability** section for the datasets, code, split
manifests, checkpoints and result repositories. Keep the existing
**Supplementary materials** section for Appendix C, the evaluation protocol and
the Android deployment package. This is cleaner for journal metadata, easier for
reviewers to find code/weights, and avoids a long final paragraph that mixes
"data" with "supplementary files".

If the journal's template only permits one section, use the merged version in
Section 4.

---

## 1. What the current manuscript says

The final paragraph of 3.23 currently says, in substance:

> Supplementary materials, including additional experimental results, evaluation
> protocols, statistical evaluation details, and Android deployment information,
> are available at https://github.com/ther608/FSDH-YOLO-Supplementary. The
> repository provides evaluation protocols and configurations, additional
> quantitative results, deployment measurements, and the exported ONNX model.
> The datasets are public and are not redistributed.

This is a **Supplementary materials** statement, not a formal Data availability
statement. It omits the reproduction repository, the fixed split manifests, the
PyTorch checkpoints and the dataset DOIs.

---

## 2. Separate version (recommended)

Place **Data availability** after the Conclusion/declarations and before the
References. Keep **Supplementary materials** at the end (or immediately after
Data availability). Exact text:

### Data availability

```text
Data availability

The two datasets used in this study are publicly available and are not
redistributed in this work. Dataset A is the RGB subset of the RGB-D lychee
dataset of Zhang et al. (2025), available at arXiv:2510.16800
(https://doi.org/10.48550/arXiv.2510.16800). Dataset B is the lichi-maturity
dataset of Zhao (2025), available at Mendeley Data
(https://doi.org/10.17632/c3rk9gv4w9.1).

The reproduction materials are available at
https://github.com/ther608/FSDH-YOLO-Reproduction. They include the modified
Ultralytics 8.4.11 source, model configurations, training and evaluation
scripts, fixed train/validation/test split manifests, annotation-cleaning and
grouping records, complete filename manifests with augmentation metadata, frozen
COCO test annotations, trained PyTorch checkpoints (best.pt for both datasets and
three random seeds), and the result tables.

Per-seed results for all experiment tables, the framework-neutral evaluation
protocol, and the Android deployment package (Kotlin application, ONNX models,
and raw on-device measurement records) are available at
https://github.com/ther608/FSDH-YOLO-Supplementary.

Code and trained checkpoints are released under AGPL-3.0; documentation, split
manifests, annotations and results are released under CC BY 4.0. Dataset images
remain under the licenses of the original dataset providers.
```

### Supplementary materials

```text
Supplementary materials

Supplementary material includes Appendix C (per-seed results for all experiment
tables), the evaluation protocol, statistical evaluation details, and the
Android deployment package (source code, ONNX models, and raw measurement
records). These materials are available at
https://github.com/ther608/FSDH-YOLO-Supplementary.
```

### Why separate is better

| Aspect | Separate | Merged |
|---|---|---|
| Journal compliance | Matches the usual Elsevier "Data availability" field/section | May be treated as a supplementary note rather than a data statement |
| Discoverability | Reviewers see data/code/weights in a dedicated section | Buried at the very end after the Appendix |
| Scope clarity | Separates datasets/code/weights from Appendix C/Android files | Mixes data, code and supplementary files in one paragraph |
| Length | Two short paragraphs | One long paragraph |
| Updates | Each repository can be updated independently | One paragraph must be edited when either repository changes |
| Duplication | No duplication; each repository has one role | Risk of repeating URLs/roles |

**Conclusion:** separate. Use Data availability for datasets + reproduction repo
+ checkpoints + result tables; use Supplementary materials for Appendix C +
protocol + Android package.

---

## 3. Where to insert in Word

1. Open the `.docx`.
2. Place the cursor after the **Declaration of generative AI and AI-assisted
   technologies** section and before **References** (or after the Conclusion,
   depending on the journal template).
3. Insert a heading **Data availability** (Heading 1 or the journal's required
   style) and paste the text from Section 2.
4. Keep the existing **Supplementary materials** heading at the end, but replace
   its paragraph with the version in Section 2 so the two statements do not
   overlap.
5. Check that both URLs are hyperlinked and that the dataset DOIs resolve.
6. If the journal requires the Data availability statement in a submission-system
   field as well, paste the same text there.

> Check the current Computers and Electronics in Agriculture (Elsevier) guide for
> authors before final submission. Elsevier journals commonly ask for a distinct
> Data availability statement; the exact heading/style may differ.

---

## 4. Merged version (only if the journal requires one section)

```text
Data availability and supplementary materials

The two datasets used in this study are publicly available and are not
redistributed. Dataset A is the RGB subset of the RGB-D lychee dataset of Zhang
et al. (2025), available at arXiv:2510.16800
(https://doi.org/10.48550/arXiv.2510.16800). Dataset B is the lichi-maturity
dataset of Zhao (2025), available at Mendeley Data
(https://doi.org/10.17632/c3rk9gv4w9.1).

Reproduction materials are available at
https://github.com/ther608/FSDH-YOLO-Reproduction: modified Ultralytics 8.4.11
source, model configurations, training/evaluation scripts, fixed split and
cleaning/grouping manifests, complete filename manifests with augmentation
metadata, frozen COCO annotations, trained PyTorch checkpoints for both datasets
and three seeds, and result tables.

Per-seed results, the evaluation protocol, and the Android deployment package
(Kotlin app, ONNX models, raw measurements) are available at
https://github.com/ther608/FSDH-YOLO-Supplementary.

Code and checkpoints are released under AGPL-3.0; documentation, split
manifests, annotations and results under CC BY 4.0. Dataset images remain under
the original dataset licenses.
```

### When merged is acceptable

- The journal template has a single "Data availability" or "Supplementary
  material" section and does not provide separate fields.
- The manuscript is close to the length limit and a second section would be
  awkward.
- The authors want one URL block for reviewers.

In that case, keep the heading **Data availability and supplementary materials**
so the statement is still discoverable.

---

## 5. Repository roles (for the response letter)

| Repository | Contents | Not included |
|---|---|---|
| `https://github.com/ther608/FSDH-YOLO-Reproduction` | Source, configs, train/eval scripts, fixed splits, cleaning/grouping/filename manifests, annotations, 12 best.pt, results | Dataset images, Android project |
| `https://github.com/ther608/FSDH-YOLO-Supplementary` | Appendix C per-seed results, evaluation protocol, Android app, ONNX, raw on-device records | Training code, PT checkpoints, dataset images |
| Dataset A DOI | `https://doi.org/10.48550/arXiv.2510.16800` | Images not redistributed |
| Dataset B DOI | `https://doi.org/10.17632/c3rk9gv4w9.1` | Images not redistributed |

---

## 6. Suggested response-to-reviewer wording

```text
We have added a separate Data availability statement. The public dataset
sources and DOIs are now stated explicitly. The reproduction repository
(https://github.com/ther608/FSDH-YOLO-Reproduction) provides the source code,
configurations, training/evaluation scripts, fixed split and cleaning/grouping
manifests, annotations, trained checkpoints and result tables. The supplementary
repository (https://github.com/ther608/FSDH-YOLO-Supplementary) provides the
per-seed Appendix C results, evaluation protocol and Android deployment package.
Dataset images are not redistributed and remain available from the original
public sources.
```