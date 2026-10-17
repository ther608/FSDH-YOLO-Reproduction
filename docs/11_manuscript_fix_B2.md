# 11. Manuscript correction: Appendix B.2 SD (0.334 -> 0.335)

## 1. Exact location

File: `论文3.23 CEA 参考文献减少至35.docx` (paper 3.23 CEA manuscript)

Section: **Appendix B.2, "Comparison Between SPDConv and FSDConv"**, table.

Row: **SPDConv+SHEB** (the fourth data row).
Column: **mAP50-95 / %**.

Current cell:

```text
66.649±0.334
```

Correct cell:

```text
66.649±0.335
```

Only the last digit of the sample SD changes. Do not change P, R, mAP50, Params,
FPS, or the row label.

## 2. Why

Per-seed mAP50-95 values for SPDConv+SHEB:

| seed | mAP50-95 |
|---|---|
| 0 | 66.701 |
| 1 | 66.291 |
| 2 | 66.954 |

- Mean = 66.648666... -> printed 66.649
- Sample SD (ddof=1) = 0.334584... -> **0.335**

The 3.12 experiment ledger and Supplementary Appendix C (C12) also give 0.335.
The current 3.23 row label is already correct (`SPDConv+SHEB`), so this is a
last-digit SD correction only.

## 3. Word steps

1. Open the `.docx` in Microsoft Word.
2. Turn on **Review > Track Changes** (recommended, so the correction is
   visible to co-authors).
3. Use **Ctrl+F** and search for `66.649` (the ± symbol may be hard to type).
4. Confirm you are in **Appendix B.2**, row **SPDConv+SHEB**, column
   **mAP50-95 / %**.
5. Select only the final `4` in `0.334` and replace it with `5`, so the cell
   reads `66.649±0.335`. Keep the `±` symbol and all other characters.
6. With the cell still selected, use **Review > New Comment** (or right-click >
   **New Comment**) and paste the comment in Section 4 below.
7. Save the file. If Track Changes is on, accept the change after co-author
   review, but keep the comment if the journal allows comments in the submitted
   file; otherwise resolve/delete it before submission and record the change in
   the response letter.

> If your Word version does not show "New Comment" in the right-click menu, use
> **Review > New Comment** in the ribbon. If the document is in compatibility
> mode, save a copy as `.docx` first.

## 4. Comment text (copy-paste)

Short version:

```text
Per-seed mAP50-95: 66.701 / 66.291 / 66.954; mean = 66.648666..., sample SD
(ddof=1) = 0.334584... -> 0.335. The 3.12 ledger and Supplementary Appendix C
also give 0.335; only the last digit changes.
```

Long version, if a fuller audit trail is preferred:

```text
Appendix B.2, SPDConv+SHEB, mAP50-95 SD. Per-seed values are 66.701, 66.291 and
66.954 (Supplementary Appendix C, C12). Mean = 66.648666..., sample SD with
ddof=1 = 0.334584..., which rounds to 0.335. The 3.12 experiment ledger also
reports 66.649±0.335. The row label SPDConv+SHEB is already correct; this edit
changes only the last digit of the SD (0.334 -> 0.335). No other cell changes.
```

## 5. What not to change

- Do **not** relabel the row to `FSDConv+SHEB`; 3.23 already has the correct
  `SPDConv+SHEB` label.
- Do **not** change the separate `FSDH-YOLO` row.
- Do **not** change mAP50, P, R, Params, or FPS.
- Do **not** change the 3.12 ledger; it already has 0.335.

## 6. After the edit

Regenerate any PDF/HTML export from the corrected `.docx`, and check that the
cell in Appendix B.2 reads `66.649±0.335` in the exported PDF. If the manuscript
is later renumbered, search for `66.649` again; the value is unique to this row.