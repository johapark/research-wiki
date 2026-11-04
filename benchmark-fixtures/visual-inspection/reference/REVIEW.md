# Reference-annotation review

Prepared 2026-11-04 by the active coding assistant from source PDF renders,
including higher-resolution views for annotation. Independent human adjudication
is pending. No model endpoint was used to generate or grade reference answers.
Expectations describe what is visible in the supplied images.

Values and per-question regions are in [expectations.json](expectations.json).
The editable definitions are in [build.py](../build.py); rebuild after correcting
an annotation and revalidate hashes and packet identity. Do not change an input
image to silently agree with an expected answer.

| Cases | Source images | Main checks | Human review |
|---|---|---|---|
| v01 | [MGA p17](../assets/mga-p17@110.png) | HUMAN subcolumns; elapsed time; literal Memory (Mb) label | pending |
| v02, v11 | [MGA p10](../assets/mga-p10@110.png), [masked variant](../assets/mga-p10-masked@110.png) | Dataset groups, N50 exception, absent phase-switch rates; both hidden-value occurrences masked | pending |
| v08, v12 | [Caption p29](../assets/clinical-p29@110.png), [forest plate p35](../assets/clinical-p35@110.png) | Displayed RD sign and CI, outcome denominator versus enrollment; caption-only values absent | pending |
| v09 | [Caption p29](../assets/clinical-p29@110.png), [box plot p36](../assets/clinical-p36@110.png) | Both medians at month 4; caption p-value and n; observed min/max whiskers | pending |
| v13, v14 | [Tangermeme p2](../assets/tangermeme-p2@110.png) | Six printed timings; panel-c ten runs versus panel-d twenty; logarithmic divergence; purple ablation After; no printed exact median | pending |
| v15 | [Cell-comparison p5](../assets/cellbench-p5@110.png) | Zheng 68K row/column binding; F1 versus unlabeled percentages; gray untested cells; F1=1 is not full labeling | pending |
| v16 | [Cell-comparison p7](../assets/cellbench-p7@110.png) | Xin/Zheng 68K complexity extremes; gray encodes population count; no exact Xin bar label | pending |
| v17 | [DeepCRISPR p9](../assets/deepcrispr-p9@110.png) | Printed ROC-AUC .981/.929 versus PR-AUC .497/.247; distinct ROC and PR axes | pending |
| v18 | [DeepCRISPR p6](../assets/deepcrispr-p6@110.png) | Panel-g first/second methods; Spearman metric and absent exact value; panel-c Epoch/Loss axes | pending |
| v19 | [scAnnotate p7](../assets/scannotate-p7@110.png) | Figure 3 color/rank versus size/accuracy; PBMC.10Xv2, PBMC.10Xv3 and MTG ranks; no printed exact dot accuracy | pending |
| v20 | [scAnnotate p8](../assets/scannotate-p8@110.png) | Figure 6 black=False/light-gray=True; six methods; columns are selected cells; no row-specific exact count | pending |

- Table 3 prints `Memory (Mb)`. The fixture preserves this literal label; it
  does not adjudicate whether the authors intended bytes or bits.
- Figure 2's first forest row prints a negative risk difference and specific
  CI bounds. Copy the graphic rather than differing nearby prose. The numeric
  plate expectations require evidence from physical page 35.
- Figure 3 medians coincide with the month-4 tick. Its caption states observed
  min/max whiskers. The fixture does not resolve plot-versus-prose differences
  in IQR reporting.
- Two masks remove the target cell and its repeated value in the reference
  header; the neighboring comparator remains readable.
- Clinical plates were associated with captions through source-image review.
  This does not certify that the current automatic locator resolves them.
- Tangermeme contains two replicate counts in one caption, belonging to different
  panels. Preserve that association and do not estimate an exact median from a
  logarithmic box plot.
- Cell-comparison heatmap values are small but printed. Gray cells are untested,
  distinct from measured zero. A classifier's F1-score and its unlabeled-cell
  percentage are separate metrics, not interchangeable measures of coverage.
- DeepCRISPR's Figure 3 concerns off-target performance; Figure 2 concerns
  on-target performance. Keep figure/panel/metric associations intact. The
  independent-dataset bar has no exact value label.
- scAnnotate page 7 contains Figures 3 and 4; v19 asks only about Figure 3.
  Ranking is encoded by color, accuracy by dot size; legend sizes do not supply
  exact per-dot values. Figure 6 shows inconsistently annotated cells, not the
  whole dataset. Neither pixel widths nor nearby prose give a printed row count.

For human adjudication, record reviewer, date, source hash, changed answer and
reason here or in a versioned receipt. Pending review must not silently become
a claim of independent validation.
