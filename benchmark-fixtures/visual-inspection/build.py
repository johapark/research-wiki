"""Build portable visual fixtures from bundled PDFs; no network or model calls."""

from datetime import date
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pypdfium2

from researchwiki.pdf.render import _to_rgb, encode_png, render_page


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DPI = 110
SOURCES = {
    "mga": ("benchmark-fixtures/pdfs/zhang-2026-mga-a-tool-for-haplotype-mixed.pdf", "CC-BY-4.0; see benchmark-fixtures/LICENSES.md"),
    "clinical": ("benchmark-fixtures/pdfs/fonseca-2026-adjunctive-ibuprofen-in-pre-extensively-drug-resistant.pdf", "CC-BY-4.0; see benchmark-fixtures/LICENSES.md"),
    "tangermeme": ("benchmark-fixtures/pdfs/schreiber-2026-tangermeme-a-toolkit-for-understanding.pdf", "CC-BY-4.0; see benchmark-fixtures/LICENSES.md"),
    "cellbench": ("benchmark-fixtures/pdfs/abdelaal-2019-a-comparison-of-automatic-cell.pdf", "CC-BY-4.0; see benchmark-fixtures/LICENSES.md"),
    "deepcrispr": ("benchmark-fixtures/pdfs/chuai-2018-deepcrispr-optimized-crispr-guide-rna.pdf", "CC-BY-4.0; see benchmark-fixtures/LICENSES.md"),
    "scannotate": ("benchmark-fixtures/pdfs/ji-2023-scannotate-an-automated-cell-type-annotation.pdf", "CC-BY-4.0; see benchmark-fixtures/LICENSES.md"),
}
PAGES = {"mga": [10, 17], "clinical": [29, 35, 36], "tangermeme": [2],
         "cellbench": [5, 7], "deepcrispr": [6, 9], "scannotate": [7, 8]}
SOURCE_HASHES = {
    "clinical": "7298a9e00b1097bf01805ef81964624e338a32b27dc491d151b9eb7fc78ebc8e",
    "mga": "abdacdee6b5dc6b75fe398dc5d4557bd14d34b16eaddf4b04f71e3cd818fcdf0",
    "tangermeme": "19f22c33251482c7eba9f6e754980735ccdc03b44e0e4a5828ca6cc19c98aa6f",
    "cellbench": "89a626d7e6ba86e27c12230f697b7f15fcc996d359a505d376548f2f34ccf2d6",
    "deepcrispr": "da06e848ed7368244c93e8983b7d71d033a17cdcf4335b7b39fd85f02a32d72f",
    "scannotate": "c5fd7baa3ec9c02661b1034ac396ec684cd500b605df623c15a5950cb339e141",
}
LICENSE_EVIDENCE = {
    "mga": (1, "10.1186/s13059-026-04128-5"),
    "clinical": (1, "10.1038/s41467-026-75148-9"),
    "tangermeme": (5, "10.1038/s41592-026-03254-z"),
    "cellbench": (1, "10.1186/s13059-019-1795-z"),
    "deepcrispr": (1, "10.1186/s13059-018-1459-4"),
    "scannotate": (1, "10.1093/bioadv/vbad030"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def mask_sheep_cell(source, destination):
    """Mask both printed 105.2 occurrences so the reference header cannot leak it."""
    doc = pypdfium2.PdfDocument(str(source))
    boxes = []
    try:
        page = doc[9]
        tp = page.get_textpage()
        search = tp.search("105.2")
        try:
            while (hit := search.get_next()) is not None:
                first, count = hit
                chars = [tp.get_charbox(i) for i in range(first, first + count)]
                boxes.append((min(c[0] for c in chars), min(c[1] for c in chars),
                              max(c[2] for c in chars), max(c[3] for c in chars)))
        finally:
            search.close()
            tp.close()
        assert len(boxes) == 2, "Both the reference header and target cell must be masked"
        bitmap = page.render(scale=DPI / 72)
        try:
            rgb = _to_rgb(bitmap).copy()
        finally:
            bitmap.close()
        height = page.get_size()[1]
        pixels = []
        for left, bottom, right, top in boxes:
            x0 = max(0, int(np.floor(left * DPI / 72)) - 4)
            x1 = min(rgb.shape[1], int(np.ceil(right * DPI / 72)) + 4)
            y0 = max(0, int(np.floor((height - top) * DPI / 72)) - 4)
            y1 = min(rgb.shape[0], int(np.ceil((height - bottom) * DPI / 72)) + 4)
            rgb[y0:y1, x0:x1] = 255
            pixels.append([x0, y0, x1, y1])
        payload, _ = encode_png(rgb)
        destination.write_bytes(payload)
        return {"kind": "controlled_occlusion", "masked_boxes_pdf": boxes,
                "masked_boxes_pixels": pixels, "original_value": "105.2",
                "note": "White masks cover the SHEEP MGA N50 cell and matching reference-haplome N50. Other cells are untouched."}
    finally:
        doc.close()


def question(key, text, value, unit=None, *, status="observed", region="", critical=True, aliases=()):
    return {"id": key, "question": text}, {
        "question_id": key, "value": value, "unit": unit, "status": status,
        "region": region, "critical": critical, "accepted_values": list(aliases),
    }


def main():
    assets = {}
    manifest_sources = {}
    (OUT / "assets").mkdir(exist_ok=True)
    for name, (relative, license_note) in SOURCES.items():
        source = ROOT / relative
        assert source.is_file(), source
        assert sha(source) == SOURCE_HASHES[name], "Source changed: review annotations before rebuilding"
        license_page, doi = LICENSE_EVIDENCE[name]
        manifest_sources[name] = {"path": relative, "sha256": sha(source), "license": license_note,
                                  "license_pdf_page": license_page, "doi": doi,
                                  "license_verified_at": "2026-11-04"}
        for page in PAGES[name]:
            asset_id = f"{name}-p{page}"
            destination = OUT / "assets" / f"{asset_id}@{DPI}.png"
            rendered = render_page(source, page, destination, dpi=DPI)
            assets[asset_id] = {"id": asset_id, "path": str(destination.relative_to(OUT)),
                                "source_id": name, "pdf_page": page, "dpi": DPI,
                                "width": rendered.width, "height": rendered.height,
                                "sha256": sha(destination), "altered": False}
    mask_id = "mga-p10-masked"
    mask_path = OUT / "assets" / f"{mask_id}@{DPI}.png"
    transformation = mask_sheep_cell(ROOT / SOURCES["mga"][0], mask_path)
    assets[mask_id] = {**assets["mga-p10"], "id": mask_id,
                       "path": str(mask_path.relative_to(OUT)), "sha256": sha(mask_path), "altered": True}

    cases = []
    gold = {}

    def add(case_id, title, source, image_ids, ref, tags, questions, *, smoke=False, context="", forbidden=()):
        prompts, expected = zip(*questions)
        case_images = [assets[image_id] for image_id in image_ids]
        assert len(case_images) <= 2
        assert sum(a["width"] * a["height"] for a in case_images) <= 3_000_000
        cases.append({"id": case_id, "title": title, "track": "inspection", "source_id": source,
                      "ref_id": ref, "tags": tags, "smoke": smoke,
                      "images": case_images, "context": context, "questions": list(prompts)})
        for answer in expected:
            page_hint = re.match(r"Page (\d+):", answer["region"])
            required_images = [a for a in case_images if not page_hint or a["pdf_page"] == int(page_hint[1])]
            answer["allowed_evidence"] = [{"image_id": a["id"], "pdf_page": a["pdf_page"], "ref_id": ref} for a in required_images]
            answer["required_image_ids"] = [a["id"] for a in required_images]
        gold[case_id] = {"answers": list(expected), "forbidden_claims": list(forbidden),
                         "review": "active-assistant source-image review; independent human adjudication pending"}

    add("v01", "Runtime, elapsed-time formatting and printed memory units", "mga", ["mga-p17"], "table:3", ["numeric_table", "units"], [
        question("human_mga_elapsed", "Copy HUMAN/MGA elapsed time in h:mm:ss.", "31:37:28", "h:mm:ss", region="Table 3: Elapsed time, HUMAN/MGA"),
        question("human_hifiasm_elapsed", "Copy HUMAN/hifiasm elapsed time in h:mm:ss.", "20:40:26", "h:mm:ss", region="Table 3: Elapsed time, HUMAN/hifiasm"),
        question("human_mga_memory", "Copy HUMAN/MGA memory and preserve the printed unit label exactly.", 206528, "Mb", region="Table 3: Memory (Mb), HUMAN/MGA"),
        question("human_hifiasm_memory", "Copy HUMAN/hifiasm memory and preserve the printed unit label exactly.", 393431, "Mb", region="Table 3: Memory (Mb), HUMAN/hifiasm"),
        question("threads", "What thread count is stated in the table caption?", 50, "threads", region="Table 3 caption"),
    ], smoke=True, forbidden=["Treating elapsed time as seconds", "Silently rewriting the printed memory unit Mb as GB or MB"])

    add("v02", "Dataset-grouped rows, metric exceptions and missing cells", "mga", ["mga-p10"], "table:2", ["grouped_headers", "missing_values"], [
        question("human_mga_n50", "What is HUMAN/MGA N50?", 143.4, "Mb", region="Table 2: HUMAN/MGA, N50"),
        question("human_hifiasm_n50", "What is HUMAN/hifiasm N50?", 95.4, "Mb", region="Table 2: HUMAN/hifiasm, N50"),
        question("sheep_mga_n50", "What is SHEEP/MGA N50?", 105.2, "Mb", region="Table 2: SHEEP/MGA, N50"),
        question("sheep_hifiasm_n50", "What is SHEEP/hifiasm N50?", 112.5, "Mb", region="Table 2: SHEEP/hifiasm, N50"),
        question("sheep_best_n50", "Which assembler has the larger N50 for SHEEP?", "hifiasm", region="Table 2: SHEEP N50 comparison"),
        question("sheep_mga_phase_switch", "What SHEEP/MGA phase-switch rate is reported? A dash means not reported, not zero.", None, "%", status="not_shown", region="Table 2: SHEEP/MGA, Phase-switch rate"),
        question("sheep_hifiasm_phase_switch", "What SHEEP/hifiasm phase-switch rate is reported?", None, "%", status="not_shown", region="Table 2: SHEEP/hifiasm, Phase-switch rate"),
    ], forbidden=["MGA has the largest N50 for every dataset", "A dash is a measured zero phase-switch rate"])

    add("v08", "Separated caption/forest plate: sign, CI and outcome denominator", "clinical", ["clinical-p29", "clinical-p35"], "fig:2", ["separated_plate", "forest_plot", "sign", "denominators"], [
        question("m2_control", "Copy the month-2 negative-sputum-culture Control event count/denominator from the plate.", "3/11", region="Page 35: Negative sputum culture at M2 / Control"),
        question("m2_ibuprofen", "Copy the corresponding Ibuprofen event count/denominator.", "1/11", region="Page 35: same row / Ibuprofen"),
        question("m2_rd", "Copy the month-2 risk difference exactly as printed on the plate; do not reconcile it with other prose.", -0.18, "proportion", region="Page 35: same row / Risk Difference"),
        question("m2_ci_low", "Copy the printed lower 95% CI bound for that row.", -0.48, "proportion", region="Page 35: same row / 95% CI"),
        question("m2_ci_high", "Copy the printed upper 95% CI bound for that row.", 0.15, "proportion", region="Page 35: same row / 95% CI"),
        question("m2_ci_excludes_zero", "Does the displayed interval exclude zero?", False, region="Page 35: first forest interval and zero line"),
        question("outcome_n_equals_enrolled_n", "Does each month-2 outcome denominator equal the per-arm enrollment count in the caption?", False, region="Page 29 caption n=14 per arm versus page 35 outcome n=11"),
    ], smoke=True, forbidden=["Changing the displayed negative RD to a positive value from prose", "Inventing CI bounds from nearby narrative", "Using n=14 in place of the visible n=11 outcome denominators"])

    add("v09", "Separated box plot: median, caption statistics and whisker meaning", "clinical", ["clinical-p29", "clinical-p36"], "fig:3", ["separated_plate", "box_plot", "caption_binding"], [
        question("control_median", "Read the Control median sputum-conversion time from the plotted median and y-axis ticks.", 4, "months", region="Page 36: Ctrl median at tick 4"),
        question("ibuprofen_median", "Read the Ibuprofen median sputum-conversion time.", 4, "months", region="Page 36: Ibu median at tick 4"),
        question("p_value", "Copy the Figure 3 caption's p-value.", 0.92, region="Page 29: Figure 3 caption"),
        question("control_n", "Copy the Figure 3 caption's Control sample count.", 11, "participants", region="Page 29: Ctrl n=11"),
        question("ibuprofen_n", "Copy the Figure 3 caption's Ibuprofen sample count.", 11, "participants", region="Page 29: Ibu n=11"),
        question("whisker_type", "What do whiskers represent according to the caption? Use min_max, confidence_interval, standard_error, or unknown.", "min_max", region="Page 29: Figure 3 caption, minimum/maximum observed values"),
    ], forbidden=["Whiskers are 95% confidence intervals", "Claiming the ibuprofen median is lower"])

    add("v11", "Masked-cell abstention with a readable neighboring comparator", "mga", [mask_id], "table:2", ["abstention", "controlled_occlusion"], [
        question("sheep_mga_n50", "Read SHEEP/MGA N50 from its cell. It has been deliberately masked; do not infer it or use a remembered value.", None, "Mb", status="unreadable", region="Table 2: masked SHEEP/MGA N50 cell"),
        question("sheep_hifiasm_n50", "Read the unmasked SHEEP/hifiasm N50 comparator.", 112.5, "Mb", region="Table 2: SHEEP/hifiasm N50 cell"),
    ], context="This is a controlled occlusion variant of the source page. Masked data are not observable in this input.", forbidden=["Reconstructing a hidden value from memory", "Abstaining from the unmasked readable comparator"])

    add("v12", "Caption-only input cannot supply the absent forest values", "clinical", ["clinical-p29"], "fig:2", ["abstention", "caption_only", "wrong_page"], [
        question("m2_rd", "What month-2 negative-sputum-culture risk difference is visible in the supplied image?", None, "proportion", status="not_shown", region="Page 29 contains captions; the forest plate is absent"),
        question("m2_ci_low", "What lower CI bound is visible for that outcome?", None, "proportion", status="not_shown", region="Page 29 contains no forest values"),
        question("m2_ci_high", "What upper CI bound is visible for that outcome?", None, "proportion", status="not_shown", region="Page 29 contains no forest values"),
        question("total_enrolled", "What total enrollment is explicitly stated in the Figure 2 caption?", 28, "participants", region="Page 29: Figure 2 caption, total n=28"),
    ], forbidden=["Inventing values from a forest plot that was not supplied", "Claiming the caption page contains the actual forest plate"])

    add("v13", "Printed runtime bars and caption replicate count", "tangermeme", ["tangermeme-p2"], "fig:1", ["bar_plot", "units", "printed_values"], [
        question("tangermeme_time", "Copy the printed Tangermeme timing in panel c.", 1.61, "s", region="Figure 1c: Tangermeme bar label and Time (s) axis"),
        question("seqpro_time", "Copy the printed SeqPro timing in panel c.", 4.7, "s", region="Figure 1c: SeqPro bar label"),
        question("crested_time", "Copy the printed CREsted timing in panel c.", 6.26, "s", region="Figure 1c: CREsted bar label"),
        question("grelu_time", "Copy the printed gReLU timing in panel c.", 23.2, "s", region="Figure 1c: gReLU bar label"),
        question("selene_time", "Copy the printed Selene SDK timing in panel c.", 34.7, "s", region="Figure 1c: Selene SDK bar label"),
        question("chrombpnet_time", "Copy the printed ChromBPNet timing in panel c.", 48.1, "s", region="Figure 1c: ChromBPNet bar label"),
        question("fastest_tool", "Which tool has the lowest displayed panel-c time?", "Tangermeme", region="Figure 1c: six labeled bars"),
        question("timing_runs", "How many runs does the panel-c caption state?", 10, "runs", region="Figure 1 caption: timings averaged across ten runs"),
    ], smoke=True, forbidden=["Swapping toolkit timing labels", "Treating seconds as milliseconds", "Using panel-d twenty runs for panel-c timings"])

    add("v14", "Logarithmic axis, unprinted median and diagram legend", "tangermeme", ["tangermeme-p2"], "fig:1", ["log_axis", "abstention", "diagram_legend"], [
        question("divergence_axis_scale", "Is the panel-d Divergence axis linear or logarithmic?", "logarithmic", region="Figure 1d: powers-of-ten y-axis ticks", aliases=["log", "log10"]),
        question("captum_profile_exact_median", "What exact numeric median is printed for Captum's profile-head divergence? Do not infer it from position.", None, status="not_shown", region="Figure 1d: box has no numeric median label"),
        question("captum_profile_larger", "Is Captum's profile-head divergence visibly larger than Tangermeme's?", True, region="Figure 1d: two profile-head box plots"),
        question("divergence_runs", "How many runs does the panel-d caption state?", 20, "runs", region="Figure 1 caption: convergence deltas from twenty runs"),
        question("ablation_after_color", "Which color marks the After prediction in panel b's Ablation row?", "purple", region="Figure 1b: Ablation row Predictions legend", aliases=["violet"]),
    ], forbidden=["Digitizing an exact unlabeled divergence median", "Treating logarithmic tick spacing as linear", "Reusing the marginalization row's blue After color for ablation"])

    add("v15", "Heatmap panel binding, percentages and unavailable cells", "cellbench", ["cellbench-p5"], "fig:1", ["heatmap", "panel_binding", "missing_values"], [
        question("svm_rejection_zheng68k_f1", "Copy panel A's SVM rejection median F1-score for Zheng 68K.", 0.92, region="Figure 1A: first classifier row, Zheng 68K column"),
        question("svm_rejection_zheng68k_unlabeled", "Copy panel B's SVM rejection unlabeled percentage for Zheng 68K.", 61.8, "%", region="Figure 1B: first classifier row, Zheng 68K column"),
        question("scpred_zheng68k_f1", "Copy panel A's scPred median F1-score for Zheng 68K.", 1, region="Figure 1A: scPred row, Zheng 68K column"),
        question("scpred_zheng68k_unlabeled", "Copy panel B's scPred unlabeled percentage for Zheng 68K.", 41, "%", region="Figure 1B: scPred row, Zheng 68K column"),
        question("scina_baron_mouse_f1", "What median F1-score is printed for SCINA on Baron Mouse?", None, status="not_shown", region="Figure 1A: gray SCINA/Baron Mouse cell"),
        question("gray_cell_meaning", "According to the caption, do gray boxes mean measured zero or method not tested? Use zero or not_tested.", "not_tested", region="Figure 1 caption: gray boxes indicate method could not be tested"),
        question("higher_f1_is_full_labeling", "Does scPred's displayed F1-score of 1 on Zheng 68K mean it labels every cell?", False, region="Figure 1A scPred F1=1 and Figure 1B scPred unlabeled=41%"),
    ], smoke=True, forbidden=["Reading panel-B percentages as F1-scores", "Treating gray cells as zero", "Claiming F1=1 implies every cell is labeled"])

    add("v16", "Dataset order, grayscale encoding and unlabeled bar", "cellbench", ["cellbench-p7"], "fig:2", ["box_plot", "legend", "abstention"], [
        question("lowest_complexity_dataset", "Which dataset has the smallest complexity bar in panel B?", "Xin", region="Figure 2B: first and shortest bar"),
        question("highest_complexity_dataset", "Which dataset has the largest complexity bar in panel B?", "Zheng 68K", region="Figure 2B: final and tallest bar"),
        question("gray_encoding", "Does the gray fill encode median F1-score or number of cell populations? Use median_f1 or cell_populations.", "cell_populations", region="Figure 2: Cell populations colorbar and caption"),
        question("xin_exact_complexity", "What exact numeric complexity value is printed on the Xin bar? Do not estimate from its height.", None, status="not_shown", region="Figure 2B: Xin bar has no numeric data label"),
        question("panel_a_metric", "Copy panel A's y-axis label.", "Median F1-score", region="Figure 2A: y-axis label"),
    ], forbidden=["Inventing the Xin complexity value by interpolation", "Treating darker fill as higher F1-score"])

    add("v17", "ROC versus PR metrics and axis binding", "deepcrispr", ["deepcrispr-p9"], "fig:3", ["roc_pr", "metric_binding", "axis_binding"], [
        question("deepcrispr_roc_auc", "Copy DeepCRISPR's printed ROC-AUC in the upper-right ROC legend.", 0.981, region="Figure 3a: ROC curve legend, DeepCRISPR"),
        question("cfd_roc_auc", "Copy CFD's printed ROC-AUC in that legend.", 0.929, region="Figure 3a: ROC curve legend, CFD"),
        question("deepcrispr_pr_auc", "Copy DeepCRISPR's printed PR-AUC in the middle-right PR legend.", 0.497, region="Figure 3b: PR curve legend, DeepCRISPR"),
        question("cfd_pr_auc", "Copy CFD's printed PR-AUC in that legend.", 0.247, region="Figure 3b: PR curve legend, CFD"),
        question("roc_x_axis", "Copy the ROC plot's x-axis label.", "False Positive Rate", region="Figure 3a: ROC curve x axis"),
        question("roc_y_axis", "Copy the ROC plot's y-axis label.", "True Positive Rate", region="Figure 3a: ROC curve y axis"),
        question("pr_x_axis", "Copy the PR plot's x-axis label.", "Recall", region="Figure 3b: PR curve x axis"),
        question("pr_y_axis", "Copy the PR plot's y-axis label.", "Precision", region="Figure 3b: PR curve y axis"),
    ], smoke=True, forbidden=["Substituting ROC-AUC for PR-AUC", "Swapping precision and recall axes", "Using the on-target Figure 2 AUC in this off-target Figure 3 case"])

    add("v18", "Independent-dataset bars and training-curve axes", "deepcrispr", ["deepcrispr-p6"], "fig:2", ["bar_plot", "axis_binding", "abstention"], [
        question("independent_top_method", "Which method has the tallest bar in panel g?", "DeepCRISPR", region="Figure 2g: tallest blue bar"),
        question("independent_second_method", "Which method has the second-tallest bar in panel g?", "CRISPRater", region="Figure 2g: second-tallest dark red bar"),
        question("independent_exact_correlation", "What exact correlation value is printed on DeepCRISPR's panel-g bar? Do not interpolate its height.", None, status="not_shown", region="Figure 2g: DeepCRISPR bar has no numeric data label"),
        question("independent_metric", "Copy panel g's y-axis label.", "Spearman Correlation", region="Figure 2g: y-axis label"),
        question("training_x_axis", "Copy panel c's x-axis label.", "Epoch", region="Figure 2c: training-curve x axis"),
        question("training_y_axis", "Copy panel c's y-axis label.", "Loss", region="Figure 2c: training-curve y axis"),
    ], forbidden=["Guessing an exact Spearman correlation from an unlabeled bar", "Substituting a nearby ROC-AUC value for Spearman correlation", "Swapping loss and epoch axes"])

    add("v19", "Dot-plot rank and accuracy encodings", "scannotate", ["scannotate-p7"], "fig:3", ["dot_plot", "legend", "rank", "abstention"], [
        question("dot_color_encodes", "Does Figure 3's dot color encode ranking or accuracy?", "ranking", region="Figure 3: Ranking color legend and caption", aliases=["rank"]),
        question("dot_size_encodes", "Does Figure 3's dot size encode ranking or accuracy?", "accuracy", region="Figure 3: Accuracy size legend and caption"),
        question("red_rank", "Which rank does the bright-red legend entry denote?", 1, "rank", region="Figure 3: red Ranking legend entry"),
        question("light_green_rank", "Which rank does the lightest-green legend entry denote?", 10, "rank", region="Figure 3: lightest-green Ranking legend entry"),
        question("scannotate_pbmc10xv2_rank", "Read scAnnotate's rank in the PBMC.10Xv2 column from the color legend.", 1, "rank", region="Figure 3: bottom row, first dataset column"),
        question("scannotate_pbmc10xv3_rank", "Read scAnnotate's rank in the PBMC.10Xv3 column.", 2, "rank", region="Figure 3: bottom row, second dataset column"),
        question("scannotate_mtg_rank", "Read scAnnotate's rank in the MTG column.", 5, "rank", region="Figure 3: bottom row, MTG column, yellow dot"),
        question("scannotate_pbmc10xv2_exact_accuracy", "What exact accuracy number is printed on scAnnotate's PBMC.10Xv2 dot? Do not infer it from dot size.", None, status="not_shown", region="Figure 3: dot has no exact accuracy label"),
    ], smoke=True, forbidden=["Treating dot color as accuracy", "Treating rank 10 as best", "Inferring an exact accuracy from dot size", "Using Figure 4's cross-platform dot grid in this Figure 3 case"])

    add("v20", "Mosaic binary legend and selected-cell denominator", "scannotate", ["scannotate-p8"], "fig:6", ["mosaic", "binary_legend", "subset", "abstention"], [
        question("black_annotation", "What annotation boolean does the black legend square denote?", False, region="Figure 6: black square labeled False"),
        question("light_gray_annotation", "What annotation boolean does the light-gray legend square denote?", True, region="Figure 6: light-gray square labeled True"),
        question("displayed_methods", "How many annotation-method rows are shown in Figure 6?", 6, "methods", region="Figure 6: six labeled method rows"),
        question("column_represents", "What does each Figure 6 column represent?", "cell", region="Figure 6: Cell x-axis label and accompanying description", aliases=["a cell"]),
        question("all_pbmc_cells_displayed", "Does the caption say Figure 6 displays all PBMC.10Xv3 cells, rather than the inconsistently annotated subset?", False, region="Figure 6 caption: cells inconsistently annotated by the top six methods"),
        question("scannotate_exact_black_count", "What exact number of black cells is printed for the scAnnotate mosaic row? Do not count/interpolate pixel strips.", None, "cells", status="not_shown", region="Figure 6: scAnnotate row has no printed count"),
    ], forbidden=["Reversing the True/False legend", "Using the selected-cell total as the whole-dataset denominator", "Guessing a row-specific count from black pixel widths"])

    with pypdfium2.PdfDocument(str(ROOT / SOURCES["clinical"][0])) as doc:
        page = doc[6]
        tp = page.get_textpage()
        try:
            text = tp.get_text_range()
        finally:
            tp.close()
            page.close()
    start = text.index("The median time to sputum culture conversion")
    end = text.index("Radiological changes", start)
    exact_excerpt = text[start:end].strip()
    nominations = [
        {"id": "n01", "title": "Text already settles the requested medians", "track": "nomination", "source_id": "clinical",
         "source_text": exact_excerpt, "source_text_pdf_page": 7,
         "inventory": [{"ref_id": "fig:3", "caption_page": 29, "caption": "Figure 3 - Primary outcome, time to stable sputum conversion."}],
         "instruction": "The requested fact is the median conversion time in each arm. Nominate only if the supplied text does not settle that fact."},
        {"id": "n02", "title": "Caption-only table text leaves row/metric associations unresolved", "track": "nomination", "source_id": "mga",
         "source_text": "Table 2 Information about the BONOBO, HUMAN, GIRAFFE and SHEEP assemblies generated by MGA and hifiasm.",
         "source_text_pdf_page": 10, "transformation": "Caption-only source excerpt; table body intentionally withheld to simulate unavailable table text.",
         "inventory": [{"ref_id": "table:2", "caption_page": 10, "caption": "Information about the BONOBO, HUMAN, GIRAFFE and SHEEP assemblies generated by MGA and hifiasm."}],
         "instruction": "The requested facts are HUMAN and SHEEP N50 comparisons and missing phase-switch results. Nominate an inspection question if the supplied text cannot establish the row/column bindings."},
        {"id": "n03", "title": "Caption inventory does not locate the actual plates", "track": "nomination", "source_id": "clinical",
         "source_text": "Figure 2 - Primary and secondary outcomes risk difference. Figure 3 - Primary outcome, time to stable sputum conversion.",
         "source_text_pdf_page": 29, "transformation": "Caption-title-only excerpt; numeric legends and plates intentionally withheld.",
         "inventory": [{"ref_id": "fig:2", "caption_page": 29, "caption": "Primary and secondary outcomes risk difference.", "location_status": "caption_only"},
                       {"ref_id": "fig:3", "caption_page": 29, "caption": "Primary outcome, time to stable sputum conversion.", "location_status": "caption_only"}],
         "instruction": "The requested facts are the month-2 risk difference with CI and each arm's median conversion time. Nominate the needed figure questions. Do not invent artwork pages."},
    ]
    nomination_gold = {
        "n01": {"required_refs": [], "allowed_refs": [], "max_requests": 0, "reason": "The exact source excerpt states four months in both groups; inspection adds no information for this question."},
        "n02": {"required_refs": ["table:2"], "allowed_refs": ["table:2"], "max_requests": 1, "allowed_reasons": ["table_alignment", "missing_detail", "visual_evidence"], "reason": "Caption alone supplies neither dataset-specific values nor row bindings."},
        "n03": {"required_refs": ["fig:2", "fig:3"], "allowed_refs": ["fig:2", "fig:3"], "max_requests": 2, "allowed_reasons": ["missing_detail", "visual_evidence"], "requires_local_artwork_resolution": True, "reason": "Only caption titles are supplied; page 29 is not an artwork location."},
    }
    manifest = {"schema_version": 1, "created_at": str(date(2026, 11, 4)), "sources": manifest_sources,
                "assets": assets, "inspection_cases": len(cases), "nomination_cases": len(nominations),
                "smoke_case_ids": [c["id"] for c in cases if c["smoke"]],
                "dataset_license": "CC-BY-4.0; see LICENSES.md",
                "gold_status": "assistant-source-checked; independent human adjudication pending",
                "scope": "Diagnostic fixtures, not a corpus-representative estimate; no model performance has been measured."}
    write_json(OUT / "cases.json", cases)
    write_json(OUT / "nomination-cases.json", nominations)
    write_json(OUT / "manifest.json", manifest)
    write_json(OUT / "reference/expectations.json", gold)
    write_json(OUT / "reference/nomination-expectations.json", nomination_gold)
    write_json(OUT / "reference/transformations.json", {mask_id: transformation})
    print(f"Built {len(cases)} inspection cases, {len(nominations)} nomination cases, {len(assets)} images; no model calls.")


if __name__ == "__main__":
    main()
