"""Portable fixture integrity and scorer failure checks, without model calls."""

from argparse import Namespace
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


FIXTURES = Path(__file__).resolve().parents[1] / "benchmark-fixtures/visual-inspection"
SPEC = importlib.util.spec_from_file_location("visual_benchmark", FIXTURES / "benchmark.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def reference_response(case_id):
    """Synthetic scorer input; never report it as model performance."""
    reference = bench.read("reference/expectations.json")[case_id]
    return {"case_id": case_id, "answers": [
        {"question_id": a["question_id"], "value": a["value"], "unit": a["unit"],
         "status": a["status"], "evidence": [{**e, "region": a["region"]} for e in a["allowed_evidence"]]}
        for a in reference["answers"]]}


def score_response(response):
    cid = response["case_id"]
    case = next(c for c in bench.read("cases.json") if c["id"] == cid)
    return bench.score_inspection(response, case, bench.read("reference/expectations.json")[cid])


def test_portable_suite_hashes_and_licensed_source_scope():
    result = bench.validate()
    assert (result["inspection_cases"], result["nomination_cases"], result["answer_atoms"]) == (14, 3, 84)
    manifest = bench.read("manifest.json")
    assert set(manifest["sources"]) == {"mga", "clinical", "tangermeme", "cellbench", "deepcrispr", "scannotate"}
    assert manifest["dataset_license"].startswith("CC-BY-4.0")
    assert manifest["smoke_case_ids"] == ["v01", "v08", "v13", "v15", "v17", "v19"]
    smoke = [c for c in bench.read("cases.json") if c["smoke"]]
    assert {c["source_id"] for c in smoke} == set(manifest["sources"])
    for source in manifest["sources"].values():
        assert source["path"].startswith("benchmark-fixtures/pdfs/")
        assert source["license"].startswith("CC-BY-4.0")


def test_each_license_grant_is_bound_to_the_exact_article():
    import pypdfium2

    for source in bench.read("manifest.json")["sources"].values():
        with pypdfium2.PdfDocument(str(bench.REPO / source["path"])) as doc:
            page = doc[source["license_pdf_page"] - 1]
            textpage = page.get_textpage()
            try:
                text = re.sub(r"\s+", "", textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
        assert "creativecommons.org/licenses/by/4.0/" in text
        assert source["doi"] in text
        assert "CreativeCommonsAttribution" in text


@pytest.mark.parametrize("case_id, question_id, changes", [
    ("v08", "m2_rd", {"value": 0.18}),
    ("v08", "m2_rd", {"evidence": [
        {"image_id": "clinical-p29", "pdf_page": 29, "ref_id": "fig:2", "region": "caption"}]}),
    ("v01", "human_mga_memory", {"unit": "MB"}),
    ("v01", "human_mga_elapsed", {"value": "20:40:26"}),
    ("v02", "sheep_mga_phase_switch", {"value": 0}),
    ("v11", "sheep_mga_n50", {"value": 105.2}),
    ("v09", "whisker_type", {"value": "confidence_interval"}),
    ("v12", "m2_rd", {"value": -0.18}),
    ("v13", "tangermeme_time", {"unit": "ms"}),
    ("v14", "divergence_axis_scale", {"value": "linear"}),
    ("v15", "gray_cell_meaning", {"value": "zero"}),
    ("v15", "higher_f1_is_full_labeling", {"value": True}),
    ("v16", "xin_exact_complexity", {"value": 0.15}),
    ("v17", "deepcrispr_pr_auc", {"value": 0.981}),
    ("v18", "independent_exact_correlation", {"value": 0.45}),
    ("v19", "dot_color_encodes", {"value": "accuracy"}),
    ("v19", "scannotate_pbmc10xv2_exact_accuracy", {"value": 0.99}),
    ("v20", "black_annotation", {"value": True}),
])
def test_scorer_rejects_wrong_values_units_and_source_pages(case_id, question_id, changes):
    response = reference_response(case_id)
    assert score_response(response)["mechanical_pass"]
    answer = next(a for a in response["answers"] if a["question_id"] == question_id)
    answer.update(changes)
    result = score_response(response)
    assert not result["mechanical_pass"]
    assert not next(a for a in result["atoms"] if a["question_id"] == question_id)["mechanical_pass"]
    if question_id in {"sheep_mga_phase_switch", "sheep_mga_n50", "xin_exact_complexity",
                       "independent_exact_correlation", "scannotate_pbmc10xv2_exact_accuracy"} or case_id == "v12":
        assert any(a["unsupported_value_assertion"] for a in result["atoms"])


@pytest.mark.parametrize("bad_answers", [[], [{"question_id": []}], "not-a-list"])
def test_malformed_answers_fail_without_crashing(bad_answers):
    result = score_response({"case_id": "v01", "answers": bad_answers})
    assert not result["answer_shape_ok"] and not result["mechanical_pass"]


def test_nomination_requires_restraint_and_rejects_invented_artwork_pages():
    cases = {c["id"]: c for c in bench.read("nomination-cases.json")}
    gold = bench.read("reference/nomination-expectations.json")
    request = {"ref_id": "table:2", "priority": "critical", "reason": "missing_detail",
               "question": "What are the requested N50 values?"}
    response = {"case_id": "n02", "visual_requests": [request]}
    assert bench.score_nomination(response, cases["n02"], gold["n02"])["mechanical_pass"]
    assert not bench.score_nomination(response, cases["n01"], gold["n01"])["mechanical_pass"]
    assert bench.score_nomination({"visual_requests": []}, cases["n01"], gold["n01"])["mechanical_pass"]
    for changes in ({"pdf_page": 10}, {"ref_id": "table:999"}, {"ref_id": []}):
        broken = deepcopy(response)
        broken["visual_requests"][0].update(changes)
        assert not bench.score_nomination(broken, cases["n02"], gold["n02"])["mechanical_pass"]


@pytest.mark.parametrize("mode, track, count, images", [
    ("vision", "inspection", 6, True), ("text-only", "inspection", 6, False),
    ("vision", "nomination", 3, False),
])
def test_prepared_packets_separate_inputs_from_gold(tmp_path, mode, track, count, images):
    result = bench.prepare(Namespace(output=tmp_path, mode=mode, track=track, suite="smoke"))
    assert result["prepared_cases"] == count and result["model_calls_made"] == 0
    for line in (tmp_path / "requests.jsonl").read_text().splitlines():
        packet = json.loads(line)
        assert set(packet) == {"case_id", "messages", "limits"}
        content = packet["messages"][1]["content"]
        text = json.loads(content[0]["text"])
        allowed = ({"id", "source_text", "inventory", "instruction"} if track == "nomination"
                   else {"case_id", "ref_id", "context", "questions", "provided_images"})
        assert set(text) == allowed
        for question in text.get("questions", []):
            assert set(question) == {"id", "question"}
        assert any(c["type"] == "image_url" for c in content) == images


def test_missing_responses_and_unknown_telemetry_are_not_success(tmp_path):
    args = Namespace(responses=tmp_path / "responses.json", output=tmp_path / "score.json",
                     suite="full", track="inspection")
    bench.dump(args.responses, [reference_response(c["id"]) for c in bench.read("cases.json")])
    result = bench.score(args)
    assert result["correct_atoms"] == 84 and result["mechanically_passing_cases"] == 14
    assert result["latency"]["mean_ms"] is None
    assert result["recorded_usage"]["input_tokens"]["sum"] is None
    bench.dump(args.responses, [])
    result = bench.score(args)
    assert len(result["missing_responses"]) == 14 and result["correct_atoms"] == 0


def test_validate_and_prepare_without_package_or_private_corpus(tmp_path):
    """Isolated checkout containing only shipped benchmark files and source PDFs."""
    isolated = tmp_path / "checkout"
    suite = isolated / "benchmark-fixtures/visual-inspection"
    shutil.copytree(FIXTURES, suite, ignore=shutil.ignore_patterns("__pycache__"))
    for source in bench.read("manifest.json")["sources"].values():
        dest = isolated / source["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bench.REPO / source["path"], dest)
    commands = [["validate"], ["prepare", "--suite", "full", "--output", str(tmp_path / "packets")]]
    for command in commands:
        process = subprocess.run([sys.executable, "-I", str(suite / "benchmark.py"), *command],
                                 cwd=tmp_path, text=True, capture_output=True, check=True)
        result = json.loads(process.stdout)
        assert result.get("inspection_cases", result.get("prepared_cases")) == 14
