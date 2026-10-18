"""Report existing checks without requesting additional verification work."""


def entailment_check(scores: dict) -> dict:
    requested = "support_check_complete" in scores or bool(scores.get("support_check_failed"))
    completed = requested and bool(scores.get("support_check_complete"))
    if not requested:
        status = "not_requested"
    elif scores.get("support_check_failed") or scores.get("n_unsupported", 0):
        status = "failed"
    elif not completed:
        status = "incomplete"
    else:
        status = "passed"
    return {"requested": requested, "completed": completed, "status": status}
