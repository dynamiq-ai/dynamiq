def review_plan(context: dict, **kwargs) -> dict:
    """Gathers human feedback on the research plan if required."""
    include_human_feedback = context.get("task").get("include_human_feedback")

    if not include_human_feedback:
        return {"human_feedback": None, "result": "ok"}

    layout = context.get("sections")
    human_feedback = input(
        f"Any feedback on this plan of topics to research? {layout}? If not, please reply with 'no'."
    )

    return {"human_feedback": human_feedback, "result": "ok"}
