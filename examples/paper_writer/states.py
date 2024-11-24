import json
import re
from typing import Any

from prompts import (
    ABSTRACT_WRITER_PROMPT,
    INTERNET_SEARCH_PROMPT,
    PAPER_WRITER_PROMPT,
    REFERENCES_PROMPT,
    REFLECTION_REVIEWER_PROMPT,
    TASK_TEMPLATE,
    TITLE_PROMPT,
    TOPIC_SENTENCE_PROMPT,
    TOPIC_SENTENCE_REVIEW_PROMPT,
    WRITER_REVIEW_PROMPT,
)

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Tavily
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.prompts import Message, Prompt

llm = OpenAI(
    name="OpenAI LLM",
    connection=OpenAIConnection(),
    model="gpt-4o-mini",
    max_tokens=5000,
)
tool_scrapper = TavilyTool(connection=Tavily())


def inference_model(messages):
    """
    Inference model
    """
    llm_result = llm.run(
        input_data={},
        prompt=Prompt(
            messages=messages,
        ),
    ).output["content"]

    return llm_result


def suggest_title_review_state(context: dict[str, Any]):
    """State that suggests user to review proposed titles"""

    messages = context.get("messages", [])

    instruction = context.get("update_instruction")
    if instruction:
        message = Message(
            role="user",
            content=(
                "Here is instruction user provided to system to refine paper."
                f"Instruction: {instruction}"
                "Just return title without any additional information"
            ),
        )
    else:
        message = Message(role="user", content="Just return the first title without any additional information")

    messages.append(message)
    result = inference_model(messages)
    messages = []

    return {"result": result, "title": result, "messages": messages, "state": "suggest_title_review_state"}


def suggest_title_state(context: dict[str, Any]):
    """State that suggests title for a paper."""

    messages_title = [
        Message(
            role="system",
            content=TITLE_PROMPT.format(
                area_of_paper=context["area_of_paper"], title=context["title"], hypothesis=context["hypothesis"]
            ),
        ),
        Message(
            role="user",
            content="Write the original title first. Then,"
            "generate 10 thought provoking titles that "
            "instigates reader's curiosity based on the given information",
        ),
    ]

    llm_result = inference_model(messages_title)

    messages_title.append(Message(role="system", content="llm_result"))
    return {"result": llm_result, "title": llm_result, "messages": messages_title, "state": "suggest_title_state"}


def internet_search_state(context: dict[str, Any]):
    messsages = context.get("messages", [])
    content = context.get("content")
    task = TASK_TEMPLATE.format(
        title=context.get("title", "(No title)"),
        type_of_document=context.get("type_of_document", "(No type_of_document)"),
        area_of_paper=context.get("area_of_paper", "(No area_of_paper)"),
        sections=context.get("sections", "(No sections)"),
        instruction=context.get("update_instruction", "(No instruction)"),
        hypothesis=context.get("hypothesis", "(No hypothesis)"),
        results=context.get("results", "(No results)"),
        references="\n".join(context.get("references", ["(No references)"])),
    )

    if not context.get("update_instruction"):

        messsages += [
            Message(
                role="system",
                content=INTERNET_SEARCH_PROMPT.format(number_of_queries=context["number_of_queries"])
                + " You must only output the response in a plain list of queries "
                "in the JSON format '{ \"queries\": list[str] }' and no other text. "
                "You MUST only cite references that are in the references "
                "section. ",
            ),
            Message(
                role="user",
                content=task,
            ),
        ]

        a = inference_model(messsages)
        generated_queries = json.loads(a)["queries"]
        for ref in context.get("references", []):
            search_match = re.search(r"http.*(\s|$)", ref)
            if search_match:
                l, r = search_match.span()
                http_ref = ref[l:r]
                generated_queries.append(http_ref)

        content = context.get("content", [])
        for query in generated_queries:
            search_result = tool_scrapper.run(input_data={"query": query}).output["content"]["raw_response"]["results"]

            for result in search_result:
                text = f"link: {result['url']}, " f"content: {result['content']}"

                content.append(text)

    return {"state": "internet_search_state", "result": "Gathered information.", "content": content, "task": task}


def write_topic(context: dict[str, Any]):
    messages = context.get("messages", [])
    content = "\n".join(context.get("content"))
    task = context.get("task")
    plan = context.get("plan")

    if not context.get("update_instruction"):
        messages = [
            Message(role="system", content=TOPIC_SENTENCE_PROMPT),
            Message(
                role="user",
                content=(
                    f"This is the content of a search on the internet for the paper:\n\n" f"{content}\n\n" f"{task}"
                ),
            ),
        ]

        plan = inference_model(messages)

        messages.append(Message(role="system", content=plan))

    return {
        "result": f"Topic plan proposed {plan}",
        "plan": plan,
        "messages": messages,
        "state": "write_topic",
    }


def review_topic_sentence(context: dict[str, Any]):

    review_topic_sentences = context.get("review_topic_sentences", [])
    messages = []
    plan = context.get("plan")

    task = context.get("task")
    result = context.get("draft")

    if instruction := context.get("update_instruction"):
        review_topic_sentences.append(instruction)
        messages.append(
            Message(
                role="system",
                content=(
                    TOPIC_SENTENCE_REVIEW_PROMPT + "\n\n"
                    f"Here is my task:\n\n{task}\n\n"
                    f"Here is previous plan:\n\n{plan}\n\n"
                    f"Here is my instruction:\n\n{instruction}\n\n"
                    "Only return the Markdown for the new plan as output."
                ),
            )
        )

        plan = inference_model(messages)

        messages.append(Message(role="system", content=result))

    return {
        "result": f"Topic after review {result}",
        "plan": plan,
        "messages": messages,
        "review_topic_sentences": review_topic_sentences,
        "state": "review_topic_sentence",
    }


def write_paper(context: dict[str, Any]):
    content = "\n\n".join(context.get("content", []))
    critique = context.get("critique", "")
    review_instructions = context.get("review_instructions", [])
    task = context.get("task")
    sentences_per_paragraph = context.get("sentences_per_paragraph")
    state = context.get("state")
    draft = context.get("draft")
    plan = context.get("plan")
    if not context.get("update_instruction"):

        if state == "internet_search":
            additional_info = " in terms of topic sentences"
        else:
            additional_info = ""

        messages = [
            Message(
                role="system",
                content=PAPER_WRITER_PROMPT.format(
                    task=task,
                    content=content,
                    review_instructions=review_instructions,
                    critique=critique,
                    sentences_per_paragraph=sentences_per_paragraph,
                ),
            ),
            Message(
                role="user",
                content=(
                    "Generate a new draft of the document based on the "
                    "information I gave you.\n\n"
                    f"Keep title {context.get("title")}"
                    f"Here is my current draft{additional_info}:\n\n"
                    f"{draft}\n\n"
                    f"Here is my plan, stick to it (use according headings):\n\n"
                    f"{plan}\n\n"
                    f"Keep response in markdown."
                ),
            ),
        ]

        draft = inference_model(messages)
    return {
        "result": "Paper rewritten successfully",
        "state": "write_paper",
        "draft": draft,
        "revision_number": context.get("revision_number", 1) + 1,
    }


def write_paper_review(context: dict[str, Any]):
    review_instructions = context.get("review_instructions", [])
    instruction = context.get("update_instruction")
    draft = result = context.get("draft")
    result = context.get("draft")
    task = context.get("task")
    if instruction:
        joined_instructions = "\n".join(review_instructions)
        messages = [
            Message(role="system", content=WRITER_REVIEW_PROMPT),
            Message(
                role="user",
                content=(
                    "Here is my task:\n\n"
                    f"{task}"
                    "\n\n"
                    "Here is my draft:\n\n"
                    f"{draft}"
                    "\n\n"
                    "Here is my instruction:\n\n"
                    f"{instruction}"
                    "\n\n"
                    "Here is my previous instructions that you must "
                    "observe:\n\n"
                    f"{joined_instructions}"
                    "\n\n"
                    "Only change in the draft what the user has requested by "
                    "the instruction.\n"
                    "Return result in markdown format (making appropriate headings)."
                ),
            ),
        ]

        if instruction not in review_instructions:
            review_instructions.append(instruction)

        result = inference_model(messages)

    return {
        "result": "Paper review finished",
        "state": "paper review",
        "review_instructions": review_instructions,
        "draft": result,
    }


def reflection_review(context: dict[str, Any]):
    review_instructions = context.get("review_instructions")
    messages = [
        Message(
            role="system",
            content=REFLECTION_REVIEWER_PROMPT.format(
                hypothesis=context.get("hypothesis"), review_instructions=review_instructions
            ),
        ),
        Message(role="user", content=context.get("draft")),
    ]
    result = inference_model(messages)
    return {"result": f"Reflection finished with critique {result}", "state": "reflection_review", "critique": result}


def additional_reflection_review(context: dict[str, Any]):
    additional_critique = context.get("update_instruction")
    critique = context.get("critique")
    if additional_critique:
        critique = critique + "\n\nAdditional User's feedback:\n" f"{additional_critique}\n"
    return {"state": "additional_reflection_review", "critique": critique}


def write_abstract(context: dict[str, Any]):
    task = context.get("task")
    draft = context.get("draft")

    messages = [
        Message(role="system", content=ABSTRACT_WRITER_PROMPT),
        Message(
            role="user",
            content=(
                f"Here is instruction from the advisor: {context.get("update_instruction")
                                                         if context.get("update_instruction") else "No instruction"}"
                f"Here is my task:\n\n{task}\n\n"
                f"Here is my current draft:\n\n{draft}\n\n"
            ),
        ),
    ]

    result = inference_model(messages)

    return {"result": f"Generated abstract of the paper {result}", "state": "write_abstract", "draft": result}


def generate_references(context: dict[str, Any]):
    content = "\n".join(context.get("content"))
    messages = [
        Message(role="system", content=REFERENCES_PROMPT),
        Message(
            role="user",
            content=(
                "Generate references for the following content entries. "
                "\n\n"
                "Content:"
                "\n\n"
                f"{
                     content}"
            ),
        ),
    ]

    result = inference_model(messages)

    return {
        "result": f"Generated references for the paper {result}",
        "state": generate_references,
        "references": result,
    }


def generate_citation(context: dict[str, Any]):
    references = context.get("references")
    draft = context.get("draft")
    draft = draft + "\n\n" + references
    return {
        "result": "Generated citation",
        "state": "generate_citation",
        "draft": draft,
    }


def generate_caption(context: dict[str, Any]):
    draft = context.get("draft")
    pattern = r"!\[([^\]]*)\]\(([^\)]*)\)"

    result = list(reversed(list(re.finditer(pattern, draft))))
    fig = len(result)

    for entry in result:
        left, right = entry.span()
        caption = f'![]({entry[2]})\n\n<div align="center">Figure {fig}:' f"{entry[1]}</div>\n"
        draft = draft[:left] + caption + draft[right:]
        fig -= 1

    return {
        "result": "Generated captions",
        "state": "generate_caption",
        "draft": draft,
    }
