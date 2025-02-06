from datetime import date, datetime, timezone


def get_search_queries_prompt() -> str:
    """Generates the prompt for refining search queries."""
    current_date = datetime.now(timezone.utc).strftime("%B %d, %Y")

    return f"""
Write {{{{max_iterations}}}} Google search queries to form an objective opinion for the task:
{{{{task}}}}".\n\n
Assume today's date is {current_date} if required.\n\n
You are a seasoned research assistant tasked with generating search queries to find relevant information
for the following task: "{{{{task}}}}".\n\n
Context: {{{{context}}}}\n\n
Use this context to inform and refine your search queries. The context provides real-time web information
that can help you generate more specific and relevant queries. Consider any current events, recent
developments, or specific details mentioned in the context that could enhance the search queries.\n\n
You must respond with a list of strings in the following format: [{{{{format}}}}].\n
The response should contain ONLY the list.
"""


def get_curate_sources_prompt() -> str:
    """Generates the prompt for evaluating and curating sources."""
    return """Your goal is to evaluate and curate the provided scraped content for the research task: {{query}}
while prioritizing the inclusion of relevant and high-quality information, especially sources containing statistics,
numbers, or concrete data.

The final curated list will be used as context for creating a research report, so prioritize:
- Retaining as much original information as possible, with extra emphasis on sources featuring
quantitative data or unique insights
- Including a wide range of perspectives and insights
- Filtering out only clearly irrelevant or unusable content

EVALUATION GUIDELINES:
1. Assess each source based on:
   - Relevance: Include sources directly or partially connected to the research query. Err on the side of inclusion.
   - Credibility: Favor authoritative sources but retain others unless clearly untrustworthy.
   - Currency: Prefer recent information unless older data is essential or valuable.
   - Objectivity: Retain sources with bias if they provide a unique or complementary perspective.
   - Quantitative Value: Give higher priority to sources with statistics, numbers, or other concrete data.
2. Source Selection:
   - Include as many relevant sources as possible, up to {{max_results}}, focusing on broad coverage and diversity.
   - Prioritize sources with statistics, numerical data, or verifiable facts.
   - Overlapping content is acceptable if it adds depth, especially when data is involved.
   - Exclude sources only if they are entirely irrelevant, severely outdated, or unusable due to poor content quality.
3. Content Retention:
   - DO NOT rewrite, summarize, or condense any source content.
   - Retain all usable information, cleaning up only clear garbage or formatting issues.
   - Keep marginally relevant or incomplete sources if they contain valuable data or insights.

SOURCES LIST TO EVALUATE:
{{sources}}

You MUST return your response in the EXACT sources JSON list format as the original sources.
The response MUST not contain any markdown format or additional text (like ```json), just the JSON list!
"""


def get_research_report_prompt(word_lower_limit=1000, report_format="apa", language="english") -> str:
    """Generates a detailed research report prompt with formatting and citation requirements."""
    return f"""
Information: "{{{{context}}}}"
---
Using the above information, answer the following query or task: "{{{{question}}}}" in a detailed report --
The report should focus on the answer to the query, should be well structured, informative,
in-depth, and comprehensive, with facts and numbers if available and at least {word_lower_limit} words.
You should strive to write the report as long as you can using all relevant and necessary information provided.

Please follow all of the following guidelines in your report:
- You MUST determine your own concrete and valid opinion based on the given information.
Do NOT defer to general and meaningless conclusions.
- You MUST write the report with markdown syntax and {report_format} format.
- You MUST prioritize the relevance, reliability, and significance of the sources you use.
Choose trusted sources over less reliable ones.
- You must also prioritize new articles over older articles if the source can be trusted.
- Use in-text citation references in {report_format} format and make it with markdown hyperlink placed at the end of
the sentence or paragraph that references them like this: ([in-text citation](url)).
- Don't forget to add a reference list at the end of the report in {report_format} format and
full url links without hyperlinks.
- You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated
sources, but only one reference for each.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report:
eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [url website](url)

You MUST write the report in the following language: {language}.
Please do your best, this is very important to my career.
Assume that the current date is {date.today()}.
"""
