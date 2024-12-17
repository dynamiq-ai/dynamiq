from dynamiq.connections import Exa, Tavily, ScaleSerp, ZenRows, Firecrawl
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """
You are an expert business intelligence and competitive analysis agent designed to:

1. Conduct Comprehensive Company Research
- Gather detailed information about target companies
- Analyze company profiles, financial performance, market positioning, and strategic insights
- Provide up-to-date and accurate company background information

2. Competitive Intelligence Analysis
- Identify and profile direct and indirect competitors
- Compare company strengths, weaknesses, market share, and strategic approaches
- Develop comparative frameworks to understand competitive landscapes

3. Research Capabilities
- Utilize multiple research methodologies
- Cross-reference information from credible business databases and public sources
- Synthesize complex information into clear, actionable insights

4. Analytical Outputs
- Generate structured reports
- Create visual comparisons and competitive matrices
- Offer strategic recommendations based on research findings

5. Ethical and Professional Standards
- Rely only on publicly available and verifiable information
- Maintain objectivity and avoid speculation
- Protect confidentiality and respect intellectual property rights

Approach each research task methodically, providing comprehensive yet concise analysis that supports strategic business decision-making.
"""

INPUT_TASK = """Conduct a comprehensive competitive analysis for Dynamiq AI, including:

    1. Company Profile
    - Full company background and history
    - Core business focus and technological offerings
    - Mission statement and strategic positioning
    - Funding history and key investors
    - Leadership team overview

    2. Technological Analysis
    - Primary AI technologies and solutions
    - Unique technological differentiators
    - Target industries and use cases
    - Technology maturity and innovation level

    3. Competitive Landscape
    - Identify direct and indirect competitors
    - Comparative analysis of:
      * Technological capabilities
      * Market share
      * Funding status
      * Key client base
      * Pricing models
      * Geographic reach

    4. Market Positioning
    - SWOT analysis
    - Competitive advantages
    - Potential market opportunities
    - Challenges and potential limitations

    5. Competitor Detailed Profiles
    - Provide in-depth profiles for top 3-5 competitors
    - Include:
      * Company overview
      * Technological similarities/differences
      * Recent notable achievements
      * Funding and financial status

    6. Strategic Insights
    - Potential partnership opportunities
    - Areas of potential technological differentiation
    - Market expansion strategies
    - Emerging competitive threats

    7. Additional Context
    - Recent industry trends affecting AI companies
    - Potential regulatory considerations
    - Technological innovation landscape

    Deliver a comprehensive report with clear, actionable insights and a strategic overview of Dynamiq AI's competitive environment."""


if __name__ == "__main__":
    connection_zenrows = ZenRows()
    connection_tavily = Tavily()
    connection_serp = ScaleSerp()
    tool_serp = ScaleSerpTool(connection=connection_serp,
                              description="A tool for searching the company details, "
                                          "all information about the company, "
                              )
    tool_tavily = TavilyTool(connection=connection_tavily,
                             description="A tool for searching the competitors of the company, "
                                            "all information about the competitors, "
                             )
    tool_zenrows = ZenRowsTool(connection=connection_zenrows,
                              )
    connection_firecrawl = Firecrawl()
    tool_firecrawl = FirecrawlTool(connection=connection_firecrawl,
                                   )
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0)
    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_serp, tool_tavily, tool_firecrawl],
        inference_mode=InferenceMode.XML,
        role=AGENT_ROLE,
        behavior=Behavior.RETURN,
        max_loops=25,

    )
    result = agent.run(input_data={
        "input": INPUT_TASK,
    })
    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
