from dynamiq.connections import Tavily, Firecrawl
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.tools.firecrawl import FirecrawlTool

from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm
from dynamiq.nodes.agents.utils import XMLParser
from dynamiq.memory.inner_memory import InnerMemoryConfig, InnerMemory

AGENT_ROLE = (
    "You are helpfull assistant that scrapes all information."
)

REQUEST_AWS_PARTNERS = """
Parse 5 pages of https://clutch.co/developers/artificial-intelligence/generative?page=1 and generate csv like file with information for this 5 pages."""

if __name__ == "__main__":
#     content = """ 
# <root>
#   <section1>
#     The section shows a cookie preferences and settings interface for the AWS Partners website, followed by extensive filtering options for AWS partners including:
#     <filters>
#       <filter>Location filters</filter>
#       <filter>Partner type filters</filter>
#       <filter>Industry vertical filters</filter>
#       <filter>AWS services filters</filter>
#       <filter>Partner program filters</filter>
#       <filter>Contract vehicle filters</filter>
#     </filters>
#   </section1>

#   <section2>
#     The section displays search results showing AWS Premier Partners with detailed information for each partner.
#     <partners>
#       <partner>SoftwareOne</partner>
#       <partner>Zero&amp;One</partner>
#       <partner>Cloud303</partner>
#       <partner>ECLOUDVALLEY</partner>
#       <partner>Eviden</partner>
#       <partner>Reply</partner>
#       <partner>Classmethod</partner>
#       <partner>tecRacer</partner>
#       <partner>AllCloud</partner>
#       <partner>MegazoneCloud</partner>
#     </partners>
#   </section2>

#   <memory>
#     <item>
#       <key>aws_partner_types</key>
#       <description>Types of AWS partners available in the directory</description>
#       <data>Software Product, Hardware Product, Communications Product, Consulting Service, Managed Service, Professional Service, Value-Added Resale AWS Service, Training Service, Distribution Service</data>
#     </item>
#     <item>
#       <key>premier_partners_2024</key>
#       <description>List of notable AWS Premier Partners identified</description>
#       <data>SoftwareOne, Zero&amp;One, Cloud303, ECLOUDVALLEY, Eviden, Reply, Classmethod, tecRacer, AllCloud, MegazoneCloud</data>
#     </item>
#     <item>
#       <key>partner_recognition</key>
#       <description>Key recognition indicators for AWS partners</description>
#       <data>AWS Competencies, Partner Programs, AWS Service Validations, AWS Certifications, AWS Customer Launches</data>
#     </item>
#   </memory>
# </root>


#         """
#     parsed_data = XMLParser.parse(content, required_tags=["section1", "section2"], optional_tags=["memory"])
    
    connection_tavily = Tavily()
    connection_firecrawl = Firecrawl()

    tool_search = TavilyTool(connection=connection_tavily)
    tool_scrape = FirecrawlTool(connection=connection_firecrawl)
    llm = setup_llm(model_provider="claude", model_name="claude-3-7-sonnet-20250219", temperature=0)

    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search, tool_scrape],
        role=AGENT_ROLE,
        max_loops = 50,
        inference_mode=InferenceMode.XML,
        inner_memory_config = InnerMemoryConfig(enabled=True, inner_memory=InnerMemory(), max_context_length=20000)
    )


    result = agent.run(input_data={"input": REQUEST_AWS_PARTNERS,
                                    "files": None})

    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
