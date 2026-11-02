from unittest.mock import patch

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.converters import LLMPDFConverter
from dynamiq.nodes.llms import OpenAI
from dynamiq.runnables import RunnableConfig


def test_llm_pdf_converter_separates_pages_in_one_document():
    converter = LLMPDFConverter(llm=OpenAI(connection=OpenAIConnection(api_key="test"), model="gpt-4o-mini"))
    pages = [{"content": "Filter QF-03"}, {"content": "Page two"}]

    with (
        patch.object(LLMPDFConverter, "_convert_images_to_urls", return_value=["page-1", "page-2"]),
        patch.object(LLMPDFConverter, "perform_llm_extraction", return_value=pages),
    ):
        documents = converter._process_images(images=[None, None], metadata={}, config=RunnableConfig())

    assert documents[0].content == "Filter QF-03\nPage two"
