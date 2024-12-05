import logging
import os

DEBUG = os.getenv("DEBUG", False)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.ERROR)

e2b_logger = logging.getLogger("e2b")
e2b_logger.setLevel(logging.ERROR)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.ERROR)

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
openai_loggers = [logger.setLevel(logging.ERROR) for logger in loggers]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
