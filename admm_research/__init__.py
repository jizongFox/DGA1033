name = "admm_research"

from absl import flags, app
import logging, sys

LOGGER = logging.getLogger(__name__)
LOGGER.parent = None
def config_logger():
    """ Get console handler """
    log_format = logging.Formatter("[%(module)s - %(asctime)s - %(levelname)s] %(message)s")
    LOGGER.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_format)
    LOGGER.handlers = [console_handler]

config_logger()