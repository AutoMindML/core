import logging
import warnings

from automind.utils.config import logger_name

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s,%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    format="[%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(logger_name)
