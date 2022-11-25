import logging
import coloredlogs


logger = logging.getLogger(__name__)

coloredlogs.install(
    fmt=f'%(asctime)s - MDDatasetBuilder {__version__} - %(levelname)s: %(message)s',
    level=logging.INFO,
    milliseconds=True,
    logger=logger,
    )
