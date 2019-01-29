# Copyright 2018 East China Normal University
"""MDDatasetBuilder."""

from .datasetbuilder import DatasetBuilder, __version__
import logging
import coloredlogs

__all__ = ['DatasetBuilder']

coloredlogs.install(
    fmt=f'%(asctime)s - MDDatasetBuilder {__version__} - %(levelname)s: %(message)s',
    level=logging.INFO, milliseconds=True)
