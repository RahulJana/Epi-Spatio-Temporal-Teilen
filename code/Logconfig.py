# -*- coding: utf-8 -*-

import os
import logging
from Constant import Paths, NAME

log_file = os.path.join(Paths.DEST_DIR, NAME.LOG_NAME)


class LoggerManager:
    _logger = None

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._logger = cls._setup_logger()
        return cls._logger

    @classmethod
    def _setup_logger(cls):

        logger = logging.getLogger("SSIRSTGNN")
        logger.setLevel(logging.INFO)

        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, mode="w")
        stream_handler = logging.StreamHandler()

        # formatter = logging.Formatter("%(asctime)s - %(processName)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        # logger.addHandler(stream_handler)

        return logger
