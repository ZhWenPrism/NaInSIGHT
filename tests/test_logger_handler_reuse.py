import logging
from unittest.mock import patch
from nainsight.utils import get_logger


def test_repeated_setup_does_not_open_an_unused_file_handler(tmp_path):
    name = "nainsight-test-handler-reuse"
    logger = logging.getLogger(name)
    library = logging.getLogger("nainsight")
    old_library = list(library.handlers)
    try:
        with patch("nainsight.utils.logging.FileHandler", wraps=logging.FileHandler) as factory:
            first = get_logger(name, tmp_path / "stage.log")
            second = get_logger(name, tmp_path / "stage.log")
            assert first is second
            assert factory.call_count == 1
        logger.info("recorded once")
        assert (tmp_path / "stage.log").read_text().count("recorded once") == 1
    finally:
        handlers = set(logger.handlers) | (set(library.handlers) - set(old_library))
        for handler in handlers:
            logger.removeHandler(handler)
            library.removeHandler(handler)
            handler.close()
