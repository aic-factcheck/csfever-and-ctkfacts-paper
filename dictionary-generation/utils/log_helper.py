
import logging

# taken from fever-baselines

class LogHelper():
    handler = None
    @staticmethod
    def setup():
        LOG_FORMAT = '[{asctime}][{levelname}][{name}:{lineno}] {message}'
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT, style='{')
        LogHelper.handler = logging.StreamHandler()
        LogHelper.handler.setLevel(logging.DEBUG)
        LogHelper.handler.setFormatter(formatter)

        LogHelper.get_logger(LogHelper.__name__).info("Log Helper set up")

    @staticmethod
    def get_logger(name, level=logging.DEBUG):
        l = logging.getLogger(name)
        l.setLevel(level)
        l.addHandler(LogHelper.handler)
        return l
