import logging
from enum import Enum
from pathlib import Path

_names = set()

_DIR = None
_EXT = '.log'


class Level(Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


def initialise(path: str, level: Level = Level.WARNING):
    global _DIR
    _DIR = Path(path)
    _DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=level.value)


def _get_log_filename(name: str) -> Path:
    """Find the next valid numbering for the log file corresponding to the
     given name.

     Parameters
     ----------
     name:
         The logger name.

    Returns
    -------
    A valid, unused log filename.
    """
    assert isinstance(_DIR, Path), \
        ("The log directory has not been initialised: call `log.initialise(<path>)` "
         "with the desired directory path.")
    count = 0
    while True:
        path = _DIR / f'{name}.{count}{_EXT}'
        if not path.exists():
            return path

        count += 1


def _custom_logger(name: str, level: int) -> logging.Logger:
    """Create a logger with the given name and level, and with preset
    customisation options.

    Parameters
    ----------
    name:
        The logger name.
    level:
        The log level.

    Returns
    -------
    A logger.
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    _names.add(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(module)s - %(levelname)s - %(message)s'
    )

    # put all outputs into a log file iff the log directory was initialised
    if isinstance(_DIR, Path):
        fh = logging.FileHandler(_get_log_filename(name))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # with a file handler, the console handler is restricted to errors only
        ch_level = Level.ERROR.value
    else:
        # without a file handler, the console handler inherits the desired log level
        ch_level = level

    # show error and critical messages only on the console
    ch = logging.StreamHandler()
    ch.setLevel(ch_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def get_logger(name: str, level: Level = Level.WARNING) -> logging.Logger:
    """Return the unique logger under the given name and, if not yet existing,
    create one with a specified level.

    Parameters
    ----------
    name:
        The logger name.
    level:
        The level to assign to the newly created logger.

    Returns
    -------
    The sought logger.
    """
    if name in _names:
        return logging.getLogger(name)
    else:
        return _custom_logger(name, level.value)
