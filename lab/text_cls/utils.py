import json
import os
import logging
import logging.handlers


def reader(input_file, encoding='utf8', fmt='raw'):
    if fmt not in ["raw", "jsonl"]:
        raise ValueError("unsupported format")
    with open(input_file, encoding=encoding) as f:
        if fmt == 'jsonl' or input_file.endswith(".jsonl"):
            for idx, line in enumerate(f):
                d = json.loads(line.strip())
                yield idx, d
        else:
            for idx, line in enumerate(f):
                yield idx, line.strip()


def init_logger(log_dir, log_name, level=logging.DEBUG, when="D", backup=7,
                format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
                datefmt="%m-%d %H:%M:%S"):
    """
    init_log - initialize log module
    Args:
        log_path - Log file path prefix.
        Log data will go to two files: log_path.log and log_path.log.wf
        Any non-exist parent directories will be created automatically
        level - msg above the level will be displayed
        DEBUG < INFO < WARNING < ERROR < CRITICAL
        the default value is logging.INFO
        when - how to split the log file by time interval
            'S' : Seconds
            'M' : Minutes
            'H' : Hours
            'D' : Days
            'W' : Week day
            default value: 'D'
        format - format of the log
        default format:
        %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
        INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
        backup - how many backup file to keep
        default value: 7

        Raises:
        OSError: fail to create log directories
        IOError: fail to open log file
    """
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger(name=log_name)
    logger.setLevel(level)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    handler = logging.handlers.TimedRotatingFileHandler(os.path.join(log_dir, log_name + ".log"),
                                                        when=when,
                                                        backupCount=backup,
                                                        encoding='utf8')
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.handlers.TimedRotatingFileHandler(os.path.join(log_dir, log_name + ".log.wf"),
                                                        when=when,
                                                        backupCount=backup,
                                                        encoding='utf8')
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger