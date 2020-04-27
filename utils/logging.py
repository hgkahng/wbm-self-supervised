# -*- coding: utf-8 -*-

import os
import sys
import logging
from colorama import Fore


def get_tqdm_config(total, leave=True, color='white'):
    fore_colors = {
        'red': Fore.LIGHTRED_EX,
        'green': Fore.LIGHTGREEN_EX,
        'yellow': Fore.LIGHTYELLOW_EX,
        'blue': Fore.LIGHTBLUE_EX,
        'magenta': Fore.LIGHTMAGENTA_EX,
        'cyan': Fore.LIGHTCYAN_EX,
        'white': Fore.LIGHTWHITE_EX,
    }
    return {
        'file': sys.stdout,
        'total': total,
        'desc': " ",
        'dynamic_ncols': True,
        'bar_format': \
            "{l_bar}%s{bar}%s| [{elapsed}<{remaining}, {rate_fmt}{postfix}]" % (fore_colors[color], Fore.RESET),
        'leave': leave
    }

def get_logger(stream=False, logfile=None, level=logging.INFO):
    """
    Arguments:
        stream: bool, default False.
        logfile: str, path.
    """
    _format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logFormatter = logging.Formatter(_format)

    rootLogger = logging.getLogger()

    if logfile:
        touch(logfile)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    if stream:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(logFormatter)
        rootLogger.addHandler(streamHandler)

    rootLogger.setLevel(level)

    return rootLogger


def touch(filepath: str, mode: str='w'):
    assert mode in ['a', 'w']
    directory, _ = os.path.split(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    open(filepath, mode).close()
