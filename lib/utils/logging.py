import time
from loguru import logger

from lib.utils.timetool import with_time


def timeit_logger(func):
    func_name = func.__qualname__

    def wrapped(*args, **kwargs):
        logger.opt(colors=True).info(
            "* Entering <u><blue>{func_name}</blue></u>...", func_name=func_name
        )
        func_wrapped = with_time(func, pretty_time=True)
        result, time_cost = func_wrapped(*args, **kwargs)
        logger.opt(colors=True).info(
            "* <u><blue>{func_name}</blue></u> finished. Time cost: {time_cost}",
            func_name=func_name,
            time_cost=time_cost,
        )
        return result

    return wrapped
