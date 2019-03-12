#Parallel execution
from multiprocessing import Pool
from functools import partial

#Logging
from common import logging

#Progress bar
from tqdm import tqdm

import time

def _execute_parallel(func, iterator, length, *args):
    #Results placeholder
    results = None

    #Multiprocessing pool and partial function for parallel execution
    partial_func = partial(func, *args)

    with Pool(5) as parallel_pool:
        worker_iterator = parallel_pool.imap(partial_func, iterator, chunksize = 8)
        iterator_with_progress_bar = tqdm(worker_iterator, total = length, desc = 'Processing in parallel: ')

        #Collect responses
        results = list(iterator_with_progress_bar)

    return results

def _execute_serial(func, iterator, length, *args):
    #Measure progress
    iterator_with_progress_bar = tqdm(iterator, total = length, desc = 'Processing in sequence: ')

    #Collect results
    results = list(map(lambda item: func(*args, item), iterator_with_progress_bar))

    return results

def execute(func, iterator, length, parallel, *args):
    logger = logging.get_logger(__name__)

    if parallel:
        print('Running parallel execution of function: ', func)

        return _execute_parallel(func, iterator, length, *args)
    else:
        logger.info('Running serial execution of function: %s', func)
        return _execute_serial(func, iterator, length, *args)