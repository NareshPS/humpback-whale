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
    results = []

    #Multiprocessing pool and partial function for parallel execution
    partial_func = partial(func, *args)

    with Pool(5) as parallel_pool:
        with tqdm(total = length, desc = 'Processing in parallel: ') as pbar:
            for result in parallel_pool.imap_unordered(partial_func, iterator):
                #Append the response
                results.append(result)

                pbar.update(1)

    return results

def _execute_serial(func, iterator, length, *args):
    #Results placeholder
    results = []

    for item in tqdm(iterator, total = length, desc = 'Processing in sequence: '):
        result = func(*args, item)

        #Append the results to the output list
        results.append(result)

    return results

def execute(func, iterator, length, parallel, *args):
    logger = logging.get_logger(__name__)

    if parallel:
        print('Running parallel execution of function: ', func)

        return _execute_parallel(func, iterator, length, *args)
    else:
        logger.info('Running serial execution of function: %s', func)
        return _execute_serial(func, iterator, length, *args)