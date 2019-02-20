"""It provides support to track execution times.
"""
#Enum
from enum import Enum, unique

#Bidirectional dictionary
from bidict import frozenbidict

#Locking
from threading import Lock                                                           
from functools import wraps as func_wrap
import time

#Save
from pickle import dump as pickle_dump

#Timing
from time import time

#Start and stop measurements
from random import randint

class Metric(object):
    #Metric data
    metric_data = {}
    metric_start_times = {}

    #Synchronization
    lock = Lock()

    @classmethod
    def create(cls, metric_name):
        """It creates a new metric.
        
        Arguments:
            metric_name {string} -- It creates a new metric.
        """
        #Initialize the metric_data if necessary
        if Metric.metric_data.get(metric_name) is None:
            Metric.metric_data[metric_name] = []

    @classmethod
    def publish(cls, metric_name, metric_value):
        """It publishes a metric to the store
        
        Arguments:
            metric_name {string} -- It creates a new metric
            metric_type {A TimingUnit object} -- The value of the metric
        """
        Metric.lock.acquire()

        try:
            Metric.metric_data[metric_name].append(metric_value)
        finally:
            Metric.lock.release()

    @classmethod
    def start(cls, metric_name):
        """It starts the measurement of time for the metric.
        
        Arguments:
            metric_name {string} -- The name of the metric
        """
        record_handle = randint(1, 123456789012345678901234567890)

        Metric.lock.acquire()

        try:
            if Metric.metric_start_times.get(record_handle):
                raise ValueError('record_handle: {} is already being measure'.format(record_handle))

            Metric.metric_start_times[record_handle] = time()
        finally:
            Metric.lock.release()

        return record_handle

    @classmethod
    def stop(cls, record_handle, metric_name):
        """It stops the measurement of time for the metric and records it.
        
        Arguments:
            record_handle {int} -- The handle to the measured record
            metric_name {string} -- The name of the metric
        """
        Metric.lock.acquire()

        try:
            if Metric.metric_start_times.get(record_handle) is None:
                raise ValueError('record_handle: {} is not being measured'.format(record_handle))

            #Compute the time elapsed from start to stop
            elapsed_time = time() - Metric.metric_start_times[record_handle]

            #Remove the start record from the register
            del Metric.metric_start_times[record_handle]

            #Save as metric
            Metric.metric_data[metric_name].append(elapsed_time)
        finally:
            Metric.lock.release()

    @classmethod
    def get(cls, metric_name):
        Metric.lock.acquire()

        try:
            return Metric.metric_data[metric_name]
        finally:
            Metric.lock.release()

    @classmethod
    def clear(cls):
        Metric.metric_data = {}

    @classmethod
    def save(cls, output_file_path):
        """It saves the metrics data to the output file.
        
        Arguments:
            output_file_path {A Path object} -- The output file path to store the metrics data.
        """
        with output_file_path.open(mode = 'wb') as handle:
            pickle_dump(Metric.metric_data, handle)