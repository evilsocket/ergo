from threading import Thread
import multiprocessing
import logging as log
import queue 
import time
import os

class TaskQueue(queue.Queue):
    def __init__(self, name, num_workers=-1, blocking=False):
        queue.Queue.__init__(self)
        self.name = name
        self.blocking = blocking
        self.num_workers = num_workers if num_workers > 0 else (multiprocessing.cpu_count() * 2)
        self._start_workers()

    def add_task(self, task, *args, **kwargs):
        args = args or ()
        kwargs = kwargs or {}
        self.put((task, args, kwargs), block=self.blocking)

    def _start_workers(self):
        log.info("starting %d workers for %s" % (self.num_workers, self.name))
        for i in range(self.num_workers):
            t = Thread(target=self._worker)
            t.daemon = True
            t.start()

    def _worker(self):
        log.debug("task queue worker started")
        while True:
            item, args, kwargs = self.get()
            item(*args, **kwargs)  
            self.task_done()

