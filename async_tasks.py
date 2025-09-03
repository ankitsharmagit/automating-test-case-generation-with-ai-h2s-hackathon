import threading
import time
import sys
import uuid
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import re

class ExecutorStopped(Exception):
    pass


class AsyncTask:
    def __init__(self, method, **kwargs):
        self.initialized_at=datetime.now()
        self.task_id = str(uuid.uuid4())
        self._method = method
        self._kwargs = kwargs
        self._task_canceled = False
        # for key, value in kwargs.items():
        #     setattr(self, key, value)

    def __call__(self,*args ,**kwargs):
        return self._method(*args, **self._kwargs, **kwargs)


class SafeCounter:
    def __init__(self, intial=0.0):
        self._lock = threading.Lock()
        self._value = intial

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, new_value):
        with self._lock:
            self._value = new_value

    def add(self, value):
        with self._lock:
            self._value += value

    def subtract(self, value):
        with self._lock:
            self._value -= value


class AsyncTasksExecutor:
    RESET_EVERY_SEC = 2
    MAX_CONCURRENT_DISPATCHES_SLEEP_SEC = 1.0

    def __init__(self, tasks_per_min, max_concurrent_dispatches=100, trials_cnt=3, silent=True):
        self.tasks_per_min = tasks_per_min
        self._tasks_queue = Queue()
        self._rate_lock = threading.Lock()
        self._running_lock = threading.Lock()
        self._last_reset_time = None
        self._tasks_sent_this_minute = SafeCounter()
        self._is_sleeping = False
        self._executor_stopped = False
        self._executor_running = False
        self.max_concurrent_dispatches = max_concurrent_dispatches
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_dispatches)
        self._results_lock = threading.Lock()
        self._results = {}
        self._trials_cnt = trials_cnt

        self._no_running_threads = SafeCounter()
        self._no_tasks_in_queue = SafeCounter()
        self.tasks_per_sec = tasks_per_min // 30
        self._silent = silent
        self.max_concurrent_threads_reached = 0

    def add_task(self, async_task:AsyncTask):
        task_id = async_task.task_id
        self._tasks_queue.put(async_task)
        self._no_tasks_in_queue.add(1)
        return task_id

    def _stop_task_executor(self):
        self._results_lock.acquire()
        self._running_lock.acquire()

        self._executor_stopped = True
        self._tasks_queue = Queue()
        self._no_tasks_in_queue.value = 0
        results_copy = self._results.copy()
        self._results = {}

        self._results_lock.release()
        self._running_lock.release()
        return results_copy

    def check_executor_stopped(self):
        with self._running_lock:
            if self._executor_stopped:
                raise ExecutorStopped("Executor stopped")

    def _start_task_executor(self):
        self._results = {}
        if not self._executor_running:
            if self._no_running_threads.value > 0:
                print(
                    f"{self._no_running_threads.value} tasks from past operation are still running sleeping till it finished")
                while self._no_running_threads.value > 0:
                    time.sleep(5)
                print(f"Remaining Tasks finished starting executor")

            self._executor_stopped = False
            self._is_sleeping = False
            executor_thread = threading.Thread(target=self._tasks_executor, daemon=True, name="cloud_tasks_executor")
            executor_thread.start()
            self._executor_running = True

    def run_async_task(self, async_task):
        self._no_running_threads.add(1)
        self.max_concurrent_threads_reached = max(self._no_running_threads.value, self.max_concurrent_threads_reached)
        req_start = time.time()
        thrown_execption = None
        result=None
        response_data = {"result": None,"succeeded":False,"inserted_to_queue_at":async_task.initialized_at ,"req_start_time": datetime.now()}
        try:
            trials_used = 1
            for trail in range(self._trials_cnt):
                self.check_executor_stopped()
                try:
                    result=async_task()


                except Exception as e:
                    result = None
                    thrown_execption = e
                if result:
                    break
                else:
                    trials_used += 1
                    time.sleep(1)
            response_data["req_end_time"] = datetime.now()
            response_data["response_time"] = round(time.time() - req_start, 5)
            response_data["no_trials"] = trials_used
            with self._results_lock:
                if result:
                    self.check_executor_stopped()
                    response_data['result'] = result
                    response_data['succeeded']=True
                else:
                    response_data['result'] = str(thrown_execption)
                self._results[async_task.task_id] = response_data

        except ExecutorStopped as e:
            pass
        finally:
            self._no_running_threads.subtract(1)

    def print_results(self):
        with self._results_lock:
            for result in self._results:
                print(result)

    def batch_processing(self, data: list):
        result_list = None
        time_remaining = "Nan"
        try:
            tasks_list = []
            self._start_task_executor()
            for task in data:
                task_id = self.add_task(task)
                tasks_list.append(task_id)
            no_finished_tasks = 0
            start = time.time()
            while True:
                with self._results_lock:
                    no_finished_tasks = len(self._results)
                if no_finished_tasks == len(tasks_list):
                    break
                time_taken = round(time.time() - start, 3)
                if no_finished_tasks > 0:
                    task_per_sec = time_taken / no_finished_tasks
                    time_remaining = str(round((task_per_sec * (len(data) - no_finished_tasks)) / 60.0, 3))

                sys.stdout.write("\r Finshed " + str(no_finished_tasks) + " of " + str(len(tasks_list)) +
                                 "in " + str(round(time.time() - start, 3)) +
                                 " current running tasks " + str(self._no_running_threads.value) +
                                 " Tasks in Queue " + str(self._no_tasks_in_queue.value) +
                                 " Time remaining (m) " + time_remaining
                                 )
                # print()
                sys.stdout.flush()
                time.sleep(1)



        except Exception as e:
            print(e)
        except KeyboardInterrupt:
            pass
        finally:
            results_dict = self._stop_task_executor()
            result_list = [results_dict[task_id] for task_id in tasks_list if task_id in results_dict]

            self._results = {}
            sys.stdout.write(r"Finshed " + str(no_finished_tasks) + " of " + str(len(tasks_list)) + "in " + str(
                round(time.time() - start, 3)))
            # sys.stdout.flush()

        return result_list

    def _tasks_executor(self):
        self._executor_stopped = False
        print("Tasks Executor Started")
        while True:
            with self._running_lock:
                if self._executor_stopped:
                    print("tasks_executor stopped")
                    break

            if self._tasks_queue.empty():
                if not self._is_sleeping:
                    self._is_sleeping = True
                    if not self._silent:
                        print("Queue Empty sleeping")
                time.sleep(0.2)
                continue
            self._is_sleeping = False
            now = time.time()
            # Reset the counter every minute
            if not self._last_reset_time or (now - self._last_reset_time >= self.RESET_EVERY_SEC):
                with self._rate_lock:
                    self._tasks_sent_this_minute.value = 0
                    self._last_reset_time = now

            if self._tasks_sent_this_minute.value < self.tasks_per_sec:

                if self._no_running_threads.value < self.max_concurrent_dispatches:
                    cloud_task = self._tasks_queue.get()

                    self._thread_pool.submit(self.run_async_task, cloud_task)
                    self._tasks_queue.task_done()
                    self._no_tasks_in_queue.subtract(1)
                    self._tasks_sent_this_minute.add(1)
                else:
                    time.sleep(self.MAX_CONCURRENT_DISPATCHES_SLEEP_SEC)
                    # print(f"Max concurrent threads limit reached sleeping {self.MAX_CONCURRENT_DISPATCHES_SLEEP_SEC}")

            else:
                sleep_time = self.RESET_EVERY_SEC - (now - self._last_reset_time)
                if not self._silent:
                    print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
            time.sleep(0.1)
        self._executor_running = False
        print("Tasks Executor Finished")

    def print_cloud_task_config(self):
        print(f"""
        Tasks Per minute : {self.tasks_per_min}
        Max concurrent tasks : {self.max_concurrent_dispatches}
        Max concurrent tasks reached : {self.max_concurrent_threads_reached}
        """)
