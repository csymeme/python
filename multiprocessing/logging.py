import logging
import multiprocessing

from multiprocessing import Process,Lock

def printer(item,lock):
	lock.acquire()
	try:
		print item
	finally:
		lock.release()

if __name__ == '__main__':
	lock=Lock()
	items=['tango','fox',10]
	multiprocessing.log_to_stderr()
	logger=multiprocessing.get_logger()
	logger.setLevel(logging.INFO)
	for item in items:
		p=Process(target=printer,args=(item,lock))
		p.start()