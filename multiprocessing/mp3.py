from multiprocessing import Process,Lock
#锁

'''
为了避免线程之间互相阻塞，我们使用Lock对象。
代码循环列表中的三个项并为它们各自都创建一个进程。
每一个进程都将调用我们的函数，
并且每次遍历到的那一项作为参数传入函数。
因为我们现在使用了锁，所以队列中下一个进程将一直阻塞，
直到之前的进程释放锁。
'''
def printer(item,lock):
	lock.acquire()
	try:
		print item
	finally：
		lock.release()

if __name__ == '__main__':
	lock=Lock()
	items=['tango','foxtrot',10]
	for item in items:
		p=Process(target=printer,args=(item,lock))
		p.start()
