'''
Created on Dec 31, 2018

@author: wangxing
'''
import multiprocessing as mp
import os
import time

def f(x):
    return x*x

# def f(conn):
#     conn.send([42, None, 'hello'])
#     conn.close()
    
def printi(l, i):
    l.acquire()
    try:
        print('hello world', i)
    finally:
        l.release()
        
def printi_nolock(i):
    print('hello world', i)

def info(title):
    print('-----', title, '------')
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id', os.getpid())

def store(n, a):
    n.value = 3.14159
    for i in range(len(a)):
        a[i] = -a[i]

def hello(name):
    info('func hello')
    print('hello', name)
    
def foo(q):
    q.put('Hello')
    
def dl(d, l):
    d[1] = '1'
    d['2'] = 2
    d[0.25] = None
    l.reverse()

if __name__=='__main__':
    print(mp.cpu_count())
    with mp.Pool(processes=4) as pool:
        print(pool.map(f, range(10)))
        print('------------------------------------------')
        for i in pool.imap_unordered(f, range(10)):
            print(i)
        
        res = pool.apply_async(f, [10])
        print(res.get(timeout=1))
        
        res = pool.apply_async(time.sleep, [10])
        print(res.get(timeout=1))
    
    
#     with mp.Manager() as manager:
#         d = manager.dict()
#         l = manager.list(range(10))
#         
#         p = mp.Process(target=dl, args=(d, l))
#         p.start()
#         p.join()
#         
#         print(d)
#         print(l)
    
#     num = mp.Value('d', 0.)
#     arr = mp.Array('i', range(10))
#     p = mp.Process(target=store, args=(num, arr))
#     p.start()
#     p.join()
#     
#     print(num.value)
#     print(arr[:])
    
# #     l = mp.Lock()
#     for i in range(100):
#         mp.Process(target=printi_nolock, args=(i,)).start()
# #         mp.Process(target=printi, args=(l, i)).start()
    
#     parent_conn, child_conn = mp.Pipe()
#     p = mp.Process(target=f, args=(child_conn,))
#     p.start()
#     print(parent_conn.recv())
#     p.join()
    
#     q = mp.Queue()
#     p = mp.Process(target=f, args=(q,))
#     p.start()
#     print(q.get())
#     p.join()

#     pool = mp.Pool(5)
#     print(pool.map(f, range(5)))
     
#     info('main line')
#     proc = mp.Process(target=hello, args=('bob',))
#     proc.start()
#     proc.join()
#     
#     ctx = mp.get_context('spawn')
# #     mp.set_start_method('spawn')
#     q = ctx.Queue()
#     p = ctx.Process(target=foo, args=(q,))
#     p.start()
#     print(q.get())
#     p.join()