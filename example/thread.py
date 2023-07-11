import time
def myfun(i):
    time.sleep(1)
    a = i
    # print(a)
    a_list.append(a)

# t = time.time()
# for _ in range(5):
#     myfun()
# print(time.time() - t)

from threading import Thread
a_list = []
ths = []
for i in range(5):
    th = Thread(target = myfun, args = (i,))
    th.start()
    ths.append(th)
for th in ths:
    th.join()
print(a_list) 
