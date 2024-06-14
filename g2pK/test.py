from g2pkc.g2pk import G2p
import time

g2p = G2p()

start = time.time()
t = '지칠줄 모르게 달려가는 그. 따라오는 3422 4명의 사람들. rlaew가 있다. 앓고 있붙|안녕하세요.'
# lst = [g2p(_t) for _t in t.split()]
print(g2p(t, use_dict=False)[:220])
end = time.time()
print(f'{round(end-start, 3)} passed during g2p')

# print(g2p(t))