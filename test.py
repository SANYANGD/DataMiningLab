
import csv
import json

# p = [{} for _ in range(0, 10)]
# k = ['1','2','3','4','5','6','8','9','10','11']
# for m in range(0, 10):
#     for n in range(0, 10):
#         p[m].update({k[n]: m})
# print(p)

p = {}
k = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'g']
for n in range(0, 10):
    p.update({k[n]: n})

json_str = json.dumps(p) #dumps
with open('test_data.txt', 'w') as f:
    f.write(json_str)