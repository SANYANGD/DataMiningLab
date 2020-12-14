p = [{} for _ in range(0, 10)]
k = ['1','2','3','4','5','6','8','9','10','11']
for m in range(0, 10):
    for n in range(0, 10):
        p[m].update({k[n]: m})
print(p)