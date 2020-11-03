from tabulate import tabulate

N = 7


nums = [i for i in range(0, N)]

times = [t for t in range(0, N)]

res = [[[] for t in times] for t in times]


for i in range(0, N):

    t = times[i]

    for j in range(0, N):
        for k in range(0, N):

            if (j * t) % N == k:
                res[t][k].append(j)

print(tabulate(res))
