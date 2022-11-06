import numpy


binaryList = [0,1]

h = []

for a in binaryList:
    for b in binaryList:
        for c in binaryList:
            answer = (a and b) or (a and c)


print(h[0])