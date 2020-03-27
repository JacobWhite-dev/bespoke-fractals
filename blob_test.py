'''
Compare blobs
'''

import ast
import numpy as np

f = open("blob elements.txt", 'r')
blob1 = ""
for line in f:
    blob1 += line
    blob1 += ","

blob1 = blob1[:-1]

blob1 = ast.literal_eval(blob1)
f.close()

f = open("blob elements 2.txt", 'r')
blob2 = ""
for line in f:
    blob2 += line
    blob2 += ","

blob2 = blob2[:-1]

blob2 = ast.literal_eval(blob2)
f.close()

print(len(blob1))
print(len(blob2))

blobels = []
blobels.extend(blob1)
blobels.extend(blob2)
blobels = set([tuple(i) for i in blobels])

print(len(blobels))
