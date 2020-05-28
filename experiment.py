import numpy as np
import re

pad = []
for i in range(0, 36):
    pad.append(0.0)




print (np.array(pad).shape)

temp = np.arange(36)

for i in range(36):
    temp[i] = 20

for i in range(2):
    temp = np.vstack((temp,pad))

print(temp)
print(temp.shape)

file = "stand.csv"
if re.search("sit", file):
    label = 0
elif re.search("stand", file):
    label = 2
elif re.search("walk", file):
    label = 3
elif re.search("bend", file):
    label = 4
elif re.search("fall", file):
    label = 5


y = []
for i in range(5):
    y.append([0])

print(y)
print(np.array(y).shape)