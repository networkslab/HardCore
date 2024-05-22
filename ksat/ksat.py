import os
from numpy import random
from tqdm import tqdm
k = 1351
names = {}
for i in tqdm(range(k)):
    m = int(random.normal(400, 100 ))
    c = random.normal(4.4, 0.05)
    n = int(m*c)
    name = str(m) + '_' + str(c) + '_' + str(n)
    if name in names.keys():
        names[name] += 1
    else:
        names[name] = 0
    os.system('cnfgen randkcnf 3 ' + str(m) + ' ' + str(n) + 
        '>/home/joseph-c/sat_gen/cnfgen/ksat/ksat_' + name + '_' + str(names[name]))
    