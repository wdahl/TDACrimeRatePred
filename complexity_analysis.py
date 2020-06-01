import numpy as np
from ripser import ripser
from persim import plot_diagrams
import numpy as np
from ripser import Rips
import pandas as pd


import time
runTime=[]
# sizes=[10,100,1000,10000,100000]
# sizes=[10,100,1000]
# for x in sizes:
xaxis=[]
for x in range(1,100):
    start_time = time.time()

    rips = Rips()
    data = np.random.random((x*100,2))
    xaxis.append(x*100)
    print(data.shape)
    diagrams = rips.fit_transform(data)
    #print(diagrams)
    #rips.plot(diagrams)

    runTime.append((time.time() - start_time))
    print("--- %s seconds ---" % (time.time() - start_time))

print(runTime)

import matplotlib.pyplot as plt
plt.plot(xaxis,runTime,'ro')

plt.xlabel('Number of data points')
plt.ylabel('run time in seconds')
plt.show()