# usage of the neuron extention can be found here - https://marketplace.visualstudio.com/items?itemName=neuron.neuron-IPE
#
# NOTE: 
#   - the path of the script and the path of the card could be different 
 

import os 
cwd = os.getcwd()
print (' Current path: {0}'.format(cwd))

# simple test
import pandas as pd
abs_path = 'c:\\repositories\\data-analysis-notebooks\\anomaly-detection\\'
data_path = os.path.join(abs_path, 'data', 'sunspots.txt')
df = pd.read_csv(data_path, sep='\t')
df.plot()
