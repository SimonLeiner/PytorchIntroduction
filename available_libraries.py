"""
Name: available_libraries.py in Project: datascience
Author: Simon Leiner
Date: 18.10.21
Description: File to test whether all libraries are set up correctly
"""

import sys
import torch as tf
import pandas as pd
import sklearn as sk
import torch.cuda

print(f"Available Libraries:")
print("")
print(f"Pytorch Version: {tf.__version__}")
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = [tf.cuda.device(i) for i in range(tf.cuda.device_count())]
print(torch.cuda.device_count())
print("GPU:", gpu)
