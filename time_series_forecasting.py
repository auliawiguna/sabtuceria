__author__ = 'ahmadauliawiguna'
# load and plot dataset

import os #import library untuk akses fitur OS

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

cwd = os.getcwd()
raw_data = cwd + "/dataset/international-airline-passengers.csv"

#Ubah kolom Month dari tahun-bulan-tanggal menjadi tahun-bulan-tanggal biar dianggap bertipe DATE
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv(raw_data,sep=";", parse_dates=['Month'], index_col='Month',date_parser=dateparse)
print data.head()
print(data.index)

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6




