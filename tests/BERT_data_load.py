## for data
import json
import pandas as pd
import numpy as np
from sklearn import metrics, manifold

lst_dics = []
with open('data.json', mode='r', errors='ignore') as json_file:
    for dic in json_file:
        lst_dics.append( json.loads(dic) )

## print the first one
print(lst_dics[0])

## create dtf
dtf = pd.DataFrame(lst_dics)

## filter categories
dtf = dtf[ dtf["category"].isin(['ENTERTAINMENT','POLITICS','TECH'])        ][["category","headline"]]

## rename columns
dtf = dtf.rename(columns={"category":"y", "headline":"text"})

## print 5 random rows
print(dtf.sample(5))