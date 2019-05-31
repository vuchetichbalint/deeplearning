#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 04:30:46 2018

@author: balint
"""

import pandas as pd
import numpy as np

import itertools

#%%

def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

d = {
        'experiment_name': ['ta_e04'],
        'env_name': [ 'CartPole-v0', 'Pendulum-v0'],
        'noise_method': ['asn', 'psn'],
        'seed': list(range(1,21)),
    }



#%%

res_ac = []
res_pc = []
res_ap = []
res_pp = []

for p in dict_product(d):
    f_path = f"/Users/balint/workspace/deeplearning/szakdolgozat_scriptek/{p['experiment_name']}/{p['env_name']}/{p['noise_method']}/{str(p['seed'])}/monitor.csv"
    print(f_path)
    df = pd.read_csv(f_path, skiprows=1)
    if p['env_name'] == 'CartPole-v0':
        if p['noise_method'] == 'asn':
            res_ac = np.append(res_ac, df.iloc[-1,2])
        else:
            res_pc = np.append(res_pc, df.iloc[-1,2])
    if p['env_name'] == 'Pendulum-v0':
        if p['noise_method'] == 'asn':
            res_ap = np.append(res_ap, df.iloc[-1,2])
        else:
            res_pp = np.append(res_pp, df.iloc[-1,2])



#%%

df = pd.DataFrame({'ac':res_ac, 'pc':res_pc, 'ap':res_ap, 'pp':res_pp})

print('mean')
print(df.mean())

print('std')
print(df.std())


#%%


#%%


#%%


#%%

