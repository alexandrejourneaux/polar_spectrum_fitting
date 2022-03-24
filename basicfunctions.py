#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:38:53 2021

@author: anatole
"""

import re
import numpy as np
import os
import math
from copy import deepcopy

def slicename(name):
    name=name[:-4]
    spots=[]
    for m in re.finditer('_', name):
        spots.append(m.start())
        
    _type=name[0:spots[0]]        
    
    if _type=='ref':
        _diam=np.float(name[spots[3]+1:spots[4]])
        _a=np.float(name[spots[4]+1:spots[5]])
        
        return [_diam, _a]
     
    if _type=='spectrepola' or 'polaspectre'or'randomdot':
        _temp=np.float(name[spots[0]+2:spots[1]])    
        _exc=name[spots[1]+1:spots[2]]  
        _powint=name[spots[2]+1:spots[3]] 
        if _powint.endswith('nW'):
            _pow=np.float(_powint[1:-2])*10**-9
        if _powint.endswith('muW'):
            _pow=np.float(_powint[1:-3])*10**-6
        if _powint.endswith('mW'):
            _pow=np.float(_powint[1:-2])*10**-3
        _center=np.float(name[spots[3]+1:spots[4]-2]) 
        _unknown=name[spots[4]+1:spots[5]] 
        _time=np.float(name[spots[5]+1:spots[6]-1])
        if len(spots)==8:
            _nacq=name[spots[6]+1:spots[7]] 
            _npola=np.float(name[spots[7]+1:])
        else:
            _nacq=name[spots[6]+1:] 
            _npola=0 
            
        return[_type,_temp,_exc,_pow,_center,_time,_npola]
            
    if _type=='randomdot':
        _temp=np.float(name[spots[0]+2:spots[1]])    
        _exc=name[spots[1]+1:spots[2]]  
        _powint=name[spots[2]+1:spots[3]] 
        if _powint.endswith('nW'):
            _pow=np.float(_powint[1:-2])*10**-9
        if _powint.endswith('muW'):
            _pow=np.float(_powint[1:-3])*10**-6
        if _powint.endswith('mW'):
            _pow=np.float(_powint[1:-2])*10**-3
        _center=np.float(name[spots[3]+1:spots[4]-2]) 
        _unknown=name[spots[4]+1:spots[5]] 
        _time=np.float(name[spots[5]+1:spots[6]-1])
        if len(spots)==8:
            _nacq=name[spots[6]+1:spots[7]] 
            _npola=np.float(name[spots[7]+1:])
        else:
            _nacq=name[spots[6]+1:] 
            _npola=0 
            
        return[_type,_temp,_exc,_pow,_center,_time,_npola]
        
    if _type=='spectre':
        _temp=np.float(name[spots[0]+2:spots[1]])    
        _exc=name[spots[1]+1:spots[2]]  
        _powint=name[spots[2]+1:spots[3]] 
        if _powint.endswith('nW'):
            _pow=np.float(_powint[1:-2])*10**-9
        if _powint.endswith('muW'):
            _pow=np.float(_powint[1:-3])*10**-6
        if _powint.endswith('mW'):
            _pow=np.float(_powint[1:-2])*10**-3
        _center=np.float(name[spots[3]+1:spots[4]-2]) 
        _unknown=name[spots[4]+1:spots[5]] 
        _time=np.float(name[spots[5]+1:spots[6]-1])
        _nacq=name[spots[6]:]
        
        return[_type,_temp,_exc,_pow,_center,_time]
        
        
    if _type=='T1':
        _temp=np.float(name[spots[0]+2:spots[1]])       
        _powint=name[spots[1]+1:spots[2]] 
        if _powint.endswith('nW'):
            _pow=np.float(_powint[1:-2])*10**-9
        if _powint.endswith('muW'):
            _pow=np.float(_powint[1:-3])*10**-6
        if _powint.endswith('mW'):
            _pow=np.float(_powint[1:-2])*10**-3
        _center=np.float(name[spots[2]+1:spots[3]-2])
        _exc=name[spots[3]+1:spots[4]]
        
        return[_type,_temp,_exc,_pow,_center]
        


def datainfolder(folder, names):
    for i in os.listdir(folder):
        if i.endswith('txt'):   
            names.append(i)
            
            
def truncate(number, digits):
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper
          

def bgremoval(x,y):
    bgg=np.mean(y[:10])
    bgd=np.mean(y[-10:])
    bg=[bgg+(x[i]-x[5])*(bgd-bgg)/(x[-5]-x[5]) for i in range(len(x))]
    y=[y[i]-bg[i] for i in range(len(y))]
    ym=np.min(y)
    y=[y[i]-ym for i in range (len(y))]
    return y

def idx_too_close(l, max_diff):
    res = []
    ll = deepcopy(l)
    for x in l:
        for idx, y in enumerate(ll):
            if y - x < max_diff and x < y:
                res.append(idx)
    return res