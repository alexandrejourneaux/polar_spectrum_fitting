#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:12:16 2021

@author: anatole
"""




import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy import optimize
import scipy.integrate as integrate
from basicfunctions import datainfolder, truncate, bgremoval, slicename, idx_too_close
import re
from tqdm import tqdm
import imageio

def nLorentzian(n):
    def f(x, *args):
        bg = args[0]
        h = args[1:n+1]
        c = args[n+1:2*n+1]
        w = args[2*n+1:]
        
        res = bg + sum([h[i]*w[i]**2/((x-c[i])**2+w[i]**2) for i in range(n)])
        
        return res
    
    return f

QD = 'C3'

folder=f'/home/alexandre/Documents/INSP/20220316/pola_analysis_{QD}/20220316'  

filesaving='/home/alexandre/Documents/INSP/codes/savings'

plot_all_spectra = True
plot_polar_separately = True

save = True

# For each QD, you need to specify :
#   - the indices of the data files that were 
#     removed (because of a dark count for instance) in the list deleted
#   - the number of pixels of separation under which two peaks are considered
#     to be the same
#   - the threshold of detection of the peaks on the averaged spectrum
#   - the number of peaks you want to fit on the spectra

if QD == 'B6':
    deleted = [54]
    position_precision = 4 # pixels
    threshold = 0.1
    npics = 2
    
if QD == 'B5':
    deleted = [8, 21]
    position_precision = 4 # pixels
    threshold = 0.1
    npics = 3
    
if QD == 'C3':
    deleted = [0, 6, 44, 57, 73, 79, 82]
    position_precision = 4 # pixels
    threshold = 0.1
    npics = 11

if QD == 'D2':
    deleted = [43, 60, 63, 68, 106]
    position_precision = 4 # pixels
    threshold = 0.1
    npics = 7

if QD == 'D6':  
    deleted = [2, 8, 42, 47, 87, 102, 105, 158, 170]
    position_precision = 4 # pixels
    threshold = 0.1
    npics = 5



# Data import

names=[]

datainfolder(folder,names)

n_pola = []
for name in names:
    for match in re.finditer('_', name):
        pos = match.end()
    n_pola.append(int(name[pos:-4]))
    
names=sorted(names, key=lambda x: n_pola[names.index(x)])

pola = np.linspace(0,375, 181)
pola= np.delete(pola, deleted)

x=np.loadtxt(folder+'/'+names[0])[:,2].tolist()


ytot=[]
ymaxtot=[]

for i in range(len(names)):
    name=names[i]
    y=bgremoval(x,np.loadtxt(folder+'/'+name)[:,-1].tolist())
    ytot.append(y)
    ymaxtot.append(np.max(y))
    
ymin=0
ymax=np.max(ymaxtot)


# Peaks finding : we find the npics highest peaks and store their position

averaged_spectrum = [np.mean([ytot[i][k] for i in range(len(ytot))]) for k in range(len(ytot[0]))]

peaks=scipy.signal.find_peaks(averaged_spectrum,height=threshold, width=0)
pos0=peaks[0].tolist()
h=peaks[1].get('peak_heights').tolist()
pos=[x[p] for p in pos0]

doubles = idx_too_close(pos0, position_precision)
for double in sorted(np.unique(doubles), reverse=True):
    pos0.pop(double)
    h.pop(double)
    pos.pop(double)


step=x[1]-x[0]

while len(pos)>npics:
    hm=h.index(np.min(h))
    h.pop(hm)
    pos.pop(hm)
    # w.pop(hm)
    pos0.pop(hm)


while len(pos)<npics:
    h.append(0)
    pos.append(pos[0])
    # w.append(0)

pos_peaks_idx = pos0
pos_peaks = pos

height = np.empty((npics, len(ytot)), dtype=float)
positions = np.empty((npics, len(ytot)), dtype=float)
widths = np.empty((npics, len(ytot)), dtype=float)


for k, y in tqdm(enumerate(ytot)):

    ym=np.max(y)
    y=[y[i]/ym for i in range(len(y))]
      
    peaks = scipy.signal.find_peaks(y,height=0.05, width=0)
    pos0=peaks[0]
    idx = []
    pos = []
    h = []
    w0 = []
    for p in pos_peaks_idx:
        idx.append((np.abs(pos0 - p)).argmin())
        pos.append(x[pos0[idx[-1]]])
        h.append(peaks[1].get('peak_heights')[idx[-1]])
        w0.append(peaks[1].get('widths')[idx[-1]])
    idx = np.array(idx)
    pos = np.array(pos)
    h = np.array(h)
    step=np.abs(x[1]-x[0])
    w=np.array(w0)*step
    bg0 = np.array((np.mean(y),))
    p0=np.concatenate((bg0, h, pos_peaks, w))
        
    bounds = ((-np.inf,) + npics*(-np.inf,) + tuple([pos_peaks[i] - 4*w[i] for i in range(npics)]) + npics*(-np.inf,),
              (np.inf,) + npics*(np.inf,) + tuple([pos_peaks[i] + 4*w[i] for i in range(npics)]) + npics*(np.inf,))
    
    try:
        p1,_=optimize.curve_fit(nLorentzian(npics), x, y, p0, bounds = bounds)
    except:
        p1 = np.array((3*npics+1) * [np.nan])
    p1=p1.tolist()
    y1=[nLorentzian(npics)(e, *p1)*ym for e in x]
    
    for i in range(npics):
        height[i, k] = p1[1+i]*ym
        positions[i, k] = p1[1 + npics + i]
        widths[i, k] = p1[1 + 2*npics + i]
    
    if plot_all_spectra:
        plt.figure()
        plt.plot(x, np.array(y)*ym)
        plt.plot(x, y1)
        plt.ylim([ymin, 1.1*ymax])
        plt.xlabel('Energy (eV)')
        plt.ylabel('Photoluminescence')
        plt.xlim([1.3406, 1.3418])
        plt.title(f"Polarisation {pola[k]}°, i = {k}")

plt.figure()
plt.title('Selected peaks on the averaged spectrum')
plt.xlabel('Energy (eV)')
plt.ylabel('Photoluminescence')
plt.ylim([0, 1.1*np.max(averaged_spectrum)])

for i, p in enumerate(pos_peaks_idx):
    # plt.plot([x[p], x[p]], [0, 1.1*np.max(averaged_spectrum)], c='r')
    plt.annotate(f'{i}', [x[p], averaged_spectrum[p]+3])
plt.plot(x, averaged_spectrum)
if save:
    plt.savefig(f'{filesaving}/{QD}_averaged_spectrum')

if not plot_polar_separately:
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
for i in range(npics):
    if plot_polar_separately:
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
    plt.title("Peaks height vs polarisation")
    ax.scatter(np.array(pola)*2*2*np.pi/360, height[i,:], c=f'C{i}', label=f'Peak {i} @ {round(np.nanmean(positions[i,:]), 3)} eV')
    plt.legend(bbox_to_anchor=(0.5,-0.1), loc=9)
    if save:
        plt.savefig(f'{filesaving}/{QD}_polar_peak_{i}@{round(np.nanmean(positions[i,:]), 3)}eV.png', bbox_inches = 'tight')


for i in range(npics):
    plt.figure()
    plt.title('Deviation from mean energy vs polarization')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Photoluminescence')
    mean = np.nanmean(positions[i,:])
    plt.plot(pola, [x - mean for x in positions[i,:]], c=f'C{i}', label=f'Peak {i} @ {round(np.nanmean(positions[i,:]), 3)} eV')
    plt.legend()
    if save:
        plt.savefig(f'{filesaving}/{QD}_energy_peak_{i}@{round(np.nanmean(positions[i,:]), 3)}eV.png')
    
for i in range(npics):
    plt.figure()
    plt.title('Peak width vs polarisation')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Photoluminescence')
    # plt.ylim([0.0002, 0.0004])
    plt.plot(pola, widths[i,:], c=f'C{i}', label=f'Peak {i} @ {round(np.nanmean(positions[i,:]), 3)} eV')
    plt.legend()
    if save: 
        plt.savefig(f'{filesaving}/{QD}_width_peak_{i}@{round(np.nanmean(positions[i,:]), 3)}eV.png')
