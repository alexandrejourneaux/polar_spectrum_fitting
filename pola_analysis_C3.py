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

def fixed_2Lorentzian(pos1, pos2):
    return lambda x, bg, h1, h2, w1, w2: (nLorentzian(2)(x, bg, h1, h2, pos1, pos2, w1, w2))

QD = 'C3'

folder=f'/home/alexandre/Documents/INSP/20220316/pola_analysis_{QD}/20220316'  

filesaving='/home/alexandre/Documents/INSP/codes/savings'

plot_all_spectra = True
plot_polar_separately = True

save = False

# For each QD, you need to specify :
#   - the indices of the data files that were 
#     removed (because of a dark count for instance) in the list deleted
#   - the number of pixels of separation under which two peaks are considered
#     to be the same
#   - the threshold of detection of the peaks on the averaged spectrum
#   - the number of peaks you want to fit on the spectra

    
if QD == 'C3':
    deleted = [0, 6, 44, 57, 73, 79, 82]
    position_precision = 4 # pixels
    threshold = 0.1
    npics = 11

npics = 2

peak_low = 1.3412648711286017
# peak_high = 1.3412942100328427
peak_high = 1.3412962100328427

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

x_study = np.array(x[640:690])
y_study = np.array(ytot)[:,640:690]

height = np.empty((npics, len(y_study)), dtype=float)
widths = np.empty((npics, len(y_study)), dtype=float)


for k, y in tqdm(enumerate(y_study)):

    ym=np.max(y)
    y=[y[i]/ym for i in range(len(y))]
    
    p0=(0.1, 0.5, 0.5, 0.00005, 0.00005)
        
    bounds = ((-np.inf,) + npics*(-np.inf,) + npics*(-np.inf,),
              (np.inf,) + npics*(np.inf,) + npics*(np.inf,))
    
    try:
        p1,_=optimize.curve_fit(fixed_2Lorentzian(peak_low, peak_high), x_study, y, p0)
    except:
        p1 = np.array((2*npics+1) * [np.nan])
    p1=p1.tolist()
    x_oversampled = np.linspace(x_study[0], x_study[-1], 1000)
    y1=[nLorentzian(1)(e, p1[0], p1[1], peak_low, p1[3])*ym for e in x_oversampled]
    y2=[nLorentzian(1)(e, p1[0], p1[2], peak_high, p1[4])*ym for e in x_oversampled]
    
    height[0, k] = p1[1]*ym
    widths[0, k] = p1[3]
    height[1, k] = p1[2]*ym
    widths[1, k] = p1[4]
    
    # for i in range(npics):
    #     height[i, k] = p1[1+i]*ym
    #     positions[i, k] = p1[1 + npics + i]
    #     widths[i, k] = p1[1 + 2*npics + i]
    
    if plot_all_spectra:
        plt.figure()
        plt.plot(x_study, np.array(y)*ym)
        plt.plot(x_oversampled, y1, c='C1')
        plt.plot(x_oversampled, y2, c='C2')
        plt.ylim([ymin, 1.1*ymax])
        plt.xlabel('Energy (eV)')
        plt.ylabel('Photoluminescence')
        # plt.xlim([1.340, 1.342])
        plt.title(f"Polarisation {pola[k]}°, i = {k}")
        
def cos_fit(x, mean, A, f, phi): 
    return mean + A*np.cos(2*np.pi*f*x + phi)

p_cos1,_ = optimize.curve_fit(cos_fit,
                            pola[np.where(height[0,:] == height[0,:])],
                            height[0,:][np.where(height[0,:] == height[0,:])],
                            (50, 100, 0.01, 0))

p_cos2,_ = optimize.curve_fit(cos_fit,
                            pola[np.where(height[1,:] == height[1,:])],
                            height[1,:][np.where(height[1,:] == height[1,:])],
                            (50, 100, 0.01, np.pi))

plt.figure()
plt.scatter(pola, height[0,:], c='C1')
plt.scatter(pola, height[1,:], c='C2')
plt.plot(np.linspace(pola[0], pola[-1], 1000),
         [cos_fit(x, *p_cos1) for x in np.linspace(pola[0], pola[-1], 1000)], c='C1')
plt.plot(np.linspace(pola[0], pola[-1], 1000),
         [cos_fit(x, *p_cos2) for x in np.linspace(pola[0], pola[-1], 1000)], c='C2')




# plt.figure()
# plt.title('Selected peaks on the averaged spectrum')
# plt.xlabel('Energy (eV)')
# plt.ylabel('Photoluminescence')
# plt.ylim([0, 1.1*np.max(averaged_spectrum)])

# for i, p in enumerate(pos_peaks_idx):
#     # plt.plot([x[p], x[p]], [0, 1.1*np.max(averaged_spectrum)], c='r')
#     plt.annotate(f'{i}', [x[p], averaged_spectrum[p]+3])
# plt.plot(x, averaged_spectrum)
# if save:
#     plt.savefig(f'{filesaving}/{QD}_averaged_spectrum')

# if not plot_polar_separately:
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='polar')
# for i in range(npics):
#     if plot_polar_separately:
#         fig = plt.figure()
#         ax = fig.add_subplot(projection='polar')
#     plt.title("Peaks height vs polarisation")
#     ax.scatter(np.array(pola)*2*2*np.pi/360, height[i,:], c=f'C{i}', label=f'Peak {i} @ {round(np.nanmean(positions[i,:]), 3)} eV')
#     plt.legend(bbox_to_anchor=(0.5,-0.1), loc=9)
#     if save:
#         plt.savefig(f'{filesaving}/{QD}_polar_peak_{i}@{round(np.nanmean(positions[i,:]), 3)}eV.png', bbox_inches = 'tight')


# for i in range(npics):
#     plt.figure()
#     plt.title('Deviation from mean energy vs polarization')
#     plt.xlabel('Energy (eV)')
#     plt.ylabel('Photoluminescence')
#     mean = np.nanmean(positions[i,:])
#     plt.plot(pola, [x - mean for x in positions[i,:]], c=f'C{i}', label=f'Peak {i} @ {round(np.nanmean(positions[i,:]), 3)} eV')
#     plt.legend()
#     if save:
#         plt.savefig(f'{filesaving}/{QD}_energy_peak_{i}@{round(np.nanmean(positions[i,:]), 3)}eV.png')
    
# for i in range(npics):
#     plt.figure()
#     plt.title('Peak width vs polarisation')
#     plt.xlabel('Energy (eV)')
#     plt.ylabel('Photoluminescence')
#     # plt.ylim([0.0002, 0.0004])
#     plt.plot(pola, widths[i,:], c=f'C{i}', label=f'Peak {i} @ {round(np.nanmean(positions[i,:]), 3)} eV')
#     plt.legend()
#     if save: 
#         plt.savefig(f'{filesaving}/{QD}_width_peak_{i}@{round(np.nanmean(positions[i,:]), 3)}eV.png')

# peak_low = np.min(positions[1,:])
# peak_high = np.max(positions[1,:])