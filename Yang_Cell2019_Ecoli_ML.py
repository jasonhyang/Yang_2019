#%% Clear and reset workspace

%clear
%reset

import timeit

#%% Import data

import pandas as pd
import numpy as np
import csv

IC50s_log2 = pd.read_csv('Yang_Cell2019_IC50s_log2.csv')
fluxData = pd.read_csv('Yang_Cell2019_Sims.csv')

# Create reaction: subsystem dictionary
with open('Yang_Cell2019_Subsystems.csv') as csvfile:
    subsystems = dict(csv.reader(csvfile))

del csvfile

#%% Convert to reversible fluxes

# Find and relabel forward reactions
rxns = list(fluxData.Reactions.values)
col = list(fluxData.columns.values)
wells = col[1:]

for idx in range(len(rxns)):
    rxn = rxns[idx]
    if rxn.endswith('_f'):
        rxns[idx] = rxn[:-2]

fluxData.Reactions = rxns

# Find and invert backward reactions
rxns = list(fluxData.Reactions.values)
fluxes = fluxData.values
idx_rmv = []

for idx in range(len(rxns)):
    rxn = rxns[idx]
    if rxn.endswith('_b'):
        rxns[idx] = rxn[:-2]
        if rxns[idx] == rxns[idx+1]:
            fluxes[idx+1,1:] = fluxes[idx+1,1:] - fluxes[idx,1:]
            idx_rmv.append(idx)
        else:
            rxns[idx] = rxn[:-2]
            
fluxData = pd.DataFrame(fluxes)
fluxData.columns = col
fluxData.Reactions = rxns
fluxData = fluxData.drop(fluxData.index[idx_rmv])

# Remove exchange and transport reactions
rxns = list(fluxData.Reactions.values)
idx_rmv = []

for idx in range(len(rxns)):
    rxn = rxns[idx]
    if subsystems[rxn] == 'Exchange':
        idx_rmv.append(idx)
    elif subsystems[rxn] == 'Transport Inner Membrane':
        idx_rmv.append(idx)
    elif subsystems[rxn] == 'Transport Outer Membrane':
        idx_rmv.append(idx)
    elif subsystems[rxn] == 'Transport Outer Membrane Porin':
        idx_rmv.append(idx)
    elif subsystems[rxn] == 'Inorganic Ion Transport and Metabolism':
        idx_rmv.append(idx)
    elif rxn == 'netFlux':
        idx_rmv.append(idx)

fluxData = fluxData.drop(fluxData.index[idx_rmv])
rxns = list(fluxData.Reactions.values)

del idx, rxn, idx_rmv, col, fluxes

#%% Filter IC50s

IC50s_log2 = IC50s_log2.set_index('Well')

# Filter log2 IC50s
log2_filt = IC50s_log2.loc[IC50s_log2.index.intersection(wells)]
amp_log2 = log2_filt['AMP IC50'].values
cip_log2 = log2_filt['CIP IC50'].values
gent_log2 = log2_filt['GENT IC50'].values
multi_log2 = np.transpose(np.array([amp_log2,cip_log2,gent_log2]))

del log2_filt

#%% Preprocess flux data
from sklearn import preprocessing

fluxes = np.transpose(fluxData.values[:,1:]).astype(float)
fluxes_scaled = preprocessing.maxabs_scale(fluxes)

#%% Elastic Net
from sklearn.linear_model import ElasticNetCV, MultiTaskElasticNetCV

%clear

# Perform MultiTask Elastic Net

tic = timeit.default_timer()
eNet_amp_0p01 = ElasticNetCV(l1_ratio=0.01, max_iter=10000, tol=0.000001, cv=50, n_jobs=-1, selection='random').fit(fluxes_scaled, amp_log2)
toc = timeit.default_timer()
print(toc-tic)

tic = timeit.default_timer()wq
eNet_cip_0p01 = ElasticNetCV(l1_ratio=0.01, max_iter=10000, tol=0.000001, cv=50, n_jobs=-1, selection='random').fit(fluxes_scaled, cip_log2)
toc = timeit.default_timer()
print(toc-tic)

tic = timeit.default_timer()
eNet_gent_0p01 = ElasticNetCV(l1_ratio=0.01, max_iter=10000, tol=0.000001, cv=50, n_jobs=-1, selection='random').fit(fluxes_scaled, gent_log2)
toc = timeit.default_timer()
print(toc-tic)

tic = timeit.default_timer()
eNet_multi_0p01 = MultiTaskElasticNetCV(l1_ratio=0.01, max_iter=10000, tol=0.000001, cv=50, n_jobs=-1, selection='random').fit(fluxes_scaled, multi_log2)
toc = timeit.default_timer()
print(toc-tic)

#%% Extract reactions
eNet_rxns_amp_0p01 = rxns[np.nonzero(eNet_amp_0p01.coef_)[0]]
eNet_rxns_cip_0p01 = rxns[np.nonzero(eNet_cip_0p01.coef_)[0]]
eNet_rxns_gent_0p01 = rxns[np.nonzero(eNet_gent_0p01.coef_)[0]]

eNet_rxns_multi_0p01 = rxns[np.nonzero(eNet_multi_0p01.coef_)[1]]
eNet_rxns_multi_0p01 = eNet_rxns_multi_0p01[0:int(len(eNet_rxns_multi_0p01)/3)]

rxns = fluxData.Reactions.values
eNet_amp_SS = []
print('\nAMP reactions: ')
for idx in range(len(eNet_rxns_amp_0p01)):
    rxn = eNet_rxns_amp_0p01[idx]
    eNet_amp_SS.append(subsystems[rxn])
    print(rxn, ':', subsystems[rxn])

eNet_cip_SS = []
print('\nCIP reactions: ')
for idx in range(len(eNet_rxns_cip_0p01)):
    rxn = eNet_rxns_cip_0p01[idx]
    eNet_cip_SS.append(subsystems[rxn])
    print(rxn, ':', subsystems[rxn])

eNet_gent_SS = []
print('\nGENT reactions: ')
for idx in range(len(eNet_rxns_gent_0p01)):
    rxn = eNet_rxns_gent_0p01[idx]
    eNet_gent_SS.append(subsystems[rxn])
    print(rxn, ':', subsystems[rxn])

# Extract coefficients
eNet_coef_amp_0p01 = eNet_amp_0p01.coef_[np.nonzero(eNet_amp_0p01.coef_)[0]]
eNet_coef_cip_0p01 = eNet_cip_0p01.coef_[np.nonzero(eNet_cip_0p01.coef_)[0]]
eNet_coef_gent_0p01 = eNet_gent_0p01.coef_[np.nonzero(eNet_gent_0p01.coef_)[0]]

eNet_coef_multi_0p01 = eNet_multi_0p01.coef_

#%% Filter coefficients
eNet_std_multi_0p01 = np.std(eNet_coef_multi_0p01,1)

amp_thresh = eNet_std_multi_0p01[0]/5
cip_thresh = eNet_std_multi_0p01[1]/5
gent_thresh = eNet_std_multi_0p01[2]/5

amp_rxns = []
amp_coef = []
cip_rxns = []
cip_coef = []
gent_rxns = []
gent_coef = []

for idx in range(len(eNet_rxns_multi_0p01)):
    if abs(eNet_coef_multi_0p01[0,idx]) >= amp_thresh:
        amp_rxns.append(eNet_rxns_multi_0p01[idx])
        amp_coef.append(eNet_coef_multi_0p01[0,idx])
    if abs(eNet_coef_multi_0p01[1,idx]) >= cip_thresh:
        cip_rxns.append(eNet_rxns_multi_0p01[idx])
        cip_coef.append(eNet_coef_multi_0p01[1,idx])
    if abs(eNet_coef_multi_0p01[2,idx]) >= gent_thresh:
        gent_rxns.append(eNet_rxns_multi_0p01[idx])
        gent_coef.append(eNet_coef_multi_0p01[2,idx])

