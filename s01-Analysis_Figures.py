# ---------------------------------------------------------------------------- #
#               Code for the Paper Analysis and Figure Generation              #
# ---------------------------------------------------------------------------- #

# ----------------------- Section 1: Import the modules ---------------------- #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors
import os
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

%matplotlib

# ---------------------------------- Metrics --------------------------------- #

#%% Metrics
# NNSE Function
def nnse(sim, obs):
    num = 0
    den = 0
    obs_mean = np.mean(obs)
    for i in range(len(obs)):
        num = num + (obs[i] - sim[i])**2
        den = den + (obs[i] - obs_mean)**2
    return 1/(2-(1 - ((num)/(den))))


# coefficeint of determination
def rsqr(sim, obs):
    num = 0
    den = 0
    den1 = 0
    den2 = 0
    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    for i in range(len(obs)):
        num = num + ((sim[i] - sim_mean)*( obs[i] - obs_mean))
        den1 = den1 + ((sim[i] - sim_mean)**2)
        den2 = den2 + (( obs[i] - obs_mean)**2)
    den = np.sqrt(den1*den2)
    rsqrc = num**2/den**2
    if (np.isnan(rsqrc)):
        return(-0.1)
    else:
        return(rsqrc)


# ----------------------- Import Data related to Rivers ---------------------- #

""" 
Make the regional Map data based on these map boundary before perfoming the analysis. 
CaMa-FLood simulation should be pefromed using this map boundary for year 2008,
for with and without bifurcation scheme 
at various resolution of 15-arcmin, 6-arcmin, 3-arcmin, and 1-arcmin

"""

## Boundary of the basin
W = -90.000
E = -40.000
N =  10.000
S = -25.000

csize01 = 1/60
csize03 = 1/20
csize06 = 1/10
csize15 = 1/4

# --------------------------- define working folder -------------------------- #
#please define the root workign folder directory
work_folder = "/..."

# ---------------------------------------------------------------------------- #

## LOCATION (row and column)
MRIVLOC15 = pd.read_csv(work_folder+'/analysis/MRIVLOC15.csv')
MRIVLOC06 = pd.read_csv(work_folder+'/analysis/MRIVLOC06.csv')
MRIVLOC03 = pd.read_csv(work_folder+'/analysis/MRIVLOC03.csv')
MRIVLOC01 = pd.read_csv(work_folder+'/analysis/MRIVLOC01.csv')

MRIVLOC15 = np.array(MRIVLOC15).astype("int32")
MRIVLOC06 = np.array(MRIVLOC06).astype("int32")
MRIVLOC03 = np.array(MRIVLOC03).astype("int32")
MRIVLOC01 = np.array(MRIVLOC01).astype("int32")

## LEN (length)
MRIVLEN15 = np.array(pd.read_csv(work_folder+'/analysis/MRIVLEN15.csv'))
MRIVLEN06 = np.array(pd.read_csv(work_folder+'/analysis/MRIVLEN06.csv'))
MRIVLEN03 = np.array(pd.read_csv(work_folder+'/analysis/MRIVLEN03.csv'))
MRIVLEN01 = np.array(pd.read_csv(work_folder+'/analysis/MRIVLEN01.csv'))


## For the 15 min data
BSN15 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/15_amz/basin.bin",dtype=np.int32).reshape(np.int32((N-S)/csize15),np.int32((E-W)/csize15))
RHGT15 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/15_amz/rivhgt.bin",dtype=np.float32).reshape(np.int32((N-S)/csize15),np.int32((E-W)/csize15))
NXTXY15 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/15_amz/nextxy.bin",dtype=np.int32).reshape(-1,np.int32((N-S)/csize15),np.int32((E-W)/csize15))
UPARA15 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/15_amz/uparea.bin",dtype=np.float32).reshape(np.int32((N-S)/csize15),np.int32((E-W)/csize15))
DXY15 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/15_amz/downxy.bin",dtype=np.int32).reshape(-1,np.int32((N-S)/csize15),np.int32((E-W)/csize15))
LONLAT15 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/15_amz/lonlat.bin",dtype=np.float32).reshape(-1,np.int32((N-S)/csize15),np.int32((E-W)/csize15))
NXTDST15 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/15_amz/nxtdst.bin",dtype=np.float32).reshape(np.int32((N-S)/csize15),np.int32((E-W)/csize15))
RIVLEN15 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/15_amz/rivlen.bin",dtype=np.float32).reshape(np.int32((N-S)/csize15),np.int32((E-W)/csize15))
ELEVTN15 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/15_amz/elevtn.bin",dtype=np.float32).reshape(np.int32((N-S)/csize15),np.int32((E-W)/csize15))
RIVWTH15 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/15_amz/rivwth_gwdlr.bin",dtype=np.float32).reshape(np.int32((N-S)/csize15),np.int32((E-W)/csize15))

## For the 06 min data
BSN06 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/06_amz/basin.bin",dtype=np.int32).reshape(np.int32((N-S)/csize06),np.int32((E-W)/csize06))
RHGT06 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/06_amz/rivhgt.bin",dtype=np.float32).reshape(np.int32((N-S)/csize06),np.int32((E-W)/csize06))
NXTXY06 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/06_amz/nextxy.bin",dtype=np.int32).reshape(-1,np.int32((N-S)/csize06),np.int32((E-W)/csize06))
UPARA06 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/06_amz/uparea.bin",dtype=np.float32).reshape(np.int32((N-S)/csize06),np.int32((E-W)/csize06))
DXY06 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/06_amz/downxy.bin",dtype=np.int32).reshape(-1,np.int32((N-S)/csize06),np.int32((E-W)/csize06))
LONLAT06 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/06_amz/lonlat.bin",dtype=np.float32).reshape(-1,np.int32((N-S)/csize06),np.int32((E-W)/csize06))
NXTDST06 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/06_amz/nxtdst.bin",dtype=np.float32).reshape(np.int32((N-S)/csize06),np.int32((E-W)/csize06))
RIVLEN06 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/06_amz/rivlen.bin",dtype=np.float32).reshape(np.int32((N-S)/csize06),np.int32((E-W)/csize06))
ELEVTN06 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/06_amz/elevtn.bin",dtype=np.float32).reshape(np.int32((N-S)/csize06),np.int32((E-W)/csize06))
RIVWTH06 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/06_amz/rivwth_gwdlr.bin",dtype=np.float32).reshape(np.int32((N-S)/csize06),np.int32((E-W)/csize06))

## For the 03 min data
BSN03 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/03_amz/basin.bin",dtype=np.int32).reshape(np.int32((N-S)/csize03),np.int32((E-W)/csize03))
RHGT03 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/03_amz/rivhgt.bin",dtype=np.float32).reshape(np.int32((N-S)/csize03),np.int32((E-W)/csize03))
NXTXY03 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/03_amz/nextxy.bin",dtype=np.int32).reshape(-1,np.int32((N-S)/csize03),np.int32((E-W)/csize03))
UPARA03 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/03_amz/uparea.bin",dtype=np.float32).reshape(np.int32((N-S)/csize03),np.int32((E-W)/csize03))
DXY03 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/03_amz/downxy.bin",dtype=np.int32).reshape(-1,np.int32((N-S)/csize03),np.int32((E-W)/csize03))
LONLAT03 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/03_amz/lonlat.bin",dtype=np.float32).reshape(-1,np.int32((N-S)/csize03),np.int32((E-W)/csize03))
NXTDST03 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/03_amz/nxtdst.bin",dtype=np.float32).reshape(np.int32((N-S)/csize03),np.int32((E-W)/csize03))
RIVLEN03 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/03_amz/rivlen.bin",dtype=np.float32).reshape(np.int32((N-S)/csize03),np.int32((E-W)/csize03))
ELEVTN03 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/03_amz/elevtn.bin",dtype=np.float32).reshape(np.int32((N-S)/csize03),np.int32((E-W)/csize03))
RIVWTH03 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/03_amz/rivwth_gwdlr.bin",dtype=np.float32).reshape(np.int32((N-S)/csize03),np.int32((E-W)/csize03))

## For the 01 min data
BSN01 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/01_amz/basin.bin",dtype=np.int32).reshape(np.int32((N-S)/csize01),np.int32((E-W)/csize01))
RHGT01 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/01_amz/rivhgt.bin",dtype=np.float32).reshape(np.int32((N-S)/csize01),np.int32((E-W)/csize01))
NXTXY01 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/01_amz/nextxy.bin",dtype=np.int32).reshape(-1,np.int32((N-S)/csize01),np.int32((E-W)/csize01))
UPARA01 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/01_amz/uparea.bin",dtype=np.float32).reshape(np.int32((N-S)/csize01),np.int32((E-W)/csize01))
DXY01 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/01_amz/downxy.bin",dtype=np.int32).reshape(-1,np.int32((N-S)/csize01),np.int32((E-W)/csize01))
LONLAT01 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/01_amz/lonlat.bin",dtype=np.float32).reshape(-1,np.int32((N-S)/csize01),np.int32((E-W)/csize01))
NXTDST01 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/01_amz/nxtdst.bin",dtype=np.float32).reshape(np.int32((N-S)/csize01),np.int32((E-W)/csize01))
RIVLEN01 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/01_amz/rivlen.bin",dtype=np.float32).reshape(np.int32((N-S)/csize01),np.int32((E-W)/csize01))
ELEVTN01 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/01_amz/elevtn.bin",dtype=np.float32).reshape(np.int32((N-S)/csize01),np.int32((E-W)/csize01))
RIVWTH01 = np.fromfile(work_folder+"/CaMa-Flood_v4/map/01_amz/rivwth_gwdlr.bin",dtype=np.float32).reshape(np.int32((N-S)/csize01),np.int32((E-W)/csize01))


#%% river channel pixels
loc_15 = pd.read_csv(work_folder+"/analysis/RIVNET15.csv")
LAT_LON_15 = np.column_stack((loc_15["LAT"],loc_15["LON"]))
ID_15 = loc_15["ID"].astype(np.int32)
loc_15 = np.column_stack((loc_15["ROW"].astype(np.int32),loc_15["COL"].astype(int)))
loc_06 = pd.read_csv(work_folder+"/analysis/RIVNET06.csv")
loc_06 = np.column_stack((loc_06["ROW"].astype(np.int32),loc_06["COL"].astype(int)))
loc_03 = pd.read_csv(work_folder+"/analysis/RIVNET03.csv")
loc_03 = np.column_stack((loc_03["ROW"].astype(np.int32),loc_03["COL"].astype(int)))
loc_01 = pd.read_csv(work_folder+"/analysis/RIVNET01.csv")
loc_01 = np.column_stack((loc_01["ROW"].astype(np.int32),loc_01["COL"].astype(int)))


# ---------------------------------------------------------------------------- #
# %% Import Discharge for 2008
AMZDIS01 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic/rivout2008.bin",dtype=np.float32).reshape(-1,2100,3000)+ np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic/fldout2008.bin",dtype=np.float32).reshape(-1,2100,3000)
AMZDIS03 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic/rivout2008.bin",dtype=np.float32).reshape(-1,700,1000)+np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic/fldout2008.bin",dtype=np.float32).reshape(-1,700,1000)
AMZDIS06 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic/rivout2008.bin",dtype=np.float32).reshape(-1,350,500)+np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic/fldout2008.bin",dtype=np.float32).reshape(-1,350,500)
AMZDIS15 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic/rivout2008.bin",dtype=np.float32).reshape(-1,140,200)+np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic/fldout2008.bin",dtype=np.float32).reshape(-1,140,200)


#%% NNSE , RMSE, R2 for discharge
## NNSE Calculation for discharge
nnse_15_06_Qamz = []
nnse_15_03_Qamz = []
nnse_15_01_Qamz = []
nnse_06_03_Qamz = []
nnse_06_01_Qamz = []
nnse_03_01_Qamz = []

for i in range(len(loc_15)):
    nnse_15_06_Qamz.append(np.round(nnse(AMZDIS15[:,loc_15[i,0],loc_15[i,1]],AMZDIS06[:,loc_06[i,0],loc_06[i,1]]),13))
    nnse_15_03_Qamz.append(np.round(nnse(AMZDIS15[:,loc_15[i,0],loc_15[i,1]],AMZDIS03[:,loc_03[i,0],loc_03[i,1]]),13))
    nnse_15_01_Qamz.append(np.round(nnse(AMZDIS15[:,loc_15[i,0],loc_15[i,1]],AMZDIS01[:,loc_01[i,0],loc_01[i,1]]),13))
    nnse_06_03_Qamz.append(np.round(nnse(AMZDIS06[:,loc_06[i,0],loc_06[i,1]],AMZDIS03[:,loc_03[i,0],loc_03[i,1]]),13))
    nnse_06_01_Qamz.append(np.round(nnse(AMZDIS06[:,loc_06[i,0],loc_06[i,1]],AMZDIS01[:,loc_01[i,0],loc_01[i,1]]),13))
    nnse_03_01_Qamz.append(np.round(nnse(AMZDIS03[:,loc_03[i,0],loc_03[i,1]],AMZDIS01[:,loc_01[i,0],loc_01[i,1]]),13))

## R2 Calculation for Discharge
rsqr_15_06_Qamz = []
rsqr_15_03_Qamz = []
rsqr_15_01_Qamz = []
rsqr_06_03_Qamz = []
rsqr_06_01_Qamz = []
rsqr_03_01_Qamz = []

for i in range(len(loc_15)):
    rsqr_15_06_Qamz.append(rsqr(AMZDIS15[:,loc_15[i,0],loc_15[i,1]],AMZDIS06[:,loc_06[i,0],loc_06[i,1]]))
    rsqr_15_03_Qamz.append(rsqr(AMZDIS15[:,loc_15[i,0],loc_15[i,1]],AMZDIS03[:,loc_03[i,0],loc_03[i,1]]))
    rsqr_15_01_Qamz.append(rsqr(AMZDIS15[:,loc_15[i,0],loc_15[i,1]],AMZDIS01[:,loc_01[i,0],loc_01[i,1]]))
    rsqr_06_03_Qamz.append(rsqr(AMZDIS06[:,loc_06[i,0],loc_06[i,1]],AMZDIS03[:,loc_03[i,0],loc_03[i,1]]))
    rsqr_06_01_Qamz.append(rsqr(AMZDIS06[:,loc_06[i,0],loc_06[i,1]],AMZDIS01[:,loc_01[i,0],loc_01[i,1]]))
    rsqr_03_01_Qamz.append(rsqr(AMZDIS03[:,loc_03[i,0],loc_03[i,1]],AMZDIS01[:,loc_01[i,0],loc_01[i,1]]))

# Print 10th and 20th percentile
qmetric_01_03 = [np.round(np.quantile(nnse_03_01_Qamz,[0.10,0.20]),2), np.round(np.quantile(rsqr_03_01_Qamz,[0.10,0.20]),2)]
qmetric_01_06 = [np.round(np.quantile(nnse_06_01_Qamz,[0.10,0.20]),2), np.round(np.quantile(rsqr_06_01_Qamz,[0.10,0.20]),2)]
qmetric_01_15 = [np.round(np.quantile(nnse_15_01_Qamz,[0.10,0.20]),2),np.round(np.quantile(rsqr_15_01_Qamz,[0.10,0.20]),2)]
qmetric_03_06 = [np.round(np.quantile(nnse_06_03_Qamz,[0.10,0.20]),2), np.round(np.quantile(rsqr_06_03_Qamz,[0.10,0.20]),2)]
qmetric_03_15 = [np.round(np.quantile(nnse_15_03_Qamz,[0.10,0.20]),2),np.round(np.quantile(rsqr_15_03_Qamz,[0.10,0.20]),2)]
qmetric_06_15 = [np.round(np.quantile(nnse_15_06_Qamz,[0.10,0.20]),2),np.round(np.quantile(rsqr_15_06_Qamz,[0.10,0.20]),2)]
print(qmetric_01_03)
print(qmetric_01_06)
print(qmetric_01_15)
print(qmetric_03_06)
print(qmetric_03_15)
print(qmetric_06_15)


# %% Import River Water Depth for 2008
AMZDPH01 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic/rivdph2008.bin",dtype=np.float32).reshape(-1,2100,3000)
AMZDPH03 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic/rivdph2008.bin",dtype=np.float32).reshape(-1,700,1000)
AMZDPH06 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic/rivdph2008.bin",dtype=np.float32).reshape(-1,350,500)
AMZDPH15 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic/rivdph2008.bin",dtype=np.float32).reshape(-1,140,200)
## NNSE, RMSE, R2 for RWD
## NNSE Calculation for RWD
nnse_15_06_Damz = []
nnse_15_03_Damz = []
nnse_15_01_Damz = []
nnse_06_03_Damz = []
nnse_06_01_Damz = []
nnse_03_01_Damz = []

for i in range(len(loc_15)):
    nnse_15_06_Damz.append(np.round(nnse(AMZDPH15[:,loc_15[i,0],loc_15[i,1]],AMZDPH06[:,loc_06[i,0],loc_06[i,1]]),13))
    nnse_15_03_Damz.append(np.round(nnse(AMZDPH15[:,loc_15[i,0],loc_15[i,1]],AMZDPH03[:,loc_03[i,0],loc_03[i,1]]),13))
    nnse_15_01_Damz.append(np.round(nnse(AMZDPH15[:,loc_15[i,0],loc_15[i,1]],AMZDPH01[:,loc_01[i,0],loc_01[i,1]]),13))
    nnse_06_03_Damz.append(np.round(nnse(AMZDPH06[:,loc_06[i,0],loc_06[i,1]],AMZDPH03[:,loc_03[i,0],loc_03[i,1]]),13))
    nnse_06_01_Damz.append(np.round(nnse(AMZDPH06[:,loc_06[i,0],loc_06[i,1]],AMZDPH01[:,loc_01[i,0],loc_01[i,1]]),13))
    nnse_03_01_Damz.append(np.round(nnse(AMZDPH03[:,loc_03[i,0],loc_03[i,1]],AMZDPH01[:,loc_01[i,0],loc_01[i,1]]),13))

## R2 Calculation for RWD
rsqr_15_06_Damz = []
rsqr_15_03_Damz = []
rsqr_15_01_Damz = []
rsqr_06_03_Damz = []
rsqr_06_01_Damz = []
rsqr_03_01_Damz = []

for i in range(len(loc_15)):
    rsqr_15_06_Damz.append(rsqr(AMZDPH15[:,loc_15[i,0],loc_15[i,1]],AMZDPH06[:,loc_06[i,0],loc_06[i,1]]))
    rsqr_15_03_Damz.append(rsqr(AMZDPH15[:,loc_15[i,0],loc_15[i,1]],AMZDPH03[:,loc_03[i,0],loc_03[i,1]]))
    rsqr_15_01_Damz.append(rsqr(AMZDPH15[:,loc_15[i,0],loc_15[i,1]],AMZDPH01[:,loc_01[i,0],loc_01[i,1]]))
    rsqr_06_03_Damz.append(rsqr(AMZDPH06[:,loc_06[i,0],loc_06[i,1]],AMZDPH03[:,loc_03[i,0],loc_03[i,1]]))
    rsqr_06_01_Damz.append(rsqr(AMZDPH06[:,loc_06[i,0],loc_06[i,1]],AMZDPH01[:,loc_01[i,0],loc_01[i,1]]))
    rsqr_03_01_Damz.append(rsqr(AMZDPH03[:,loc_03[i,0],loc_03[i,1]],AMZDPH01[:,loc_01[i,0],loc_01[i,1]]))

## 10th and 20th percentile 
dmetric_01_03 = [np.round(np.quantile(nnse_03_01_Damz,[0.10, 0.20]),2), np.round(np.quantile(rsqr_03_01_Damz,[0.10, 0.20]),2)]
dmetric_01_06 = [np.round(np.quantile(nnse_06_01_Damz,[0.10, 0.20]),2), np.round(np.quantile(rsqr_06_01_Damz,[0.10, 0.20]),2)]
dmetric_01_15 = [np.round(np.quantile(nnse_15_01_Damz,[0.10, 0.20]),2),np.round(np.quantile(rsqr_15_01_Damz,[0.10, 0.20]),2)]
dmetric_03_06 = [np.round(np.quantile(nnse_06_03_Damz,[0.10, 0.20]),2), np.round(np.quantile(rsqr_06_03_Damz,[0.10, 0.20]),2)]
dmetric_03_15 = [np.round(np.quantile(nnse_15_03_Damz,[0.10, 0.20]),2),np.round(np.quantile(rsqr_15_03_Damz,[0.10, 0.20]),2)]
dmetric_06_15 = [np.round(np.quantile(nnse_15_06_Damz,[0.10, 0.20]),2),np.round(np.quantile(rsqr_15_06_Damz,[0.10, 0.20]),2)]
print(dmetric_01_03)
print(dmetric_01_06)
print(dmetric_01_15)
print(dmetric_03_06)
print(dmetric_03_15)
print(dmetric_06_15)


# %% Import Water surface elevation for 2008
AMZWSE01 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic/sfcelv2008.bin",dtype=np.float32).reshape(-1,2100,3000)
AMZWSE03 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic/sfcelv2008.bin",dtype=np.float32).reshape(-1,700,1000)
AMZWSE06 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic/sfcelv2008.bin",dtype=np.float32).reshape(-1,350,500)
AMZWSE15 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic/sfcelv2008.bin",dtype=np.float32).reshape(-1,140,200)

## Main River Discharge
MDIS01 = AMZDIS01[:,MRIVLOC01[:,0],MRIVLOC01[:,1]]
MDIS03 = AMZDIS03[:,MRIVLOC03[:,0],MRIVLOC03[:,1]]
MDIS06 = AMZDIS06[:,MRIVLOC06[:,0],MRIVLOC06[:,1]]
MDIS15 = AMZDIS15[:,MRIVLOC15[:,0],MRIVLOC15[:,1]]
## Main River Water Depth
MDPH01 = AMZDPH01[:,MRIVLOC01[:,0],MRIVLOC01[:,1]]
MDPH03 = AMZDPH03[:,MRIVLOC03[:,0],MRIVLOC03[:,1]]
MDPH06 = AMZDPH06[:,MRIVLOC06[:,0],MRIVLOC06[:,1]]
MDPH15 = AMZDPH15[:,MRIVLOC15[:,0],MRIVLOC15[:,1]]


#%% Lat lon of the basin data
# 15 min
FLOC = np.argwhere((BSN15==1)&(NXTXY15[0]!=-9))  ## location of pixel with highest value of river depth (river mouth)
FLON15 = LONLAT15[0,FLOC[:,0],FLOC[:,1]]
FLAT15 = LONLAT15[1,FLOC[:,0],FLOC[:,1]]
TLOCROW15 = NXTXY15[1,FLOC[:,0],FLOC[:,1]]-1
TLOCCOL15 = NXTXY15[0,FLOC[:,0],FLOC[:,1]]-1
TLON15 = LONLAT15[0,TLOCROW15,TLOCCOL15]
TLAT15 = LONLAT15[1,TLOCROW15,TLOCCOL15]
RIVLATLON15 = np.column_stack((FLON15,FLAT15,TLON15,TLAT15))


# 06 min
FLOC = np.argwhere((BSN06==1)&(NXTXY06[0]!=-9))  ## location of pixel with highest value of river depth (river mouth)
FLON06 = LONLAT06[0,FLOC[:,0],FLOC[:,1]]
FLAT06 = LONLAT06[1,FLOC[:,0],FLOC[:,1]]
TLOCROW06 = NXTXY06[1,FLOC[:,0],FLOC[:,1]]-1
TLOCCOL06 = NXTXY06[0,FLOC[:,0],FLOC[:,1]]-1
TLON06 = LONLAT06[0,TLOCROW06,TLOCCOL06]
TLAT06 = LONLAT06[1,TLOCROW06,TLOCCOL06]
RIVLATLON06 = np.column_stack((FLON06,FLAT06,TLON06,TLAT06))


# 03 min
FLOC = np.argwhere((BSN03==1)&(NXTXY03[0]!=-9))  ## location of pixel with highest value of river depth (river loc_15mouth)
FLON03 = LONLAT03[0,FLOC[:,0],FLOC[:,1]]
FLAT03 = LONLAT03[1,FLOC[:,0],FLOC[:,1]]
TLOCROW03 = NXTXY03[1,FLOC[:,0],FLOC[:,1]]-1
TLOCCOL03 = NXTXY03[0,FLOC[:,0],FLOC[:,1]]-1
TLON03 = LONLAT03[0,TLOCROW03,TLOCCOL03]
TLAT03 = LONLAT03[1,TLOCROW03,TLOCCOL03]
RIVLATLON03 = np.column_stack((FLON03,FLAT03,TLON03,TLAT03))


# 01 min
FLOC = np.argwhere((BSN01==1)&(NXTXY01[0]!=-9))  ## location of pixel with highest value of river depth (river mouth)
FLON01 = LONLAT01[0,FLOC[:,0],FLOC[:,1]]
FLAT01 = LONLAT01[1,FLOC[:,0],FLOC[:,1]]
TLOCROW01 = NXTXY01[1,FLOC[:,0],FLOC[:,1]]-1
TLOCCOL01 = NXTXY01[0,FLOC[:,0],FLOC[:,1]]-1
TLON01 = LONLAT01[0,TLOCROW01,TLOCCOL01]
TLAT01 = LONLAT01[1,TLOCROW01,TLOCCOL01]
RIVLATLON01 = np.column_stack((FLON01,FLAT01,TLON01,TLAT01))



#%% make the data for the lat lon of river pixels 
riv15lon = LONLAT15[0,loc_15[:,0],loc_15[:,1]]
riv15lat = LONLAT15[1,loc_15[:,0],loc_15[:,1]]

riv06lon = LONLAT06[0,loc_06[:,0],loc_06[:,1]]
riv06lat = LONLAT06[1,loc_06[:,0],loc_06[:,1]]

riv03lon = LONLAT03[0,loc_03[:,0],loc_03[:,1]]
riv03lat = LONLAT03[1,loc_03[:,0],loc_03[:,1]]

riv01lon = LONLAT01[0,loc_01[:,0],loc_01[:,1]]
riv01lat = LONLAT01[1,loc_01[:,0],loc_01[:,1]]


# Linewith of rivers

lwidth_06 = (np.log10(RIVWTH06[TLOCROW06,TLOCCOL06]))/5
lwidth_03 = (np.log10(RIVWTH03[TLOCROW03,TLOCCOL03]))/5
lwidth_01 = (np.log10(RIVWTH01[TLOCROW01,TLOCCOL01]))/5




# ------------------------------ NNSE and R2 violin plot (Discharge and River water depth)------------------------------------------#

# %% update to violin plot with cut = 0 and wihout inner plot mark maybe errorbars or lines for 80th and 90th percentile 

#%% Dataframe and plot using seaborn 
# violinplot: distributio NNSE values over the Amzaon
NNSEQAMZ = pd.DataFrame(np.column_stack((nnse_03_01_Qamz,nnse_06_01_Qamz,nnse_15_01_Qamz,nnse_06_03_Qamz,nnse_15_03_Qamz,nnse_15_06_Qamz)),
columns=['1 & 3 min','1 & 6 min','1 & 15 min','3 & 6 min','3 & 15 min','6 & 15 min'])
fig, ax= plt.subplots(2,2,figsize=(28,20))
vplot1 = sns.violinplot(data=NNSEQAMZ,  palette=['green','blue','red','green','blue','green'],ax=ax[0,0],cut=0, inner=None)
clrs = ['green','blue','red','green','blue','green']
qtls = np.round(np.array(NNSEQAMZ.quantile(q=[0.90,0.80])),3)
alpha=[0.7,0.5,0.3,0.5,0.3,0.3]
for violin, alpha, clr in zip(ax[0,0].collections, alpha,clrs):
    violin.set_alpha(alpha)
    violin.set_edgecolor(clr)
    violin.set_linewidth(1)

for i, column in enumerate(NNSEQAMZ.columns):
    #get the vertices
    x_coords = vplot1.collections[i].get_paths()[0].vertices[:, 0]
    xmin, xmax = x_coords.min(), x_coords.max()
    # Calculate the 90th percentile
    p10 = NNSEQAMZ[column].quantile(0.10)
    p20 = NNSEQAMZ[column].quantile(0.20)
    
    ax[0, 0].scatter(x=(xmin+xmax)/2,y=p10,s=30,marker="*",c='k',label="10$^{th}$ percentile")
    ax[0, 0].scatter(x=(xmin+xmax)/2,y=p20,s=30,marker="+",c='k',label="20$^{th}$ percentile")


# sns.stripplot(data=NNSEQAMZ, alpha=0.5,edgecolor='k',palette=['green','blue','red','green','blue','green'],size=1,ax=ax[0,0])
ax[0,0].set_xlabel("Spatial resolution",fontsize=22)
ax[0,0].set_ylabel("NNSE",fontsize=22)
ax[0,0].set_title("(a) NNSE comparison for discharge", fontsize=22, weight="bold",loc='left')
ax[0,0].grid(color='k', linestyle='--', linewidth=0.2, which='both')
ax[0,0].set_yticks(np.arange(0,1.1,0.1))
ax[0,0].tick_params(axis='both', which='major', labelsize=18)
lines_labels = [ax[0,0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax[0,0].legend(lines[0:2], labels[0:2],loc='lower right', markerscale=3, fontsize=18) 

# violinplot: distribution of R2 values over amazon
RSQRQAMZ = pd.DataFrame(np.column_stack((rsqr_03_01_Qamz,rsqr_06_01_Qamz,rsqr_15_01_Qamz,rsqr_06_03_Qamz,rsqr_15_03_Qamz,rsqr_15_06_Qamz)),
columns=['1 & 3 min','1 & 6 min','1 & 15 min','3 & 6 min','3 & 15 min','6 & 15 min'])
# fig, ax = plt.subplots(figsize=(16,9))
vplot2 = sns.violinplot(data=RSQRQAMZ,  palette=['green','blue','red','green','blue','green'],ax=ax[1,0],cut=0, inner=None)
clrs = ['green','blue','red','green','blue','green']
alpha=[0.7,0.5,0.3,0.5,0.3,0.3]
for violin, alpha,clr in zip(ax[1,0].collections, alpha,clrs):
    violin.set_alpha(alpha)
    violin.set_edgecolor(clr)
    violin.set_linewidth(1)

for i, column in enumerate(RSQRQAMZ.columns):
    #get the vertices
    x_coords = vplot2.collections[i].get_paths()[0].vertices[:, 0]
    xmin, xmax = x_coords.min(), x_coords.max()
    # Calculate the 90th percentile
    p10 = RSQRQAMZ[column].quantile(0.10)
    p20 = RSQRQAMZ[column].quantile(0.20)
    
    ax[1, 0].scatter(x=(xmin+xmax)/2,y=p10,s=30,marker="*",c='k',label="10$^{th}$ percentile")
    ax[1, 0].scatter(x=(xmin+xmax)/2,y=p20,s=30,marker="+",c='k',label="20$^{th}$ percentile")


# sns.stripplot(data=RSQRQAMZ, alpha=0.5,edgecolor='k',palette=['green','blue','red','green','blue','green'],size=1,ax=ax[1,0])
ax[1,0].set_xlabel("Spatial resolution",fontsize=22)
ax[1,0].set_ylabel("R\N{SUPERSCRIPT TWO}",fontsize=22)
ax[1,0].set_title("(b) R\N{SUPERSCRIPT TWO} comparison for discharge", fontsize=22, weight="bold", loc='left')
ax[1,0].grid(color='k', linestyle='--', linewidth=0.2, which='both')
ax[1,0].set_yticks(np.arange(0,1.1,0.1))
ax[1,0].tick_params(axis='both', which='major', labelsize=18)
lines_labels = [ax[1,0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax[1,0].legend(lines[0:2], labels[0:2],loc='lower right', markerscale=3, fontsize=18)  

# violinplot: distributio NNSE values over the Amzaon
NNSEDAMZ = pd.DataFrame(np.column_stack((nnse_03_01_Damz,nnse_06_01_Damz,nnse_15_01_Damz,nnse_06_03_Damz,nnse_15_03_Damz,nnse_15_06_Damz)),
columns=['1 & 3 min','1 & 6 min','1 & 15 min','3 & 6 min','3 & 15 min','6 & 15 min'])
vplot1 = sns.violinplot(data=NNSEDAMZ, palette=['green','blue','red','green','blue','green'],ax=ax[0,1],cut=0, inner=None)
clrs = ['green','blue','red','green','blue','green']
alpha=[0.7,0.5,0.3,0.5,0.3,0.3]
for violin, alpha,clr in zip(ax[0,1].collections, alpha,clrs):
    violin.set_alpha(alpha)
    violin.set_edgecolor(clr)
    violin.set_linewidth(1)

for i, column in enumerate(NNSEDAMZ.columns):
    #get the vertices
    x_coords = vplot1.collections[i].get_paths()[0].vertices[:, 0]
    xmin, xmax = x_coords.min(), x_coords.max()
    # Calculate the 90th percentile
    p10 = NNSEDAMZ[column].quantile(0.10)
    p20 = NNSEDAMZ[column].quantile(0.20)
    
    ax[0,1].scatter(x=(xmin+xmax)/2,y=p10,s=30,marker="*",c='k',label="10$^{th}$ percentile")
    ax[0,1].scatter(x=(xmin+xmax)/2,y=p20,s=30,marker="+",c='k',label="20$^{th}$ percentile")

# sns.stripplot(data=NNSEDAMZ, alpha=0.5,edgecolor='k',palette=['green','blue','red','green','blue','green'],size=1,ax=ax[0,1])
ax[0,1].set_xlabel("Spatial resolution",fontsize=22)
ax[0,1].set_ylabel("NNSE",fontsize=22)
ax[0,1].set_title("(c) NNSE comparison for water depth", fontsize=22, weight="bold",loc='left')
ax[0,1].grid(color='k', linestyle='--', linewidth=0.2, which='both')
ax[0,1].set_yticks(np.arange(0,1.1,0.1))
ax[0,1].tick_params(axis='both', which='major', labelsize=18)
lines_labels = [ax[0,1].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax[0,1].legend(lines[0:2], labels[0:2],loc='lower right', markerscale=3, fontsize=18)  

# violinplot: distribution of R2 values over amazon
RSQRDAMZ = pd.DataFrame(np.column_stack((rsqr_03_01_Damz,rsqr_06_01_Damz,rsqr_15_01_Damz,rsqr_06_03_Damz,rsqr_15_03_Damz,rsqr_15_06_Damz)),
columns=['1 & 3 min','1 & 6 min','1 & 15 min','3 & 6 min','3 & 15 min','6 & 15 min'])
vplot2 = sns.violinplot(data=RSQRDAMZ,  palette=['green','blue','red','green','blue','green'],ax=ax[1,1],cut=0, inner=None)
clrs = ['green','blue','red','green','blue','green']
alpha=[0.7,0.5,0.3,0.5,0.3,0.3]
for violin, alpha,clr in zip(ax[1,1].collections, alpha,clrs):
    violin.set_alpha(alpha)
    violin.set_edgecolor(clr)
    violin.set_linewidth(1)

for i, column in enumerate(RSQRDAMZ.columns):
    #get the vertices
    x_coords = vplot2.collections[i].get_paths()[0].vertices[:, 0]
    xmin, xmax = x_coords.min(), x_coords.max()
    # Calculate the 90th percentile
    p10 = RSQRDAMZ[column].quantile(0.10)
    p20 = RSQRDAMZ[column].quantile(0.20)
    
    ax[1,1].scatter(x=(xmin+xmax)/2,y=p10,s=30,marker="*",c='k',label="10$^{th}$ percentile")
    ax[1,1].scatter(x=(xmin+xmax)/2,y=p20,s=30,marker="+",c='k',label="20$^{th}$ percentile")

# sns.stripplot(data=RSQRDAMZ, alpha=0.5,edgecolor='k',palette=['green','blue','red','green','blue','green'],size=1,ax=ax[1,1])
ax[1,1].set_xlabel("Spatial resolution",fontsize=22)
ax[1,1].set_ylabel("R\N{SUPERSCRIPT TWO}",fontsize=22)
ax[1,1].set_title("(d) R\N{SUPERSCRIPT TWO} comparison for water depth", fontsize=22, weight="bold", loc='left')
ax[1,1].grid(color='k', linestyle='--', linewidth=0.2, which='both')
ax[1,1].set_yticks(np.arange(0,1.1,0.1))
ax[1,1].tick_params(axis='both', which='major', labelsize=18)
lines_labels = [ax[1,1].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax[1,1].legend(lines[0:2], labels[0:2],loc='lower right', markerscale=3, fontsize=18)  
plt.subplots_adjust(wspace=0.15)
plt.savefig(work_folder+"/analysis/boxplt_QAMZ_DAMZ_NNSE_R2_v4.jpg",dpi=500,bbox_inches='tight')



########### ------------------------------- Spatial Distrbution of NNSE and R2 --------------------------------------- ################

# # --------------- ## NNSE and R2 spatial distrbution for both Q and RWD only for 1min and 6 min in single Figure (4 panels)-------------------- # #

norm1 = [-9999,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130,140,141,142,143,144,145,146,147,148,149,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,
         300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,
         3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000]

## for discharge 
fig, ax = plt.subplots(2,2,figsize=(18,15))
# m = Basemap(projection='cyl',llcrnrlat=-25,urcrnrlat=10,llcrnrlon=-90,urcrnrlon=-40,resolution='h')
m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawcoastlines(color='lightgray',ax=ax[0,0])
m.drawcountries(color='lightgray',ax=ax[0,0])
m.fillcontinents(color='white', lake_color='#eeeeee',ax=ax[0,0])
m.drawmapboundary(fill_color='#eeeeee',ax=ax[0,0])
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[0,0])
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[0,0])
m.shadedrelief(zorder=1 ,ax=ax[0,0])
# Linewith of rivers
lwidth_15 = (np.log10(RIVWTH15[TLOCROW15,TLOCCOL15]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON15[:,0], RIVLATLON15[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON15[:,2], RIVLATLON15[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[0,0].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_15)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv15lon,riv15lat)
im1 = ax[0,0].scatter(x4,y4,s=14,c=nnse_06_01_Qamz,facecolor=nnse_06_01_Qamz,edgecolor='none',linewidth=0.02,alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar1 = m.colorbar(im1,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[0,0])
cbar1.set_label("NNSE",fontsize=15)
ax[0,0].set_title("(a) NNSE (1min & 6min) for discharge",loc='left',fontsize=16,weight='bold')


# m = Basemap(projection='cyl',llcrnrlat=-25,urcrnrlat=10,llcrnrlon=-90,urcrnrlon=-40,resolution='h')
m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawcoastlines(color='lightgray',ax=ax[1,0])
m.drawcountries(color='lightgray',ax=ax[1,0])
m.fillcontinents(color='white', lake_color='#eeeeee',ax=ax[1,0])
m.drawmapboundary(fill_color='#eeeeee',ax=ax[1,0])
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[1,0] )
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[1,0])
m.shadedrelief(zorder=1 ,ax=ax[1,0])
# Linewith of rivers
lwidth_15 = (np.log10(RIVWTH15[TLOCROW15,TLOCCOL15]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON15[:,0], RIVLATLON15[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON15[:,2], RIVLATLON15[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[1,0].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_15)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv15lon,riv15lat)
im3 = ax[1,0].scatter(x4,y4,s=14,c=rsqr_06_01_Qamz,facecolor=rsqr_06_01_Qamz,edgecolor='none',linewidth=0.02,alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar3 = m.colorbar(im3,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[1,0])
cbar3.set_label("R\N{SUPERSCRIPT TWO}",fontsize=15)
ax[1,0].set_title("(b) R\N{SUPERSCRIPT TWO} (1min & 6min) for discharge",loc='left',fontsize=16,weight='bold')



# m = Basemap(projection='cyl',llcrnrlat=-25,urcrnrlat=10,llcrnrlon=-90,urcrnrlon=-40,resolution='h')
m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawcoastlines(color='lightgray',ax=ax[0,1])
m.drawcountries(color='lightgray',ax=ax[0,1])
m.fillcontinents(color='white', lake_color='#eeeeee',ax=ax[0,1])
m.drawmapboundary(fill_color='#eeeeee',ax=ax[0,1])
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[0,1])
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[0,1])
m.shadedrelief(zorder=1 ,ax=ax[0,1])
# Linewith of rivers
lwidth_15 = (np.log10(RIVWTH15[TLOCROW15,TLOCCOL15]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON15[:,0], RIVLATLON15[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON15[:,2], RIVLATLON15[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[0,1].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_15)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv15lon,riv15lat)
im1 = ax[0,1].scatter(x4,y4,s=14,c=nnse_06_01_Damz,facecolor=nnse_06_01_Damz,edgecolor='none',linewidth=0.02,alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar1 = m.colorbar(im1,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[0,1])
cbar1.set_label("NNSE",fontsize=15)
ax[0,1].set_title("(c) NNSE (1min & 6min) for water depth",loc='left',fontsize=16,weight='bold')


# m = Basemap(projection='cyl',llcrnrlat=-25,urcrnrlat=10,llcrnrlon=-90,urcrnrlon=-40,resolution='h')
m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawcoastlines(color='lightgray',ax=ax[1,1])
m.drawcountries(color='lightgray',ax=ax[1,1])
m.fillcontinents(color='white', lake_color='#eeeeee',ax=ax[1,1])
m.drawmapboundary(fill_color='#eeeeee',ax=ax[1,1])
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[1,1] )
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[1,1])
m.shadedrelief(zorder=1 ,ax=ax[1,1])
# Linewith of rivers
lwidth_15 = (np.log10(RIVWTH15[TLOCROW15,TLOCCOL15]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON15[:,0], RIVLATLON15[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON15[:,2], RIVLATLON15[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[1,1].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_15)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv15lon,riv15lat)
im3 = ax[1,1].scatter(x4,y4,s=14,c=rsqr_06_01_Damz,facecolor=rsqr_06_01_Damz,edgecolor='none',linewidth=0.02,alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar3 = m.colorbar(im3,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[1,1])
cbar3.set_label("R\N{SUPERSCRIPT TWO}",fontsize=15)
ax[1,1].set_title("(d) R\N{SUPERSCRIPT TWO} (1min & 6min) for water depth",loc='left',fontsize=16,weight='bold')

plt.subplots_adjust(hspace=0.2,wspace=0.03)

# show the locations on each point used in the maintext
# Marker 351 → right
for i in range(2):
    for j in range(2):
        ax[i,j].annotate(text="351", xy=(x4[350], y4[350]), xytext=(10, 0),
                         xycoords='data', textcoords='offset points',
                         arrowprops=dict(facecolor='black', width=0.5, headwidth=2, headlength=4, edgecolor='black'),
                         horizontalalignment='left', verticalalignment='center',
                         fontsize=16, color='k', weight='bold')

# Marker 9 → lower right
for i in range(2):
    for j in range(2):
        ax[i,j].annotate(text="9", xy=(x4[8], y4[8]), xytext=(10, -10),
                         xycoords='data', textcoords='offset points',
                         arrowprops=dict(facecolor='black', width=0.5, headwidth=2, headlength=4, edgecolor='black'),
                         horizontalalignment='left', verticalalignment='top',
                         fontsize=16, color='k', weight='bold')

# Marker 2 → above
for i in range(2):
    for j in range(2):
        ax[i,j].annotate(text="2", xy=(x4[247], y4[247]), xytext=(0, 10),
                         xycoords='data', textcoords='offset points',
                         arrowprops=dict(facecolor='black', width=0.5, headwidth=2, headlength=4, edgecolor='black'),
                         horizontalalignment='center', verticalalignment='bottom',
                         fontsize=16, color='red', weight='bold')

# Marker 12 → lower left
for i in range(2):
    for j in range(2):
        ax[i,j].annotate(text="12", xy=(x4[546], y4[546]), xytext=(-10, -10),
                         xycoords='data', textcoords='offset points',
                         arrowprops=dict(facecolor='black', width=0.5, headwidth=2, headlength=4, edgecolor='black'),
                         horizontalalignment='right', verticalalignment='top',
                         fontsize=16, color='red', weight='bold')

# Marker 6 → above
for i in range(2):
    for j in range(2):
        ax[i,j].annotate(text="6", xy=(x4[418], y4[418]), xytext=(0, 10),
                         xycoords='data', textcoords='offset points',
                         arrowprops=dict(facecolor='black', width=0.5, headwidth=2, headlength=4, edgecolor='black'),
                         horizontalalignment='center', verticalalignment='bottom',
                         fontsize=16, color='red', weight='bold')

# Marker 7 → upper right
for i in range(2):
    for j in range(2):
        ax[i,j].annotate(text="7", xy=(x4[620], y4[620]), xytext=(10, 10),
                         xycoords='data', textcoords='offset points',
                         arrowprops=dict(facecolor='black', width=0.5, headwidth=2, headlength=4, edgecolor='black'),
                         horizontalalignment='left', verticalalignment='bottom',
                         fontsize=16, color='purple', weight='bold')

# Marker 8 → upper left
for i in range(2):
    for j in range(2):
        ax[i,j].annotate(text="8", xy=(x4[621], y4[621]), xytext=(-10, 10),
                         xycoords='data', textcoords='offset points',
                         arrowprops=dict(facecolor='black', width=0.5, headwidth=2, headlength=4, edgecolor='black'),
                         horizontalalignment='right', verticalalignment='bottom',
                         fontsize=16, color='purple', weight='bold')

plt.savefig(work_folder+"/analysis/Spatial_Q_D_NNSE_rsqr_01_06_p8_wloc.jpg",dpi=300,bbox_inches='tight')


# ------------------------------ Mainstream (Discharge and River water depth)------------------------------------------#

# Peak Flow Timing (out of 366 days)
PTQ01 = []
PTQ03 = []
PTQ06 = []
PTQ15 = []
p=0.90
for i in range(0,len(MRIVLOC15),1):
    j = (np.argwhere(MDIS15[:,i]>=np.quantile(MDIS15[:,i],p)))
    maxj = np.max(j[:,0])
    minj = np.min(j[:,0])
    PTQ15.append([maxj,minj])
PTQ15 = np.array(PTQ15)
PTQ15[PTQ15 > 250] -= 366
PTQ15 = PTQ15.transpose()

for i in range(0,len(MRIVLOC06),1):
    j = (np.argwhere(MDIS06[:,i]>=np.quantile(MDIS06[:,i],p)))
    maxj = np.max(j[:,0])
    minj = np.min(j[:,0])
    PTQ06.append([maxj,minj])
PTQ06 = np.array(PTQ06)
PTQ06[PTQ06 > 250] -= 366
PTQ06 = PTQ06.transpose()

for i in range(0,len(MRIVLOC03),1):
    j = (np.argwhere(MDIS03[:,i]>=np.quantile(MDIS03[:,i],p)))
    maxj = np.max(j[:,0])
    minj = np.min(j[:,0])
    PTQ03.append([maxj,minj])
PTQ03 = np.array(PTQ03)
PTQ03[PTQ03 > 250] -= 366
PTQ03 = PTQ03.transpose()

for i in range(0,len(MRIVLOC01),1):
    j = (np.argwhere(MDIS01[:,i]>=np.quantile(MDIS01[:,i],p)))
    maxj = np.max(j[:,0])
    minj = np.min(j[:,0])
    PTQ01.append([maxj,minj])
PTQ01 = np.array(PTQ01)
PTQ01[PTQ01 > 250] -= 366
PTQ01 = PTQ01.transpose()



# Peak depth timing (out of 366 days)
PTD01 = []
PTD03 = []
PTD06 = []
PTD15 = []

for i in range(0,len(MRIVLOC01),1):
    j = (np.argwhere(MDPH01[:,i]>=np.quantile(MDPH01[:,i],p)))
    maxj = np.max(j[:,0])
    minj = np.min(j[:,0])
    PTD01.append([maxj,minj])
PTD01 = np.array(PTD01)
PTD01[PTD01 > 250] -= 366
PTD01 = PTD01.transpose()


for i in range(0,len(MRIVLOC03),1):
    j = (np.argwhere(MDPH03[:,i]>=np.quantile(MDPH03[:,i],p)))
    maxj = np.max(j[:,0])
    minj = np.min(j[:,0])
    PTD03.append([maxj,minj])
PTD03 = np.array(PTD03)
PTD03[PTD03 > 250] -= 366
PTD03 = PTD03.transpose()


for i in range(0,len(MRIVLOC06),1):
    j = (np.argwhere(MDPH06[:,i]>=np.quantile(MDPH06[:,i],p)))
    maxj = np.max(j[:,0])
    minj = np.min(j[:,0])
    PTD06.append([maxj,minj])
PTD06 = np.array(PTD06)
PTD06[PTD06 > 250] -= 366
PTD06 = PTD06.transpose()

for i in range(0,len(MRIVLOC15),1):
    j = (np.argwhere(MDPH15[:,i]>=np.quantile(MDPH15[:,i],p)))
    maxj = np.max(j[:,0])
    minj = np.min(j[:,0])
    PTD15.append([maxj,minj])
PTD15 = np.array(PTD15)
PTD15[PTD15 > 250] -= 366
PTD15 = PTD15.transpose()





# -------------------Time series along the mainstream --------------------------------------------------------- #
## Plot Q
fig, ax2 = plt.subplots(2,2,figsize=(26,18))
ax2[0,0].fill_between(MRIVLEN15.flatten()/1000,MDIS15.max(axis=0)/10**4,MDIS15.min(axis=0)/10**4,facecolor=(0,0,1,0.4),edgecolor=(0,0,1,0.8),label='15min',linestyle='-',alpha=0.4)
ax2[0,0].fill_between(MRIVLEN06.flatten()/1000,MDIS06.max(axis=0)/10**4,MDIS06.min(axis=0)/10**4,facecolor=(0,0.6,0,0.4),edgecolor=(0,0.6,0,0.8),label='6min',linestyle='-',alpha=0.4)
ax2[0,0].fill_between(MRIVLEN03.flatten()/1000,MDIS03.max(axis=0)/10**4,MDIS03.min(axis=0)/10**4,facecolor=(1,1,0,0.4),edgecolor=(1,1,0,0.8),label='3min',linestyle='-',alpha=0.4)
ax2[0,0].fill_between(MRIVLEN01.flatten()/1000,MDIS01.max(axis=0)/10**4,MDIS01.min(axis=0)/10**4,facecolor=(1,0,0,0.4),edgecolor=(1,0,0,0.8),label='1min',linestyle='-',alpha=0.4)
ax2[0,0].set_ylabel("Q range (m\N{SUPERSCRIPT THREE}/s) x 10\N{SUPERSCRIPT FOUR}",fontsize=22)
ax2[0,0].grid(color='k', linestyle='--', linewidth=0.2, which='both')
ax2[0,0].legend(title="Model resolution",loc='upper left',facecolor=(255/256,255/256,255/256),fontsize=17, title_fontsize=18)
ax2[0,0].set_xlabel("<----- U/S    Distance from river mouth (km)    D/S ------> ",fontsize=22)
ax2[0,0].invert_xaxis()
ax2[0,0].set_title("(a) Discharge range", fontsize=22, weight="bold", loc='left')
ax2[0,0].tick_params(axis='both', which='major', labelsize=16)

## Plot Q timing
ax2[1,0].fill_between(MRIVLEN15.flatten()/1000,PTQ15.max(axis=0),PTQ15.min(axis=0),facecolor=(0,0,1,0.4),edgecolor=(0,0,1,0.8),label='15min',linestyle='-',alpha=0.4)
ax2[1,0].fill_between(MRIVLEN06.flatten()/1000,PTQ06.max(axis=0),PTQ06.min(axis=0),facecolor=(0,0.6,0,0.4),edgecolor=(0,0.6,0,0.8),label='6min',linestyle='-',alpha=0.4)
ax2[1,0].fill_between(MRIVLEN03.flatten()/1000,PTQ03.max(axis=0),PTQ03.min(axis=0),facecolor=(1,1,0,0.4),edgecolor=(1,1,0,0.8),label='3min',linestyle='-',alpha=0.4)
ax2[1,0].fill_between(MRIVLEN01.flatten()/1000,PTQ01.max(axis=0),PTQ01.min(axis=0),facecolor=(1,0,0,0.4),edgecolor=(1,0,0,0.8),label='1min',linestyle='-',alpha=0.4)
ax2[1,0].set_ylabel("Time range (days)",fontsize=22)
ax2[1,0].grid(color='k', linestyle='--', linewidth=0.2, which='both')
ax2[1,0].legend(title="Model resolution",loc='lower right',facecolor=(255/256,255/256,255/256),fontsize=17, title_fontsize=18)
ax2[1,0].set_xlabel("<----- U/S    Distance from river mouth (km)    D/S ------> ",fontsize=22)
ax2[1,0].invert_xaxis()
ax2[1,0].set_title("(b) Discharge timing variation", fontsize=22, weight="bold", loc='left')
ax2[1,0].tick_params(axis='both', which='major', labelsize=16)

## Plot D
ax2[0,1].fill_between(MRIVLEN15.flatten()/1000,MDPH15.max(axis=0),MDPH15.min(axis=0),facecolor=(0,0,1,0.4),edgecolor=(0,0,1,0.8),label='15min',linestyle='-')
ax2[0,1].fill_between(MRIVLEN06.flatten()/1000,MDPH06.max(axis=0),MDPH06.min(axis=0),facecolor=(0,0.6,0,0.4),edgecolor=(0,0.6,0,0.8),label='6min',linestyle='-')
ax2[0,1].fill_between(MRIVLEN03.flatten()/1000,MDPH03.max(axis=0),MDPH03.min(axis=0),facecolor=(1,1,0,0.4),edgecolor=(1,1,0,0.8),label='3min',linestyle='-')
ax2[0,1].fill_between(MRIVLEN01.flatten()/1000,MDPH01.max(axis=0),MDPH01.min(axis=0),facecolor=(1,0,0,0.4),edgecolor=(1,0,0,0.8),label='1min',linestyle='-')
ax2[0,1].set_ylabel("River depth range (m)",fontsize=22)
ax2[0,1].grid(color='k', linestyle='--', linewidth=0.2, which='both')
ax2[0,1].legend(title="Model resolution",loc='upper left',facecolor=(255/256,255/256,255/256),fontsize=17, title_fontsize=18)
ax2[0,1].set_xlabel("<----- U/S    Distance from river mouth (km)    D/S ------> ",fontsize=22)
ax2[0,1].invert_xaxis()
ax2[0,1].set_title("(c) Water depth range", fontsize=22, weight="bold", loc='left')
ax2[0,1].tick_params(axis='both', which='major', labelsize=16)

## Plot D timing
ax2[1,1].fill_between(MRIVLEN15.flatten()/1000,PTD15.max(axis=0),PTD15.min(axis=0),facecolor=(0,0,1,0.4),edgecolor=(0,0,1,0.8),label='15min',linestyle='-')
ax2[1,1].fill_between(MRIVLEN06.flatten()/1000,PTD06.max(axis=0),PTD06.min(axis=0),facecolor=(0,0.6,0,0.4),edgecolor=(0,0.6,0,0.8),label='6min',linestyle='-')
ax2[1,1].fill_between(MRIVLEN03.flatten()/1000,PTD03.max(axis=0),PTD03.min(axis=0),facecolor=(1,1,0,0.4),edgecolor=(1,1,0,0.8),label='3min',linestyle='-')
ax2[1,1].fill_between(MRIVLEN01.flatten()/1000,PTD01.max(axis=0),PTD01.min(axis=0),facecolor=(1,0,0,0.4),edgecolor=(1,0,0,0.8),label='1min',linestyle='-')
ax2[1,1].set_ylabel("Time range (days)",fontsize=22)
ax2[1,1].grid(color='k', linestyle='--', linewidth=0.2, which='both')
ax2[1,1].legend(title="Model resolution",loc='lower right',facecolor=(255/256,255/256,255/256),fontsize=17, title_fontsize=18)
ax2[1,1].set_xlabel("<----- U/S    Distance from river mouth (km)    D/S ------> ",fontsize=22)
ax2[1,1].invert_xaxis()
ax2[1,1].set_title("(d) Water depth timing variation", fontsize=22, weight="bold", loc='left')
ax2[1,1].tick_params(axis='both', which='major', labelsize=16)
plt.savefig(work_folder+'/analysis/MRIV_Q_D_Peak_Time_90p.jpg',dpi=500,bbox_inches='tight')

# ---------------------------------------------------------------------------- #


# --- Good discharge and river water depth and stage dicharge curve for it --- #
# at 351 loc 350
# at 9 loc 8
pos1 = 350
pos2 = 8
fig,ax = plt.subplots(3,2,figsize=(16,14))

ax[0,0].plot(AMZDIS01[:,loc_01[pos1,0],loc_01[pos1,1]],c='red',label="1min",alpha=0.6)
ax[0,0].plot(AMZDIS03[:,loc_03[pos1,0],loc_03[pos1,1]],c='yellow',label="3min")
ax[0,0].plot(AMZDIS06[:,loc_06[pos1,0],loc_06[pos1,1]],c='green',label="6min")
ax[0,0].plot(AMZDIS15[:,loc_15[pos1,0],loc_15[pos1,1]],c='blue',label="15min")
ax[0,0].grid(linestyle=':')
ax[0,0].set_xlabel("days",fontsize=12)
ax[0,0].set_ylabel("Discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=12)
ax[0,0].set_title("(a) Virtual station 351",fontsize=16, weight="bold", loc='left')

ax[0,0].text(0.6, .95,"NNSE", ha='left', va='top', transform=ax[0,0].transAxes, weight='bold')
ax[0,0].text(0.6, .90,"1 & 3 min ="+str(round(nnse_03_01_Qamz[pos1],3)),ha='left', va='top', transform=ax[0,0].transAxes)
ax[0,0].text(0.6, .85,"1 & 6 min ="+str(round(nnse_06_01_Qamz[pos1],3)),ha='left', va='top', transform=ax[0,0].transAxes)
ax[0,0].text(0.6, .80,"1 & 15 min ="+str(round(nnse_15_01_Qamz[pos1],3)),ha='left', va='top', transform=ax[0,0].transAxes)
ax[0,0].text(0.6, .75,"3 & 6 min ="+str(round(nnse_06_03_Qamz[pos1],3)),ha='left', va='top', transform=ax[0,0].transAxes)
ax[0,0].text(0.6, .70,"3 & 15 min ="+str(round(nnse_15_03_Qamz[pos1],3)),ha='left', va='top', transform=ax[0,0].transAxes)
ax[0,0].text(0.6, .65,"6 & 15 min ="+str(round(nnse_15_06_Qamz[pos1],3)),ha='left', va='top', transform=ax[0,0].transAxes)

ax[1,0].plot(AMZDPH01[:,loc_01[pos1,0],loc_01[pos1,1]],c='red')
ax[1,0].plot(AMZDPH03[:,loc_03[pos1,0],loc_03[pos1,1]],c='yellow')
ax[1,0].plot(AMZDPH06[:,loc_06[pos1,0],loc_06[pos1,1]],c='green')
ax[1,0].plot(AMZDPH15[:,loc_15[pos1,0],loc_15[pos1,1]],c='blue')
ax[1,0].grid(linestyle=':')
ax[1,0].set_xlabel("days",fontsize=12)
ax[1,0].set_ylabel("Water depth (m)",fontsize=12)
ax[1,0].set_title("(b)  Virtual station 351",fontsize=16, weight="bold", loc='left')

ax[1,0].text(0.6, .95,"NNSE",ha='left', va='top', transform=ax[1,0].transAxes, weight='bold')
ax[1,0].text(0.6, .90,"1 & 3min ="+str(round(nnse_03_01_Damz[pos1],3)),ha='left', va='top', transform=ax[1,0].transAxes)
ax[1,0].text(0.6, .85,"1 & 6min ="+str(round(nnse_06_01_Damz[pos1],3)),ha='left', va='top', transform=ax[1,0].transAxes)
ax[1,0].text(0.6, .80,"1 & 15min ="+str(round(nnse_15_01_Damz[pos1],3)),ha='left', va='top', transform=ax[1,0].transAxes)
ax[1,0].text(0.6, .75,"3 & 6min ="+str(round(nnse_06_03_Damz[pos1],3)),ha='left', va='top', transform=ax[1,0].transAxes)
ax[1,0].text(0.6, .70,"3 & 15min ="+str(round(nnse_15_03_Damz[pos1],3)),ha='left', va='top', transform=ax[1,0].transAxes)
ax[1,0].text(0.6, .65,"6 & 15min ="+str(round(nnse_15_06_Damz[pos1],3)),ha='left', va='top', transform=ax[1,0].transAxes)

ax[2,0].scatter(AMZDIS01[:,loc_01[pos1,0],loc_01[pos1,1]],AMZDPH01[:,loc_01[pos1,0],loc_01[pos1,1]],c='red')
ax[2,0].scatter(AMZDIS03[:,loc_03[pos1,0],loc_03[pos1,1]],AMZDPH03[:,loc_03[pos1,0],loc_03[pos1,1]],c='yellow')
ax[2,0].scatter(AMZDIS06[:,loc_06[pos1,0],loc_06[pos1,1]],AMZDPH06[:,loc_06[pos1,0],loc_06[pos1,1]],c='green')
ax[2,0].scatter(AMZDIS15[:,loc_15[pos1,0],loc_15[pos1,1]],AMZDPH15[:,loc_15[pos1,0],loc_15[pos1,1]],c='blue')
ax[2,0].grid(linestyle=':')
ax[2,0].set_xlabel("Discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=12)
ax[2,0].set_ylabel("Water depth (m)",fontsize=12)
ax[2,0].set_title("(c)  Virtual station 351",fontsize=16, weight="bold", loc='left')

ax[0,1].plot(AMZDIS01[:,loc_01[pos2,0],loc_01[pos2,1]],c='red')
ax[0,1].plot(AMZDIS03[:,loc_03[pos2,0],loc_03[pos2,1]],c='yellow')
ax[0,1].plot(AMZDIS06[:,loc_06[pos2,0],loc_06[pos2,1]],c='green')
ax[0,1].plot(AMZDIS15[:,loc_15[pos2,0],loc_15[pos2,1]],c='blue')
ax[0,1].grid(linestyle=':')
ax[0,1].set_xlabel("days",fontsize=12)
ax[0,1].set_ylabel("Discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=12)
ax[0,1].set_title("(d)  Virtual station 9",fontsize=16, weight="bold", loc='left')

ax[0,1].text(0.70, .95,"NNSE",ha='left', va='top', transform=ax[0,1].transAxes, weight='bold')
ax[0,1].text(0.70, .90,"1 & 3 min ="+str(round(nnse_03_01_Qamz[pos2],3)),ha='left', va='top', transform=ax[0,1].transAxes)
ax[0,1].text(0.70, .85,"1 & 6 min ="+str(round(nnse_06_01_Qamz[pos2],3)),ha='left', va='top', transform=ax[0,1].transAxes)
ax[0,1].text(0.70, .80,"1 & 15 min ="+str(round(nnse_15_01_Qamz[pos2],3)),ha='left', va='top', transform=ax[0,1].transAxes)
ax[0,1].text(0.70, .75,"3 & 6 min ="+str(round(nnse_06_03_Qamz[pos2],3)),ha='left', va='top', transform=ax[0,1].transAxes)
ax[0,1].text(0.70, .70,"3 & 15 min ="+str(round(nnse_15_03_Qamz[pos2],3)),ha='left', va='top', transform=ax[0,1].transAxes)
ax[0,1].text(0.70, .65,"6 & 15 min ="+str(round(nnse_15_06_Qamz[pos2],3)),ha='left', va='top', transform=ax[0,1].transAxes)

ax[1,1].plot(AMZDPH01[:,loc_01[pos2,0],loc_01[pos2,1]],c='red')
ax[1,1].plot(AMZDPH03[:,loc_03[pos2,0],loc_03[pos2,1]],c='yellow')
ax[1,1].plot(AMZDPH06[:,loc_06[pos2,0],loc_06[pos2,1]],c='green')
ax[1,1].plot(AMZDPH15[:,loc_15[pos2,0],loc_15[pos2,1]],c='blue')
ax[1,1].grid(linestyle=':')
ax[1,1].set_xlabel("days",fontsize=12)
ax[1,1].set_ylabel("Water depth (m)",fontsize=12)
ax[1,1].set_title("(e)  Virtual station 9",fontsize=16, weight="bold", loc='left')

ax[1,1].text(0.70, .95,"NNSE",ha='left', va='top', transform=ax[1,1].transAxes, weight='bold')
ax[1,1].text(0.70, .90,"1 & 3 min ="+str(round(nnse_03_01_Damz[pos2],3)),ha='left', va='top', transform=ax[1,1].transAxes)
ax[1,1].text(0.70, .85,"1 & 6 min ="+str(round(nnse_06_01_Damz[pos2],3)),ha='left', va='top', transform=ax[1,1].transAxes)
ax[1,1].text(0.70, .80,"1 & 15 min ="+str(round(nnse_15_01_Damz[pos2],3)),ha='left', va='top', transform=ax[1,1].transAxes)
ax[1,1].text(0.70, .75,"3 & 6 min ="+str(round(nnse_06_03_Damz[pos2],3)),ha='left', va='top', transform=ax[1,1].transAxes)
ax[1,1].text(0.70, .70,"3 & 15 min ="+str(round(nnse_15_03_Damz[pos2],3)),ha='left', va='top', transform=ax[1,1].transAxes)
ax[1,1].text(0.70, .65,"6 & 15 min ="+str(round(nnse_15_06_Damz[pos2],3)),ha='left', va='top', transform=ax[1,1].transAxes)


ax[2,1].scatter(AMZDIS01[:,loc_01[pos2,0],loc_01[pos2,1]],AMZDPH01[:,loc_01[pos2,0],loc_01[pos2,1]],c='red')
ax[2,1].scatter(AMZDIS03[:,loc_03[pos2,0],loc_03[pos2,1]],AMZDPH03[:,loc_03[pos2,0],loc_03[pos2,1]],c='yellow')
ax[2,1].scatter(AMZDIS06[:,loc_06[pos2,0],loc_06[pos2,1]],AMZDPH06[:,loc_06[pos2,0],loc_06[pos2,1]],c='green')
ax[2,1].scatter(AMZDIS15[:,loc_15[pos2,0],loc_15[pos2,1]],AMZDPH15[:,loc_15[pos2,0],loc_15[pos2,1]],c='blue')
ax[2,1].grid(linestyle=':')
ax[2,1].set_xlabel("Discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=12)
ax[2,1].set_ylabel("Water depth (m)",fontsize=12)
ax[2,1].set_title("(f)  Virtual station 9",fontsize=16, weight="bold", loc='left')

lines_labels = [fig.axes[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels,loc='lower center',ncol=4, markerscale=20,fontsize=12) 

plt.subplots_adjust(hspace=0.35,bottom=0.075)

plt.savefig(work_folder+'/analysis/timseries_stage_discahrge.jpg',dpi=300,bbox_inches='tight')

######### --------------------------------------------- All Others (For Supplementary) ------------------------------ ###########

# --------------------- Spatial Distrbution of Locations --------------------- #
Q_Low_loc_15_01 = np.argwhere(np.array(nnse_15_01_Qamz)<0.3)
D_Low_loc_15_01 = np.argwhere(np.array(nnse_15_01_Damz)<0.3)

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import copy


# For Meridians and Parallels (Subbasin Boundaries)
def getGridLines(dat, east, west, north, south, nx, ny):
    lonsize = float(east - west) / nx
    latsize = float(south - north) / ny

    # Get meridians
    lons, lats = [], []
    lats_north = np.linspace(north, south, ny + 1)[:-1]
    for ix in range(nx - 1):
        lats_inter = copy.copy(lats_north)
        lats_inter[dat[:, ix] == dat[:, ix + 1]] = np.nan
        lats_this = np.r_[np.c_[lats_north, lats_inter].reshape(-1), south]
        lons_this = np.ones((ny * 2 + 1)) * (west + (ix + 1) * lonsize)
        lons.append(np.r_[lons_this, np.nan])
        lats.append(np.r_[lats_this, np.nan])
    meridians = (np.array(lons).reshape(-1), np.array(lats).reshape(-1))

    # Get parallels
    lons, lats = [], []
    lons_west = np.linspace(west, east, nx + 1)[:-1]
    for iy in range(ny - 1):
        lons_inter = copy.copy(lons_west)
        lons_inter[dat[iy, :] == dat[iy + 1, :]] = np.nan
        lons_this = np.r_[np.c_[lons_west, lons_inter].reshape(-1), east]
        lats_this = np.ones((nx * 2 + 1)) * (north + (iy + 1) * latsize)
        lons.append(np.r_[lons_this, np.nan])
        lats.append(np.r_[lats_this, np.nan])
    parallels = (np.array(lons).reshape(-1), np.array(lats).reshape(-1))

    return meridians, parallels


#%% Lat lon of the basin data
# 15 min
FLOC = np.argwhere((BSN15==1)&(NXTXY15[0]!=-9))  ## location of pixel with highest value of river depth (river mouth)
FLON15 = LONLAT15[0,FLOC[:,0],FLOC[:,1]]
FLAT15 = LONLAT15[1,FLOC[:,0],FLOC[:,1]]
TLOCROW15 = NXTXY15[1,FLOC[:,0],FLOC[:,1]]-1
TLOCCOL15 = NXTXY15[0,FLOC[:,0],FLOC[:,1]]-1
TLON15 = LONLAT15[0,TLOCROW15,TLOCCOL15]
TLAT15 = LONLAT15[1,TLOCROW15,TLOCCOL15]
RIVLATLON15 = np.column_stack((FLON15,FLAT15,TLON15,TLAT15))

lwidth = np.log10(RIVWTH15[TLOCROW15,TLOCCOL15])/2

WEST, EAST, SOUTH, NORTH = -80, -49, -22, 7

# Set up Cartopy map
fig, ax = plt.subplots(figsize=(18, 18), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([WEST, EAST, SOUTH, NORTH], crs=ccrs.PlateCarree())
# Draw coastlines and fill continents
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.add_feature(cfeature.LAND, facecolor='white')
# ax.stock_img()

# Draw parallels and meridians
parallels = np.arange(SOUTH, NORTH, 5)
meridians = np.arange(WEST, EAST, 5)

gl = ax.gridlines(draw_labels=True, xlocs=meridians, ylocs=parallels, color='gray', linewidth=0.1,  crs=ccrs.PlateCarree())
gl.bottom_labels = False
gl.right_labels = False

# Define the extent for the elevation plot
cx0, cx1 = WEST, EAST
cy0, cy1= NORTH, SOUTH

# Normalize the color scale for the elevation plot
norm1 = [-9999, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290,
         300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
         3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000]


# Transform river coordinates
x, y = RIVLATLON15[:,0], RIVLATLON15[:,1]
x1, y1 = RIVLATLON15[:,2], RIVLATLON15[:,3]
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax.add_collection(LineCollection(pts, color="gray", label="River network", alpha=0.6, linewidth=lwidth))


BSN = BSN01.copy()
BSN[BSN!=1]=0
# Plot basin boundaries (replace `meridians1` and `parallels1` with your actual boundary data)
meridians1, parallels1 = getGridLines(BSN, E, W, N, S, np.int32((E-W)/csize01), np.int32((N-S)/csize01))
ax.plot(meridians1[0], meridians1[1], linewidth=1, color='k', alpha=0.3)
ax.plot(parallels1[0], parallels1[1], linewidth=1, color='k', alpha=0.3)

#plot virtual staions  
printed_locations = set()

ax.scatter(LAT_LON_15[:,1],LAT_LON_15[:,0],s=22,marker='^',facecolor='coral',edgecolors='None',linewidths=0.01,label='Virtual stations')

for j in Q_Low_loc_15_01:
    if ID_15[j[0]] not in printed_locations:
        ax.text(LAT_LON_15[j[0],1],LAT_LON_15[j[0],0],s=str(ID_15[j[0]]),c='red',weight='bold',fontsize=8)
        printed_locations.add(ID_15[j[0]])

for k in D_Low_loc_15_01:
    if ID_15[k[0]] not in printed_locations:
        ax.text(LAT_LON_15[k[0],1],LAT_LON_15[k[0],0],s=str(ID_15[k[0]]),c='red',weight='bold',fontsize=8)
        printed_locations.add(ID_15[k[0]])

for i in range(0,ID_15.shape[0],2):
    if ID_15[i] not in printed_locations:
        ax.text(LAT_LON_15[i,1],LAT_LON_15[i,0],s=str(ID_15[i]),c='k',fontsize=8)
        printed_locations.add(ID_15[i])


# Create a Rectangle patch
rect1 = patches.Rectangle(xy=(-75.5,-9.5), width=3, height=-4.5, linewidth=1.2, edgecolor='blue', facecolor='none', label="Analysis zones",zorder=3,ls='--')
rect2 = patches.Rectangle(xy=(-64.5,-3), width=3.5, height=-1.5, linewidth=1.2, edgecolor='blue', facecolor='none',zorder=3,ls='--')
rect3 = patches.Rectangle(xy=(-71.5,-11), width=3.5, height=-2.5, linewidth=1.2, edgecolor='blue', facecolor='none',zorder=3,ls='--')
rect4 = patches.Rectangle(xy=(-66.5,0.5), width=3.5, height=-1.5, linewidth=1.2, edgecolor='blue', facecolor='none',zorder=3,ls='--')
rect5 = patches.Rectangle(xy=(-62.5,-2.5), width=4, height=-2.5, linewidth=1.2, edgecolor='blue', facecolor='none',zorder=3,ls='--') 

# Add the patch to the Axes
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)

ax.annotate("5",(-75.5+0.5,-9.5-2.25),color='blue', weight='bold', fontsize=12, ha='center', va='center',alpha=0.8)
ax.annotate("2",(-64.5+0.5,-3-1.15),color='blue', weight='bold', fontsize=12, ha='center', va='center',alpha=0.8)
ax.annotate("4",(-71.5+1.75,-11-1.25),color='blue', weight='bold', fontsize=12, ha='center', va='center',alpha=0.8)
ax.annotate("3",(-66.5+1.75,0.5-0.35),color='blue', weight='bold', fontsize=12, ha='center', va='center',alpha=0.8)
ax.annotate("1",(-62.5+2,-2.5-1.45),color='blue', weight='bold', fontsize=12, ha='center', va='center',alpha=0.8)


plt.legend(loc="lower right", fontsize=18,markerscale=3)
plt.savefig(work_folder+"/analysis/vitual_stations_1.jpg",dpi=300,bbox_inches='tight')

# -------------------- Spatial distrbution of NNSE and R2 -------------------- #

# --------------------------------- Discharge -------------------------------- #
norm1 = [-9999,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130,140,141,142,143,144,145,146,147,148,149,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,
         300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,
         3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000]

fig, ax = plt.subplots(3,2,figsize=(16,22))
m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[2,0])
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[2,0])
m.shadedrelief(zorder=1 ,ax=ax[2,0])

# Linewith of rivers
lwidth_15 = (np.log10(RIVWTH15[TLOCROW15,TLOCCOL15]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON15[:,0], RIVLATLON15[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON15[:,2], RIVLATLON15[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[2,0].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_15)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv15lon,riv15lat)
im1 = ax[2,0].scatter(x4,y4,s=10,c=nnse_15_01_Qamz,facecolor=nnse_15_01_Qamz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar1 = m.colorbar(im1,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[2,0])
cbar1.set_label("NNSE",fontsize=15)
ax[2,0].set_title("(c) NNSE (1min & 15min)",loc='left',fontsize=16,weight='bold')

m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[2,1] )
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[2,1])
m.shadedrelief(zorder=1 ,ax=ax[2,1])
# Linewith of rivers
lwidth_15 = (np.log10(RIVWTH15[TLOCROW15,TLOCCOL15]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON15[:,0], RIVLATLON15[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON15[:,2], RIVLATLON15[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[2,1].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_15)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv15lon,riv15lat)
im3 = ax[2,1].scatter(x4,y4,s=10,c=rsqr_15_01_Qamz,facecolor=rsqr_15_01_Qamz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar3 = m.colorbar(im3,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[2,1])
cbar3.set_label("R\N{SUPERSCRIPT TWO}",fontsize=15)
ax[2,1].set_title("(f) R\N{SUPERSCRIPT TWO} (1min & 15min)",loc='left',fontsize=16,weight='bold')

m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[1,0])
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[1,0])
m.shadedrelief(zorder=1 ,ax=ax[1,0])
# Linewith of rivers
lwidth_06 = (np.log10(RIVWTH06[TLOCROW06,TLOCCOL06]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON06[:,0], RIVLATLON06[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON06[:,2], RIVLATLON06[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[1,0].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_06)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv06lon,riv06lat)
im5 = ax[1,0].scatter(x4,y4,s=10,c=nnse_06_01_Qamz,facecolor=nnse_06_01_Qamz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar5 = m.colorbar(im5,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[1,0])
cbar5.set_label("NNSE",fontsize=15)
ax[1,0].set_title("(b) NNSE (1min & 6min)",loc='left',fontsize=16,weight='bold')

m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[1,1] )
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[1,1])
m.shadedrelief(zorder=1 ,ax=ax[1,1])
# Linewith of rivers
lwidth_06 = (np.log10(RIVWTH06[TLOCROW06,TLOCCOL06]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON06[:,0], RIVLATLON06[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON06[:,2], RIVLATLON06[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[1,1].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_06)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv06lon,riv06lat)
im7 = ax[1,1].scatter(x4,y4,s=10,c=rsqr_06_01_Qamz,facecolor=rsqr_06_01_Qamz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar7 = m.colorbar(im7,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[1,1])
cbar7.set_label("R\N{SUPERSCRIPT TWO}",fontsize=15)
ax[1,1].set_title("(e) R\N{SUPERSCRIPT TWO} (1min & 6min)",loc='left',fontsize=16,weight='bold')

m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[0,0] )
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[0,0])
m.shadedrelief(zorder=1 ,ax=ax[0,0])
# Linewith of rivers
lwidth_03 = (np.log10(RIVWTH03[TLOCROW03,TLOCCOL03]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON03[:,0], RIVLATLON03[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON03[:,2], RIVLATLON03[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[0,0].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_03)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv03lon,riv03lat)
im9 = ax[0,0].scatter(x4,y4,s=10,c=nnse_03_01_Qamz,facecolor=nnse_03_01_Qamz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar9 = m.colorbar(im9,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[0,0])
cbar9.set_label("NNSE",fontsize=15)
ax[0,0].set_title("(a) NNSE (1min & 3min)",loc='left',fontsize=16,weight='bold')

m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[0,1])
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[0,1])
m.shadedrelief(zorder=1 ,ax=ax[0,1])
# Linewith of rivers
lwidth_03 = (np.log10(RIVWTH03[TLOCROW03,TLOCCOL03]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON03[:,0], RIVLATLON03[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON03[:,2], RIVLATLON03[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[0,1].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_03)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv03lon,riv03lat)
im11 = ax[0,1].scatter(x4,y4,s=10,c=rsqr_03_01_Qamz,facecolor=rsqr_03_01_Qamz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar11 = m.colorbar(im11,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[0,1])
cbar11.set_label("R\N{SUPERSCRIPT TWO}",fontsize=15)
ax[0,1].set_title("(d) R\N{SUPERSCRIPT TWO} (1min & 3min)",loc='left',fontsize=16,weight='bold')
# # # plt.savefig(work_folder+'/analysis/Spatial_Q_NNSE_rsqr_01_03_06_15.jpg",dpi=500,bbox_inches='tight')
plt.subplots_adjust(hspace=0.2,wspace=0.05)
plt.savefig(work_folder+"/analysis/Spatial_Q_NNSE_rsqr_01_03_06_15_p1.jpg",dpi=500,bbox_inches='tight')


# -------------------------------- Water depth ------------------------------- #
norm1 = [-9999,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130,140,141,142,143,144,145,146,147,148,149,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,
         300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,
         3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000]

fig, ax = plt.subplots(3,2,figsize=(16,22))
m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[2,0])
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[2,0])
m.shadedrelief(zorder=1 ,ax=ax[2,0])
# Linewith of rivers
lwidth_15 = (np.log10(RIVWTH15[TLOCROW15,TLOCCOL15]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON15[:,0], RIVLATLON15[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON15[:,2], RIVLATLON15[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[2,0].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_15)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv15lon,riv15lat)
im1 = ax[2,0].scatter(x4,y4,s=10,c=nnse_15_01_Damz,facecolor=nnse_15_01_Damz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar1 = m.colorbar(im1,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[2,0])
cbar1.set_label("NNSE",fontsize=15)
ax[2,0].set_title("(c) NNSE (1min & 15min)",loc='left',fontsize=16,weight='bold')

m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[2,1] )
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[2,1])
m.shadedrelief(zorder=1 ,ax=ax[2,1])
# Linewith of rivers
lwidth_15 = (np.log10(RIVWTH15[TLOCROW15,TLOCCOL15]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON15[:,0], RIVLATLON15[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON15[:,2], RIVLATLON15[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[2,1].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_15)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv15lon,riv15lat)
im3 = ax[2,1].scatter(x4,y4,s=10,c=rsqr_15_01_Damz,facecolor=rsqr_15_01_Damz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar3 = m.colorbar(im3,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[2,1])
cbar3.set_label("R\N{SUPERSCRIPT TWO}",fontsize=15)
ax[2,1].set_title("(f) R\N{SUPERSCRIPT TWO} (1min & 15min)",loc='left',fontsize=16,weight='bold')

m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[1,0])
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[1,0])
m.shadedrelief(zorder=1 ,ax=ax[1,0])
# Linewith of rivers
lwidth_06 = (np.log10(RIVWTH06[TLOCROW06,TLOCCOL06]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON06[:,0], RIVLATLON06[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON06[:,2], RIVLATLON06[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[1,0].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_06)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv06lon,riv06lat)
im5 = ax[1,0].scatter(x4,y4,s=10,c=nnse_06_01_Damz,facecolor=nnse_06_01_Damz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar5 = m.colorbar(im5,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[1,0])
cbar5.set_label("NNSE",fontsize=15)
ax[1,0].set_title("(b) NNSE (1min & 6min)",loc='left',fontsize=16,weight='bold')

m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[1,1] )
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[1,1])
m.shadedrelief(zorder=1 ,ax=ax[1,1])
# Linewith of rivers
lwidth_06 = (np.log10(RIVWTH06[TLOCROW06,TLOCCOL06]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON06[:,0], RIVLATLON06[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON06[:,2], RIVLATLON06[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[1,1].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_06)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv06lon,riv06lat)
im7 = ax[1,1].scatter(x4,y4,s=10,c=rsqr_06_01_Damz,facecolor=rsqr_06_01_Damz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar7 = m.colorbar(im7,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[1,1])
cbar7.set_label("R\N{SUPERSCRIPT TWO}",fontsize=15)
ax[1,1].set_title("(e) R\N{SUPERSCRIPT TWO} (1min & 6min)",loc='left',fontsize=16,weight='bold')

m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[0,0] )
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[0,0])
m.shadedrelief(zorder=1 ,ax=ax[0,0])
# Linewith of rivers
lwidth_03 = (np.log10(RIVWTH03[TLOCROW03,TLOCCOL03]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON03[:,0], RIVLATLON03[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON03[:,2], RIVLATLON03[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[0,0].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_03)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv03lon,riv03lat)
im9 = ax[0,0].scatter(x4,y4,s=10,c=nnse_03_01_Damz,facecolor=nnse_03_01_Damz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar9 = m.colorbar(im9,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[0,0])
cbar9.set_label("NNSE",fontsize=15)
ax[0,0].set_title("(a) NNSE (1min & 3min)",loc='left',fontsize=16,weight='bold')

m = Basemap(projection='cyl',llcrnrlat=-20,urcrnrlat=5,llcrnrlon=-80,urcrnrlon=-50,resolution='h')
m.drawparallels(range(-25, 15, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 ,ax=ax[0,1])
m.drawmeridians(range(-90, -30, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0,ax=ax[0,1])
m.shadedrelief(zorder=1 ,ax=ax[0,1])
# Linewith of rivers
lwidth_03 = (np.log10(RIVWTH03[TLOCROW03,TLOCCOL03]))/5
# define extent and transform the coordinate
cx0,cy0 = m(-90,10)
cx1,cy1=m(-40,-25)
x, y = m(RIVLATLON03[:,0], RIVLATLON03[:,1])  # transform coordinates
x1, y1 = m(RIVLATLON03[:,2], RIVLATLON03[:,3])
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax[0,1].add_collection(LineCollection(pts, color="k", label="River",alpha=0.8, linewidth=lwidth_03)) # drawing lines
# for perfromance 
norm2 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
x4,y4 = m(riv03lon,riv03lat)
im11 = ax[0,1].scatter(x4,y4,s=10,c=rsqr_03_01_Damz,facecolor=rsqr_03_01_Damz,edgecolor='none',alpha=0.8,norm=colors.BoundaryNorm(norm2,256),cmap='hot', zorder=3)
cbar11 = m.colorbar(im11,location='bottom',extend='both',aspect=20,shrink=0.5,pad=0.3,ax=ax[0,1])
cbar11.set_label("R\N{SUPERSCRIPT TWO}",fontsize=15)
ax[0,1].set_title("(d) R\N{SUPERSCRIPT TWO} (1min & 3min)",loc='left',fontsize=16,weight='bold')
plt.subplots_adjust(hspace=0.2,wspace=0.05)
plt.savefig(work_folder+"/analysis/Spatial_D_NNSE_rsqr_01_03_06_15_p1.jpg",dpi=500,bbox_inches='tight')



################## - ---------------------- Flooded Area and Flood Extent -------------------- ##################

# ------------------------------- flooded area ------------------------------- #

#  %% Flooded Area Comparion for Various Zones 
fldare_15 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic/fldare2008.bin",np.float32).reshape(-1,140,200)
fldare_06 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic/fldare2008.bin",np.float32).reshape(-1,350,500)
fldare_03 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic/fldare2008.bin",np.float32).reshape(-1,700,1000)
fldare_01 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic/fldare2008.bin",np.float32).reshape(-1,2100,3000)

# Lat=-2.5 to -5.0 Lon= -62.5 to -58.5
fld15 = []
fld06 = []
fld03 = []
fld01 = []
for i in range(365):
    fld15.append(np.nansum(fldare_15[i,50:60,110:126]/10**9))
    fld06.append(np.nansum(fldare_06[i,125:150,275:315]/10**9))
    fld03.append(np.nansum(fldare_03[i,250:300,550:630]/10**9))
    fld01.append(np.nansum(fldare_01[i,750:900,1650:1890]/10**9))

plt.figure(figsize=(16,9))
plt.plot(fld15,label="15 min",color='blue',alpha=0.7)
plt.plot(fld06,label="06 min",color='green',alpha=0.7)
plt.plot(fld03,label="03 min",color='yellow',alpha=0.7)
plt.plot(fld01,label="01 min",color='red',alpha=0.7)
plt.legend()
plt.xlabel("Days")
plt.ylim(0,40)
plt.ylabel("Flooded Area (m\N{SUPERSCRIPT TWO}) x 10\N{SUPERSCRIPT NINE}")
# plt.title("Flooded area (m\N{SUPERSCRIPT TWO}, Lat=-2.5 to -5.0 Lon= -62.5 to -58.5)")
plt.savefig(work_folder+"/analysis/Zone1_FA_Lat=-2.5 to -5.0 Lon= -62.5 to -58.5).png",dpi=500,bbox_inches='tight')
    
# Lat=-3 to -4.5 lon= -64.5 to -61
fld15 = []
fld06 = []
fld03 = []
fld01 = []
for i in range(365):
    fld15.append(np.nansum(fldare_15[i,52:58,102:116]/10**9))
    fld06.append(np.nansum(fldare_06[i,130:145,255:290]/10**9))
    fld03.append(np.nansum(fldare_03[i,260:290,510:580]/10**9))
    fld01.append(np.nansum(fldare_01[i,780:870,1530:1740]/10**9))

plt.figure(figsize=(16,9))
plt.plot(fld15,label="15 min",color='blue',alpha=0.7)
plt.plot(fld06,label="06 min",color='green',alpha=0.7)
plt.plot(fld03,label="03 min",color='yellow',alpha=0.7)
plt.plot(fld01,label="01 min",color='red',alpha=0.7)
plt.legend()
plt.xlabel("Days")
plt.ylabel("Flooded Area (m\N{SUPERSCRIPT TWO}) x 10\N{SUPERSCRIPT NINE}")
# plt.title("Flooded area (m\N{SUPERSCRIPT TWO}, Lat=-3 to -4.5 lon= -64.5 to -61)")
plt.savefig(work_folder+"/analysis/Zone2_FA_Lat=-3 to -4.5 lon= -64.5 to -61).png",dpi=500,bbox_inches='tight')

# Lat=0.5 to -1.0 Lon= -66.5 to -63.0
fld15 = []
fld06 = []
fld03 = []
fld01 = []
for i in range(365):
    fld15.append(np.nansum(fldare_15[i,38:44,94:108]/10**9))
    fld06.append(np.nansum(fldare_06[i,95:110,235:270]/10**9))
    fld03.append(np.nansum(fldare_03[i,190:220,470:540]/10**9))
    fld01.append(np.nansum(fldare_01[i,570:660,1410:1620]/10**9))

plt.figure(figsize=(16,9))
plt.plot(fld15,label="15 min",color='blue',alpha=0.7)
plt.plot(fld06,label="06 min",color='green',alpha=0.7)
plt.plot(fld03,label="03 min",color='yellow',alpha=0.7)
plt.plot(fld01,label="01 min",color='red',alpha=0.7)
plt.legend()
plt.xlabel("Days")
plt.ylabel("Flooded Area (m\N{SUPERSCRIPT TWO}) x 10\N{SUPERSCRIPT NINE}")
# plt.title("Flooded area (m\N{SUPERSCRIPT TWO}, Lat=0.5 to -1.0 Lon= -66.5 to -63.0)")
plt.savefig(work_folder+"/analysis/Zone3_FA_Lat=0.5 to -1.0 Lon= -66.5 to -63.0).png",dpi=500,bbox_inches='tight')


# Lat=-11 to -13.5 Lon= -71.5 to -68
fld15 = []
fld06 = []
fld03 = []
fld01 = []
for i in range(365):
    fld15.append(np.nansum(fldare_15[i,84:94,74:88]/10**9))
    fld06.append(np.nansum(fldare_06[i,210:235,185:220]/10**9))
    fld03.append(np.nansum(fldare_03[i,420:470,370:440]/10**9))
    fld01.append(np.nansum(fldare_01[i,1260:1410,1110:1320]/10**9))

plt.figure(figsize=(16,9))
plt.plot(fld15,label="15 min",color='blue',alpha=0.7)
plt.plot(fld06,label="06 min",color='green',alpha=0.7)
plt.plot(fld03,label="03 min",color='yellow',alpha=0.7)
plt.plot(fld01,label="01 min",color='red',alpha=0.7)
plt.legend()
plt.xlabel("Days")
plt.ylabel("Flooded Area (m\N{SUPERSCRIPT TWO}) x 10\N{SUPERSCRIPT NINE}")
# plt.title("Flooded area (m\N{SUPERSCRIPT TWO}, Lat=-11 to -13.5 Lon= -71.5 to -68)")
plt.savefig(work_folder+"/analysis/Zone4_FA_Lat=-11 to -13.5 Lon= -71.5 to -68).png",dpi=500,bbox_inches='tight')


## Lat=-9.5 to -14 lon= -75.5 to -72.5
fld15 = []
fld06 = []
fld03 = []
fld01 = []
for i in range(365):
    fld15.append(np.nansum(fldare_15[i,78:96,58:70]/10**9))
    fld06.append(np.nansum(fldare_06[i,195:240,145:175]/10**9))
    fld03.append(np.nansum(fldare_03[i,390:480,290:350]/10**9))
    fld01.append(np.nansum(fldare_01[i,1170:1440,870:1050]/10**9))

plt.figure(figsize=(16,9))
plt.plot(fld15,label="15 min",color='blue',alpha=0.7)
plt.plot(fld06,label="06 min",color='green',alpha=0.7)
plt.plot(fld03,label="03 min",color='yellow',alpha=0.7)
plt.plot(fld01,label="01 min",color='red',alpha=0.7)
plt.legend()
plt.xlabel("Days")
plt.ylabel("Flooded Area (m\N{SUPERSCRIPT TWO}) x 10\N{SUPERSCRIPT NINE}")
plt.savefig(work_folder+"/analysis/Zone5_FA_Lat=-9.5 to -14 lon= -75.5 to -72.5).png",dpi=500,bbox_inches='tight')


# ------------------------------- Map River Network and Zones ------------------------------- #

## for the colors normalisation
norm1 = [-9999,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130,140,141,142,143,144,145,146,147,148,149,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,
         300,310,320,330,340,350,360,370,380,390,400,410,420,430,440,450,460,470,480,490,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,
         3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000]

## Figure Size
fig, ax = plt.subplots(figsize=(16,9))

# m = Basemap(projection='cyl',llcrnrlat=-25,urcrnrlat=10,llcrnrlon=-90,urcrnrlon=-40,resolution='h')
m = Basemap(projection='cyl',llcrnrlat=-22,urcrnrlat=8,llcrnrlon=-82,urcrnrlon=-47,resolution='h')
m.drawcoastlines(color='lightgray')
m.drawcountries(color='lightgray')
m.fillcontinents(color='white', lake_color='#eeeeee');
m.drawmapboundary(fill_color='#eeeeee')
m.drawparallels(range(-25, 10, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0 )
m.drawmeridians(range(-82, -42, 5), linewidth=2, dashes=[4, 2], labels=[1,0,0,1], color='r', zorder=0)

# transform coordinates
x, y = m(RIVLATLON01[:,0], RIVLATLON01[:,1])  
x1, y1 = m(RIVLATLON01[:,2], RIVLATLON01[:,3])

# Linewidth of rivers
lwidth = (np.log10(RIVWTH01[TLOCROW01,TLOCCOL01]))/5

# set of points for line drawing 
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)

# drawing lines
plt.gca().add_collection(LineCollection(pts, color="blue", label="River",alpha=1, linewidth=lwidth))

# Create a Rectangle patch
rect1 = patches.Rectangle(xy=(-75.5,-9.5), width=3, height=-4.5, linewidth=2, edgecolor='purple', facecolor='none', label="Analysis zones",zorder=3,ls='--')
rect2 = patches.Rectangle(xy=(-64.5,-3), width=3.5, height=-1.5, linewidth=2, edgecolor='purple', facecolor='none',zorder=3,ls='--')
rect3 = patches.Rectangle(xy=(-71.5,-11), width=3.5, height=-2.5, linewidth=2, edgecolor='purple', facecolor='none',zorder=3,ls='--')
rect4 = patches.Rectangle(xy=(-66.5,0.5), width=3.5, height=-1.5, linewidth=2, edgecolor='purple', facecolor='none',zorder=3,ls='--')
rect5 = patches.Rectangle(xy=(-62.5,-2.5), width=4, height=-2.5, linewidth=2, edgecolor='purple', facecolor='none',zorder=3,ls='--') 

# Add the patch to the Axes
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(rect4)
ax.add_patch(rect5)

ax.annotate("5",(-75.5+1.5,-9.5-2.25),color='crimson', weight='bold', fontsize=14, ha='center', va='center',alpha=1)
ax.annotate("2",(-64.5+1.75,-3-0.75),color='crimson', weight='bold', fontsize=14, ha='center', va='center',alpha=1)
ax.annotate("4",(-71.5+1.75,-11-1.25),color='crimson', weight='bold', fontsize=14, ha='center', va='center',alpha=1)
ax.annotate("3",(-66.5+1.75,0.5-0.75),color='crimson', weight='bold', fontsize=14, ha='center', va='center',alpha=1)
ax.annotate("1",(-62.5+2,-2.5-1.25),color='crimson', weight='bold', fontsize=14, ha='center', va='center',alpha=1)

plt.legend(loc='upper right')
plt.savefig(work_folder+"/analysis/amazon_river_floodzone_01min_wo_ele.png",dpi=500,bbox_inches='tight')

# ---------------------------------------------------------------------------- #


## Map of High resolution (3 arc-sec maps of flooded pixels)
import gdal

amz_01 = gdal.Open(work_folder+"/analysis/dsl_amaz_3sec/max_depth/merge_amazon_01min_3sec")
amz_03 = gdal.Open(work_folder+"/analysis/dsl_amaz_3sec/max_depth/merge_amazon_03min_3sec")
amz_06 = gdal.Open(work_folder+"/analysis/dsl_amaz_3sec/max_depth/merge_amazon_06min_3sec")
amz_15 = gdal.Open(work_folder+"/analysis/dsl_amaz_3sec/max_depth/merge_amazon_15min_3sec")

amz_01_ar = np.array(amz_01.GetRasterBand(1).ReadAsArray())
amz_03_ar = np.array(amz_03.GetRasterBand(1).ReadAsArray())
amz_06_ar = np.array(amz_06.GetRasterBand(1).ReadAsArray())
amz_15_ar = np.array(amz_15.GetRasterBand(1).ReadAsArray())


amz_01_ar[amz_01_ar<0]=0 
amz_03_ar[amz_03_ar<0]=0 
amz_06_ar[amz_06_ar<0]=0  
amz_15_ar[amz_15_ar<0]=0 

amz_01_ar[amz_01_ar>0]=1
amz_03_ar[amz_03_ar>0]=2 
amz_06_ar[amz_06_ar>0]=4  
amz_15_ar[amz_15_ar>0]=8 


amz_all_ar = amz_01_ar + amz_03_ar + amz_06_ar + amz_15_ar


# # Calcluate the CSI (Flooded pixels matching)
# # zone 5
a1 = (np.count_nonzero((amz_01_ar[78*300:96*300,58*300:70*300]+amz_03_ar[78*300:96*300,58*300:70*300])==3)/np.count_nonzero((amz_01_ar[78*300:96*300,58*300:70*300]+amz_03_ar[78*300:96*300,58*300:70*300])))
b1 = (np.count_nonzero((amz_01_ar[78*300:96*300,58*300:70*300]+amz_06_ar[78*300:96*300,58*300:70*300])==5)/np.count_nonzero((amz_01_ar[78*300:96*300,58*300:70*300]+amz_06_ar[78*300:96*300,58*300:70*300])))
c1 = (np.count_nonzero((amz_01_ar[78*300:96*300,58*300:70*300]+amz_15_ar[78*300:96*300,58*300:70*300])==9)/np.count_nonzero((amz_01_ar[78*300:96*300,58*300:70*300]+amz_15_ar[78*300:96*300,58*300:70*300])))
d1 = (np.count_nonzero((amz_03_ar[78*300:96*300,58*300:70*300]+amz_06_ar[78*300:96*300,58*300:70*300])==6)/np.count_nonzero((amz_03_ar[78*300:96*300,58*300:70*300]+amz_06_ar[78*300:96*300,58*300:70*300])))
e1 = (np.count_nonzero((amz_03_ar[78*300:96*300,58*300:70*300]+amz_15_ar[78*300:96*300,58*300:70*300])==10)/np.count_nonzero((amz_03_ar[78*300:96*300,58*300:70*300]+amz_15_ar[78*300:96*300,58*300:70*300])))
f1 = (np.count_nonzero((amz_06_ar[78*300:96*300,58*300:70*300]+amz_15_ar[78*300:96*300,58*300:70*300])==12)/np.count_nonzero((amz_06_ar[78*300:96*300,58*300:70*300]+amz_15_ar[78*300:96*300,58*300:70*300])))

## Zone 2
a2 = (np.count_nonzero((amz_01_ar[52*300:58*300,102*300:116*300]+amz_03_ar[52*300:58*300,102*300:116*300])==3)/np.count_nonzero((amz_01_ar[52*300:58*300,102*300:116*300]+amz_03_ar[52*300:58*300,102*300:116*300])))
b2 = (np.count_nonzero((amz_01_ar[52*300:58*300,102*300:116*300]+amz_06_ar[52*300:58*300,102*300:116*300])==5)/np.count_nonzero((amz_01_ar[52*300:58*300,102*300:116*300]+amz_06_ar[52*300:58*300,102*300:116*300])))
c2 = (np.count_nonzero((amz_01_ar[52*300:58*300,102*300:116*300]+amz_15_ar[52*300:58*300,102*300:116*300])==9)/np.count_nonzero((amz_01_ar[52*300:58*300,102*300:116*300]+amz_15_ar[52*300:58*300,102*300:116*300])))
d2 = (np.count_nonzero((amz_03_ar[52*300:58*300,102*300:116*300]+amz_06_ar[52*300:58*300,102*300:116*300])==6)/np.count_nonzero((amz_03_ar[52*300:58*300,102*300:116*300]+amz_06_ar[52*300:58*300,102*300:116*300])))
e2 = (np.count_nonzero((amz_03_ar[52*300:58*300,102*300:116*300]+amz_15_ar[52*300:58*300,102*300:116*300])==10)/np.count_nonzero((amz_03_ar[52*300:58*300,102*300:116*300]+amz_15_ar[52*300:58*300,102*300:116*300])))
f2 = (np.count_nonzero((amz_06_ar[52*300:58*300,102*300:116*300]+amz_15_ar[52*300:58*300,102*300:116*300])==12)/np.count_nonzero((amz_06_ar[52*300:58*300,102*300:116*300]+amz_15_ar[52*300:58*300,102*300:116*300])))


## Zone 4
a3 = (np.count_nonzero((amz_01_ar[84*300:94*300,74*300:88*300]+amz_03_ar[84*300:94*300,74*300:88*300])==3)/np.count_nonzero((amz_01_ar[84*300:94*300,74*300:88*300]+amz_03_ar[84*300:94*300,74*300:88*300])))
b3 = (np.count_nonzero((amz_01_ar[84*300:94*300,74*300:88*300]+amz_06_ar[84*300:94*300,74*300:88*300])==5)/np.count_nonzero((amz_01_ar[84*300:94*300,74*300:88*300]+amz_06_ar[84*300:94*300,74*300:88*300])))
c3 = (np.count_nonzero((amz_01_ar[84*300:94*300,74*300:88*300]+amz_15_ar[84*300:94*300,74*300:88*300])==9)/np.count_nonzero((amz_01_ar[84*300:94*300,74*300:88*300]+amz_15_ar[84*300:94*300,74*300:88*300])))
d3 = (np.count_nonzero((amz_03_ar[84*300:94*300,74*300:88*300]+amz_06_ar[84*300:94*300,74*300:88*300])==6)/np.count_nonzero((amz_03_ar[84*300:94*300,74*300:88*300]+amz_06_ar[84*300:94*300,74*300:88*300])))
e3 = (np.count_nonzero((amz_03_ar[84*300:94*300,74*300:88*300]+amz_15_ar[84*300:94*300,74*300:88*300])==10)/np.count_nonzero((amz_03_ar[84*300:94*300,74*300:88*300]+amz_15_ar[84*300:94*300,74*300:88*300])))
f3 = (np.count_nonzero((amz_06_ar[84*300:94*300,74*300:88*300]+amz_15_ar[84*300:94*300,74*300:88*300])==12)/np.count_nonzero((amz_06_ar[84*300:94*300,74*300:88*300]+amz_15_ar[84*300:94*300,74*300:88*300])))

## Zone 3
a4 = (np.count_nonzero((amz_01_ar[38*300:44*300,94*300:108*300]+amz_03_ar[38*300:44*300,94*300:108*300])==3)/np.count_nonzero((amz_01_ar[38*300:44*300,94*300:108*300]+amz_03_ar[38*300:44*300,94*300:108*300])))
b4 = (np.count_nonzero((amz_01_ar[38*300:44*300,94*300:108*300]+amz_06_ar[38*300:44*300,94*300:108*300])==5)/np.count_nonzero((amz_01_ar[38*300:44*300,94*300:108*300]+amz_06_ar[38*300:44*300,94*300:108*300])))
c4 = (np.count_nonzero((amz_01_ar[38*300:44*300,94*300:108*300]+amz_15_ar[38*300:44*300,94*300:108*300])==9)/np.count_nonzero((amz_01_ar[38*300:44*300,94*300:108*300]+amz_15_ar[38*300:44*300,94*300:108*300])))
d4 = (np.count_nonzero((amz_03_ar[38*300:44*300,94*300:108*300]+amz_06_ar[38*300:44*300,94*300:108*300])==6)/np.count_nonzero((amz_03_ar[38*300:44*300,94*300:108*300]+amz_06_ar[38*300:44*300,94*300:108*300])))
e4 = (np.count_nonzero((amz_03_ar[38*300:44*300,94*300:108*300]+amz_15_ar[38*300:44*300,94*300:108*300])==10)/np.count_nonzero((amz_03_ar[38*300:44*300,94*300:108*300]+amz_15_ar[38*300:44*300,94*300:108*300])))
f4 = (np.count_nonzero((amz_06_ar[38*300:44*300,94*300:108*300]+amz_15_ar[38*300:44*300,94*300:108*300])==12)/np.count_nonzero((amz_06_ar[38*300:44*300,94*300:108*300]+amz_15_ar[38*300:44*300,94*300:108*300])))


## Zone 1
a5 = (np.count_nonzero((amz_01_ar[50*300:60*300,110*300:126*300]+amz_03_ar[50*300:60*300,110*300:126*300])==3)/np.count_nonzero((amz_01_ar[50*300:60*300,110*300:126*300]+amz_03_ar[50*300:60*300,110*300:126*300])))
b5 = (np.count_nonzero((amz_01_ar[50*300:60*300,110*300:126*300]+amz_06_ar[50*300:60*300,110*300:126*300])==5)/np.count_nonzero((amz_01_ar[50*300:60*300,110*300:126*300]+amz_06_ar[50*300:60*300,110*300:126*300])))
c5 = (np.count_nonzero((amz_01_ar[50*300:60*300,110*300:126*300]+amz_15_ar[50*300:60*300,110*300:126*300])==9)/np.count_nonzero((amz_01_ar[50*300:60*300,110*300:126*300]+amz_15_ar[50*300:60*300,110*300:126*300])))
d5 = (np.count_nonzero((amz_03_ar[50*300:60*300,110*300:126*300]+amz_06_ar[50*300:60*300,110*300:126*300])==6)/np.count_nonzero((amz_03_ar[50*300:60*300,110*300:126*300]+amz_06_ar[50*300:60*300,110*300:126*300])))
e5 = (np.count_nonzero((amz_03_ar[50*300:60*300,110*300:126*300]+amz_15_ar[50*300:60*300,110*300:126*300])==10)/np.count_nonzero((amz_03_ar[50*300:60*300,110*300:126*300]+amz_15_ar[50*300:60*300,110*300:126*300])))
f5 = (np.count_nonzero((amz_06_ar[50*300:60*300,110*300:126*300]+amz_15_ar[50*300:60*300,110*300:126*300])==12)/np.count_nonzero((amz_06_ar[50*300:60*300,110*300:126*300]+amz_15_ar[50*300:60*300,110*300:126*300])))

a = [a1,a2,a3,a4,a5]
b = [b1,b2,b3,b4,b5]
c = [c1,c2,c3,c4,c5]
d = [d1,d2,d3,d4,d5]
e = [e1,e2,e3,e4,e5]
f = [f1,f2,f3,f4,f5]

# ---------------------------- save the CSI value ---------------------------- #

import os
## Save as a CSV File
mypath = work_folder+"/analysis"
filename = 'floodzones_csi.csv'
file_path = os.path.join(mypath, filename)
with open(file_path,'ab') as f1:
    np.savetxt(f1,np.column_stack((a,b,c,d,e,f)),delimiter=',',header="01&03,01&06,01&15,03&06,03&15,06&15", comments='')
f1.close()


# # -------------- FLood Extent Plots --------------- # #


# -------------- Zone 1 and Zone 3--------------------------#

# zone 1 and zone 3 only for 1 and 6min
amz_01_06_ar = amz_01_ar + amz_06_ar
## Zone 1
## Lat=-2.5 to -5.0 Lon= -62.5 to -58.5
## 50:60,110:126
fig, ax = plt.subplots(2,1,figsize=(16,19))
plt.subplots_adjust(hspace=0.002)
cx0 = -62.5
cx1 = -58.5
cy0 = -2.5
cy1 = -5.0
norm1 = np.array([0,1,4,5,6])
im = ax[0].imshow(amz_01_06_ar[50*300:60*300,110*300:126*300],cmap='terrain_r',norm=colors.BoundaryNorm(norm1,256),extent=[cx0,cx1,cy0,cy1])
cbar1 = plt.colorbar(im, ticks=(norm1[0:-1]+norm1[1::])/2, orientation='vertical',extend='both',fraction=0.03, pad=0.02,ax=ax[0])
cbar1.ax.set_yticklabels(['No Flooding','1min','6min','1min & 6min'], fontsize=14)
cbar1.set_label("Resolution (min)",fontsize=16,rotation=90)
ax[0].set_title("(a) Zone 1 (CSI=0.91)",loc='left',fontsize=20,weight='bold')
ax[0].tick_params(axis='both', which='both', labelsize=14)  # Set tick size for latitude and longitude

## Zone 3
## Lat=0.5 to -1.0 Lon= -66.5 to -63.0
## 38:44,94:108
cx0 = -66.5
cx1 = -63.0
cy0 = -1.0
cy1 = 0.5
im = ax[1].imshow(amz_01_06_ar[38*300:44*300,94*300:108*300],cmap='terrain_r',norm=colors.BoundaryNorm(norm1,256),extent=[cx0,cx1,cy0,cy1])
cbar2 = plt.colorbar(im, ticks=(norm1[0:-1]+norm1[1::])/2, orientation='vertical',extend='both',fraction=0.02, pad=0.02,ax=ax[1])
cbar2.ax.set_yticklabels(['No Flooding','1min','6min','1min & 6min'], fontsize=14)
cbar2.set_label("Resolution (min)",fontsize=16,rotation=90)
ax[1].set_title("(b) Zone 3 (CSI=0.59)",loc='left',fontsize=20,weight='bold')
ax[1].tick_params(axis='both', which='both', labelsize=14)  # Set tick size for latitude and longitude
plt.subplots_adjust(hspace=0.01)
plt.savefig(work_folder+'/analysis/zone_1_zone3_3sec_1_6_min.jpg',dpi=300,bbox_inches='tight')


## All resolutions
## Zone 1
## Lat=-2.5 to -5.0 Lon= -62.5 to -58.5
## 50:60,110:126
fig, ax = plt.subplots(2,1,figsize=(16,20))
plt.subplots_adjust(hspace=0.002)
cx0 = -62.5
cx1 = -58.5
cy0 = -2.5
cy1 = -5.0
norm1 = np.arange(0,17,1)
im = ax[0].imshow(amz_all_ar[50*300:60*300,110*300:126*300],cmap='terrain_r',norm=colors.BoundaryNorm(norm1,256),extent=[cx0,cx1,cy0,cy1])
cbar1 = plt.colorbar(im, ticks=norm1[0:-1]+0.5, orientation='vertical',extend='both',fraction=0.03, pad=0.02,ax=ax[0])
cbar1.ax.set_yticklabels(['No Flooding','01','03','01 & 03','06','01 & 06','03 & 06', '01 & 03 & 06','15',
'01 & 15','03 & 15', '01 & 03 & 15', '06 & 15', '01 & 06 & 15', '03 & 06 & 15', '01 & 03 & 06 & 15'], fontsize=8)
cbar1.set_label("Resolution (min)",fontsize=12,rotation=90)
ax[0].set_title("(a) Zone 1",loc='left',fontsize=14,weight='bold')


## Zone 3
## Lat=0.5 to -1.0 Lon= -66.5 to -63.0
## 38:44,94:108
cx0 = -66.5
cx1 = -63.0
cy0 = -1.0
cy1 = 0.5
norm1 = np.arange(0,17,1)
im = ax[1].imshow(amz_all_ar[38*300:44*300,94*300:108*300],cmap='terrain_r',norm=colors.BoundaryNorm(norm1,256),extent=[cx0,cx1,cy0,cy1])
cbar2 = plt.colorbar(im, ticks=norm1[0:-1]+0.5, orientation='vertical',extend='both',fraction=0.02, pad=0.02,ax=ax[1])
cbar2.ax.set_yticklabels(['No Flooding','01','03','01 & 03','06','01 & 06','03 & 06', '01 & 03 & 06','15',
'01 & 15','03 & 15', '01 & 03 & 15', '06 & 15', '01 & 06 & 15', '03 & 06 & 15', '01 & 03 & 06 & 15'], fontsize=8)
cbar2.set_label("Resolution (min)",fontsize=12,rotation=90)
ax[1].set_title("(b) Zone 3",loc='left',fontsize=14,weight='bold')
plt.savefig(work_folder+'/analysis/zone_1_zone3_3sec.jpg',dpi=500,bbox_inches='tight')


# ------------------------------ Seperate Zones ------------------------------ #

## Zone 1
## Lat=-2.5 to -5.0 Lon= -62.5 to -58.5
## 50:60,110:126
fig, ax = plt.subplots(figsize=(16,7))
cx0 = -62.5
cx1 = -58.5
cy0 = -2.5
cy1 = -5.0
norm1 = np.arange(0,17,1)
im = ax.imshow(amz_all_ar[50*300:60*300,110*300:126*300],cmap='terrain_r',norm=colors.BoundaryNorm(norm1,256),extent=[cx0,cx1,cy0,cy1])
cbar = plt.colorbar(im, ticks=norm1[0:-1]+0.5, orientation='vertical',extend='both',fraction=0.05, pad=0.02)
cbar.ax.set_yticklabels(['No Flooding','01','03','01 & 03','06','01 & 06','03 & 06', '01 & 03 & 06','15',
'01 & 15','03 & 15', '01 & 03 & 15', '06 & 15', '01 & 06 & 15', '03 & 06 & 15', '01 & 03 & 06 & 15'], fontsize=8)
cbar.set_label("Resolution (min)",fontsize=12,rotation=90)
ax.set_title("    Zone 1",loc='left',fontsize=14,weight='bold')
plt.savefig(work_folder+'/analysis/zone1_3sec.jpg',dpi=500,bbox_inches='tight')


## Zone 2
## # Lat=-3 to -4.5 lon= -64.5 to -61
## 52:58,102:116
fig, ax = plt.subplots(figsize=(16,7))
cx0 = -64.5
cx1 = -61.0
cy0 = -3.0
cy1 = -4.5
norm1 = np.arange(0,17,1)
im = ax.imshow(amz_all_ar[52*300:58*300,102*300:116*300],cmap='terrain_r',norm=colors.BoundaryNorm(norm1,256),extent=[cx0,cx1,cy0,cy1])
cbar = plt.colorbar(im, ticks=norm1[0:-1]+0.5, orientation='vertical',extend='both',fraction=0.03, pad=0.02)
cbar.ax.set_yticklabels(['No Flooding','01','03','01 & 03','06','01 & 06','03 & 06', '01 & 03 & 06','15',
'01 & 15','03 & 15', '01 & 03 & 15', '06 & 15', '01 & 06 & 15', '03 & 06 & 15', '01 & 03 & 06 & 15'], fontsize=8)
cbar.set_label("Resolution (min)",fontsize=12,rotation=90)
ax.set_title("    Zone 2",loc='left',fontsize=14,weight='bold')
plt.savefig(work_folder+'/analysis/zone2_3sec.jpg',dpi=500,bbox_inches='tight')


## Zone 3
## Lat=0.5 to -1.0 Lon= -66.5 to -63.0
## 38:44,94:108
fig, ax = plt.subplots(figsize=(16,7))
cx0 = -66.5
cx1 = -63.0
cy0 = -1.0
cy1 = 0.5
norm1 = np.arange(0,17,1)
im = ax.imshow(amz_all_ar[38*300:44*300,94*300:108*300],cmap='terrain_r',norm=colors.BoundaryNorm(norm1,256),extent=[cx0,cx1,cy0,cy1])
cbar = plt.colorbar(im, ticks=norm1[0:-1]+0.5, orientation='vertical',extend='both',fraction=0.05, pad=0.02)
cbar.ax.set_yticklabels(['No Flooding','01','03','01 & 03','06','01 & 06','03 & 06', '01 & 03 & 06','15',
'01 & 15','03 & 15', '01 & 03 & 15', '06 & 15', '01 & 06 & 15', '03 & 06 & 15', '01 & 03 & 06 & 15'], fontsize=8)
cbar.set_label("Resolution (min)",fontsize=12,rotation=90)
ax.set_title("    Zone 3",loc='left',fontsize=14,weight='bold')
plt.savefig(work_folder+'/analysis/zone3_3sec.jpg',dpi=500,bbox_inches='tight')


## Zone 4
## Lat=-11 to -13.5 Lon= -71.5 to -68
## 84:94,74:88
fig, ax = plt.subplots(figsize=(16,7))
cx0 = -71.5
cx1 = -68.0
cy0 = -11.0
cy1 = -13.5
norm1 = np.arange(0,17,1)
im = ax.imshow(amz_all_ar[84*300:94*300,74*300:88*300],cmap='terrain_r',norm=colors.BoundaryNorm(norm1,256),extent=[cx0,cx1,cy0,cy1])
cbar = plt.colorbar(im, ticks=norm1[0:-1]+0.5, orientation='vertical',extend='both',fraction=0.05, pad=0.02)
cbar.ax.set_yticklabels(['No Flooding','01','03','01 & 03','06','01 & 06','03 & 06', '01 & 03 & 06','15',
'01 & 15','03 & 15', '01 & 03 & 15', '06 & 15', '01 & 06 & 15', '03 & 06 & 15', '01 & 03 & 06 & 15'], fontsize=8)
cbar.set_label("Resolution (min)",fontsize=12,rotation=90)
ax.set_title("    Zone 4",loc='left',fontsize=14,weight='bold')
plt.savefig(work_folder+'/analysis/zone4_3sec.jpg',dpi=500,bbox_inches='tight')



## Zone 5
## Lat=-9.5 to -14 lon= -75.5 to -72.5
fig, ax = plt.subplots(figsize=(16,7))
cx0 = -75.5
cx1 = -72.5
cy0 = -14
cy1 = -9.5
norm1 = np.arange(0,17,1)
im = ax.imshow(amz_all_ar[78*300:96*300,58*300:70*300],cmap='terrain_r',norm=colors.BoundaryNorm(norm1,256),extent=[cx0,cx1,cy0,cy1])
cbar = plt.colorbar(im, ticks=norm1[0:-1]+0.5, orientation='vertical',extend='both',fraction=0.03, pad=0.02)
cbar.ax.set_yticklabels(['No Flooding','01','03','01 & 03','06','01 & 06','03 & 06', '01 & 03 & 06','15',
'01 & 15','03 & 15', '01 & 03 & 15', '06 & 15', '01 & 06 & 15', '03 & 06 & 15', '01 & 03 & 06 & 15'], fontsize=8)
cbar.set_label("Resolution (min)",fontsize=12,rotation=90)
ax.set_title("    Zone 5",loc='left',fontsize=14,weight='bold')
plt.savefig(work_folder+'/analysis/zone5_3sec.jpg',dpi=500,bbox_inches='tight')



################## ------------------ Lowest Performance ------------------ ####################

## Import river discharge with bifurcation scheme
AMZRDIS01 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic/rivout2008.bin",dtype=np.float32).reshape(-1,2100,3000)
AMZRDIS03 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic/rivout2008.bin",dtype=np.float32).reshape(-1,700,1000)
AMZRDIS06 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic/rivout2008.bin",dtype=np.float32).reshape(-1,350,500)
AMZRDIS15 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic/rivout2008.bin",dtype=np.float32).reshape(-1,140,200)

# import bifurcation outflow 
AMZBF01 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic/pthout2008.bin",dtype=np.float32).reshape(-1,2100,3000)
AMZBF03 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic/pthout2008.bin",dtype=np.float32).reshape(-1,700,1000)
AMZBF06 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic/pthout2008.bin",dtype=np.float32).reshape(-1,350,500)
AMZBF15 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic/pthout2008.bin",dtype=np.float32).reshape(-1,140,200)

## Import river discharge wthout bifurcation scheme
AMZRDISWBF01 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic_nobif/rivout2008.bin",dtype=np.float32).reshape(-1,2100,3000)
AMZRDISWBF03 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic_nobif/rivout2008.bin",dtype=np.float32).reshape(-1,700,1000)
AMZRDISWBF06 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic_nobif/rivout2008.bin",dtype=np.float32).reshape(-1,350,500)
AMZRDISWBF15 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic_nobf/rivout2008.bin",dtype=np.float32).reshape(-1,140,200)

## Import discharge wthout bifurcation scheme
AMZDISWBF01 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic_nobif/fldout2008.bin",dtype=np.float32).reshape(-1,2100,3000) + AMZRDISWBF01
AMZDISWBF03 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic_nobif/fldout2008.bin",dtype=np.float32).reshape(-1,700,1000) + AMZRDISWBF03
AMZDISWBF06 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic_nobif/fldout2008.bin",dtype=np.float32).reshape(-1,350,500) + AMZRDISWBF06
AMZDISWBF15 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic_nobf/fldout2008.bin",dtype=np.float32).reshape(-1,140,200) + AMZRDISWBF15


# import without bifurcation river depth 
AMZDPHWBF01 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic_nobif/rivdph2008.bin",dtype=np.float32).reshape(-1,2100,3000)
AMZDPHWBF03 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic_nobif/rivdph2008.bin",dtype=np.float32).reshape(-1,700,1000)
AMZDPHWBF06 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic_nobif/rivdph2008.bin",dtype=np.float32).reshape(-1,350,500)
AMZDPHWBF15 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic_nobf/rivdph2008.bin",dtype=np.float32).reshape(-1,140,200)


PTHFLW01 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/01_amz_vic/pthflw2008.pth",dtype=np.float32).reshape(366,5,-1)
PTHFLW03 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/03_amz_vic/pthflw2008.pth",dtype=np.float32).reshape(366,5,-1)
PTHFLW06 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/06_amz_vic/pthflw2008.pth",dtype=np.float32).reshape(366,5,-1)  
PTHFLW15 = np.fromfile(work_folder+"/CaMa-Flood_v4/out/15_amz_vic/pthflw2008.pth",dtype=np.float32).reshape(366,5,-1)


PTHFLW01 = np.nansum(PTHFLW01,axis=1)
PTHFLW03 = np.nansum(PTHFLW03,axis=1)
PTHFLW06 = np.nansum(PTHFLW06,axis=1)
PTHFLW15 = np.nansum(PTHFLW15,axis=1)


## Takeout all the pixels with NNSE <0.3
## for discharge

Q_Low_loc_15_01 = np.argwhere(np.array(nnse_15_01_Qamz)<0.3)
nxtloc_15 = NXTXY15[:,loc_15[Q_Low_loc_15_01[:,0],0],loc_15[Q_Low_loc_15_01[:,0],1]]
nxtloc_06 = NXTXY06[:,loc_06[Q_Low_loc_15_01[:,0],0],loc_06[Q_Low_loc_15_01[:,0],1]]
nxtloc_03 = NXTXY03[:,loc_03[Q_Low_loc_15_01[:,0],0],loc_03[Q_Low_loc_15_01[:,0],1]]
nxtloc_01 = NXTXY01[:,loc_01[Q_Low_loc_15_01[:,0],0],loc_01[Q_Low_loc_15_01[:,0],1]]

## Row column locations
# Current pixels
QPXY15 = loc_15[Q_Low_loc_15_01[:,0]]
QPXY06 = loc_06[Q_Low_loc_15_01[:,0]]
QPXY03 = loc_03[Q_Low_loc_15_01[:,0]]
QPXY01 = loc_01[Q_Low_loc_15_01[:,0]]

# Next Pixels
QPNXY15 = np.column_stack((nxtloc_15[1,:]-1,nxtloc_15[0,:]-1))
QPNXY06 = np.column_stack((nxtloc_06[1,:]-1,nxtloc_06[0,:]-1))
QPNXY03 = np.column_stack((nxtloc_03[1,:]-1,nxtloc_03[0,:]-1))
QPNXY01 = np.column_stack((nxtloc_01[1,:]-1,nxtloc_01[0,:]-1))

nse_dis  = np.array(nnse_15_01_Qamz)[np.array(nnse_15_01_Qamz)<0.3]

bifprm15 = pd.read_csv(work_folder+"/CaMa-Flood_v4/map/15_amz/bifprm.txt",skiprows=1,header=None,delimiter="\s+")
bifprm06 = pd.read_csv(work_folder+"/CaMa-Flood_v4/map/06_amz/bifprm.txt",skiprows=1,header=None,delimiter="\s+")
bifprm03 = pd.read_csv(work_folder+"/CaMa-Flood_v4/map/03_amz/bifprm.txt",skiprows=1,header=None,delimiter="\s+")
bifprm01 = pd.read_csv(work_folder+"/CaMa-Flood_v4/map/01_amz/bifprm.txt",skiprows=1,header=None,delimiter="\s+")

bifprm15 = np.array(bifprm15)

LOWDISLATLON = np.column_stack((np.arange(1,QPXY15.shape[0]+1),LONLAT15[0,QPXY15[:,0],QPXY15[:,1]],LONLAT15[1,QPXY15[:,0],QPXY15[:,1]],
                                LONLAT06[0,QPXY06[:,0],QPXY06[:,1]],LONLAT06[1,QPXY06[:,0],QPXY06[:,1]],
                                LONLAT03[0,QPXY03[:,0],QPXY03[:,1]],LONLAT03[1,QPXY03[:,0],QPXY03[:,1]],
                                LONLAT01[0,QPXY01[:,0],QPXY01[:,1]],LONLAT01[1,QPXY01[:,0],QPXY01[:,1]]))

# ------------------------------- save the data ------------------------------ #
mypath = work_folder+"/analysis"
filename = 'LOWDISLATLON.cs'
file_path = os.path.join(mypath, filename)
with open(file_path,'ab') as f1:
    np.savetxt(f1,LOWDISLATLON,delimiter=',',header=("ID,LON15,LAT15,LON06,LAT06,LON03,LAT03,LON01,LAT01"), comments='')
f1.close()


# ----------------------------- Bifurcation plot ----------------------------- #
fig,ax=plt.subplots(figsize=(16,17.8),nrows=3,ncols=2,gridspec_kw={'height_ratios': [1, 1, 1], 'width_ratios': [1, 1]})

# Set the ratios using subplots_adjust
plt.subplots_adjust(hspace=0.2,wspace=0.1)
i=0;k=1;
for j in range(1):

    l, b, w, h = ax[i,j].get_position().bounds
    ax[i,j].set_position([l,b+0.06,w,h-0.06])
    ax[i,j].plot(AMZDIS01[:,QPXY01[k+j,0],QPXY01[k+j,1]],c='red',linestyle='-',alpha=0.5, label="1min BF")
    ax[i,j].plot(AMZDIS03[:,QPXY03[k+j,0],QPXY03[k+j,1]],c='yellow',linestyle='-',alpha=0.5,label="3min BF")
    ax[i,j].plot(AMZDIS06[:,QPXY06[k+j,0],QPXY06[k+j,1]],c='green',linestyle='-',alpha=0.5,label="6min BF")
    ax[i,j].plot(AMZDIS15[:,QPXY15[k+j,0],QPXY15[k+j,1]],c='blue',linestyle='-',alpha=0.5,label="15min BF")


    ax[i,j].plot(AMZDISWBF01[:,QPXY01[k+j,0],QPXY01[k+j,1]],c='red',label='1min NBF',linestyle='--',alpha=0.5)
    ax[i,j].plot(AMZDISWBF03[:,QPXY03[k+j,0],QPXY03[k+j,1]],c='yellow',label='3min NBF',linestyle='--',alpha=0.5)
    ax[i,j].plot(AMZDISWBF06[:,QPXY06[k+j,0],QPXY06[k+j,1]],c='green',label='6min NBF',linestyle='--',alpha=0.5)
    ax[i,j].plot(AMZDISWBF15[:,QPXY15[k+j,0],QPXY15[k+j,1]],c='blue',label='15min NBF',linestyle='--',alpha=0.5)
    ax[i,j].plot()

    if(j==0):
        ax[i,j].plot(PTHFLW15[:,299-1],c='blue',label='PTHFLW 15min',linestyle=':',alpha=0.5)
        ax[i,j].plot(PTHFLW06[:,2606-1],c='green',label='PTHFLW 6min',linestyle=':',alpha=0.5)

    ax[i,j].set_title("("+chr(97+j)+")"+ " at location "+ str(k+1+j),fontsize=12, loc='left',weight="bold")
    ax[i,j].set_xlabel("Days",fontsize=10)
    ax[i,j].set_ylabel("Discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=10)
    ax[i,j].grid(linestyle=':')

lines_labels = [fig.axes[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels,loc='lower center',ncol=5, bbox_to_anchor=(0.30, 0.660),borderpad=0.2,labelspacing=0.5,handletextpad=0.05,columnspacing=0.6) 
for i in range(0,3):
    for j in range(0,2):
        if(i==0 and j==0):
            continue
        else:
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])

ax[0,1].set_title("("+chr(98)+") Schematic diagram",fontsize=16, loc='left',weight="bold")
ax[1,0].set_title("("+chr(99)+") 15min",fontsize=16, loc='left',weight="bold")
ax[1,1].set_title("("+chr(100)+") 6min",fontsize=16, loc='left',weight="bold")
ax[2,0].set_title("("+chr(101)+") 3min",fontsize=16, loc='left',weight="bold")
ax[2,1].set_title("("+chr(102)+") 1min",fontsize=16, loc='left',weight="bold")

plt.savefig(work_folder+'/analysis/bifurcation_BF_2.jpg',dpi=500,bbox_inches='tight')



# ------------------------------ for location 6  backwater effect ------------------------------ #
fig,ax = plt.subplots(figsize=(22,5),nrows=1,ncols=2)
ax[0].set_title("(a) Backwater effect",fontsize=22, loc='left',weight="bold")
ax[0].set_xticks([])
ax[0].set_yticks([])

k = 5
ax[1].scatter(AMZDIS01[:,QPXY01[k,0],QPXY01[k,1]],AMZDPH01[:,QPXY01[k,0],QPXY01[k,1]],c='red',label='1min',linestyle='--',alpha=0.5)
ax[1].scatter(AMZDIS03[:,QPXY03[k,0],QPXY03[k,1]],AMZDPH03[:,QPXY03[k,0],QPXY03[k,1]],c='yellow',label='3min',linestyle='--',alpha=0.5)
ax[1].scatter(AMZDIS06[:,QPXY06[k,0],QPXY06[k,1]],AMZDPH06[:,QPXY06[k,0],QPXY06[k,1]],c='green',label='6min',linestyle='--',alpha=0.5)
ax[1].scatter(AMZDIS15[:,QPXY15[k,0],QPXY15[k,1]],AMZDPH15[:,QPXY15[k,0],QPXY15[k,1]],c='blue',label='15min',linestyle='--',alpha=0.5)

ax[1].set_title("(b) Stage-discharge curve for location 6",fontsize=22, loc='left',weight="bold")
ax[1].set_ylabel("Water depth (m)",fontsize=18)
ax[1].set_xlabel("Discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=18)
ax[1].grid(linestyle=':')
ax[1].legend(loc="lower right",fontsize=18)
plt.subplots_adjust(wspace=0.1)
plt.savefig(work_folder+'/analysis/backwater_BF_loc6.jpg',dpi=500,bbox_inches='tight')


# ----------------------- floodplain conveyance effect ----------------------- #


fig,ax = plt.subplots(figsize=(14,8),nrows=2, ncols=2)

k=11
ax[0,0].plot(AMZDIS01[:,QPXY01[k,0],QPXY01[k,1]]-AMZRDIS01[:,QPXY01[k,0],QPXY01[k,1]],c='red',linestyle='-',alpha=0.5, label="1min floodplain discharge")
ax[0,0].plot(AMZDIS03[:,QPXY03[k,0],QPXY03[k,1]]-AMZRDIS03[:,QPXY03[k,0],QPXY03[k,1]],c='yellow',linestyle='-',alpha=0.5,label="3min floodplain discharge")
ax[0,0].plot(AMZDIS06[:,QPXY06[k,0],QPXY06[k,1]]-AMZRDIS06[:,QPXY06[k,0],QPXY06[k,1]],c='green',linestyle='-',alpha=0.5,label="6min floodplain discharge")
ax[0,0].plot(AMZDIS15[:,QPXY15[k,0],QPXY15[k,1]]-AMZRDIS15[:,QPXY15[k,0],QPXY15[k,1]],c='blue',linestyle='-',alpha=0.5,label="15min floodplain discharge")

ax[0,0].plot(AMZRDIS01[:,QPXY01[k,0],QPXY01[k,1]],c='red',label='1min river channel discharge',linestyle='--',alpha=0.5)
ax[0,0].plot(AMZRDIS03[:,QPXY03[k,0],QPXY03[k,1]],c='yellow',label='3min river channel discharge',linestyle='--',alpha=0.5)
ax[0,0].plot(AMZRDIS06[:,QPXY06[k,0],QPXY06[k,1]],c='green',label='6min river channel discharge',linestyle='--',alpha=0.5)
ax[0,0].plot(AMZRDIS15[:,QPXY15[k,0],QPXY15[k,1]],c='blue',label='15min river channel discharge',linestyle='--',alpha=0.5)

ax[0,0].set_title("(a) at location 12",fontsize=16, loc='left',weight="bold")
ax[0,0].set_xlabel("Days",fontsize=10)
ax[0,0].set_ylabel("Discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=10)
ax[0,0].grid(linestyle=':')

ax[0,1].scatter(AMZRDIS01[:,QPXY01[k,0],QPXY01[k,1]],AMZDPH01[:,QPXY01[k,0],QPXY01[k,1]],c='red',s=10,alpha=0.5, label="1min river channel discharge")
ax[0,1].scatter(AMZRDIS03[:,QPXY03[k,0],QPXY03[k,1]],AMZDPH03[:,QPXY03[k,0],QPXY03[k,1]],c='yellow',s=10,alpha=0.5,label="3min river channel discharge")
ax[0,1].scatter(AMZRDIS06[:,QPXY06[k,0],QPXY06[k,1]],AMZDPH06[:,QPXY06[k,0],QPXY06[k,1]],c='green',s=10,alpha=0.5,label="6min river channel discharge")
ax[0,1].scatter(AMZRDIS15[:,QPXY15[k,0],QPXY15[k,1]],AMZDPH15[:,QPXY15[k,0],QPXY15[k,1]],c='blue',s=10,alpha=0.5,label="15min river channel discharge")
ax[0,1].grid(linestyle=':')
ax[0,1].set_title("(b) River channel discharge",fontsize=16, loc='left',weight="bold")
ax[0,1].set_xlabel("River channel discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=10)
ax[0,1].set_ylabel("Water depth (m)",fontsize=10)

ax[1,0].scatter(AMZDIS01[:,QPXY01[k,0],QPXY01[k,1]]-AMZRDIS01[:,QPXY01[k,0],QPXY01[k,1]],AMZDPH01[:,QPXY01[k,0],QPXY01[k,1]],c='red',s=10,alpha=0.5, label="1min floodplain discharge")
ax[1,0].scatter(AMZDIS03[:,QPXY03[k,0],QPXY03[k,1]]-AMZRDIS03[:,QPXY03[k,0],QPXY03[k,1]],AMZDPH03[:,QPXY03[k,0],QPXY03[k,1]],c='yellow',s=10,alpha=0.5,label="3min floodplain discharge")
ax[1,0].scatter(AMZDIS06[:,QPXY06[k,0],QPXY06[k,1]]-AMZRDIS06[:,QPXY06[k,0],QPXY06[k,1]],AMZDPH06[:,QPXY06[k,0],QPXY06[k,1]],c='green',s=10,alpha=0.5,label="6min floodplain discharge")
ax[1,0].scatter(AMZDIS15[:,QPXY15[k,0],QPXY15[k,1]]-AMZRDIS15[:,QPXY15[k,0],QPXY15[k,1]],AMZDPH15[:,QPXY15[k,0],QPXY15[k,1]],c='blue',s=10,alpha=0.5,label="15min floodplain discharge")
ax[1,0].grid(linestyle=':')
ax[1,0].set_title("(c) Floodplain discharge",fontsize=16, loc='left',weight="bold")
ax[1,0].set_xlabel("Floodplain discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=10)
ax[1,0].set_ylabel("Water depth (m)",fontsize=10)

lines_labels = [fig.axes[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels,loc='lower center',ncol=4,fontsize=10) 
plt.subplots_adjust(hspace=0.3,bottom=0.15)

ax[1,1].set_title("(d) Conceptual diagram  ",fontsize=16, loc='left',weight="bold")
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])

plt.savefig(work_folder+'/analysis/fldpln_conv_loc12.jpg',dpi=300,bbox_inches='tight')



# ------------------ Dsicharge issue plots * 3 examples each ----------------- #

fig,ax = plt.subplots(figsize=(20,14),nrows=3, ncols=3)
plt.subplots_adjust(hspace=0.5)
# fig,ax = plt.subplots(figsize=(9,7))
# bifurcation issue
i=0;k=1;
for j in range(3):
    
    ax[i,j].plot(AMZDIS01[:,QPXY01[k+j,0],QPXY01[k+j,1]],c='red',linestyle='-',alpha=0.5, label="1min BF")
    ax[i,j].plot(AMZDIS03[:,QPXY03[k+j,0],QPXY03[k+j,1]],c='yellow',linestyle='-',alpha=0.7,label="3min BF")
    ax[i,j].plot(AMZDIS06[:,QPXY06[k+j,0],QPXY06[k+j,1]],c='green',linestyle='-',alpha=0.5,label="6min BF")
    ax[i,j].plot(AMZDIS15[:,QPXY15[k+j,0],QPXY15[k+j,1]],c='blue',linestyle='-',alpha=0.5,label="15min BF")


    ax[i,j].plot(AMZDISWBF01[:,QPXY01[k+j,0],QPXY01[k+j,1]],c='red',label='1min NBF',linestyle='--',alpha=0.5)
    ax[i,j].plot(AMZDISWBF03[:,QPXY03[k+j,0],QPXY03[k+j,1]],c='yellow',label='3min NBF',linestyle='--',alpha=0.7)
    ax[i,j].plot(AMZDISWBF06[:,QPXY06[k+j,0],QPXY06[k+j,1]],c='green',label='6min NBF',linestyle='--',alpha=0.5)
    ax[i,j].plot(AMZDISWBF15[:,QPXY15[k+j,0],QPXY15[k+j,1]],c='blue',label='15min NBF',linestyle='--',alpha=0.5)
    ax[i,j].plot()

    ax[i,j].set_title("("+chr(97+j)+")"+ " at location "+ str(k+1+j),fontsize=12, loc='left',weight="bold")
    ax[i,j].set_xlabel("Days",fontsize=11)
    ax[i,j].set_ylabel("Discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=11)
    ax[i,j].grid(linestyle=':')



lines_labels = [fig.axes[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels,loc='lower center',ncol=8, bbox_to_anchor=(0.5, 0.63),fontsize=11) 

#Floodplian flow 
i=1;k=10;
for j in range(3):
    
    ax[i,j].plot(AMZDIS01[:,QPXY01[k+j,0],QPXY01[k+j,1]]-AMZRDIS01[:,QPXY01[k+j,0],QPXY01[k+j,1]],c='red',linestyle='-',alpha=0.5, label="01min floodplain discharge")
    ax[i,j].plot(AMZDIS03[:,QPXY03[k+j,0],QPXY03[k+j,1]]-AMZRDIS03[:,QPXY03[k+j,0],QPXY03[k+j,1]],c='yellow',linestyle='-',alpha=0.7,label="03min floodplain discharge")
    ax[i,j].plot(AMZDIS06[:,QPXY06[k+j,0],QPXY06[k+j,1]]-AMZRDIS06[:,QPXY06[k+j,0],QPXY06[k+j,1]],c='green',linestyle='-',alpha=0.5,label="06min floodplain discharge")
    ax[i,j].plot(AMZDIS15[:,QPXY15[k+j,0],QPXY15[k+j,1]]-AMZRDIS15[:,QPXY15[k+j,0],QPXY15[k+j,1]],c='blue',linestyle='-',alpha=0.5,label="15min floodplain discharge")


    ax[i,j].plot(AMZRDIS01[:,QPXY01[k+j,0],QPXY01[k+j,1]],c='red',label='01min river channel discharge',linestyle='--',alpha=0.5)
    ax[i,j].plot(AMZRDIS03[:,QPXY03[k+j,0],QPXY03[k+j,1]],c='yellow',label='03min river channel discharge',linestyle='--',alpha=0.7)
    ax[i,j].plot(AMZRDIS06[:,QPXY06[k+j,0],QPXY06[k+j,1]],c='green',label='06min river channel discharge',linestyle='--',alpha=0.5)
    ax[i,j].plot(AMZRDIS15[:,QPXY15[k+j,0],QPXY15[k+j,1]],c='blue',label='15min river channel discharge',linestyle='--',alpha=0.5)

    ax[i,j].set_title("("+chr(100+j)+")"+ " at location "+ str(k+1+j),fontsize=12, loc='left',weight="bold")
    ax[i,j].set_xlabel("Days",fontsize=11)
    ax[i,j].set_ylabel("Discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=11)
    ax[i,j].grid(linestyle=':')

lines_labels = [fig.axes[3].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels,loc='lower center',ncol=4, bbox_to_anchor=(0.5, 0.325),fontsize=11) 

i=2;k=1;
# Backwater effect
for j in range(3):
    ax[i,j].scatter(AMZDIS01[:,QPXY01[k+j,0],QPXY01[k+j,1]],AMZDPH01[:,QPXY01[k+j,0],QPXY01[k+j,1]],c='red',label='1min',linestyle='--',alpha=0.5)
    ax[i,j].scatter(AMZDIS03[:,QPXY03[k+j,0],QPXY03[k+j,1]],AMZDPH03[:,QPXY03[k+j,0],QPXY03[k+j,1]],c='yellow',label='3min',linestyle='--',alpha=0.5)
    ax[i,j].scatter(AMZDIS06[:,QPXY06[k+j,0],QPXY06[k+j,1]],AMZDPH06[:,QPXY06[k+j,0],QPXY06[k+j,1]],c='green',label='6min',linestyle='--',alpha=0.5)
    ax[i,j].scatter(AMZDIS15[:,QPXY15[k+j,0],QPXY15[k+j,1]],AMZDPH15[:,QPXY15[k+j,0],QPXY15[k+j,1]],c='blue',label='15min',linestyle='--',alpha=0.5)

    ax[i,j].set_title("("+chr(103+j)+")"+ " at location "+ str(k+1+j),fontsize=12, loc='left',weight="bold")
    ax[i,j].set_ylabel("Water depth (m)",fontsize=11)
    ax[i,j].set_xlabel("Discharge (m\N{SUPERSCRIPT THREE}/s)",fontsize=11)
    ax[i,j].grid(linestyle=':')

lines_labels = [fig.axes[6].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels,loc='lower center',ncol=8, bbox_to_anchor=(0.5, 0.05),fontsize=11) 

plt.savefig(work_folder+'/analysis/low_q_nnse_various.jpg',dpi=500,bbox_inches='tight')


## ---------------------------- Lowest perfomance river water depth --------------------------------------###########
## Takeout all the pixels with NNSE <0.3
## for water depth
D_Low_loc_15_01 = np.argwhere(np.array(nnse_15_01_Damz)<0.3)
nxtloc_15 = NXTXY15[:,loc_15[D_Low_loc_15_01[:,0],0],loc_15[D_Low_loc_15_01[:,0],1]]
nxtloc_06 = NXTXY06[:,loc_06[D_Low_loc_15_01[:,0],0],loc_06[D_Low_loc_15_01[:,0],1]]
nxtloc_03 = NXTXY03[:,loc_03[D_Low_loc_15_01[:,0],0],loc_03[D_Low_loc_15_01[:,0],1]]
nxtloc_01 = NXTXY01[:,loc_01[D_Low_loc_15_01[:,0],0],loc_01[D_Low_loc_15_01[:,0],1]]

## Row column locations
# Current pixels
DPXY15 = loc_15[D_Low_loc_15_01[:,0]]
DPXY06 = loc_06[D_Low_loc_15_01[:,0]]
DPXY03 = loc_03[D_Low_loc_15_01[:,0]]
DPXY01 = loc_01[D_Low_loc_15_01[:,0]]

# Next Pixels
PNXY15 = np.column_stack((nxtloc_15[1,:]-1,nxtloc_15[0,:]-1))
PNXY06 = np.column_stack((nxtloc_06[1,:]-1,nxtloc_06[0,:]-1))
PNXY03 = np.column_stack((nxtloc_03[1,:]-1,nxtloc_03[0,:]-1))
PNXY01 = np.column_stack((nxtloc_01[1,:]-1,nxtloc_01[0,:]-1))


LOWRWDLATLON = np.column_stack((np.arange(1,DPXY15.shape[0]+1),LONLAT15[0,DPXY15[:,0],DPXY15[:,1]],LONLAT15[1,DPXY15[:,0],DPXY15[:,1]],
                                LONLAT06[0,DPXY06[:,0],DPXY06[:,1]],LONLAT06[1,DPXY06[:,0],DPXY06[:,1]],
                                LONLAT03[0,DPXY03[:,0],DPXY03[:,1]],LONLAT03[1,DPXY03[:,0],DPXY03[:,1]],
                                LONLAT01[0,DPXY01[:,0],DPXY01[:,1]],LONLAT01[1,DPXY01[:,0],DPXY01[:,1]]))

# ------------------------------- save the data ------------------------------ #
mypath = work_folder+'/analysis"
filename = 'LOWRWDLATLON.csv'
file_path = os.path.join(mypath, filename)
with open(file_path,'ab') as f1:
    np.savetxt(f1,LOWRWDLATLON,delimiter=',',header=("ID,LON15,LAT15,LON06,LAT06,LON03,LAT03,LON01,LAT01"), comments='')
f1.close()


# ----------------------------- 15min 6min and 1min all three ----------------------------- #
#6min is ahead of the 3975m so add this value to 1min and 15min
pos = 586
loc = [loc_15[pos]]  ## location of pixel with highest value of river depth (river mouth)
rowloc = []
colloc = []
MWSE15 = []
rowloc.append(loc[0][0])
colloc.append(loc[0][1])
MWSE15.append(np.nanmean(AMZWSE15[:,loc[0][0],loc[0][1]]))
while(UPARA15[loc[0][0],loc[0][1]]>0):
    temploc = np.argwhere((NXTXY15[1]==loc[0][0]+1)&(NXTXY15[0]==loc[0][1]+1))
    if (temploc.shape[0]>0):
        loc = np.argwhere(max(UPARA15[temploc[:,0],temploc[:,1]])==UPARA15[temploc[:,0],temploc[:,1]])
        loc = [[temploc[loc[0][0],0],temploc[loc[0][0],1]]]
        rowloc.append(loc[0][0])
        colloc.append(loc[0][1])
        MWSE15.append(np.nanmean(AMZWSE15[:,loc[0][0],loc[0][1]]))
    else: 
        break
MRIVLOC15 = np.column_stack((rowloc,colloc))     ## main river channel downstream to upstream
MRIVLEN15 = RIVLEN15[MRIVLOC15[:,0],MRIVLOC15[:,1]]
# MRIVLEN15 = np.insert(MRIVLEN15,0,0)
MRIVLEN15 = np.insert(MRIVLEN15,0,3975)
MRIVLEN15 = np.cumsum(MRIVLEN15)
MRIVLEN15 = MRIVLEN15[0:-1]
MRIVWTH15 = RIVWTH15[MRIVLOC15[:,0],MRIVLOC15[:,1]]
NNSE_15_1 = nnse_15_01_Damz[pos:pos+MRIVWTH15.shape[0]]             # 1min-15min nnse 

loc = [loc_06[pos]]  ## location of pixel with highest value of river depth (river mouth)
rowloc = []
colloc = []
MWSE06 = []
rowloc.append(loc[0][0])
colloc.append(loc[0][1])
MWSE06.append(np.nanmean(AMZWSE06[:,loc[0][0],loc[0][1]]))
while(UPARA06[loc[0][0],loc[0][1]]>=0):
    temploc = np.argwhere((NXTXY06[1]==loc[0][0]+1)&(NXTXY06[0]==loc[0][1]+1))
    if (temploc.shape[0]>0):
        loc = np.argwhere(max(UPARA06[temploc[:,0],temploc[:,1]])==UPARA06[temploc[:,0],temploc[:,1]])
        loc = [[temploc[loc[0][0],0],temploc[loc[0][0],1]]]
        rowloc.append(loc[0][0])
        colloc.append(loc[0][1])
        MWSE06.append(np.nanmean(AMZWSE06[:,loc[0][0],loc[0][1]]))
    else: 
        break
MRIVLOC06 = np.column_stack((rowloc,colloc))     ## main river channel downstream to upstream
MRIVLEN06 = RIVLEN06[MRIVLOC06[:,0],MRIVLOC06[:,1]]
MRIVLEN06 = np.insert(MRIVLEN06,0,0)
MRIVLEN06 = np.cumsum(MRIVLEN06)
MRIVLEN06 = MRIVLEN06[0:-1]
MRIVWTH06 = RIVWTH06[MRIVLOC06[:,0],MRIVLOC06[:,1]]

NNSE_06_1 = nnse_06_01_Damz[pos:pos+MRIVWTH15.shape[0]]             # 1min-6min nnse

MRIVLOC06_P = loc_06[pos:pos+MRIVWTH15.shape[0]]
MRIVWTH06_P = []
MRIVLEN06_P = []

for i in range(0,MRIVLOC06_P.shape[0]):
    pos1 = np.argwhere((MRIVLOC06_P[i,0]==MRIVLOC06[:,0])& (MRIVLOC06_P[i,1]==MRIVLOC06[:,1]))
    if (pos1.size>0):
        MRIVLEN06_P.append(MRIVLEN06[int(pos1[0,0])])
        MRIVWTH06_P.append(MRIVWTH06[int(pos1[0,0])])


loc = [loc_01[pos]]  ## location of pixel with highest value of river depth (river mouth)
rowloc = []
colloc = []
MWSE01 = []
rowloc.append(loc[0][0])
colloc.append(loc[0][1])
MWSE01.append(np.nanmean(AMZWSE01[:,loc[0][0],loc[0][1]]))
while(UPARA01[loc[0][0],loc[0][1]]>0):
    temploc = np.argwhere((NXTXY01[1]==loc[0][0]+1)&(NXTXY01[0]==loc[0][1]+1))
    if (temploc.shape[0]>0):
        loc = np.argwhere(max(UPARA01[temploc[:,0],temploc[:,1]])==UPARA01[temploc[:,0],temploc[:,1]])
        loc = [[temploc[loc[0][0],0],temploc[loc[0][0],1]]]
        rowloc.append(loc[0][0])
        colloc.append(loc[0][1])
        MWSE01.append(np.nanmean(AMZWSE01[:,loc[0][0],loc[0][1]]))
    else: 
        break
MRIVLOC01 = np.column_stack((rowloc,colloc))     ## main river channel downstream to upstream
MRIVLEN01 = RIVLEN01[MRIVLOC01[:,0],MRIVLOC01[:,1]]
MRIVLEN01 = np.cumsum(MRIVLEN01)
MRIVLEN01 = np.insert(MRIVLEN01,0,3975)
MRIVLEN01 = MRIVLEN01[0:-1]
MRIVWTH01 = RIVWTH01[MRIVLOC01[:,0],MRIVLOC01[:,1]]

# ----------------------------- Plot elevation and WSE ----------------------------- #

## Plot elevation and WSE
fig, ax2 = plt.subplots(figsize=(16,8),nrows=1,ncols=1)
ax2.plot(MRIVLEN15/1000,MRIVWTH15[0:],'o-',color='royalblue',markersize=2,label="15 min",alpha=0.7,linewidth=0.8)
ax2.plot(MRIVLEN06/1000,MRIVWTH06[0:],'o-',color='lime',markersize=0,label="6 min",alpha=0.7,linewidth=0.8)
ax2.plot(MRIVLEN01/1000,MRIVWTH01[0:],'o-',color='coral',markersize=0,label="1 min",alpha=0.7,linewidth=0.8)

for i in range(MRIVLEN15.shape[0]-1):
    if((MRIVLEN15[i]/1000>=500)&(MRIVLEN15[i]/1000<=1600)):
        ax2.text(MRIVLEN15[i]/1000,MRIVWTH15[i]+10,str(np.round(NNSE_15_1[i],2)),color='blue',fontsize=8)
        ax2.text(MRIVLEN06_P[i]/1000,MRIVWTH06_P[i]-5,str(np.round(NNSE_06_1[i],2)),color='green',fontsize=8,ha='right')
        ax2.scatter(MRIVLEN06_P[i]/1000,MRIVWTH06_P[i],s=2,color='green')
ax2.set_ylabel("River width (m)",fontsize=18)
ax2.legend(title="River width", loc='upper right', facecolor=(255/256,255/256,255/256), bbox_to_anchor=(1.00,1), fontsize=10, title_fontsize=10)
ax2.set_xlim(500,1600)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylim(0,3500)
ax=ax2.twinx()
ax.plot(MRIVLEN15/1000,MWSE15,'--',color='blue',markersize=5,label="15 min",alpha=1,linewidth=0.8)
ax.plot(MRIVLEN06/1000,MWSE06,'--',color='green',markersize=3,label="6 min",alpha=1,linewidth=0.8)
ax.plot(MRIVLEN01/1000,MWSE01,'--',color='red',markersize=3,label="1 min",alpha=1,linewidth=0.8)
ax.set_ylim(0,200)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylabel("Mean WSE (m)",fontsize=18)
ax.legend(title="Mean WSE ",loc='upper right',facecolor=(255/256,255/256,255/256),bbox_to_anchor=(0.91,1), fontsize=10, title_fontsize=10)
ax2.set_xlabel("<----- U/S    Distance from tributary mouth (virtual station 587 )  D/S ------> ",fontsize=18)
ax2.invert_xaxis()
ax.grid(color='k', linestyle=':', linewidth=0.2, which='both')
ax2.grid(color='k', linestyle=':', linewidth=0.2, which='both',axis='x')
ax2.set_title("(a) Water depth bad perfomance for location 7 & 8",loc='left', fontsize=22, weight="bold")
plt.savefig(work_folder+'/analysis/low_rwd_loc7_1_6_15min.jpg',dpi=500,bbox_inches='tight')


# --------------- spatial distrbution of various bad locations --------------- #
# For Meridians and Parallels (Subbasin Boundaries)
def getGridLines(dat, east, west, north, south, nx, ny):
    lonsize = float(east - west) / nx
    latsize = float(south - north) / ny

    # Get meridians
    lons, lats = [], []
    lats_north = np.linspace(north, south, ny + 1)[:-1]
    for ix in range(nx - 1):
        lats_inter = copy.copy(lats_north)
        lats_inter[dat[:, ix] == dat[:, ix + 1]] = np.nan
        lats_this = np.r_[np.c_[lats_north, lats_inter].reshape(-1), south]
        lons_this = np.ones((ny * 2 + 1)) * (west + (ix + 1) * lonsize)
        lons.append(np.r_[lons_this, np.nan])
        lats.append(np.r_[lats_this, np.nan])
    meridians = (np.array(lons).reshape(-1), np.array(lats).reshape(-1))

    # Get parallels
    lons, lats = [], []
    lons_west = np.linspace(west, east, nx + 1)[:-1]
    for iy in range(ny - 1):
        lons_inter = copy.copy(lons_west)
        lons_inter[dat[iy, :] == dat[iy + 1, :]] = np.nan
        lons_this = np.r_[np.c_[lons_west, lons_inter].reshape(-1), east]
        lats_this = np.ones((nx * 2 + 1)) * (north + (iy + 1) * latsize)
        lons.append(np.r_[lons_this, np.nan])
        lats.append(np.r_[lats_this, np.nan])
    parallels = (np.array(lons).reshape(-1), np.array(lats).reshape(-1))

    return meridians, parallels


#%% Lat lon of the basin data
# 15 min
FLOC = np.argwhere((BSN15==1)&(NXTXY15[0]!=-9))  ## location of pixel with highest value of river depth (river mouth)
FLON15 = LONLAT15[0,FLOC[:,0],FLOC[:,1]]
FLAT15 = LONLAT15[1,FLOC[:,0],FLOC[:,1]]
TLOCROW15 = NXTXY15[1,FLOC[:,0],FLOC[:,1]]-1
TLOCCOL15 = NXTXY15[0,FLOC[:,0],FLOC[:,1]]-1
TLON15 = LONLAT15[0,TLOCROW15,TLOCCOL15]
TLAT15 = LONLAT15[1,TLOCROW15,TLOCCOL15]
RIVLATLON15 = np.column_stack((FLON15,FLAT15,TLON15,TLAT15))


lwidth = np.log10(RIVWTH15[TLOCROW15,TLOCCOL15])/2

WEST, EAST, SOUTH, NORTH = -80, -49, -22, 7

# request = cimgt.OSM()

import random as rm
from itertools import cycle

ha_val = ['left','right']
va_val_q=['top','bottom']
va_val_d=['bottom','top']
ha_cycle = cycle(ha_val)
va_cycle_q = cycle(va_val_q)
va_cycle_d = cycle(va_val_d)

# Set up Cartopy map
fig, ax = plt.subplots(figsize=(18, 18), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([WEST, EAST, SOUTH, NORTH], crs=ccrs.PlateCarree())
# Draw coastlines and fill continents
ax.add_feature(cfeature.COASTLINE, edgecolor='black')
ax.add_feature(cfeature.LAND, facecolor='white')

# Draw parallels and meridians
parallels = np.arange(SOUTH, NORTH, 5)
meridians = np.arange(WEST, EAST, 5)

gl = ax.gridlines(draw_labels=True, xlocs=meridians, ylocs=parallels, color='gray', linewidth=0.1, crs=ccrs.PlateCarree())
gl.xlabel_style = {'fontsize': 16}
gl.ylabel_style = {'fontsize': 16}
gl.bottom_labels = False
gl.right_labels = False

# Define the extent for the elevation plot
cx0, cx1 = WEST, EAST
cy0, cy1= NORTH, SOUTH

BSN = BSN01.copy()
BSN[BSN!=1]=0
# Plot basin boundaries (replace `meridians1` and `parallels1` with your actual boundary data)
meridians1, parallels1 = getGridLines(BSN, E, W, N, S, np.int32((E-W)/csize01), np.int32((N-S)/csize01))
ax.plot(meridians1[0], meridians1[1], linewidth=1, color='k', alpha=0.3)
ax.plot(parallels1[0], parallels1[1], linewidth=1, color='k', alpha=0.3)


# Transform river coordinates
x, y = RIVLATLON15[:,0], RIVLATLON15[:,1]
x1, y1 = RIVLATLON15[:,2], RIVLATLON15[:,3]
pts = np.c_[x, y, x1, y1].reshape(len(x1), 2, 2)
ax.add_collection(LineCollection(pts, color="gray", label="River network", alpha=0.6, linewidth=lwidth))

## Plot discharge bad locations 

#1 Major BF: 1, 17, 24-26, 28-29
ax.scatter(LONLAT15[0,QPXY15[0,0],QPXY15[0,1]],LONLAT15[1,QPXY15[0,0],QPXY15[0,1]],s=90,marker='o',facecolor='crimson',edgecolors='crimson',linewidths=0.01,alpha=0.7,label='Discharge bifurcation')
k=1+0
ax.text(LONLAT15[0,QPXY15[0,0],QPXY15[0,1]],LONLAT15[1,QPXY15[0,0],QPXY15[0,1]],s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle_q))

ax.scatter(LONLAT15[0,QPXY15[16,0],QPXY15[16,1]],LONLAT15[1,QPXY15[16,0],QPXY15[16,1]],s=90,marker='o',facecolor='crimson',edgecolors='crimson',linewidths=0.01,alpha=0.7)
k=1+16
ax.text(LONLAT15[0,QPXY15[16,0],QPXY15[16,1]],LONLAT15[1,QPXY15[16,0],QPXY15[16,1]],s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle_q))

ax.scatter(LONLAT15[0,QPXY15[23:26,0],QPXY15[23:26,1]],LONLAT15[1,QPXY15[23:26,0],QPXY15[23:26,1]],s=90,marker='o',facecolor='crimson',edgecolors='crimson',linewidths=0.01,alpha=0.7)
k=1+23
for i,j in zip(LONLAT15[0,QPXY15[23:26,0],QPXY15[23:26,1]],LONLAT15[1,QPXY15[23:26,0],QPXY15[23:26,1]]):
    ax.text(i,j,s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle_q))
    k=k+1

ax.scatter(LONLAT15[0,QPXY15[27:29,0],QPXY15[27:29,1]],LONLAT15[1,QPXY15[27:29,0],QPXY15[27:29,1]],s=90,marker='o',facecolor='crimson',edgecolors='crimson',linewidths=0.01,alpha=0.7)
k=1+27
for i,j in zip(LONLAT15[0,QPXY15[27:29,0],QPXY15[27:29,1]],LONLAT15[1,QPXY15[27:29,0],QPXY15[27:29,1]]):
    ax.text(i,j,s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle_q))
    k=k+1


#2 BF and BW 2-9, 18, 26, 
ax.scatter(LONLAT15[0,QPXY15[1:9,0],QPXY15[1:9,1]],LONLAT15[1,QPXY15[1:9,0],QPXY15[1:9,1]],s=90,marker='o',facecolor='teal',edgecolors='teal',linewidths=0.01,alpha=0.7,label='Discharge bifurcation & backwater')
k=1+1
for i,j in zip(LONLAT15[0,QPXY15[1:9,0],QPXY15[1:9,1]],LONLAT15[1,QPXY15[1:9,0],QPXY15[1:9,1]]):
    ax.text(i,j,s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle_q))
    k=k+1

ax.scatter(LONLAT15[0,QPXY15[17,0],QPXY15[17,1]],LONLAT15[1,QPXY15[17,0],QPXY15[17,1]],s=90,marker='o',facecolor='teal',edgecolors='teal',linewidths=0.01,alpha=0.7)
k=1+17
ax.text(LONLAT15[0,QPXY15[17,0],QPXY15[17,1]],LONLAT15[1,QPXY15[17,0],QPXY15[17,1]],s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle_q))

ax.scatter(LONLAT15[0,QPXY15[25,0],QPXY15[25,1]],LONLAT15[1,QPXY15[25,0],QPXY15[25,1]],s=90,marker='o',facecolor='teal',edgecolors='teal',linewidths=0.01,alpha=0.7)
k=1+25
ax.text(LONLAT15[0,QPXY15[25,0],QPXY15[25,1]],LONLAT15[1,QPXY15[25,0],QPXY15[25,1]],s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle_q))


ha_val = ['right','left']
va_val=['top','bottom']
ha_cycle = cycle(ha_val)
va_cycle = cycle(va_val)

ax.scatter(LONLAT15[0,QPXY15[9:16,0],QPXY15[9:16,1]],LONLAT15[1,QPXY15[9:16,0],QPXY15[9:16,1]],s=90,marker='o',facecolor='gold',edgecolors='gold',linewidths=0.01,alpha=0.7,label='Discharge floodplain conveyance')
k=1+9
for i,j in zip(LONLAT15[0,QPXY15[9:16,0],QPXY15[9:16,1]],LONLAT15[1,QPXY15[9:16,0],QPXY15[9:16,1]]):
    ax.text(i,j,s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle))
    k=k+1

ha_val = ['right','left']
va_val=['bottom','top']
ha_cycle = cycle(ha_val)
va_cycle = cycle(va_val)
ax.scatter(LONLAT15[0,QPXY15[18:23,0],QPXY15[18:23,1]],LONLAT15[1,QPXY15[18:23,0],QPXY15[18:23,1]],s=90,marker='o',facecolor='gold',edgecolors='gold',linewidths=0.01,alpha=0.7)
k=1+18
for i,j in zip(LONLAT15[0,QPXY15[18:23,0],QPXY15[18:23,1]],LONLAT15[1,QPXY15[18:23,0],QPXY15[18:23,1]]):
    ax.text(i,j,s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle))
    k=k+1

ax.scatter(LONLAT15[0,QPXY15[26,0],QPXY15[26,1]],LONLAT15[1,QPXY15[26,0],QPXY15[26,1]],s=90,marker='o',facecolor='gold',edgecolors='gold',linewidths=0.01,alpha=0.7)
k=1+26
ax.text(LONLAT15[0,QPXY15[26,0],QPXY15[26,1]],LONLAT15[1,QPXY15[26,0],QPXY15[26,1]],s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle_q))


ha_val = ['right','left']
va_val=['top','bottom']
ha_cycle = cycle(ha_val)
va_cycle = cycle(va_val)
# River water depth 
k=1
ax.scatter(LONLAT15[0,DPXY15[:,0],DPXY15[:,1]],LONLAT15[1,DPXY15[:,0],DPXY15[:,1]],s=90,marker='^',facecolor='indigo',edgecolors='indigo',linewidths=0.01,alpha=0.7, label='Water depth')
for i,j in zip(LONLAT15[0,DPXY15[:,0],DPXY15[:,1]],LONLAT15[1,DPXY15[:,0],DPXY15[:,1]]):
    ax.text(i,j,s=str(k),c='k',fontsize=14,ha=next(ha_cycle),va=next(va_cycle))
    k+=1


plt.legend(loc="lower right", fontsize=16,markerscale=2)
plt.savefig(work_folder+"/analysis/bad_Q_RWD.jpg",dpi=300,bbox_inches='tight')
