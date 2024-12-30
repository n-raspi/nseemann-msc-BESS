import pyomo.environ as pyo
from pyomo.util.model_size import build_model_size_report

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from components import storage

import plotly.express as px

from import_data import import_data
DA_p, IP_p, aFRR_p, PCR_p, aFRR_E, aFRR_pE = import_data('/Users/nicolas/Documents/GitHub/nseemann-msc-BESS/dev_OOP/30_src/energy_modeling/dataset_prep/combined_market_data.csv')

start_datetime = min(IP_p.index)
print(start_datetime)

chosen_start = pd.to_datetime('2021-06-15 00:00:00')
DA_p = DA_p.loc[chosen_start:]
IP_p = IP_p.loc[chosen_start:]
aFRR_p = aFRR_p.loc[chosen_start:]
PCR_p = PCR_p.loc[chosen_start:]
aFRR_pE = aFRR_pE.loc[chosen_start:]

combined_data = pd.concat([IP_p, aFRR_pE], axis=1)
combined_data
# # ## Market prices
# IP_p = IP_p.to_list()
# DA_p = DA_p.to_list()
# aFRRpos_p = aFRR_p['aFRRpos_SMARD_15min_pP'].to_list()
# aFRRneg_p = aFRR_p['aFRRneg_SMARD_15min_pP'].to_list()

## aFRR market data import
# aFRR_market_interpolation = pd.read_csv('/Users/nicolas/Documents/GitHub/nseemann-msc-BESS/dev_OOP/30_src/energy_modeling/dataset_prep/aFRR_activation_interpolated.csv', index_col=0, parse_dates=True)

## First stage optimization results
# first_stage_results = pd.read_csv('/Users/nicolas/Documents/GitHub/nseemann-msc-BESS/dev_OOP/30_src/energy_modeling/model/all_markets_results_new.csv', index_col=0, parse_dates=True)

# aFRR_market_interpolation