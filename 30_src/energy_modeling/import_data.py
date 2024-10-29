# import pyomo.environ as pyo
# from pyomo.util.model_size import build_model_size_report

import numpy as np
import pandas as pd

def import_data():
    # import matplotlib.pyplot as plt

    date_parser = pd.to_datetime
    combined_market_data = pd.read_csv('../build_dataset/combined_market_data.csv', index_col=0, parse_dates=True, date_parser=date_parser)
    
    # When debugging:
    # combined_market_data = pd.read_csv('nseemann-msc-BESS/30_src/build_dataset/combined_market_data.csv', index_col=0, parse_dates=True, date_parser=date_parser)


    DA_IP = combined_market_data[['IPindex_netztransparenz_15min_pE','DADELU_ENTSOE_60min_pE']]
    DA_IP_lim = DA_IP[DA_IP['DADELU_ENTSOE_60min_pE'].first_valid_index():DA_IP['DADELU_ENTSOE_60min_pE'].last_valid_index()][DA_IP['IPindex_netztransparenz_15min_pE'].first_valid_index():DA_IP['IPindex_netztransparenz_15min_pE'].last_valid_index()]

    #DA_IP_lim['IPindex_netztransparenz_15min_pE']
    DA = DA_IP_lim['DADELU_ENTSOE_60min_pE']
    IP = DA_IP_lim['IPindex_netztransparenz_15min_pE']
    DA = DA.fillna(DA.shift(24, freq='h')).asfreq('1h')
    IP = IP.fillna(IP.shift(96, freq='15min'))  # 96 * 15min = 24 hours


    DA = DA[DA.index < DA.last_valid_index()] #- pd.Timedelta('1min')]
    IP = IP[IP.index < DA.last_valid_index() + pd.Timedelta('60min')]

    # Check both indexes are continuous (should be 1 for first index)
    print(sum(DA.index.diff() != pd.Timedelta('1h')))
    print(sum(IP.index.diff() != pd.Timedelta('15min')))

    # Check all indexes match when IP is resampled to 1h (should be 0)
    print((DA.index != IP.asfreq('1h').index).sum()) 
    return DA, IP