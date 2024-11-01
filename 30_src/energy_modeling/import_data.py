# import pyomo.environ as pyo
# from pyomo.util.model_size import build_model_size_report

import numpy as np
import pandas as pd

def import_data(min_date):

    date_parser = pd.to_datetime
    combined_market_data = pd.read_csv('../build_dataset/combined_market_data.csv', index_col=0, parse_dates=True, date_parser=date_parser)
    # When debugging:
    # combined_market_data = pd.read_csv('nseemann-msc-BESS/30_src/build_dataset/combined_market_data.csv', index_col=0, parse_dates=True, date_parser=date_parser)
    combined_market_data = combined_market_data[combined_market_data.index > min_date]

    ################## DA and ID data ##################
    IP = combined_market_data['IPindex_netztransparenz_15min_pE']
    IP = IP[IP.first_valid_index():IP.last_valid_index()] #Crop nas
    IP = IP.fillna(IP.shift(96, freq='15min'))  # Fillna from 24h before


    DA = combined_market_data['DADELU_ENTSOE_60min_pE'] 
    DA = DA[DA.first_valid_index():DA.last_valid_index()] #Crop nas
    DA = DA.fillna(DA.shift(24, freq='h')).asfreq('1h') #Fillna from 24h before


    ################## aFRR data ##################
    aFRR_p = combined_market_data[['aFRRpos_SMARD_15min_pP','aFRRneg_SMARD_15min_pP']]
    aFRR_p = aFRR_p[aFRR_p.apply(lambda col: col.first_valid_index()).max(): aFRR_p.apply(lambda col: col.last_valid_index()).min()]
    aFRR_p = aFRR_p.fillna(aFRR_p.shift(6, freq='4h')).resample('4h').sum()  #Fillna from 24h before


    ################## FCR data ##################

    FCR_p = combined_market_data['FCR_regelleistung_4h_pP']
    FCR_p = FCR_p[FCR_p.first_valid_index():FCR_p.last_valid_index()]
    FCR_p = FCR_p.fillna(FCR_p.shift(6, freq='4h')).resample('4h').sum()


    ################## Align the start and end of the datasets ##################
    start = max(DA.first_valid_index(), IP.first_valid_index(), aFRR_p.apply(lambda col: col.first_valid_index()).max(), FCR_p.first_valid_index())
    end = min(DA.last_valid_index(), IP.last_valid_index(), aFRR_p.apply(lambda col: col.last_valid_index()).min(), FCR_p.last_valid_index())

    print(start,end)

    IP = IP[start:end][:-1]#.dropna()
    DA = DA[start:end][:-1]#.dropna()
    aFRR_p = aFRR_p[start:end][:-1]#.dropna()

    ################## CHECKS ##################

    # # Check both indexes are continuous (should be 1 for first index)
    # print(sum(DA.index.diff() != pd.Timedelta('1h')))
    # print(sum(IP.index.diff() != pd.Timedelta('15min')))
    # print(sum(DA.index.diff() != pd.Timedelta('1h')))
    # print(sum(aFRR_p.index.diff() != pd.Timedelta('4h')))

    # # Check all indexes match when IP is resampled to 1h (should be 0)
    # print((DA.index != IP.asfreq('1h').index).sum()) 

    # # Check all indexes match when DA is resampled to 4h (should be 0)
    # print((aFRR_p.index != DA.asfreq('4h').index).sum()) 

    # print(len(aFRR_p.index))
    # print(len(DA.index)/4)
    # print(len(IP.index)/16)

    return DA, IP, aFRR_p, FCR_p

