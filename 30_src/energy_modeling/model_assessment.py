import pyomo.environ as pyo
from pyomo.util.model_size import build_model_size_report

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# import os
# print(os.getcwd())

from import_data import import_data
# combined_market_data = pd.read_csv('30_src/build_dataset/combined_market_data.csv', index_col=0, parse_dates=True)#, date_parser=date_parser)


DA_p, IP_p, aFRR_p, PCR_p = import_data()

start_datetime = min(IP_p.index)
print(start_datetime)

# ## Market prices
IP_p = IP_p.to_list()
DA_p = DA_p.to_list()
aFRRpos_p = aFRR_p['aFRRpos_SMARD_15min_pP'].to_list()
aFRRneg_p = aFRR_p['aFRRneg_SMARD_15min_pP'].to_list()

# Needed data to import:
# - DA prices
# - ID prices
# - PCR prices
# - aFRR capacity prices
# - aFRR activation prices


N_days = 3
N_days_fix = 1
day = 0

outputQ = pd.DataFrame(columns=[
        'ID_buy',
        'ID_sell',
        'SOC_BESS',
        'x_grid_out',
        'x_BESS_in',
        'x_BESS_out',
        'priceQ'
])

outputH = pd.DataFrame(columns=[
    'DA_buy',
    'DA_sell',
    'priceH'
])

outputH4 = pd.DataFrame(columns=[
    'PCR',
    'aFRR_pos',
    'aFRR_neg',
    'priceH4_PCR',
    'priceH4_aFRR_pos',
    'priceH4_aFRR_neg'
])

def define_model():
    model = pyo.AbstractModel()

    model.Q = pyo.RangeSet(0, N_days*24*4-1)
    model.H = pyo.RangeSet(0, N_days*24-1)
    model.H4 = pyo.RangeSet(0, N_days*6-1)  
    # parameters to define and instantiate:
     # ID prices
    model.pricesQ = pyo.Param(model.Q, mutable=True)

    # DA prices
    model.pricesH = pyo.Param(model.H, mutable=True)

    # PCR prices
    model.pricesH4_PCR = pyo.Param(model.H4, mutable=True)

    #  aFRR_pos capacity prices
    model.pricesH4_aFRR_pos = pyo.Param(model.H4, mutable=True)

    #  aFRR_neg capacity prices
    model.pricesH4_aFRR_neg = pyo.Param(model.H4, mutable=True)

    # # Peak prices
    # model.p_peak = pyo.Param()

    # # Emissions
    # model.emissions = pyo.Param(model.Q)

    # Start and end SOC
    model.start_SOC = pyo.Param(mutable=True)

    ## System data

    # Efficiency
    e_BESS = 0.9 # Battery efficiency
    SD_BESS = 0.01 # Self discharge rate of battery in %/quarter-hour

    BESS_capacity = 1 # MWh
    BESS_c_rate = 0.5 # C-rate of battery

    # PV prod
    PV_prod = np.zeros(len(model.Q)) #random.rand(len(model.Q))

    # Local load
    local_load = np.zeros(len(model.Q)) #np.random.rand(len(model.Q))


    ## Main trading decision variables
    model.ID_buy = pyo.Var(model.Q, within=pyo.PositiveReals, bounds = (0,100))
    model.ID_sell = pyo.Var(model.Q, within=pyo.PositiveReals, bounds = (0,100))
  
    pyo.sum_product(model.ID_buy, model.pricesQ, index=model.ID_buy)

    model.DA_buy = pyo.Var(model.H, within=pyo.PositiveReals, bounds = (0,100))
    model.DA_sell = pyo.Var(model.H, within=pyo.PositiveReals, bounds = (0,100))

    model.PCR = pyo.Var(model.H4, within=pyo.PositiveReals)

    model.aFRR_pos = pyo.Var(model.H4, within=pyo.PositiveReals)#, initialize = 0)
    model.aFRR_neg = pyo.Var(model.H4, within=pyo.PositiveReals)#, initialize = 0)

    # model.peak = pyo.Var(model.Q, within=pyo.PositiveReals)

    ## Supporting decision variables

    # Decision flows in and out of components

    model.x_grid_out = pyo.Var(model.Q, within=pyo.Reals)#, bounds = (-0.1,0.1))

    model.x_BESS_in = pyo.Var(model.Q, within=pyo.PositiveReals)#, bounds = (0,1))
    model.x_BESS_out = pyo.Var(model.Q, within=pyo.PositiveReals)#, bounds = (0,1))

    # model.x_supercap_in = pyo.Var(model.Q, within=pyo.PositiveReals)
    # model.x_supercap_out = pyo.Var(model.Q, within=pyo.PositiveReals)

    model.SOC_BESS = pyo.Var(model.Q, within=pyo.PositiveReals)

    ## Constraints

    #  Equilibrium constraints (nodal and trade balancing)
    model.node_in = pyo.Constraint(model.Q, rule = lambda model, q: 0 == PV_prod[q] - local_load[q]  + model.x_grid_out[q] + model.x_BESS_out[q] - model.x_BESS_in[q]) # + model.x_supercap_out[q] - model.x_supercap_in[q])

    # Financial balancing: net power from grid must be equal to aggregated trades on both markets. DA is hourly, so divided by 4.
    model.balancing_in = pyo.Constraint(model.Q, rule = lambda model, q: model.x_grid_out[q] == model.ID_buy[q] - model.ID_sell[q] + model.DA_buy[q//4]/4 - model.DA_sell[q//4]/4)

    # Big M constraint for simulaneous buying and selling:
    # Buying and selling simultaneously is not allowed: (ID_buy + DA_buy)*(ID_sell + DA_sell) = 0) 
    model.y = pyo.Var(model.Q, within=pyo.Binary)#, bounds = (-1,1))
    M = 1000
    model.const_bigM_ID_sell = pyo.Constraint(model.Q, rule = lambda model, q: (model.ID_sell[q] + model.DA_sell[q//4]/4) <= M*model.y[q])
    model.const_bigM_ID_buy = pyo.Constraint(model.Q, rule = lambda model, q: (model.ID_buy[q] + model.DA_buy[q//4]/4) <= M*(1-model.y[q]))

    # Storage continuity
    def const_SOC_BESS(model, q):
        if q == 0:
            return model.SOC_BESS[q] == (1-SD_BESS)*(model.start_SOC + (e_BESS*(model.x_BESS_in[q])) - ((1/e_BESS)*model.x_BESS_out[q]))
        else:
            return model.SOC_BESS[q] == (1-SD_BESS)*(model.SOC_BESS[q-1] + (e_BESS*(model.x_BESS_in[q])) - ((1/e_BESS)*model.x_BESS_out[q]))
    model.BESS_cont = pyo.Constraint(model.Q, rule = const_SOC_BESS)

    def end_SOC_BESS(model):
        return model.SOC_BESS[len(model.Q)-1] == model.start_SOC
    model.SOC_BESS_end = pyo.Constraint(rule = end_SOC_BESS)

    # BESS power
    model.const_BESS_power_in = pyo.Constraint(model.Q, rule = lambda model, q: model.x_BESS_in[q] <= 0.5)# BESS_c_rate*BESS_capacity)
    model.const_BESS_power_out = pyo.Constraint(model.Q, rule = lambda model, q: model.x_BESS_out[q] <= 0.5)# BESS_c_rate*BESS_capacity)

    # Supercap power
    # model.const_supercap_power_in = pyo.Constraint(model.Q, rule = lambda model, q: model.x_supercap_in[q] <= 0.5)
    # model.const_supercap_power_out = pyo.Constraint(model.Q, rule = lambda model, q: model.x_supercap_out[q] <= 0.5)


    # SOC bidding limits (fix)
    model.const_SOC_BESS_min_reg = pyo.Constraint(model.Q, rule = lambda model, q: model.SOC_BESS[q] >=  model.aFRR_pos[q//16]*1  + model.PCR[q//16]*0.5)
    model.const_SOC_BESS_max_reg = pyo.Constraint(model.Q, rule = lambda model, q: model.SOC_BESS[q] <= BESS_capacity - model.aFRR_neg[q//16]*1 - model.PCR[q//16]*0.5)

    # Peak constraint
    model.const_peak = pyo.Constraint(model.Q, rule = lambda model, q: model.peak[q] >= -1*model.x_grid_out[q])

    # Degradation here the constants, variables and constraints are defined together
    # n=5 # Number of segments
    # alpha = np.random.rand(5)
    # beta = np.random.rand(5)

    # model.wq = pyo.Var(model.Q, within=pyo.PositiveReals)

    # def const_deg_lin(model,q,j):
    #     return model.wq[q] >= alpha[j]*model.SOC_BESS[q] + beta[j]

    # model.const_deg_lin = pyo.Constraint(model.Q*pyo.RangeSet(0,n-1,1), rule = const_deg_lin)

    # Dsh = 0.004 # Calendar degradation

    # model.dq = pyo.Var(model.Q, within=pyo.PositiveReals)

    # def const_dq(model,q):
    #     if q == 0:
    #         return pyo.Constraint.Skip
    #     else:
    #         return model.dq[q] >= model.wq[q] - model.wq[q-1] + Dsh
        
    # def const_dq_dsh(model,q):
    #     if q == 0:
    #         return pyo.Constraint.Skip
    #     else:
    #         return model.dq[q] >= Dsh

    # model.const_dq = pyo.Constraint(model.Q, rule = const_dq)
    # model.const_dq_dsh = pyo.Constraint(model.Q, rule = const_dq_dsh)

    # deg_total = sum(model.dq[q] for q in model.Q)
    

    def obj_expression(model):
        return pyo.sum_product(model.ID_buy, model.pricesQ, index=model.ID_buy) \
            - pyo.sum_product(model.ID_sell, model.pricesQ, index=model.ID_buy)\
            + pyo.sum_product(model.DA_buy, model.pricesH, index=model.DA_buy)\
            - pyo.sum_product(model.DA_sell, model.pricesH, index=model.DA_buy)\
            - pyo.sum_product(model.aFRR_pos, model.pricesH4_aFRR_pos, index=model.aFRR_pos)\
            - pyo.sum_product(model.aFRR_neg, model.pricesH4_aFRR_neg, index=model.aFRR_neg)\
            - pyo.sum_product(model.PCR, model.pricesH4_PCR, index=model.PCR)  #  + pyo.sum_product(model.peak[q], p_peak) + pyo.sum_product(model.dq[q], emissions[q])

    model.obj = pyo.Objective(rule = obj_expression, sense = pyo.minimize)

    return model
    


    # print('Obj for exclusive buy/sell: '+ str(results['Problem'][0]['Upper bound']))
    # print('DA_buy: '+ str(sum(model.DA_buy.extract_values().values())))
    # print('DA_sell: '+ str(sum(model.DA_sell.extract_values().values())))

    # print('ID_buy: '+ str(sum(model.ID_buy.extract_values().values())))
    # print('ID_sell: '+ str(sum(model.ID_sell.extract_values().values())))

    # print('Abs SOC change: ' + str(sum(abs(model.SOC_BESS.extract_values()[q] - model.SOC_BESS.extract_values()[q-1]) for q in model.Q if q > 0)))


    # SOC = list(model.SOC_BESS.extract_values().values())
    # print(SOC[0], SOC[-1])
    # plt.plot(model.SOC_BESS.extract_values().values())
    # plt.show()
    # print('done')
    # print(model.obj())

    # pd.DataFrame(model.ID_buy.extract_values())

def define_instance(model):
    # def run_model(day, start_SOC):
    shiftQ = day*24*4 
    shiftH = day*24
    shiftH4 = day*6 
    instance = model.create_instance(
        data = {None:{
            'pricesQ': dict(zip(model.Q, [IP_p[shiftQ+q] for q in model.Q])),
            'pricesH': dict(zip(model.H, [DA_p[shiftH+h] for h in model.H])),
            'pricesH4_PCR': dict(zip(model.H4, [PCR_p[shiftH4+h] for h in model.H4])),
            'pricesH4_aFRR_pos': dict(zip(model.H4, [aFRRpos_p[shiftH4+h] for h in model.H4])),
            'pricesH4_aFRR_neg': dict(zip(model.H4, [aFRRneg_p[shiftH4+h] for h in model.H4])),
            'start_SOC': {None: 0.5}
            }
        }
    )
    return instance
    

def get_list(pyomo_var):
        return list(pyomo_var.extract_values().values())

def run_instance(instance, start_SOC, day):
    shiftQ = day*24*4 
    shiftH = day*24
    shiftH4 = day*6 
    instance.start_SOC = start_SOC

    for q in instance.Q:
        instance.pricesQ[q] = IP_p[shiftQ + q]
    for h in instance.H:
        instance.pricesH[h] = DA_p[shiftH + h]
    for h4 in instance.H4:
        instance.pricesH4_PCR[h4] = PCR_p[shiftH4 + h4]
        instance.pricesH4_aFRR_pos[h4] = aFRRpos_p[shiftH4 + h4]
        instance.pricesH4_aFRR_neg[h4] = aFRRneg_p[shiftH4 + h4]
    
    # instance.pricesQ = dict(zip(instance.Q, [IP_p[shiftQ+q] for q in instance.Q]))
    # instance.pricesH = dict(zip(instance.H, [DA_p[shiftH+h] for h in instance.H]))
    # instance.pricesH4_PCR = dict(zip(instance.H4, [PCR_p[shiftH4+h] for h in instance.H4]))
    # instance.pricesH4_aFRR_pos = dict(zip(instance.H4, [aFRRpos_p[shiftH4+h] for h in instance.H4]))
    # instance.pricesH4_aFRR_neg = dict(zip(instance.H4, [aFRRneg_p[shiftH4+h] for h in instance.H4]))
    
    solver = pyo.SolverFactory('gurobi') 
    
    options = {
        "MIPGap":  0.005,
        "OutputFlag": 1
    }

    print('Starting instance: ', day)
    solver.solve(instance,tee=True, options = options)
    print('Finished instance: ', day)
    # SOC_end = get_list(instance.SOC_BESS)[-1]

    outputQ = pd.DataFrame({
        'ID_buy': get_list(instance.ID_buy),
        'ID_sell': get_list(instance.ID_sell),
        'SOC_BESS': get_list(instance.SOC_BESS),
        'x_grid_out': get_list(instance.x_grid_out),
        'x_BESS_in': get_list(instance.x_BESS_in),
        'x_BESS_out': get_list(instance.x_BESS_out),
        'priceQ': get_list(instance.pricesQ)
    })

    outputH = pd.DataFrame({
        'DA_buy': get_list(instance.DA_buy),
        'DA_sell': get_list(instance.DA_sell),
        'priceH': get_list(instance.pricesH)
    })

    outputH4 = pd.DataFrame({
        'PCR': get_list(instance.PCR),
        'aFRR_pos': get_list(instance.aFRR_pos),
        'aFRR_neg': get_list(instance.aFRR_neg),
        'priceH4_PCR': get_list(instance.pricesH4_PCR),
        'priceH4_aFRR_pos': get_list(instance.pricesH4_aFRR_pos),
        'priceH4_aFRR_neg': get_list(instance.pricesH4_aFRR_neg)
    })

    return outputQ, outputH, outputH4


model = define_model()
instance = define_instance(model)

# To get duals
instance.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# outputQ, outputH, outputH4
last_SOC = 0.5

results_df = pd.DataFrame()

for day in range(0,1):
    outputQ, outputH, outputH4 = run_instance(instance, 0.5,day)
    outputQ.index  = [start_datetime + day*pd.to_timedelta('1day') + i*pd.to_timedelta('15min') for i in range(N_days*24*4)]
    outputH.index = [start_datetime + day*pd.to_timedelta('1day') + i*pd.to_timedelta('1h') for i in range(N_days*24)]
    outputH4.index = [start_datetime + day*pd.to_timedelta('1day') + i*pd.to_timedelta('4h') for i in range(N_days*6)]
    output_window = pd.concat([outputQ,outputH,outputH4], axis=1)[start_datetime + day*pd.to_timedelta('1day'):start_datetime + (N_days_fix + day)*pd.to_timedelta('1day')-pd.to_timedelta('15min')]
    last_SOC = output_window.SOC_BESS.iloc[-1]

    results_df = pd.concat([results_df, output_window])

results_df.to_csv('results_test.csv')

# # ID prices
# instance.pricesQ = IP_p[shiftQ:shiftQ+len(model.Q)]

# # DA prices
# instance.pricesH = DA_p[shiftH:shiftH+len(model.H)]

# # PCR prices
# instance.pricesH4_PCR = PCR_p[shiftH4:shiftH4+len(model.H4)]

# #  aFRR_pos capacity prices
# instance.pricesH4_aFRR_pos = aFRRpos_p[shiftH4:shiftH4+len(model.H4)]

# #  aFRR_neg capacity prices
# instance.pricesH4_aFRR_neg = aFRRneg_p[shiftH4:shiftH4+len(model.H4)]

# # # Peak prices
# # instance.p_peak = np.random.rand(1)

# # Emissions
# instance.emissions = np.random.rand(len(model.Q))







# output = run_model(0,0.5)
# # print(output[0].model.SOC_BESS[N_days_fix*24*4-1])
# output = run_model(1,output[0]['SOC_BESS'].iloc[N_days_fix*24*4-1])
