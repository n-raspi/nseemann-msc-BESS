import pyomo.environ as pyo
from pyomo.util.model_size import build_model_size_report

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from components import storage

import plotly.express as px

# import os
# print(os.getcwd())

from import_data import import_data
# combined_market_data = pd.read_csv('30_src/build_dataset/combined_market_data.csv', index_col=0, parse_dates=True)#, date_parser=date_parser)


DA, IP, aFRR_p, FCR_p, aFRR_E, aFRR_pE = import_data('../build_dataset/combined_market_data.csv')

start_datetime = min(IP_p.index)
print(start_datetime)

chosen_start = pd.to_datetime('2023-01-01 00:00:00')
DA_p = DA_p.loc[chosen_start:]
IP_p = IP_p.loc[chosen_start:]
aFRR_p = aFRR_p.loc[chosen_start:]
PCR_p = PCR_p.loc[chosen_start:]

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





def define_model(N_days=3, N_deg_segments = 4):
    print('Starting model definition')

    ############## SETS ##############

    model = pyo.AbstractModel()

    model.storage_units = pyo.Set()

    model.Q = pyo.RangeSet(0, N_days*24*4-1)
    model.H = pyo.RangeSet(0, N_days*24-1)
    model.H4 = pyo.RangeSet(0, N_days*6-1) 

    model.deg_segments = pyo.RangeSet(0,N_deg_segments-1)

    ############## PARAMETERS ##############

    ############## Market parameters

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

    ############## Storage parameters

    model.storage_capacity_MWh= pyo.Param(model.storage_units, mutable=True)
    model.storage_max_charge_MW= pyo.Param(model.storage_units, mutable=False)
    model.storage_max_discharge_MW= pyo.Param(model.storage_units, mutable=False)
    model.storage_eff_charge= pyo.Param(model.storage_units, mutable=False)
    model.storage_eff_discharge=pyo.Param(model.storage_units, mutable=False)
    model.storage_SD=pyo.Param(model.storage_units, mutable=False)
    model.deg_sh = pyo.Param(model.storage_units, mutable=False)
    model.deg_eqtn = pyo.Param(model.storage_units*model.deg_segments*['alpha','beta'], mutable=False)
    #model.storage_financial_params=pyo.Param(model.storage_units*['capcostdiff'], mutable=False)
    #model.storage_CO2_params=pyo.Param(model.storage_units, mutable=False)


    # Start and end SOC
    model.start_SOC = pyo.Param(model.storage_units,mutable=True)
    # Start and end degradation psi
    model.start_psi = pyo.Param(model.storage_units, initialize = 0, mutable=True)
    model.start_deg_cap = pyo.Param(model.storage_units, initialize = 0, mutable=True)

    # PV prod
    PV_prod = np.zeros(len(model.Q)) #random.rand(len(model.Q))

    # Local load
    local_load = np.zeros(len(model.Q)) #np.random.rand(len(model.Q))

    ############## DECISION VARIABLES ##############

    ############## Market participation variables

    # Intraday trades
    model.ID_buy = pyo.Var(model.Q, within=pyo.PositiveReals, bounds = (0,100))
    model.ID_sell = pyo.Var(model.Q, within=pyo.PositiveReals, bounds = (0,100))
  
    # Day-ahead trades
    model.DA_buy = pyo.Var(model.H, within=pyo.PositiveReals, bounds = (0,100))
    model.DA_sell = pyo.Var(model.H, within=pyo.PositiveReals, bounds = (0,100))

    # PCR trades
    model.PCR = pyo.Var(model.H4, within=pyo.PositiveReals)

    # aFRR capacity trades
    model.aFRR_pos = pyo.Var(model.H4, within=pyo.PositiveReals)#, initialize = 0)
    model.aFRR_neg = pyo.Var(model.H4, within=pyo.PositiveReals)#, initialize = 0)

    # Peak power output
    model.peak_x_grid_out = pyo.Var(model.Q, within=pyo.PositiveReals)
    
    ############## Flow variables

    # Grid flow, out means out of grid into node
    model.x_grid_out = pyo.Var(model.Q, within=pyo.Reals)#, bounds = (-0.1,0.1))

    # Storage flow, out means out of grid into node
    model.x_storage_in = pyo.Var(model.storage_units*model.Q, within=pyo.PositiveReals)#, bounds = (0,1))
    model.x_storage_out = pyo.Var(model.storage_units*model.Q, within=pyo.PositiveReals)#, bounds = (0,1))
    
    # Storage state of charge in MWh
    model.SOC_storage = pyo.Var(model.storage_units*model.Q, within=pyo.PositiveReals)

    ############## CONSTRAINTS ##############

    ############## Equilibrium constraints

    # Nodal equilibrium
    model.node_in = pyo.Constraint(model.Q, rule = lambda model, q: 0 ==\
                PV_prod[q]\
                - local_load[q]\
                + model.x_grid_out[q]\
                + sum(model.x_storage_out[storage_unit,q] for storage_unit in model.storage_units)\
                - sum(model.x_storage_in[storage_unit,q] for storage_unit in model.storage_units))

    # Financial balancing: net power from grid must be equal to aggregated trades on both markets. DA is hourly, so divided by 4.
    model.balancing_in = pyo.Constraint(model.Q, rule = lambda model, q: model.x_grid_out[q] == model.ID_buy[q] - model.ID_sell[q] + model.DA_buy[q//4]/4 - model.DA_sell[q//4]/4)


    # Big M constraint for simulaneous buying and selling:
    # Buying and selling simultaneously is not allowed: (ID_buy + DA_buy)*(ID_sell + DA_sell) = 0) 
    model.y = pyo.Var(model.Q, within=pyo.Binary)#, bounds = (-1,1))
    M = 10
    model.const_bigM_ID_sell = pyo.Constraint(model.Q, rule = lambda model, q: (model.ID_sell[q] + model.DA_sell[q//4]/4) <= M*model.y[q])
    model.const_bigM_ID_buy = pyo.Constraint(model.Q, rule = lambda model, q: (model.ID_buy[q] + model.DA_buy[q//4]/4) <= M*(1-model.y[q]))

    # Big M constraint for simulaneous charging and discharging:
    # Buying and selling simultaneously is not allowed: (x_storage_out)*(x_storage_in) = 0) 
    model.x_simult_bin = pyo.Var(model.Q, within=pyo.Binary)#, bounds = (-1,1))
    model.const_bigM_x_storage_out = pyo.Constraint(model.Q, rule = lambda model, q: sum([model.x_storage_in[storage_unit,q] for storage_unit in model.storage_units]) <= M*model.x_simult_bin[q])
    model.const_bigM_x_storage_in = pyo.Constraint(model.Q, rule = lambda model, q: sum([model.x_storage_out[storage_unit,q] for storage_unit in model.storage_units]) <= M*(1-model.x_simult_bin[q]))
    
 
    ############## Storage-specific constraints

    # Storage continuity
    def const_SOC_storage(model, storage_unit, q):
        if q == 0:
            return model.SOC_storage[storage_unit,q] == (1-model.storage_SD[storage_unit] )*(model.start_SOC[storage_unit] + (model.storage_eff_charge[storage_unit]*(model.x_storage_in[storage_unit,q])) - ((1/model.storage_eff_discharge[storage_unit])*model.x_storage_out[storage_unit,q]))
        else:
            return model.SOC_storage[storage_unit,q] == (1-model.storage_SD[storage_unit])*(model.SOC_storage[storage_unit,q-1] + (model.storage_eff_charge[storage_unit]*(model.x_storage_in[storage_unit,q])) - ((1/model.storage_eff_discharge[storage_unit])*model.x_storage_out[storage_unit,q]))
    model.const_storage_cont = pyo.Constraint(model.storage_units*model.Q, rule = const_SOC_storage)

    # Fix the SOC at the last SOC of storage tech to the beginning SOC
    def end_SOC_storage(model,storage_unit):
        return model.SOC_storage[storage_unit, len(model.Q)-1] == model.start_SOC[storage_unit]
    model.SOC_storage_end = pyo.Constraint(model.storage_units, rule = end_SOC_storage)

    # Storage power limits for each storage unit
    model.const_storage_power_in = pyo.Constraint(model.storage_units*model.Q, rule = lambda model, storage_unit, q: model.x_storage_in[storage_unit,q]*4 <= model.storage_max_charge_MW[storage_unit])# BESS_c_rate*BESS_capacity)
    model.const_storage_power_out = pyo.Constraint(model.storage_units*model.Q, rule = lambda model, storage_unit, q: model.x_storage_out[storage_unit,q]*4 <= model.storage_max_discharge_MW[storage_unit])# BESS_c_rate*BESS_capacity)

    model.const_SOC_min = pyo.Constraint(model.storage_units*model.Q, rule = lambda model, storage_unit, q: model.SOC_storage[storage_unit,q] >=  0)
    model.const_SOC_max = pyo.Constraint(model.storage_units*model.Q, rule = lambda model, storage_unit, q: model.SOC_storage[storage_unit,q] <= model.storage_capacity_MWh[storage_unit] - model.start_deg_cap[storage_unit])

    # SOC limits with capacity bidding
    model.const_SOC_tot_min_reg = pyo.Constraint(model.Q, rule = lambda model,  q: sum(model.SOC_storage[storage_unit,q] for storage_unit in model.storage_units) >=  model.aFRR_pos[q//16]*1  + model.PCR[q//16]*0.5)
    model.const_SOC_tot_max_reg = pyo.Constraint(model.Q, rule = lambda model, q: sum(model.SOC_storage[storage_unit,q] for storage_unit in model.storage_units) <= sum(model.storage_capacity_MWh[storage_unit] for storage_unit in model.storage_units) - model.aFRR_neg[q//16]*1 - model.PCR[q//16]*0.5)

    # Peak constraint
    model.const_peak = pyo.Constraint(model.Q, rule = lambda model, q: model.peak_x_grid_out[q] >= model.x_grid_out[q])

    ############## Storage degradation constraints 
    ############## Constants, variables and constraints are defined together for now

    model.psi_j = pyo.Var(model.storage_units*model.Q*model.deg_segments, within=pyo.PositiveReals)
    model.psi = pyo.Var(model.storage_units*model.Q, within=pyo.PositiveReals)

    model.w = pyo.Var(model.storage_units*model.Q*model.deg_segments, within=pyo.Binary)

    model.D = pyo.Var(model.storage_units*model.Q, within=pyo.PositiveReals)


    ######### METHOD  WITH MULTIPLIED BINARY AND CONTINUOUS
    # betabound = 200

#     model.const_uqj_0 = pyo.Constraint(model.storage_units*model.Q*model.deg_segments, rule = lambda model, storage_unit, q, j: model.psi[storage_unit,q] >= model.deg_eqtn[storage_unit,j,'alpha']*model.SOC_storage[storage_unit,q] + model.deg_eqtn[storage_unit,j,'beta'])

#     model.const_uqj_1 = pyo.Constraint(model.storage_units*model.Q*model.deg_segments, rule = lambda model, storage_unit, q, j: (model.w[storage_unit,q,j]*model.psi[storage_unit,q] - betabound*(1-model.w[storage_unit,q,j])) <= (model.deg_eqtn[storage_unit,j,'alpha']*model.SOC_storage[storage_unit,q] + model.deg_eqtn[storage_unit,j,'beta']))

#     model.const_uq_0 = pyo.Constraint(model.storage_units,model.Q, rule = lambda model, storage_unit, q: 1 == sum(model.w[storage_unit,q,j] for j in model.deg_segments))

#  # for all u,q
#     # model.D[storage_unit,q] >= (psi[storage_unit,q] - psi[storage_unit,q-1])/2 
#     def const_uq_2(model, storage_unit, q):
#         if q == 0:
#             return model.D[storage_unit,q] >= (model.psi[storage_unit,q] - model.start_psi[storage_unit])/2
#         else:
#             return model.D[storage_unit,q] >= (model.psi[storage_unit,q] - model.psi[storage_unit,q-1])/2
#     model.const_uq_2 = pyo.Constraint(model.storage_units*model.Q, rule = const_uq_2)

#     # for all u,q
#     # model.D[storage_unit,q] >= (psi[storage_unit,q-1] - psi[storage_unit,q])/2 
#     def const_uq_3(model, storage_unit, q):
#         if q == 0:
#             return model.D[storage_unit,q] >= (model.start_psi[storage_unit] - model.psi[storage_unit,q])/2
#         else:
#             return model.D[storage_unit,q] >= (model.psi[storage_unit,q-1] - model.psi[storage_unit,q])/2
#     model.const_uq_3 = pyo.Constraint(model.storage_units*model.Q, rule = const_uq_3)

#     model.const_uq_4 = pyo.Constraint(model.storage_units*model.Q, rule = lambda model, storage_unit, q: model.D[storage_unit,q] >= model.deg_sh[storage_unit])

    ######### END METHOD  WITH MULTIPLIED BINARY AND CONTINUOUS

    # ######### alternative model.const_uqj_1 with big M
    # model.z = pyo.Var(model.storage_units*model.Q*model.deg_segments, within=pyo.PositiveReals)

    # # w * psi - betabound*(1-w) <= alpha*SOC + beta
    # model.const_uqj_1 = pyo.Constraint(model.storage_units*model.Q*model.deg_segments, rule = lambda model, storage_unit, q, j: (model.z[storage_unit,q,j] - betabound*(1-model.w[storage_unit,q,j])) <= (model.deg_eqtn[storage_unit,j,'alpha']*model.SOC_storage[storage_unit,q] + model.deg_eqtn[storage_unit,j,'beta']))

    # # z <= psi
    # model.const_uqj_2 = pyo.Constraint(model.storage_units*model.Q*model.deg_segments, rule = lambda model, storage_unit, q, j: model.z[storage_unit,q,j] <= model.psi[storage_unit,q])

    # # z <= betabound*w
    # model.const_uqj_3 = pyo.Constraint(model.storage_units*model.Q*model.deg_segments, rule = lambda model, storage_unit, q, j: model.z[storage_unit,q,j] <= betabound*model.w[storage_unit,q,j])

    # # z >= psi - betabound*(1-w)
    # model.const_uqj_4 = pyo.Constraint(model.storage_units*model.Q*model.deg_segments, rule = lambda model, storage_unit, q, j: model.z[storage_unit,q,j] >= model.psi[storage_unit,q] - (1-model.w[storage_unit,q,j])*betabound)

    # ### end alternative model.const_eqj_1


   
      
    ######### METHOD 3 KESKE (not quicker than method 2)

    # xs = {0:0,1:0.25,2:0.5,3:0.75,4:1}
    # model.const_uq_0 = pyo.Constraint(model.storage_units*model.Q, rule = lambda model, storage_unit, q: model.psi[storage_unit,q] == sum((model.deg_eqtn[storage_unit,j,'alpha']*model.SOC_storage[storage_unit,q] + model.deg_eqtn[storage_unit,j,'beta'])*model.w[storage_unit,q,j] for j in model.deg_segments))
    
    # model.const_uq_1 = pyo.Constraint(model.storage_units,model.Q, rule = lambda model, storage_unit, q: 1 >= sum(model.w[storage_unit,q,j] for j in model.deg_segments))
    
    # model.const_uq_2 = pyo.Constraint(model.storage_units,model.Q, rule = lambda model, storage_unit, q: model.SOC_storage[storage_unit,q] >= sum(model.w[storage_unit,q,j]*xs[j] for j in model.deg_segments))
    # model.const_uq_3 = pyo.Constraint(model.storage_units,model.Q, rule = lambda model, storage_unit, q: model.SOC_storage[storage_unit,q] <= sum(model.w[storage_unit,q,j]*xs[j+1] for j in model.deg_segments))


# # for all u,q
#     # model.D[storage_unit,q] >= (psi[storage_unit,q] - psi[storage_unit,q-1])/2 
#     def const_uq_20(model, storage_unit, q):
#         if q == 0:
#             return model.D[storage_unit,q] >= (model.psi[storage_unit,q] - model.start_psi[storage_unit])/2
#         else:
#             return model.D[storage_unit,q] >= (model.psi[storage_unit,q] - model.psi[storage_unit,q-1])/2
#     model.const_uq_20 = pyo.Constraint(model.storage_units*model.Q, rule = const_uq_20)

#     # for all u,q
#     # model.D[storage_unit,q] >= (psi[storage_unit,q-1] - psi[storage_unit,q])/2 
#     def const_uq_30(model, storage_unit, q):
#         if q == 0:
#             return model.D[storage_unit,q] >= (model.start_psi[storage_unit] - model.psi[storage_unit,q])/2
#         else:
#             return model.D[storage_unit,q] >= (model.psi[storage_unit,q-1] - model.psi[storage_unit,q])/2
#     model.const_uq_30 = pyo.Constraint(model.storage_units*model.Q, rule = const_uq_30)

#     model.const_uq_40 = pyo.Constraint(model.storage_units*model.Q, rule = lambda model, storage_unit, q: model.D[storage_unit,q] >= model.deg_sh[storage_unit])
 
    ############## OBJECTIVE FUNCTION ##############    

    def obj_expression(model):
        return \
            + pyo.sum_product(model.ID_buy, model.pricesQ, index=model.ID_buy) \
            - pyo.sum_product(model.ID_sell, model.pricesQ, index=model.ID_buy)\
            + pyo.sum_product(model.DA_buy, model.pricesH, index=model.DA_buy)\
            - pyo.sum_product(model.DA_sell, model.pricesH, index=model.DA_buy)\
            - pyo.sum_product(model.aFRR_pos, model.pricesH4_aFRR_pos, index=model.aFRR_pos)\
            - pyo.sum_product(model.aFRR_neg, model.pricesH4_aFRR_neg, index=model.aFRR_neg)\
            - pyo.sum_product(model.PCR, model.pricesH4_PCR, index=model.PCR)#\
            #+ sum(model.D[storage_unit,q] for storage_unit in model.storage_units for q in model.Q) #  + pyo.sum_product(model.peak[q], p_peak) + pyo.sum_product(model.dq[q], emissions[q])

    model.obj = pyo.Objective(rule = obj_expression, sense = pyo.minimize)

    print('Model definition done')

    return model
    
def define_instance(model, storage_units):
    # is it necessary to define the parameters in the instance definition?
    # model.deg_eqtn = pyo.Param(model.storage_units*model.deg_segments*['alpha','beta'])

    instance = model.create_instance(
        data = {None:{
            'storage_units': {None: [i.name for i in storage_units]},
            'start_SOC': dict(zip([i.name for i in storage_units],[0.5])),
            'storage_capacity_MWh': dict(zip([i.name for i in storage_units],[i.capacity_MWh for i in storage_units])),
            'storage_max_charge_MW': dict(zip([i.name for i in storage_units],[i.max_charge_MW for i in storage_units])),
            'storage_max_discharge_MW': dict(zip([i.name for i in storage_units],[i.max_discharge_MW for i in storage_units])),
            'storage_eff_charge': dict(zip([i.name for i in storage_units],[i.eff_charge for i in storage_units])),
            'storage_eff_discharge': dict(zip([i.name for i in storage_units],[i.eff_discharge for i in storage_units])),
            'storage_SD': dict(zip([i.name for i in storage_units],[i.SD for i in storage_units])),
            'deg_sh': dict(zip([i.name for i in storage_units],[i.degradation_sh for i in storage_units])),
            'deg_eqtn':  {(unit.name, seg, param): value for unit in storage_units for seg, (alpha, beta) in enumerate(unit.degradation_eqtn) for param, value in zip(['alpha', 'beta'], [alpha, beta])},
            #'storage_degradation_params': dict(zip([i.name for i in storage_units],[i.degradation_params for i in storage_units])),
            # 'storage_financial_params': dict(zip([i.name for i in storage_units],[i.financial_params for i in storage_units])),
            # 'storage_CO2_params': dict(zip([i.name for i in storage_units],[i.CO2_params for i in storage_units]))
            }
        }
    )
    return instance
    
def get_list(pyomo_var):
        return list(pyomo_var.extract_values().values())


def run_instance(instance, day, start_SOC, start_deg_cap, start_psi):
    shiftQ = day*24*4 
    shiftH = day*24
    shiftH4 = day*6 

    for j in instance.storage_units:
        instance.start_SOC[j] = start_SOC
        instance.start_psi[j] = start_psi
        instance.start_deg_cap[j] = start_deg_cap
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
        "MIPGap":  0.001, #0.005,
        "OutputFlag": 1,
        #"TimeLimit": 60
    }

    print('Starting instance: ', day)
    solver.solve(instance,tee=True, options = options)
    print('Finished instance: ', day)
    # SOC_end = get_list(instance.SOC_BESS)[-1]


    outputQ = pd.DataFrame({
        'ID_buy': get_list(instance.ID_buy),
        'ID_sell': get_list(instance.ID_sell),
        'x_grid_out': get_list(instance.x_grid_out),
        'priceQ': get_list(instance.pricesQ)
    })

    outputQ_units = pd.DataFrame()

    for i in storage_units:
        outputQ_units['SOC_' + i.name] = list(instance.SOC_storage[i.name,:].value)
        outputQ_units['x_' + i.name + '_in'] = list(instance.x_storage_in[i.name,:].value)
        outputQ_units['x_' + i.name + '_out'] = list(instance.x_storage_out[i.name,:].value)

    
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

    return outputQ, outputQ_units, outputH, outputH4

# Other misc. functions

def multiply_nested_array(nested_array, multiplier):
    return [[element * multiplier for element in sub_array] for sub_array in nested_array]

def psif(A,B,SOC):
    return A* ((1-SOC)**B)

# Setting up current instance and model

cost_mult = 95000 # 95000 EUR/MWh

lifetime = 20*365*24*4 # 20 years in quarters

deg_eqtn = [[-0.0004969829179274153, 0.000274],\
            [-0.0003433660423914995, 0.00023559578111602105],\
            [-0.00019601832609530072, 0.00016192192296792166],\
            [-5.963271358578449e-05, 5.963271358578449e-05]]
deg_eqtn_mult = multiply_nested_array(deg_eqtn, cost_mult)

BESS = storage(
    name='BESS',
    capacity_MWh=1,
    max_charge_MW=0.5,
    max_discharge_MW=0.5,
    eff_charge=0.90,
    eff_discharge=0.90,
    SD=0.00001,
    degradation_eqtn = multiply_nested_array(deg_eqtn, cost_mult), # alpha, beta for each segment
    degradation_sh=cost_mult/lifetime,
    financial_params=[cost_mult],
    CO2_params=10)


# cost_mult_SC = 5000 # 95000 EUR/MWh

# lifetime_SC = 20*365*24*4 # 20 years in quarters

# deg_eqtn_SC = [[-0.0004969829179274153, 0.000274],\
#             [-0.0003433660423914995, 0.00023559578111602105],\
#             [-0.00019601832609530072, 0.00016192192296792166],\
#             [-5.963271358578449e-05, 5.963271358578449e-05]]

# SC = storage(
#     name='SC',
#     capacity_MWh=0.5,
#     max_charge_MW=1,
#     max_discharge_MW=1,
#     eff_charge=0.95,
#     eff_discharge=0.95,
#     SD=0.00005,#0.000007,
#     degradation_eqtn = multiply_nested_array(deg_eqtn_SC, cost_mult_SC), # alpha, beta for each segment
#     degradation_sh=cost_mult_SC/lifetime_SC,
#     financial_params=[cost_mult_SC],
#     CO2_params=10)


storage_units = [BESS]#, SC]

# Number of days simulated
N_days = 2

# Number of days outputted
N_days_fix = 2

# First day of simulation =  "chosen_start" + "day". e.g. if chosen_start is 2023-01-01 and day is 1, the simulation starts at 2023-01-02
day = 0

model = define_model(N_days=2)
instance = define_instance(model,storage_units)

# results_instance = run_instance(instance, day, 0.5, 0, 95000*psif(0.000274, 2.1, 0.5))

results_df = pd.DataFrame()


for day in range(0,1):
    outputQ, outputQ_units, outputH, outputH4 = run_instance(instance, day, 0.5, 0, 95000*psif(0.000274, 2.1, 0.5))
    outputQ.index  = [chosen_start + day*pd.to_timedelta('1day') + i*pd.to_timedelta('15min') for i in range(N_days*24*4)]
    outputQ_units.index  = [chosen_start + day*pd.to_timedelta('1day') + i*pd.to_timedelta('15min') for i in range(N_days*24*4)]
    outputH.index = [chosen_start + day*pd.to_timedelta('1day') + i*pd.to_timedelta('1h') for i in range(N_days*24)]
    outputH4.index = [chosen_start + day*pd.to_timedelta('1day') + i*pd.to_timedelta('4h') for i in range(N_days*6)]
    output_window = pd.concat([outputQ,outputQ_units,outputH,outputH4], axis=1)[chosen_start + day*pd.to_timedelta('1day'):chosen_start + (N_days_fix + day)*pd.to_timedelta('1day')-pd.to_timedelta('15min')]
    last_SOC = output_window.SOC_BESS.iloc[-1]
    results_df = pd.concat([results_df, output_window])
    results_df.to_csv('all_markets_results_1041.csv')
    print(pyo.value(instance.obj))

# fig = px.line()
# fig.add_scatter(y=list(instance.D.extract_values().values()), name = 'D')

# fig.add_scatter(y=list(instance.SOC_storage.extract_values().values()), mode='lines', name='SOC')
# fig.add_scatter(y=list(instance.psi.extract_values().values()), mode='lines', name='psi')
# fig.show()


# results_df.to_csv('results_test.csv')

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


##### Degradation analysis:
# SOC = pyo.value(instance.SOC_storage['BESS',:])
# psi = pyo.value(instance.psi['BESS',:])
# fig = px.scatter(x=SOC,y=psi)

# A = 0.000274
# B = 2.1
# # Generate 200 sample points for SOC
# SOC_values = np.linspace(0, 1, 200)
# psi_values = cost_mult*[psif(A, B, soc) for soc in SOC_values]

# fig.add_scatter(x=SOC_values, y=psi_values, mode='lines')

# for alpha, beta in BESS.degradation_eqtn:
#     psi_values = [alpha * soc + beta for soc in SOC_values]
#     fig.add_scatter(x=SOC_values, y=psi_values, mode='lines')
    
# fig.show()
