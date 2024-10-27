import pyomo.environ as pyo
from pyomo.util.model_size import build_model_size_report

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Needed data to import:
# - DA prices
# - ID prices
# - PCR prices
# - aFRR capacity prices
# - aFRR activation prices

model = pyo.ConcreteModel()

N_days = 7

model.Q = pyo.RangeSet(0, N_days*24*4-1)
model.H = pyo.RangeSet(0, N_days*24-1)
model.H4 = pyo.RangeSet(0, N_days*6-1)

## Market prices

# ID prices
pricesQ = 100*np.random.rand(len(model.Q))

# DA prices
pricesH = 100*np.random.rand(len(model.H))

# PCR prices
pricesH4_PCR = np.random.rand(len(model.H4))

#  aFRR_pos capacity prices
pricesH4_aFRR_pos = np.random.rand(len(model.H4))

#  aFRR_neg capacity prices
pricesH4_aFRR_neg = np.random.rand(len(model.H4))

# Peak prices
p_peak = np.random.rand(1)

# Emissions
emissions = np.random.rand(len(model.Q))

## System data

# Efficiency
e_BESS = 1 # Battery efficiency
SD_BESS = 0.01 # Self discharge rate of battery in %/quarter-hour

BESS_capacity = 1 # MWh
BESS_c_rate = 1 # C-rate of battery

# PV prod
PV_prod = np.random.rand(len(model.Q))

# Local load
local_load = np.random.rand(len(model.Q))


## Main trading decision variables
model.ID_buy = pyo.Var(model.Q, within=pyo.PositiveReals, bounds = (0,100))
model.ID_sell = pyo.Var(model.Q, within=pyo.PositiveReals, bounds = (0,100))

model.DA_buy = pyo.Var(model.H, within=pyo.PositiveReals, bounds = (0,100))
model.DA_sell = pyo.Var(model.H, within=pyo.PositiveReals, bounds = (0,100))

model.PCR = pyo.Var(model.H4, within=pyo.PositiveReals)

model.aFRR_pos = pyo.Var(model.H4, within=pyo.PositiveReals)
model.aFRR_neg = pyo.Var(model.H4, within=pyo.PositiveReals)

model.peak = pyo.Var(model.Q, within=pyo.PositiveReals)

## Supporting decision variables

# Decision flows in and out of components
#model.x_grid_in = pyo.Var(model.Q, within=pyo.PositiveReals)
model.x_grid_out = pyo.Var(model.Q, within=pyo.Reals)#, bounds = (-0.1,0.1))

model.x_BESS_in = pyo.Var(model.Q, within=pyo.PositiveReals)#, bounds = (0,1))
model.x_BESS_out = pyo.Var(model.Q, within=pyo.PositiveReals)#, bounds = (0,1))

# model.x_supercap_in = pyo.Var(model.Q, within=pyo.PositiveReals)
# model.x_supercap_out = pyo.Var(model.Q, within=pyo.PositiveReals)

model.SOC_BESS = pyo.Var(model.Q, within=pyo.PositiveReals)

## Constraints

#  Equilibrium constraints (nodal and trade balancing)
model.node_in = pyo.Constraint(model.Q, rule = lambda model, q: 0 == PV_prod[q] - local_load[q]  + model.x_grid_out[q] + model.x_BESS_out[q] - model.x_BESS_in[q]) # + model.x_supercap_out[q] - model.x_supercap_in[q])
#model.node_out = pyo.Constraint(model.Q, rule = lambda model, q: 0 == PV_prod[q] - local_load[q] + model.x_grid_out[q])# + model.x_BESS_out[q] - model.x_BESS_in[q] ) # + model.x_supercap_out[q] - model.x_supercap_in[q])

# Financial balancing: net power from grid must be equal to aggregated trades on both markets. DA is hourly, so divided by 4.
model.balancing_in = pyo.Constraint(model.Q, rule = lambda model, q: model.x_grid_out[q] == model.ID_buy[q] - model.ID_sell[q] + model.DA_buy[q//4]/4 - model.DA_sell[q//4]/4)


# Big M constraint for simulaneous buying and selling:
# Buying and selling simultaneously is not allowed: (ID_buy + DA_buy)*(ID_sell + DA_sell) = 0)
# 
model.y = pyo.Var(model.Q, within=pyo.Binary)#, bounds = (-1,1))
M = 1000
model.const_bigM_ID_sell = pyo.Constraint(model.Q, rule = lambda model, q: (model.ID_sell[q] + model.DA_sell[q//4]/4) <= M*model.y[q])
model.const_bigM_ID_buy = pyo.Constraint(model.Q, rule = lambda model, q: (model.ID_buy[q] + model.DA_buy[q//4]/4) <= M*(1-model.y[q]))

# Alternative big M constraint
# Difference between buying and selling simultaneously must be bigger than certain value: (ID_buy + DA_buy)*(ID_sell + DA_sell) = 0)
# OR, for every hour where we buy and sell simultaneously, we must also buy and buy simultaneously
# OR, for every hour sum of ID_buy must aggregate to 0

# model.const_bigM_alt = pyo.Constraint(model.H, rule = lambda model, h:  sum(model.ID_sell[4*h + k] - model.ID_buy[4*h + k] for k in range(0,4)) == 0 )

# M = 1000 

# Either ID buy or ID sell
# model.y = pyo.Var(model.Q, within=pyo.Binary)#, bounds = (-1,1))
# model.const_bigM_ID1_sell = pyo.Constraint(model.Q, rule = lambda model, q: (model.ID_sell[q]<= M*model.y[q]))
# model.const_bigM_ID1_buy = pyo.Constraint(model.Q, rule = lambda model, q: (model.ID_buy[q] <= M*(1-model.y[q])))

# Either DA buy or DA sell
# model.y1 = pyo.Var(model.Q, within=pyo.Binary)#, bounds = (-1,1))
# model.const_bigM_DA1_sell = pyo.Constraint(model.Q, rule = lambda model, q: (model.DA_sell[q//4]<= M*model.y1[q//4]))
# model.const_bigM_DA1_buy = pyo.Constraint(model.Q, rule = lambda model, q: (model.DA_buy[q//4] <= M*(1-model.y1[q//4])))

# Other alternative big M constraint
# For each hour q0 and q3 must be opposite sign
# M = 1000 
# model.y2 = pyo.Var(model.H, within=pyo.Binary)#, bounds = (-1,1))

# model.const_bigM_altq01 = pyo.Constraint(model.H, rule = lambda model, h:  model.ID_sell[4*h] - model.ID_buy[4*h] <= M*model.y2[h])
# model.const_bigM_altq02 = pyo.Constraint(model.H, rule = lambda model, h:  model.ID_sell[4*h] - model.ID_buy[4*h] >= -1* M*(1-model.y2[h]))

# model.const_bigM_altq31 = pyo.Constraint(model.H, rule = lambda model, h:  model.ID_sell[4*h + 3] - model.ID_buy[4*h + 3] >= -1*M*model.y2[h])
# model.const_bigM_altq32 = pyo.Constraint(model.H, rule = lambda model, h:  model.ID_sell[4*h + 3] - model.ID_buy[4*h + 3] <=  M*(1-model.y2[h]))

# # # Either ID buy or ID sell
# model.y = pyo.Var(model.Q, within=pyo.Binary)#, bounds = (-1,1))
# model.const_bigM_ID1_sell = pyo.Constraint(model.Q, rule = lambda model, q: (model.ID_sell[q]<= M*model.y[q]))
# model.const_bigM_ID1_buy = pyo.Constraint(model.Q, rule = lambda model, q: (model.ID_buy[q] <= M*(1-model.y[q])))

# # Either DA buy or DA sell
# model.y1 = pyo.Var(model.Q, within=pyo.Binary)#, bounds = (-1,1))
# model.const_bigM_DA1_sell = pyo.Constraint(model.Q, rule = lambda model, q: (model.DA_sell[q//4]<= M*model.y1[q//4]))
# model.const_bigM_DA1_buy = pyo.Constraint(model.Q, rule = lambda model, q: (model.DA_buy[q//4] <= M*(1-model.y1[q//4])))


# Storage continuity
def const_SOC_BESS(model, q):
    if q == 0:
        return model.SOC_BESS[q] == 0.5
    else:
        return model.SOC_BESS[q] == (1-SD_BESS)*(model.SOC_BESS[q-1] + (e_BESS*(model.x_BESS_in[q])) - ((1/e_BESS)*model.x_BESS_out[q]))
model.BESS_cont = pyo.Constraint(model.Q, rule = const_SOC_BESS)

# BESS power
model.const_BESS_power_in = pyo.Constraint(model.Q, rule = lambda model, q: model.x_BESS_in[q] <= 0.5)# BESS_c_rate*BESS_capacity)
model.const_BESS_power_out = pyo.Constraint(model.Q, rule = lambda model, q: model.x_BESS_out[q] <= 0.5)# BESS_c_rate*BESS_capacity)

# Supercap power
# model.const_supercap_power_in = pyo.Constraint(model.Q, rule = lambda model, q: model.x_supercap_in[q] <= 0.5)
# model.const_supercap_power_out = pyo.Constraint(model.Q, rule = lambda model, q: model.x_supercap_out[q] <= 0.5)

# SOC limits
model.const_SOC_BESS_min = pyo.Constraint(model.Q, rule = lambda model, q: model.SOC_BESS[q] >= 0)
model.const_SOC_BESS_max = pyo.Constraint(model.Q, rule = lambda model, q: model.SOC_BESS[q] <= BESS_capacity)

# SOC bidding limits (fix)
# model.const_SOC_BESS_min_reg = pyo.Constraint(model.Q, rule = lambda model, q: model.SOC_BESS[q] >= model.PCR[q//16]*0.5 + model.aFRR_pos[q//16]*1)
# model.const_SOC_BESS_min_reg = pyo.Constraint(model.Q, rule = lambda model, q: model.SOC_BESS[q] <= BESS_capacity - model.PCR[q//16]*0.5 + model.aFRR_neg[q//16]*1)

# Peak constraint
model.const_peak = pyo.Constraint(model.Q, rule = lambda model, q: model.peak[q] >= -1*model.x_grid_out[q])

# Degradation here the constants, variables and constraints are defined together
n=5 # Number of segments
alpha = np.random.rand(5)
beta = np.random.rand(5)

model.wq = pyo.Var(model.Q, within=pyo.PositiveReals)

def const_deg_lin(model,q,j):
    return model.wq[q] >= alpha[j]*model.SOC_BESS[q] + beta[j]

model.const_deg_lin = pyo.Constraint(model.Q*pyo.RangeSet(0,n-1,1), rule = const_deg_lin)

Dsh = 0.004 # Calendar degradation

model.dq = pyo.Var(model.Q, within=pyo.PositiveReals)

def const_dq(model,q):
    if q == 0:
        return pyo.Constraint.Skip
    else:
        return model.dq[q] >= model.wq[q] - model.wq[q-1] + Dsh
    
def const_dq_dsh(model,q):
    if q == 0:
        return pyo.Constraint.Skip
    else:
        return model.dq[q] >= Dsh

model.const_dq = pyo.Constraint(model.Q, rule = const_dq)
model.const_dq_dsh = pyo.Constraint(model.Q, rule = const_dq_dsh)

deg_total = sum(model.dq[q] for q in model.Q)


def obj_expression(model):
    return pyo.sum_product(model.ID_buy, pricesQ, index=model.ID_buy) \
          - pyo.sum_product(model.ID_sell, pricesQ, index=model.ID_buy)\
             + pyo.sum_product(model.DA_buy, pricesH, index=model.DA_buy) \
                - pyo.sum_product(model.DA_sell, pricesH, index=model.DA_buy) # + pyo.sum_product(model.PCR[q//16], pricesH4_PCR[q//16]) + pyo.sum_product(model.aFRR_pos[q//16], pricesH4_aFRR_pos[q//16]) + pyo.sum_product(model.aFRR_neg[q//16], pricesH4_aFRR_neg[q//16])# + pyo.sum_product(model.peak[q], p_peak) + pyo.sum_product(model.dq[q], emissions[q])

model.obj = pyo.Objective(rule = obj_expression, sense = pyo.minimize)
solver = pyo.SolverFactory('gurobi') 

options = {
    "MIPGap":  0.0001,
    "OutputFlag": 1
}

results = solver.solve(model,tee=True, options = options)
print('Obj for exclusive buy/sell: '+ str(results['Problem'][0]['Lower bound']))
print('DA_buy: '+ str(sum(model.DA_buy.extract_values().values())))
print('DA_sell: '+ str(sum(model.DA_sell.extract_values().values())))

print('ID_buy: '+ str(sum(model.ID_buy.extract_values().values())))
print('ID_sell: '+ str(sum(model.ID_sell.extract_values().values())))

print('Abs SOC change: ' + str(sum(abs(model.SOC_BESS.extract_values()[q] - model.SOC_BESS.extract_values()[q-1]) for q in model.Q if q > 0)))

# plt.plot(model.SOC_BESS.extract_values().values())
# plt.show()
# print('done')
# print(model.obj())

# pd.DataFrame(model.ID_buy.extract_values())