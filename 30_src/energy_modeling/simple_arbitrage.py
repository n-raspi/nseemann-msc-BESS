import pyomo.environ as pyo

prices = [10,3,4,2,5,67,4,3,2,6,4,90,10,3,4,2,5,67,4,3,2,6,4,90]

SOC0 = 0.5
SOCmin = 0
SOCmax = 1
power = 0.5
# Create a model
model = pyo.ConcreteModel()
print('asdf')
model.hours = pyo.RangeSet(0, 23)
model.possible_trades = pyo.RangeSet(-power,power,0.1*power)


# Decision variable for trading up or down
model.trade = pyo.Var(model.hours, within=model.possible_trades)

# Decision variable for SOC
model.SOC = pyo.Var(model.hours, within=pyo.NonNegativeReals)

# Constraint for continuity of SOC
def con_SOC(model,h):
    if h < 23:
        return model.SOC[h+1] == model.SOC[h] + model.trade[h]
    else:
        return pyo.Constraint.Skip
model.con_SOC = pyo.Constraint(model.hours, rule = con_SOC)


# Constraint for initial SOC
model.con_0 = pyo.Constraint(rule = model.SOC[0] == SOC0)

# Constraint for maxSOC adnd minSOC
def con_min_SOC(model, h):
    return model.SOC[h] >= SOCmin
model.con_min = pyo.Constraint(model.hours, rule=con_min_SOC)
def con_max_SOC(model, h):
    return model.SOC[h] <= SOCmax
model.con_max = pyo.Constraint(model.hours, rule=con_max_SOC)

model.con_max_end = pyo.Constraint(rule=model.SOC[23] + model.trade[23] <= SOCmax)

# Define the objective function
def obj_expression(model):
    return sum(prices[h] * model.trade[h] for h in model.hours)
model.obj = pyo.Objective(rule=obj_expression, sense=pyo.maximize)

# Solve the model
solver = pyo.SolverFactory('glpk')
solver.solve(model)

# Print the results
print(f"trades = {list(model.trade.extract_values().values())}")
print(f"SOC = {list(model.SOC.extract_values().values())}")
print(f"Objective = {model.obj()}")
print(model.obj())

