import pyomo.environ as pyo


# Create a model
model = pyo.ConcreteModel()

# Define decision variables
model.x = pyo.Var(within=pyo.NonNegativeReals)
model.v = pyo.Var(within=pyo.NonNegativeReals)

# Define the objective function
model.obj = pyo.Objective(expr=2*model.x + 3*model.v, sense=pyo.maximize)

# Define constraints
model.con1 = pyo.Constraint(expr=model.x + model.v <= 10)
model.con2 = pyo.Constraint(expr=model.x - model.v >= 3)

# Solve the model
solver = pyo.SolverFactory('glpk')
solver.solve(model)

# Print the results
print(f"x = {model.x()}")
print(f"v = {model.v()}")
print(f"Objective = {model.obj()}")
print(model.obj())
