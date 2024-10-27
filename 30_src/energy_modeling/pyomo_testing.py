import pyomo.environ as pyo
from pyomo.util.model_size import build_model_size_report

model = pyo.ConcreteModel()

model.D = pyo.RangeSet(0,10)

model.x = model.D * model.E