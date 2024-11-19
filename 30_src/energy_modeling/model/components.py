class storage:
    def __init__(self, name, capacity_MWh, max_charge_MW, max_discharge_MW, eff_charge, eff_discharge, SD, degradation_eqtn, degradation_sh, financial_params, CO2_params):
        self.name = name
        self.capacity_MWh = capacity_MWh
        self.max_charge_MW = max_charge_MW
        self.max_discharge_MW = max_discharge_MW
        self.eff_charge = eff_charge
        self.eff_discharge = eff_discharge
        self.SD = SD
        self.degradation_eqtn = degradation_eqtn # [[alpha0, beta0], [alpha1, beta1], [alpha2, beta2]]...
        self.degradation_sh = degradation_sh # 
        self.financial_params = financial_params # LCA cost at 100 & 80 SOH
        self.CO2_params = CO2_params # CO2 for manufacturing and recycling

# class market:
#     def __init__(self, name, type, timescale, data):
#         self.name = name # IP, DA, aFRR, FCR
#         self.type = type # energy or capacity
#         self.timescale = timescale # 15min, 1h, 4h
#         self.data = data # dataframe

# class timescale:
#     def __init__(self, name, interval_duration):
#         self.name = name
#         self.interval_duration = interval_duration

# class fixed_profile:
#     def __init__(self, name, timescale, data):
#         self.name = name
#         self.timescale = timescale
#         self.data = data