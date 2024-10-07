class product:
    ## Define products
    ## name: name of the product
    ## product_type: type of the product 1 = power, 2 = energy
    ## length: duration of the product in hours (e.g. 0.25 = 15 minutes)
    ## ramp_min_limits: [ramp_up_min, ramp_down_min] in %/min
    def __init__(self, name, product_type, length, ramp_min_limits):
        self.name = name
        self.product_type = product_type
        self.length = length
        self.ramp_limits = ramp_min_limits

class market:
    ## Define market
    ## name: name of the market
    ## product: product object delivered on market
    ## bidding_time: pattern of producte (1= T-i at fixed time/date, 2= T-i)
    ## bidding_struct: payment structure (1= pay-as-bid, 2= pay-as-clear)
    ## product_pattern: how the product repeats (0= once, 1= continuously repeats (starting at T0 in horizon), 2= repeats every n intervals)
    ## price_t_series: time series of prices
    def __init__(self, name, product, bidding_time, bidding_struct, product_distribution, price_t_series):
        self.name = name
        self.product = product
        self.bidding_time = bidding_time
        self.bidding_struct = bidding_struct
        self.product_distribution = product_distribution
        self.price = price_t_series


class storage_device:
    ## Define storage
    ## name: name of the storage
    ## capacity: capacity of the storage in MWh
    ## power: power of the storage in MW
    ## efficiency: efficiency of the storage in %
    ## ramp_limits: [ramp_up, ramp_down] in %/min
    def __init__(self, name, capacity, power, efficiency, ramp_limits, start_SOC):
        self.name = name
        self.capacity = capacity
        self.power = power
        self.efficiency = efficiency
        self.ramp_limits = ramp_limits
        self.start_SOC = start_SOC

    def constraints():
        print('aaa')


class storage:
    def __init__(self, devices):
        self.devices = devices

    def constraints():
        print('aaa')


# Example usage
if __name__ == "__main__":
    model = product('Primary reserves', 1, 0.25, [10,10])
    print(model.length)
