from traffic.config import delayBudgets

# simulation parameters
timeStep = 1e-3  # size of a simulation step in seconds
startTime = 0  # time in a week in seconds (peak time: 307800 = Thu 13:30)
actionInterval = 20  # sim steps between two actions (1 env step = 20 sim steps)
accelRate = 600  # 1h in real time = 6s (600 env steps) in simulation time
episodeTimeLen = 60 * 60 * 24 * 7  # the duration of an episode in simulation (seconds in a week)

# traffic parameters
trafficScenario = 'B'  # 'A', 'B', 'C'

# action parameters
numAntSwitchOpts = 5
numSleepModes = 4
numConnModes = 3

# reward parameters
dropAppWeights = [.5, .3, .2]  # normalized to [0, 1]
delayAppWeights = [1/t/len(delayBudgets) for t in delayBudgets]  # normalized to [0, 1]
dropRatioWeight = 3.
delayWeight = 0.1
powerConsumptionWeight = 1e-3  # per watt
