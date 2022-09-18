import numpy as np
from traffic import TrafficType

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
dropAppWeights = [.75, .2, .05]
delayAppWeights = [.66, .22, .12]
dropRatioWeight = 4.
delayWeight = 2.  # per second
powerConsumptionWeight = 1e-3  # per watt
