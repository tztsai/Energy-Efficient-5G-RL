import numpy as np
from traffic import TrafficType

# simulation parameters
timeStep = 1e-3  # size of a simulation step in seconds
startTime = 0  # time in a week in seconds (peak time: 307800 = Thu 13:30)
actionInterval = 20  # sim steps between two actions (1 env step = 10 sim steps)
accelRate = 600  # 1h in real time = 6s (600 env steps) in simulation time
statsInterval = 5  # env steps between two stats printouts
episodeTimeLen = 60 * 60 * 24 * 7  # the duration of an episode in simulation (seconds in a week)

# traffic parameters
trafficType = 'A'

# action parameters
numAntSwitchOpts = 5
numSleepModes = 4
numConnModes = 4

# reward parameters
droppedAppWeights = [.6, .3, .1]
droppedTrafficWeight = 0.01
powerConsumptionWeight = 1.
