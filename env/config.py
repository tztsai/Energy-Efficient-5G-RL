from traffic.config import delayBudgets

# simulation parameters
timeStep = 1e-3  # size of a simulation step in seconds
startTime = 0  # time in a week in seconds (peak time: 307800 = Thu 13:30)
actionInterval = 20  # sim steps between two actions (1 env step = 20 sim steps)
accelRate = 6000  # equivalent seconds in real time for 1 second in simulation time
episodeTimeLen = 60 * 60 * 24 * 7  # the duration of an episode in simulation (seconds in a week)

# traffic parameters
trafficScenario = 'RANDOM'  # 'A', 'B', 'C'

# action parameters
numAntSwitchOpts = 3
numSleepModes = 4
numConnModes = 3

# reward parameters
# dropAppWeights = [66., 27., 7.]  # weighted sum in the range of [0, 100]
# delayAppWeights = [1/t/3 for t in delayBudgets]  # weighted average in the range of [0, 1]
# dropRatioWeight = 0.5  # penalty for average drop ratio = 1%
# delayWeight = 0.1  # penalty for average delay = delay budget
# pcWeight = 1e-3  # PC (unit: W) penalty weight
qosWeight = 4  # QoS reward weight
extraQosWeight = 0.005 # QoS reward weight for done UEs
