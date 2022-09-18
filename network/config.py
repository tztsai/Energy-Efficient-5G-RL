import numpy as np
from utils import dB2lin, lin2dB

renderMode = 'none'

# base station params
numBS = 7
interBSDist = 400  # the distance between two adjacent BSs
cellRadius = 600  # the radius of a hexagon cell in meters
antennaPower = 0.2  # average antenna power in watts (src: Energy Saving Game for Massive MIMO)
maxAntPower = 20 * antennaPower  # maximum antenna power in watts
bsFrequency = 5e9  # carrier frequency in Hz
feederLoss = 1  # feeder loss in dB (XXX: include in antennaGain)
antennaGain = 19 - feederLoss  # power gain in dB of each antenna of a BS
totalAntennas = 64  # max number of antennas
bandWidth = 20e6  # communication bandwidth in Hz
bsHeight = 30  # height of a BS in meters
# powerAllocWeights = [95, 4, 1]  # weights of the power allocation
powerAllocBase = 2.
antennaSwitchOpts = [-16, -4, 0, 4, 16]
sleepModeDeltas = [1, 0.69, 0.50, 0.29]
wakeupDelays = [0, 1e-3, 1e-2, 1e-1]
antSwitchEnergy = 5e-3  # energy consumption of switch per antenna in Joules
sleepSwitchEnergy = [0, 0.01, 0.01, 0.01]  # energy consumption of switching sleep mode in Joules
disconnectEnergy = 0.02  # energy consumption of an early disconnection in Joules
bufferShape = (50, 2)  # shape of the buffer used to record past observations 
bufferChunkSize = 5  # chunk size to apply average pooling
bufferNumChunks = bufferShape[0] // bufferChunkSize

# channel model params
noisePower = bandWidth * dB2lin(-174 - 30 + 7)
# thermal noise variance

# user equipment params
ueHeight = 1.5  # height of a UE in meters

# default network configuration
areaSize = np.array([2.5, 2.5]) * interBSDist
bsPositions = np.vstack([
    np.array(
        [[areaSize[0]/2 + interBSDist * np.cos(a + np.pi/6),
          areaSize[1]/2 + interBSDist * np.sin(a + np.pi/6)]
         for a in np.linspace(0, 2*np.pi, 7)[:-1]]),
    # np.array(
    #     [[AREA[0]/2 + R * 3 * np.cos(a),
    #       AREA[1]/2 + R * 3 * np.sin(a)]
    #      for a in np.linspace(0, 2*np.pi, 13)[:-1]]),
    areaSize[None, :] / 2])

# obs names
public_obs_keys = ['num_antennas', 'responding', 'sleep_mode']
buffer_record_keys = ['pc', 'arrival_rate']
private_obs_keys = ['next_sleep_mode', 'wakeup_time',
                    *[f'{k}{i}' for i in range(-bufferNumChunks, 0) for k in buffer_record_keys],
                    'num_served', 'num_queued', 'num_idle', 'num_covered',
                    'thrp_served', 'thrp_covered', 'log_ratio_served', 'log_ratio_covered',
                    'thrp_req_served', 'thrp_req_queued', 'thrp_req_idle', 'thrp_req_covered']
mutual_obs_keys = ['dist', 'own_thrp_req', 'own_log_ratio', 'other_thrp_req', 'other_log_ratio']
other_obs_keys = [f'nb{i}_{k}' for i in range(numBS - 1) for k in public_obs_keys + mutual_obs_keys]
all_obs_keys = public_obs_keys + private_obs_keys + other_obs_keys
