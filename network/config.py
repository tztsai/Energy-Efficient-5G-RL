import numpy as np
from utils import dB2lin, lin2dB

renderMode = 'none'

# base station params
numBS = 7
interBSDist = 200  # the distance between two adjacent BSs
cellRadius = 500  # the radius of a hexagon cell in meters
antennaPower = 0.2  # maximum BS power in watts (src: Energy Saving Game for Massive MIMO)
bsFrequency = 1.9e9  # carrier frequency in Hz
antennaGain = 18  # power gain in dB of each antenna of a BS
# feederLoss = 1  # feeder loss in dB (XXX: included in antennaGain)
numAntennas = 64  # max number of antennas
bandWidth = 20e6  # communication bandwidth in Hz
bsHeight = 30  # height difference between a BS and a user in meters
powerAllocWeights = [128, 2, 1]  # weights of the power allocation
# powerAllocWeights = [1, 1, 1]  # weights of the power allocation
antennaSwitchOpts = [-16, -4, 0, 4, 16]
sleepDiscounts = [1, 0.195, 0.114, 0.076]
wakeupDelays = [0, 1e-3, 1e-2, 1]
antSwitchEnergy = 0.02  # energy consumption of switch per antenna in Joules
sleepSwitchEnergy = [0, 0.02, 0.04, 0.4]  # energy consumption of switching sleep mode in Joules
disconnectEnergy = 0.03  # energy consumption of an early disconnection in Joules

# channel model params
noiseVariance = bandWidth * dB2lin(-174 - 30 + 7)
# thermal noise variance

# user equipment params
ueHeight = 1.5  # height of a user equipment in meters

# default network configuration
areaSize = np.array([4, 4]) * interBSDist
bsPositions = np.vstack([
    np.array(
        [[areaSize[0]/2 + interBSDist * np.sqrt(3) * np.cos(a + np.pi/6),
          areaSize[1]/2 + interBSDist * np.sqrt(3) * np.sin(a + np.pi/6)]
         for a in np.linspace(0, 2*np.pi, 7)[:-1]]),
    # np.array(
    #     [[AREA[0]/2 + R * 3 * np.cos(a),
    #       AREA[1]/2 + R * 3 * np.sin(a)]
    #      for a in np.linspace(0, 2*np.pi, 13)[:-1]]),
    areaSize[None, :] / 2])
