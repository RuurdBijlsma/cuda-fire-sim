import numpy as np
import cuda_python

width = 3
height = 4
timeSteps = 5
checkpoints = 3
weatherElements = 2
psoParams = 500
batchSize = 500
landCoverTypes = 8

landCoverGrid = np.zeros((width, height), dtype=np.int16)
elevation = np.ones((width, height), dtype=np.int16) * 2
fire = np.ones((width, height, checkpoints), dtype=bool)
weather = np.ones((width, height, timeSteps, weatherElements), dtype=np.float64) * 4
psoConfigs = np.ones((psoParams, batchSize), dtype=float) * 5
landCoverRates = np.ones((landCoverTypes, batchSize), dtype=float) * 6

landCoverGrid[0, :] = 1

print("Sending data!!!!\n", weather)
result = cuda_python.batch_simulate(landCoverGrid, elevation, fire, weather, psoConfigs, landCoverRates)
print("Result from batch_simulate = " + str(result))
