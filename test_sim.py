import numpy as np
import cuda_python

cuda_python.main()
print("DONE")

# width = 500
# height = 500
# timeSteps = 300
# weatherElements = 2
# psoParams = 10
# batchSize = 1
# landCoverTypes = 8
#
# landCoverGrid = np.zeros((width, height), dtype=np.int16) * 1
# landCoverRates = np.ones((landCoverTypes, batchSize), dtype=np.float64) * 1
# elevation = np.ones((width, height), dtype=np.int16) * 3
# fire = np.zeros((width, height), dtype=bool)
# weather = np.ones((width, height, timeSteps, weatherElements), dtype=np.float64) * 2
# params = np.ones((psoParams, batchSize), dtype=np.float64) * 1
#
# params[0, :] = 0.3
# params[1, :] = 0.1
# params[2, :] = 0.1
# params[3, :] = 2
# params[4, :] = 0.1
# params[5, :] = 2
# params[6, :] = .2
# params[7, :] = 1
# params[8, :] = .1
# params[9, :] = 1
#
# fire[int(width / 2) - 1:int(width / 2) + 1, int(height / 2) - 1:int(height / 2) + 1] = True
# print("FIRE", fire[:, :])
#
# # print("Sending data!!!!\n", fire)
# result = cuda_python.batch_simulate(landCoverGrid, landCoverRates, elevation, fire, weather, params)
# grid = result.squeeze()
# print("Result from batch_simulate = ", grid)
