require 'optim'

optimOptimiser = optim.nag

optimConfig = {}
optimConfig.learningRate = 1e-3
optimConfig.learningRateDecay = 0.0
optimConfig.weightDecay = 1e-4
optimConfig.momentum = 0.9
optimConfig.dampening = 0.0
optimConfig.nesterov = true


function defineCustomLearningRate(epoch, currentLearningRate)
	return currentLearningRate
end

--[[NOTE: The table optimConfig should be specific to the optimiser you are using!
Feel free to add if-statements inside defineCustomLearningRate() to for example cut the learning rate after 50 epochs]]--