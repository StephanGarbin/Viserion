require 'nn'

--From torch tutorial https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua

model = nn.Sequential()
------------------------------------------------------------
-- convolutional network 
------------------------------------------------------------
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolution(1, 64, 5, 5))
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolution(64, 128, 5, 5))
model:add(nn.SpatialBatchNormalization(128))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 3 : standard 3-layer MLP:
local numE = 128*2*2
model:add(nn.Reshape(numE))
model:add(nn.Linear(numE, numE))
model:add(nn.Dropout())
model:add(nn.ReLU())
model:add(nn.Linear(numE, numE))
model:add(nn.Dropout())
model:add(nn.ReLU())
model:add(nn.Linear(numE, 10))
------------------------------------------------------------
model:add(nn.LogSoftMax())




--[[NOTE: All we need to do here is to define an nn model called 'model'.
This can be an nn.gModule and Viserion will act accordingly 
(for example, print out graphs of the model structure - don't forget to use -modelName)]]--

--[[NOTE: To have different behaviours for training and testing create the model only when opts.doTraining is true,
load it from disk otherwise. This could also be conditional on opts.startEpoch]]--