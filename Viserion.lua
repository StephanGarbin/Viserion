require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cudnn'
require 'cutorch'
local ViserionTrainer = require 'Viserion/ViserionTrainer'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-ioFile', '', 'Script specifing IO')
cmd:option('-modelFile', '', 'Script defining model OR .t7 archive to load')
cmd:option('-criterionFile', '', 'Script defining criterion to optimise')
cmd:option('-optimFile', '', 'Script defining optimisation functions')  
cmd:option('-doTraining', false, 'Training/Test switch')
cmd:option('-batchSize', 1, 'Batch size')
cmd:option('-numEpochs', 100, '#Epochs')
cmd:option('-startEpoch', 0, 'Starting epoch for training')
cmd:option('-numThreads', 1, '#Threads for data loading')
cmd:option('-saveStateInterval', -1, 'Interval in which saveState function in your IO script is called. Set to -1 to disable')
cmd:option('-passFullOutput2saveState', false, 'Set this to false so that full state output is not passed to the saveState function in your IO script.')
cmd:option('-disableCUDNN', false, 'Use this to disable the CUDNN backend')

opts = cmd:parse(arg)

print(opts)

torch.setdefaulttensortype('torch.FloatTensor')

cudnn.benchmark = true
cudnn.fastest = true

--DATA HANDLING
print('Loading Data...')
dofile(opts.ioFile)

--DEFINE MODEL
if(opts.doTraining) then
	print('Creating Model...(NOTE: Your IO File should respect this!)')
else
	print('Loading Model...(NOTE: Your IO File should respect this!)')
end

mFile = loadfile(opts.modelFile)
	mFile()

--DEFINE CRITERION
print('Creating Criterion...')
cFile = loadfile(opts.criterionFile)
cFile()

--ENABLE GPU SUPPORT
print('Converting Criterion to CUDA...')
criterion:cuda()
print('Converting Model to CUDA...')
model:cuda()
if not opts.disableCUDNN then
	print('Converting Model to CUDNN...')
	cudnn.convert(model, cudnn)
end
print(model)

--DEFINE OPTIMISATION
print('Defining Optimisation Parameters...')
oFile = loadfile(opts.optimFile)
oFile()

--CREATE TRAINER
local trainer = ViserionTrainer(model, criterion, optimOptimiser, optimConfig, defineCustomLearningRate, opts)

print('Finished all preliminaries...\n')
--TRAIN
if(opts.doTraining) then
	print('Starting training from epoch ' .. tostring(opts.startEpoch) .. '... ')

	lossAll = torch.Tensor(opts.numEpochs)

	for epoch = opts.startEpoch, opts.numEpochs do

		-- Train
		trainer:train(epoch, trainDataLoader)

		-- Test
		local loss = trainer:test(epoch, testDataLoader, opts.passFullOutput2saveState)

		lossAll[epoch + 1] = loss

		if(opts.saveStateInterval > 0) then
			if(epoch % opts.saveStateInterval == 0 and epoch > 0) then
				saveState(epoch, loss, trainer.testOutput)
			end
		end
	end
	
	saveState(epoch, loss, trainer.testOutput)
	print(lossAll)
else
	print('Just testing ' .. tostring(opts.startEpoch) .. '... ')
	-- Test
	local loss = trainer:test(opts.startEpoch, testDataLoader, true)
	saveTestState(epoch, loss, trainer.testOutput)
end



