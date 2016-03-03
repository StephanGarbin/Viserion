require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cudnn'
require 'cutorch'
local ViserionTrainer = require 'Viserion/ViserionTrainer'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-name', 'project', 'Project name')
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
cmd:option('-cudnnVerbose', false, 'Enable verbose output for CUDNN debug')
cmd:option('-log', false, 'Enable logging')

opts = cmd:parse(arg)

print(opts)

torch.setdefaulttensortype('torch.FloatTensor')

cudnn.benchmark = true
cudnn.fastest = true
if opts.cudnnVerbose then
	cudnn.verbose = true
end

--DATA HANDLING
print('Loading Data...')
dofile(opts.ioFile)

--DEFINE MODEL
if(opts.doTraining) then
	print('Creating Model...(NOTE: Your IO File should respect this!)')
else
	print('Loading Model...(NOTE: Your IO File should respect this!)')
end

--mFile = loadfile(opts.modelFile)
--mFile()
dofile(opts.modelFile)

--DEFINE CRITERION
print('Creating Criterion...')
--cFile = loadfile(opts.criterionFile)
--cFile()
dofile(opts.criterionFile)

--ENABLE GPU SUPPORT
print('Converting Criterion to CUDA...')
criterion:cuda()
print('Converting Model to CUDA...')
model:cuda()
if not opts.disableCUDNN then
	print('Converting Model to CUDNN...')
	cudnn.convert(model, cudnn)
	cudnn.convert(criterion, cudnn)
end
print(model)

--CREATE LOGGER
if opts.log then
	logger = optim.Logger(opts.name .. '.log')
	logger:setNames{'training loss', 'test loss'}
	logger:style{'-', '-'}
	logger.showPlot = false
end

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
		local train_loss = trainer:train(epoch, trainDataLoader)

		-- Test
		local test_loss = trainer:test(epoch, testDataLoader, opts.passFullOutput2saveState)

		lossAll[epoch + 1] = test_loss

		if(opts.saveStateInterval > 0) then
			if(epoch % opts.saveStateInterval == 0 and epoch > 0) then
				saveState(epoch, test_loss, trainer.testOutput)
			end
		end

		-- Log and plot
		if opts.log then
			logger:add{train_loss, test_loss}
			logger:plot()
		end

	end

	saveState(opts.numEpochs, test_loss, trainer.testOutput)
	print(lossAll)
else
	print('Just testing ' .. tostring(opts.startEpoch) .. '... ')
	-- Test
	local test_loss = trainer:test(opts.startEpoch, testDataLoader, true)
	saveState(opts.startEpoch, test_loss, trainer.testOutput)
end
