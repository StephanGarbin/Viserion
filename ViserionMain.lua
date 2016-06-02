require 'torch'
require 'nn'
require 'optim'
require 'cunn'

if not opts.disableCUDNN then 
	require 'cudnn'
end

require 'cutorch'
require 'nngraph'

autograd = require 'autograd'

require 'Viserion/ViserionPlotting'

ViserionTrainer = require 'Viserion/ViserionTrainer'

function printDebug(m)
	if opts.debug then
		print('DEBUG: ', m)
	end
end

function printWarning(m)
	print('WARNING: ', m)
end

function printError(m)
	print('ERROR: ', m)
end

VISERION_TERMINATED = false
function ViserionTerminate()
	VISERION_TERMINATED = true
end

function getCurrentMinibatchLoss()
	if trainer ~= nil then
		return trainer.getCurrentMinibatchLoss()
	else
		printError('Cannot call \'getCurrentMinibatchLoss\', trainer is not created yet')
		return nil
	end
end

if opts.batchSize < 2 then
	printWarning("Batch sizes of 1 are highly discouraged due to internal indexing!")
end

torch.setdefaulttensortype('torch.FloatTensor')

if not opts.disableCUDNN then 
	cudnn.benchmark = true
	cudnn.fastest = true
	print('Using cudnn version: ', cudnn.version)
	if opts.cudnnVerbose then
		cudnn.verbose = true
	end
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

--Create DPT if multi-gpu use is enabled
local function separateDigits(x)
	components = {}
	counter = 0
	while x > 0 do
		table.insert(components, x%10)
		x = math.floor(x / 10)
		counter = counter + 1
	end
	return counter, components
end

opts.numGPUs, opts.gpuIDXs = separateDigits(opts.specifyGPUs)

if not opts.disableCUDA then
	if opts.numGPUs > 1 then
		print('Using GPUs: ', opts.gpuIDXs)
		print('GPU Host: ', opts.gpuIDXs[1])
		--enable flattenParams and NCLL,... splitting the minibatch!
		model = nn.DataParallelTable(1, true, true):add(model, opts.gpuIDXs)
		--potentially disable ncll
		--enable asyncronous kernel launches
		local options = opts
		local nnCpy = nn
		local cunnCpy = cunn
		local cudnnCpy = cudnn
		local cutorchCpy = cutorch
		if not opts.disableThreadedGPUCopies then
			print('Spawning ' .. tostring(opts.numGPUs) .. ' Threads for CUDA MemCpy operations...')
			print('WARNING: If this causes FATAL_THREAD_PANIC, use -disableThreadedGPUCopies')

			model:threads(function(idx)
				print('Spawning Thread', idx)	
				local nn = require 'nn'
				local cudnn = require 'cudnn'
				local cunn = require 'cunn'
				local cutorch = require 'cutorch'
				local torch = require 'torch'
				local nngraph = require 'nngraph'
				cudnn.benchmark = true
				cudnn.fastest = true
				print('Using cudnn version: ', cudnn.version)
				if options.cudnnVerbose then
	        			cudnn.verbose = true
				end
				return idx
				end)
		end
	else
		print('Using GPU: ', opts.gpuIDXs[1])
	end
	cutorch.setDevice(opts.gpuIDXs[1])
end

--DEFINE CRITERION
print('Parsing Criterion Defintions:')
dofile(opts.criterionFile)

--DETERMINE IF COMPLICATED MODEL IS BEING USED
if(criteria == nil and defineModelFlow == nil) then
	--do nothing, this is the default case
	print('\tUsing single Criterion ...')
	opts.usingMultiCriteria = false
else
	if criterion ~= nil then
		--CHECK that everything is defined correctly!
	end

	print('\tUsing Multi-Criteria setup and custom data flow in optimisation...')
	opts.usingMultiCriteria = true
end


--ENABLE GPU SUPPORT
if not opts.disableCUDA then
	if opts.usingMultiCriteria then
		print('Converting Criteria to CUDA...')
		for i, c in ipairs(criteria) do
			c:cuda()
		end
	else
		print('Converting Criterion to CUDA...')
		criterion:cuda()
	end

	print('Converting Model to CUDA...')
	model:cuda()
	if not opts.disableCUDNN then
		print('Converting Model to CUDNN...')
		cudnn.convert(model, cudnn)
		if opts.usingMultiCriteria then
			print('Converting Criteria to CUDNN...')
			for i, c in ipairs(criteria) do
				cudnn.convert(c, cudnn)
			end
		else
			print('Converting Criterion to CUDNN...')
			cudnn.convert(criterion, cudnn)
		end
	end
end
print(model)

if torch.typename(model) == 'nn.gModule' then
	print('Saving graphics of the defined model...')
	graph.dot(model.fg, opts.modelName .. '_fg', opts.modelName .. '_fg')
	graph.dot(model.bg, opts.modelName .. '_bg', opts.modelName .. '_bg')
end

--DEFINE OPTIMISATION
print('Defining Optimisation Parameters...')
dofile(opts.optimFile)

--CREATE TRAINER
if not opts.usingMultiCriteria then
	trainer = ViserionTrainer(model, criterion, optimOptimiser, optimConfig, defineCustomLearningRate, opts)
else
	trainer = ViserionTrainer(model, criteria, optimOptimiser, optimConfig, defineCustomLearningRate, opts)
end

print('Finished all preliminaries...\n')
--TRAIN
if(opts.doTraining) then
	print('Starting training from epoch ' .. tostring(opts.startEpoch) .. '... ')

	--Save the loss in training and testing
	if not opts.usingMultiCriteria then
		overallLossTrain = torch.Tensor(opts.numEpochs):fill(0)
		overallLossTest = torch.Tensor(opts.numEpochs):fill(0)
	end
	
	for epoch = opts.startEpoch, opts.numEpochs do

		if VISERION_TERMINATED == true then
			printWarning('Viserion terminated by user')
			break
		end

		-- Train
		lossTrain = trainer:train(epoch, trainDataLoader)

		if opts.debug then
			print('DEBUG: Training completed, switching to testing')
		end

		if VISERION_TERMINATED == true then
			printWarning('Viserion terminated by user')
			break
		end
		
		-- Test
		lossTest = trainer:test(epoch, testDataLoader, opts.passFullOutput2saveState)

		if(opts.saveStateInterval > 0) then
			if(epoch % opts.saveStateInterval == 0 and epoch > 0) then
				if opts.debug then
					print('DEBUG: calling saveState()')
				end
				saveState(epoch, lossTrain, lossTest, trainer.testOutput)
			end
		end

		if opts.enablePlots then
			--do print out automatically if necessary
			if not opts.usingMultiCriteria then
				if opts.debug then
					print('DEBUG: generating figures for loss')
					print('DEBUG: Note: there are known issues with gnuplot that can make this segfault')
				end
				overallLossTrain[epoch] = lossTrain
				overallLossTest[epoch] = lossTest
				plotLoss(overallLossTrain, opts.modelName .. '_Train')
				plotLoss(overallLossTest, opts.modelName .. '_Test')
			end
		end
	end

	saveState(opts.numEpochs, lossTrain, lossTest, trainer.testOutput)
else
	print('Just testing ' .. tostring(opts.startEpoch) .. '... ')
	-- Test
	local loss = trainer:test(opts.startEpoch, testDataLoader, true)
	saveState(opts.startEpoch, loss, trainer.testOutput)
end
