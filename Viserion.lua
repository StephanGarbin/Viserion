require 'torch'
require 'nn'
require 'optim'
require 'cunn'
require 'cudnn'
require 'cutorch'
autograd = require 'autograd'

local ViserionTrainer = require 'Viserion/ViserionTrainer'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-ioFile', '', 'Script specifing IO')
cmd:option('-modelFile', '', 'Script defining model')
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
cmd:option('-disableCUDA', false, 'Uses the CPU instead')
cmd:option('-specifyGPUs', 1, 'Specify which GPUS on the system to use, for example, to use 3 and 4, use 34')
cmd:option('-multiThreadGPUCopies', false, 'Faster for nn.Sequential modules, but does not work for nn.gModules at the moment')
cmd:option('-debug', false, 'Prints detailed debug output to identify where bugs are occuring')
opts = cmd:parse(arg)

print(opts)

torch.setdefaulttensortype('torch.FloatTensor')

cudnn.benchmark = true
cudnn.fastest = true
print('Using cudnn version: ', cudnn.version)
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
		model = nn.DataParallelTable(1, false, false):add(model, opts.gpuIDXs)
		--potentially disable ncll
		--enable asyncronous kernel launches
		local options = opts
		local nnCpy = nn
		local cunnCpy = cunn
		local cudnnCpy = cudnn
		local cutorchCpy = cutorch
		if opts.multiThreadGPUCopies then
			print('Spawning ' .. tostring(opts.numGPUs) .. ' Threads for CUDA MemCpy operations...')
			print('WARNING: If this causes FATAL_THREAD_PANIC, disable -multiThreadGPUCopies')

			model:threads(function(idx)
				print('Spawning Thread', idx)	
				local nn = require 'nn'
				local cudnn = require 'cudnn'
				local cunn = require 'cunn'
				local cutorch = require 'cutorch'
				local torch = require 'torch'
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
		cutorch.setDevice(opts.gpuIDXs[1])
		print('Using GPU: ', opts.gpuIDXs[1])
	end
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

graph.dot(model.fg, 'myModel_fg', 'myModel_fg')
graph.dot(model.bg, 'myModel_bg', 'myModel_bg')

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

	for epoch = opts.startEpoch, opts.numEpochs do

		-- Train
		trainer:train(epoch, trainDataLoader)

		if opts.debug then
			print('DEBUG: Training completed, switching to testing')
		end

		-- Test
		trainer:test(epoch, testDataLoader, opts.passFullOutput2saveState)

		if(opts.saveStateInterval > 0) then
			if(epoch % opts.saveStateInterval == 0 and epoch > 0) then
				if opts.debug then
					print('DEBUG: calling saveState()')
				end
				saveState(epoch, loss, trainer.testOutput)
			end
		end
	end

	saveState(opts.numEpochs, loss, trainer.testOutput)
	print(lossAll)
else
	print('Just testing ' .. tostring(opts.startEpoch) .. '... ')
	-- Test
	local loss = trainer:test(opts.startEpoch, testDataLoader, true)
	saveState(opts.startEpoch, loss, trainer.testOutput)
end
