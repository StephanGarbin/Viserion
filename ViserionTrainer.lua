require 'optim'
require 'torch'
require 'xlua'

local X = {}
local ViserionTrainer = torch.class('ViserionTrainer', X)

function ViserionTrainer:__init(model, criterion, optimOptimiser, optimOptions, learningRateFunc, options)
	self.model = model
	self.criterion = criterion
	self.optimOptimiser = optimOptimiser
	self.optimOptions = optimOptions
	self.learningRateFunc = learningRateFunc
	self.opts = options
	self.params, self.gradParams = model:getParameters()
end

function ViserionTrainer:train(epoch, dataloader)
	--Scale learning rate if required
	self.optimOptions.learningRate = defineCustomLearningRate(epoch, self.optimOptions.learningRate)

	--The evaluation function for the optimiser
	local function feval()
		return self.criterion.output, self.gradParams
	end

	--Time what we do
	local modelTimer = torch.Timer()
	local dataTimer = torch.Timer()

	local numBatches = math.ceil(dataloader:size() / self.opts.batchSize)

	--Store progression of loss, time to load data & to execute
	local loss = torch.Tensor(numBatches)
	local avgModelTime = 0
	local avgDataTime = 0

	--Switch Model to training
	self.model:training()

	print('TRAIN: Processing Epoch # ' .. epoch .. ' (LR = ' .. tostring(self.optimOptions.learningRate) .. ')')

	ProgressBarStep = 1
	--Process all batches
	for n, sample in dataloader:run() do
		avgDataTime = avgDataTime + dataTimer:time().real

		xlua.progress(ProgressBarStep, numBatches)

		--Copy input and target to the GPU
		self:cudaDeviceCopy(sample)

		--Do forward pass
		self.model:forward(self.input)

		--Compute loss
		local local_loss = self.criterion:forward(self.model.output, self.target)
     		loss[n] = local_loss
		--Erase prev gradient params
		self.model:zeroGradParameters()

		--Do backward pass
		self.criterion:backward(self.model.output, self.target)
		self.model:backward(self.input, self.criterion.gradInput)

		--Do Optim step
      	self.optimOptimiser(feval, self.params, self.optimOptions)

		--Save some debug info
		avgModelTime = avgModelTime + modelTimer:time().real

		modelTimer:reset()
		dataTimer:reset()
		ProgressBarStep = ProgressBarStep + 1
	end

	print('\n')
	print('Loss = ' .. tostring(loss:mean()))
	--print('Avg Model Time = ' .. tostring(avgModelTime / numBatches))
	--print('Avg Data Time = ' .. tostring(avgDataTime / numBatches))
	print('\n')
end

function ViserionTrainer:test(epoch, dataloader, saveTestOutput)
	--Time what we do
	local modelTimer = torch.Timer()
	local dataTimer = torch.Timer()

	local numBatches = math.ceil(dataloader:size() / self.opts.batchSize)

	--Store progression of loss, time to load data & to execute
	local loss = torch.Tensor(numBatches)
	local avgModelTime = 0
	local avgDataTime = 0

	if saveTestOutput then
		self.testOutput = torch.Tensor(dataloader:ySize())
	else
		self.testOutput = {}
	end

	--Switch Model to evaluation
	self.model:evaluate()

	print('TEST: Processing Epoch # ' .. epoch)

	ProgressBarStep = 1
	--Process all batches
	for n, sample in dataloader:runNoShuffle() do

		xlua.progress(ProgressBarStep, numBatches)

		local dataTime = dataTimer:time().real

		--Copy input and target to the GPU
		self:cudaDeviceCopy(sample)

		--Do forward pass
		self.model:forward(self.input)

		--Compute loss
		local local_loss = self.criterion:forward(self.model.output, self.target)
     		loss[n] = local_loss

     	--Save data if required
     	if saveTestOutput then
     		for i = 1, (#sample.target)[1] do
     			self.testOutput[(n - 1) * self.opts.batchSize + i] = self.model.output:float()[i]
     		end
     	end

		--Save some debug info
		avgModelTime = avgModelTime + modelTimer:time().real
		avgDataTime = avgDataTime + dataTimer:time().real

		modelTimer:reset()
		dataTimer:reset()

		ProgressBarStep = ProgressBarStep + 1
	end

	print('\n')
	print('Loss = ' .. tostring(loss:mean()))
	--print('Avg Model Time = ' .. tostring(avgModelTime / numBatches))
	--print('Avg Data Time = ' .. tostring(avgDataTime / numBatches))
	print('----------------------------------------------------------------------------------------------');
	print('----------------------------------------------------------------------------------------------');
	print('\n\n')

	return loss:mean()
end

function ViserionTrainer:cudaDeviceCopy(sample)
	self.input = torch.CudaTensor()
	self.target = torch.CudaTensor()

	self.input:resize(sample.input:size()):copy(sample.input)
	self.target:resize(sample.target:size()):copy(sample.target)
end


return X.ViserionTrainer
