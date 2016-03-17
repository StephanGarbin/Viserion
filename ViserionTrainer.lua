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
	if self.opts.debug then
		print('DEBUG: Calling defineCustomLearningRate()')
	end
	self.optimOptions.learningRate = defineCustomLearningRate(epoch, self.optimOptions.learningRate)

	--update learning rate across devices if the model is a DPT
	--if self.opts.numGPUs > 1 then
	--	self.model:updateParameters(self.optimOptions.learningRate)
	--end

	--The evaluation function for the optimiser
	local function feval()
		return 0, self.gradParams
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
	if self.opts.debug then
		print('DEBUG: Switching model to training')
	end
	self.model:training()

	print('TRAIN: Processing Epoch # ' .. epoch .. ' (LR = ' .. tostring(self.optimOptions.learningRate) .. ')')

	ProgressBarStep = 1
	--Process all batches
	for n, sample in dataloader:run() do
		avgDataTime = avgDataTime + dataTimer:time().real

		if not self.opts.debug then
			xlua.progress(ProgressBarStep, numBatches)
		else
			print('DEBUG: Processing Batch ' .. tostring(n))
		end

		--Copy input and target to the GPU
		if self.opts.debug then
			print('DEBUG: Copying data to the host GPU')
			print('DEBUG: Input size: ', sample.input:size())
			print('DEBUG: Target size: ', sample.target:size())
		end
		self:cudaDeviceCopy(sample)

		--Do forward pass
		if self.opts.debug then
			print('DEBUG: Forward pass model')
			print([[DEBUG: If this segfaults, check sizes carefully (especially of your convolutions). If using autograd and declaring local tensors, make sure they are the correct type, especially when using CUDA, as the defaulttensortype in Viserion is a floatTensor.]])
		end
		self.model:forward(self.input)
		if self.opts.debug then

			--print('DEBUG: model output size', self.model.output:size())
		end
		
		if self.opts.debug then
			print('DEBUG: Resetting  model gradient parameters')
		end
		self.model:zeroGradParameters()
		
		--Criterion Forward
		if not self.opts.usingMultiCriteria then
			if self.opts.debug then
				print('DEBUG: Forward pass criterion')
			end
			local local_loss = self.criterion:forward(self.model.output, self.target)
	     	
			loss[n] = local_loss
		else
			criteriaForwardOutput = {}
			for i, c in ipairs(self.criterion) do
				if self.opts.debug then
					print('DEBUG: Evaluating defineCriteriaFlowForward() for ' .. tostring(i))
				end
				--1. Evaluate user definitions
				criteriaInputs, criteriaTargets =
					defineCriteriaFlowForward(i, self.model.output, self.input, criteriaForwardOutput)

				if self.opts.debug then
					print('DEBUG: Forward pass criterion ' .. tostring(i))
				end
				--2. Do Forward Pass
				criteriaForwardOutput[i] = c:forward(criteriaInputs, criteriaTargets)
			end
		end

		--Do backward pass
		if not self.opts.usingMultiCriteria then
			if self.opts.debug then
				print('DEBUG: Backward pass criterion')
			end
			self.criterion:backward(self.model.output, self.target)

			if self.opts.debug then
				print('DEBUG: Backward pass model')
			end
			self.model:backward(self.input, self.criterion.gradInput)
		else

			criteriaBackwardOutput = {}
			for i, c in self.criterion do
				if self.opts.debug then
					print('DEBUG: Evaluating defineCriteriaFlowBackward() for ' .. tostring(i))
				end
				--1. Evaluate user definitions
				criteraInputs, criteraTargets =
					defineCriteriaFlowBackward(self.model.output, self.input,
						criteriaForwardOutput, criteriaBackwardOutput)

				if self.opts.debug then
					print('DEBUG: Backward pass criterion ' .. tostring(i))
				end
				--2. Do backward pass
				criteriaBackwardOutput[i] = c:backward(criteriaInputs, criteriaTargets)
			end

			if self.opts.debug then
					print('DEBUG: Evaluating defineModelFlowBackward()')
				end
			modelTarget =
				defineModelFlowBackward(self.model.output, criteriaOutput, self.input)

			if self.opts.debug then
				print('DEBUG: Backward pass model')
			end
			self.model.backward(self.input, modelTarget)
		end
		--Do Optim step
		if self.opts.debug then
			print('DEBUG: Calling the optimiser')
		end
      	self.optimOptimiser(feval, self.params, self.optimOptions)

      	--https://github.com/soumith/cunnsparse/blob/master/doc/cunnmodules.md
      	if self.opts.numGPUs > 1 then
      		self.model:syncParameters()
      	end

		--Save some debug info
		avgModelTime = avgModelTime + modelTimer:time().real

		modelTimer:reset()
		dataTimer:reset()
		ProgressBarStep = ProgressBarStep + 1
	end

	print('\n')
	if not self.opts.usingMultiCriteria then
		print('Loss = ' .. tostring(loss:mean()))
	else
		defineCriteriaPrintOut(criteriaForwardOutput)
	end
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
		if self.opts.debug then
			print([[DEBUG: Creating Tensor to hold full size output,
				this will call size() on your dataloader for the targets/labels]])
		end
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

		if not self.opts.debug then
			xlua.progress(ProgressBarStep, numBatches)
		else
			print('DEBUG: Processing Batch ' .. tostring(n))
		end

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
     		local tmp = self.model.output:float()
     		for i = 1, (#sample.target)[1] do
     			self.testOutput[(n - 1) * self.opts.batchSize + i] = tmp[i]
     		end
     		--collectgarbage()
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
end

function ViserionTrainer:cudaDeviceCopy(sample)
	if self.opts.debug then
			print('DEBUG: Your Host GPU is ' .. tostring(self.opts.gpuIDXs[1]))
	end
	cutorch.setDevice(self.opts.gpuIDXs[1])
	if self.opts.numGPUs > 1 then
		self.input = cutorch.createCudaHostTensor()
		--self.target = cutorch.createCudaHostTensor()
	else
		self.input = torch.CudaTensor()
		--self.target = torch.CudaTensor()
	end

	self.target = torch.CudaTensor()

	self.input:resize(sample.input:size()):copy(sample.input)
	self.target:resize(sample.target:size()):copy(sample.target)
end


return X.ViserionTrainer
