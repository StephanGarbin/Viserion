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

	self.totalNumTrainingMiniBatches = 0
	self.totalNumTestingMiniBatches = 0
	self.currentMiniBatchLoss = 0
end

function ViserionTrainer:train(epoch, dataloader)

	dataloader.opts.currentEpoch = epoch

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
	local loss = torch.Tensor(numBatches):fill(0)
	local avgModelTime = 0
	local avgDataTime = 0

	local t1 = nil
	if opts.printCLErrors then
		t1 = torch.Tensor(numBatches):fill(0)
	end

	--Switch Model to training
	if self.opts.debug then
		print('DEBUG: Switching model to training')
	end
	self.model:training()

	print('TRAIN: Processing Epoch # ' .. epoch .. ' (LR = ' .. tostring(self.optimOptions.learningRate) .. ')')

	criteriaForwardOutputs = {}
	ProgressBarStep = 1

	if processBeforeEpoch ~= nil then
		printDebug([[Calling 'processBeforeEpoch']])
		processBeforeEpoch(true)
	end

	self.abort = nil
	--Process all batches
	for n, sample in dataloader:run() do	

		if self.abort == false then break end

		avgDataTime = avgDataTime + dataTimer:time().real

		if processBeforeMinibatch ~= nil then
			printDebug([[Calling 'processBeforeMinibatch']])
			self.abort = processBeforeMinibatch(true, n, self.totalNumTrainingMiniBatches)
		end

		if not self.opts.debug then
			xlua.progress(ProgressBarStep, numBatches)
		else
			print('DEBUG: Processing Batch ' .. tostring(n))
		end

		--Copy input and target to the GPU
		if self.opts.debug then
			print('DEBUG: Copying data...')
			if type(sample.input) == 'table' then
				print('DEBUG: Input size: ', sample.input)
			else
				print('DEBUG: Input size: ', sample.input:size())
			end

			if type(sample.target) == 'table' then
				print('DEBUG: Target size: ', sample.target)
			else
				print('DEBUG: Target size: ', sample.target:size())
			end
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
			self.currentMiniBatchLoss = local_loss

			if opts.printCLErrors then
				lt1 = self:computeClassificationErrors(self.model.output, sample.target)
				t1[n] = lt1
			end
		else
			criteriaForwardOutput = {}
			for i, c in ipairs(self.criterion) do
				if self.opts.debug then
					print('DEBUG: Evaluating defineCriteriaFlowForward(), Forward pass criterion ' .. tostring(i))
				end
				--2. Do Forward Pass
				criteriaForwardOutput[i] =
					c:forward(defineCriteriaFlowForward(i, self.model.output, self.input, self.target, criteriaForwardOutput))
			end

			criteriaForwardOutputs[n] = criteriaForwardOutput
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
			for i, c in ipairs(self.criterion) do
				if self.opts.debug then
					print('DEBUG: Evaluating defineCriteriaFlowBackward(), Backward pass criterion ' .. tostring(i))
				end
				--2. Do backward pass
				criteriaBackwardOutput[i] = 
					c:backward(defineCriteriaFlowBackward(i, self.model.output, self.input, self.target,
						criteriaForwardOutput, criteriaBackwardOutput))
			end

			if self.opts.debug then
					print('DEBUG: Evaluating defineModelFlowBackward()')
			end
			modelTarget =
				defineModelFlowBackward(self.model.output, criteriaBackwardOutput, self.input, self.target)

			if self.opts.debug then
				print('DEBUG: Backward pass model')
				if type(modelTarget) == 'table' then
					print('DEBUG: Your target is:', modelTarget)
				else
					print('DEBUG: Your target is:', modelTarget:size())
				end

				if type(self.model.output) == 'table' then
					print('DEBUG: Your model forward output was:', self.model.output)
				else
					print('DEBUG: Your model forward output was:', self.model.output:size())
				end
			end

			self.model:backward(self.input, modelTarget)
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

		self.totalNumTrainingMiniBatches = self.totalNumTrainingMiniBatches + 1

		if processAfterMinibatch ~= nil then
			printDebug([[Calling 'processAfterMinibatch']])
			self.abort = processAfterMinibatch(true, n, self.totalNumTrainingMiniBatches)
		end
	end

	if processAfterEpoch ~= nil then
			print('\n')
			printDebug([[Calling 'processAfterEpoch']])
			processAfterEpoch(true)
	end

	print('\n')
	if not self.opts.usingMultiCriteria then
		print('Loss = ' .. tostring(loss:mean()))

		if opts.printCLErrors then
			print('Top1Error = ' .. tostring(t1:mean()))
		end
	else
		defineCriteriaPrintOut(epoch, true, criteriaForwardOutputs)
	end
	--print('Avg Model Time = ' .. tostring(avgModelTime / numBatches))
	--print('Avg Data Time = ' .. tostring(avgDataTime / numBatches))
	print('\n')

	if not self.opts.usingMultiCriteria then
		return loss:mean()
	end
end

function ViserionTrainer:test(epoch, dataloader, saveTestOutput)
	
	dataloader.opts.currentEpoch = epoch

	--Time what we do
	local modelTimer = torch.Timer()
	local dataTimer = torch.Timer()

	local numBatches = math.ceil(dataloader:size() / self.opts.batchSize)

	--Store progression of loss, time to load data & to execute
	local loss = torch.Tensor(numBatches):fill(0)
	local t1 = nil
	if opts.printCLErrors then
		t1 = torch.Tensor(numBatches):fill(0)
	end

	local avgModelTime = 0
	local avgDataTime = 0

	self.testOutput = nil

	--Switch Model to evaluation
	self.model:evaluate()

	print('TEST: Processing Epoch # ' .. epoch)

	criteriaForwardOutputs = {}

	ProgressBarStep = 1

	if processBeforeEpoch ~= nil then
		printDebug([[Calling 'processBeforeEpoch']])
		processBeforeEpoch(false)
	end

	self.abort = nil

	--Process all batches
	for n, sample in dataloader:runNoShuffle() do
		local dataTime = dataTimer:time().real


		if processBeforeMinibatch ~= nil then
			printDebug([[Calling 'processBeforeMinibatch']])
			self.abort = processBeforeMinibatch(false, n, self.totalNumTestingMiniBatches)
		end

		if self.abort == false then break end

		if not self.opts.debug then
			xlua.progress(ProgressBarStep, numBatches)
		else
			print('DEBUG: Processing Batch ' .. tostring(n))
		end

		--Copy input and target to the GPU
		self:cudaDeviceCopy(sample)

		--Do forward pass
		self.model:forward(self.input)

		if self.testOutput == nil and saveTestOutput then
			if self.opts.debug then
				print('DEBUG: Creating data structure to hold full size output of your model')
			end
			if type(self.model.output) == 'table' then
				self.testOutput = {}

				for i, e in ipairs(self.model.output) do
					local cSize = self.model.output[i]:size()
					cSize[1] = dataloader:size()
					self.testOutput[i] = torch.Tensor(cSize)
				end

				if self.opts.debug then
					print('DEBUG: Created table of dimension', self.testOutput)
				end
			else
				if self.opts.debug then
					print('DEBUG: Creating tensor to hold full size output of your model')
				end
				local cSize = self.model.output:size()
				cSize[1] = dataloader:size()
				self.testOutput = torch.Tensor(cSize)
				if self.opts.debug then
					print('DEBUG: Created tensor of size', cSize)
				end
			end
		end

		--Compute loss
		if not self.opts.usingMultiCriteria then
			if self.opts.debug then
				print('DEBUG: Forward pass criterion')
			end
			local local_loss = self.criterion:forward(self.model.output, self.target)
			loss[n] = local_loss
			self.currentMiniBatchLoss = local_loss

			if opts.printCLErrors then
				lt1 = self:computeClassificationErrors(self.model.output, sample.target)
				t1[n] = lt1
			end
		else
			criteriaForwardOutput = {}
			for i, c in ipairs(self.criterion) do
				if self.opts.debug then
					print('DEBUG: Evaluating defineCriteriaFlowForward(), Forward pass criterion ' .. tostring(i))
				end
				--2. Do Forward Pass
				criteriaForwardOutput[i] =
					c:forward(defineCriteriaFlowForward(i, self.model.output, self.input, self.target, criteriaForwardOutput))
			end
			criteriaForwardOutputs[n] = criteriaForwardOutput
		end

     	--Save data if required
     	if saveTestOutput then
     		if self.opts.debug then
				print('DEBUG: Saving model output of batch ' .. tostring(n))
			end
     		if type(self.model.output) == 'table' then
     			for i, e in ipairs(self.model.output) do
     				local tmp = self.model.output[i]:float()
		     		for b = 1, (#(self.model.output[i]))[1] do
		     			self.testOutput[i][(n - 1) * self.opts.batchSize + b] = tmp[b]
		     		end
     			end
     		else
     			local tmp = self.model.output:float()
	     		for b = 1, (#self.model.output)[1] do
	     			self.testOutput[(n - 1) * self.opts.batchSize + b] = tmp[b]
	     		end
     		end
     	end

		--Save some debug info
		avgModelTime = avgModelTime + modelTimer:time().real
		avgDataTime = avgDataTime + dataTimer:time().real

		modelTimer:reset()
		dataTimer:reset()

		ProgressBarStep = ProgressBarStep + 1
		self.totalNumTestingMiniBatches = self.totalNumTestingMiniBatches + 1

		if processAfterMinibatch ~= nil then
			printDebug([[Calling 'processAfterMinibatch']])
			self.abort = processAfterMinibatch(false, n, self.totalNumTestingMiniBatches)
		end
	end

	if processAfterEpoch ~= nil then

		print('\n')
		printDebug([[Calling 'processAfterEpoch']])
		processAfterEpoch(false)
	end

	print('\n')

	if not self.opts.usingMultiCriteria then
		print('Loss = ' .. tostring(loss:mean()))

		if opts.printCLErrors then
			print('Top1Error = ' .. tostring(t1:mean()))
		end
	else
		defineCriteriaPrintOut(epoch, false, criteriaForwardOutputs)
	end
	--print('Avg Model Time = ' .. tostring(avgModelTime / numBatches))
	--print('Avg Data Time = ' .. tostring(avgDataTime / numBatches))
	print('----------------------------------------------------------------------------------------------');
	print('----------------------------------------------------------------------------------------------');
	print('\n\n')

	if not self.opts.usingMultiCriteria then
		return loss:mean()
	end
end

function ViserionTrainer:cudaDeviceCopy(sample)
	if not self.opts.disableCUDA then
		if self.opts.debug then
				print('DEBUG: Your Host GPU is ' .. tostring(self.opts.gpuIDXs[1]))
		end
		cutorch.setDevice(self.opts.gpuIDXs[1])
	end

	if type(sample.input) == 'table' then
		if self.input == nil then
			self.input = {}
			for i,e in ipairs(sample.input) do
				if not self.opts.disableCUDA then
					if self.opts.numGPUs > 1 then
						self.input[i] = cutorch.createCudaHostTensor()
					else
						self.input[i] = torch.CudaTensor()
					end
				else
					self.input[i] = torch.Tensor()
				end

				self.input[i]:resize(sample.input[i]:size()):copy(sample.input[i])
			end
		else
			for i,e in ipairs(sample.input) do
				self.input[i]:resize(sample.input[i]:size()):copy(sample.input[i])
			end
		end
	else
		if self.input == nil then
			if not self.opts.disableCUDA then
				if self.opts.numGPUs > 1 then
					self.input = cutorch.createCudaHostTensor()
				else
					self.input = torch.CudaTensor()
				end
			else
				self.input = torch.Tensor()
			end

			self.input:resize(sample.input:size()):copy(sample.input)
		else
			self.input:resize(sample.input:size()):copy(sample.input)
		end
	end
	
	if type(sample.target) == 'table' then
		if self.target == nil then
			self.target = {}
			for i,e in ipairs(sample.target) do
				if not self.opts.disableCUDA then
					if self.opts.numGPUs > 1 then
						self.target[i] = torch.CudaTensor()
					else
						self.target[i] = torch.CudaTensor()
					end
				else
					self.target[i] = torch.Tensor()
				end

				self.target[i]:resize(sample.target[i]:size()):copy(sample.target[i])
			end
		else
			for i,e in ipairs(sample.target) do
				self.target[i]:resize(sample.target[i]:size()):copy(sample.target[i])
			end
		end
	else
		if self.target == nil then
			if not self.opts.disableCUDA then
				if self.opts.numGPUs > 1 then
					self.target = torch.CudaTensor()
				else
					self.target = torch.CudaTensor()
				end
			else
				self.target = torch.Tensor()
			end

			self.target:resize(sample.target:size()):copy(sample.target)
		else
			self.target:resize(sample.target:size()):copy(sample.target)
		end
	end
end

function ViserionTrainer:computeClassificationErrors(modelForward, target)
	target = target:view(torch.LongStorage{target:size()[1], 1})

	sorted, idxs = torch.sort(modelForward, 2, true)

	idxs = idxs - 1

	local tmp = torch.clamp((idxs[{{}, 1}]:float() - target:float()):abs(),0,1)

	top1Error = 1 - torch.sum(tmp) / modelForward:size()[1]

	return top1Error
end

function ViserionTrainer:getCurrentMinibatchLoss()
	return self.currentMiniBatchLoss
end

return X.ViserionTrainer
