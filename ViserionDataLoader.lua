local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

X = {}

ViserionDataLoader = torch.class("ViserionDataLoader", X)


function ViserionDataLoader:__init(options, Xs, Ys)
	assert(Xs:size() == Ys:size())
	
	self.Xs = Xs
	self.Ys = Ys
	self.__size = Xs:size()
	self.batchSize = options.batchSize
	self.numThreads = options.numThreads

	--Create the parallel thread pool
	self.pool = threads.Threads(self.numThreads,
		function(idx)
			--We need to declare required files here,
			--otherwise serialisation will make this fail
			requireF = loadfile('Viserion/ViserionDataLoaderRequire.lua')
			requireF()
		end,
		function(idx)
			print('Spawing IO Thread...')
			_G.x = Xs
			_G.y = Ys
		end
		)
end


function ViserionDataLoader:size()
	return self.__size
end

function ViserionDataLoader:xDim()
	return self.Xs:dim()
end

function ViserionDataLoader:yDim()
	return self.Ys:dim()
end


function ViserionDataLoader:run()
	
	local perm = torch.randperm(self.__size)
	
	numEnqueuedBatches = 0
	numBatches = math.ceil(self.__size / self.batchSize)

	local sample

	local function createJobs()
		while numEnqueuedBatches < numBatches and self.pool:acceptsjob() do
			self.pool:addjob(
				function(batchSize, batchNum, totalSize, perm)

					--1. Determine batch size
					local bSize = -1
					if batchNum * batchSize + batchSize >= totalSize then
						bSize = batchSize - ((batchNum * batchSize + batchSize) - totalSize)
					else
						bSize = batchSize
					end

					local sample_ = {}
					sample_.input = _G.x:getNarrowChunkNonContiguous(1, perm:narrow(1, 1 + batchSize * batchNum, bSize))
					sample_.target = _G.y:getNarrowChunkNonContiguous(1, perm:narrow(1, 1 + batchSize * batchNum, bSize))
					
					return sample_
				end,
				function(sample_)
					sample = sample_
				end,
				self.batchSize, numEnqueuedBatches, self:size(), perm)

			numEnqueuedBatches = numEnqueuedBatches + 1
		end
	end

	local currentBatch = 0

	local function loop()
		createJobs()

		if not self.pool:hasjob() then
        	return nil
      	end
		self.pool:dojob()

		--Check for errors
		if self.pool:haserror() then
			print('ERROR: Thread Pool of DataLoader Class has encountered a critical error...')
			self.pool:synchronize()
		end

		currentBatch = currentBatch + 1

		return currentBatch, sample
	end

	return loop
end


function ViserionDataLoader:runNoShuffle()
	numEnqueuedBatches = 0
	numBatches = math.ceil(self.__size / self.batchSize)

	local sample

	local function createJobs()
		while numEnqueuedBatches < numBatches and self.pool:acceptsjob() do
			self.pool:addjob(
				function(batchSize, batchNum, totalSize)

					--1. Determine batch size
					local bSize = -1
					if batchNum * batchSize + batchSize >= totalSize then
						bSize = batchSize - ((batchNum * batchSize + batchSize) - totalSize)
					else
						bSize = batchSize
					end

					local sample_ = {}
					sample_.input = _G.x:getNarrowChunk(1, 1 + batchSize * batchNum, bSize)
					sample_.target = _G.y:getNarrowChunk(1, 1 + batchSize * batchNum, bSize)
					
					return sample_
				end,
				function(sample_)
					sample = sample_
				end,
				self.batchSize, numEnqueuedBatches, self:size())

			numEnqueuedBatches = numEnqueuedBatches + 1
		end
	end

	local currentBatch = 0

	local function loop()
		createJobs()

		if not self.pool:hasjob() then
        	return nil
      	end
		self.pool:dojob()

		--Check for errors
		if self.pool:haserror() then
			print('ERROR: Thread Pool of DataLoader Class has encountered a critical error...')
			self.pool:synchronize()
		end

		currentBatch = currentBatch + 1

		return currentBatch, sample
	end

	return loop
end

function ViserionDataLoader:augmentWithHFlips(perc)
	local perm = torch.randperm(self.__size)
	numAdditions = self.__size * (perc / 100)
	
	local newDataX = torch.Tensor(self.__size + numAdditions, (#self.Xs.data)[2], (#self.Xs.data)[    3], (#self.Xs.data)[4])
	
	local newDataY = torch.Tensor(self.__size + numAdditions, (#self.Ys.data)[2])

	for i=1,self.__size do
		newDataX[i] = self.Xs.data[i]
		newDataY[i] = self.Ys.data[i]
	end
	
	width = (#self.Xs.data)[2]
	height = (#self.Xs.data)[3]
	
	tmp = torch.Tensor(1, (#self.Ys.data)[2]):zero()
	tmp[1][1] = 1
	tmp[1][3] = 1
	tmp[1][5] = 1
	tmp[1][7] = 1
	
	for i=1,numAdditions do
		newDataX[self.__size + i] = image.hflip(self.Xs.data[perm[i]])
		newDataY[self.__size + i] = (tmp -  self.Ys.data[perm[i]]):abs()
	end

	self.Xs.data = newDataX
	collectgarbage()
	self.Ys.data = newDataY
	collectgarbage()
	self.__size = self.__size + numAdditions
	self.Xs.__size = self.__size
	self.Ys.__size = self.__size
	print('Added ' .. tostring(numAdditions) .. ' random hflips to data')
	print('New training set size is: ' .. tostring(self.__size))
end



return X.ViserionDataLoader
