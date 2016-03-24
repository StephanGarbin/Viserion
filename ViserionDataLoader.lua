threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

X = {}

ViserionDataLoader = torch.class("ViserionDataLoader", X)


function ViserionDataLoader:__init(options, Xs, Ys)
	if torch.type(Xs:size()) ~= 'table' then
		self.__size = Xs:size()[1]
	else
		self.__size = Xs:size()[1][1]
	end

	self.Xs = Xs
	self.Ys = Ys
	--shorthand as these are used all the time
	self.batchSize = options.batchSize
	self.numThreads = options.numThreads
	--but save all options
	self.opts = options
end


function ViserionDataLoader:size()
	return self.__size
end

function ViserionDataLoader:xSize()
	return self.Xs:size()
end

function ViserionDataLoader:ySize()
	return self.Ys:size()
end


function ViserionDataLoader:run()
	local Xs = self.Xs
	local Ys = self.Ys
	local options = self.opts

	if self.opts.debug then
		print('DEBUG: Creating Thread Pool with ' .. tostring(self.numThreads) .. ' threads')
	end

	local lViserionBinaryLoader = require('Viserion/dataLoaders/ViserionBinaryLoader')
	local lViserionMNISTLoader = require('Viserion/dataLoaders/ViserionMNISTLoader')
	local lViserionStereoHDF5Loader = require('Viserion/dataLoaders/ViserionStereoHDF5Loader')
	--Create the parallel thread pool
	local pool = threads.Threads(self.numThreads,
		function(idx)
			--We need to declare required files here,
			--otherwise serialisation will make this fail
			if options.debug then
				print('DEBUG: Loading the require file on ', idx)
			end
			ViserionBinaryLoader = lViserionBinaryLoader
			--require('Viserion/dataLoaders/ViserionMNISTLoader')
			ViserionStereoHDF5Loader = lViserionStereoHDF5Loader

			--Load custom dataloaders if necessary
			if options.customDataLoaderFile ~= '' then
				if options.debug then
					print('DEBUG: Loading Custom DataLoader')
				end
				dofile(options.customDataLoaderFile)
			end

			return idx
		end,
		function(idx)

			if options.debug then
				print('DEBUG: Spawing IO Thread...', idx)
			end
			
			_G.x = Xs
			_G.y = Ys
			_G.opts = options

		end
	)

	if self.opts.debug then
		print('DEBUG: Creating Random Permutation')
	end
	local perm = torch.randperm(self.__size)
	
	numEnqueuedBatches = 0
	numBatches = math.ceil(self.__size / self.batchSize)

	local sample
	local currentBatch

	local function createJobs()
		while numEnqueuedBatches < numBatches and pool:acceptsjob() do
			pool:addjob(
				function(batchSize, batchNum, totalSize, perm)

					--1. Determine batch size
					local bSize = -1
					if batchNum * batchSize + batchSize >= totalSize then
						bSize = batchSize - ((batchNum * batchSize + batchSize) - totalSize)
					else
						bSize = batchSize
					end

					if _G.opts.debug then
						print('DEBUG: Getting Data, calling getNarrowChunkNonContiguous() on your dataloader')
					end

					local sample_ = {}
					sample_.input = _G.x:getNarrowChunkNonContiguous(1, perm:narrow(1, 1 + batchSize * batchNum, bSize))
					sample_.target = _G.y:getNarrowChunkNonContiguous(1, perm:narrow(1, 1 + batchSize * batchNum, bSize))
					
					return batchNum, sample_
				end,
				function(batchNum, sample_)
					sample = sample_
					currentBatch = batchNum + 1
				end,
				self.batchSize, numEnqueuedBatches, self:size(), perm)

			numEnqueuedBatches = numEnqueuedBatches + 1
		end
	end

	if self.opts.debug then
			print('DEBUG: Creating Training Loop')
	end
	local function loop()
		if self.opts.debug then
			print('DEBUG: Creating Jobs')
		end
		createJobs()

		if not pool:hasjob() then
			if self.opts.debug then
				print('DEBUG: Thread Pool has no more jobs')
			end
			pool:synchronize()
        	return nil
      	end

      	if self.opts.debug then
			print('DEBUG: Running dojob()')
		end
		pool:dojob()

		--Check for errors
		if pool:haserror() then
			print('ERROR: Thread Pool of DataLoader Class has encountered a critical error...')
			pool:synchronize()
		end

		return currentBatch, sample
	end

	return loop
end


function ViserionDataLoader:runNoShuffle()
	local Xs = self.Xs
	local Ys = self.Ys
	local options = self.opts
	--Compile custom dataloaders into function if required
	local customDLs = nil
	if options.customDataLoaderFile ~= '' then
		customDLs = loadfile(opts.customDataLoaderFile)
	end

	--Create the parallel thread pool
	--print('Num Threads: ', self.numThreads)
	local pool = threads.Threads(self.numThreads,
		function(idx)
			--We need to declare required files here,
			--otherwise serialisation will make this fail
			requireF = loadfile('Viserion/ViserionDataLoaderRequire.lua')
			requireF()

			--Load custom dataloaders if necessary
			if customDLs ~= nil then
				customDLs()
			end

			return idx
		end,
		function(idx)
			--print('Spawing IO Thread...', idx)
			_G.x = Xs
			_G.y = Ys
			_G.opts = options
		end
		)

	numEnqueuedBatches = 0
	numBatches = math.ceil(self.__size / self.batchSize)

	local currentBatch = 0

	local sample

	local function createJobs()
		while numEnqueuedBatches < numBatches and pool:acceptsjob() do
			pool:addjob(
				function(batchSize, batchNum, totalSize)

					--1. Determine batch size
					local bSize = -1
					if batchNum * batchSize + batchSize >= totalSize then
						bSize = batchSize - ((batchNum * batchSize + batchSize) - totalSize)
					else
						bSize = batchSize
					end

					if _G.opts.debug then
						print('DEBUG: Getting Data, calling getNarrowChunk() on your dataloader')
					end
					local sample_ = {}
					sample_.input = _G.x:getNarrowChunk(1, 1 + batchSize * batchNum, bSize)
					sample_.target = _G.y:getNarrowChunk(1, 1 + batchSize * batchNum, bSize)
					
					return batchNum, sample_
				end,
				function(batchNum, sample_)
					sample = sample_
					currentBatch = batchNum + 1
				end,
				self.batchSize, numEnqueuedBatches, self:size())

			numEnqueuedBatches = numEnqueuedBatches + 1
		end
	end

	local function loop()
		createJobs()

		if not pool:hasjob() then
        	return nil
      	end
		pool:dojob()

		--Check for errors
		if pool:haserror() then
			print('ERROR: Thread Pool of DataLoader Class has encountered a critical error...')
			pool:synchronize()
		end

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
