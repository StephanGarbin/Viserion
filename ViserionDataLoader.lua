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

	--Create the parallel thread pool
	local pool = threads.Threads(self.numThreads,
		function(idx)
			--We need to declare required files here,
			--otherwise serialisation will make this fail
			if options.debug then
				print('DEBUG: Loading the require file on ', idx)
			end

			require('Viserion/dataLoaders/ViserionMNISTLoader')
			require('Viserion/dataLoaders/ViserionCIFAR10Loader')
			require('Viserion/dataLoaders/ViserionSVHNLoader')

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
	
	self.perm = perm

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
					sample_.input = _G.x:getNarrowChunkNonContiguous(1, perm:narrow(1, 1 + batchSize * batchNum, bSize), batchNum, _G.opts.currentEpoch)
					sample_.target = _G.y:getNarrowChunkNonContiguous(1, perm:narrow(1, 1 + batchSize * batchNum, bSize), batchNum, _G.opts.currentEpoch)
					
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
			print('DEBUG: Creating Shuffled Loop')
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

		createJobs()

		return currentBatch, sample
	end

	return loop
end


function ViserionDataLoader:runNoShuffle()
	local Xs = self.Xs
	local Ys = self.Ys
	local options = self.opts

	--Create the parallel thread pool
	--print('Num Threads: ', self.numThreads)
	local pool = threads.Threads(self.numThreads,
		function(idx)
			--We need to declare required files here,
			--otherwise serialisation will make this fail
			require('Viserion/dataLoaders/ViserionMNISTLoader')
			require('Viserion/dataLoaders/ViserionCIFAR10Loader')
			require('Viserion/dataLoaders/ViserionSVHNLoader')

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
			--print('Spawing IO Thread...', idx)
			_G.x = Xs
			_G.y = Ys
			_G.opts = options
		end
		)

	numEnqueuedBatches = 0
	numBatches = math.ceil(self.__size / self.batchSize)

	local sample
	local currentBatch

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
						print('DEBUG: Getting Data, calling getNarrowChunkNonContiguous() on your dataloader')
					end

					local sample_ = {}
					--sample_.input = _G.x:getNarrowChunkNonContiguous(1, perm:narrow(1, 1 + batchSize * batchNum, bSize), batchNum, _G.opts.currentEpoch)
					--sample_.target = _G.y:getNarrowChunkNonContiguous(1, perm:narrow(1, 1 + batchSize * batchNum, bSize), batchNum, _G.opts.currentEpoch)
					sample_.input = _G.x:getNarrowChunk(1, 1 + batchSize * batchNum, bSize, batchNum, _G.opts.currentEpoch)
					sample_.target = _G.y:getNarrowChunk(1, 1 + batchSize * batchNum, bSize, batchNum, _G.opts.currentEpoch)
					
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

	printDebug('DEBUG: Creating Non-Shuffled Loop')

	local function loop()
		createJobs()

		if not pool:hasjob() then
			pool:synchronize()
        	return nil
      	end
		pool:dojob()

		--Check for errors
		if pool:haserror() then
			print('ERROR: Thread Pool of DataLoader Class has encountered a critical error...')
			pool:synchronize()
		end

		createJobs()

		return currentBatch, sample
	end

	return loop
end

return X.ViserionDataLoader
