require 'image'
require 'io'

local X = {}

local ViserionBinaryLoader = torch.class('ViserionBinaryLoader', X)

function ViserionBinaryLoader:__init(directory)
	self.directory = directory

	print('Finding Files in Directory...')

	self.files = {}

	for file in paths.files(directory) do
			if file:find('bin' .. '$') then
				table.insert(self.files, paths.concat(directory, file))
			end
	end

	table.sort(self.files, function (a,b) return a < b end)
	
	count = 0
	for x in pairs(self.files) do count = count + 1 end
	
	self.numFiles = count
	self.__size = 0

	print('Found ' .. tostring(self.numFiles) .. ' files.')
end

function ViserionBinaryLoader:size()
	return self.data:size()
end

function ViserionBinaryLoader:getNarrowChunk(dim, index, size)
	--print('Test', index, size)
	return self.data:narrow(dim, index, size)
end

function ViserionBinaryLoader:getNarrowChunkNonContiguous(dim, idxList)
	--print('Train', idxList[1])
	chunk = torch.Tensor(idxList:size()[1], self.data:size()[2], self.data:size()[3], self.data:size()[4])
	for i = 1, idxList:size()[1] do
		chunk[i] = self.data[idxList[i]]
	end 
	return chunk
end

function ViserionBinaryLoader:readArchives()
	--determine size of total dataset first
	
	numSamples = 0
	numChannels = 0	
	height = 0
	width = 0
	for i, f in pairs(self.files) do
		currentFile = torch.DiskFile(f, 'r')
		currentFile:binary();
		format = currentFile:readInt()
		numBlocks = currentFile:readInt()
		numChannels = currentFile:readInt()
		width = currentFile:readInt()
		height = currentFile:readInt()
		--print('Tensor of size ' .. tostring(numBlocks) .. 'x' .. tostring(numChannels) .. 'x' .. tostring(width) .. 'x' .. tostring(height))
		numSamples = numSamples + numBlocks
		currentFile:close()				
	end		
	
	print('Total # samples to read: ' .. tostring(numSamples)) 	

	--now read dataset
	self.data = torch.Tensor(numSamples, numChannels, height, width)
	
	collectgarbage()

	currentSample = 1
	for i, f in pairs(self.files) do
		currentFile = torch.DiskFile(f, 'r')
		currentFile:binary();
		currentFile:readInt();
		numBlocks = currentFile:readInt()
		currentFile:readInt(3)

		numFloatsPerBlock = numChannels * width * height;
		
		for i = 1, numBlocks do					
			self.data[{ currentSample,{},{},{} }] = torch.Tensor(currentFile:readFloat(numFloatsPerBlock))
			currentSample = currentSample + 1
		end

		currentFile:close()
		collectgarbage()
	end

	print('Finished Reading...')
		
	self.__size = (#self.data)[1]
	
	self.width = width
	self.height = height
	self.numChannels = numChannels

	collectgarbage()
end

return X.ViserionBinaryLoader


