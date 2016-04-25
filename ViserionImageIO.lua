require 'init'

local X = {}

local ViserionImageIO = torch.class('ViserionImageIO', X)

function ViserionImageIO:__init(datasetName, directory, cacheDirectory, recreateCache, recursive, compression, normalise, whiten, percentage2Use)

	--check if dataset file exists without needing any other libraries
	f_temp = io.open(cacheDirectory .. "/" .. datasetName .. ".exr","r")
	f_Stat = f_temp
	if f_Stat ~= nil then
		io.close(f_temp)
	end

	if recreateCache or f_Stat == nil then
		print('Creating cache for dataset '.. datasetName)
		clib.createDataSetCache(datasetName, directory, cacheDirectory, compression, recursive, normalise, whiten)
	else
		print('Cache for dataset ' .. datasetName .. ' exists, no caching to be done.');
	end

	self.percentage2Use = percentage2Use;

	print('Creating Instance...')
	self.cPtr = clib.createIOInstance(311)
end

function ViserionImageIO:destroyInstance()
	print('Destroying Instance...')
	clib.destroyIOInstance(self.cPtr)
end

function ViserionImageIO:size()
	return self.data:size()
end

function ViserionImageIO:getNarrowChunk(dim, index, size)
	return self.data:narrow(dim, index, size)
end

function ViserionImageIO:getNarrowChunkNonContiguous(dim, idxList)

		chunk = torch.Tensor(idxList:size()[1], self.data:size()[2], self.data:size()[3], self.data:size()[4])
		for i = 1, idxList:size()[1] do
			chunk[i] = self.data[idxList[i]]
		end
		return chunk

end

function ViserionImageIO:readSingleExr(fileName)
	result = torch.FloatTensor()
	clib.readEXR(self.cPtr, fileName, result:cdata())

	return result
end

return X.ViserionImageIO
