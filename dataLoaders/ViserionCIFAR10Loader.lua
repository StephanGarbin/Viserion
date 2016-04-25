local X = {}

local ViserionCIFAR10Loader = torch.class('ViserionCIFAR10Loader', X)

function ViserionCIFAR10Loader:__init(directory, isLabels, isTest)
	self.isLabels = isLabels
	self.isTest = isTest

	--find files in directories
	--print('Finding CIFAR binary files in Directory...')

	self.files = {}

	for file in paths.files(directory) do
			if file:find('bin' .. '$') then
				table.insert(self.files, paths.concat(directory, file))
			end
	end

	table.sort(self.files, function (a,b) return a < b end)
	local count = 0
	for x in pairs(self.files) do count = count + 1 end
	
	if count ~= 6 then
		printError("CIFAR binary file missing!")
	end
	
	if self.isTest then
		for i=1, 5, 1 do 
			table.remove(self.files, 1)
		end
		self.numFiles = 1
	else
		table.remove(self.files, 6)
		self.numFiles = 5
	end

	collectgarbage()

	self.__size = 0

	if self.isLabels then
		self.data = torch.FloatTensor(self.numFiles * 10000, 1):zero()
	else
		self.data = torch.FloatTensor(self.numFiles * 10000, 3, 32, 32):zero()
	end

	for i, file in pairs(self.files) do
		local offset = (i - 1) * 10000
		f = torch.DiskFile(file, 'r'):binary()

		for j=1,10000,1 do
			local label = torch.ByteTensor(f:readByte(1))
			local image = torch.ByteTensor(f:readByte(1024 * 3))
			if self.isLabels then
				self.data[offset + j] = label:float()
			else 
				self.data[offset + j] = image:float()
			end
		end
		f:close()
	end

	collectgarbage()
end

function ViserionCIFAR10Loader:size()
	return self.data:size()
end

function ViserionCIFAR10Loader:getNarrowChunk(dim, index, size)
	return self.data:narrow(dim, index, size)
end

function ViserionCIFAR10Loader:getNarrowChunkNonContiguous(dim, idxList)
	if self.isLabels then
		chunk = torch.Tensor(idxList:size()[1])
		for i = 1, idxList:size()[1] do
			chunk[i] = self.data[idxList[i]] + 1
		end
		return chunk
	else
		chunk = torch.Tensor(idxList:size()[1], self.data:size()[2], self.data:size()[3], self.data:size()[4])
		for i = 1, idxList:size()[1] do
			chunk[i] = self.data[idxList[i]]
		end
		return chunk
	end
end

return X.ViserionCIFAR10Loader
