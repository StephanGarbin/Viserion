require 'image'

local X = {}

local ViserionSVHNLoader = torch.class('ViserionSVHNLoader', X)

function ViserionSVHNLoader:__init(directory, isLabels, isTest)
	self.isLabels = isLabels
	self.isTest = isTest

	if isTest then
		n = 26032

		extents = {}
		extents[1] = {1, 10000}
		extents[2] = {10001, 26032}
	else
		n = 73257

		extents = {}
		extents[1] = {1, 10000}
		extents[2] = {10001, 20000}
		extents[3] = {20001, 30000}
		extents[4] = {30001, 40000}
		extents[5] = {40001, 50000}
		extents[6] = {50001, 60000}
		extents[7] = {60001, n}
	end

	self.files = {}

	for file in paths.files(directory) do
			if file:find('bin' .. '$') then
				table.insert(self.files, paths.concat(directory, file))
			end
	end

	table.sort(self.files, function (a,b) return a < b end)
	local count = 0
	for x in pairs(self.files) do count = count + 1 end
	
	if count ~= 11 then
		printError("CIFAR binary file error!")
	end

	if self.isTest then
		if self.isLabels then
			table.remove(self.files, 1)
			table.remove(self.files, 1)

			for i=1, 8, 1 do 
				table.remove(self.files, 2)
			end
			self.numFiles = 1
		else
			for i=1, 9, 1 do 
				table.remove(self.files, 3)
			end
			self.numFiles = 2
		end
	else
		if self.isLabels then
			for i=1, 10, 1 do 
				table.remove(self.files, 1)
			end
			self.numFiles = 1
		else
			for i=1, 3, 1 do 
				table.remove(self.files, 1)
			end
			table.remove(self.files, 8)
			self.numFiles = 7
		end
	end

	collectgarbage()

	--print(isLabels, isTest, self.files, self.numFiles)

	if not isLabels then
		self.data = torch.FloatTensor(n, 3, 32, 32):zero()
		for i, file in ipairs(self.files) do
			f = torch.DiskFile(file, 'r'):binary()
			local_n = extents[i][2] - extents[i][1] + 1
			self.data[{{extents[i][1], extents[i][2]}, {}, {}, {}}] = torch.ByteTensor(f:readByte(local_n * 3 * 32 * 32)):view(torch.LongStorage{local_n, 3, 32, 32}):float() / 255
		  	
			f:close()
		end


		for i=1, n, 1 do
			self.data[{i, {}, {}, {}}] = image.rgb2yuv(self.data[{i, {}, {}, {}}]:squeeze())
		end
		--[[if isTest then
			image.save('/home/stephan/tempTest.png', self.data[{torch.random(1, n), {}, {}, {}}])
		else
			image.save('/home/stephan/tempTrain.png', self.data[{torch.random(1, n), {}, {}, {}}])
		end]]--
	else
		f = torch.DiskFile(self.files[1], 'r'):binary()

		self.data = torch.ByteTensor(f:readByte(n)):view(torch.LongStorage{n}):float()
	  	
		f:close()
	end
end

function ViserionSVHNLoader:size()
	return self.data:size()
end

function ViserionSVHNLoader:getNarrowChunk(dim, index, size)
	if self.isLabels then
		return self.data:narrow(dim, index, size)
	else
		return self.data:narrow(dim, index, size)
	end
end

function ViserionSVHNLoader:getNarrowChunkNonContiguous(dim, idxList)
	if self.isLabels then
		chunk = torch.Tensor(idxList:size()[1])
		for i = 1, idxList:size()[1] do
			chunk[i] = self.data[idxList[i]]
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

return X.ViserionSVHNLoader
