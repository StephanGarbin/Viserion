local X = {}

local ViserionMNISTLoader = torch.class('ViserionMNISTLoader', X)

function ViserionMNISTLoader:__init(filename)

	f = torch.DiskFile(self.filename, 'r'):binary():bigEndianEncoding()
  s = f:readInt(4)
  n = s[2]
  f:close()

  self.data = torch.ByteTensor(f:readByte(n * 28 * 28)):view(torch.LongStorage{n, 1, 28, 28}):float() / 255

end

function ViserionMNISTLoader:size()
	return self.data:size()
end

function ViserionMNISTLoader:getNarrowChunk(dim, index, size)
	return self.data:narrow(dim, index, size)
end

function ViserionMNISTLoader:getNarrowChunkNonContiguous(dim, idxList)
	chunk = torch.Tensor(idxList:size()[1], self.data:size()[2], self.data:size()[3], self.data:size()[4])
	for i = 1, idxList:size()[1] do
		chunk[i] = self.data[idxList[i]]
	end
	return chunk
end

return X.ViserionMNISTLoader
