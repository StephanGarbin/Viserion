require 'image'
require 'io'
require 'hdf5'

function resize_image_tensor(t, downsample_factor)
	local os = t:size()
	local height = os[3] / math.pow(2, downsample_factor)
	local width  = os[4] / math.pow(2, downsample_factor)
	local nt = torch.ByteTensor(os[1], os[2], height, width)
	for i = 1, os[1] do
			nt[i] = image.scale(t[i], width, height)
	end
	collectgarbage(); collectgarbage();
	return nt
end

local X = {}

local ViserionStereoHDF5Loader = torch.class('ViserionStereoHDF5Loader', X)

function ViserionStereoHDF5Loader:__init(filename, side, nb_sequences, nb_ips, downsample_factor)
	self.nb_sequences      = nb_sequences
	self.side              = side
	self.downsample_factor = downsample_factor
	self.nb_ips            = nb_ips

	print('loading hdf5')

	self.sequences  = {}

	f = hdf5.open(filename, 'r')
	for i = 1, nb_sequences do
		local t = f:read(side .. '/' .. (i - 1)):all()
		table.insert(self.sequences, resize_image_tensor(t, self.downsample_factor))
	end
	f:close()
	collectgarbage(); collectgarbage();

	self.height = self.sequences[1]:size()[3]
	self.width  = self.sequences[1]:size()[4]

	self.nb_layers = 0
	if side == 'left' then
		self.nb_layers = 15
	else
		self.nb_layers = 3
	end

-- indices[i] contains {sequence, start, end}
	self.indices    = {}
	self.total_size = 0

	for i = 1, nb_sequences do
		local current_sequence_size = self.sequences[i]:size()[1]
  	self.total_size = self.total_size + current_sequence_size - nb_ips + 1
		for j = 1, current_sequence_size - self.nb_ips + 1 do
			local t = {}
			if side == 'left' then
				t = {i, j, nb_ips + (j - 1)}
			else
				t = {i, j + 2, j + 2}
			end
			table.insert(self.indices, t)
		end
  end

	self.data_size = torch.LongStorage(4)
	self.data_size[1] = self.total_size
	self.data_size[2] = self.nb_layers
	self.data_size[3] = self.height
	self.data_size[4] = self.width

	print('loading of ' .. side .. ' done. ' .. 'total_size: ' .. self.total_size)
	print(self.data_size)

	collectgarbage(); collectgarbage();

end

function ViserionStereoHDF5Loader:size()
	return self.data_size
end

function ViserionStereoHDF5Loader:getNarrowChunk(dim, index, size)
	return self:getNarrowChunkNonContiguous(dim, torch.range(index, index + size - 1))
end

function ViserionStereoHDF5Loader:getNarrowChunkNonContiguous(dim, idxList)
	local chunk = torch.Tensor(idxList:size()[1], self.data_size[2], self.data_size[3], self.data_size[4])
	for i = 1, idxList:size()[1] do
		local chunk_index = self.indices[idxList[i]]
		chunk[i] = self.sequences[chunk_index[1]][{{chunk_index[2], chunk_index[3]}, {}, {}, {}}]:float()
	end
	return chunk / 255
end

return X.ViserionStereoHDF5Loader
