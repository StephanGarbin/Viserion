require 'image'
require 'torch'

local ViserionImageIO = require 'ViserionImageIO'

t0 = torch.Timer()
myInstance = ViserionImageIO('db_1', '/home/stephan/testImages/testSequence/exrs', '/home/stephan/testImages/testCache', true, false, true, true, 100)
print(t0:time().real)

myInstance:destroyInstance()

t1 = torch.Timer()
for i = 1, 1000, 1 do
	im = myInstance:readSingleExr('/home/stephan/testImages/tinyRGB.exr')
end
print(t1:time().real)

print(im:size())
print(im:stride())

image.save('/home/stephan/testImages/tinyRGB_back.png', im)

collectgarbage()

t2 = torch.Timer()
for i = 1, 1000, 1 do
	im = image.load('/home/stephan/testImages/tinyRGB_2.png', 3, 'float')
end
print(t2:time().real)

collectgarbage()

--im1 = myInstance:readSingleExr('/home/stephan/testImages/tinyRGB.exr')
--im2 = image.load('/home/stephan/testImages/tinyRGB.png', 3, 'float')
--image.save('/home/stephan/testImages/tinyRGB_backFromPNG.png', im1)

