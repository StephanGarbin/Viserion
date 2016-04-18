local superDataLoader = require 'Viserion/ViserionDataLoader'
local ViserionMNISTLoader = require 'Viserion/dataLoaders/ViserionMNISTLoader'

trainXPath = 'data/train-images.idx3-ubyte'
trainYPath = 'data/train-labels.idx1-ubyte'

testXPath = 'data/t10k-images.idx3-ubyte'
testYPath = 'data/t10k-labels.idx1-ubyte'

trainXLoader = ViserionMNISTLoader(trainXPath, false)
testXLoader = ViserionMNISTLoader(testXPath, false)

trainYLoader = ViserionMNISTLoader(trainYPath, true)
testYLoader = ViserionMNISTLoader(testYPath, true)


--[[NOTE: All we need to do now is give our dataloaders to the ViserionDataLoader (here renamed to be called 'superdataLoader'),
the first argument to the superdataLoader MUST be opts, i.e. the command line options you have access to anywhere in your scripts.
Let's also define an example print statement that will only happen when you call your scripts with -debug]]--

printDebug("Constructing the ViserionDataLoader Instances...")


trainDataLoader = superDataLoader(opts, trainXLoader, trainYLoader)
testDataLoader = superDataLoader(opts, testXLoader, testYLoader)

function saveState(epoch, loss, testOutput)
		print('Save State Not Implemented...')
end
