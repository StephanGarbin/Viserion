local superDataLoader = require 'Viserion/ViserionDataLoader'
local ViserionCIFAR10Loader = require 'Viserion/dataLoaders/ViserionCIFAR10Loader'

path = 'data/'

trainXLoader = ViserionCIFAR10Loader(path, false, false)
testXLoader = ViserionCIFAR10Loader(path, false, true)

trainYLoader = ViserionCIFAR10Loader(path, true, false)
testYLoader = ViserionCIFAR10Loader(path, true, true)


--[[NOTE: All we need to do now is give our dataloaders to the ViserionDataLoader (here renamed to be called 'superdataLoader'),
the first argument to the superdataLoader MUST be opts, i.e. the command line options you have access to anywhere in your scripts.
Let's also define an example print statement that will only happen when you call your scripts with -debug]]--

printDebug("Constructing the ViserionDataLoader Instances...")


trainDataLoader = superDataLoader(opts, trainXLoader, trainYLoader)
testDataLoader = superDataLoader(opts, testXLoader, testYLoader)

function saveState(epoch, lossTrain, lossTest, testOutput)
		print('Save State Not Implemented...')
end
