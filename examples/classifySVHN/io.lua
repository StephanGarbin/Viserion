local superDataLoader = require 'Viserion/ViserionDataLoader'
local ViserionSVHNLoader = require 'Viserion/dataLoaders/ViserionSVHNLoader'

dataPath = 'data/'

trainXLoader = ViserionSVHNLoader(dataPath, false, false)
testXLoader = ViserionSVHNLoader(dataPath, false, true)

trainYLoader = ViserionSVHNLoader(dataPath, true, false)
testYLoader = ViserionSVHNLoader(dataPath, true, true)


--[[NOTE: All we need to do now is give our dataloaders to the ViserionDataLoader (here renamed to be called 'superdataLoader'),
the first argument to the superdataLoader MUST be opts, i.e. the command line options you have access to anywhere in your scripts.
Let's also define an example print statement that will only happen when you call your scripts with -debug]]--

printDebug("Constructing the ViserionDataLoader Instances...")


trainDataLoader = superDataLoader(opts, trainXLoader, trainYLoader)
testDataLoader = superDataLoader(opts, testXLoader, testYLoader)

function saveState(epoch, lossTrain, lossTest, testOutput)
		print('Save State Not Implemented...')
end
