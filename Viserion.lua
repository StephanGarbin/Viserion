require 'torch'

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-ioFile', '', 'Script specifing IO')
cmd:option('-modelFile', '', 'Script defining model')
cmd:option('-criterionFile', '', 'Script defining criterion to optimise')
cmd:option('-optimFile', '', 'Script defining optimisation functions')
cmd:option('-doTraining', false, 'Training/Test switch')
cmd:option('-batchSize', 10, 'Batch size')
cmd:option('-numEpochs', 100, '#Epochs')
cmd:option('-startEpoch', 1, 'Starting epoch for training')
cmd:option('-numThreads', 1, '#Threads for data loading')
cmd:option('-saveStateInterval', -1, 'Interval in which saveState function in your IO script is called. Set to -1 to disable')
cmd:option('-passFullOutput2saveState', false, 'Set this to false so that full state output is not passed to the saveState function in your IO script.')
cmd:option('-disableCUDNN', false, 'Use this to disable the CUDNN backend')
cmd:option('-cudnnVerbose', false, 'Enable verbose output for CUDNN debug')
cmd:option('-disableCUDA', false, 'Uses the CPU instead')
cmd:option('-specifyGPUs', 1, 'Specify which GPUS on the system to use, for example, to use 3 and 4, use 34')
cmd:option('-multiThreadGPUCopies', true, 'Faster Multi-GPU transfers')
cmd:option('-debug', false, 'Prints detailed debug output to identify where bugs are occuring')
cmd:option('-customDataLoaderFile', '', 'Specify this if you are using your own dataloaders')
cmd:option('-modelName', 'myModel', 'Specify model name. This is used when saving a gModule grap for example')
cmd:option('-enablePlots', false, 'Plots loss to a file')
cmd:option('-printCLErrors', false, 'Switches plots to doClassification')
cmd:option('-wildcard', '', 'Pass a custom string')
opts = cmd:parse(arg)

print(opts)

viserionAsFunc = loadfile('Viserion/ViserionMain.lua')

if opts.debug then
	print('Running Viserion in DEBUG mode ...')
	function errorHandler(error)
		print('Viserion has reported an error: ', error)
	end

	status = xpcall(viserionAsFunc, errorHandler)
	print('Viserion exited with status: ', status)
else
	viserionAsFunc()
end