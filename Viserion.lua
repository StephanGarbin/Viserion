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
cmd:option('-disableThreadedGPUCopies', false, 'Disable faster Multi-GPU transfers')
cmd:option('-debug', false, 'Prints detailed debug output to identify where bugs are occuring')
cmd:option('-customDataLoaderFile', '', 'Specify this if you are using your own dataloaders')
cmd:option('-modelName', 'myModel', 'Specify model name. This is used when saving a gModule grap for example')
cmd:option('-enablePlots', false, 'Plots loss to a file')
cmd:option('-printCLErrors', false, 'Switches plots to doClassification')
cmd:option('-wildcard', '', 'Pass a custom string')
cmd:option('-wildcards', '', 'Pass a custom string resembling standard terminal arguments which gets converted to a table')

--Pass cmd args
opts = cmd:parse(arg)
--Pass wildcards args
if opts.wildcards ~= '' then
	local args = {}
	for word in opts.wildcards:gmatch("%w+") do
		if word[1] == '-' then
			word = string.sub(word, 1)
		end
		table.insert(args, word)
	end

	opts.wildcards = {}
	if math.fmod(#args, 2) == 0 then
		for i=1,#args,2 do
			if tonumber(args[i + 1]) ~= nil then
				opts.wildcards[args[i]] = tonumber(args[i + 1])
			elseif args[i + 1] == 'true' then
				opts.wildcards[args[i]] = true
			elseif args[i + 1] == 'false' then
				opts.wildcards[args[i]] = false
			else
				opts.wildcards[args[i]] = args[i + 1]
			end
		end
	else
		if #args >= 1 then
			print('ERROR:', '-wildcards expects <key, value> pairs to construct a table!')
		end
	end
else
	opts.wildcards = {}
end

print('You supplied the following arguments:')
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