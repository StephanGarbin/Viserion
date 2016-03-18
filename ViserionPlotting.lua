

function plotLoss(loss, modelName)
	local lPlot = require 'gnuplot'
	local x = torch.linspace(1, loss:size()[1], loss:size()[1])
	local fileName = modelName ..'Loss.eps'
	lPlot.epsfigure(fileName)
	lPlot.plot(loss, '-')
	lPlot.xlabel('Epoch')
	lPlot.ylabel('Error')
	lPlot.title('Loss')
	lPlot.plotflush()
	lPlot.close()

	collectgarbage()
end