require 'gnuplot'


function plotLoss(loss, modelName)
	local x = torch.linspace(1, loss:size()[1], loss:size()[1])
	gnuplot.epsfigure(modelName .. '_loss.eps')
	gnuplot.plot('Loss', x, loss, '-')
	gnuplot.xlabel('Epoch')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end