require 'gnuplot'


function plotLoss(loss, modelName)
	x = torch.linspace(1, loss:size()[1], loss:size()[1])
	gnuplot.epsfigure(fileName .. '_loss.eps')
	gnuplot.plot({'Loss for model ' .. modelName, x, loss, '-'})
	gnuplot.xlabel('Epoch')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end