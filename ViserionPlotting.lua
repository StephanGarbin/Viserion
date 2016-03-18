

function plotLoss(loss, modelName)
	local x = torch.linspace(1, loss:size()[1], loss:size()[1])
	gnuplot.epsfigure(modelName .. '_loss.eps')
	gnuplot.plot(loss, '-')
	gnuplot.xlabel('Epoch')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
	gnuplot.close()
end