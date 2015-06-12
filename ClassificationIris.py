from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer
from pybrain.structure import SigmoidLayer
from pybrain.structure import SoftmaxLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
import xmlparse

parser = xmlparse.Parser('mlp_iris.xml')


file=open(parser.datafile)

#for the momment, jsut check this since I don't have time to add in the other options yet.
assert(parser.is_classification())
assert(parser.algorithm.tag == 'feedforward_classifier')

(inputs, outputs, classes) = parser.parse_classification()
DS=ClassificationDataSet(inputs,outputs,nb_classes=classes)

#How do we want to deal with this stuff?
#Should there be a way to handle a split in here, or do we assume the user has done that?
#I lean towards the latter.
for line in file.readlines():
	data=[float(x) for x in line.strip().split(',') if x != '']
	inp=tuple(data[:outputs])
	output=tuple(data[outputs:])
	DS.addSample(inp,output)

tstdata,trndata=DS.splitWithProportion(0.25)
trdata=ClassificationDataSet(trndata.indim,1,nb_classes=classes)
tsdata=ClassificationDataSet(tstdata.indim,1,nb_classes=classes)

for i in xrange(trndata.getLength()):
    trdata.addSample(trndata.getSample(i)[0],trndata.getSample(i)[1])

for i in xrange(tstdata.getLength()):
    tsdata.addSample(tstdata.getSample(i)[0],tstdata.getSample(i)[1])


trdata._convertToOneOfMany()
tsdata._convertToOneOfMany()


((classifier_tag ,trainer_tuple, layers)) = parser.parse_algorithm()

fnn=FeedForwardNetwork()
i = 1
hiddenLayers = []
if(layers[0][0] == "linear"):
        inputLayer = LinearLayer(inputs)
elif(layers[0][0] == "sigmoid"):
        inputLayer = SigmoidLayer(inputs)
elif(layers[0][0] == "softmax"):
        inputLayer = SoftmaxLayer(inputs)
fnn.addInputModule(inputLayer)
for i in range(1, len(layers)- 1):
        num = layers[i][1]
        if(layers[i][0] == "linear"):
                hiddenLayers.append( LinearLayer(num))
        elif(layers[i][0] == "sigmoid"):
                hiddenLayers.append(SigmoidLayer(num))
        elif(layers[i][0] == "softmax"):
                hiddenLayers.append(SoftmaxLayer(num))
for hiddenLayer in hiddenLayers:
        fnn.addModule(hiddenLayer)`
outputLayer=SoftmaxLayer(trdata.outdim)

if(layers[0][0] == "linear"):
        outputLayer = LinearLayer(outputs)
elif(layers[0][0] == "sigmoid"):
        outputLayer = SigmoidLayer(outputs)
elif(layers[0][0] == "softmax"):
        outputLayer = SoftmaxLayer(outputs)
fnn.addOutputModule(outputLayer)
nnlayers = [inputLayer] + hiddenLayers + [outputLayer]
connections = []
inputCons = layers[0][2]
for connection in inputCons:
        typ = connection[0]
        if typ = 'full':
                #we aren't handeling any other type yet, unfortunately.
                tmpcon = FullConnection(inputLayer
in_to_hidden=FullConnection(inputLayer,hiddenLayer)
hidden_to_outputLayer=FullConnection(hiddenLayer,outputLayer)


fnn.addConnection(in_to_hidden)
fnn.addConnection(hidden_to_outputLayer)

for connection in connections:
        fnn.add(connection)

fnn.sortModules()

trainer=BackpropTrainer(fnn,dataset=trdata,momentum=0.1,verbose=True,weightdecay=0.01)
 #training for 50 epochs
#for i in xrange(50):
trainer.trainEpochs(i)

trresult=percentError(trainer.testOnClassData(),trdata['class'])

testingResult=percentError(trainer.testOnClassData(dataset=tsdata),tsdata['class'])


print "%f %f" %(trresult,testingResult)                 
print "Training accuracy : %f , Testing Accuracy: %f" % (100-trresult,100-testingResult)


    



