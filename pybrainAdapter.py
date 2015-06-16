from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer
from pybrain.structure import SigmoidLayer
from pybrain.structure import SoftmaxLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
import xmlparse


#still need to figure out the best way to do this.
provides = {'feedforward_clasifier':{'backprop_trainer':{'momentum':{}, 'weight':{}, 'epochs':{}},
                                     'linear_layer':{}, 'sigmoid_layer':{}, 'softmax_layer':{},
                                     'full_connection':{}}}


def buildModel(xmlfile):

    parser = xmlparse.Parser(xmlfile)


    file=open(parser.datafile)
    
    #for the momment, just check this since I don't have time to add in the other options yet.

    assert(parser.is_classification())
    assert(provides.hasKey(parser.algorithm.tag))
    
    (inputs, outputs, classes) = parser.parse_classification()
    DS=ClassificationDataSet(inputs,outputs,nb_classes=classes)


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

    if(classes > 2):
        DS._convertToOneOfMany()

    ((classifier_tag ,trainer_tuple, layers)) = parser.parse_algorithm()

    if (parser.algorithm.tag =='feedforward_classifier'):
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
            fnn.addModule(hiddenLayer)

        if(layers[0][0] == "linear"):
            outputLayer = LinearLayer(outputs)
        elif(layers[0][0] == "sigmoid"):
            outputLayer = SigmoidLayer(outputs)
        elif(layers[0][0] == "softmax"):
            outputLayer = SoftmaxLayer(outputs)
            fnn.addOutputModule(outputLayer)
        nnlayers = [inputLayer] + hiddenLayers + [outputLayer]
        for i in range (0, len(layers) - 1):
            connections = []
            inputCons = layers[0][2]
            for connection in inputCons:
                typ = connection[0]
                if typ == 'full':
                #we aren't handeling any other type yet, unfortunately.
                    tmpcon = FullConnection(nnlayers[i], nnlayers[connection[1] + i])
                    fnn.addConnection(tmpcon)
                    
        fnn.sortModules()
        trainer = None
        if trainer_tuple[0] == 'backprop_trainer':
            momentum = 0.2
            weightdecay = 0.05
            epochs = 25
            #just some defaults that aren't what's in the xml for now...

            if 'momentum' in trainer_tuple[1]:
                momentum = float((trainer_tuple[1])['momentum'])
            if 'weightdecay' in trainer_tuple[1]:
                weightdecay = float((trainer_tuple[1])['weightdecay'])
            if 'epochs' in trainer_tuple[1]:
                epochs = int((trainer_tuple[1])['epochs'])

            trainer=BackpropTrainer(fnn,dataset=DS,momentum=momentum,verbose=True,weightdecay=weightdecay)
            
            for i in range(0, epochs):
                trainer.trainEpochs(i)

            trresult=percentError(trainer.testOnClassData(),DS['class'])
            print ((100 - trresult), " on training set.")
            return trainer #not at all how we actually want to do this!

#trresult=percentError(trainer.testOnClassData(),DS['class'])

#testingResult=percentError(trainer.testOnClassData(dataset=tsdata),tsdata['class'])


#print "%f %f" %(trresult,testingResult)                 
#print "Training accuracy : %f , Testing Accuracy: %f" % (100-trresult,100-testingResult)


    

buildModel('mlp_iris.xml')

