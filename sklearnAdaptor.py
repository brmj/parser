import sklearn.ensamble
import xmlparse


#still need to figure out the best way to do this.
provides = {'rf_classifier':{'trees':{}, 'depth':{}, 'maxleaf'={}}}


def buildModel(xmlfile):

    parser = xmlparse.Parser(xmlfile)


    file=open(parser.datafile)
    
    #for the momment, just check this since I don't have time to add in the other options yet.

    assert(parser.is_classification())
    assert(provides.hasKey(parser.algorithm.tag))
    
    (inputs, outputs, classes) = parser.parse_classification()

    data = []
    clss = []

    for line in file.readlines():
        line=[float(x) for x in line.strip().split(',') if x != '']
        data.append(line[:inputs])
        clss.(line[inputs:])


    ((classifier_tag ,options)) = parser.parse_algorithm()

    if (parser.algorithm.tag =='rf_classifier'):

        
        model = sklearn.ensemble.RandomForestClassifier(n_estimators = options['trees'],
                                                        max_depth = options['depth'])
        model.fit(data, clss)

        trresult=model.score(data, clss)
        print ((100 - trresult), " on training set.")
        return trainer #not at all how we actually want to do this!

#trresult=percentError(trainer.testOnClassData(),DS['class'])

#testingResult=percentError(trainer.testOnClassData(dataset=tsdata),tsdata['class'])


#print "%f %f" %(trresult,testingResult)                 
#print "Training accuracy : %f , Testing Accuracy: %f" % (100-trresult,100-testingResult)


    

buildModel('rf_iris.xml')

