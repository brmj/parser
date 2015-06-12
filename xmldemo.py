import xml.etree.cElementTree as ET
 
#----------------------------------------------------------------------
classification_algorithms = {"mlp_classifier", "rf_classifier", "svm_classifier"}
#Will want a nicer way to extend this sort of thing.



class Parser:
    def __init__(self, filename):
        self.tree = ET.ElementTree(file=filename)
        self.root = self.tree.getroot()
        self.problem = self.root.find("problem")[0]
        self.algorithm = self.root.find("algorithm")[0]
        self.datafile = self.problem.filename.text
        self.alg_parse_funcs = {'mlp_classifier':self.parse_mlp_classifier,
                                'rf_classifier':self.parse_rf_classifier}

    def is_classification(self):
        return self.problem.tag == "classification_problem"

    def is_regression(self):
        return self.problem.tag == "regression_problem"

    def is_prediction(self):
        return self.problem.tag == "prediction_problem"

    def parse_classification(self):
        assert(self.problem.tag == "classification_problem")
        classes = 2
        clss = self.problem.findall("classes")
        if(len(clss) > 0):
            classes = int(clss[0].text)
        inputs = int(self.problem.find("inputs").text)
        outputs = int(self.problem.find("outputs").text)
        return (inputs, outputs, classes)

    def parse_regression(self):
        assert(self.problem.tag == "regression_problem")
        inputs = int(self.problem.find("inputs").text)
        outputs = int(self.problem.find("outputs").text)
        return (inputs, outputs)

    def is_classification_algorithm(self):
        return self.algorithm.tag in classification_algorithms

    def parse_algorithm(self):
        return self.alg_parse_funcs[self.algorithm.tag](self.algorithm)

    

    def parse_mlp_classifier(self, classifier):
        trainer = classifier.find("trainer")[0]
        #if (trainer.tag == "backprop_trainer"):
        trainer_options = {}
        for child in trainer:
            trainer_options[child.tag] = child.text #probably the cleanest easy way to do this...
        trainer_tuple = (trainer.tag, trainer_options)

        layers = []
        input = trainer.find("input")
        layers.append((input.get('type'), "I"))
        done = False
        parent = input
        while not done:
            hidden = parent.findall("hidden")
            if (len(hidden) > 0):
                layer = hidden[0]
                layers.append((layer.get('type'), int(layer.get('number'))))
                parent = layer
            else:
                output = parent.find("output")
                layers.append((output.get('type'), "O"))
                done = True

        return ((classifier.tag ,trainer_tuple, layers))
        

    def parse_rf_classifier(self, classifier):
        trs = classifier.findall("trees")
        options = {}
        if (len(trs) == 0):
           options['trees'] = 30 #An arbitrary default.
        else:
            options['trees'] = trs[0].text
        dpth = algorithm.findall("max_depth")
        if (len(dpth) == 0):
            options['depth'] = -1 #for unlimited.
        else:
            options['depth'] = dpth[0].text

        return((classifier.tag, options))

if __name__ == "__main__":
    parseXML("xmldemo.xml")
