import csv
import pandas as pd
import math

class Node:
    def __init__(self, isLeaf, label, threshold):
        self.label = label
        self.nodeValue = []
        self.threshold = threshold
        self.isLeaf = isLeaf
        self.children = []

class C4point5:
    def __init__(self, datasetPath):
        self.datasetPath = datasetPath
        self.data = []
        self.classes = []
        self.numAttributes = -1
        self.attrValues = {}
        self.attributes = []
        self.tree = None
        self.pred = None
        
    def readDataset(self):
        with open(self.datasetPath, mode ='r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            for index,row in enumerate(csv_reader):
                if index == 0:
                    self.attributes = [x.strip() for x in list(row[:len(row)-1])]
                    self.numAttributes = len(self.attributes)
                    for i in range(self.numAttributes):
                        self.attrValues[self.attributes[i]] = []
                    continue
            
                for i in range(self.numAttributes):
                    if row[i].strip() not in self.attrValues[self.attributes[i]]:
                        self.attrValues[self.attributes[i]].append(row[i].strip())
                    
                try:
                    if row[self.numAttributes].strip() not in self.classes:
                        self.classes.append(row[self.numAttributes].strip())
                except:
                    print('class value missing!')
                
                line = [x.strip() for x in list(row)]
                if line!= [] or line != [""]:
                    self.data.append(line)
                    
            for key,val in self.attrValues.items():
                flag = 0
                for x in val:
                    try:
                        p = float(x)
                        flag = 1
                        break
                    except:
                        pass
                if flag == 1:
                    self.attrValues[key] = ["continuous"]
    
    def preprocessData(self):
        for index,row in enumerate(self.data):
            #print(index, row)
            for attr_index in range(self.numAttributes):
                #print(attr_index)
                #print(type(self.data[index][attr_index]))
                if(not self.isAttrDiscrete(self.attributes[attr_index])):
                    self.data[index][attr_index] = float(self.data[index][attr_index])
                #print(type(self.data[index][attr_index]))
                
                
    def isAttrDiscrete(self, attribute):
        if attribute not in self.attributes:
            raise ValueError("Attribute not listed")
        elif len(self.attrValues[attribute]) == 1 and self.attrValues[attribute][0] == "continuous":
            return False
        else:
            return True
    
    def allSameClass(self, data):
        for row in data:
            if row[-1] != data[0][-1]:
                return False
        # if all same, then returning the class value
        return data[0][-1]
    
    def generateTree(self):
        self.tree = self.recursiveGenerateTree(self.data, self.attributes)
        
    def recursiveGenerateTree(self, curData, curAttributes):
        allSame = self.allSameClass(curData)
        
        if len(curData) == 0:
            return Node(True, "Fail", None)
        
        elif allSame is not False:
            return Node(True, allSame, None)
        
        elif len(curAttributes) == 0:
            majClass = self.getMajClass(curData)
            return Node(True, majClass, None)
        
        else:
            (best, best_threshold, splitted) = self.splitAttribute(curData, curAttributes)
            remainingAttributes = curAttributes[:]
            #print(self.attrValues[best])
            if self.attrValues[best][0] != 'continuous':
                remainingAttributes.remove(best)
            node = Node(False, best, best_threshold)
            for subset in splitted:
                indx = self.attributes.index(best)
                node.nodeValue.append(subset[0][indx])
            node.children = [self.recursiveGenerateTree(subset, remainingAttributes) for subset in splitted]
            return node
        
    def getMajClass(self, curData):
        freq = [0]*len(self.classes) #[4, 5]
        for row in curData:
            index = self.classes.index(row[-1])
            freq[index] += 1
        maxInd = freq.index(max(freq))
        return self.classes[maxInd]
    
    def splitInfo(self, S, subsets):
        weights = [len(subset)/S for subset in subsets]
        spi = 0
        for i in weights:
            spi += -i * self.log(i)
        return spi
        
    
    def splitAttribute(self, curData, curAttributes):
        splitted = []
        maxInfoGainRatio = -1*float("inf")
        best_attribute = -1
        
        #None for discrete attributes, threshold value for continuous attributes
        best_threshold = None
        
        for attribute in curAttributes:
            indexOfAttribute = self.attributes.index(attribute)
            if self.isAttrDiscrete(attribute):
                valuesForAttribute = self.attrValues[attribute]
                subsets = [[] for a in valuesForAttribute]
                for row in curData:
                    for index in range(len(valuesForAttribute)):
                        if row[indexOfAttribute] == valuesForAttribute[index]:
                            subsets[index].append(row)
                            break
                e = self.gain(curData, subsets)
                s = self.splitInfo(len(curData), subsets)
                """if s!= 0:
                    gr = e/s
                else:
                    gr = e"""
                gr = e/s
                if gr > maxInfoGainRatio:
                    maxInfoGainRatio = gr
                    splitted = subsets
                    best_attribute = attribute
                    best_threshold = None
                    
            else:
                curData.sort(key = lambda x: x[indexOfAttribute])
                indexOfClass = len(attribute)
                #print(curData)
                #exit()
                for j in range(0, len(curData) - 1):
                    if curData[j][indexOfClass] != curData[j+1][indexOfClass]:
                        threshold = (curData[j][indexOfAttribute] + curData[j+1][indexOfAttribute]) / 2
                        less = []
                        greater = []
                        for row in curData:
                            if(row[indexOfAttribute] > threshold):
                                greater.append(row)
                            else:
                                less.append(row)
                        e = self.gain(curData, [less, greater])
                        s = self.splitInfo(len(curData), [less, greater])
                        """if s!= 0:
                            gr = e/s
                        else:
                            gr = e"""
                        gr = e/s
                        if gr >= maxInfoGainRatio:
                            splitted = [less, greater]
                            maxInfoGainRatio = gr
                            best_attribute = attribute
                            best_threshold = threshold
        return (best_attribute,best_threshold,splitted)
    
    def gain(self,unionSet, subsets):
        #input : data and disjoint subsets of it
        #output : information gain
        S = len(unionSet)
        #calculate impurity before split
        impurityBeforeSplit = self.entropy(unionSet) # entropy
        #calculate impurity after split
        weights = [len(subset)/S for subset in subsets]
        impurityAfterSplit = 0
        for i in range(len(subsets)):
            impurityAfterSplit += weights[i]*self.entropy(subsets[i])
        #calculate total gain
        totalGain = impurityBeforeSplit - impurityAfterSplit
        return totalGain
    
    def entropy(self, dataSet):
        S = len(dataSet)
        if S == 0:
            return 0
        num_classes = [0 for i in self.classes]
        for row in dataSet:
            classIndex = list(self.classes).index(row[-1])
            num_classes[classIndex] += 1
        num_classes = [x/S for x in num_classes]
        ent = 0
        for num in num_classes:
            ent += num*self.log(num)
        return ent*-1
    
    def log(self, x):
        if x==0:
            return 0
        else:
            return math.log(x,2)
        
    def printTree(self):
        self.printNode(self.tree)
        
    def printNode(self, node, indent=""):
        if not node.isLeaf:
            if node.threshold is None:
                #discrete
                for index,child in enumerate(node.children):
                    if child.isLeaf:
                        print(indent + node.label + " = " + node.nodeValue[index] + " : " + child.label)
                    else:
                        print(indent + node.label + " = " + node.nodeValue[index] + " : ")
                        self.printNode(child, indent + "    ")
            else:
                #continuous
                leftChild = node.children[0]
                rightChild = node.children[1]
                if leftChild.isLeaf:
                    print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
                else:
                    print(indent + node.label + " <= " + str(node.threshold)+" : ")
                    self.printNode(leftChild, indent + "    ")
                
                if rightChild.isLeaf:
                    print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
                else:
                    print(indent + node.label + " > " + str(node.threshold) + " : ")
                    self.printNode(rightChild , indent + "    ")
        pass
    
    def predict(self):
        row = []
        for i in range(len(self.attributes)):
            inp = input('Enter value for '+str(self.attributes[i])+str(self.attrValues[self.attributes[i]])).strip()
            row.append(inp)
            
        self.prediction(self.tree, row)
        print('prediction :',self.pred)
        pass
    
    def prediction(self, node, row):
        if not node.isLeaf:
            if node.threshold is None:
                #discrete
                indexOfAttribute = self.attributes.index(node.label)
                inp = row[indexOfAttribute]
                for index,child in enumerate(node.children):
                    if node.nodeValue[index] == inp:
                        if child.isLeaf:
                            #print('prediction :',child.label)
                            self.pred = child.label
                        else:
                            self.prediction(child, row)
                
            else:
                #continuous (have threshold value)
                indexOfAttribute = self.attributes.index(node.label)
                inp = float(row[indexOfAttribute])
                leftChild = node.children[0]
                rightChild = node.children[1]
                if inp <= node.threshold:
                    if leftChild.isLeaf:
                        #print('prediction :',leftChild.label)
                        self.pred = leftChild.label
                    else:
                        self.prediction(leftChild, row)
                else:
                    if rightChild.isLeaf:
                        #print('prediction :',rightChild.label)
                        self.pred = rightChild.label
                    else:
                        self.prediction(rightChild, row)
        pass
    
    def evaluate(self, dataPath):
        data = pd.read_csv(dataPath)
        length = len(data)
        true = []
        predicted = []
        for i in range(length):
            row = [str(i).strip() for i in list(data.iloc[i])[:-1]]
            true.append(str(data.iloc[i][-1]).strip())
            #row = list(data.iloc[i])
            #print(row)
            self.prediction(self.tree, row)
            #print(self.pred)
            predicted.append(self.pred)
        
        accurate = 0
        error = 0
        for i in range(length):
            if true[i] == predicted[i]:
                accurate += 1
            else:
                error += 1
                
        print('Accuracy:', (accurate/(accurate+error))*100, '%')
            #print(true[i] == predicted[i])
            
    def viewTruePred(self, dataPath):
        data = pd.read_csv(dataPath)
        length = len(data)
        true = []
        predicted = []
        for i in range(length):
            row = [str(i).strip() for i in list(data.iloc[i])[:-1]]
            true.append(data.iloc[i][-1].strip())
            self.prediction(self.tree, row)
            predicted.append(self.pred)
            
        print('True label', 'Predicted label')
        for i in range(length):
            print(true[i], predicted[i])
