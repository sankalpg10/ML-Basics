"""
Write a Python function that implements the decision tree learning algorithm for classification.
The function should use recursive binary splitting based on entropy and information gain to build a decision tree.
 It should take a list of examples (each example is a dict of attribute-value pairs) and a list of attribute names as
  input, and return a nested dictionary representing the decision tree.
"""
from tkinter.scrolledtext import example

#input :
"""examples = [
                    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
                    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
                    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
                    {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
                ],
                attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
"""



from collections import Counter
import math

class DecisionTree:
    def __init__(self,depth,y_label):
        self.depth = depth
        self.label = y_label
        self.tree = None

    def get_entropy(self,subset): #label = y_label column name, subset = data subset

        label_counts = Counter(example[self.label] for example in subset)
        total = len(subset)

        entropy = -sum((count/total)*math.log2((count/total)) for count in label_counts.values())

        return entropy


    def get_information_gain(self,parent,left,right):

        total = len(parent)

        info_gain = (self.get_entropy(parent) -
                     (self.get_entropy(left)*(len(left)/total)) -
                     (self.get_entropy(right)*(len(right)/total)))

        return info_gain

    def find_best_split(self,subset,features):

        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None

        #calculate parent entropy
        parent_entropy = self.get_entropy(subset)

        for feature in features:

            #gathering unique values for the current feature
            unique_values = sorted(set(row[feature] for row in subset))

            #try midpoint split for consecutive values for this feature

            for i in range(len(unique_values)-1):

                threshold = (unique_values[i] + unique_values[i+1])/2

                #split the current subset on this threshold

                left = [row for row in subset if row[feature]<= threshold]
                right = [row for row in subset if row[feature]> threshold]

                #ignore bad splits

                if not left or not right:
                    continue

                #calculate info gain from this above split

                info_gain_curr = self.get_information_gain(subset,left,right)

                #update best gain if curr_info_gain is better

                if info_gain_curr > best_gain:
                    best_gain = info_gain_curr
                    best_feature = feature
                    best_threshold = threshold
                    best_left = left
                    best_right = right

        return best_threshold,best_feature,best_gain, best_left, best_right

    def majority_label(self,subset):

        labels = Counter(row[self.label] for row in subset)

        return labels.most_common(1)[0][0] #get the 1st most common label, [0][0] its a tuple


    def decision_tree_training(self,subset,features,parent_majority_label,curr_depth=0,):


        #Checking base cases
        #base case - 1: all labels in subset are same
        if all(row[self.label]== subset[0][self.label] for row in subset):
            return {self.label : subset[0][self.label]} #return this label cause anyway everyone has the same label

        #base case - 2: curr_depth > max_depth : avoid overfitting
        if curr_depth > self.depth:
            return {self.label : self.majority_label(subset)}

        #base case -3: np subset or no features
        if not subset or not features: #if not subset : when no rows left- we might return a defailt label form the parent
                                      #or to keep things simple we can just return "no examples"
                                        #no features : if categorical features - we shud update the attributes list, if numerical we can keep it
            return {self.label : parent_majority_label}

        best_threshold, best_feature, best_gain, left, right = self.find_best_split(subset,features)

        if not best_feature:
            return {self.label : self.majority_label(subset)}


        #recursively build tree

        return {
            'feature' : best_feature,
            'threshold' : best_threshold,
            'left' : self.decision_tree_training(left,features,parent_majority_label,curr_depth+1),
            'right': self.decision_tree_training(left,features,parent_majority_label,curr_depth+1)
        }


    def predict(self,tree,test_case):

        #base case
        if self.label in tree:
            return tree[self.label]

        feature = tree['feature']
        threshold = tree['threshold']


        #traverse the tree
        if test_case[feature] <= threshold:
            return self.predict(tree['left'],test_case)
        else:
            return self.predict(tree['right'],test_case)



if __name__ == "__main__":
    training_data = [
        {'Age': 22, 'Income': 30000, 'Label': 'No'},
        {'Age': 25, 'Income': 35000, 'Label': 'No'},
        {'Age': 30, 'Income': 40000, 'Label': 'No'},
        {'Age': 35, 'Income': 50000, 'Label': 'No'},
        {'Age': 40, 'Income': 60000, 'Label': 'No'},
        {'Age': 45, 'Income': 70000, 'Label': 'Yes'},
        {'Age': 50, 'Income': 80000, 'Label': 'Yes'},
        {'Age': 55, 'Income': 90000, 'Label': 'Yes'}
    ]
    features = ['Age', 'Income']
    decision_tree = DecisionTree(depth = 3,y_label='Label')
    parent_majority_label = decision_tree.majority_label(training_data)
    decision_tree.tree = decision_tree.decision_tree_training(training_data,features,parent_majority_label)

    # Example test case
    test_case = {'Age': 39, 'Income': 100000}

    # Predict the label for the test case
    predicted_label = decision_tree.predict(decision_tree.tree, test_case)

    print(f"Predicted Label: {predicted_label}")
