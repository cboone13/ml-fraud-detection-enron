#!/usr/bin/python

import sys
import pickle
sys.path.append("C:/Users/cboone/Repo/ud120-projects/tools/")
# sys.path.append("../tools/")
import matplotlib.pyplot as plot
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from sklearn.ensemble import RandomForestClassifier
from tester import dump_classifier_and_data
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import average_precision_score,\
                            accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from tester import test_classifier
import warnings

warnings.filterwarnings("ignore")

def tuning(grid_search, features, labels,params, iters = 100):

    accuracy = []
    precision = []
    recall = []

    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size = 0.3, random_state = i)

        grid_search.fit(features_train, labels_train)
        predict = grid_search.predict(features_test)

        accuracy = accuracy + [accuracy_score(labels_test, predict)]
        precision = precision = [precision_score(labels_test, predict)]
        recall = recall + [recall_score(labels_test, predict)]

    print "accuracy_score: ", np.mean(accuracy) 
    print "precision: ", np.mean(precision)
    print "recall: ", np.mean(recall)

    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'long_term_incentive', 
                 'expenses', 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'to_messages', 'from_messages' ] # You will need to use more features
poi_label = ['poi']
total_features = poi_label + features_list
### Load the dictionary containing the dataset
with open("C:/Users/cboone/Repo/ud120-projects/final_project/final_project_dataset.pkl",
          "r") as data_file:
    # data_dict = pickle.load()
    data_dict = pickle.load( open("C:/Users/cboone/Repo/ud120-projects/final_project/final_project_dataset.pkl", "r") )


### Task 2: Remove outliers
            
def show_scatter_plot(dataset, feature1, feature2):
    """ This function takes two features and plots them
        on a scatter plot to aid in identifying and 
        removing outliers.
    """
    data = featureFormat(dataset, [feature1, feature2])
    for point in data:
        x = point[0]
        y = point[1]
        plot.scatter(x, y)

    plot.xlabel(feature1)
    plot.ylabel(feature2)
    plot.show()

# identify outliers
# show_scatter_plot(data_dict, "salary", "bonus")

data_dict.pop("TOTAL", 0)

# save dataset with outliers removed to new variable.
my_dataset = data_dict

# show_scatter_plot(data_dict, "salary", "bonus")

### Task 3: Create new feature(s)

def computeFraction( poi_messages, all_messages ):
    """ given a number messages from POI (numerator) 
        and number of all messages to a person (denominator),
        return the fraction of messages to that person
        that are from/to a POI
   """

    ### number of emails from an loi / total number of emails recieved.
    fraction = 0

    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = (poi_messages / float(all_messages))

    return fraction

# parse out the from_poi_to_this_person and from_this_person_to_poi from dataset.
# will allow to compute fraction for emails to/from poi.

for employee in my_dataset:
    from_poi_to_this_person = my_dataset[employee]['from_poi_to_this_person']
    to_messages = my_dataset[employee]['to_messages']
    from_this_person_to_poi = my_dataset[employee]['from_this_person_to_poi']
    from_messages = my_dataset[employee]['from_messages']

    # get fractions for to/from
    fraction_from = computeFraction(from_poi_to_this_person, to_messages)
    fraction_to = computeFraction(from_this_person_to_poi, from_messages)

    # add fraction features to dataset.
    my_dataset[employee]['fraction_from'] = fraction_from
    my_dataset[employee]['fraction_to'] = fraction_to

new_features_list = total_features + ['fraction_from', 'fraction_to']

def sort_key(elem):
    
    return elem[1]

# select features 
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
selector = SelectKBest(f_classif, k = 10)
selector.fit(features, labels)
scores = zip(new_features_list[1:], selector.scores_)
sorted_scores = sorted(scores, key = sort_key, reverse = True)
print 'Sorted scores: ', sorted_scores

# Appends the poi feature name inside sorted scores list to the poi_label list 
# and assign to their own variable for use with kBest
kBest_features = poi_label + [(i[0]) for i in sorted_scores[0:10]]
print 'KBest', kBest_features

# Replace NaN values with a 0
for emp in data_dict:
    for f in data_dict[emp]:
        if data_dict[emp][f] == 'NaN':
            # fill NaN values
            data_dict[emp][f] = 0

my_dataset = data_dict

kBest_features.remove('fraction_to')

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# split 40% of the data for testing


data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# select features using SelectKBest.
selector = SelectKBest(f_classif, k = 10)
selector.fit(features, labels)

# store SelectKBest scores and match/add to features list for each employee.
selectK_scores = zip(new_features_list[1:], selector.scores_)
      
#print (selectK_scores)
#print('-----------')

features_train, features_test, labels_train, labels_test \
        = train_test_split( 
                           features,
                           labels,
                           test_size=0.4,
                           random_state=42
                          )

# Provided to give you a starting point. Try a variety of classifiers.

# Naive Bayes Algorithm

clf1 = GaussianNB()
clf1.fit(features_train, labels_train)
nb_pred = clf1.predict(features_test)
nb_score = clf1.score(features_test, labels_test)

nb_acc = accuracy_score(labels_test, nb_pred)
nb_pre = precision_score(labels_test, nb_pred)
nb_rec = recall_score(labels_test, nb_pred)
print "NAIVE BAYES TEST"
print "NB accuracy: ", nb_acc
print "NB precision: ", nb_pre
print "NB recall: ", nb_rec

# Send to GridSearch for tuning parameters
nb_param = {}
nb_grid_search = GridSearchCV(estimator = clf1, param_grid = nb_param)

print "\n NAIVE BAYES TUNING PARAMS"
tuning(nb_grid_search, features, labels, nb_param)

# Random Forest Classifier

clf2 = RandomForestClassifier(n_estimators = 10)
clf2.fit(features_train, labels_train)
rfc_pred = clf2.predict(features_test)

rf_acc = accuracy_score(labels_test, rfc_pred)
rf_pre = precision_score(labels_test, rfc_pred)
rf_rec = recall_score(labels_test, rfc_pred)

print "\n RANDOM FOREST CLASSIFIER TEST"
print "RF accuracy: ", rf_acc
print "RF precision: ", rf_pre
print "RF recall: ", rf_rec

# Send to GridSearch for tuning parameters
rf_param = {}
rf_grid_search = GridSearchCV(estimator = clf2, param_grid = rf_param)

print "RANDOM FOREST CLASSIFIER TUNING PARAMS"
tuning(rf_grid_search, features, labels, rf_param)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


        
# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#test_clf_list = [ 
                # clf1,
                # clf2
                #]

#features_list
#for i in test_clf_list:
#    print " "           
#    test_classifier(i, my_dataset, kBest_features)

clf = clf2

test_classifier(clf, my_dataset, kBest_features)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

