#!/usr/bin/python

import sys
import pickle
import pprint
import matplotlib.pyplot 
sys.path.append("../tools/")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC #SVC or SVM?
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.lda import LDA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'pct_poi_inbound',
                 'pct_poi_outbound'] 
                 
financial_features = ['salary', 'deferral_payments', 'total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred', 
                      'deferred_income', 'total_stock_value', 'expenses', 
                      'exercised_stock_options', 'other', 
                      'long_term_incentive', 'restricted_stock', 
                      'director_fees']
                      
email_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 
                  'from_messages', 'from_this_person_to_poi', 
                  'shared_receipt_with_poi']
email_features.remove('email_address')

                  
# You will need to use more features
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

#pprint.pprint(data_dict['TOTAL'])
data_dict.pop('TOTAL', 0)

# Do any records have no financial data? If so, no action is required
# featureFormat takes care of that (below)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

def compute_fraction( numerator, denominator ):
    if numerator == 'NaN' or denominator == 'NaN':
        fraction = 0
    else:
        fraction = float(numerator)/float(denominator)    
    return round(fraction, 2)
    
    
def add_fraction_to_dict(dict, numerator, denominator, new_variable_name):
    num = dict[numerator]
    den = dict[denominator]
    fraction = compute_fraction(num, den)
    dict[new_variable_name] = fraction
    return dict
    
    
my_dataset = data_dict
for p in my_dataset:   
    # Calculate inbound POI email fraction
    my_dataset[p] = add_fraction_to_dict(my_dataset[p],
                                        'from_poi_to_this_person', 
                                        'to_messages',
                                        'fraction_from_poi')
        
    # Calculate outbound POI email fraction
    my_dataset[p] = add_fraction_to_dict(my_dataset[p],
                                        'from_this_person_to_poi',
                                        'from_messages',
                                        'fraction_to_poi')    
    
    # Calculate Exercised Stock Options as fraction of Total Stock Value
#    my_dataset[p] = add_fraction_to_dict(my_dataset[p],
#                                        'exercised_stock_options',
#                                        'total_stock_value',
#                                        'fraction_stock_exercised')
   
    # Calculate Salary as fraction of Total Payments
    my_dataset[p] = add_fraction_to_dict(my_dataset[p],
                                        'salary',
                                        'total_payments',
                                        'fraction_salary_total_payments') 
                                        
    
email_features = email_features + ['fraction_from_poi', 'fraction_to_poi']
financial_features = financial_features + ['fraction_salary_total_payments']
                  
features_list = ['poi'] + financial_features + email_features
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#print "length of data numpy array:", len(data)
#print "Features List:", features_list
#print "Labels:", labels
#print "Features:", features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
algorithms = [ 'Naive_Bayes', 
               'SVC',
               'Standard_Decision_Tree', 
               'K_Nearest_Neighbors',
               'Adaboost',
               'Random_Forest',
               'LDA'
             ]

# Created by DKS 1/26/16 to facilitate comparing different algorithms
def create_classifier_step(algorithm):
    cl_params = {}
    if algorithm == 'Naive_Bayes':
        cl = GaussianNB()
    elif algorithm == 'SVC':
        cl = SVC()
        cl_params = { algorithm + '__kernel' : ['rbf', 'poly'],
                      algorithm + '__C' : [1000, 10000, 100000]
                    }
    elif algorithm == 'Standard_Decision_Tree':
        cl = tree.DecisionTreeClassifier()
        cl_params = { algorithm + '__min_samples_split' : [30, 40, 50] }
    elif algorithm == 'K_Nearest_Neighbors':
        cl = KNeighborsClassifier()
        cl_params = { algorithm + '__n_neighbors' : [6, 8, 10],
                      algorithm + '__weights' : ['uniform']
                    }
    elif algorithm == 'Adaboost':
        cl = AdaBoostClassifier()
        cl_params = { algorithm + '__n_estimators' : [5, 8, 10, 20, 30, 50, 100],
                      algorithm + '__learning_rate' : [0.025, 0.05, 0.1, 0.5, 1, 2, 4, 6]
                    }
    elif algorithm == 'Random_Forest':
        cl = RandomForestClassifier()
        cl_params = { algorithm + '__max_features' : ['sqrt', 'log2'],
                      algorithm + '__n_estimators' : [2, 5, 7, 10, 15]}
    elif algorithm == 'LDA':
        cl = LDA()
        cl_params = { algorithm + '__solver' : ['svd', 'lsqr', 'eigen']}
    return (algorithm, cl), cl_params
    

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.
### StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


features_train, features_test, labels_train, labels_test = train_test_split(
                                                            features, 
                                                            labels, 
                                                            test_size=0.2, 
                                                            random_state=42)

#pipe = Pipeline(steps=[
#                        ('SKB', SelectKBest(f_classif)),
#                        ('PCA', PCA()),
#                        ('NaiveBayes', GaussianNB())
#                      ]
#                )

algorithm_comparison = [['ALGORITHM', 'ACCU', 'PREC', 'RECA', 'F1', 'F2']]                
for a in algorithms:
    classifer_step, clf_step_params = create_classifier_step(a)
    print '\nNOW RUNNING', classifer_step[0].upper()
    
    min_max_scaler = MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(features_train)
    #print "Scaled Values:\n:", x_train_minmax
    
    
                                            
    pipe = Pipeline(steps=[
                            ('MMS', MinMaxScaler()),
                            ('SKB', SelectKBest()),
                            #('PCA', PCA()),
                            classifer_step
                          ]
                    )
    
    params = {
                #'PCA__n_components': [2],
                'SKB__k' : [5, 6, 7, 8, 9, 10, 11, 12],
                'SKB__score_func' : [f_classif]
             }
    params.update(clf_step_params)
    
    sss = StratifiedShuffleSplit(labels_train, n_iter = 20, test_size = 0.5,
                                 random_state = 0)
                                 
                 
    gscv = GridSearchCV(pipe,
                       params,
                       verbose = 0,
                       scoring = 'f1_weighted',
                       cv=sss
                       )
    
    gscv.fit(features_train, labels_train)
    
    pred = gscv.predict(features_test)

    clf = gscv.best_estimator_
    
    
    # Get the selected features
#    pipe.fit(features_train, labels_train)
    selected_features = gscv.best_estimator_.named_steps['SKB'].get_support(indices=True)
    feature_scores = gscv.best_estimator_.named_steps['SKB'].scores_
    sfs = []
    for sf in selected_features: 
        sfs.append((features_list[sf + 1], feature_scores[sf]))         
    print len(sfs), "best parameters with scores:"
    for f, s in sfs: print f, "{0:.3f}".format(s)
    

#    print len(feature_scores), "feature scores:\n", feature_scores
#    print len(features_list), "features", features_list
#    features_and_scores = zip(features_list[1:], feature_scores)
#    print "\nAll", len(features_and_scores), "features with scores:"
#    agg_score = 0
#    for f, s in features_and_scores: 
#        print f, "{0:.3f}".format(s)
#        agg_score += s
#    print "\nSum of scores", agg_score
    
    # Test the model using the hold-out test data
#    pred = clf.predict(features_test)
    print "\n", a, "performance report:"
    print(classification_report(labels_test, pred))

    #Will need this line eventually, because this is the test that needs to pass
    print '\nNow running test_classifier...'
    #print "\nFeatures_list for test_classifier:", features_list
    acc, prec, rec, f1, f2 = test_classifier(clf, my_dataset, features_list)
    algorithm_comparison.append([a, 
                                 "{0:.2f}".format(acc), 
                                 "{0:.2f}".format(prec), 
                                 "{0:.2f}".format(rec), 
                                 "{0:.2f}".format(f1), 
                                 "{0:.2f}".format(f2)
                                ]
                               )

print "\n"    
for algo in algorithm_comparison:
    print algo[0].ljust(22), algo[1], algo[2], algo[3]

'''
for a in algorithms:
    clf = create_classifier(a)        
    clf = clf.fit(features_train, labels_train)  
    pred = clf.predict(features_test)
    acc = accuracy_score(pred, labels_test)
    print 'Algorithm:', a
    print "Accuracy:", acc
    #print "Feature Importances:", clf.feature_importances_
    print "Sum of predictions:", sum(pred)
    print "Number of predictions:", len(pred)
    print "Predictions * Labels Test:", pred*labels_test
    print classification_report(labels_test, pred)
    test_classifier(clf, my_dataset, features_list)
'''

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
