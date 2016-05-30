# Machine learning analysis of Enron email corpus
## Looking for "persons of interest" in the email scandal
## 
## 
### Overview
The Enron email corpus is a compilation of emails sent to and from important
Enron employees during the period during which major financial fraud was being
committed. By evaluating data from the Enron email corpus and public financial
reports using machine learning techniques, we are trying to determine who within
the Enron organization should be considered a “person of interest?” The dataset
contains information about many potential POIs’ financial interest in Enron as
well as their email activity to and from other Enron employees, including
potential POIs.

### Feature Selection
The features (with SelectKBest feature scores) I ended up using are the
following: 
* salary (19.025) 
* bonus (30.076) 
* total\_stock\_value (15.964) 
* exercised\_stock\_options (15.823) 
* long\_term\_incentive (11.365) 
* shared\_receipt\_with\_poi (10.777) 
* fraction\_to\_poi (15.716)

### Algorithm Selection and Tuning
In this case, we are looking for the highest weighted F1 score, which combines
precision and recall into one value that an algorithm can maximize. If you don’t
do this well, the classifier will not be optimized to be as predictive as it
could be.

I used GridSearchCV, an automated way of running multiple iterations of the same
algorithm using different parameter combinations in search of the one that
yields the highest score. As mentioned above, in this case, the high score is
based on the “F1_Weighted,” During the GridSearchCV step, I used Stratified
Shuffle Split on the training labels in an effort to randomize the selection of
testing data due to the small sample size.

I ultimately selected Naive Bayes because it gives me the best combination of
precision and recall for this data set. However,when I tested other algorithms,
like Adaboost, for example, I tried to tune them as best I could to see if the
F1 score could exceed that of Naive Bayes. For Adaboost in particular, I tried
an array of values for both the n_estimators and learning_rate parameters. Since
there is a tradeoff between those two parameters, I wanted to see which
combination worked the best. I ended up with a combination with a very high
precision (0.71), but lower recall (0.12).

### Other Notes
poi\_id\_testing\_all\_algorithms.py is where I ran seven different algorithms
through test_classifier to assess the results

I modified test\_classifier in tester.py to return results to assist with
testing the algorithms in poi\_id\_testing\_all\_algorithms.py

