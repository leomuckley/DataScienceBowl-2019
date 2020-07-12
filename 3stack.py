#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter
from functools import partial
import scipy as sp
from scipy.stats import mode


from sklearn.ensemble import ExtraTreesRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


from ngboost.scores import MLE
from ngboost.learners import default_tree_learner
from ngboost import NGBRegressor

from mlxtend.regressor import StackingCVRegressor, LinearRegression


# pip install --upgrade git+https://github.com/stanfordmlgroup/ngboost.git


pd.set_option('display.max_columns', 1000)

SEED = 42

def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code

# this is the function that convert the raw data into processed features
def get_data(user_sample, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0
    
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    
    # new features: time spent in each activity
    last_session_time_sec = 0
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}
        
    # last features
    sessions_count = 0
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
                    
            
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())
            features['installation_session_count'] = sessions_count
            
            variety_features = [('var_event_code', event_code_count),
                              ('var_event_id', event_id_count),
                               ('var_title', title_count),
                               ('var_title_event_code', title_event_code_count)]
            
            for name, dict_counts in variety_features:
                arr = np.array(list(dict_counts.values()))
                features[name] = np.count_nonzero(arr)
                 
            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
                features['duration_std'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
                features['duration_std'] = np.std(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions
            
            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
                
            counter += 1
        
        sessions_count += 1
        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter
            
        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type 
                        
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0
    
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    
    # new features: time spent in each activity
    event_code_count = {eve: 0 for eve in list_of_event_code}
    last_session_time_sec = 0
    
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
                    
            
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(event_code_count.copy())
            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions
            
            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
                
            counter += 1
        
        # this piece counts how many actions was made in each event_code so far
        n_of_event_codes = Counter(session['event_code'])
        
        for key in n_of_event_codes.keys():
            event_code_count[key] += n_of_event_codes[key]

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type 
                        
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments

def get_train_and_test(train, test):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in enumerate(train.groupby('installation_id', sort = False)):
        compiled_train += get_data(user_sample)
    for ins_id, user_sample in test.groupby('installation_id', sort = False):
        test_data = get_data(user_sample, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    categoricals = ['session_title']
    return reduce_train, reduce_test, categoricals


def qwk(act,pred,n=4,hist_range=(0,3)):
    
    # Calculate the percent each class was tagged each label
    O = confusion_matrix(act,pred)
    # normalize to sum 1
    O = np.divide(O,np.sum(O))
    
    # create a new matrix of zeroes that match the size of the confusion matrix
    # this matriz looks as a weight matrix that give more weight to the corrects
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # makes a weird matrix that is bigger in the corners top-right and botton-left (= 1)
            W[i][j] = ((i-j)**2)/((n-1)**2)
            
    # make two histograms of the categories real X prediction
    act_hist = np.histogram(act,bins=n,range=hist_range)[0]
    prd_hist = np.histogram(pred,bins=n,range=hist_range)[0]
    
    # multiply the two histograms using outer product
    E = np.outer(act_hist,prd_hist)
    E = np.divide(E,np.sum(E)) # normalize to sum 1
    
    # apply the weights to the confusion matrix
    num = np.sum(np.multiply(W,O))
    # apply the weights to the histograms
    den = np.sum(np.multiply(W,E))
    
    return 1-np.divide(num,den)


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

        return -qwk(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']


       

# read data
train, test, train_labels, specs, sample_submission = read_data()

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)
# tranform function to get the train and test set
reduce_train, reduce_test, categoricals = get_train_and_test(train, test)
 

# this list comprehension create the list of features that will be used on the input dataset X
# all but accuracy_group, that is the label y
all_features = [x for x in reduce_train.columns if x not in ['accuracy_group', 'installation_id']]
# this cat_feature must be declared to pass later as parameter to fit the model
cat_features = ['session_title']
# here the dataset select the features and split the input ant the labels
X, y = reduce_train[all_features].values, reduce_train['accuracy_group'].apply(float)


# Define groups for group K-fold
groups = train_labels['installation_id']


def make_regressor(iterations=6000, clf=None):
    
    if clf == 'cat':            
        clf = CatBoostRegressor(
            loss_function='RMSE',
            # eval_metric="WKappa",
            task_type="CPU",
            #learning_rate=0.01,
            iterations=iterations,
            od_type="Iter",
            #depth=4,
            early_stopping_rounds=500,
            #l2_leaf_reg=10,
            #border_count=96,
            random_seed=SEED,
            #use_best_model=True
            )
        
    if clf == 'xgb':        
        clf = xgb.XGBRegressor(
            n_estimators = 5000,
            max_depth = 10,
            min_child_weight = 3,
            gamma = 0.25,
            n_jobs = -1,
            #verbosity=3,
            random_state=SEED
            )
        
    if clf == 'lgb':        
        clf = lgb.LGBMRegressor(
            learning_rate = 0.01,
            n_estimators = 2000,
            max_depth = 15,
            #reg_alpha = 1,
            #reg_lambda = 1,
            random_state=SEED
            )
            
        
    if clf == 'ngb':        
        clf = NGBRegressor(
            Dist=Normal,
            Score=MLE,
            Base=default_tree_learner,
            natural_gradient=True,
            n_estimators=2000,
            learning_rate=0.01,
            minibatch_frac=0.6,
            verbose=True,
            verbose_eval=50
            )
        
    if clf == 'ext':        
        clf = ExtraTreesRegressor(
            #learning_rate = 0.01,
            n_estimators = 2000,
            max_depth = 15,
            #reg_alpha = 1,
            #reg_lambda = 1,
            random_state=SEED,
            verbose=3,
            #n_jobs=-1
            )
            
    return clf


def get_coef(model, X, y):
    """ Return model coefficients. """
    pr = model.predict(X)
    optR.fit(pr.reshape(-1,), y)
    return optR.coefficients()
    

def make_predictions(model, coefs):
    """ Return predictions based on coefficient values. """
    predictions = model.predict(reduce_test[all_features].values)
    # Maintian oridnal structure           
    predictions[predictions <= coefs[0]] = 0
    predictions[np.where(np.logical_and(predictions > coefs[0], predictions <= coefs[1]))] = 1
    predictions[np.where(np.logical_and(predictions > coefs[1], predictions <= coefs[2]))] = 2
    predictions[predictions > coefs[2]] = 3
    
    return predictions.astype('int')




# Define models for the first ensemble (using group k-fold)
model1 = make_regressor(clf='cat')

model2 = make_regressor(clf='xgb')

model3 = make_regressor(clf='lgb')

lr = LinearRegression()

stack_one = StackingCVRegressor(regressors=[model1, model2, model3], 
                            meta_regressor=lr, 
                            cv=5,
                            verbose=0,
                            store_train_meta_features=True,
                            random_state=SEED)

stack_one.fit(X, y, groups=groups)

## Return ordinal predictions
pr1 = stack_one.predict(X)
optR = OptimizedRounder()
coefs_one = get_coef(stack_one, X, y)
#print(coefs_one)
#one_preds = optR.predict(pr1.reshape(-1, ), coefs_one)
#print(np.round(qwk(y, one_preds), 3))
# lr.fit(stack_one.train_meta_features_, y)
# p1 = lr.predict(stack_one.predict_meta_features(X))



# Define models for the second ensemble (using group k-fold)
model4 = make_regressor(clf='ngb')

model5 = make_regressor(clf='cat')

model6 = make_regressor(clf='lgb')

lr = LinearRegression()


stack_two = StackingCVRegressor(regressors=[model4, model5, model6], 
                            meta_regressor=lr, 
                            cv=5,
                            verbose=0,
                            store_train_meta_features=True,
                            random_state=SEED)

stack_two.fit(X, y, groups=groups)

## Return ordinal predictions
pr2 = stack_two.predict(X)
optR = OptimizedRounder()
coefs_two = get_coef(stack_two, X, y)
#print(coefs_two)
#two_preds = optR.predict(pr2.reshape(-1, ), coefs_two)
#print(np.round(qwk(y, two_preds), 3))




# Define models for the thrid ensemble (using group k-fold)
model7 = make_regressor(clf='cat')
model8 = make_regressor(clf='ext')
model9 = make_regressor(clf='lgb')

lr = LinearRegression()


stack_three = StackingCVRegressor(regressors=[model7, model8, model9], 
                            meta_regressor=lr, 
                            cv=5,
                            verbose=0,
                            store_train_meta_features=True,
                            random_state=SEED)

stack_three.fit(X, y, groups=groups)

# Return ordinal predictions
pr3 = stack_three.predict(X)
optR = OptimizedRounder()
coefs_three = get_coef(stack_three, X, y)
#print(coefs_three)
#three_preds = optR.predict(pr3.reshape(-1, ), coefs_three)
#print(np.round(qwk(y, three_preds), 3))




# Get final preds
stack1_preds = make_predictions(stack_one, coefs_one)
stack2_preds = make_predictions(stack_two, coefs_two)
stack3_preds = make_predictions(stack_three, coefs_three)


res_df = pd.DataFrame(mode([stack1_preds, stack2_preds, stack3_preds])[0].reshape((1000,)))

res_df.to_csv('result.csv', index=False)




























