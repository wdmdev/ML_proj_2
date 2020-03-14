#Add ann to path to use data functions
import sys
sys.path.insert(1,'../')
from ann import create_dataset as cd, get_data, fix_nans, TARGET_FEATURE, FEATURES, EPS

import numpy as np

def to_risk_cat(y):
    '''
    Transforms the standardized(subtracted mean and divided std)
    y data point into string labels for classification based on the intervals:
    Low:    [-2.2,1.5]
    Medium: ]1.5,3.1]
    High:   ]3.1,4.6] 
    '''
    return 'low' if (y <= 0.07) else 'medium' if (0.07 < y <= 2.3) else 'high'

def create_data_with_classes():
    '''
    Transformation of dataset adding new column with string labels for categories of each record

    Output: X and y with y being a collection of string labels
    '''
    X = np.array([])
    y = np.array([])

    #Features to use for the models
    feature_idcs = np.arange(len(FEATURES))
    target_feature_idx = FEATURES.index(TARGET_FEATURE)
    feature_idcs = np.delete(feature_idcs, target_feature_idx)

    #Getting municipality data from 2007 until 2018
    data = get_data(aarstal=[str(x) for x in range(2007, 2018+1)])
    data = fix_nans(data)

    #Create model data
    X = []
    y = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            X.append(data[i,j, feature_idcs].ravel())
            y.append(data[i,j, target_feature_idx].ravel())
    
    #Standardize
    X = np.array(X)
    y = np.array(y)
    mean_X = X.mean(axis=0)
    std_X = X.std(axis=0) + EPS
    X = (X - mean_X) / std_X
    mean_y = y.mean()
    print('Mean of y: {}'.format(mean_y))
    std_y = y.std() + EPS
    print('Standard deviation of y: {}\n'.format(std_y))
    y = (y - mean_y) / std_y

    #Transform intervals into labels for categorization
    class_y = np.array([to_risk_cat(target) for target in y])
    classes, count = np.unique(class_y, return_counts=True)
    print('\nClass count after transformation:')
    for i in range(len(classes)):
        print('{0} risk: {1}'.format(classes[i], count[i]))
    
    print('\n')

    return X , class_y