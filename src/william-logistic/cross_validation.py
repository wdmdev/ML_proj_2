import numpy as np
import os

#Global variables
SEED = 30

def _save_npy(file, arr):
    '''
    Saves the given array to a npy file in the current working directory
    Needs src as working directory

    Input:
    :param file: Name of the npy file to save
    :param arr: The array to save in the npy file
    '''
    path = os.path.join(os.path.join(os.getcwd(), 'william-logistic/npys'),file)
    np.save(path, arr)

def _shuffle(D, seed=SEED):
    '''
    Shuffles the collection of data sets D using the given seed.
    The datasets are shuffled in place

    Input:
    :param D: Collection of datasets to shuffle
    :param seed: Seed to use for random shuffle for all the datasets D
    '''
    for d in D:
        np.random.seed(seed)
        np.random.shuffle(d)



def _split(X,y, K, seed=SEED):
    '''
    Splits data X and y into K chunks

    Input:
    :param X: The training features
    :param y: The target features
    :param K: The number of chunks
    '''
    #Shuffles X and y in-place with the same seed
    _shuffle([X,y])

    #Find the split size of the K chuncks
    split_size = int(np.floor(len(y)/K))
    remainder = len(y) % (K*split_size)
    #Split X and y
    X_splits = np.split(X[:len(X)-remainder], split_size)
    y_splits = np.split(y[:len(X)-remainder], split_size)

    #Find the remaining elements 
    if remainder > 0:
        X_remainder = X[-remainder:]
        y_remainder = y[-remainder:]
        for i in range(len(y_remainder)):
            X_splits[i] = np.append(X_splits[i],[X_remainder[i]], axis=0)
            y_splits[i] = np.append(y_splits[i],[y_remainder[i]])

    return X_splits, y_splits

def _train_test_split(X_splits, y_splits, i):
    '''
    Gets the X, y training and test data from collections of X, y splitted data
    
    Input:
    :param X_splits: The collection of X data splits
    :param y_spltis: The collection of y data splits
    :param i: The index of the test fold-chunk

    Output: X and y training data
    '''
    X_train = np.vstack(np.delete(X_splits, i, 0))
    X_test = X_splits[i]
    y_train = np.concatenate(np.delete(y_splits, i, 0))
    y_test = y_splits[i]

    return X_train, X_test, y_train, y_test


def _select_model(validation_errors, D_val_length, D_par_length):
    '''
    Selects the model with the lowest error E_gen

    Input:
    :param validation_errors: The validation errors of the model from inner CV
    :param D_val_length: The length of the y_test data used in the inner CV
    :param D_par_length: The length of the training data partition in the outer CV

    Output: The model with the lowest E_gen error
    '''
    E_gen = []
    for v in validation_errors:
        E_gen.append(np.sum(v*(D_val_length/D_par_length)))
    
    return np.argmin(E_gen)


def nested_cv(X, y, models, K_out, K_in, model_name, lambdas=None, seed=SEED):
    print('Cross Validation on {}'.format(model_name))
    '''
    Performs nested cross-validation, comparing the given models against the data X and y

    Input:
    :param X: The training feature values
    :param y: The target values
    :param models: The models to train and evaluate on the data X and y. One model is selected based on the CV
    :param K_out: Number of outer folds
    :param K_in: Number of inner folds
    :param seed: The seed number for the CV, to be able to compare statistically

    Output: The generalization of the model as calculated from the CV
    '''
    #Split into K outer folds
    X_splits, y_splits = _split(X, y, K_out, seed)

    #Outer validation errors
    out_validation_errors = np.zeros(K_out)
    #Lambdas for optimal models
    opt_lambdas = []
    #Outer test set length
    outer_D_val_lengths = np.zeros(K_out)
    #Predictions for optimal models
    opt_predictions = np.array([])

    #Go through the K outer folds
    for i in range(K_out):
        #Divide K outer folds into training and test set
        X_par, out_X_test, y_par, out_y_test = _train_test_split(X_splits, y_splits, i)
        outer_D_val_lengths[i] = len(out_y_test)

        #Inner split of each K outer fold training data
        in_X_splits, in_y_splits = _split(X_par, y_par, K_in, seed)

        #Model validation errors
        in_validation_errors = np.empty((len(models), K_in))
        #Length of inner validation sets
        inner_D_val_lengths = np.array([])

        #Go through the K inner folds
        for j in range(K_in):
            #Divide K inner folds into training and test set
            X_train, X_test, y_train, y_test = _train_test_split(in_X_splits, in_y_splits, j)
            inner_D_val_lengths = np.append(inner_D_val_lengths, len(y_test))

            #Run through models: train and eval
            for idx, m in enumerate(models):
                #Train model m
                train_model = m(X_train, y_train)
                #Predict on test data
                test_est = train_model(X_test)
                #Calculate and save validation error for each model
                in_validation_errors[idx, j] = np.sum(test_est != y_test) / len(y_test)
            
        #Select optimal inner model
        opt_idx = _select_model(in_validation_errors, inner_D_val_lengths, len(y_par))
        opt_inner_model = models[opt_idx]
        opt_lambdas.append(opt_idx)
        #Train optimal model on X_par
        opt_trained_model = opt_inner_model(X_par, y_par)
        #Test on outer fold X test data
        opt_test_est = opt_trained_model(out_X_test)
        #Calculate predictions for McNemar
        opt_predictions = np.append(opt_predictions, [int(o) for o in opt_test_est == out_y_test])
        #Calculate outer test error, Ei_test
        out_validation_errors[i] = np.sum(opt_test_est != out_y_test) / len(out_y_test)
        
    #Save predictions for McNemar
    _save_npy('{0}_predictions'.format(model_name), opt_predictions)
    #Save validation error for optimal models
    _save_npy(model_name, out_validation_errors)
    #Save lambdas for optimal models
    if lambdas is not None:
        _save_npy('{}_lambdas'.format(model_name), lambdas[opt_lambdas])


    #Compute generalization error
    E_gen = np.sum(out_validation_errors*(outer_D_val_lengths/len(y)))
    return E_gen