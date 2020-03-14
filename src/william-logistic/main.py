from classification import test_models
from McNemar import McNemar

if __name__ == '__main__':
    #Names for npy files
    model_names = ['logistic', 'tree', 'baseline']

    #Test models against each other using nested K-fold cross validation
    #test_models(model_names)

    #Comparing using McNemar
    #for m in model_names[1:]:
    #    McNemar(model_names[0], m)
    McNemar('tree', 'baseline')