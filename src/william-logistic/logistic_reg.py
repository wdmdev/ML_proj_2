from sklearn.linear_model import LogisticRegression
from classification_dataset import create_data_with_classes as cdc
from ann import FEATURES
import numpy as np

def run():
    #Load standardized data
    X, y = cdc()

    mdl = LogisticRegression(penalty='l2', C=1.84)
    mdl.fit(X, y)
    largest = np.argmax(np.asarray(mdl.coef_[0]))
    print(largest)
    print(FEATURES[largest])


if __name__ == '__main__':
    run()