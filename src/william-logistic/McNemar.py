from classification import logistic_reg, base_line, _load_npy
from classification_dataset import create_data_with_classes as cdc
from scipy.stats import beta, binom
import sys
import numpy as np

def McNemar(model_name_A, model_name_B):
    '''
    Evaluates model A and B using McNemar

    Input:
    :param model_name_A: First model to evaluate
    :param model_name_B: Second model to evaluate

    Output: The model that is deemed best from the McNemar evaluation
    '''
    #Load predictions from CV
    cA = _load_npy(model_name_A+'_predictions')
    cB = _load_npy(model_name_B+'_predictions')

    #Compute matched pair matrix
    n = len(cA)
    n12 = np.sum(cA*(1-cB))
    n21 = np.sum((1-cA)*cB)
    print('n12: {}'.format(n12))
    print('n21: {}\n'.format(n21))

    #Check n12 + n21 needs to be >= 5
    if n12 + n21 < 5:
        print('Cannot perform McNemar with n12 + n21 = {0} + {1} < 5'.format(n12, n21))
        return None

    #Estimate difference in accuracy
    theta = (n12-n21)/n
    print('theta: {}'.format(theta))
    Q = ((n**2)*(n+1)*(theta+1)*(1-theta))/(n*(n12+n21)-(n12-n21)**2)
    print('Q: {}\n'.format(Q))

    #Approximate confidence interval
    p = ((theta+1)/2)*(Q-1)
    q = ((1-theta)/2)*(Q-1)
    thetaL = 2*beta.ppf(0.95/2,p,q)-1
    thetaU = 2*beta.ppf(1-0.95/2,p,q)-1

    print('p: {}'.format(p))
    print('q: {}\n'.format(q))
    print('Confidence Interval for theta (difference in model A and B)')
    print('Lower: {0} , Upper:{1}\n'.format(thetaL, thetaU))

    #Calculate p-value
    p_val = 2*binom.cdf(np.min([n12, n21]),n12+n21, 1/2)
    print('Testing hypothesis, H0: No difference in A and B performance')
    print('p-value: {}'.format(p_val))


if __name__ == '__main__':
    McNemar('logistic', 'baseline')