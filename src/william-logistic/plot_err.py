import matplotlib.pyplot as plt
import numpy as np


def plot_classification_err(train_error_rates, test_error_rates, lambda_interval):
    font_size = 24
    plt.rcParams.update({'font.size': font_size})
    
    #Classification error against lambda
    logi_opt_lambda = 0.0
    logi_min_error = 0.0
    plt.figure(figsize=(8, 8))
    for i in range(len(train_error_rates)):
        test_error_rate = np.asarray(test_error_rates[i])
        train_error_rate = np.asarray(train_error_rates[i])

        min_error = np.min(test_error_rate)
        opt_lambda_idx = np.argmin(test_error_rate)
        opt_lambda = lambda_interval[opt_lambda_idx]
        if i == 0: 
            logi_opt_lambda = opt_lambda
            logi_min_error = min_error

        plt.semilogx(lambda_interval, train_error_rate*100)
        plt.semilogx(lambda_interval, test_error_rate*100)
        plt.semilogx(opt_lambda, min_error*100, 'o')

    logi_test_min = str(np.round(logi_min_error*100,2)) + ' % at e$^{' + str(np.round(np.log10(logi_opt_lambda),2))+'}$'
    plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
    plt.ylabel('Error rate (%)')
    plt.title('Classification error')
    plt.legend(['Logistic Training error','Logistic Test error','Logistic Test minimum {}'.format(logi_test_min),
                    'Baseline Training error', 'Baseline Test error', 'Baseline Test Minimum'],loc='upper right')
    plt.ylim([0, 50])
    plt.grid()
    plt.show()    