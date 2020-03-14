import os
from os.path import dirname, realpath
os.chdir(realpath(dirname(__file__)))

from glob import glob
from dataclasses import dataclass
from datetime import datetime
from timeit import default_timer
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression

#Needs sys for import to work on my laptop - William
import sys
sys.path.insert(1, './')
from dataclasses import dataclass
from data_extractor import get_data
from lr import lr_l2

np.random.seed(30)

CPU = torch.device("cpu")
GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = np.finfo(np.float32).eps

START_TIME = "-".join(str(datetime.now()).split(".")[0].split(":")).replace(" ", "_")
TARGET_FEATURE = "anmeldte tyverier/indbrud pr. 1.000 indb."
with open("attrs.out", encoding="utf-8") as f:
  FEATURES = f.read().split("\n")[2].split(";")

def log(*tolog):

  with open("ann_out/%s_log.txt" % START_TIME, "a", encoding="utf-8") as logfile:
    tolog = " ".join([str(x) for x in tolog])
    logfile.write(tolog+"\n")
    print(tolog)

@dataclass
class Data:
  x: np.ndarray
  y: np.ndarray
  mean_x: np.ndarray = None
  mean_y: np.float = None
  std_x: np.ndarray = None
  std_y: np.float = None

class Net(nn.Module):

  def __init__(self, in_features: int, hidden_layers: tuple, for_classification: bool):

    super().__init__()

    self.model = nn.Sequential()

    if for_classification:
      layers = (in_features, ) + hidden_layers + (3, )
    else:
      layers = (in_features, ) + hidden_layers + (1, )
    for i in range(len(layers)-1):
      self.model.add_module(
        "linear_" + str(i),
        nn.Linear(
          in_features=layers[i],
          out_features=layers[i+1]
        )
      )
      if i < len(layers) - 2:
        self.model.add_module(
          "relu_" + str(i),
          nn.ReLU()
        )
        self.model.add_module(
          "dropout_" + str(i),
          nn.Dropout(.4)
        )
        self.model.add_module(
          "batchnorm_" + str(i),
          nn.BatchNorm1d(layers[i+1])
        )
    if for_classification:
      self.model.add_module("sigmoid", nn.Sigmoid())
  
  def forward(self, x):
    
    return self.model(x)

class NetWrapper:

  def __init__(self, in_features, hidden_layers, lr=5e-4, momentum=0, weight_decay=1e-5, for_classification=False):

    self.net = Net(in_features, hidden_layers, for_classification).to(GPU)
    self.loss = nn.modules.MSELoss().to(GPU)
    self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    self.hasdata = False

    # log("Done initializing network")
    # log(self.net)
    # log("Loss function:", self.loss)
    # log("Optimizer:", self.optimizer)
    # if torch.cuda.is_available():
    #   log("Using CUDA\n")
    # else:
    #   log("Using CPU\n")
  
  def load_data(self, data: Data, split: tuple=(.8, 0, .2)):

    """
    x: input data, MxD numpy matrix, where M is the number of features (usually n_years * n_noegletal)
    y: targets, D long numpy vector
    split: the data is split into train, validation, and test according to the given proportions
    """

    self.mean_x = data.mean_x
    self.mean_y = data.mean_y
    self.std_x = data.std_x
    self.std_y = data.std_y

    try:
      assert sum(split) == 1
    except AssertionError:
      if sum(split) < 1:
        log("Warning: Split sums to %.4f but must sum to 1. Some data will not be used." % sum(split))
      else:
        raise ValueError("Split sums to %.4f but must sum to 1." % sum(split))

    # Splits data into train, validation, and test sets
    split_idcs = (
      int(data.y.size*split[0]),
      int(data.y.size*sum(split[:2])),
      int(data.y.size*sum(split[:3])),
    )
    
    self.train_x = torch.from_numpy(data.x.astype("float32")).to(GPU)
    self.train_y = torch.from_numpy(data.y.astype("float32")).to(GPU)

    # self.val_x = torch.from_numpy(x[split_idcs[0]:split_idcs[1]].astype("float32")).to(GPU)
    # self.val_y = torch.from_numpy(y[split_idcs[0]:split_idcs[1]].astype("float32")).to(GPU)

    # self.test_x = torch.from_numpy(data.x[split_idcs[1]:split_idcs[2]].astype("float32")).to(GPU)
    # self.test_y = torch.from_numpy(data.y[split_idcs[1]:split_idcs[2]].astype("float32")).to(GPU)

    # self.rest_x = torch.from_numpy(x[split_idcs[2]:].astype("float32")).to(GPU)
    # self.rest_y = torch.from_numpy(y[split_idcs[2]:].astype("float32")).to(GPU)

    self.hasdata = True

    # log("Finished loading data")
    # log("Train, validation, and test split:", ", ".join(["%i%%" % (x*100) for x in split]))
    # log("train_x:", self.train_x.size())
    # log("train_y:", self.train_y.size())
    # # log("val_x:", self.val_x.size(), self.val_y.dtype)
    # # log("val_y:", self.val_y.size(), self.val_y.dtype)
    # log("test_x:", self.test_x.size())
    # log("test_y:", self.test_y.size())
    log()
  
  def train(self, epochs=3000):

    if not self.hasdata:
      raise Exception("Error: Data has not been loaded")

    train_losses = []
    val_losses = []

    # Trains the network
    # log(("Training network over %i epochs on " + ("CUDA" if torch.cuda.is_available() else "cpu")) % epochs)
    train_start = default_timer()
    for i in range(epochs):
      # Network validation
      # with torch.no_grad():
      #   output = self._forward(self.val_x)
      #   loss = self.loss(output.squeeze(), self.val_y)
      #   val_losses.append(loss.detach().cpu())

      # Network training
      self.net.train()
      output = self._forward(self.train_x)
      loss = self.loss(output.squeeze(), self.train_y)
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()
      train_losses.append(loss.detach().cpu())

      if i % (epochs // 10) == 0:
        new_time = default_timer()
        try:
          elapsed_time = new_time - prev_time
        except NameError:
          elapsed_time = 0
        # log("Epoch %i (%.2fs): Train loss: %.6f" % (
        #   i,
        #   elapsed_time,
        #   train_losses[-1],
        #   val_losses[-1]
        # ))
        prev_time = default_timer()
    log("Total training time: %.4f s\n" % (default_timer() - train_start))
    
    return train_losses, val_losses
  
  def _forward(self, x):
    return self.net(x)
  
  def train_plot(self, train_losses, val_losses, path):

    plt.figure(figsize=(19.2, 10.8))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.title("Loss over time")
    plt.xlabel("Epochs")
    plt.ylabel(str(self.loss))
    # plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    # plt.show()
    plt.clf()
  
  def test(self, test: Data):

    try:
      test.x = torch.from_numpy(test.x.astype("float32")).to(GPU)
      test.y = torch.from_numpy(test.y.astype("float32")).to(GPU)
    except:
      pass

    # Evaluates the network using test data
    with torch.no_grad():
      output = self._forward(test.x).squeeze()
      loss = self.loss(output, test.y)
      log("%s evaluated on %i test data points: %.6f" % (self.loss, len(test.y), loss))
      log("Specific cases")
      # for i in range(min(max_test_cases, len(self.rest_x))):
      #   out = self._forward(self.rest_x[i:i+2])[0] * self.std_y + self.mean_y
      #   target = self.rest_y[i] * self.std_y + self.mean_y
      #   log("MUNICIPALITY YEAR X - Predicted: %.4f, actual: %.4f" % (out, target))
      log()
    
    return output.cpu().numpy(),\
           test.y.cpu().numpy(),\
           loss
  
  @staticmethod
  def test_plot(targets, outputs, n_years, path):

      plt.figure(figsize=(19.2, 10.8))
      x = np.linspace(0, max(targets))
      plt.plot(x, x)
      plt.scatter(targets, outputs)
      plt.title("Network Evaluation\nEmpirical Correlation: %.4f" % np.corrcoef(targets, outputs)[1, 0])
      plt.xlabel("Reported number of thefts and burglaries after %s years" % str(n_years))
      plt.ylabel("Predicted")
      plt.savefig(path)
      # plt.show()
      plt.clf()
  
  @staticmethod
  def corr_plot(json_path):

    with open(json_path) as f:
      losses = json.load(f)
    
    x = np.arange(1, len(losses["layers"].keys())+1)
    y = [losses["layers"][x] for x in losses["layers"]]
    plt.plot(x, y)
    plt.title("Layer configurations")
    plt.xticks(x, losses["layers"].keys())
    plt.xlabel("Hidden layers")
    plt.ylabel("MSE loss")
    plt.grid(True)
    plt.savefig("ann_out/%s_loss_layers.png" % START_TIME)
    plt.clf()

    x = np.arange(1, len(losses["weight_decay"].keys())+1)
    y = [losses["weight_decay"][x] for x in losses["weight_decay"]]
    # plt.xscale("log")
    plt.plot(x, y)
    plt.title("L2 Regularization")
    plt.xticks(x, losses["weight_decay"].keys())
    plt.xlabel("L2 regularization parameter")
    plt.ylabel("MSE loss")
    plt.grid(True)
    plt.savefig("ann_out/%s_loss_wd.png" % START_TIME)
    plt.clf()


def fix_nans(data: np.ndarray):

  # Loops over municipalities
  for i in range(data.shape[0]):
    # Loops over features
    for j in range(data.shape[2]):
      dat = data[i, :, j]
      nan_idcs = np.isnan(dat)
      if nan_idcs.all():
        # If all values are nans, they are set to 0
        data[i, :, j] = 0
        continue
      elif (~nan_idcs).all():
        continue
      # Sets the first and last values to the first and last nans
      val_idcs = np.where(~np.isnan(dat))[0]
      dat[0] = dat[val_idcs[0]]
      dat[-1] = dat[val_idcs[-1]]
      nan_idcs = np.isnan(dat)
      val_idcs = np.where(~np.isnan(dat))[0]
      # Loops over years where the observation is not nan
      # The magic happens here
      for k in range(val_idcs.size-1):
        x0, x1 = val_idcs[k], val_idcs[k+1]
        y0, y1 = dat[val_idcs[k]], dat[val_idcs[k+1]]
        a = (y1 - y0) / (x1 - x0)
        b = y1 - a * x1
        x = np.arange(x0+1, x1)
        y = a * x + b
        dat[x0+1:x1] = y
      data[i, :, j] = dat
  
  return data

def to_risk_cat(y):
    '''
    Transforms the standardized(subtracted mean and divided std)
    y data point into string labels for classification based on the intervals:
    Low:    [-2.2,1.5]
    Medium: ]1.5,3.1]
    High:   ]3.1,4.6] 
    '''
    return 'low' if (y <= 1.5) else 'medium' if (1.5 < y <= 3.1) else 'high'

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

    return Data(X, class_y, mean_X, mean_y, std_X, std_y)

def create_dataset(feature_years=4, predict_years=5, start_year=2007, end_year=2018):

  """
  Creates a dataset with the following properties
  x is a data matrix of size nxm where m is the number of features: feature_years * n_features
  n is the number of data vectors: n_municipalities * (n_years-feature_years+1)
  y is a vector of length n with the reported number of crimes predict_years after the end of a data vector
  """

  target_feature_idx = FEATURES.index(TARGET_FEATURE)
  feature_idcs = np.arange(len(FEATURES))
  # Comment out this line to include previous crime rates in features
  # feature_idcs = feature_idcs[feature_idcs!=target_feature_idx]

  data = get_data(aarstal=[str(x) for x in range(start_year, end_year+1)])
  log("Estimating nans")
  data = fix_nans(data)
  x_total = data[:, :-predict_years, feature_idcs]
  y_total = data[:, predict_years:, target_feature_idx]

  x_new = []
  y_new = []
  # Loops through all years that are at the start of an observation
  for i in range(x_total.shape[1]-feature_years+1):
    # Loops through all municipalities
    for j in range(x_total.shape[0]):
      x_new.append(x_total[j, i:i+feature_years, :].ravel())
      y_new.append(y_total[j, i+feature_years-1])
  
  x = np.array(x_new)
  mean_x = x.mean(axis=0)
  std_x = x.std(axis=0) + EPS
  x = (x - mean_x) / std_x
  y = np.array(y_new)
  mean_y = y.mean()
  std_y = y.std() + EPS
  y = (y - mean_y) / std_y
  
  return Data(x, y, mean_x, mean_y, std_x, std_y)


def kfold_cv(data: Data, k: int):

  log("Running %i-fold cross validation for linear regression" % k)
  lambdas = np.logspace(-1, 2.5, 30)
  #lambdas = np.logspace(-3, 3, 30)
  nobs_per_batch = data.x.shape[0] // k
  train_losses = np.empty((k, len(lambdas)))
  losses = np.empty((k, len(lambdas)))
  for i in range(k):
    idcs = np.zeros(data.x.shape[0], dtype=np.bool)
    idcs[i*nobs_per_batch:(i+1)*nobs_per_batch] = 1
    inner_train = Data(
      data.x[idcs],
      data.y[idcs],
    )
    inner_test = Data(
      data.x[~idcs],
      data.y[~idcs],
    )
    
    model_train_losses = np.empty(len(lambdas))
    model_losses = np.empty(len(lambdas))
    for k1, cfg in enumerate(lambdas):
      w = lr_l2(inner_train.x, inner_train.y, cfg)
      yhat = inner_test.x @ w
      # lm = LinearRegression(fit_intercept=False)
      # lm.fit(inner_train.x, inner_train.y)
      # yhat = lm.predict(inner_test.x)
      # print(inner_test.y.mean(), yhat.mean(), w)
      train_loss = ((inner_train.y - inner_train.x @ w)**2).sum() / inner_train.y.size
      model_train_losses[k1] = train_loss
      loss = ((inner_test.y - yhat)**2).sum() / inner_test.y.size
      model_losses[k1] = loss
    train_losses[i] = model_train_losses
    losses[i] = model_losses
    
  losses = losses.mean(axis=0)
  train_losses = train_losses.mean(axis=0)

  # Bruges til del a
  plt.figure(figsize=(15, 10))
  plt.subplot(121)
  plt.scatter(lambdas, losses)
  plt.plot(lambdas, losses)
  plt.scatter(lambdas,train_losses)
  plt.plot(lambdas, train_losses)
  plt.subplots_adjust(left=None, bottom=None, right= None, top=None, wspace= 0.4, hspace=None)
  plt.title("Loss as function of regularization",fontsize = 14)
  plt.legend(["Test","Train"], fontsize = 14)
  plt.ylabel("Mean squared error",fontsize = 14)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel("Lambda",fontsize = 14)
  plt.xscale("log")
  plt.yscale("log")
  plt.grid(True)


  plt.subplot(122)
  plt.scatter(lambdas,losses)
  plt.plot(lambdas, losses)
  plt.title("Optimal lambda: %s " % np.argmin(losses))
  plt.legend(["Test"],fontsize = 14)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.ylabel("Mean squared error",fontsize = 14)
  plt.xlabel("Lambda",fontsize = 14)
  plt.xscale("log")
  plt.yscale("log")
  
  plt.grid(True)
  plt.show()

  print("Test error:",np.min(losses))

def twofold_cv(data: Data, k1: int, k2: int):

  log("Performing two-layer cross-validation")
  configs = (1, 2**3, 2**5, 2**7, 2**8, 2**9, 2**10)
  n_configs = len(configs)
  outer_nobs_per_batch = data.x.shape[0] // k1
  E_i_test = np.empty(k1)
  optimals = []
  all_losses = []
  for i in range(k1):
    log("Starting outer loop %i" % i)
    outer_idcs = np.ones(data.x.shape[0], dtype=np.bool)
    outer_idcs[i*outer_nobs_per_batch:(i+1)*outer_nobs_per_batch] = 0
    outer_train = Data(
      data.x[outer_idcs],
      data.y[outer_idcs],
    )
    outer_test = Data(
      data.x[~outer_idcs],
      data.y[~outer_idcs],
    )
    inner_nobs_per_batch = outer_train.x.shape[0] // k2
    inner_losses = np.empty((k2, n_configs))
    for j in range(k2):
      log("Starting inner loop %i" % j)
      model_losses = np.empty(n_configs)
      inner_idcs = np.ones(outer_train.x.shape[0], dtype=np.bool)
      inner_idcs[j*inner_nobs_per_batch:(j+1)*inner_nobs_per_batch] = 0
      inner_train = Data(
        outer_train.x[inner_idcs],
        outer_train.y[inner_idcs],
      )
      inner_test = Data(
        outer_train.x[~inner_idcs],
        outer_train.y[~inner_idcs],
      )
      log(inner_train.x.shape, inner_test.x.shape)
      for k, cfg in enumerate(configs):
        # w = lr_l2(inner_train.x, inner_train.y, cfg)
        # yhat = inner_test.x @ w
        # loss = ((inner_test.y - yhat)**2).sum() / inner_test.y.size
        net = NetWrapper(inner_train.x.shape[1], (cfg,))
        net.load_data(inner_train)
        net.train()
        _, _, loss = net.test(inner_test)
        model_losses[k] = loss
      # mean = inner_train.y.mean()
      # model_losses = (float(((mean-inner_test.y)**2).sum()) / inner_test.y.size,)*n_configs
      inner_losses[j] = model_losses
    inner_losses_save = inner_losses.copy()
    inner_losses = inner_losses.mean(axis=0)
    D_j_val = (~outer_idcs).sum()
    D_i_par = outer_idcs.sum()
    E_s_gen = D_j_val / D_i_par * inner_losses
    optimal = configs[np.argmin(E_s_gen)]
    optimals.append(optimal)
    # w = lr_l2(outer_train.x, outer_train.y, optimal)
    # yhat = outer_test.x @ w
    # loss = ((outer_test.y - yhat)**2).sum() / outer_test.y.size
    mean = outer_train.y.mean()
    loss = float(((mean-outer_test.y)**2).sum()) / outer_test.y.size
    [all_losses.append(x) for x in inner_losses_save[:, np.argmin(E_s_gen)]]
    net = NetWrapper(outer_train.x.shape[1], (optimal,))
    net.load_data(outer_train)
    net.train()
    _, _, loss = net.test(outer_test)
    E_i_test[i] = loss
  D_i_test = (~outer_idcs).sum()
  N = data.x.shape[0]
  E_gen = np.sum(D_i_test / N * E_i_test)
  
  return E_gen, E_i_test, optimals, all_losses

def run():

  log("Starting network training")
  log("Time: %s" % START_TIME)

  # Gets indices of features
  target_feature = "anmeldte tyverier/indbrud pr. 1.000 indb."
  with open("attrs.out", encoding="utf-8") as f:
    features = f.read().split("\n")[2].split(";")
  target_feature_idx = features.index(target_feature)
  feature_idcs = np.arange(len(features))
  feature_idcs = feature_idcs[feature_idcs!=target_feature_idx]

  # Creates dataset
  feature_years = 4
  predict_years = 5
  log(
    "Attemptning to predict reported thefts and burglaries %s years in the future using %s years of data\n"
    % (predict_years, feature_years)
  )
  start_year, end_year = 2007, 2018
  data = create_data_with_classes()
  log("Done creating dataset (observations, features)")
  log("x:", data.x.shape)
  log("y:", data.y.shape)
  log()

  kfold_cv(data, 10)
  # E_gen, E_i_test, optimals, all_losses = twofold_cv(data, 10, 10)
    # log("Training with hidden layers %s" % (layers,))
    # net = NetWrapper(x.shape[1], layers, lr=1e-3, momentum=0, weight_decay=1e-6)
    # net.load_data(data, split=(.8, .1, .1))
    # train_losses, val_losses = net.train(epochs=int(1e4))
    # torch.save(net.net.state_dict(), "ann_out/%s_%s_model.pt" % (START_TIME, layers))
    # test_outputs, test_y, loss = net.test()
    # net.train_plot(train_losses, val_losses, "ann_out/%s_%s_test" % (START_TIME, layers))
    # net.test_plot(test_outputs, test_y, predict_years, "ann_out/%s_%s_test" % (START_TIME, layers))
    # losses["layers"][str(layers)] = float(loss)
  # means = [float(x) for x in means]
  # log("Done performing two-layer cross-validation.\n%s\n%s\n%s" % (E_gen, E_i_test, optimals))

  # with open("ann_out/%s_lr_tl-cv.json" % START_TIME, "w", encoding="utf-8") as f:
  #   json.dump({
  #     "E_gen": E_gen,
  #     "E_i_test": [float(x) for x in E_i_test],
  #     "optimal": optimals,
  #     "all_losses": all_losses
  #   }, f, indent=4)
  
  # net.corr_plot("ann_out/%s_corrs.json" % START_TIME)

if __name__ == "__main__":

  run()
