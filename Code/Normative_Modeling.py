from google.colab import drive
drive.mount('/content/drive', force_remount=True)

"""# **Install pcntoolkit**"""

#!pip install --upgrade numpy==1.21.2 #1.20.3

!pip install h5py

!pip install typing-extensions

!pip install wheel

!pip install pcntoolkit

"""# **Normative model**"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pcntoolkit.dataio.fileio as fileio
from pcntoolkit.util.utils import create_bspline_basis, compute_MSLL, create_design_matrix
from pcntoolkit.normative import estimate, predict, evaluate
from scipy.stats import norm
import statsmodels.api as sm
import scipy.stats as stats

def load_2d(filename):
    """
        - Loads a dataset using PCNtoolkit's file handling utilities.
        - Ensures the data is in a 2D NumPy array format.
        - If the data is 1D, it converts it into a column vector.
    """
    x = fileio.load(filename)
    if len(x.shape) == 1:
        x = x[:, np.newaxis]
    return x

def calibration_descriptives(x):
  """
      Computes statistical properties of a dataset x:
        - Skewness (asymmetry of the distribution).
        - Standard deviation of skewness.
        - Kurtosis (tailedness of the distribution).
        - Standard deviation of kurtosis.
        - Standard error of the mean.
        - Standard error of standard deviation.
        - Returns these values as a list.
  """
  n = np.shape(x)[0]
  m1 = np.mean(x)
  m2 = sum((x-m1)**2)
  m3 = sum((x-m1)**3)
  m4 = sum((x-m1)**4)
  s1 = np.std(x)
  skew = n*m3/(n-1)/(n-2)/s1**3
  sdskew = np.sqrt( 6*n*(n-1) / ((n-2)*(n+1)*(n+3)) )
  kurtosis = (n*(n+1)*m4 - 3*m2**2*(n-1)) / ((n-1)*(n-2)*(n-3)*s1**4)
  sdkurtosis = np.sqrt( 4*(n**2-1) * sdskew**2 / ((n-3)*(n+5)) )
  semean = np.sqrt(np.var(x)/n)
  sesd = s1/np.sqrt(2*(n-1))
  cd = [skew, sdskew, kurtosis, sdkurtosis, semean, sesd]
  return cd

def confidence_interval(s2,x,z,x_forward):
  """
      Calculates confidence intervals at specific points (x_forward).
      For each x_forward value, it:
        - Finds matching values in x.
        - Computes the mean variance (S_hat).
        - Uses the Z-score (z) to determine the confidence interval.
      Returns a matrix with confidence intervals.
  """
  CI=np.zeros((len(x_forward),4))
  for i,xdot in enumerate(x_forward):
    ci_inx=np.isin(x,xdot)
    S2=s2[ci_inx]
    S_hat=np.mean(S2,axis=0)
    n=S2.shape[0]
    CI[i,:]=z*np.power(S_hat/n,.5)
  return CI

def normative_model(X_train, y_train, out_dir, X_test=None, y_test=None, warp=None, cols_cov=['age','sex'], k_fold=None, alg='blr', nknots=5, p=3, opt='powell', cov_x='age',qq='yes', varcov=None, varcov_te=None, X_ad=None, y_ad=None):

  """
    Estimates a normative model using Bayesian Linear Regression (BLR) or Gaussian Process Regression (GPR).
    This function iterates through each region of interest in the brain, fitting a separate model for each.
    The model is then evaluated on an independent test set. Running the models iteratively allows flexibility,
    as it does not require the same subjects to have data available for every brain region.

    * Inputs:
        - X_train (str): Path to a .txt/.csv file containing covariates for the training set (e.g., age, sex, scanner info).
          - If different scanners were used, a separate binary column for each scanner must be included.
          - The file must have a header with column names.

        - y_train (str): Path to a .txt/.csv file containing brain features for the training set.
          - Format: N subjects × M brain regions.
          - The file must have a header with column names.

        - X_test (str, optional): Path to a .txt file with covariates for the test set.
          - Same format as X_train.
          - If None, cross-validation will be performed.

        - y_test (str, optional): Path to a .txt file containing brain features for the test set.
          - Format: N subjects × M brain regions.
          - The file must have a header with column names.
          - If None, cross-validation is used instead.

        - k_fold (int, optional): Number of cross-validation folds.

        - warp (str, optional): Nonlinear warping function for non-Gaussian data.
          - Options: 'WarpSinArcsinh', 'WarpBoxCox', 'WarpAffine', 'WarpCompose'.
          - Default is None for Gaussian data.

        - out_dir (str): Path where input data is located and output results will be stored.

        - cols_cov (list, optional): List of covariate names used in model estimation.
          - Do not include scanner/site columns.
          - Default: ['age', 'sex'].

        - alg (str): Algorithm used to estimate the normative model.
          - 'blr': Bayesian Linear Regression.
          - 'gpr': Gaussian Process Regression.

        - nknots (int, optional): Number of knot points for nonlinear basis expansion (default: 5, cubic B-spline).

        - p (int, optional): Order of the spline (default: 3, cubic).

        - opt (str, optional): Optimization algorithm used to estimate the model.
          - Options: 'powell' (default), 'l-bfgs-b'.

        - cov_x (str, optional): Covariate to be plotted on the x-axis for model visualization.

        - qq (str, optional): If set to "yes", Q-Q plots will be generated and saved.

        - varcov (str, optional): Specifies the training covariates used to model heteroscedasticity.

        - varcov_te (str, optional): Specifies the test covariates used to model heteroscedasticity.

        - X_ad (str, optional): Path to a .txt file with covariates for the adaptation set.
          - If different scanners were used, include a separate binary column for each scanner.

        - y_ad (str, optional): Path to a .txt file containing brain features for the adaptation set.
          - Format: N subjects × M brain regions.
          - The file must have a header with column names.

    * Outputs:
        A directory named 'Feature_models' containing a subdirectory for each brain region. Each subdirectory includes:

        - Models directory:
          - The trained model stored in pickle format along with metadata.

        - Plotting directory:
          - Visualizations of model fit, centiles, and upper/lower bound for centiles.

        - ev_metrics.txt:
          - Evaluation metrics comparing test set values to predicted values in the original (input) space.

        - *_predict.txt:
          - Evaluation metrics comparing test set values to predicted values in the transformed (Gaussian) space.
  """

  os.chdir(out_dir)

  # Select order of columns of covariate files to have the first column the x-axis covariate and in the second column 'sex'
  cols_order = [cov_x, 'sex']


  """
  Read and process the train dataset, test dataset (if provided), and adaptation dataset (if provided):
  - Validate the file format.
  - Read the file and ensure the correct column order.
  - Raise an error if the file format is invalid.
  """
  # Read train, test and adpatation sets. Order columns: first the variable to plot in the x axis and then sex and scanner/site variables
  if X_train.endswith(".txt") or X_train.endswith(".csv"):
    sep = '\s+' if X_train.endswith(".txt") else ','
    X_train = pd.read_csv(X_train, header=0, sep=sep)
    y_train = pd.read_csv(y_train, header=0, sep=sep)
    cols = cols_order + [col for col in X_train.columns if col not in cols_order]
    X_train = X_train[cols]
  else:
      raise ValueError("Invalid file format. Enter .txt or .csv files.")

  if X_test is not None:
    if X_test.endswith(".txt") or X_test.endswith(".csv"):
      sep = '\s+' if X_test.endswith(".txt") else ','
      X_test = pd.read_csv(X_test, header=0, sep=sep)
      y_test = pd.read_csv(y_test, header=0, sep=sep)
      cols = cols_order + [col for col in X_test.columns if col not in cols_order]
      X_test = X_test[cols]
    else:
      raise ValueError("Invalid file format. Enter .txt or .csv files.")

  if X_ad is not None:
    if X_ad.endswith(".txt") or X_ad.endswith(".csv"):
      sep = '\s+' if X_ad.endswith(".txt") else ','
      X_ad = pd.read_csv(X_ad, header=0, sep=sep)
      y_ad = pd.read_csv(y_ad, header=0, sep=sep)
      cols = cols_order + [col for col in X_ad.columns if col not in cols_order]
      X_ad = X_ad[cols]
    else:
      raise ValueError("Invalid file format. Enter .txt or .csv files.")


  """
  Extract scanner/site information:
  - Identify scanner columns in the training dataset.
  - Identify scanner columns in the test dataset (if available).
  - Identify scanner columns in the adaptation dataset (if available) and check consistency with test scanners.
  """
  # Create list with scanner names for the train set
  scanner_ids = [c for c in X_train.columns if c not in cols_cov]

  # Create list with scanner names for the test set
  if X_test is not None:
    scanner_ids_te = [c for c in X_test.columns if c not in cols_cov]

  # If there is an adaptation set, create list with scanner names
  if X_ad is not None:
    scanner_ids_ad = [c for c in X_ad.columns if c not in cols_cov]
    if not all(elem in scanner_ids_ad for elem in scanner_ids_te):
      print('Warning: some of the testing sites are not in the adaptation data')


  if alg == 'gpr':
    X_train_initial = X_train
    # Training set of covariables consists of age and sex
    X_train = X_train_initial.loc[:,[cov_x, 'sex']]
    # Training set of covariables only consists of age for calculation of forward model of centiles_both
    X_train_both = X_train_initial.loc[:,cov_x]
    X_train_both = X_train_both.to_frame()

    if len(cols_cov) > 2 or len(scanner_ids) != 0:
      # Training set of batch effects consists of the rest of covariables (euler, scanners...)
      batch_effects = X_train_initial.drop(columns=[cov_x, 'sex']) # Extract batch effects by dropping 'age' and 'sex'

      if len(scanner_ids) != 0: # Identify batch effect variables that are not scanner-related
        batch_vars_notscanners = list(set(batch_effects.columns) & set(cols_cov)) #get columns in batch_effects that are not scanners
        batch_effects_melt = pd.melt(batch_effects, id_vars=batch_vars_notscanners, value_vars=scanner_ids, var_name='scanner', ignore_index=False)
        batch_effects_melt = batch_effects_melt[batch_effects_melt.value != 0]
        batch_effects_melt = batch_effects_melt.drop(columns='value')
        batch_effects_melt['scanner'] = batch_effects_melt['scanner'].factorize()[0]
      else:
        batch_effects_melt = batch_effects
    # Training set of batch effects consists of the rest of covariables (sex, euler, scanners...) used for forward model of centiles_both
    batch_effects_both = X_train_initial.drop(columns=cov_x)

    if len(scanner_ids) != 0:
      batch_vars_notscanners_both = list(set(batch_effects_both.columns) & set(cols_cov)) # Get columns in batch_effects_both that are not scanners
      batch_effects_melt_both = pd.melt(batch_effects_both, id_vars=batch_vars_notscanners_both, value_vars=scanner_ids, var_name='scanner', ignore_index=False)
      batch_effects_melt_both = batch_effects_melt_both[batch_effects_melt_both.value != 0]
      batch_effects_melt_both = batch_effects_melt_both.drop(columns='value')
      batch_effects_melt_both['scanner'] = batch_effects_melt_both['scanner'].factorize()[0]
    else:
      batch_effects_melt_both = batch_effects_both

    # Test set
    if X_test is not None:
      X_test_initial = X_test
      # Training set of covariables only consists of age
      X_test = X_test_initial.loc[:, [cov_x, 'sex']]
      X_test_both = X_test_initial.loc[:,cov_x]
      X_test_both = X_test_both.to_frame()

      if len(cols_cov) > 2 or len(scanner_ids) != 0:
        # Training set of batch effects consists of the rest of covariables (euler, scanners...)
        batch_effects_te = X_test_initial.drop(columns=[cov_x, 'sex'])
        if len(scanner_ids) != 0:
          batch_vars_notscanners_te = list(set(batch_effects_te.columns) & set(cols_cov)) #get columns in batch_effects that are not scanners
          batch_effects_melt_te = pd.melt(batch_effects_te, id_vars=batch_vars_notscanners_te, value_vars=scanner_ids, var_name='scanner', ignore_index=False)
          batch_effects_melt_te = batch_effects_melt_te[batch_effects_melt_te.value != 0]
          batch_effects_melt_te = batch_effects_melt_te.drop(columns='value')
          batch_effects_melt_te['scanner'] = batch_effects_melt_te['scanner'].factorize()[0]
        else:
          batch_effects_melt_te = batch_effects_te


  # Create list of regions
  roi_ids=list()
  for c in y_train.columns:
    roi_ids.append(c)

  # Only get max and min "covariate to plot in the x-axis" of train and test subjects to estimate the model and visualize centiles
  min_train=X_train.min()[0]
  max_train=X_train.max()[0]
  if X_test is not None:
    min_test=X_test.min()[0]
    max_test=X_test.max()[0]

  if cov_x == "age":
    min_train = min_train - 5
    max_train = max_train + 5
    if X_test is not None:
      min_test = min_test - 5
      max_test = max_test + 5

  # Train set
  df = open(os.path.join(out_dir, 'roi_dir_names.txt'),'w')
  for c in y_train.columns:
    y_train[c].to_csv(os.path.join(out_dir, 'resp_tr_' + c + '.txt'), header=False, index=False)
    df.write(c)
    df.write('\n')
  df.close()
  X_train.to_csv(os.path.join(out_dir, 'cov_tr.txt'), sep = '\t', header=False, index = False)
  y_train.to_csv(os.path.join(out_dir, 'resp_tr.txt'), sep = '\t', header=False, index = False)
  if alg == 'gpr':
    if len(cols_cov) > 2 or len(scanner_ids) != 0:
      batch_effects_melt.to_csv(os.path.join(out_dir, 'batch_tr.txt'), sep = '\t', header=False, index = False)
    batch_effects_melt_both.to_csv(os.path.join(out_dir, 'batch_tr_both.txt'), sep = '\t', header=False, index = False)
    X_train_both.to_csv(os.path.join(out_dir, 'cov_tr_both.txt'), sep = '\t', header=False, index = False)

  # Test set
  if X_test is not None:
    for c in y_test.columns:
      y_test[c].to_csv(os.path.join(out_dir, 'resp_te_' + c + '.txt'), header=False, index=False)
    X_test.to_csv(os.path.join(out_dir, 'cov_te.txt'), sep = '\t', header=False, index = False)
    y_test.to_csv(os.path.join(out_dir, 'resp_te.txt'), sep = '\t', header=False, index = False)
    if alg == 'gpr':
      if len(cols_cov) > 2 or len(scanner_ids) != 0:
        batch_effects_melt_te.to_csv(os.path.join(out_dir, 'batch_te.txt'), sep = '\t', header=False, index = False)

  # Adaptation set
  if X_ad is not None:
    for c in y_ad.columns:
      y_ad[c].to_csv(os.path.join(out_dir, 'resp_ad_' + c + '.txt'), header=False, index=False)
    #X_ad.to_csv(os.path.join(out_dir, 'cov_ad.txt'), sep = '\t', header=False, index = False)
    y_ad.to_csv(os.path.join(out_dir, 'resp_ad.txt'), sep = '\t', header=False, index = False)

  # if there is no "Feature_models" folder, create the directory
  if not os.path.exists("Feature_models/"):
    os.mkdir("Feature_models/")
    print('Feature_models directory sucessfully created.')

  else:
    print('Warning. Feature_models directory was already created.')
    os.mkdir("Feature_models/")

  # If there are files with features and covariables for a region, create a folder inside "Feature_models" for that region
  for i in open(os.path.join(out_dir, 'roi_dir_names.txt')).read().splitlines():
    if os.path.exists("resp_tr_{}.txt".format(i)):
      os.chdir("Feature_models")
      os.mkdir(i)
      os.chdir("..")
      os.rename("resp_tr_{}.txt".format(i), "Feature_models/{}/resp_tr.txt".format(i))
      os.rename("cov_tr.txt", "Feature_models/{}/cov_tr.txt".format(i))
      X_train.to_csv(os.path.join(out_dir, 'cov_tr.txt'), sep = '\t', header=False, index = False)
      if X_test is not None:
        os.rename("resp_te_{}.txt".format(i), "Feature_models/{}/resp_te.txt".format(i))
        os.rename("cov_te.txt", "Feature_models/{}/cov_te.txt".format(i))
        X_test.to_csv(os.path.join(out_dir, 'cov_te.txt'), sep = '\t', header=False, index = False)
      if X_ad is not None:
        os.rename("resp_ad_{}.txt".format(i), "Feature_models/{}/resp_ad.txt".format(i))
      if alg == 'gpr':
        os.rename("cov_tr_both.txt", "Feature_models/{}/cov_tr_both.txt".format(i))
        X_train_both.to_csv(os.path.join(out_dir, 'cov_tr_both.txt'), sep = '\t', header=False, index = False)
        if len(cols_cov) > 2 or len(scanner_ids) != 0:
          os.rename("batch_tr.txt", "Feature_models/{}/batch_tr.txt".format(i))
          batch_effects_melt.to_csv(os.path.join(out_dir, 'batch_tr.txt'), sep = '\t', header=False, index = False)
          if X_test is not None:
            os.rename("batch_te.txt", "Feature_models/{}/batch_te.txt".format(i))
            batch_effects_melt_te.to_csv(os.path.join(out_dir, 'batch_te.txt'), sep = '\t', header=False, index = False)
        os.rename("batch_tr_both.txt", "Feature_models/{}/batch_tr_both.txt".format(i))
        batch_effects_melt_both.to_csv(os.path.join(out_dir, 'batch_tr_both.txt'), sep = '\t', header=False, index = False)

  # clean up files of features
  os.system("rm resp_*.txt")
  # clean up files of covariables
  os.system("rm cov_*.txt")

  if alg == 'gpr':
    os.system("rm batch_*.txt")
    # Create pandas dataframes with header names to save out the overall model evaluation metrics
    if X_test is None:
      gpr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'] + [f'BIC_fold{i+1}' for i in range(k_fold)])
    else:
      gpr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho', 'BIC'])

  out_dir = os.path.join(out_dir, 'Feature_models')

  col_rest = []
  if len(cols_cov) > 2:
    for col in X_train.columns:
      if col not in cols_order:
        if len(scanner_ids) != 0:
          if col not in scanner_ids:
            col_rest.append(col)
        else:
          col_rest.append(col)
    cols_cov_order = cols_order + col_rest
  else:
    cols_cov_order = cols_order


  if alg == 'blr':
    if len(scanner_ids) != 0:
      # Get scanner column (not binary) used to create the design matrices
      X_train_melted = pd.melt(X_train, id_vars=cols_cov_order, value_vars=scanner_ids, var_name='scanner', ignore_index=False)
      X_train_melt = X_train_melted[X_train_melted.value != 0].copy()
      X_train_melt.sort_index(inplace=True)
      if X_test is not None:
        X_test_melted = pd.melt(X_test, id_vars=cols_cov_order, value_vars=scanner_ids_te, var_name='scanner', ignore_index=False)
        X_test_melt = X_test_melted[X_test_melted.value != 0].copy()
        X_test_melt.sort_index(inplace=True)
      if X_ad is not None:
        X_ad_melted = pd.melt(X_ad, id_vars=cols_cov_order, value_vars=scanner_ids_ad, var_name='scanner', ignore_index=False)
        X_ad_melt = X_ad_melted[X_ad_melted.value != 0].copy()
        X_ad_melt.sort_index(inplace=True)

      for roi_num, roi in enumerate(roi_ids):
          print('Running feature', roi_num+1, roi)
          roi_dir = os.path.join(out_dir, roi)
          os.chdir(roi_dir)

          cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')
          cov_file_tr = os.path.join(roi_dir, 'cov_bspline_tr.txt')

          X_tr = create_design_matrix(X_train[cols_cov_order],
                                      site_ids = X_train_melt['scanner'],
                                      all_sites = scanner_ids,
                                      basis = 'bspline',
                                      xmin = min_train,
                                      xmax = max_train,
                                      nknots = nknots,
                                      p = p)
          np.savetxt(cov_file_tr, X_tr)
          if X_test is not None:
            X_te = create_design_matrix(X_test[cols_cov_order],
                                        site_ids = X_test_melt['scanner'],
                                        all_sites = scanner_ids,
                                        basis = 'bspline',
                                        xmin = min_test,
                                        xmax = max_test,
                                        nknots = nknots,
                                        p = p)
            np.savetxt(cov_file_te, X_te)

      # Create pandas dataframes with header names to save out the overall evaluation metrics
      if X_test is None:
        blr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'] + [f'BIC_fold{i+1}' for i in range(k_fold)])
      else:
        if warp is None:
          blr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho', 'BIC'])
        elif warp is not None:
          blr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho', 'BIC', 'NLL', 'Skew', 'Kurtosis'])

    else:
      print("Warning. No scanner covariables specified")
      for roi_num, roi in enumerate(roi_ids):
          print('Running feature', roi_num+1, roi)
          roi_dir = os.path.join(out_dir, roi)
          os.chdir(roi_dir)

          cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')
          cov_file_tr = os.path.join(roi_dir, 'cov_bspline_tr.txt')

          X_tr = create_design_matrix(X_train[cols_cov_order],
                                      basis = 'bspline',
                                      xmin = min_train,
                                      xmax = max_train,
                                      nknots = nknots,
                                      p=p)
          np.savetxt(cov_file_tr, X_tr)
          if X_test is not None:
            X_te = create_design_matrix(X_test[cols_cov_order],
                                        basis = 'bspline',
                                        xmin = min_test,
                                        xmax = max_test,
                                        nknots = nknots,
                                        p=p)
            np.savetxt(cov_file_te, X_te)
      # Create pandas dataframes with header names to save out the overall evaluation metrics
      if X_test is None:
        blr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho'] + [f'BIC_fold{i+1}' for i in range(k_fold)])
      else:
        if warp is None:
          blr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho', 'BIC'])
        elif warp is not None:
          blr_metrics = pd.DataFrame(columns = ['ROI', 'MSLL', 'EV', 'SMSE', 'RMSE', 'Rho', 'BIC', 'NLL', 'Skew', 'Kurtosis'])

    if warp is not None:
      warp_reparam = True
    else:
      warp_reparam = False

  ##-- Estimate normative model cv --## (for both blr and gpr algorithms)

  if X_test is None and y_test is None:
    # Cross-validation
    for roi in roi_ids:
        print('Running feature:', roi)
        roi_dir = os.path.join(out_dir, roi)
        os.chdir(roi_dir)

        if alg == 'blr':
          # configure the covariates to use.
          cov_file_tr = os.path.join(roi_dir, 'cov_bspline_tr.txt')

          # load train & test response files
          resp_file_tr = os.path.join(roi_dir, 'resp_tr.txt')

          # run a basic model
          yhat_te, s2_te, nm, Z, metrics_te = estimate(cov_file_tr,
                                                      resp_file_tr,
                                                      cvfolds=k_fold,
                                                      alg = 'blr',
                                                      optimizer = opt,
                                                      savemodel = True,
                                                      saveoutput = False,
                                                      standardize = False,
                                                      warp=warp,
                                                      warp_reparam=warp_reparam)
          # save metrics
          bic_values = np.array(metrics_te['BIC'][0], dtype=np.float64)
          blr_metrics.loc[len(blr_metrics)] = [roi, metrics_te['MSLL'][0], metrics_te['EXPV'][0], metrics_te['SMSE'][0], metrics_te['RMSE'][0], metrics_te['Rho'][0]] + bic_values.tolist()
          np.savetxt(os.path.join(roi_dir, 'Z_crossval.txt'), Z)
          blr_metrics.to_csv(os.path.join(out_dir, 'blr_metrics_estimate.txt'), sep = '\t', header=True, index = False)

        elif alg == 'gpr':
          plot_dir = os.path.join(roi_dir,'plotting')
          os.mkdir(plot_dir)
          os.chdir(roi_dir)

          # configure the covariates to use.
          cov_file_tr = os.path.join(roi_dir, 'cov_tr.txt')
          cov_file_tr_both = os.path.join(roi_dir, 'cov_tr_both.txt')

          # load train & test response files
          resp_file_tr = os.path.join(roi_dir, 'resp_tr.txt')

          # load batch effects file
          trbefile_both =  os.path.join(roi_dir, 'batch_tr_both.txt')

          # run a basic model
          if len(cols_cov) > 2 or len(scanner_ids) != 0:
            trbefile =  os.path.join(roi_dir, 'batch_tr.txt')
            if varcov is None:
              yhat_cv, s2_cv, nm, Z, metrics_cv = estimate(covfile=cov_file_tr,
                                                          respfile=resp_file_tr,
                                                          trbefile=trbefile,
                                                          cvfolds=k_fold,
                                                          alg = 'gpr',
                                                          optimizer=opt,
                                                          savemodel = True,
                                                          saveoutput = False,
                                                          standardize = False)
            elif varcov is not None:
              yhat_cv, s2_cv, nm, Z, metrics_cv = estimate(covfile=cov_file_tr,
                                                          respfile=resp_file_tr,
                                                          trbefile=trbefile,
                                                          cvfolds=k_fold,
                                                          alg = 'gpr',
                                                          optimizer=opt,
                                                          savemodel = True,
                                                          saveoutput = False,
                                                          standardize = False,
                                                          varcovfile=varcov)
          if len(cols_cov) <= 2 and len(scanner_ids) == 0:
            yhat_cv, s2_cv, nm, Z, metrics_cv = estimate(covfile=cov_file_tr,
                                                        respfile=resp_file_tr,
                                                        cvfolds=k_fold,
                                                        alg = 'gpr',
                                                        optimizer=opt,
                                                        savemodel = True,
                                                        saveoutput = False,
                                                        standardize = False)

          np.savetxt(os.path.join(roi_dir, 's2_cv.txt'), s2_cv)
          bic_values = np.array(metrics_cv['BIC'][0], dtype=np.float64)
          gpr_metrics.loc[len(gpr_metrics)] = [roi, metrics_cv['MSLL'][0], metrics_cv['EXPV'][0], metrics_cv['SMSE'][0], metrics_cv['RMSE'][0], metrics_cv['Rho'][0]] + bic_values.tolist()
          gpr_metrics.to_csv(os.path.join(out_dir, 'gpr_metrics_estimate.txt'), sep = '\t', header=True, index = False)

          if qq is not None:
            Z = Z[:, 0]
            sm.qqplot(Z, line = '45')
            plt.savefig(os.path.join(roi_dir, "QQ_theoretical_vs_sample"), bbox_inches='tight')
            plt.show()
            plt.close()

          a = np.linspace(min_train,max_train,20) # list of 20 equally spaced numbers from min to max age
          b = np.linspace(min_train,max_train,20)
          age_forw = np.concatenate((a, b)) # concatenate both lists one after the other

          sex_covariates=['male','female','both']

          # Creating plots for Females, Males and both
          for i,sex in enumerate(sex_covariates):
          # Find the index of the data exclusively for one sex. Female:2, Male: 1
            if sex=='male':
              clr='blue'
              inx=np.where(X_train.sex==i+1)[0]
              x = X_train.values[inx,0] # get males age
            elif sex=='female':
              clr='red'
              inx=np.where(X_train.sex==i+1)[0]
              x = X_train.values[inx,0] # get females age
            else:
              clr='green'
              x=X_train.values[:,0] # get subjects age
              i=0

            if sex != 'both':
              ones = np.repeat(1, 20)
              twos = np.repeat(2, 20)
              sex_forw = np.concatenate((ones, twos))
              sex_forw = sex_forw.tolist()
              covariate_forwardmodel = {cov_x: age_forw,
                                        'sex': sex_forw}
              covariate_forwardmodel = pd.DataFrame(data=covariate_forwardmodel)

              covariate_forwardmodel.to_csv('covariate_forwardmodel.txt', sep = ' ', header = False, index = False)

              #estimate forward model
              if len(cols_cov) > 2 or len(scanner_ids) != 0:
                if varcov is None:
                  estimate(covfile = cov_file_tr,
                          respfile = resp_file_tr,
                          trbefile = trbefile,
                          testcov = 'covariate_forwardmodel.txt',
                          cvfolds = None,
                          alg = 'gpr',
                          outputsuffix = '_forward')
                elif varcov is not None:
                  estimate(covfile = cov_file_tr,
                          respfile = resp_file_tr,
                          trbefile = trbefile,
                          testcov = 'covariate_forwardmodel.txt',
                          cvfolds = None,
                          alg = 'gpr',
                          varcovfile=varcov,
                          outputsuffix = '_forward')
              if len(cols_cov) <= 2 and len(scanner_ids) == 0:
                estimate(covfile = cov_file_tr,
                        respfile = resp_file_tr,
                        testcov = 'covariate_forwardmodel.txt',
                        cvfolds = None,
                        alg = 'gpr',
                        outputsuffix = '_forward')
            if sex == 'both':
              covariate_forwardmodel = {cov_x: age_forw}
              covariate_forwardmodel = pd.DataFrame(data=covariate_forwardmodel)

              covariate_forwardmodel.to_csv('covariate_forwardmodel.txt', sep = ' ', header = False, index = False)

              # estimate forward model both males and females
              if varcov is None:
                estimate(covfile = cov_file_tr_both,
                        respfile = resp_file_tr,
                        trbefile = trbefile_both,
                        testcov = 'covariate_forwardmodel.txt',
                        cvfolds = None,
                        alg = 'gpr',
                        outputsuffix = '_forward')
              elif varcov is not None:
                estimate(covfile = cov_file_tr_both,
                        respfile = resp_file_tr,
                        trbefile = trbefile_both,
                        testcov = 'covariate_forwardmodel.txt',
                        cvfolds = None,
                        alg = 'gpr',
                        varcovfile=varcov,
                        outputsuffix = '_forward')

            # forward model data
            forward_yhat = pd.read_csv('yhat_forward.txt', sep = ' ', header=None)
            yhat_forward=forward_yhat.values
            yhat_forward=yhat_forward[20*i:20*(i+1)]
            x_sorted = np.sort(x)
            n = len(x_sorted) // 19
            x_forward = [x_sorted[0]] + [x_sorted[i*n] for i in range(1, 19)] + [x_sorted[-1]] # forward age expand over real test age

            # actual data
            y = pd.read_csv(resp_file_tr, sep = ' ', header=None)
            if sex != 'both':
              y = y.values[inx]
            # confidence Interval yhat+ z *(std/n^.5)-->.95 % CI:z=1.96, 99% CI:z=2.58
            s2= pd.read_csv('s2_cv.txt', sep = ' ', header=None)
            if sex != 'both':
              s2=s2.values[inx]
            CI_95=confidence_interval(s2,x,1.96,x_forward)
            CI_99=confidence_interval(s2,x,2.58,x_forward)

            # Create a trayectory for each point
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(x_forward,yhat_forward[:,0], linewidth=2.5,c=clr, label='Normative trajectory')
            ax.plot(x_forward,CI_95[:,0]+yhat_forward[:,0], linewidth=2, linestyle='--',c=clr, label='95% confidence interval')
            ax.plot(x_forward,-CI_95[:,0]+yhat_forward[:,0], linewidth=2, linestyle='--',c=clr)
            ax.plot(x_forward,CI_99[:,0]+yhat_forward[:,0], linewidth=1.5, linestyle='--',c=clr, label='99% confidence interval')
            ax.plot(x_forward,-CI_99[:,0]+yhat_forward[:,0], linewidth=1.5, linestyle='--',c=clr)
            ax.fill_between(x_forward, CI_95[:,0]+yhat_forward[:,0], -CI_95[:,0]+yhat_forward[:,0], alpha = 0.1,color=clr)
            ax.fill_between(x_forward, CI_99[:,0]+yhat_forward[:,0], -CI_99[:,0]+yhat_forward[:,0], alpha = 0.1,color=clr)

            # save plotting files
            np.savetxt(os.path.join(plot_dir, 'dummy_'+cov_x+sex+'.txt'), x_forward)
            np.savetxt(os.path.join(plot_dir, 'fit_'+sex+'.txt'), yhat_forward[:,0])
            np.savetxt(os.path.join(plot_dir, 'CI_95l_'+sex+'.txt'), -CI_95[:,0]+yhat_forward[:,0])
            np.savetxt(os.path.join(plot_dir, 'CI_95u_'+sex+'.txt'), CI_95[:,0]+yhat_forward[:,0])
            np.savetxt(os.path.join(plot_dir, 'CI_99l_'+sex+'.txt'), -CI_99[:,0]+yhat_forward[:,0])
            np.savetxt(os.path.join(plot_dir, 'CI_99u_'+sex+'.txt'), CI_99[:,0]+yhat_forward[:,0])

            ax.scatter(x, y, s=5, color=clr, alpha = 0.3, label=roi)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('Normative trajectory of ' +roi+' in '+sex+' cohort')
            plt.savefig(os.path.join(roi_dir, 'centiles_' + str(sex)),  bbox_inches='tight')
            plt.show()
            plt.close()

            if qq is not None and sex == 'both':
              fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17, 4))
              for j, xaxisrange in enumerate(['all_range','<50%','>50%']):
                # x_axis_feature, feature
                sorted_y = np.column_stack((x, y))
                #order by x_axis_feature
                sort_index = np.argsort(sorted_y[:,0])
                y_sorted_array = sorted_y[sort_index]
                #total number of subjects for visualization
                subjs = y_sorted_array.shape[0]
                print("Running qq plot for feature", roi, "; Sex:", sex, "; X-axis range:", xaxisrange)
                if xaxisrange == '<50%':
                  #filter features for subjects <50 xaxisrange
                  y_sorted_array = y_sorted_array[y_sorted_array[:,0] < (min_train + max_train)/2]
                  subjs = y_sorted_array.shape[0]
                elif xaxisrange == '>50%':
                  #filter features for subjects >50 xaxisrange
                  y_sorted_array = y_sorted_array[y_sorted_array[:,0] > (min_train + max_train)/2]
                  subjs = y_sorted_array.shape[0]
                bounds = [[i] for i in np.arange(0, 1, 0.01)]
                for cent in bounds:
                  z = norm.ppf(cent)
                  pr_int=confidence_interval(s2,x,z,x_forward)
                  pr_int=pr_int[:,0]+yhat_forward[:,0]
                  # get indices of the percentile that matches the xcovar_feature of subjects
                  xcovar_indices = np.searchsorted(x_forward,y_sorted_array[:,0])
                  # Use the xcovar_indices to extract the corresponding pr_int values
                  pr_int_values = pr_int[xcovar_indices]
                  # Count the number of y_values that are less than the corresponding pr_int values
                  num_below = np.count_nonzero(y_sorted_array[:, 1] <= pr_int_values)

                  axs = axes[j]
                  axs.scatter(cent[0], num_below, s=4, c=clr)
                # Ideal line
                axs.plot([0, 1], [0, subjs], color='red', alpha=0.5)
                axs.set_xlabel('Quantile (q)')
                axs.set_ylabel('N(q)')
                axs.set_title(xaxisrange + " " + roi)
            plt.savefig(os.path.join(roi_dir, "QQ_plots_counting"), bbox_inches='tight')
            plt.show()
            plt.close()

          os.chdir(out_dir)

  ##-- Estimate normative model (train and test) --## (for both blr and gpr algorithms)

  else:
    if alg == 'blr':
      # Use training and test data to estimate the model
      # Loop through features
      for roi in roi_ids:
          print('Running feature:', roi)
          roi_dir = os.path.join(out_dir, roi)
          os.chdir(roi_dir)

          # configure the covariates to use
          cov_file_tr = os.path.join(roi_dir, 'cov_bspline_tr.txt')
          cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')

          # load train & test response files
          resp_file_tr = os.path.join(roi_dir, 'resp_tr.txt')
          resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

          # run a basic model
          if varcov is not None:
            yhat_te, s2_te, nm, Z, metrics_te = estimate(cov_file_tr,
                                                        resp_file_tr,
                                                        testresp=resp_file_te,
                                                        testcov=cov_file_te,
                                                        alg = 'blr',
                                                        optimizer = opt,
                                                        savemodel = True,
                                                        saveoutput = False,
                                                        standardize = False,
                                                        warp=warp,
                                                        warp_reparam=warp_reparam,
                                                        varcovfile=varcov,
                                                        testvarcovfile=varcov_te)
          elif varcov is None:
            yhat_te, s2_te, nm, Z, metrics_te = estimate(cov_file_tr,
                                                        resp_file_tr,
                                                        testresp=resp_file_te,
                                                        testcov=cov_file_te,
                                                        alg = 'blr',
                                                        optimizer = opt,
                                                        savemodel = True,
                                                        saveoutput = False,
                                                        standardize = False,
                                                        warp=warp,
                                                        warp_reparam=warp_reparam)

          # get metrics
          if warp == None:
            np.savetxt(os.path.join(roi_dir, 'Z_scores.txt'), Z)
            blr_metrics.loc[len(blr_metrics)] = [roi, metrics_te['MSLL'][0], metrics_te['EXPV'][0], metrics_te['SMSE'][0], metrics_te['RMSE'][0], metrics_te['Rho'][0], metrics_te['BIC'][0]]
          if warp is not None:
            y_tr = load_2d(os.path.join(roi_dir, 'resp_tr.txt')) #features train
            # load the normative model
            with open(os.path.join(roi_dir,'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
                nm = pickle.load(handle)
            [skew, sdskew, kurtosis, sdkurtosis, semean, sesd] = calibration_descriptives(Z)
            BIC = len(nm.blr.hyp) * np.log(y_tr.shape[0]) + 2 * nm.neg_log_lik

            with open('skew_predict.txt', 'w') as f:
              f.write(str(skew))
            with open('kurtosis_predict.txt', 'w') as f:
              f.write(str(kurtosis))
            with open('BIC_predict.txt', 'w') as f:
              f.write(str(BIC))
            blr_metrics.loc[len(blr_metrics)] = [roi, metrics_te['MSLL'][0], metrics_te['EXPV'][0], metrics_te['SMSE'][0], metrics_te['RMSE'][0], metrics_te['Rho'][0], metrics_te['BIC'][0], nm.neg_log_lik, skew, kurtosis]

    elif alg == 'gpr':
      # Use training and test data to estimate the model
      # Loop through ROIs
      for roi in roi_ids:
          print('Running feature:', roi)
          roi_dir = os.path.join(out_dir, roi)
          os.chdir(roi_dir)
          plot_dir = os.path.join(roi_dir,'plotting')
          os.mkdir(plot_dir)
          os.chdir(roi_dir)

          # configure the covariates to use.
          cov_file_tr = os.path.join(roi_dir, 'cov_tr.txt')
          cov_file_te = os.path.join(roi_dir, 'cov_te.txt')
          cov_file_tr_both = os.path.join(roi_dir, 'cov_tr_both.txt')

          # load train & test response files
          resp_file_tr = os.path.join(roi_dir, 'resp_tr.txt')
          resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

          # load batch effects file
          trbefile_both =  os.path.join(roi_dir, 'batch_tr_both.txt')

          # run a basic model
          if len(cols_cov) > 2 or len(scanner_ids) != 0:
            trbefile =  os.path.join(roi_dir, 'batch_tr.txt')
            tebefile =  os.path.join(roi_dir, 'batch_te.txt')
            if varcov is not None:
              yhat_te, s2_te, nm, Z, metrics_te = estimate(covfile=cov_file_tr,
                                                        respfile=resp_file_tr,
                                                        trbefile=trbefile,
                                                        testresp=resp_file_te,
                                                        testcov=cov_file_te,
                                                        tebefile=tebefile,
                                                        alg = 'gpr',
                                                        optimizer=opt,
                                                        savemodel = True,
                                                        saveoutput = False,
                                                        standardize = False,
                                                        varcovfile=varcov,
                                                        testvarcovfile=varcov_te)
            elif varcov is None:
              yhat_te, s2_te, nm, Z, metrics_te = estimate(covfile=cov_file_tr,
                                                        respfile=resp_file_tr,
                                                        trbefile=trbefile,
                                                        testresp=resp_file_te,
                                                        testcov=cov_file_te,
                                                        tebefile=tebefile,
                                                        alg = 'gpr',
                                                        optimizer=opt,
                                                        savemodel = True,
                                                        saveoutput = False,
                                                        standardize = False)
          if len(cols_cov) <= 2 and len(scanner_ids) == 0:
            yhat_te, s2_te, nm, Z, metrics_te = estimate(covfile=cov_file_tr,
                                                      respfile=resp_file_tr,
                                                      testresp=resp_file_te,
                                                      testcov=cov_file_te,
                                                      alg = 'gpr',
                                                      optimizer=opt,
                                                      savemodel = True,
                                                      saveoutput = False,
                                                      standardize = False)

          np.savetxt(os.path.join(roi_dir, 's2_te.txt'), s2_te)
          gpr_metrics.loc[len(gpr_metrics)] = [roi, metrics_te['MSLL'][0], metrics_te['EXPV'][0], metrics_te['SMSE'][0], metrics_te['RMSE'][0], metrics_te['Rho'][0], metrics_te['BIC'][0]]
          np.savetxt(os.path.join(roi_dir, 'Z_scores.txt'), Z)
          gpr_metrics.to_csv(os.path.join(out_dir, 'gpr_metrics_estimate.txt'), sep = '\t', header=True, index = False)

          if qq is not None:
            Z = Z[:, 0]
            sm.qqplot(Z, line = '45')
            plt.savefig(os.path.join(roi_dir, "QQ_theoretical_vs_sample"), bbox_inches='tight')
            plt.show()
            plt.close()

          a = np.linspace(min_train,max_train,20) # list of 20 equally spaced numbers from min to max age
          b = np.linspace(min_train,max_train,20)
          age_forw = np.concatenate((a, b)) # concatenate both lists one after the other

          sex_covariates=['male','female','both']

          # Creating plots for Females, Males and both
          for i,sex in enumerate(sex_covariates):
          # Find the index of the data exclusively for one sex. Female:2, Male: 1
            if sex=='male':
              clr='blue'
              inx=np.where(X_test.sex==i+1)[0]
              x = X_test.values[inx,0] # get test males age
            elif sex=='female':
              clr='red'
              inx=np.where(X_test.sex==i+1)[0]
              x = X_test.values[inx,0] # get test females age
            else:
              clr='green'
              x=X_test.values[:,0] # get test subjects age
              i=0

            if sex != 'both':
              ones = np.repeat(1, 20)
              twos = np.repeat(2, 20)
              sex_forw = np.concatenate((ones, twos))
              sex_forw = sex_forw.tolist()
              covariate_forwardmodel = {cov_x: age_forw,
                                        'sex': sex_forw}
              covariate_forwardmodel = pd.DataFrame(data=covariate_forwardmodel)

              covariate_forwardmodel.to_csv('covariate_forwardmodel.txt', sep = ' ', header = False, index = False)

              #estimate forward model
              if len(cols_cov) > 2 or len(scanner_ids) != 0:
                if varcov is not None:
                  estimate(covfile = cov_file_tr,
                          respfile = resp_file_tr,
                          trbefile = trbefile,
                          testcov = 'covariate_forwardmodel.txt',
                          cvfolds = None,
                          alg = 'gpr',
                          varcovfile=varcov,
                          outputsuffix = '_forward')
                elif varcov is None:
                  estimate(covfile = cov_file_tr,
                          respfile = resp_file_tr,
                          trbefile = trbefile,
                          testcov = 'covariate_forwardmodel.txt',
                          cvfolds = None,
                          alg = 'gpr',
                          outputsuffix = '_forward')
              if len(cols_cov) <= 2 and len(scanner_ids) == 0:
                estimate(covfile = cov_file_tr,
                        respfile = resp_file_tr,
                        testcov = 'covariate_forwardmodel.txt',
                        cvfolds = None,
                        alg = 'gpr',
                        outputsuffix = '_forward')

            if sex == 'both':
              covariate_forwardmodel = {cov_x: age_forw}
              covariate_forwardmodel = pd.DataFrame(data=covariate_forwardmodel)

              covariate_forwardmodel.to_csv('covariate_forwardmodel.txt', sep = ' ', header = False, index = False)

              # estimate forward model
              if varcov is not None:
                estimate(covfile = cov_file_tr_both,
                        respfile = resp_file_tr,
                        trbefile = trbefile_both,
                        testcov = 'covariate_forwardmodel.txt',
                        cvfolds = None,
                        alg = 'gpr',
                        varcovfile=varcov,
                        outputsuffix = '_forward')
              elif varcov is None:
                estimate(covfile = cov_file_tr_both,
                        respfile = resp_file_tr,
                        trbefile = trbefile_both,
                        testcov = 'covariate_forwardmodel.txt',
                        cvfolds = None,
                        alg = 'gpr',
                        outputsuffix = '_forward')

            # forward model data
            forward_yhat = pd.read_csv('yhat_forward.txt', sep = ' ', header=None)
            yhat_forward=forward_yhat.values
            yhat_forward=yhat_forward[20*i:20*(i+1)]
            x_sorted = np.sort(x)
            n = len(x_sorted) // 19
            x_forward = [x_sorted[0]] + [x_sorted[i*n] for i in range(1, 19)] + [x_sorted[-1]] # forward age expand over real test age

            # actual data
            y = pd.read_csv(resp_file_te, sep = ' ', header=None)
            if sex != 'both':
              y = y.values[inx]
            # confidence Interval yhat+ z *(std/n^.5)-->.95 % CI:z=1.96, 99% CI:z=2.58
            s2= pd.read_csv('s2_te.txt', sep = ' ', header=None)
            if sex != 'both':
              s2=s2.values[inx]
            CI_95=confidence_interval(s2,x,1.96,x_forward)
            CI_99=confidence_interval(s2,x,2.58,x_forward)

            # Create a trayectory for each point
            fig=plt.figure()
            ax=fig.add_subplot(111)
            ax.plot(x_forward,yhat_forward[:,0], linewidth=2.5,c=clr, label='Normative trajectory')
            ax.plot(x_forward,CI_95[:,0]+yhat_forward[:,0], linewidth=2, linestyle='--',c=clr, label='95% confidence interval')
            ax.plot(x_forward,-CI_95[:,0]+yhat_forward[:,0], linewidth=2, linestyle='--',c=clr)
            ax.plot(x_forward,CI_99[:,0]+yhat_forward[:,0], linewidth=1.5, linestyle='--',c=clr, label='99% confidence interval')
            ax.plot(x_forward,-CI_99[:,0]+yhat_forward[:,0], linewidth=1.5, linestyle='--',c=clr)
            ax.fill_between(x_forward, CI_95[:,0]+yhat_forward[:,0], -CI_95[:,0]+yhat_forward[:,0], alpha = 0.1,color=clr)
            ax.fill_between(x_forward, CI_99[:,0]+yhat_forward[:,0], -CI_99[:,0]+yhat_forward[:,0], alpha = 0.1,color=clr)

            # save plotting files
            np.savetxt(os.path.join(plot_dir, 'dummy_'+cov_x+sex+'.txt'), x_forward)
            np.savetxt(os.path.join(plot_dir, 'fit_'+sex+'.txt'), yhat_forward[:,0])
            np.savetxt(os.path.join(plot_dir, 'CI_95l_'+sex+'.txt'), -CI_95[:,0]+yhat_forward[:,0])
            np.savetxt(os.path.join(plot_dir, 'CI_95u_'+sex+'.txt'), CI_95[:,0]+yhat_forward[:,0])
            np.savetxt(os.path.join(plot_dir, 'CI_99l_'+sex+'.txt'), -CI_99[:,0]+yhat_forward[:,0])
            np.savetxt(os.path.join(plot_dir, 'CI_99u_'+sex+'.txt'), CI_99[:,0]+yhat_forward[:,0])

            ax.scatter(x, y, s=5, color=clr, alpha = 0.3, label=roi)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('Normative trajectory of ' +roi+' in '+sex+' cohort')
            plt.savefig(os.path.join(roi_dir, 'centiles_' + str(sex)),  bbox_inches='tight')
            plt.show()
            plt.close()

            if qq is not None and sex == 'both':
              fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17, 4))
              for j, xaxisrange in enumerate(['all_range','<50%','>50%']):
                # x_axis_feature, feature
                sorted_y = np.column_stack((x, y)) # this is the test set, unless you use the train set as test
                #order by x_axis_feature
                sort_index = np.argsort(sorted_y[:,0])
                y_sorted_array = sorted_y[sort_index]
                #total number of subjects for visualization
                subjs = y_sorted_array.shape[0]
                print("Running qq plot for feature", roi, "; Sex:", sex, "; X-axis range:", xaxisrange)
                if xaxisrange == '<50%':
                  #filter features for subjects <50 xaxisrange
                  y_sorted_array = y_sorted_array[y_sorted_array[:,0] < (min_train + max_train)/2]
                  subjs = y_sorted_array.shape[0]
                elif xaxisrange == '>50%':
                  #filter features for subjects >50 xaxisrange
                  y_sorted_array = y_sorted_array[y_sorted_array[:,0] > (min_train + max_train)/2]
                  subjs = y_sorted_array.shape[0]
                bounds = [[i] for i in np.arange(0, 1, 0.01)]
                for cent in bounds:
                  z = norm.ppf(cent)
                  pr_int=confidence_interval(s2,x,z,x_forward)
                  pr_int=pr_int[:,0]+yhat_forward[:,0]
                  # get indices of the percentile that matches the xcovar_feature of subjects
                  xcovar_indices = np.searchsorted(x_forward,y_sorted_array[:,0])
                  # Use the xcovar_indices to extract the corresponding pr_int values
                  pr_int_values = pr_int[xcovar_indices]
                  # Count the number of y_values that are less than the corresponding pr_int values
                  num_below = np.count_nonzero(y_sorted_array[:, 1] <= pr_int_values)

                  axs = axes[j]
                  axs.scatter(cent[0], num_below, s=4, c=clr)
                # Ideal line
                axs.plot([0, 1], [0, subjs], color='red', alpha=0.5)
                axs.set_xlabel('Quantile (q)')
                axs.set_ylabel('N(q)')
                axs.set_title(xaxisrange + " " + roi)
            plt.savefig(os.path.join(roi_dir, "QQ_plots_counting"), bbox_inches='tight')
            plt.show()
            plt.close()

          os.chdir(out_dir)

  if alg == 'blr':
    if X_test is not None and y_test is not None:
      # Save overall test set evaluation metrics
      if warp is not None:
        blr_metrics['Skew'] = blr_metrics['Skew'].str[0].astype(float)
        blr_metrics['Kurtosis'] = blr_metrics['Kurtosis'].str[0].astype(float)
      blr_metrics['BIC'] = blr_metrics['BIC'].str[0].astype(float)
    blr_metrics.to_csv(os.path.join(out_dir, 'blr_metrics_estimate.txt'), sep = '\t', header=True, index = False)

    # Make predictions for each region separately and Z-statistics for the test dataset
    if X_test is not None:
      for roi_num, roi in enumerate(roi_ids):
        if len(scanner_ids) != 0:
          print('Running feature', roi_num+1, roi)
          roi_dir = os.path.join(out_dir, roi)
          os.chdir(roi_dir)

          resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

          # configure and save the design matrix
          cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')
          X_te = create_design_matrix(X_test_melt[cols_cov_order],
                                      site_ids = X_test_melt['scanner'],
                                      all_sites = scanner_ids,
                                      basis = 'bspline',
                                      xmin = min_test,
                                      xmax = max_test)
          np.savetxt(cov_file_te, X_te)

          # check whether all sites in the test set are represented in the training set
          if all(elem in scanner_ids for elem in scanner_ids_te):
              print('All sites/scanners in test data are present in the training data.')

              # just make predictions #predict only works with warped data
              if varcov_te is not None and warp is not None:
                yhat_te, s2_te, Z = predict(cov_file_te,
                                            alg='blr',
                                            respfile=resp_file_te,
                                            model_path=os.path.join(roi_dir, 'Models'),
                                            testvarcovfile=varcov_te,
                                            nm=nm)
              elif varcov_te is None and warp is not None:
                yhat_te, s2_te, Z = predict(cov_file_te,
                                            alg='blr',
                                            respfile=resp_file_te,
                                            model_path=os.path.join(roi_dir, 'Models'),
                                            nm=nm)
          else:
              print('Some sites/scanners missing from the training data. Adapting model')

              # Save the covariates for the adaptation data
              X_adap = create_design_matrix(X_ad[cols_cov_order],
                                          site_ids = X_ad_melt['scanner'],
                                          all_sites = scanner_ids,
                                          basis = 'bspline',
                                          xmin = min_train,
                                          xmax = max_train)
              cov_file_ad = os.path.join(roi_dir, 'cov_bspline_ad.txt')
              np.savetxt(cov_file_ad, X_adap)

              resp_file_ad = os.path.join(roi_dir, 'resp_ad.txt')

              # save the site numbers ids for the adaptation data
              site_num_ad = pd.factorize(X_ad_melt['scanner'])[0]
              sitenum_file_ad = os.path.join(roi_dir, 'sitenum_ad.txt')
              np.savetxt(sitenum_file_ad, site_num_ad)

              # save the site ids for the test data
              site_num_te = pd.factorize(X_test_melt['scanner'])[0]
              sitenum_file_te = os.path.join(roi_dir, 'sitenum_te.txt')
              np.savetxt(sitenum_file_te, site_num_te)

              if varcov_te is not None and warp is not None: # predict only works with warped data
                yhat_te, s2_te, Z = predict(cov_file_te,
                                            alg = 'blr',
                                            respfile = resp_file_te,
                                            model_path = os.path.join(roi_dir,'Models'),
                                            adaptrespfile = resp_file_ad,
                                            adaptcovfile = cov_file_ad,
                                            adaptvargroupfile = sitenum_file_ad,
                                            testvargroupfile = sitenum_file_te,
                                            testvarcovfile=varcov_te)
              elif varcov_te is None and warp is not None:
                yhat_te, s2_te, Z = predict(cov_file_te,
                                            alg = 'blr',
                                            respfile = resp_file_te,
                                            model_path = os.path.join(roi_dir,'Models'),
                                            adaptrespfile = resp_file_ad,
                                            adaptcovfile = cov_file_ad,
                                            adaptvargroupfile = sitenum_file_ad,
                                            testvargroupfile = sitenum_file_te)
        if len(scanner_ids) == 0:
          print('Running feature', roi_num+1, roi)
          roi_dir = os.path.join(out_dir, roi)
          os.chdir(roi_dir)

          resp_file_te = os.path.join(roi_dir, 'resp_te.txt')

          # configure and save the design matrix
          cov_file_te = os.path.join(roi_dir, 'cov_bspline_te.txt')
          # just make predictions #predict only works with warped data
          if varcov_te is not None and warp is not None:
            yhat_te, s2_te, Z = predict(cov_file_te,
                                        alg='blr',
                                        respfile=resp_file_te,
                                        model_path=os.path.join(roi_dir, 'Models'),
                                        testvarcovfile=varcov_te)
          elif varcov_te is None and warp is not None:
            yhat_te, s2_te, Z = predict(cov_file_te,
                                        alg='blr',
                                        respfile=resp_file_te,
                                        model_path=os.path.join(roi_dir, 'Models'))

    # Configure dummy data
    sx = [1,2,'both'] # 1 = male 2 = female

    for sex in sx:
      if sex == 1:
        sex = int(sex)
        clr = 'blue';
      elif sex == 2:
        clr = 'red'
        sex = int(sex)
      else:
        clr = 'green'

      # create dummy data for visualisation
      xx = np.linspace(min_train, max_train, 30)
      X0_dummy = np.zeros((len(xx), len(cols_cov)))
      X0_dummy[:,0] = xx
      if sex != 'both':
        X0_dummy[:,1] = sex

      # create the design matrix
      X_dummy = create_design_matrix(X0_dummy, xmin=min_train, xmax=max_train, site_ids=None, all_sites=scanner_ids, nknots=nknots, p=p)

      # save the dummy covariates
      cov_file_dummy = os.path.join(out_dir,'cov_bspline_dummy_mean.txt')
      np.savetxt(cov_file_dummy, X_dummy)

      if X_test is not None:
        sns.set(style='whitegrid')

        for roi_num, roi in enumerate(roi_ids):
          print('Running feature', roi_num, roi, ':')
          roi_dir = os.path.join(out_dir, roi)
          os.chdir(roi_dir)
          plot_dir = os.path.join(roi_dir,'plotting')
          if sex==1:
            os.mkdir(plot_dir)

          if warp is not None:
            # load the true data points (predicted from the test set)
            yhat_te = load_2d(os.path.join(roi_dir, 'yhat_predict.txt')) # predicted mean
            s2_te = load_2d(os.path.join(roi_dir, 'ys2_predict.txt')) # predicted variance

          y_te = load_2d(os.path.join(roi_dir, 'resp_te.txt')) #features test
          y_tr = load_2d(os.path.join(roi_dir, 'resp_tr.txt')) #features train

          # set up the covariates for the dummy data
          print('Making predictions with dummy covariates (for visualisation)')
          yhat, s2 = predict(cov_file_dummy,
                             alg = 'blr',
                             respfile = None,
                             model_path = os.path.join(roi_dir,'Models'),
                             outputsuffix = '_dummy')

          # load the normative model
          with open(os.path.join(roi_dir,'Models', 'NM_0_0_estimate.pkl'), 'rb') as handle:
              nm = pickle.load(handle)

          if warp is not None:
            # get the warp and warp parameters
            W = nm.blr.warp
            warp_param = nm.blr.hyp[1:nm.blr.warp.get_n_params()+1]

            # first, we warp predictions for the true data and compute evaluation metrics
            med_te = W.warp_predictions(np.squeeze(yhat_te), np.squeeze(s2_te), warp_param)[0]
            med_te = med_te[:, np.newaxis]
            metrics = evaluate(y_te, med_te)

            # ev_metrics: predictions in the input non-gaussian space vs. test data
            # *_predict: predictions in the warped gaussian space vs. test data

            # then, we warp dummy predictions to create the plots
            # med: median, pr_int: predictive interval
            med, pr_int = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param)
            Z = np.loadtxt(os.path.join(roi_dir, 'Z_predict.txt'))

          else:
            metrics = evaluate(y_te, yhat_te)
            med = yhat
            Z = np.loadtxt(os.path.join(roi_dir, 'Z_scores.txt'))

          if sex == 'both' and qq is not None:
            sm.qqplot(Z, line = '45')
            plt.savefig(os.path.join(roi_dir, "QQ_theoretical_vs_sample"), bbox_inches='tight')
            plt.show()
            plt.close()

          df = pd.DataFrame(list(metrics.items()),columns=['metric', 'value'])
          df.to_csv('ev_metrics.txt', index=False)

          # extract the different variance components to visualise
          beta, junk1, junk2 = nm.blr._parse_hyps(nm.blr.hyp, X_dummy)
          s2n = 1/beta # variation (aleatoric uncertainty)
          s2s = s2-s2n # modelling uncertainty (epistemic uncertainty)

          # plot the data points
          y_te_rescaled_list = np.zeros(len(y_te))
          if len(scanner_ids) != 0:
            for sid, site in enumerate(scanner_ids_te):
                # plot the true test data points
                if all(elem in scanner_ids for elem in scanner_ids_te):
                  # all data in the test set are present in the training set
                  # first, we select the data points belonging to this particular site
                  if sex != 'both':
                    idx = np.where(np.bitwise_and(X_te[:,2] == sex, X_te[:,sid+len(cols_cov)+1] !=0))[0]
                    if len(idx) == 0:
                      print('No data for site', sid, site, 'skipping...')
                      continue
                  else:
                    idx = np.where(X_te[:,sid+len(cols_cov)+1] !=0)[0]
                    if len(idx) == 0:
                      print('No data for scanner', sid, site, 'skipping...')
                      continue

                  # then directly adjust the data
                  idx_dummy = np.bitwise_and(X_dummy[:,1] > X_te[idx,1].min(), X_dummy[:,1] < X_te[idx,1].max())
                  y_te_rescaled = y_te[idx] - np.median(y_te[idx]) + np.median(med[idx_dummy])

                else:
                  # we need to adjust the data based on the adaptation dataset
                  # load the adaptation data
                  y_ad = load_2d(os.path.join(roi_dir, 'resp_ad.txt'))
                  X_ad = load_2d(os.path.join(roi_dir, 'cov_bspline_ad.txt'))
                  # first, select the data point belonging to this particular site
                  if sex != 'both':
                    idx = np.where(np.bitwise_and(X_te[:,2] == sex, (X_test_melt['scanner'] == site).to_numpy()))[0]
                    idx_a = np.where(np.bitwise_and(X_ad[:,2] == sex, (X_ad_melt['scanner'] == site).to_numpy()))[0]
                  else:
                    idx = np.where((X_test_melt['scanner'] == site).to_numpy())[0]
                    idx_a = np.where((X_ad_melt['scanner'] == site).to_numpy())[0]
                  if len(idx) < 2 or len(idx_a) < 2:
                      print('Insufficent data for site', sid, site, 'skipping...')
                      continue

                  # adjust and rescale the data
                  y_te_rescaled, s2_rescaled = nm.blr.predict_and_adjust(nm.blr.hyp,
                                                                        X_ad[idx_a,:],
                                                                        np.squeeze(y_ad[idx_a]),
                                                                        Xs=None,
                                                                        ys=np.squeeze(y_te[idx]))
                  # plot the (adjusted) data points
                  if warp is None:
                    idx_dummy = np.bitwise_and(X_dummy[:,1] > X_te[idx,1].min(), X_dummy[:,1] < X_te[idx,1].max())
                    y_te_rescaled = y_te[idx] - np.median(y_te[idx]) + np.median(med[idx_dummy])
                plt.scatter(X_te[idx,1], y_te_rescaled, s=4, color=clr, alpha = 0.1)
                y_te_rescaled = y_te_rescaled.ravel()
                y_te_rescaled_list[idx] = y_te_rescaled
          else:
            # no scanner/site covariables
            if sex != 'both':
              idx = np.where(X_te[:,2] == sex)[0]
              # then directly adjust the data
              idx_dummy = np.bitwise_and(X_dummy[:,1] > X_te[idx,1].min(), X_dummy[:,1] < X_te[idx,1].max())
              y_te_rescaled = y_te[idx] - np.median(y_te[idx]) + np.median(med[idx_dummy])

              plt.scatter(X_te[idx,1], y_te_rescaled, s=4, color=clr, alpha = 0.1)
            else:
              # adjust the data for test set
              idx_dummy = np.bitwise_and(X_dummy[:,1] > X_te[:,1].min(), X_dummy[:,1] < X_te[:,1].max())
              y_te_rescaled = y_te - np.median(y_te) + np.median(med[idx_dummy])

              #Visualize test data
              plt.scatter(X_te[:,1], y_te_rescaled, s=4, color=clr, alpha = 0.1)

              y_te_rescaled = y_te_rescaled.ravel()
              y_te_rescaled_list = y_te_rescaled

          if sex == 'both':
            np.savetxt(os.path.join(plot_dir, 'y_te_rescaled.txt'), y_te_rescaled_list)

          # plot the median of the dummy data
          plt.plot(xx, med, clr)
          if sex == 'both':
            np.savetxt(os.path.join(plot_dir, 'fit.txt'), med)
            np.savetxt(os.path.join(plot_dir, cov_x+'_dummy.txt'),xx)

          if warp is not None:
              # fill the gaps in between the centiles
              junk, pr_int25 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.25,0.75])
              junk, pr_int95 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.05,0.95])
              junk, pr_int99 = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[0.01,0.99])
              plt.fill_between(xx, pr_int25[:,0], pr_int25[:,1], alpha = 0.1,color=clr)
              plt.fill_between(xx, pr_int95[:,0], pr_int95[:,1], alpha = 0.1,color=clr)
              plt.fill_between(xx, pr_int99[:,0], pr_int99[:,1], alpha = 0.1,color=clr)

              # make the width of each centile proportional to the epistemic uncertainty
              junk, pr_int25l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.25,0.75])
              junk, pr_int95l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.05,0.95])
              junk, pr_int99l = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2-0.5*s2s), warp_param, percentiles=[0.01,0.99])
              junk, pr_int25u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.25,0.75])
              junk, pr_int95u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.05,0.95])
              junk, pr_int99u = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2+0.5*s2s), warp_param, percentiles=[0.01,0.99])
              plt.fill_between(xx, pr_int25l[:,0], pr_int25u[:,0], alpha = 0.3,color=clr)
              plt.fill_between(xx, pr_int95l[:,0], pr_int95u[:,0], alpha = 0.3,color=clr)
              plt.fill_between(xx, pr_int99l[:,0], pr_int99u[:,0], alpha = 0.3,color=clr)
              plt.fill_between(xx, pr_int25l[:,1], pr_int25u[:,1], alpha = 0.3,color=clr)
              plt.fill_between(xx, pr_int95l[:,1], pr_int95u[:,1], alpha = 0.3,color=clr)
              plt.fill_between(xx, pr_int99l[:,1], pr_int99u[:,1], alpha = 0.3,color=clr)
              if sex == 'both':
                np.savetxt(os.path.join(plot_dir, 'pr_int25l.txt'), pr_int25l)
                np.savetxt(os.path.join(plot_dir, 'pr_int25u.txt'), pr_int25u)
                np.savetxt(os.path.join(plot_dir, 'pr_int95l.txt'), pr_int95l)
                np.savetxt(os.path.join(plot_dir, 'pr_int95u.txt'), pr_int95u)
                np.savetxt(os.path.join(plot_dir, 'pr_int99l.txt'), pr_int99l)
                np.savetxt(os.path.join(plot_dir, 'pr_int99u.txt'), pr_int99u)
              # plot actual centile lines
              plt.plot(xx, pr_int25[:,0],color=clr, linewidth=0.5)
              plt.plot(xx, pr_int25[:,1],color=clr, linewidth=0.5)
              plt.plot(xx, pr_int95[:,0],color=clr, linewidth=0.5)
              plt.plot(xx, pr_int95[:,1],color=clr, linewidth=0.5)
              plt.plot(xx, pr_int99[:,0],color=clr, linewidth=0.5)
              plt.plot(xx, pr_int99[:,1],color=clr, linewidth=0.5)
              if sex == 'both':
                np.savetxt(os.path.join(plot_dir, 'pr_int25.txt'), pr_int25)
                np.savetxt(os.path.join(plot_dir, 'pr_int95.txt'), pr_int95)
                np.savetxt(os.path.join(plot_dir, 'pr_int99.txt'), pr_int99)

              plt.xlabel(cov_x)
              plt.ylabel(roi)
              plt.title(roi)
              if cov_x == 'age':
                if sex == 'both':
                  plt.xlim((X_test.min()[0]-2,X_test.max()[0]+2))
                if sex != 'both':
                  plt.xlim((X_test[X_test['sex'] == sex]['age'].min()-2,X_test[X_test['sex'] == sex]['age'].max()+2))
              else:
                if sex == 'both':
                  plt.xlim((X_test.min()[0],X_test.max()[0]))
                if sex != 'both':
                  plt.xlim((X_test[X_test['sex'] == sex][cov_x].min(),X_test[X_test['sex'] == sex][cov_x].max()))
              plt.savefig(os.path.join(roi_dir, 'centiles_' + str(sex)),  bbox_inches='tight')
              plt.show()
              plt.close()

          elif warp == None:
            # compute the lower and upper bounds of the 75%, 95% and 99% predictive intervals
            bounds=[[0.25,0.75],[0.05, 0.95],[0.01, 0.99]]
            for b in bounds:
                Z = norm.ppf(b)

                pred_interval = np.zeros((len(yhat), len(Z)))
                mu=yhat.ravel()
                s=s2.ravel()
                for i, z in enumerate(Z):
                    pred_interval[:,i] = mu + np.sqrt(s)*z

                pred_interval_lower = np.zeros((len(yhat), len(Z)))
                s_lower=s2-0.5*s2s
                s_lower=s_lower.ravel()
                for i, z in enumerate(Z):
                    pred_interval_lower[:,i] = mu + np.sqrt(s_lower)*z

                pred_interval_upper = np.zeros((len(yhat), len(Z)))
                s_upper=s2+0.5*s2s
                s_upper=s_upper.ravel()
                for i, z in enumerate(Z):
                    pred_interval_upper[:,i] = mu + np.sqrt(s_upper)*z

                plt.fill_between(xx, pred_interval_lower[:,0], pred_interval_upper[:,0], alpha = 0.3,color=clr)
                plt.fill_between(xx, pred_interval_lower[:,1], pred_interval_upper[:,1], alpha = 0.3,color=clr)
                plt.plot(xx, pred_interval[:,0],color=clr, linewidth=0.5)
                plt.plot(xx, pred_interval[:,1],color=clr, linewidth=0.5)
                plt.fill_between(xx, pred_interval[:,0], pred_interval[:,1], alpha = 0.1,color=clr)
                if sex == 'both':
                    np.savetxt(os.path.join(plot_dir, 'pr_int_'+str(b[1])+'l.txt'), pred_interval_lower)
                    np.savetxt(os.path.join(plot_dir, 'pr_int_'+str(b[1])+'u.txt'), pred_interval_upper)
                    np.savetxt(os.path.join(plot_dir, 'pr_int_'+str(b[1])+'.txt'), pred_interval)
            plt.xlabel(cov_x)
            plt.ylabel(roi)
            plt.title(roi)
            if cov_x == 'age':
              if sex == 'both':
                  plt.xlim((X_test.min()[0]-2,X_test.max()[0]+2))
              if sex != 'both':
                plt.xlim((X_test[X_test['sex'] == sex]['age'].min()-2,X_test[X_test['sex'] == sex]['age'].max()+2))
            else:
              if sex == 'both':
                plt.xlim((X_test.min()[0],X_test.max()[0]))
              if sex != 'both':
                plt.xlim((X_test[X_test['sex'] == sex][cov_x].min(),X_test[X_test['sex'] == sex][cov_x].max()))
            plt.savefig(os.path.join(roi_dir, 'centiles_' + str(sex)),  bbox_inches='tight')
            plt.show()
            plt.close()

          # QQ-plot
          if sex == 'both' and qq is not None:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(17, 4))
            for j, xaxisrange in enumerate(['all_range','<50%','>50%']):
              # x_axis_feature, feature
              sorted_y = np.column_stack((X_tr[:,1], y_tr))  #sorted_y = np.column_stack((X_te[:,1], y_te_rescaled_list))
              #order by x_axis_feature
              sort_index = np.argsort(sorted_y[:,0])
              y_sorted_array = sorted_y[sort_index]
              #total number of subjects for visualization
              subjs = y_sorted_array.shape[0]
              print("Running qq plot for feature", roi, "; Sex:", sex, "; X-axis range:", xaxisrange)
              if xaxisrange == '<50%':
                #filter features for subjects <50 xaxisrange
                y_sorted_array = y_sorted_array[y_sorted_array[:,0] < (min_train + max_train)/2] #y_sorted_array = y_sorted_array[y_sorted_array[:,0] < (min_test + max_test)/2]
                subjs = y_sorted_array.shape[0]
              elif xaxisrange == '>50%':
                #filter features for subjects >50 xaxisrange
                y_sorted_array = y_sorted_array[y_sorted_array[:,0] > (min_train + max_train)/2]
                subjs = y_sorted_array.shape[0]

              if warp is None:
                bounds = [[i] for i in np.arange(0, 1, 0.01)]
                for b in bounds:
                  Z = norm.ppf(b)
                  pred_interval = np.zeros((len(yhat), len(Z)))
                  mu=yhat.ravel()
                  s=s2.ravel()
                  for i, z in enumerate(Z):
                    pred_interval[:,i] = mu + np.sqrt(s)*z
                  pr_int = pred_interval[:,0]
                  # get indices of the percentile that matches the xcovar_feature of subjects
                  xcovar_indices = np.searchsorted(xx,y_sorted_array[:,0])
                  # Use the xcovar_indices to extract the corresponding pr_int values
                  pr_int_values = pr_int[xcovar_indices]
                  # Count the number of y_values that are less than the corresponding pr_int values
                  num_below = np.count_nonzero(y_sorted_array[:, 1] < pr_int_values)

                  axs = axes[j]
                  axs.scatter(b[0], num_below, s=4, c=clr)

              else:
                percentiles = np.arange(0, 1, 0.01)
                for perc in percentiles:
                  junk, pr_int = W.warp_predictions(np.squeeze(yhat), np.squeeze(s2), warp_param, percentiles=[perc])
                  pr_int = pr_int[:,0]

                  # get indices of the percentile that matches the xcovar_feature of subjects
                  xcovar_indices = np.searchsorted(xx,y_sorted_array[:,0])
                  # Use the xcovar_indices to extract the corresponding pr_int values
                  pr_int_values = pr_int[xcovar_indices]
                  # Count the number of y_values that are less than the corresponding pr_int values
                  num_below = np.count_nonzero(y_sorted_array[:, 1] < pr_int_values)

                  axs = axes[j]
                  axs.scatter(perc, num_below, s=4, c=clr)

              # Ideal line
              axs.plot([0, 1], [0, subjs], color='red', alpha=0.5)
              axs.set_xlabel('Quantile (q)')
              axs.set_ylabel('N(q)')
              axs.set_title(xaxisrange + " " + roi)
            plt.savefig(os.path.join(roi_dir, "QQ_plots_counting"), bbox_inches='tight')
            plt.show()
            plt.close()

  os.chdir(out_dir)

  return

"""# **Run normative model**"""

# PREPARE TRAINING DATA

# Read 90% split of controls from non-clinical datasets
sanos_90 = pd.read_csv('/content/drive/MyDrive/Carmen_Rueda/Width_Thickness_Adjusted/80clinical/Splits/controls_split_90_filtered.csv', header=0)

# Read 80% split of controls from clinical datasets
sanos_clinicas_80 = pd.read_csv('/content/drive/MyDrive/Carmen_Rueda/Width_Thickness_Adjusted/80clinical/Splits/controls_clinic_split_80_filtered.csv', header=0)

# Combine both subsets to create the final training data
train_data = pd.concat([sanos_90, sanos_clinicas_20])
train_data.to_csv('/content/drive/MyDrive/final/train_data.csv', index=False)

# Prepare training covariates (X)
# Select columns for covariates: age, sex, scanner ID, and Euler number
train = train_data.loc[:, ['age', 'sex', 'scanner', 'euler_med']]

# One-hot encode the 'scanner' column to convert it into binary columns (e.g. scanner_1, scanner_2, ...)
dummies = pd.get_dummies(train['scanner'], prefix='scanner')

# Append the binary scanner columns to the training covariates
train = pd.concat([train, dummies], axis=1)

# Remove the original 'scanner' column as it's now redundant
train.drop('scanner', axis=1, inplace=True)

# Save the covariates to CSV
train.to_csv('/content/drive/MyDrive/final/X_train.csv', index=False)

# Prepare training features (y) 
# Select the sulcal feature columns (from column 9 onward)
train_feat = train_data.iloc[:, 9:]
train_feat.to_csv('/content/drive/MyDrive/final/y_train.csv', index=False)

# PREPARE TESTING DATA

# Read 10% split of controls from non-clinical datasets
sanos_10 = pd.read_csv('/content/drive/MyDrive/Carmen_Rueda/Width_Thickness_Adjusted/80clinical/Splits/controls_split_10_filtered.csv', header=0)

# Read 20% of controls from clinical datasets
sanos_clinicas_20 = pd.read_csv('/content/drive/MyDrive/Carmen_Rueda/Width_Thickness_Adjusted/80clinical/Splits/controls_clinic_split_20_filtered.csv', header=0)

# Read all patients from clinical datasets
pac_clinicas = pd.read_csv('/content/drive/MyDrive/Carmen_Rueda/Width_Thickness_Adjusted/80clinical/Splits/patients_clinic_filtered.csv', header=0)

# Combine all the above to create the final test dataset
test_data = pd.concat([sanos_10, sanos_clinicas_20, pac_clinicas])
test_data.to_csv('/content/drive/MyDrive/final/test_data.csv', index=False)

# Prepare testing covariates (X) 
test = test_data.loc[:, ['age', 'sex', 'scanner', 'euler_med']]
dummies = pd.get_dummies(test['scanner'], prefix='scanner')
test = pd.concat([test, dummies], axis=1)
test.drop('scanner', axis=1, inplace=True)
test.to_csv('/content/drive/MyDrive/final/X_test.csv', index=False)

# Prepare testing features (y) 
test_feat = test_data.iloc[:, 9:]
test_feat.to_csv('/content/drive/MyDrive/final/y_test.csv', index=False)

# Define paths to the generated training and testing files
X_train = '/content/drive/MyDrive/final/X_train.csv'
y_train = '/content/drive/MyDrive/final/y_train.csv'
X_test = '/content/drive/MyDrive/final/X_test.csv'
y_test = '/content/drive/MyDrive/final/y_test.csv'
out_dir = '/content/drive/MyDrive/final'

# Define the covariates to use in the normative model
cols_cov = ['age', 'sex', 'euler_med']

# ---- Apply the Normative Model ----
# Train and test the normative model using the provided covariates and features
normative_model(
    X_train,         # Path to training covariates
    y_train,         # Path to training sulcal features
    out_dir,         # Output directory for model results
    X_test=X_test,   # Path to test covariates (omit for cross-validation)
    y_test=y_test,   # Path to test features (omit for cross-validation)
    warp='WarpSinArcsinh',  # Non-linear transformation for modeling
    cols_cov=cols_cov,      # List of covariates to include
    k_fold=None,            # No cross-validation used (specify for cross-validation)
    alg='blr',              # Bayesian Linear Regression
    nknots=5,               # Number of knots for warping
    p=3,                    # Polynomial degree
    opt='powell',           # Optimization method
    cov_x='age',            # Variable to evaluate QQ-plots on
    qq="yes",               # Generate QQ-plots
    varcov=None,            # No additional covariate variance
    varcov_te=None,         # No test variance
    X_ad=None,              # No adaptation data
    y_ad=None               # No adaptation targets
)
