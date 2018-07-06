import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn import model_selection
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA 
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb 
import xgboost as xgb 

from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

#Read train data file
train_df = pd.read_csv('./train.csv')

#Training set
print("Training set:")
n_data = len(train_df)
n_features = train_df.shape[1]
print("Number of Records: {}".format(n_data))
print("Number of Features: {}".format(n_features))

train_df.head(n=10)
train_df.info()

#Check for Missing Data
print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size) )

if(train_df.columns[train_df.isnull().sum() != 0].size):
	print("Features with NaN => {}".format(list(train_df.columns[train_df.isnull().sum() != 0])))



