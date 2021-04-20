"""Unearthed Training Template"""
import argparse
import logging
import pickle
import sys
from scipy.stats import norm, skew, boxcox_normmax
from scipy.special import boxcox1p

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_transformer

from io import StringIO
from os import getenv
from os.path import abspath, join
import numpy as np
import pandas as pd
from preprocess import preprocess
from ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath('/opt/ml/code') not in sys.path:
    sys.path.append(abspath('/opt/ml/code'))


def train(args):
    """Train

    Your model code goes here.
    """
    logger.info('calling training function')

    # preprocess
    # if you require any particular preprocessing to create features then this 
    # *must* be contained in the preprocessing function for the Unearthed pipeline
    # apply it to the private data
    df = preprocess(join(args.data_dir, 'public.csv.gz'))

    # the list of 49 elements which are target variables for this challenge
    target_columns = ['AgPPM', 'AsPPM', 'AuPPM', 'BaPPM', 'BiPPM', 'CdPPM', 'CoPPM', 'CuPPM', 'FePCT', 'MnPPM', 'MoPPM', 'NiPPM', 'PPCT','PbPPM', 'SPPM', 'SbPPM', 'SePPM', 'SnPPM', 'SrPPM', 'TePPB', 'ThPPB', 'UPPB', 'VPCT', 'WPPM', 'ZnPPM', 'ZrPPM', 'BePPM', 'AlPPM', 'CaPPM', 'CePPM', 'CrPPM', 'CsPPM', 'GaPPM', 'GePPM', 'HfPPM', 'InPPM', 'KPPM', 'LaPPM', 'LiPPM', 'MgPPM', 'NaPPM', 'NbPPM', 'RbPPM', 'RePPM', 'ScPPM', 'TaPPM', 'TiPPM', 'TlPPM', 'YPPM']
    gr_1 = ['AgPPM','AuPPM','BiPPM','CdPPM', 'CuPPM', 'FePCT', 'NiPPM','PPCT','SPPM', 'SePPM','SnPPM', 'TePPB', 'ZnPPM','InPPM']
    gr_2 = ['AsPPM', 'CoPPM']
    gr_3 = ['BaPPM', 'SrPPM', 'ThPPB', 'UPPB', 'WPPM']
    gr_4 = ['MnPPM', 'PbPPM', 'SbPPM']
    gr_5 = ['MoPPM', 'RePPM']
    gr_6 = ['VPCT', 'CrPPM', 'CsPPM', 'GaPPM', 'LiPPM', 'MgPPM', 'NbPPM', 'RbPPM', 'ScPPM', 'TaPPM','TiPPM', 'TlPPM']
    gr_7 = [ 'ZrPPM', 'CaPPM', 'HfPPM','KPPM']
    gr_8 = ['BePPM', 'AlPPM', 'NaPPM']
    gr_9 = ['CePPM', 'GePPM', 'LaPPM', 'YPPM']
    y_train = df[target_columns]

    logger.info(f"training target shape is {y_train.shape}")
    X_train = df.drop(columns = target_columns)

    model = EnsembleModel()
    model.fit1(X_train, y_train[gr_1])
    model.fit2(X_train, y_train[gr_2])
    model.fit3(X_train, y_train[gr_3])
    model.fit4(X_train, y_train[gr_4])
    model.fit5(X_train, y_train[gr_5])
    model.fit6(X_train, y_train[gr_6])
    model.fit7(X_train, y_train[gr_7])
    model.fit8(X_train, y_train[gr_8])
    model.fit9(X_train, y_train[gr_9])


    logger.info(f"training input shape is {X_train.shape}")
    # save the model to disk
    save_model(model, args.model_dir)


def save_model(model, model_dir):
    """Save model to a binary file.

    This function must write the model to disk in a format that can
    be loaded from the model_fn.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info(f"saving model to {model_dir}")
    with open(join(model_dir, 'model.pkl'), 'wb') as model_file:
        pickle.dump(model, model_file)


def model_fn(model_dir):
    """Load model from binary file.

    This function loads the model from disk. It is called by SageMaker.

    WARNING - modifying this function may case the submission process to fail.
    """
    logger.info("loading model")
    with open(join(model_dir, 'model.pkl'), 'rb') as file:
        return pickle.load(file)


def input_fn(input_data, content_type):
    """Take request data and de-serialize the data into an object for prediction.

    In the Unearthed submission pipeline the data is passed as "text/csv". This
    function reads the CSV into a Pandas dataframe ready to be passed to the model.

    WARNING - modifying this function may cause the submission process to fail.
    """
    return pd.read_csv(StringIO(input_data), index_col = 0)

if __name__ == '__main__':
    """Training Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed train" command.
    
    WARNING - modifying this function may cause the submission process to fail.

    The main function must call preprocess, arrange th
    """    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=getenv('SM_MODEL_DIR', '/opt/ml/models'))
    parser.add_argument('--data_dir', type=str, default=getenv('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    train(parser.parse_args())
