"""Preprocess data for the challenge.

This script will be invoked in two ways during the Unearthed scoring pipeline:
 - first during model training on the 'public' dataset
 - secondly during generation of predictions on the 'private' dataset
"""
import argparse
import logging
import numpy  as np
import pandas as pd
from scipy.stats import norm, skew, boxcox_normmax
from scipy.special import boxcox1p
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess(data_file):

    df = pd.read_csv(data_file, index_col=0)

    target_columns = ['AgPPM', 'AsPPM', 'AuPPM', 'BaPPM', 'BiPPM', 'CdPPM', 'CoPPM', 'CuPPM', 'FePCT', 'MnPPM', 'MoPPM', 'NiPPM', 'PPCT','PbPPM', 'SPPM', 'SbPPM', 'SePPM', 'SnPPM', 'SrPPM', 'TePPB', 'ThPPB', 'UPPB', 'VPCT', 'WPPM', 'ZnPPM', 'ZrPPM', 'BePPM', 'AlPPM', 'CaPPM', 'CePPM', 'CrPPM', 'CsPPM', 'GaPPM', 'GePPM', 'HfPPM', 'InPPM', 'KPPM', 'LaPPM', 'LiPPM', 'MgPPM', 'NaPPM', 'NbPPM', 'RbPPM', 'RePPM', 'ScPPM', 'TaPPM', 'TiPPM', 'TlPPM', 'YPPM']

    logger.info(f"preprocess input is {df.columns}")
    df['depth_diff'] = df['DepthTo']-df['DepthFrom']

    
    df['350*351']     = df['350']*df['351']
    df['350*352']     = df['352']*df['350']
    
    
    df['2516-351']        = df['2516'] - df['351']
    df['2516-350']        = df['2516'] - df['350']
    df['350*353']         = df['353']  * df['350']

    for x in df.columns:
        if x in target_columns:
            pass
        else:
            df["{}_mean".format(x)] = df.groupby('SiteID')[x].transform('mean')

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/opt/ml/processing/input/public/public.csv.gz')
    parser.add_argument('--output', type=str, default='/opt/ml/processing/output/preprocess/public.csv')
    args, _ = parser.parse_known_args()

    # call preprocessing on private data
    df = preprocess(args.input)
    
    
    # write to the output location
    df.to_csv(args.output)
    