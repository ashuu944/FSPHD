from logger import log
from logger import Logs
import configuration as conf
from preprocessor import preprocess
from timeseries_lstm import timeseries_lstm
from timeseries_rf import timeseries_random_forest
from structure_rf import conjunctural_structural_rf
from spatial_cnn import cnn
from rf_feature_fusion import feature_fusion_rf
import argparse, sys

def get_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some arguments.")
    
    # Add arguments
    parser.add_argument('-country', type=str, required=True, help='The name of the country (burkina_faso/rwanda/tanzania)')
    parser.add_argument('-algorithm', type=str, required=True, help='The name of the algorithm(classification/regression)')
    parser.add_argument('-tt_split', type=str, required=False, help='The name of the algorithm(temporal/percentage)')
    
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Access arguments
    country = args.country
    algorithm = args.algorithm
    if args.tt_split is not None:
        tt_split = args.tt_split
    else:
        tt_split = 'percentage'
    # Ensure arguments are not None
    if not country or not algorithm:
        print("Error: Both -country,-algorithm must be provided.")
        parser.print_help()
        sys.exit(1)
    return country, algorithm, tt_split
      
country, algorithm, tt_split = get_arguments()

if country:
    for r_split in [1]:  # [1, 2, 3, 4, 5]

        for rep in conf.OUTPUT_VARIABLES[country][algorithm]:  
            
            # perform preprocessing
            preprocess(rep, r_split, country, algorithm, tt_split)

            timeseries_lstm(rep, algorithm, r_split, country,tt_split) # Timeseries with LSTM

            #timeseries_random_forest(rep, algorithm, r_split, country, tt_split) #Timeseries with RF

            cnn(rep, algorithm, r_split, country, tt_split)  # High Spatial with CNN

            conjunctural_structural_rf(rep, algorithm, r_split, country, tt_split) # Conjuctural & Spatial with RF

            #feature_fusion_rf(rep, algorithm, r_split, country, tt_split) # Not stable yet

