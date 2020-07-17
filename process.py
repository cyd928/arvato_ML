# load packages
import numpy as np
import pandas as pd



# consolidate all cleanning steps into one function

def clean_data(df, features):
    '''
    Function to clean up raw data file
    Step 1: Drop Duplicates
    Step 2: Convert Unknown Values to NaNs
    Step 3: Analyse Columns/Rows with Missing Values
    Step 4: Encoding Categorical String Columns
    Step 5: Reduce Columns with the Same Meaning
    Step 6: Reduce Highly Correlated Columns
    
    INPUT: 
    df - raw df
    features - features.csv

    OUTPUT:
    df - cleaned df

    '''


    # STEP 1
    df = df.drop_duplicates()
    print('Done removing duplicates')
    
    # STEP 2
    df[['CAMEO_DEUG_2015','CAMEO_DEU_2015','CAMEO_INTL_2015']] = df[['CAMEO_DEUG_2015','CAMEO_DEU_2015','CAMEO_INTL_2015']].replace(['X','XX'], '-1').apply(pd.to_numeric, errors='coerce')
    features['unknown_value'] = features['unknown_value'].replace(["['-1','X']"], "[-1]")
    features['unknown_value'] = features['unknown_value'].replace(["['-1','XX']"], "[-1]")
    
    for idx in features.index.values: 
        column_name = features.feature[idx]
        lst = features.iloc[idx, 2].strip('][').split(',') 
    
        if lst[0] != '': # if unknown value list is not empty, change list values to NaNs
            df[column_name] = df[column_name].replace(lst, np.nan)
    print('Done Converting unknown numerical values to NaN')
    
    # STEP 3
    missing_col_df = df.isnull().sum().reset_index(name = 'Number of Missing Values')
    missing_col_df['Percentage of Missing Values'] = missing_col_df['Number of Missing Values']/df.shape[0]
    col_to_drop = missing_col_df[missing_col_df['Percentage of Missing Values'] > 0.3]['index'].tolist()
    df = df.drop(labels = col_to_drop, axis = 1)
    print('Done dropping columns with more than 30% missing values')
    
    missing_row_df = df.isnull().sum(axis=1).reset_index(name = 'Number of Missing Values')
    missing_row_df['Percentage of Missing Values'] = missing_row_df['Number of Missing Values']/df.shape[1]
    rows_to_drop = missing_row_df[missing_row_df['Percentage of Missing Values'] > 0.3]['index'].tolist()
    df = df.drop(labels = rows_to_drop)
    print('Done dropping rows with more than 30% missing values')
    
    # STEP 4
    df = df.drop(labels=['D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM'], axis = 1)
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].replace(['O','W'],[0, 1]).astype(float)
    print('Done encoding categorical columns')
    
    # STEP 5
    df = df.drop(labels = ['ALTERSKATEGORIE_FEIN','LP_FAMILIE_GROB','LP_STATUS_GROB','LP_LEBENSPHASE_GROB'], 
                                axis = 1)
    print('Done keeping either GROB or FEIN version of the feature')
    
    # STEP 6
    corr_matrix = df.corr().abs()     # Create correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))# Select upper triangle of correlation matrix
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]# Find index of feature columns with correlation greater than 0.85
    df = df.drop(df[to_drop], axis=1)
    print('Done dropping highly correlated features')

    print('Data cleaning process complete')

    return df 
    