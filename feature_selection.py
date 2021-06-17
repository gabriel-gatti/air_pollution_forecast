import sklearn
import sklearn.feature_selection as fs
import pandas as pd


path        = '/media/gabriel-gatti/HDD/Dados TCC/Unified Pickles/merged/Hourly_Merged_CO-NO2-OZONE-PM25-PRESS-RH_DP-SO2-TEMP-WIND.pkl'
y_col       = 'PM25'
x_cols      = ['PRESS', 'RH_DP', 'SO2', 'TEMP', 'WIND', 'CO', 'NO2', 'Ozone']
df_input    = pd.read_pickle(path)[[*x_cols, y_col]]
config_args =  {'treshold': 0.5,
                'percentile': 0.6,
                'kbest': 3
                }

def run_feature_selection(df_input, x_cols, y_col, config: dict) -> dict:
    X = df_input[x_cols]
    y = df_input[y_col]

    treshold_arg    = config.get('treshold')
    percentile_arg  = config.get('percentile',0)
    kbest_arg       = config.get('kbest', 0)

    get_cols = lambda selector, x_cols: [col_name for boolean, col_name in zip(selector.get_support(), x_cols) if boolean]

    selectors_list = [
        fs.VarianceThreshold(threshold=(treshold_arg)).fit(X),
        fs.SelectKBest(fs.f_regression, k=kbest_arg).fit(X, y),
        fs.SelectKBest(fs.mutual_info_regression, k=kbest_arg).fit(X, y),
        fs.SelectPercentile(fs.f_regression, percentile=percentile_arg).fit(X, y),
        fs.SelectPercentile(fs.mutual_info_regression, percentile=percentile_arg).fit(X, y)
    ]

    result_dict = {str(selector): cols for selector, cols in zip(selectors_list, map(lambda x: get_cols(x, x_cols), selectors_list))}
    
    return result_dict