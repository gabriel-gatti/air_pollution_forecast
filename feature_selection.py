from sklearn.tree import DecisionTreeRegressor
import sklearn.feature_selection as fs
from utils import time_it, normalize_min_max

def run_feature_selection(df_input, config: dict, x_cols=['PRESS', 'RH_DP', 'SO2', 'TEMP', 'WIND', 'CO', 'NO2', 'OZONE'], y_col=['PM25']) -> dict:
    df_norm = normalize_min_max(df_input)
    X = df_norm[x_cols]
    y = df_norm[y_col]
    get_cols = lambda selector, x_cols: [col_name for boolean, col_name in zip(selector.get_support(), x_cols) if boolean]

    tempo_execucao      = {}
    resultado_execucao  = {}
    execucao_dict = {
        'VarianceTreshold': (fs.VarianceThreshold, {'threshold': config.get('treshold', 0)}, (X,)),
        'SelectKBest_f_regression': (fs.SelectKBest,{'score_func':fs.f_regression, 'k': config.get('kbest', 0)},(X, y)),
        'SelectKBest_mutual_info_regression': (fs.SelectKBest,{'score_func':fs.mutual_info_regression, 'k': config.get('kbest', 0)},(X, y)),
        'SelectPercentile_f_regression': (fs.SelectPercentile,{'score_func':fs.f_regression, 'percentile': config.get('percentile', 0)},(X, y)),
        'SelectPercentile_mutual_info_regression': (fs.SelectPercentile,{'score_func':fs.mutual_info_regression, 'percentile': config.get('percentile', 0)},(X, y)),
        'SequentialFeatureSelector': (fs.SequentialFeatureSelector, {'estimator': DecisionTreeRegressor(), 'n_features_to_select':config.get('kbest', 0)}, (X, y)),
        'RFE': (fs.RFE, {'estimator': DecisionTreeRegressor(), 'n_features_to_select':config.get('kbest', 0)},(X, y)),
        'RFECV': (fs.RFECV, {'estimator': DecisionTreeRegressor(), 'min_features_to_select':config.get('kbest', 0)},(X, y)),
        }

    for name, (func, param_func, param_fit) in execucao_dict.items():
        selector, tempo = time_it(lambda: func(**param_func).fit(*param_fit))()
        tempo_execucao[name] = tempo
        resultado_execucao[name] = get_cols(selector, x_cols)

    return (resultado_execucao, tempo_execucao)
