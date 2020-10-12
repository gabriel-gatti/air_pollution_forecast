# -*- coding: utf-8 -*-
"""
@author: Gabriel Gatti Lima
e-mail: gabriel.gatti.lima@usp.br
"""
import support_module as sup
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.dummy import DummyRegressor

sns.set()
keras = tf.keras

# %% Getting Data
API_lables = {'42101': 'Carbon monoxide',
              '42401': 'Sulfur dioxide',
              '42402': 'Hydrogen sulfide',
              '42600': 'Reactive oxides of nitrogen (NOy)',
              '42601': 'Nitric oxide (NO)',
              '42602': 'Nitrogen dioxide (NO2)',
              '42603': 'Oxides of nitrogen (NOx)',
              '42612': 'NOy - NO',
              '43102': 'Total NMOC (non-methane organic compound)',
              '44201': 'Ozone'}

uri = 'https://raw.githubusercontent.com/gabriel-gatti/air_pollution_forecast/'
uri += 'master/Site_3_NY_Agrupado.csv'

df_result = pd.read_csv(uri, low_memory=False)
features = df_result.reset_index().interpolate(method='cubic').dropna(axis=0)
features = features.set_index(pd.date_range(
    "2019-01-01 00:00", "2019-12-31 23:00", freq="60min"))
features = features.drop(['index', 'Date-Time'], axis=1).rename(
    columns=API_lables)
features['Month'] = features.index.month
features['Week'] = features.index.weekday
# features['Day'] = features.index.dayofyear
features['Hour'] = features.index.hour

fig = features.plot(subplots=True, layout=(9, 1), figsize=(22, 22))
for ax in fig:
    ax[0].legend(bbox_to_anchor=(1.01, 0.5), loc='center left', ncol=1)
plt.show()

features_indexes = features.index
features_columns = features.columns
out_col = "Ozone (Parts per million)"

features.head()


# %% Visualizadno correlações

X = features.drop(out_col, axis=1)  # independent columns
y = features[out_col]               # target column i.e price range

# get correlations of each features in dataset
corrmat = features.corr()
top_corr_features = corrmat.index

plt.figure(figsize=(10, 10))
plt.title("Heatmap Matriz de Correlação")
# plot heat map
g = sns.heatmap(features[top_corr_features].corr(), annot=True)

""" Seleciando Variáveis K-Best (Informação Mutua)"""
# %% Feature Selection K-Best(Mutual Information)"""
k = 4
f_regression_selector = SelectKBest(f_regression, k=k)
selected_features = f_regression_selector.fit_transform(features,
                                                        features[out_col])
features_names = list()
for i in range(len(features.columns)):
    if f_regression_selector.get_support()[i]:
        features_names.append(features.columns[i])

selected_features = pd.DataFrame(selected_features, columns=features_names,
                                 index=features_indexes)
selected_features[out_col] = features[out_col]

fig = selected_features.plot(subplots=True, layout=(
    len(selected_features.columns), 1), figsize=(22, 10),
    title='Dados selecionados não normalizados')

for ax in fig:
    ax[0].legend(bbox_to_anchor=(1.01, 0.5), loc='center left', ncol=1)
# plt.ylabel('S&P500 Points')
plt.show()

# %% Normalizando os Dados
features_normalizadas, stats = sup.normalize(selected_features)
stats_out = (stats[0][out_col], stats[1][out_col])

fig = features_normalizadas.plot(subplots=True,
                                 layout=(k+1, 1), figsize=(22, 10))
plt.title('Dados selecionados e normalizados')
for ax in fig:
    ax[0].legend(bbox_to_anchor=(1.01, 0.5), loc='center left', ncol=1)

# %% Benchmarks
df_naive = sup.naive_forecast(features_normalizadas[out_col].values,
                              days_in_the_future=1, stats_out=stats_out,
                              denorm=False)

# df_naive.plot()
"""Multivariate LSTM Model"""

# %% Tunning Hyper-Parameters

days_future = 1
patience_ = 50
train_perc = (0.75, 0.15, 0.10)

hyper_par = {'n_layers': (2, 5),
             'batch_size': [32, 64, 128, 256, 512],
             'length': (8, 256),
             'learning_rate': (-1, -6)}

model, history, model_evl, valid_sets = sup.model_pipeline(
    dataframe=features_normalizadas, output_column=out_col,
    days_in_the_future=days_future, n_layers=n_layers, train_perc=train_perc,
    batch_size=batch_size, length=length, patience_=patience_,
    learning_rate=learning_rate)

# %% Predicting and plotting predictions (Denormalizing)"""
df_predicted = sup.model_prediction(model, valid_sets[0], valid_sets[1],
                                    denorm=True,
                                    stats_out=stats_out)

df_naive2 = sup.naive_forecast(serie=features_normalizadas[out_col].values,
                               days_in_the_future=days_future,
                               stats_out=stats_out, denorm=True)

df_naive2 = df_naive2[-len(df_predicted):].reset_index(drop=True)
df_summ = pd.concat([df_predicted, df_naive2['Naive Forecast']], axis=1)
df_summ['\u0394 Naive'] = df_summ['Naive Forecast'] - df_summ['Actual']
df_summ.index = features_indexes[-len(df_predicted):]
df_summ[['Actual', 'Predicted', '\u0394 Naive', '\u0394 Predicted']].plot(
    figsize=(22, 10), xlim=(datetime.strptime(
        '19/12/01 00:00', '%y/%m/%d %H:%M'),
        datetime.strptime('19/12/31 00:00', '%y/%m/%d %H:%M')))

plt.title('Visualizando os reultados')
# plt.ylabel('S&P500 Points')
plt.show()

# %% Comparing Metrics"""

mape = tf.keras.losses.MeanAbsolutePercentageError()
mae = tf.keras.losses.mean_absolute_error

naive_MAE = mae(df_summ['Actual'].values,
                df_summ['Naive Forecast'].values).numpy()
model_MAE = mae(df_summ['Actual'].values,
                df_summ['Predicted'].values).numpy()

naive_MAPE = mape(df_summ['Actual'].values,
                  df_summ['Naive Forecast'].values).numpy()
model_MAPE = mape(df_summ['Actual'].values,
                  df_summ['Predicted'].values).numpy()

naive_STD = np.std(df_summ['Naive Forecast'])
model_STD = np.std(df_summ['Predicted'])

print('Model Mean Absolute Error: %.5f' % (model_MAE))
print('--------------------------------')
print('Naive Mean Absolute Error: %.5f' % (naive_MAE))
print('\n')
print('Model Mean Absolute Percentage Error: %.2f%%' % (model_MAPE))
print('--------------------------------')
print('Naive Mean Absolute Percentage Error: %.2f%%' % (naive_MAPE))
print('\n')
print('Model Std: %.5f' % (model_STD))
print('--------------------------------')
print('Naive Std: %.5f' % (naive_STD))
print('\n')

# %%Plotting Histogram of Errros"""

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
sns.distplot(df_summ['\u0394 Predicted'], hist=True, kde=True,
             label='\u0394 Predicted')
sns.distplot(df_summ['\u0394 Naive'], hist=True, kde=True,
             label='\u0394 Naive')
plt.legend(prop={'size': 12})
plt.title('Histogram of Errors')
plt.xlabel('S&P500 Points')
plt.ylabel('Density')
plt.show()

"""## Plotting Model History"""

# summarize history for accuracy
plt.figure(figsize=(22, 8))

# summarize history for loss
plt.subplot(1, 4, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Huber Loss')
plt.ylabel('Huber Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')


plt.subplot(1, 4, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')


# summarize history for loss
plt.subplot(1, 4, 3)
plt.plot(history.history['mape'])
plt.plot(history.history['val_mape'])
plt.title('MAPE')
plt.ylabel('MAPE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')


# summarize history for loss
plt.subplot(1, 4, 3)
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Root Mean Square Error')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

"""##Exemplos de Predição
sup.show_predictions(model, valid_ds[0], valid_ds[1], stats_out, denorm=True,
                     output_column=out_col, length=length,
                     days_in_the_future=days_future)"""

# %% Plotting Actual x Predictions"""

x = min([min(df_summ['Actual']), min(df_summ['Predicted'])])
y = max([max(df_summ['Actual']), max(df_summ['Predicted'])])

plt.figure(figsize=(15, 15))
plt.scatter(df_summ['Actual'].values, df_summ['Predicted'].values)
sns.lineplot([x, y], [x, y], color='red')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title('Predições x Real')

# %%

days_future = 1
patience_ = 3
train_perc = (0.75, 0.15, 0.10)

hyper_par = {'n_layers': (2, 5),
             'batch_size': [32, 64, 128, 256, 512],
             'length': (8, 256),
             'learning_rate': (-1, -6)}

df_evl, models = sup.random_tune_hyperpar(dataframe=features_normalizadas,
                                          output_column=out_col,
                                          hyper_dict=hyper_par,
                                          days_in_the_future=1,
                                          patience_=5, sampling_rate=1,
                                          train_perc=(0.75, 0.15, 0.10),
                                          epochs=1000, random_searchs=3)

