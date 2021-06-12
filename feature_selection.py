import sklearn.feature_selection as fs

fs.VarianceThreshold(threshold=(.8 * (1 - .8))).fit_transform(X)
