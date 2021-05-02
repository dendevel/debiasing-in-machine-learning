from fairml.datasets import DataSet
from typing import Union
import pandas as pd
import fairml.metrics as metrics
import numpy as np

class Evaluator():
    _test_data_by_group: dict
    _fairness_scores: pd.DataFrame
    _positive_predicted_probability: float
    _predictor: any
    _fairness: pd.Series()

    def __init__(self, predictor):
        self._predictor = predictor

    def bias(self, data: DataSet):
        result = pd.Series(name="Bias")
        for sensitive_feature in data.get_sensitive_feature_names():
            result[sensitive_feature] = metrics.variation_coefficient(data.get_x(), sensitive_feature)
        return result

    def fit_predict(self, train_data:DataSet, test_data: DataSet, fairness_metric:metrics.FairnessMetric, sensitive_features:[str]=[]):
        self._predictor.fit(train_data.get_x(), train_data.get_y())
        sensitive_features = train_data.get_sensitive_feature_names() if len(sensitive_features) == 0 else sensitive_features
        self._test_data_by_group = dict()
        df = test_data.to_dataframe()

        df = test_data.to_dataframe(include_groups=True)
        reduced = df.groupby(test_data.get_groups().name)
        groups = test_data.get_groups().unique()
        for group in groups:
            X = reduced.get_group(group)
            g = X.pop(X.columns[-1]) 
            y = X.pop(X.columns[-1])
            y_dict = dict()
            y_dict['y_true'] = y
            y_dict['y_pred'] = self._predictor.predict(X)
            self._test_data_by_group[group] = y_dict
        
        self._positive_predicted_probability = metrics.positive_predicted_probability(self._test_data_by_group)
        self._fairness_scores = fairness_metric.run(self._test_data_by_group)
        self._fairness_scores.name = fairness_metric.get_name()
        return self._fairness_scores

    def fairness(self):
        self._fairness = pd.Series(name=self._fairness_scores.name)
        for metric, values in self._fairness_scores.items():
            self._fairness[metric] = np.min(values)/np.max(values)
        return self._fairness

    def is_fair(self, threshold:float=0.8) -> bool:
        return self._fairness.apply(lambda x: x>=threshold).all()

    def set_sensitive_features(self, sensitive_feature_names:[str]):
        self._sensitive_features = sensitive_feature_names