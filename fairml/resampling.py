from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, OneSidedSelection, CondensedNearestNeighbour, TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN, NeighbourhoodCleaningRule, InstanceHardnessThreshold
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, ADASYN, SMOTENC, SMOTEN, BorderlineSMOTE, KMeansSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline
from collections import Counter
import pandas as pd
from fairml.datasets import DataSet, concat_dataframes
import numpy as np
import statistics


class Resampler():
# -------------------------------- Properties -------------------------------- #
    _strategy: str
    _fitted: bool
    _weights: pd.Series
    _group_target_amount: dict
    _data: pd.DataFrame
    _target: pd.Series
    
    MIMIMUM_SAMPLES = 6

# -------------------------------- Constructor ------------------------------- #

    def __init__(self, strategy:str or None=None):
        self._fitted=False
        if strategy:
            self.set_strategy(strategy)
        return 
# ------------------------------- Getter/Setter ------------------------------ #

    def set_strategy(self, strategy:str):
        allowed_strategies = ['combined', 'over', 'under']
        if not strategy in allowed_strategies:
            raise Exception("Invalid strategy '"+strategy+"'. Possible values are: " + str(allowed_strategies))
        self._strategy = strategy

    def strategy(self):
        return self._strategy

# -------------------------------- strategies -------------------------------- #

    def fit(self, data: DataSet, positive_predicted:float, fairness_evaluation:pd.Series):
        if not hasattr(self, '_strategy'):
            raise Exception('No resampling strategy set yet. Please set using `set_strategy(strategy)` method.')
        self._data = data.to_dataframe()
        self._sensitive_features = data.get_sensitive_features()
        self._target = data.get_groups().copy()

        self.calculate_weights(positive_predicted, fairness_evaluation)
        self.weights_to_number()

        self._target_init = dict()
        self._target_under = dict()
        self._target_over = dict()
        
        value_counts = self._target.value_counts()
        for group in self._group_target_amount.keys():
            if self._group_target_amount[str(group)] < value_counts[str(group)]:
                self._target_under[str(group)] = self._group_target_amount[str(group)]
            elif self._group_target_amount[str(group)] > value_counts[str(group)]:
                self._target_over[str(group)] = self._group_target_amount[str(group)]
            elif value_counts[str(group)] < self.MIMIMUM_SAMPLES:
                self._target_init[str(group)] = self.MIMIMUM_SAMPLES
        self._categorical_features = [self._data.columns.get_loc(col) for col in list(self._data.select_dtypes(include=['category', object]).columns)]
        self._fitted = True

    def calculate_weights(self, positive_predicted:float, fairness_scores:pd.Series):
        self._weights = pd.Series()
        for group in fairness_scores.index:
            rate = fairness_scores.loc[str(group)].values
            if len(rate)>1:
                tpr, tnr = rate
                if not np.isnan(tpr) and not float(tpr) == 0:
                    self._weights.loc[str(group)] = float(positive_predicted / tpr)
                elif not np.isnan(tnr) and not float(tnr) == 0:
                    self._weights.loc[str(group)] = float((1-positive_predicted) / tnr)
            elif not float(rate) == 0:
                self._weights.loc[str(group)] = float(positive_predicted / rate)
        if self._strategy=='combined':
            comp = np.median(self._weights)
        elif self._strategy =='over':
            comp = np.min(self._weights)
        elif self._strategy == 'under':
            comp = np.max(self._weights)
        self._weights = self._weights.apply(lambda x: x/comp)
            

    def weights_to_number(self):
        self._group_target_amount = dict()
        value_counts = self._target.value_counts()
        for group in self._weights.index:
            self._group_target_amount[str(group)] = int(np.ceil(value_counts[str(group)] * self._weights[str(group)]))

    def resample(self):
        if not self._strategy in ['combined', 'over', 'under']:
            raise Exception("invalid strategy " + self._strategy)
        if self._strategy in ['combined', 'over']:
            if len(self._target_init) > 0:
                init = RandomOverSampler(sampling_strategy=self._target_init)
                print('Initial RandomOversampling started')
                self._data, self._target = init.fit_resample(self._data, self._target)
                print('Initial RandomOversampling finished')
            over = BorderlineSMOTE(sampling_strategy=self._target_over)
            print("Oversampling started")
            self._data, self._target = over.fit_resample(self._data, self._target)
            print("Oversampling finished")
        if self._strategy in ['combined', 'under']:
            print("Undersampling started")
            resampler = NearMiss(version=1, sampling_strategy=self._target_under)
            self._data, self._target = resampler.fit_resample(self._data, self._target)
            print("Undersampling finished")
        return self._data.iloc[:, :-1], self._data.iloc[:, -1]
