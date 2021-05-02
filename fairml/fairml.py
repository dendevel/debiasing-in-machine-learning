from fairml.datasets import DataSet
from fairml.preprocesing import Scaler, Encoder
from fairml.evaluation import Evaluator
from fairml.resampling import Resampler
import fairml.metrics as metrics
import pandas as pd
import copy


class FairML():
    _train_data: DataSet
    _test_data: DataSet
    _last_fairness_evaluation: pd.DataFrame
    _fairness_metric: metrics.FairnessMetric
    
    _encoder: Encoder
    _scaler: Scaler
    _evaluator: Evaluator
    _resampler: Resampler


    def __init__(self, X, y, fairness_metric: metrics.FairnessMetric, sensitive_feature_names=[]):
        self._train_data = DataSet(X, y, sensitive_feature_names=sensitive_feature_names, include_label = fairness_metric.get_includes_label())
        self._encoder = Encoder()
        self._scaler = Scaler()
        self._resampler = Resampler()
        self._fairness_metric = fairness_metric

# ------------------------------- PREPROCESSING ------------------------------ #

    def check_for_missing_values(self):
        return self._train_data.get_x().isna().sum()

    def encode_and_scale(self):
        if not hasattr(self, '_scaler'):
            print("You have to set a scaler first.")
            return
        if not self._train_data.get_encoded():
            self._encoder.fit_transform(self._train_data, inplace=True)
            self._scaler.fit_transform(self._train_data, inplace=True)
        else:
            print("Data is already encoded.")

    def decode_and_descale(self):
        if self._train_data.get_encoded():
            self._scaler.inverse_transform(self._train_data, inplace=True)
            self._encoder.inverse_transform(self._train_data, inplace=True)
        else:
            print("Data is already decoded.")

# -------------------------------- EVALUATION -------------------------------- #

    def evaluate_bias(self):
        return self._evaluator.bias(self.get_train_data())

    def evaluate_fairness(self, threshold:float=0.8, plot:bool=True, compare:bool=False, show_description:bool=False):
        if not hasattr(self, '_evaluator'): 
            print("You have to set an evaluator first.")
            return
        if not hasattr(self, '_test_data'):
            self._train_data, self._test_data = self._train_data.split_train_test(shuffle=False)

        self._evaluator.fit_predict(self._train_data, self._test_data, self._fairness_metric)
       
        if plot:
            if compare:
                if hasattr(self, '_last_fairness_evaluation'):
                    compare_fairness = pd.concat([self._last_fairness_evaluation, self._evaluator._fairness_scores], axis = 1)
                    compare_fairness.columns = ['old', 'new']
                    metrics.visualize_fairness(compare_fairness.copy(), title=self._fairness_metric.get_name(), show_description=False)
                else:
                    print("Nothing to compare yet.")
                    metrics.visualize_fairness(self._evaluator._fairness_scores.copy(), title=self._fairness_metric.get_name(), show_description=show_description)
            else:
                metrics.visualize_fairness(self._evaluator._fairness_scores.copy(), title=self._fairness_metric.get_name(), show_description=show_description)
        self._last_fairness_evaluation = self._evaluator._fairness_scores
        return self._evaluator.fairness()
        

# --------------------------------- DEBIASING -------------------------------- #

    def drop_sensitive(self, correlation_threshold:float=0.1):
        dropped=[]
        correlation_by_feature = self.get_train_data().evaluate_correlation(threshold=correlation_threshold)
        for feature in correlation_by_feature:
            if feature[1].isnull().all().all():
                self.get_train_data().drop_feature(feature[0])
                if hasattr(self, '_test_data'):
                    self.get_test_data().drop_feature(feature[0])
                dropped.append(feature[0])
        if len(dropped)>0:
            print("The following features were dropped: " + str(dropped))
        else:
            print("No features were dropped.")

    def fit(self, strategy:str="combined"):
        self._resampler.set_strategy(strategy)
        self._resampler.fit(self._train_data, self._evaluator._positive_predicted_probability, self._evaluator._fairness_scores)
        
    def resample(self, inplace:bool=False):
        data_res, label_res = self._resampler.resample()

        if inplace:
            self._train_data.set_x(data_res)
            self._train_data.set_y(label_res)
            self._train_data.set_sensitive_feature_names(self._train_data.get_sensitive_feature_names())
            return self
        else:
            self_copy = copy.deepcopy(self) 
            self_copy.get_train_data().set_x(data_res)
            self_copy.get_train_data().set_y(label_res)
            self_copy.get_train_data().set_sensitive_feature_names(self.get_train_data().get_sensitive_feature_names())
            return self_copy

    def fit_resample(self, strategy:str="combined", inplace:bool = False):
        self.fit(strategy=strategy)
        return self.resample(inplace=inplace)
        

# ----------------------------------- UTILS ---------------------------------- #

    def describe(self):
        description = pd.Series(name='description of '+self._train_data._name)
        description['name'] = self._train_data._name
        description['rows'] = self._train_data._data.shape[0]
        description['columns'] = self._train_data._data.shape[1]
        description['all features'] = list(self._train_data._data.columns)
        description['of which numeric'] = list(self._train_data._data.select_dtypes(include=['number']).columns)
        description['of which categorical'] = list(self._train_data._data.select_dtypes(include=['category', object]).columns)
        description['of which set as sensitive'] = self._train_data.get_sensitive_feature_names()
        description['classes'] = list(self._train_data._label.unique())
        description['encoded'] = self._train_data._encoded
        description['standardized'] = self._train_data._scaled
        description['group_bias'] = 0 #TODO

        return description

# ------------------------------- getter/setter ------------------------------ #
        
    def set_sensitive_features(self, sensitive_feature_names):
        if self._train_data.get_encoded():
            print("Sensitive features can be set only if data is not encoded. Please decode and try again.")
            return
        self._train_data.set_sensitive_feature_names(sensitive_feature_names, rebuild_groups=True)
    
    def set_scaler(self, scaler):
        self._scaler = Scaler(scaler)

    def set_evaluator(self, predictor):
        self._evaluator = Evaluator(predictor)

    def set_resampler(self, resampler):
        self._resampler = Resampler()

    def get_train_data(self):
        return self._train_data

    def get_test_data(self):
        if hasattr(self, '_test_data'):
            return self._test_data
        else:
            raise Exception("No test data set yet.")


        
    



