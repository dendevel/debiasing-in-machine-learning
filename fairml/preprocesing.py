from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
from fairml.datasets import DataSet
import pandas as pd
import numpy as np
from typing import Union

class Encoder():
    _feature_names: [str]
    _sensitive_feature_names: [str]
    _prefix_sep: str
    _le: LabelEncoder
    
    def __init__(self, prefix_sep:str="=&="):
        self._prefix_sep = prefix_sep
        self._le = LabelEncoder()
    
    def fit_transform(self, data: DataSet, inplace:bool=False):
        if not inplace:
            data = data.copy()
        self._feature_names = data.get_x().columns
        self._sensitive_feature_names = data.get_sensitive_feature_names()
        data.set_x(self.encode_data(data.get_x()))
        data.set_y(self.encode_label(data.get_y()))
        data.set_sensitive_feature_names(self.encode_sensitive_feature_names(data))
        data.set_encoded()
        data.set_prefix_sep(self._prefix_sep)
        return data 

    def inverse_transform(self, data: DataSet, inplace:bool=False):
        if not inplace:
            data = data.copy()
        data.set_x(self.decode_data(data.get_x()))
        data.set_y(self.decode_label(data.get_y()))
        data.set_sensitive_feature_names(self.decode_sensitive_feature_names(data), rebuild_groups=True)
        data.set_encoded(False)
        data.set_prefix_sep(None)
        return data

    def encode_data(self, X:pd.DataFrame):
        return pd.get_dummies(X, drop_first=False, prefix_sep=self._prefix_sep)

    def encode_label(self, y:pd.Series):
        return pd.Series(self._le.fit_transform(y), name=y.name)

    def encode_sensitive_feature_names(self, data: DataSet):
        return [col for col in data.to_dataframe().columns if col.startswith(tuple(self._sensitive_feature_names))]

    def decode_data(self, X_enc:pd.DataFrame):
        return self._from_dummies(X_enc)

    def decode_label(self, y_enc:pd.Series):
        if isinstance(y_enc, pd.Series):
            return pd.Series(self._le.inverse_transform(y_enc), name=y_enc.name)
        else:
            return self._le.inverse_transform(y_enc)

    def decode_sensitive_feature_names(self, data: DataSet):
        return self._sensitive_feature_names

    def _from_dummies(self, X_enc: pd.DataFrame):
        X_dec = pd.DataFrame()
        for feature in self._feature_names:
            if feature in list(X_enc.columns):
                col = X_enc[feature]
            else:
                df = X_enc.filter(regex='^'+feature+self._prefix_sep)
                if df.shape[1] > 1:
                    col = pd.Series(df.columns[np.where(df!=0)[1]], name=feature).str.replace('^.+'+self._prefix_sep, '', regex=True)
                else:
                    col = pd.Series()
            X_dec = pd.concat([X_dec, col], axis=1)
        return X_dec


class Scaler():
    _scaler: Union[MinMaxScaler, MaxAbsScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer]
    def __init__(self, scaler=MaxAbsScaler()):
        self._scaler = scaler

    def fit(self, data: DataSet):
        self._scaler.fit(data.to_dataframe())

    def transform(self, data: DataSet, inplace:bool=False):
        if not inplace:
            data = data.copy()
        data.set_x(pd.DataFrame(self._scaler.transform(data.get_x()), columns=data.get_x().columns))
        data.set_scaled()
        return data

    def fit_transform(self, data: DataSet, inplace:bool=False):
        if not inplace:
            data = data.copy()
        data.set_x(pd.DataFrame(self._scaler.fit_transform(data.get_x()), columns=data.get_x().columns))
        data.set_scaled()
        return data

    def inverse_transform(self, data: DataSet, inplace:bool=False):
        if not inplace:
            data = data.copy()
        data.set_x(pd.DataFrame(self._scaler.inverse_transform(data.get_x()), columns=data.get_x().columns))
        data.set_scaled(False)
        return data