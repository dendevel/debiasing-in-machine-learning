import pandas as pd
from sklearn.model_selection import train_test_split
import copy

def concat_dataframes(*dataframes, axis:int or str='columns') -> pd.DataFrame:
    for df in dataframes:
        df.reset_index(drop=True, inplace=True)
    return pd.concat([*dataframes], axis=axis)


class DataSet():
    _name: str
    _data: pd.DataFrame
    _label: pd.Series
    _sensitive_features: pd.DataFrame
    _sensitive_feature_names: [str]
    _groups: pd.Series
    _correlation: pd.DataFrame
    _prefix_sep: str
    _include_label: bool

    _encoded: bool
    _scaled: bool

    def __init__(self, X:pd.DataFrame, y:pd.Series, sensitive_feature_names:[str]=[], name:str='untitled', encoded:bool=False, scaled:bool=False, prefix_sep:str or None = None, include_label:bool=True) -> 'DataSet':
        self.set_x(X)
        self.set_y(y)
        self.set_scaled(scaled)
        self.set_encoded(encoded)
        self.set_prefix_sep(prefix_sep)
        self._include_label = include_label
        self.set_sensitive_feature_names(sensitive_feature_names)
        self.set_sensitive_features()
        self.set_name(name)

    def copy(self):
        return copy.deepcopy(self)

    def split_train_test(self, test_ratio:float=0.33, shuffle:bool=True):
        X_train, X_test, y_train, y_test = train_test_split(self.get_x(), self.get_y(), test_size=test_ratio, shuffle=shuffle)
        train_data = DataSet(X_train, y_train, sensitive_feature_names=self.get_sensitive_feature_names(), encoded=self.get_encoded(), scaled=self.get_scaled(), prefix_sep=self.get_prefix_sep(), include_label=self._include_label)
        test_data = DataSet(X_test, y_test, sensitive_feature_names=self.get_sensitive_feature_names(), encoded=self.get_encoded(), scaled=self.get_scaled(),prefix_sep=self.get_prefix_sep(),  include_label=self._include_label)
        return train_data, test_data

    def drop_feature(self, feature:str):
        columns = self.get_x().filter(regex='^'+ feature).columns
        self.get_x().drop(columns=columns, inplace=True)
        sensitive_features = self.get_sensitive_feature_names()
        for c in columns:
            sensitive_features.remove(c)
        self.set_sensitive_feature_names(sensitive_features)

    def to_dataframe(self, include_groups:bool=False) -> pd.DataFrame:
        if include_groups:
            return concat_dataframes(self.get_x(), self.get_y(), self.get_groups(), axis='columns')
        else:
            return concat_dataframes(self.get_x(), self.get_y(), axis='columns')

    def evaluate_correlation(self, method:str='pearson', features:[str]=[], threshold:float=0.1):
        if not self.get_encoded():
            raise Exception('Data has to be encoded before correlation evaluation.')
        features = features if len(features) > 0 else self.get_sensitive_feature_names().copy()
        self._correlation = self.get_x().corr(method=method)
        features = [col for col in self.get_x().columns if col.startswith(tuple(features))]
        result = self._correlation.filter(regex='^('+'|'.join(features)+')').copy()
        result.drop(index=features, inplace=True)
        result = result[abs(result)>threshold]
        result_by_feature = result.groupby(axis=1, by=lambda x: str(x).split(self._prefix_sep)[0])
        return result_by_feature

# ------------------------------- getter/setter ------------------------------ #

    def get_x(self, feature:str or None=None) -> pd.DataFrame or pd.Series:
        if feature:
            if self.get_encoded():
                return self._data.filter(regex='^'+feature)
            else:
                return self._data[feature]
        else:
            return self._data

    def set_x(self, X) -> None:
        self._data = X
    
    def get_y(self) -> pd.DataFrame:
        return self._label

    def set_y(self, y) -> None:
        self._label = y

    def get_sensitive_feature_names(self) -> [str]:
        sensitive_feaures_names = self._sensitive_feature_names.copy()
        if self.get_y().name in sensitive_feaures_names:
            sensitive_feaures_names.remove(self.get_y().name)
        return sensitive_feaures_names

    def set_sensitive_feature_names(self, sensitive_feature_names:[str], rebuild_groups:bool=False) -> None:
        if self._include_label:
            sensitive_feature_names.append(self.get_y().name)
        self._sensitive_feature_names = sensitive_feature_names
        self.set_sensitive_features()
        self.create_groups()

    def get_sensitive_features(self) -> pd.DataFrame:
        return self._sensitive_features

    def set_sensitive_features(self) -> None:
        if len(self.get_sensitive_feature_names())>0:
            self._sensitive_features = (self.to_dataframe().filter(regex='^('+'|'.join(self._sensitive_feature_names)+')')).copy()
        else: self._sensitive_features = pd.DataFrame()

    def get_groups(self):
        return self._groups

    def create_groups(self):
        if len(self._sensitive_feature_names) > 0:
            groups = []
            if self.get_encoded():
                for _, row in self._sensitive_features.iterrows():
                        values = []
                        for feature in self.get_sensitive_feature_names():
                            if row[feature] == 1:
                                values.append(feature.split(self.get_prefix_sep())[-1])
                        if self._include_label:
                            values.append(str(row[self.get_y().name]))
                        groups.append('+'.join(values))
            else:
                for _, row in self._sensitive_features.iterrows():
                        values = []
                        for feature in self._sensitive_feature_names:
                            values.append(str(row[feature]))
                        groups.append('+'.join(str(values)))

            self._groups = pd.Series(groups, name="groups")
        else:
            self._groups = pd.Series()


    def get_encoded(self) -> bool:
        return self._encoded
    
    def set_encoded(self, encoded:bool=True) -> None:
        self._encoded = encoded

    def get_scaled(self) -> bool:
        return self._scaled
    
    def set_scaled(self, standardized:bool=True) -> None:
        self._scaled = standardized

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    def set_prefix_sep(self, prefix_sep):
        self._prefix_sep = prefix_sep

    def get_prefix_sep(self):
        return self._prefix_sep
