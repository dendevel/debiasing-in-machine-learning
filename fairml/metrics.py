from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from scipy.stats import variation
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import textwrap as tw
from abc import ABC, abstractmethod



class MLMetrics():
    def __init__(self, y_pred, y_true):
        self.cm = confusion_matrix(y_true, y_pred)
        if self.cm.shape != (2,2):
            cm = np.array([[0, 0], [0, 0]])
            for kombi in zip(y_true, y_pred):
                cm[kombi]+=1
            self.cm = cm
        tn, fp, fn, tp = self.cm.ravel()
        #accuracy
        if self.cm.sum() == 0: self.accuracy = 0
        else:                  self.accuracy = (tp+tn)/self.cm.sum()
        #precision
        if tp+fp == 0: self.precision = 0
        else:          self.precision = tp/(tp+fp)
        #tpr & fnr
        if tp+fn == 0: self.tpr = self.fnr = 0
        else:
            self.tpr = tp/(tp+fn)
            self.fnr = fn/(tp+fn)
        #tnr & fpr
        if tn+fp == 0: self.tnr = self.fpr = 0
        else:
            self.tnr = tn/(tn+fp)
            self.fpr = fp/(tn+fp)
        #F1
        if self.tpr + self.precision == 0: self.F1 = 0
        else:
            self.F1 = 2*self.tpr*self.precision/(self.tpr + self.precision)


class FairnessMetric(ABC):
    _name: str
    _type: str
    _includes_label: bool

    def get_name(self):
        return self._name

    def get_type(self):
        return self._type

    def get_includes_label(self):
        return self._includes_label

    @abstractmethod
    def run(self, y_per_group:dict):
        pass


# ---------------------------------------------------------------------------- #
#                             distribution metrics                             #
# ---------------------------------------------------------------------------- #

def variation_coefficient(X:pd.DataFrame, sensitive_feature_name:str):
    return variation(X[sensitive_feature_name].value_counts())

# ---------------------------------------------------------------------------- #
#                      metrics based on predicted outcome                      #
# ---------------------------------------------------------------------------- #
def positive_predicted_probability(y_per_group):
    y_pred      = [results['y_pred'] for results in y_per_group.values()]
    y_pred_flat = [element for sublist in y_pred for element in sublist]
    return np.count_nonzero(y_pred_flat)/len(y_pred_flat)


class DemographicParity(FairnessMetric):
    def __init__(self):
        self._name = "Demographic Parity"
        self._type = "Independence"
        self._includes_label = True

    def run(self, y_per_group):
        result = pd.DataFrame(data = {'group': y_per_group.keys()}, columns = ['group', 'DP']).set_index('group')
        for group, results in y_per_group.items():
            if any(results['y_true']):
                result.loc[group]['DP'] = np.count_nonzero(results['y_pred'])/len(results['y_pred'])
        result.name = "demographic_parity"
        return result.dropna(how='all')

# ---------------------------------------------------------------------------- #
#                 metrics based on predicted and actual outcome                #
# ---------------------------------------------------------------------------- #
class EqualOpportunity(FairnessMetric):
    def __init__(self):
        self._name = "Equal Opportunity"
        self._type = "Separation"
        self._includes_label = True

    def run(self, y_per_group):
        result = pd.DataFrame(data = {'group': y_per_group.keys()}, columns = ['group', 'TPR']).set_index('group')
        for group, results in y_per_group.items():
            if any(results['y_true']):
                result.loc[group]['TPR'] = MLMetrics(results['y_pred'], results['y_true']).tpr
        result.name = 'equal_opportunity'
        return result.dropna(how='all')


class PredictiveEquality(FairnessMetric):
    def __init__(self):
        self._name = "Predictive Equality"
        self._type = "Separation"
        self._includes_label = True

    def run(self, y_per_group):
        result = pd.DataFrame(data = {'group': y_per_group.keys()}, columns = ['group', 'TNR']).set_index('group')
        for group, results in y_per_group.items():
            if not any(results['y_true']):
                result.loc[group]['TNR'] = MLMetrics(results['y_pred'], results['y_true']).tnr
        result.name = 'predictive_equality'
        return result.dropna(how='all')


class EqualizedOdds(FairnessMetric):
    def __init__(self):
        self._name = "Equalized Odds"
        self._type = "Separation"
        self._includes_label = True

    def run(self, y_per_group):
        result = pd.DataFrame(data = {'group': y_per_group.keys()}, columns = ['group', 'TPR', 'FPR']).set_index('group')
        for group, results in y_per_group.items():
            if all(results['y_true']):
                result.loc[group]['TPR'] = MLMetrics(results['y_pred'], results['y_true']).tpr
            elif not any(results['y_true']):
                result.loc[group]['FPR'] = MLMetrics(results['y_pred'], results['y_true']).fpr
        result.name = 'equalized_odds'
        return result.dropna(how='all')


class OverallAccuracyEquality(FairnessMetric):
    def __init__(self):
        self._name = "Overall Accuracy Equality"
        self._type = "Separation"
        self._includes_label = True

    def run(self, y_per_group):
        y_per_group_new = dict()
        for group, results in y_per_group.items():
            new_group_name = '+'.join(group.split('+')[:-1])
            for g, res in y_per_group.items():
                new_g_name = '+'.join(g.split('+')[:-1])
                if new_group_name == new_g_name and not group == g:
                    new_y_true = list(results['y_true']) + list(res['y_true'])
                    new_y_pred = list(results['y_pred']) + list(res['y_pred'])
                    y_per_group_new[group] = {'y_true': new_y_true, 'y_pred': new_y_pred}
                    y_per_group_new[g] = {'y_true': new_y_true, 'y_pred': new_y_pred}

        result = pd.DataFrame(data = {'group': y_per_group_new.keys()}, columns = ['group', 'accuracy']).set_index('group')
        for group, results in y_per_group_new.items():
            result.loc[group]['accuracy'] = MLMetrics(results['y_pred'], results['y_true']).accuracy
        result.name = 'overall_accuracy_equality'
        return result.dropna(how='all')


class ConditionalUseAccuracy(FairnessMetric):
    def __init__(self):
        self._name = "Conditional Use Accuracy"
        self._type = "Separation"
        self._includes_label = True

    def run(self, y_per_group):
        result = pd.DataFrame(data = {'group': y_per_group.keys()}, columns = ['group', 'TPR', 'FNR']).set_index('group')
        for group, results in y_per_group.items():
            if any(results['y_true']):
                result.loc[group]['TPR'] = MLMetrics(results['y_pred'], results['y_true']).tpr
            else:
                result.loc[group]['FNR'] = MLMetrics(results['y_pred'], results['y_true']).fnr
        result.name = 'conditional_use_accuracy'
        return result.dropna(how='all')


class TreatmentEquality(FairnessMetric):
    def __init__(self):
        self._name = "Treatment Equality"
        self._type = "Separation"
        self._includes_label = True

    def run(self, y_per_group):
        result = pd.DataFrame(data = {'group': y_per_group.keys()}, columns = ['group', 'FPR/FNR']).set_index('group')
        for group, results in y_per_group.items():
            cm = MLMetrics(results['y_pred'], results['y_true'])
            result.loc[group]['FPR/FNR'] = cm.fpr/cm.fnr
        result.name = 'treatment_equality'
        return result.dropna(how='all')


class EqualizingDisincentives(FairnessMetric):
    def __init__(self):
        self._name = "Equalizing Disincentives"
        self._type = "Separation"
        self._includes_label = True

    def run(self, y_per_group):
        result = pd.DataFrame(data = {'group': y_per_group.keys()}, columns = ['group', 'TPR-FNR']).set_index('group')
        for group, results in y_per_group.items():
            if any(results['y_true']):
                cm = MLMetrics(results['y_pred'], results['y_true'])
                result.loc[group]['TPR-FNR'] = cm.tpr - cm.fpr
        result.name = 'equalizing_disincentives'
        return result.dropna(how='all')

def roc_auc_score(y_per_group):
    fpr= dict()
    tpr = dict()
    roc_auc = dict()
    for group, results in y_per_group.items():
        # if any(results['y_true']):
        fpr[group], tpr[group], _ = roc_curve(results['y_true'], results['y_pred'])
        roc_auc[group] = auc(fpr[group], tpr[group])
        plt.plot(fpr[group], tpr[group], label = group)
    
    plt.title('ROC-Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return roc_auc, tpr, fpr

# ---------------------------------------------------------------------------- #
#                 visualization                                                #
# ---------------------------------------------------------------------------- #

def fairness_title(name): return name.replace("_", " ").title()
        

def visualize_fairness(fairness_df, title='', show_description = False):
    fig, ax = plt.subplots()
    for index in fairness_df.index:
        indices = index.split('+')
        indices.pop(-1)
        fairness_df.rename(index={index: '+'.join(indices)}, inplace=True)
    fairness_df.drop_duplicates(inplace=True)
    ax = fairness_df.plot.bar()
    title = fairness_title(title)
    ax.set_title(title)
    ax.set_ylabel('metric')
    plt.xticks(rotation = 45)
    ax.grid(axis='y')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    if show_description:
        if len(fairness_df.columns) == 1:
            text = '''Der Algorithmus ist fair, wenn für alle Gruppen eine ungefähr gleiche "{0}" besteht.'''
            text = text.format(fairness_df.columns[0])
            y = -0.08
        elif len(fairness_df.columns) == 2:
            text = '''Der Algorithmus ist fair, wenn für alle Gruppen eine ungefähr gleiche "{0}" und eine ungefähr gleiche "{1}" besteht.'''
            text = text.format(fairness_df.columns[0], fairness_df.columns[1])
            y = -0.1
        else:
            raise Exception("Only one and two metrics are supported")
        
        fig_txt = tw.fill(text, width=50)
        plt.figtext(0.5, y, fig_txt, horizontalalignment='center',
                fontsize=10, multialignment='left',
                bbox=dict(boxstyle="round", facecolor='#D8D8D8',
                          ec="0.5", pad=0.5, alpha=1), fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    