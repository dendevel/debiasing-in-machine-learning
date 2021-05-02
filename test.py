#%%
from sklearn.datasets import fetch_openml
from fairml.metrics import EqualOpportunity
from sklearn.svm import SVC
from fairml.fairml import FairML
# %% read data
adult = fetch_openml(name='adult')
X = adult.data
y = adult.target
# %%
fairml = FairML(X, y, EqualOpportunity(), ['race'])
# %%
fairml.encode_and_scale()
# %%
fairml.set_evaluator(SVC(kernel='linear'))
#%%
fairml.evaluate_fairness(plot=True, show_description=True)
# %%
fairml = fairml.fit_resample(strategy='combined', inplace=True)
# %%
fairml.evaluate_fairness(plot=True, show_description=True, compare=True)
# %%
