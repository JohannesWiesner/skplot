import sys
sys.path.append('C:/Users/Johannes.Wiesner/Documents/repos/skplot/')

from skplot import extraction
from skplot import preparation
from skplot import plotting

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

## Set user defined variables #################################################

RANDOM_SEED=42
LR_C=np.linspace(start=1,stop=10,num=5,endpoint=True)
INNER_CV=5
OUTER_CV=5

## Load Data  #################################################################

X,y = datasets.load_breast_cancer(return_X_y=True)

## Run classification pipeline for different solvers and penalty types ########

# The output of each combination of solver and penalty type will be stored in a list
scores_list = []

for n_components in [0.9,0.1]:
    for penalty_type in ['l2','l1']:
    
        # create a random number generator
        rng = np.random.RandomState(RANDOM_SEED)
        
        # z-standardize
        scaler = StandardScaler()
        
        # PCA
        pca = PCA(n_components=n_components,svd_solver='full',random_state=rng)
        
        # use linear L2-regularized Logistic Regression as classifier
        lr = LogisticRegression(penalty=penalty_type,
                                solver='liblinear',
                                random_state=rng,)
        
        # define parameter grid to optimize over
        p_grid = {'lr__C':LR_C}
        
        # create pipeline
        lr_pipe = Pipeline([
                ('scaler',scaler),
                ('pca',pca),
                ('lr',lr)
                ])
        
        # define inner and outer folds
        skf_inner = StratifiedKFold(n_splits=INNER_CV)
        skf_outer = StratifiedKFold(n_splits=INNER_CV)
        
        # implement GridSearch (inner cross validation)
        grid = GridSearchCV(lr_pipe,
                            param_grid=p_grid,
                            cv=skf_inner,
                            )
        
        # implement cross_validate (outer cross validation)
        nested_cv_scores = cross_validate(grid,
                                          X,
                                          y,
                                          cv=skf_outer,
                                          return_train_score=True,
                                          return_estimator=True,
                                          )
        
        scores_list.append(nested_cv_scores)
    
# Extract data ################################################################

# get the coefficient value for each pca component 
def get_intercept_value(scores,idx):
    return scores['estimator'][idx].best_estimator_._final_estimator.intercept_[0]

scores_df = extraction.get_measures_df(scores_list,
                                       data_kind=['lr__C','train_and_test_scores','n_coefs',get_intercept_value])
        
# Prepare data for plotting  ##################################################

iterator_dict = {'pca':[0.9,0.1],
                 'regularization type':['l2','l1']}

scale_type_dict = {'lr__C':'nominal',
                   'n_coefs':'interval',
                   'train_score':'interval',
                   'test_score':'interval'}

plot_df = preparation.get_plot_df(scores_df,
                                  measures=['test_score','train_score','lr__C'],
                                  measure_labels=['Test Score','Train Score','C'],
                                  scale_type_dict=scale_type_dict)

# Plot data ###################################################################

g = plotting.catplot_scores(plot_df,
                            iterator_dict=iterator_dict,
                            separate_measures=True,
                            train_scores='train_score')