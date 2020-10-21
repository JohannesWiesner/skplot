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

RANDOM_SEED = 42
LR_C = np.linspace(start=1,stop=10,num=5,endpoint=True)
INNER_CV = 5
OUTER_CV = 10

## Load Data  #################################################################

X,y = datasets.load_breast_cancer(return_X_y=True)

## Run several classification pipelines with different combinations of input
# parameters 

# The output of each pipeline must be stored in a list
scores_list = []

# Try out for different amounts of explained variances and different regularization
# types
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
        skf_outer = StratifiedKFold(n_splits=OUTER_CV)
        
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

# Create your personal custom function: get the coefficient value for each feature
def get_intercept_value(scores,idx):
    return scores['estimator'][idx].best_estimator_._final_estimator.intercept_[0]

# Extract information for all different pipelines using build-in settings
# and your custom function
scores_df = extraction.get_measures_df(scores_list,
                                       data_kind=['lr__C','train_and_test_scores','n_coefs',get_intercept_value])
        
# Prepare data for plotting  ##################################################

# Set the iterator dict (Note, that the order of keys is important here)
iterator_dict = {'PCA (explained variance)':['90%','10%'],
                 'Regularization Type':['L2','L1']}

# Skplot is dumb. It doesn't know  which scale each of your extracted measure is using,
# so you have to provide this manually 
scale_type_dict = {'lr__C':'nominal',
                   'n_coefs':'interval',
                   'train_score':'interval',
                   'test_score':'interval'}

# Use the preparation module to process your data to a plot-ready format
plot_df = preparation.get_plot_df(plot_data=scores_df,
                                  measures=['test_score','train_score','lr__C'],
                                  measure_labels=['Test Score','Train Score','C'],
                                  iterator_dict=iterator_dict,
                                  scale_type_dict=scale_type_dict)

# Plot data ###################################################################

# Plot as categorical plots
g = plotting.catplot_scores(plot_df,
                            iterator_dict=iterator_dict,
                            separate_measures=True,
                            kind='point',
                            train_scores='train_score')
# Plot as lineplots
g = plotting.lineplot_scores(plot_df,
                             col_wrap=2,
                             x_tick_spacing=1)