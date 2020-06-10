import numpy as np
import pandas as pd
import itertools as it
import warnings

# FIXME: this function is not really necessary here, except for validation
# in get_plot_df (get_plot_df doesn't need this function anymore). Transfer
# this function over to preparation module where it is needed. For user 
# validation input, create new validation module which only purpose is
# to validate user input.
def get_iterator_combinations(iterator_dict):
    '''Generate all combinations of elements from a dictionary of iterators.
        
    Takes a dictionary of iterators and calculates the combinations of all the 
    iterators elements.  Note that the order of the items defines the way 
    combinations are  made. It is assumed that the first iterator is the 
    `main-loop` and all following iterators are nested within that loop.
    
    See also:
    https://www.hackerrank.com/challenges/itertools-product/problem
    
    Parameters
    ----------
    iterator_dict: dict of iterators
        A dictionary of iterators.

    Returns
    -------
    iterator_combinations: list
        A list of tuples, where each tuple represents a combination of the 
        iterators elements.
    '''
    
    # get list of combinations
    iterator_dict_keys = iterator_dict.keys()
    iterator_combinations = it.product(*(iterator_dict[key] for key in iterator_dict_keys))
    iterator_combinations = list(iterator_combinations)
    
    return iterator_combinations

def validate_input(scores,iterator_dict):
    if isinstance(scores,dict):
        if iterator_dict != None:
            warnings.warn('You provided an iterator dict but only one output dictionary. Iterator dict will be ignored.',UserWarning)
    
    elif isinstance(scores,list):
        if iterator_dict:
            iterator_combinations = get_iterator_combinations(iterator_dict=iterator_dict)
            if len(iterator_combinations) != len(scores):
                raise ValueError('Number of provided score dictionaries and number of iterator combinations must be equal')

# TO-DO: (Make explicit): Add column with type of extracted score,
# e.g. type = 'train_or_test_score','hyperparameter', etc.
# This makes filtering at later stages easier.
# currently, type is found out by using regex functions in preparation modules
# TO-DO: Allow to pass function to extract custom information
def extract_single_plot_data(scores,data_kind='train_and_test_scores'):
    '''Extract information from a single dict as provided from 
    sklearn.model_selection.cross_validate
    
    Parameters
    ----------
    scores: dict of float arrays of shape=(n_splits,)
        A single dictionary as returned from `cross_validate`.
        
    data_kind: str or list of str(default='train_and_test_scores')
        Use 'train_and_test_scores' if you want to extract training and test
        scores for each outer fold.
        
        Use 'n_coefs' if you want to get the number of coefficients 
        for for each outer fold.
        
        Any other string will be interpreted as hyperparameter.
        
        Use a list of strings if you want to extract multiple information.
        
    Returns
    -------
    single_plot_data: pd.DataFrame
        a DataFrame containing the specified information for each outer fold.
    
    See also
    ----
    See also: `sklearn.model_selection.cross_validate <https://scikit-learn.org/
    stable/modules/generated/sklearn.model_selection.cross_validate.html>`_
    '''
    
    # if data_kind is single string, convert to list of single string
    if isinstance(data_kind,str):
        data_kind_ = [data_kind]
    elif isinstance(data_kind,list):
        data_kind_ = data_kind
    
    # create an empty Data Frame
    plot_dict = {}
    
    # get number of outer folds
    n_folds = len(scores['estimator'])
    folds = np.linspace(start=1,stop=n_folds,num=n_folds,dtype= int)
    plot_dict['outer_fold'] = folds
    
    for kind in data_kind_:
        
        if kind == 'train_and_test_scores':
            
            train_and_test_scores = {k: v for k, v in scores.items() if k != 'estimator' and k != 'fit_time' and k != 'score_time'}
            plot_dict.update(train_and_test_scores)
        
        elif kind == 'n_coefs':
            
            n_coefs = []
        
            for idx in range(0,n_folds):
                n_coefs_fold = scores['estimator'][idx].best_estimator_._final_estimator.coef_.size
                n_coefs.append(n_coefs_fold)
    
            plot_dict['n_coefs'] = n_coefs
            
        elif kind == 'sum_explained_variance':
            
            sum_explained_variance = []
            
            for idx in range(0,n_folds):
                sum_explained_variance_fold = np.sum(scores['estimator'][idx].best_estimator_.steps[1][1].explained_variance_ratio_)
                sum_explained_variance.append(sum_explained_variance_fold)
    
            plot_dict['sum_explained_variance'] = sum_explained_variance
        
        else:
            
            hyper_params = []
    
            for idx in range(0,n_folds):
                hyper_param = scores['estimator'][idx].best_params_[kind]
                hyper_params.append(hyper_param)
    
            plot_dict[kind] = hyper_params
        
    single_plot_data = pd.DataFrame(plot_dict)
        
    return single_plot_data

# FIXME: Iterator dict is not necessary here. See FIXMES above.
# FIXME: Make model_number categorical and ordered. Right now 
# it is unordered which creates problems in later functions such
# as preparation.get_continuous_stats_df
def extract_plot_data(scores,iterator_dict=None,data_kind='train_and_test_scores'):
    '''Extract plottable information from one or multiple output dicts as 
    provided from sklearn.model_selection.cross_validate
    
    Parameters
    ----------
    scores: dict or list of dicts
        One or multiple dictionaries as returned from `sklearn.model_selection.cross_validate`.
    
    iterator_dict: dict of iterators or None (default = None)
        If iterator_dict is `None`, it is assumed that a single model-building-procedure
        was run.
        
        If iterator_dict has one item, it is assumed that different
        model-building procedures where running within a single loop. 
        
        If iterator_dict contains multiple items, it is assumed that different
        model-building procedures where running within nested-loops.
        
    data_kind: str (default = 'train_and_test_scores')
        See documentation of ``skplot.extract_single_plot_data``
    
    Returns
    -------
    plot_data: pd.DataFrame()
        A dataframe which containing extracted information.
    
    
    '''
    
    validate_input(scores=scores,iterator_dict=iterator_dict)

    plot_data = pd.DataFrame()
    
    if isinstance(scores,dict):
        measures_df = extract_single_plot_data(scores,data_kind=data_kind)
        measures_df['model_number'] = 0
        plot_data = plot_data.append(measures_df)

    elif isinstance(scores,list):
        for idx,score_dict in enumerate(scores):
            measures_df = extract_single_plot_data(score_dict,data_kind=data_kind)
            measures_df['model_number'] = idx
            plot_data = plot_data.append(measures_df)

    return plot_data

if __name__ == '__main__':
    pass