import numpy as np
import pandas as pd

def extract_measure(scores,data_kind='train_and_test_scores'):
    '''Extract information from a single dict as provided from 
    sklearn.model_selection.cross_validate
    
    Parameters
    ----------
    scores: dict of float arrays of shape=(n_splits,)
        A single dictionary as returned from `cross_validate`.
        
    data_kind: str or list of str(default='train_and_test_scores')
    
        Use a list of strings if you want to extract multiple information.
    
        Use 'train_and_test_scores' if you want to extract training and test
        scores for each outer fold.
        
        Use 'n_coefs' if you want to get the number of coefficients for each outer fold.
        
        Any other string will be interpreted as hyperparameter.
        

        
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
        
        elif callable(kind):
            
            returned_values = []
            
            for idx in range(0,n_folds):
                returned_values.append(kind(scores,idx))
            
            plot_dict[kind.__name__] = returned_values
                
        else:
            
            hyper_params = []
    
            for idx in range(0,n_folds):
                hyper_param = scores['estimator'][idx].best_params_[kind]
                hyper_params.append(hyper_param)
    
            plot_dict[kind] = hyper_params
        
    single_plot_data = pd.DataFrame(plot_dict)
        
    return single_plot_data

def get_measures_df(scores,data_kind='train_and_test_scores',data_types=None):
    '''Extract plottable information from one or multiple output dicts as 
    provided from sklearn.model_selection.cross_validate
    
    Parameters
    ----------
    scores: dict or list of dicts
        One or multiple dictionaries as returned from `sklearn.model_selection.cross_validate`.
    
    data_kind: str (default = 'train_and_test_scores')
        See documentation of ``skplot.extract_measure``
    
    Returns
    -------
    plot_data: pd.DataFrame()
        A dataframe which containing extracted information.
    '''
    
    plot_data = pd.DataFrame()
    
    if isinstance(scores,dict):
        measures_df = extract_measure(scores,data_kind=data_kind)
        measures_df['model_number'] = 0
        plot_data = plot_data.append(measures_df)

    elif isinstance(scores,list):
        for idx,score_dict in enumerate(scores):
            measures_df = extract_measure(score_dict,data_kind=data_kind)
            measures_df['model_number'] = idx
            plot_data = plot_data.append(measures_df)
    
    return plot_data

if __name__ == '__main__':
    pass