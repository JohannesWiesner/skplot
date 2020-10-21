import pandas as pd
import itertools as it
import warnings

def validate_measures(measures,measure_labels):
    
    if measure_labels is not None and measures is None:
        warnings.warn('Measure labels argument provided but not measures. Measure Labels will be ignored.')
    
    if isinstance(measures,list) and measure_labels is not None:
        if not isinstance(measure_labels,list):
            raise TypeError('If measures is provided as list, you must provide measure labels as list, too')
        if not len(measures) == len(measure_labels):
            raise ValueError('List of measures and list of measure labels must be of the same length')
    
    elif isinstance(measures,str) and measure_labels is not None:
        if not isinstance(measure_labels,str):
            raise TypeError('If measures is provided as single string, you must provide measure labels as single string, too')

def check_scale_type_dict(scale_type_dict):

    valid_scale_types = set(['nominal','ordinal','interval','ratio'])
    scale_type_dict_values = set(scale_type_dict.values())
    
    if not scale_type_dict_values.issubset(valid_scale_types):
        raise ValueError('scale type dict contains one or more invalid scale types. Allowed scale types are: nominal, ordinal, interval, ratio')

def check_for_scale_type(plot_df):
    if not 'scale_type' in plot_df.columns:
        raise ValueError('input plot df must contain a column scale_type')

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
    
    iterator_dict_keys = iterator_dict.keys()
    iterator_combinations = it.product(*(iterator_dict[key] for key in iterator_dict_keys))
    iterator_combinations = list(iterator_combinations)
    
    return iterator_combinations

def get_keys_and_combos(iterator_dict):

    iterator_keys = list(iterator_dict.keys())
    iterator_combinations = get_iterator_combinations(iterator_dict)

    return iterator_keys,iterator_combinations

def get_number_of_iterators(iterator_dict):
    return len(iterator_dict)

def get_model_titles(iterator_keys,iterator_combinations):
    
    model_titles = []
    
    for combo in iterator_combinations:
        
        title = []
        
        for value,name in zip(combo,iterator_keys):
            description = name + ': ' + str(value)
            title.append(description) 
        
        title = ' & '.join(title)
        model_titles.append(title)
    
    return model_titles

def get_plot_df(plot_data,measures=None,measure_labels=None,iterator_dict=None,
                scale_type_dict=None):
    
    '''Prepare extracted measures for plotting.
    
    Parameters
    ----------
    plot_data: pd.DataFrame
        A dataframe provided from skplot.extraction.get_measures_df()
    
    measures: list
        A list of strings defining the names of the measures that where
        extracted using skplot.extraction.get_measures_df()
    
    measure_labels: list
        A list of strings defining the names of the measures as they should 
        be plotted
     
    scale_type_dict: dict
        A dictionary where the keys are the measures and the values are one 
        of 'nominal','ordinal','interval','ratio'. This dictionary defines the
        type of scale for each measure.
    
    iterator_dict: dict of iterators or None (default = None)
        If iterator_dict is `None`, it is assumed that a single model-building-procedure
        was run.
        
        If iterator_dict has one item, it is assumed that different
        model-building procedures where running within a single loop. 
        
        If iterator_dict contains multiple items, it is assumed that different
        model-building procedures where running within nested-loops.
    
    Returns
    -------
    plot_df: pd.DataFrame
        A dataframe that holds the extracted information in a plot-ready
        format.
    
    '''
    
    # validate input
    validate_measures(measures,measure_labels)
    
    # if measures is not provided, treat all columns except fold and model 
    # number columns as measure columns
    if not measures:
        measures = [col_name for col_name in plot_data.columns if col_name not in ['model_number','outer_fold']]
    
    # melt plot df
    plot_df = pd.melt(plot_data,
                      id_vars=['outer_fold','model_number'],
                      value_vars=measures,
                      var_name='measure_name',
                      value_name='measure_value')
    

    if scale_type_dict:
        check_scale_type_dict(scale_type_dict)
        plot_df['scale_type'] = plot_df['measure_name'].map(scale_type_dict)
    
    # Add measure labels column
    if measure_labels:
        
        if isinstance(measure_labels,list):
            mapping_dict = dict(zip(measures,measure_labels))
            plot_df['measure_label'] = plot_df['measure_name'].map(mapping_dict)
        
        elif isinstance(measure_labels,str):
            plot_df['measure_label'] = measure_labels
    
    # Add model infos
    if iterator_dict:
        
        iterator_keys,iterator_combinations = get_keys_and_combos(iterator_dict)
        
        # add iterator columns
        iterator_df = pd.DataFrame(iterator_combinations,columns=iterator_keys)
        plot_df = plot_df.merge(iterator_df,left_on='model_number',right_index=True)
    
        # add model title column
        model_titles = get_model_titles(iterator_keys,iterator_combinations)
        plot_df['model_name'] = [model_titles[model_number] for model_number in plot_df['model_number']]

    return plot_df

def get_continuous_stats_df(plot_df):
    
    check_for_scale_type(plot_df)
    
    groupby_vars = ['measure_name','model_number']
    
    if 'model_name' in plot_df.columns:
        groupby_vars.append('model_name')

    continuous_vars_df = plot_df.loc[plot_df['scale_type'] == 'interval']
    continuous_stats_df = continuous_vars_df.groupby(groupby_vars)['measure_value'].agg(['mean','median','std']).reset_index()
    continuous_stats_df = continuous_stats_df.sort_values(['model_number','measure_name']).reset_index(drop=True)
    
    return continuous_stats_df

def get_categorical_stats_df(plot_df):
    
    check_for_scale_type(plot_df)
    
    groupby_vars = ['measure_name','model_number','measure_value']
    
    if 'model_name' in plot_df.columns:
        groupby_vars.append('model_name')
    
    categorical_vars_df = plot_df.loc[plot_df['scale_type'] == 'nominal']
    categorical_stats_df = categorical_vars_df.groupby(groupby_vars)['measure_value'].count()
    categorical_stats_df = categorical_stats_df.rename('count')
    categorical_stats_df = categorical_stats_df.reset_index()
    
    return categorical_stats_df
    
if __name__ == '__main__':
    pass