#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:30:20 2020

@author: jwiesner
"""
import numpy as np
import pandas as pd
from skplot import extraction

import re

def validate_measures(measures,measure_labels):
    if measure_labels is not None and measures is None:
        
        # FIXME: This should be changed to a warning. If measures is not given,
        # all measures will be used and measure labels will be ignored
        raise ValueError('If you provide measure labels argument, you must provide measures, too')
    
    if isinstance(measures,list) and measure_labels is not None:
        if not isinstance(measure_labels,list):
            raise TypeError('If measures is provided as list, you must provide measure labels as list, too')
        if not len(measures) == len(measure_labels):
            raise ValueError('List of measures and list of measure labels must be of the same length')
    
    elif isinstance(measures,str) and measure_labels is not None:
        if not isinstance(measure_labels,str):
            raise TypeError('If measures is provided as single string, you must provide measure labels as single string, too')
            
# get the keys of each iterator and the list of combinations for the
# iterators elements.
def get_keys_and_combos(iterator_dict):
    iterator_keys = list(iterator_dict.keys())
    iterator_combinations = extraction.get_iterator_combinations(iterator_dict)

    return iterator_keys,iterator_combinations

def get_number_of_iterators(iterator_dict):
    return len(iterator_dict)

# FIXME: get_model_titles not really necessary, model_title
# could also be created from iterator columns
# use this approach: https://stackoverflow.com/a/54298586/8792159
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

# FIXME: For measure labels, why not just rename measures to measure_labels?
def get_plot_df(plot_data,measures=None,measure_labels=None,iterator_dict=None):
    
    validate_measures(measures,measure_labels)
    
    if iterator_dict:
        iterator_keys,iterator_combinations = get_keys_and_combos(iterator_dict)
    
    # FIXME: Shorten this, use isin or ~isin
    # if measures is not provided, treat all columns except fold and model 
    # number columns as measure columns
    if not measures:
        measures = [column_name for column_name in plot_data.columns.values.tolist() if column_name != 'outer_fold' and column_name != 'model_number']
    
    # melt plot df
    plot_df = pd.melt(plot_data,
                           id_vars=['outer_fold','model_number'],
                           value_vars=measures,
                           var_name='measure_name',
                           value_name='measure_value')
    
    # convert measure name column to categorical
    if isinstance(measures,list):
        plot_df['measure_name'] = pd.Categorical(plot_df['measure_name'],categories=measures,ordered=True)
        
    # optional: add measure labels column
    if measure_labels:
        if isinstance(measure_labels,list):
            plot_df['measure_label'] = [measure_labels[measure_name] for measure_name in plot_df['measure_name'].cat.codes]
        
        elif isinstance(measure_labels,str):
            plot_df['measure_label'] = measure_labels
    
    # add iterator columns
    if iterator_dict:
        
        # FIXME: any way to make this better?
        iterator_df = pd.DataFrame(iterator_combinations,columns=iterator_keys)
        plot_df.set_index('model_number', inplace=True)
        plot_df = plot_df.merge(iterator_df, left_index = True, right_index=True)
        plot_df = plot_df.reset_index().rename(columns={'index':'model_number'})
    
        # add model title column
        # FIXME: get_model_titles not really necessary here, model_title
        # could also be created from iterator columns as create above
        # use this approach: https://stackoverflow.com/a/54298586/8792159
        model_titles = get_model_titles(iterator_keys,iterator_combinations)
        plot_df['model_name'] = [model_titles[model_number] for model_number in plot_df['model_number']]
        plot_df['model_name'] = pd.Categorical(plot_df['model_name'],categories=model_titles,ordered=True)
        
    # convert model number to categorical
    # FIXME: if possible, this should be done already in extraction module
    plot_df['model_number'] = pd.Categorical(plot_df['model_number'],ordered=True)
    
    return plot_df

def get_number_of_models(plot_df):
    n_models = plot_df['model_number'].value_counts().count()
    return n_models

def get_number_of_measures(plot_df):
    n_measures = plot_df['measure_name'].value_counts().count()
    return n_measures

# FIXME: might be redudant if variable type is already set in an earlier step
def remove_hyperparameters(plot_df):

    # get list of names of all hyperparameters (search for categories that contain '__')
    measure_names = plot_df['measure_name'].cat.categories.to_list()
    hyper_param_names = [measure_name for measure_name in measure_names if re.match('^.+__.+$',measure_name)]
    
    # get list of names of all measures except hyperparameters
    measure_names_ = [measure_name for measure_name in measure_names if measure_name not in hyper_param_names]
    
    # subset to data frame without hyperparameters
    # drop rows in dataframe and drop all hyperparameter categories
    plot_df_ = plot_df.loc[~plot_df['measure_name'].isin(hyper_param_names),:]
    plot_df_.loc[:,'measure_name'] = pd.Categorical(plot_df_['measure_name'],categories=measure_names_,ordered=True)
    
    return plot_df_

# get descriptive statistics for all continous measures
# TO-DO: This should be also be callable from report module
def get_continuous_stats_df(plot_df):
    
    plot_df = remove_hyperparameters(plot_df)
    
    continous_stats_df = plot_df.groupby(['measure_name','model_number'])['measure_value'].agg(['mean','median','std']).reset_index()
    continous_stats_df = continous_stats_df.sort_values(['model_number','measure_name']).reset_index(drop=True)
    
    if 'model_name' in plot_df.columns:
        model_numbers = continous_stats_df['model_number'].cat.categories
        model_names = plot_df['model_name'].cat.categories
        mapped_categories = dict(zip(model_numbers,model_names))
        continous_stats_df['model_name'] = continous_stats_df['model_number'].cat.codes.map(mapped_categories)

    return continous_stats_df

# add a new column that indicates which scores are train scores and which are test scores
# Doesn't this belong to plotting module?
def add_score_type(plot_df,train_scores):
    
    if not train_scores:
        raise ValueError('If separate_scores == True, train_scores must be provided')
    
    train_scores_set = set(train_scores)
    scores_set = set(plot_df['measure_name'].cat.categories)
    
    if not train_scores_set.issubset(scores_set):
        raise ValueError('all elements of train_scores must appear as score type in plot df')
    
    plot_df['score'] = np.where(plot_df['measure_name'].isin(train_scores),'train_score','test_score')
    
    return plot_df

if __name__ == '__main__':
    pass