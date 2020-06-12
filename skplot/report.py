import pandas as pd
from skplot import preparation

def check_plot_df(plot_df):
    
    if 'scale_type' not in plot_df.columns:
        raise ValueError('input plot_df must contain a column scale_type')

def get_categorical_stats_report(plot_df):
    
    check_plot_df(plot_df)
    categorical_stats_df = preparation.get_categorical_stats_df(plot_df)
    
    index_vars = 'model_number'
    
    if 'model_name' in plot_df.columns:
        index_vars = ['model_number','model_name']
    else:
        index_vars = 'model_number'
    
    categorical_stats_report = pd.pivot_table(data=categorical_stats_df,
                                              columns=['measure_name','measure_value'],
                                              values='count',
                                              index=index_vars)
    
    return categorical_stats_report

def get_continuous_stats_report(plot_df):
    
    check_plot_df(plot_df)
    continuous_stats_df = preparation.get_continuous_stats_df(plot_df)
    continuous_stats_df = continuous_stats_df.round({'mean':2,'median':2,'std': 2})
    continuous_stats_df['descriptive_statistics'] = continuous_stats_df.agg('{0[mean]} ({0[std]})'.format,axis=1)
    
    if 'model_name' in continuous_stats_df.columns:
        index_vars = ['model_number','model_name']
    else:
        index_vars = 'model_number'
    
    continuous_stats_report = pd.pivot_table(data=continuous_stats_df,
                                             columns='measure_name',
                                             values='descriptive_statistics',
                                             index=index_vars,
                                             aggfunc='first')
    
    return continuous_stats_report

if __name__ == '__main__':
    pass