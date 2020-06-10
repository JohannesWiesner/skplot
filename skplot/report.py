import pandas as pd
import re
from skplot import preparation

# FIXME: See extraction module. The type of measure should already be
# added in extraction.extract_single_plot data. With this, you don't have
# to use regex anymore.
def get_categorical_stats_report(plot_data,**kwargs):
    
    plot_df = preparation.get_plot_df(plot_data,**kwargs)

    # get list of names of all hyperparameters (search for categories that contain '__')
    measure_names = plot_df['measure_name'].cat.categories.to_list()
    categorical_params = [measure_name for measure_name in measure_names if re.match('^.+__.+$',measure_name)]
    
    # subset to data frame that only contains hyperparameters
    plot_df_ = plot_df.loc[plot_df['measure_name'].isin(categorical_params),:]
    plot_df_.loc[:,'measure_name'] = pd.Categorical(plot_df_['measure_name'],categories=categorical_params,ordered=True)
        
    # treat hyperparameter values as categories
    plot_df_.loc[:,'measure_value'] = plot_df_['measure_value'].astype('category')
    
    # count hyperparameter categories
    categorical_stats_report = plot_df_.groupby(['model_number','measure_name','measure_value',]).count().reset_index()
    categorical_stats_report.rename(columns={'outer_fold':'count'},inplace=True)
    categorical_stats_report.loc[:,'count'].fillna(0,inplace=True)
    
    categorical_stats_report = pd.pivot_table(data=categorical_stats_report,
                                              columns=['measure_name','measure_value'],
                                              values='count',
                                              index='model_number')
    
    return categorical_stats_report

# TO-DO: allow for measure_names
# TO-DO: allow to reorder the columns to make it more pretty, i.e.
# the order how measures is defined should define the order of how colums appear
def get_continuous_stats_report(plot_data,**kwargs):
    
    plot_df = preparation.get_plot_df(plot_data,**kwargs)
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
    
    # convert categorical columns to objects in order allow multiple 
    # stats reports to be concatenated together at later stages. Without this, you get error
    # See: https://github.com/pandas-dev/pandas/issues/19136#issuecomment-597884616
    continuous_stats_report.columns = continuous_stats_report.columns.astype(list)
    
    # FIXME: If row MultiIndex (that is (model_number,model_name), create single
    # index from this, and also convert to list in order to fix problem as above
    # continuous_stats_report.index = continuous_stats_report.index.astype(list)
    
    return continuous_stats_report

if __name__ == '__main__':
    pass