from . import preparation
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd
import numpy as np

# TO-DO: Make plot 'pretty', e.g. set figure background transparent with plt.gcf().patch.set_alpha(0.0)
# TO-DO: define one function for setting style (setting seaborn stile, font, transparent plot, etc.)
def get_number_of_models(plot_df):
    return plot_df['model_number'].value_counts().count()

def get_number_of_measures(plot_df):
    return plot_df['measure_name'].value_counts().count()

def get_measure_labels(plot_df):
    return plot_df['measure_label'].unique()

def save_plot(dst_dir,filename):
    if dst_dir:
        if not filename:
            raise ValueError('Please provide a filename')
        else:
            dst_path = dst_dir + filename
            plt.savefig(dst_path,dpi=600,bbox_inches = "tight")
            
def set_x_and_y_axis_layout(g,xlim='auto',ylim='auto',x_tick_spacing='auto',y_tick_spacing='auto'):
    """Set both the tick spacing and upper and lower limits for both x- and y-axis.
    """
    
    if xlim != 'auto':
        g.set(xlim=xlim)
    
    if ylim != 'auto':
        g.set(ylim=ylim)
    
    if y_tick_spacing != 'auto':
        loc_y = plticker.MultipleLocator(base=y_tick_spacing)
        plt.gca().yaxis.set_major_locator(loc_y)
    
    if x_tick_spacing != 'auto':
        loc_x = plticker.MultipleLocator(base=x_tick_spacing)
        plt.gca().xaxis.set_major_locator(loc_x)

def plot_mean_lines(axes_flatten,continuous_stats_df,n_models,n_measures):
    """Plot mean lines for all measures appearing in the plot.
    
    Parameters
    ----------
    axes_flatten : TYPE
        Flattened array of all axes that belong to the FacetGrid object
        
    continuous_stats_df : pd.DataFrame
        A dataframe obtained from skplot.preparation.get_continuous_stats_df
        
    n_models : int
        Number of all models (aka. combinations of different pipeline parameters)
    
    n_measures : int
        Number of all measures that are plotted.

    Returns
    -------
    None.

    """
    
    
    # get colors for each score
    current_palette = sns.color_palette()
    colors = []
    for idx in range(0,n_measures):
        colors.append(current_palette[idx])
        
    # plot mean lines
    row_idx = 0
    
    for idx in range(0,n_models):    
        for line_idx in range(0,n_measures):
            axes_flatten[idx].axhline(y=continuous_stats_df.loc[row_idx,'mean'],color=colors[line_idx])
            row_idx += 1
            
def get_nested_titles_and_labels(iterator_dict):
    """Get upper and all lower x-axis titles and labels for all combinations
    of different pipelines.

    """
    
    iterator_keys, iterator_combos = preparation.get_keys_and_combos(iterator_dict)
    
    # get dataframe with labels for all x-axes in the order of the x-axes
    titles_and_labels = pd.DataFrame(iterator_combos)
    titles_and_labels.columns = iterator_keys
    titles_and_labels = titles_and_labels.iloc[:, ::-1]
    
    # get labels and title of only the main x-axis
    main_xaxis_elements = titles_and_labels.iloc[:,0]
    
    # get data frame with only secondary x-axis labels
    secondary_xaxis_elements = titles_and_labels.iloc[:,1:]

    return main_xaxis_elements,secondary_xaxis_elements

def draw_nested_xaxes(g,y_axis_title,iterator_dict):
    """Draw nested x-axes for all combinations of different pipelines.
    """

    # get x-axis elements for main and secondary x-axes
    main_xaxis_elements,secondary_xaxis_elements = get_nested_titles_and_labels(iterator_dict)
    n_secondary_axes = len(iterator_dict)-1
    
    # draw main x-axis
    g.set_xticklabels(main_xaxis_elements)
    g.set_axis_labels(x_var=main_xaxis_elements.name,y_var=y_axis_title)
    
    for ax in g.axes.flat:
        for idx in range(0,n_secondary_axes):
            
            # create additional ax object
            twin_ax = ax.twiny()
                
            # Move twinned axis ticks and label from top to bottom
            twin_ax.xaxis.set_ticks_position("bottom")
            twin_ax.xaxis.set_label_position("bottom")
            
            # Offset the twin axis below the host
            twin_ax.spines["bottom"].set_position(("axes",-0.20 * (idx + 1)))
                    
            # Hide grid lines
            twin_ax.grid(False)
            
            # set ticks and labels
            twin_ax.set_xticks(secondary_xaxis_elements.index)
            twin_ax.set_xticklabels(secondary_xaxis_elements.iloc[:,idx])
            
            # align ticks with first x-axis
            twin_ax.set_xlim(ax.get_xlim())
            
            # set title
            twin_ax.set_xlabel(secondary_xaxis_elements.columns[idx])

# FIXME: Allow for input dictionary where keys define groups and 
# list as values define which measure belongs to which group.
# Currently this function is restricted to training and test scores. 
# It's important that measures WITHIN one plot share the same scale.
def add_score_type(plot_df,train_scores):
    
    if not train_scores:
        raise ValueError('If separate_scores == True, train_scores must be provided')
     
    if isinstance(train_scores,str):
        train_scores = [train_scores]
        
    train_scores_set = set(train_scores)
    scores_set = set(plot_df['measure_name'].unique())
    
    if not train_scores_set.issubset(scores_set):
        raise ValueError('all elements of train_scores must appear as score type in plot df')
    
    plot_df['score'] = np.where(plot_df['measure_name'].isin(train_scores),'train_score','test_score')
    
    return plot_df

def lineplot_scores(plot_df,height=4,aspect=2,col_wrap=1,x_tick_spacing='auto',
                    y_tick_spacing='auto',x_axis_title='Outer Fold',y_axis_title ='Score',xlim='auto',ylim='auto',
                    legend_title='Measures',mean_lines=True,dst_dir=None,filename=None):
    '''Plot measures from one or multiple sklearn pipelines as lineplots.
    
    Parameters
    ----------
    plot_df: pd.DataFrame
        A dataframe obtained from skplot.preparation.get_plot_df()
    
    height: float
        Height (in inches) of each facet 
        
        Default: 4

    aspect: float 
        Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches. 
        
        Default: 2
    
    col_wrap: int 
        “Wrap” the column variable at this width, so that the column facets span multiple rows. Incompatible with a row facet.
        
        Default: 1
    
    x_tick_spacing:  float or 'auto'
        Set the spacing between the x-axis numbers automatically or use 
        a user defined spacing
    
        Default: 'auto'
    
    y_tick_spacing:  float or 'auto'
        Set the spacing between the y-axis numbers automatically or use 
        a user defined spacing
    
        Default: 'auto'
        
    x_axis_title: str
        Set the title(s) for the x-axis / x-axes
        
        Default: 'Outer Fold'
    
    y_axis-title: str
        Set the title(s) for the y-axis / y-axes
        
        Default: 'Score'
    
    x_lim: tuple or 'auto'
        Set the lower and upper limit of the x-axis or use 
        user defined limits
    
        Default: 'auto'
    
    y_lim: tuple or 'auto'
        Set the lower and upper limit of the x-axis or use 
        user defined limits
    
        Default: 'auto'
    
    legend_title: 'str'
        Set the title of the legend
        
        Default: 'Measures'
    
    mean_lines: boolean
        Draw mean lines for each of your measures.
        
        Default: True
    
    dst_dir: str
        Path to the directory where the plot should be saved.
        
    filename: str
        Name of the file


    Returns
    -------
    g: seaborn.FacetGrid object
        https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
        
    See also
    --------
    https://seaborn.pydata.org/generated/seaborn.lineplot.html

    '''
    
    # subset to only measures with interval scale
    plot_df = plot_df.loc[plot_df['scale_type'] == 'interval']
    
    # seaborn settings
    sns.set(font='Open Sans',style='whitegrid')
    
    # get variables that are required for later functions #####################
    n_models = get_number_of_models(plot_df)
    n_measures = get_number_of_measures(plot_df)
    continuous_stats_df = preparation.get_continuous_stats_df(plot_df)
    
    # Plot measures as lines ##################################################
    g = sns.FacetGrid(plot_df,
                      col='model_number',
                      height=height,
                      aspect=aspect,
                      col_wrap=col_wrap)
    
    g.map(sns.lineplot,'outer_fold','measure_value','measure_name')
    
    # set matplotlib variables that are required for later functions
    axes_flatten = g.axes.flatten()
    axes_iterator = g.axes.flat
    
    # draw subplot title(s) ###################################################
    if 'model_name' in plot_df.columns:
        model_titles = plot_df['model_name'].unique()
        for title,ax in zip(model_titles,axes_iterator):
            ax.set_title(title)
    
    if n_models == 1:
        g.set_titles('') 
                
    ## Add and modify legend title and labels #################################
    # FIXME: Legend is sometimes inside plot
    g.add_legend()
    legend_texts = g._legend.texts
    legend_texts[0].set_text(legend_title)
    
    if 'measure_label' in plot_df.columns:
        
        measure_labels = get_measure_labels(plot_df)
     
        for t,l in zip(legend_texts[1:],measure_labels):
            t.set_text(l)
    
    ## Modify x and y-axis intervals ##########################################
    set_x_and_y_axis_layout(g,xlim,ylim,x_tick_spacing,y_tick_spacing)
    
    # Modify x and y-axis titles ##############################################
    for ax in axes_flatten:
        ax.set_ylabel(y_axis_title)
        ax.set_xlabel(x_axis_title)
    
    ## Plot Mean Lines ########################################################
    if mean_lines == True:
        plot_mean_lines(axes_flatten=axes_flatten,
                        continuous_stats_df=continuous_stats_df,
                        n_models=n_models,
                        n_measures=n_measures)
    
    ## Save plot ##############################################################
    save_plot(dst_dir,filename)

# FIXME: if only one model is provided and separate measures=True, boxplots
# are not centered
# TO-DO: When using kind = 'box', draw mean line
def catplot_scores(plot_df,iterator_dict,height=4,aspect=2,kind='point',separate_measures=False,
                   train_scores=None,x_axis_title='Model',y_axis_title ='Score',ylim=(0,1),
                   y_tick_spacing=0.05,legend_title='Measures',ci='sd',dst_dir=None,
                   filename=None,**kwargs):
    """Plot measures from one or multiple sklearn pipelines as categorical plots.
    

    Parameters
    ----------
    plot_df: pd.DataFrame
        A dataframe obtained from skplot.preparation.get_plot_df()
        
    iterator_dict: dict of iterators or None (default = None)
        If iterator_dict is `None`, it is assumed that a single model-building-procedure
        was run.
        
        If iterator_dict has one item, it is assumed that different
        model-building procedures where running within a single loop. 
        
        If iterator_dict contains multiple items, it is assumed that different
        model-building procedures where running within nested-loops.
        
    height: float
        Height (in inches) of each facet 
        
        Default: 4

    aspect: float 
        Aspect ratio of each facet, so that aspect * height gives the width of each facet in inches. 
        
        Default: 2
    
    kind: str
        The kind of plot to draw, corresponds to the name of a categorical axes-level plotting function. Options are: “strip”, “swarm”, “box”, “violin”, “boxen”, “point”, “bar”, or “count”.
        
        Default: 'point'
        
    separate_measures : bool
        Plot train and test scores separately in different subplots. If true,
        you must provide train_scores
        
        Default: False
    
    train_scores : str or list
        A list of names that define the names of all train scores.
        
    x_axis_title: str
        Set the title(s) for the x-axis / x-axes
        
        Default: 'Model'
    
    y_axis-title: str
        Set the title(s) for the y-axis / y-axes
        
        Default: 'Score'
    
    y_lim: tuple or 'auto'
        Set the lower and upper limit of the x-axis or use 
        user defined limits
    
        Default: (0,1)
    
    y_spacing: float
        Set the y-tick spacing between the y-axis ticks.
        
        Default: 0.5
    
    legend_title: 'str'
        Set the title of the legend.
        
        Default: 'Measures'
        
    ci : float or “sd” or None, optional
        Size of confidence intervals to draw around estimated values. If “sd”, skip bootstrapping and draw the standard deviation of the observations. If None, no bootstrapping will be performed, and error bars will not be drawn.
    
        Default: 'sd'
        
    dst_dir: str
        Path to the directory where the plot should be saved.
        
    filename: str
        Name of the file
        
    kwargs: key, value pairings
        Other keyword arguments are passed through to the underlying plotting function.

    Returns
    -------
    g: seaborn.FacetGrid object
        Returns the FacetGrid object with the plot on it for further tweaking.

    """
    
    # subset to only measures with interval scale ############################
    plot_df = plot_df.loc[plot_df['scale_type'] == 'interval']

    # seaborn settings ########################################################
    sns.set(font='Open Sans',style='whitegrid',font_scale=1)
    
    ## get and set variables that are required for later functions ############
    n_models = get_number_of_models(plot_df)

    if iterator_dict:
        iterator_keys,iterator_combinations = preparation.get_keys_and_combos(iterator_dict)
        n_iterators = preparation.get_number_of_iterators(iterator_dict)
        
    if separate_measures:
        plot_df = add_score_type(plot_df,train_scores)
        col = 'score'
    else:
        col = None

    # Plot ####################################################################
        
    g = sns.catplot(x='model_number',
                    y='measure_value',
                    hue='measure_name',
                    col=col,
                    kind=kind,
                    data=plot_df,
                    height=height,
                    aspect=aspect,
                    ci=ci,
                    legend=False,
                    **kwargs)
    
    if separate_measures:
        
        # set subplot titles
        axes_iterator = g.axes.flat
        for title,ax in zip(['Train Scores','Test Scores'],axes_iterator):
            ax.set_title(title)


    ## Set legend title and labels ############################################
    g.add_legend()
    legend_texts = g._legend.texts
    g._legend.set_title(legend_title)
    
    if 'measure_label' in plot_df.columns:
        
        measure_labels = get_measure_labels(plot_df)
     
        for t,l in zip(legend_texts,measure_labels):
            t.set_text(l)
            
            
    # Draw x- and y-axis titles and labels ####################################
    if n_models == 1:
        g.set_xticklabels('')
        g.set_axis_labels(x_var='',y_var=y_axis_title)
    

    elif n_models > 1:
        
        if iterator_dict:
            
            if n_iterators > 1:
                draw_nested_xaxes(g,y_axis_title,iterator_dict)
            
            elif n_iterators == 1:
                iterator_name = list(iterator_dict.keys())[0]
                iterator_list = list(iterator_dict.values())[0]
                g.set_xticklabels(iterator_list)
                g.set_axis_labels(x_var=iterator_name,y_var=y_axis_title)
        
        else:
            model_numbers = np.linspace(start=1,stop=n_models,num=n_models,dtype=int)
            g.set_xticklabels(model_numbers)
            g.set_axis_labels(x_var=x_axis_title,y_var=y_axis_title)

    # Modify x and y-axis intervals ##########################################
    set_x_and_y_axis_layout(g,xlim='auto',x_tick_spacing='auto',ylim=ylim,y_tick_spacing=y_tick_spacing)
        
    ## Save plot ##############################################################
    save_plot(dst_dir,filename)
    
    return g
    
if __name__ == '__main__':
    pass