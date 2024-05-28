import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the csv file
df = pd.read_csv('medical_examination.csv')


# Normalize the data for overweight either 0 or 1
df['overweight'] = df['weight'] / ((df['height'] / 100) ** 2)
df.loc[df['overweight'] <= 25, 'overweight'] = 0
df.loc[df['overweight'] > 25, 'overweight'] = 1
df['overweight'] = df['overweight'].astype('int64')

# Normalize the data for cholesterol and gluc either 0 or 1
for key in ['cholesterol', 'gluc']:
    df.loc[df[key] == 1, key] = 0
    df.loc[df[key] > 1, key] = 1


# Define plotter
def draw_cat_plot():
    # Melt our table so cardio will be the subject 
    df_cat = pd.melt(df, id_vars = ['cardio'], value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Draw plot using seaborn sns.FacetGrid and sns.countplot
    fig = sns.FacetGrid(data = df_cat, col = 'cardio', col_wrap = 2, height = 6)
    fig.map_dataframe(sns.countplot, 'variable', hue = 'value', order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], legend = False, palette = sns.color_palette(n_colors = 2))

    # Add legend and y-label
    fig.add_legend(title = 'value', labels = [0,1])
    fig.set(ylabel = 'total')

    """ fig, ax = plt.subplots(1,2, figsize = (12, 6), sharex = False, sharey = True)
    plt1 = sns.countplot(data = df_cat.loc[df_cat['cardio'] == 0], x = 'variable', hue = 'value', order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], legend = False, ax = ax[0])
    plt2 = sns.countplot(data = df_cat.loc[df_cat['cardio'] == 1], x = 'variable', hue = 'value', order = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], legend = False, ax = ax[1])

    ax[0].title.set_text("cardio = 0")
    ax[1].title.set_text("cardio = 1")
    ax[0].set_ylabel('total')
    ax[1].set_ylabel('total')
    ax[0].spines[['top', 'right']].set_visible(False)
    ax[1].spines[['top', 'right']].set_visible(False)
    fig.legend([plt1, plt2], title = 'value', labels = ['0','1'], loc = 'center right', frameon = False)
    fig.subplots_adjust(left = 0.075, right = 0.95, top = 0.925, bottom = 0.1, wspace = 0.05) """

    # Save figure or plot
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(corr)

    # 14
    fig, ax = plt.subplots(1,1, figsize = (9,9))

    # 15
    sns.heatmap(data = corr, mask = mask, annot = True, annot_kws = {'size' : 10}, fmt = ".1f", ax = ax, center = 0, cmap = "twilight", cbar_kws = {'shrink' : 0.6})

    # 16
    fig.savefig('heatmap.png')
    return fig
