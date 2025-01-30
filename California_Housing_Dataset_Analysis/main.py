# California Housing Dataset (Analysis)

# Import libraries and custom module
import pandas as pd

from data_utils import one_hot_encoding, plot_geospatial_data, plot_data_distribution, scale_cols, plot_one_hot_boxplot, plot_one_hot_corr_matrix, plot_one_hot_significance


# Import data and handle NaNs and categorical feature
data = pd.read_csv(r'housing.csv')
#print(data.describe())

# perform one-hot encoding
encoded_data = one_hot_encoding(data, 'ocean_proximity')

handledna_encoded_data = encoded_data.dropna()
#print(abs((encoded_data[encoded_data.columns[2:9]].mean()-handledna_encoded_data[encoded_data.columns[2:9]].mean())/encoded_data[encoded_data.columns[2:9]].mean())*100) # less than 0.04% change in all features (one hot features are not included)
#print(abs((encoded_data[encoded_data.columns[2:9]].median()-handledna_encoded_data[encoded_data.columns[2:9]].median())/encoded_data[encoded_data.columns[2:9]].median())*100) # less than 0.05% change in all features (one hot features are not included)
encoded_data = handledna_encoded_data
#print(abs((encoded_data[encoded_data.columns[2:9]].mean()-encoded_data[encoded_data.columns[2:9]].median())/encoded_data[encoded_data.columns[2:9]].mean())*100) # get a first idea of the data distribution
#print(encoded_data.describe())


# Plot geospacial data
plot_geospatial_data(
    dataset=encoded_data,
    geo_cols=encoded_data.columns[0:2],
    feature_cols=encoded_data.columns[2:9], 
    one_hot_cols=encoded_data.columns[9:14],
    one_hot_title='ocean_proximity', 
    tot_rows=2, 
    tot_cols=4, 
    figsize=(17, 9),
    output_file="geospatial_data.png")


# Plot data distribution
plot_data_distribution(
    dataset=encoded_data, 
    feature_cols=encoded_data.columns[2:9], 
    one_hot_cols=encoded_data.columns[9:14], 
    one_hot_title='ocean_proximity', 
    bins=24, 
    tot_rows=2, 
    tot_cols=4, 
    figsize=(19, 9), 
    output_file="data_distribution.png")


# Scale data | Min-Max normalization
scaled_data = scale_cols(encoded_data, encoded_data.columns[2:9])


# Plot box plots
plot_one_hot_boxplot(
    dataset=scaled_data, 
    feature_cols=scaled_data.columns[2:9], 
    one_hot_cols=scaled_data.columns[9:14], 
    one_hot_title='OCEAN PROXIMITY', 
    tot_rows=2, 
    tot_cols=3, 
    figsize=(19,12), 
    output_file="box_plot.png")


# Plot correlation matrices by category
plot_one_hot_corr_matrix(
    dataset=scaled_data, 
    feature_cols=scaled_data.columns[2:9], 
    one_hot_cols=scaled_data.columns[9:14], 
    one_hot_title='OCEAN PROXIMITY', 
    tot_rows=2, 
    tot_cols=3, 
    figsize=(19,12), 
    output_file="correlation_matrix.png")


# Plot correlation matrices by category filtered
plot_one_hot_significance(
    dataset=scaled_data, 
    feature_cols=scaled_data.columns[2:9], 
    one_hot_cols=scaled_data.columns[9:14], 
    one_hot_title='OCEAN PROXIMITY', 
    athreshold=0.05,
    tot_rows=2, 
    tot_cols=3, 
    figsize=(19,12), 
    output_file="sig_corr_matrix.png")