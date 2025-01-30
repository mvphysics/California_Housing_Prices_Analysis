# California Housing Dataset (Analysis)

# Import libraries and custom module
import pandas as pd

from data_utils import one_hot_encoding, plot_geospatial_data, plot_data_distribution, scale_cols, plot_one_hot_boxplot, plot_one_hot_corr_matrix, plot_one_hot_significance, log_scaling, remove_outliers


# Import data
data = pd.read_csv(r'housing.csv')
# Create new calculated columns
rooms_per_household = data["total_rooms"] / data["households"]
bedrooms_per_person = data["total_bedrooms"] / data["population"]
people_per_household = data["population"] / data["households"]
# Create the new DataFrame while replacing the original columns
data.rename(columns = {'total_rooms': 'rooms_per_household', 'total_bedrooms': 'bedrooms_per_person', 'population': 'people_per_household'}, inplace = True)  # Drop old columns
data["rooms_per_household"] = rooms_per_household
data["bedrooms_per_person"] = bedrooms_per_person
data["people_per_household"] = people_per_household

#print(data.describe())
# perform one-hot encoding
encoded_data = one_hot_encoding(data, 'ocean_proximity')

handledna_encoded_data = encoded_data.dropna()
encoded_data = handledna_encoded_data


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
    output_file="alt_geospatial_data.png")


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
    output_file="alt_data_distribution.png")


# Automatic log scale columns with wide range
encoded_data = log_scaling(encoded_data)

# Automatically remove outliers
encoded_data = remove_outliers(encoded_data, encoded_data.columns[2:9], threshold=5)
print(len(encoded_data)) # current number of entries


# Plot data distribution after log scaling and outlier removal
plot_data_distribution(
    dataset=encoded_data, 
    feature_cols=encoded_data.columns[2:9], 
    one_hot_cols=encoded_data.columns[9:14], 
    one_hot_title='ocean_proximity', 
    bins=24, 
    tot_rows=2, 
    tot_cols=4, 
    figsize=(19, 9), 
    output_file="alt_out_data_distribution.png")


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
    output_file="alt_box_plot.png")


# Plot correlation matrices by category
plot_one_hot_corr_matrix(
    dataset=scaled_data, 
    feature_cols=scaled_data.columns[2:9], 
    one_hot_cols=scaled_data.columns[9:14], 
    one_hot_title='OCEAN PROXIMITY', 
    tot_rows=2, 
    tot_cols=3, 
    figsize=(19,12), 
    output_file="alt_correlation_matrix.png")


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
    output_file="alt_sig_corr_matrix.png")

