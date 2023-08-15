# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:26:55 2023

@author: Administrator
"""
import pandas as pd
import geopandas as gpd
import shapely.geometry as geom
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches


# Read the JSON file into a DataFrame
df = pd.read_json('../SG_landuse_wGPR.json')

# Define a function to round the coordinates
def round_coordinates(coords):
    for ring in coords['rings']:
        for coord in ring:
            coord[0] = round(coord[0], 13)
            coord[1] = round(coord[1], 13)
    return coords

# Apply the function to the 'rings' list in the 'geometry' column
df['geometry'] = df['geometry'].apply(round_coordinates)

# Convert DataFrame to GeoDataFrame
gdf = gpd.GeoDataFrame(df)

#Change the geometry format 
gdf['geometry'] = gdf['geometry'].apply(lambda x: geom.Polygon(x['rings'][0]))

# Get the center point of each polygon and add it as a new column 'center'
gdf['center'] = gdf['geometry'].centroid

# Get the area from SHAPE_Area and add it as a new column 'area'
gdf['area'] = gdf['attributes'].apply(lambda x: x['SHAPE_Area'])

#Get Lu_DESC
gdf['LU_DESC'] = gdf['attributes'].apply(lambda x: x['LU_DESC'])

#Get 'GPR_NUM'
gdf['GPR_NUM'] = gdf['attributes'].apply(lambda x: x['GPR_NUM'])

#Replace attributes with 'OBJECTID'
gdf['attributes'] = gdf['attributes'].apply(lambda x: x['OBJECTID'])

#select the rows where the 'GPR_NUM' column has NaN values
nan_rows = df[df['GPR_NUM'].isna()]

#group the 'nan_rows' DataFrame by the 'LU_DESC' column
nan_rows_count_by_lu = nan_rows.groupby('LU_DESC').size()

#find the mean value of the 'GPR_NUM' column for their respective 'LU_DESC'
mean_gpr_by_lu = df.groupby('LU_DESC')['GPR_NUM'].mean()

#set nan and 0 to the mean value of the 'GPR_NUM' column for their respective 'LU_DESC'
df['GPR_NUM'] = df.groupby('LU_DESC')['GPR_NUM'].apply(lambda x: x.replace(0,np.nan).fillna(x.mean()))

#calculate the expected population
df['EP'] = df['area'] * df['GPR_NUM']

#set white EP to be 0
df.loc[df['LU_DESC'] == 'WHITE', 'EP'] = 0

# Group the DataFrame by 'LU_DESC'
grouped = df.groupby('LU_DESC')

# Define the area boundaries
min_lat, max_lat, min_lon, max_lon = [1.26523 , 1.2986 , 103.8100, 103.8550]

# Filter the DataFrame
filtered_df = df[(df['center'].apply(lambda p: p.x) >= min_lon) & 
                 (df['center'].apply(lambda p: p.x) <= max_lon) &
                 (df['center'].apply(lambda p: p.y) >= min_lat) & 
                 (df['center'].apply(lambda p: p.y) <= max_lat)]
# Extract latitude and longitude from "center" column
filtered_df['latitude'] = filtered_df["center"].apply(lambda x: x.y)
filtered_df['longitude'] = filtered_df["center"].apply(lambda x: x.x)
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6367 * 1000


#Read the CSV file
df_road = pd.read_csv('../2019.02.19.csv',nrows=58780)

# Create a dictionary mapping link ids to their respective start and end points
locations = df_road["Location"].str.split().apply(lambda x: tuple(map(float, x))).tolist()
sp = [tuple(map(float, x.split()))[:2] for x in df_road["Location"]]
ep = [tuple(map(float, x.split()))[2:] for x in df_road["Location"]]
location_dict = dict(zip(df_road["LinkID"], zip(sp, ep)))

# Create an empty list to store the node data
node_data = []

# Define the area boundaries
min_lat, max_lat, min_lon, max_lon = [min_lat-0.0018, max_lat-0.0018, min_lon-0.0018, max_lon-0.0018]

# Iterate over the location dictionary and only keep the links within the boundary
for link_id, (start_point, end_point) in location_dict.items():
    # Check if the start and end points are within the area boundaries
    if min_lat <= start_point[0] <= max_lat and min_lat <= end_point[0] <= max_lat and min_lon <= start_point[1] <= max_lon and min_lon <= end_point[1] <= max_lon:
        # Append the link_id, start_point, and end_point as a tuple
        node_data.append((link_id, start_point, end_point))
# Create a Pandas DataFrame from the link data
node_df = pd.DataFrame(node_data, columns=["LinkID", "StartPoint", "EndPoint"])

# Calculate the center point for each link
node_df["CenterPoint"] = node_df[["StartPoint", "EndPoint"]].apply(lambda row: ((row["StartPoint"][0] + row["EndPoint"][0]) / 2, (row["StartPoint"][1] + row["EndPoint"][1]) / 2), axis=1)
node_df["length"] = node_df.apply(lambda row: haversine(row["CenterPoint"][0], row["CenterPoint"][1], row["EndPoint"][0], row["EndPoint"][1]), axis=1)
#this is already half(length)
# Create separate "Latitude" and "Longitude" columns from "CenterPoint"
node_df[["latitude", "longitude"]] = node_df["CenterPoint"].apply(lambda point: pd.Series([point[0], point[1]]))

filtered_df['No_nodes_around']=np.zeros(len(filtered_df))
# loop through each unique LU_DESC value in filtered_df
for lu in filtered_df['LU_DESC'].unique():
    # Subset filtered_df to only rows with matching LU_DESC value
    filtered_lu = filtered_df[filtered_df['LU_DESC'] == lu]
    # Create a new column "EP" with zero values for all LU_DESC values
    node_df["EP_"+lu] = np.zeros(len(node_df))
    # loop through each row in filtered_lu
    for index, row in filtered_lu.iterrows():
        # initialize variables to keep track of nodes around
        nodes_around = []
        # loop through each row in the updated_node_df
        for node_index, node_row in node_df.iterrows():
            # calculate distance between current node and current row in filtered_df
            dist = haversine(row['longitude'], row['latitude'], node_row['longitude'], node_row['latitude'])
            # if distance is around, add node to list
            if dist <= 303:
                nodes_around.append(node_index)
        if len(nodes_around) == 0:
            filtered_df.loc[filtered_lu.index, 'No_nodes_around'] = 1
        # add EP values of filtered_lu to node_df for nodes around
        if len(nodes_around) > 0:
            node_df.loc[nodes_around, 'EP_'+lu] += row['EP'] / len(nodes_around)
#print the number that have nodes around
print('the number that have nodes around',filtered_df['No_nodes_around'].value_counts()[0],'total number',len(filtered_df))
ep_columns = [col for col in node_df.columns if "EP_" in col]
# use the max function to find the maximum value for each row, only including the specified columns
node_df["max_EP"] = node_df[ep_columns].max(axis=1)
# get the column index for the maximum value of the specified columns
max_cols = node_df[ep_columns].idxmax(axis=1)
# extract the relevant part of the column name to get the corresponding LU_DESC value
max_EP_lu = max_cols.apply(lambda col: col.split("_")[1])
# add the new column to the dataframe
node_df["max_EP_lu"] = max_EP_lu
node_df.loc[node_df["max_EP"] == 0, "max_EP_lu"] = 'None'
unique_max_EP_lu = node_df['max_EP_lu'].unique()

# Create a new column 'group' by encoding the 'max_EP_lu' values as integers
node_df['group'], _ = pd.factorize(node_df['max_EP_lu'])

# Create a new empty graph
G = nx.DiGraph()

# Add all nodes to the graph with their attributes
for _, row in node_df.iterrows():
    G.add_node(row['LinkID'], max_EP_lu=row['max_EP_lu'], group=row['group'], latitude=row['latitude'], longitude=row['longitude'], EP=row['max_EP'])

# Iterate over the rows in the node_df data frame
for _, row in node_df.iterrows():
    # Check if the end point of a link matches the start point of another link
    next_links = node_df[node_df['StartPoint'] == row['EndPoint']]
    for _, next_row in next_links.iterrows():
        # Check if the start point of the next link equals the end point of the current link and vice versa
        if not ((row['StartPoint'] == next_row['EndPoint']) and (row['EndPoint'] == next_row['StartPoint'])):
            # Calculate the distance between the two nodes using the haversine function
            distance = row['length']+next_row['length']
            # Add an edge between the two nodes with the weight set to the distance between them
            G.add_edge(next_row['LinkID'], row['LinkID'], weight=distance)
node_df.to_csv('output_land_find_road_303.csv')
#%%
# Create a subset of nodes with max_EP_lu values 'RESIDENTIAL' and 'COMMERCIAL'
source_nodes = node_df[node_df['max_EP_lu'] == 'RESIDENTIAL']['LinkID'].tolist()
target_nodes = node_df[(node_df['max_EP_lu'] == 'COMMERCIAL')]['LinkID'].tolist()

# get strongly connected components
scc_gen = nx.strongly_connected_components(G)
# sort components by size in descending order
scc_list = sorted(scc_gen, key=len, reverse=True)
# get largest component
largest_scc = scc_list[0]
    
# Remove isolated nodes from source and target nodes
source_nodes = [node for node in source_nodes if node in largest_scc]
target_nodes = [node for node in target_nodes if node in largest_scc]
for node in G.nodes():
    G.nodes[node]['my_betweenness'] = 0
SUM=0
no_path_count=0
from tqdm import tqdm

for source_idx, source in enumerate(tqdm(source_nodes, desc='Source nodes')):
    for target_idx, target in enumerate(tqdm(target_nodes, desc='Target nodes', leave=False)):
        all_paths = nx.all_shortest_paths(G, source=source, target=target, weight='weight')
        num_paths = len(list(all_paths))  # count the number of shortest paths
        shortest_lengths = nx.shortest_path_length(G, source=source, target=target, weight='weight')
        
        all_paths = nx.all_shortest_paths(G, source=source, target=target, weight='weight')
        for path in all_paths:
            for node in path:
                G.nodes[node]['my_betweenness'] += G.nodes[source]['EP'] * G.nodes[target]['EP'] / (num_paths*shortest_lengths)
                SUM += G.nodes[source]['EP'] * G.nodes[target]['EP'] / (num_paths*shortest_lengths)

    progress_target = (target_idx + 1) / len(target_nodes)
    tqdm.write(f'Target nodes: {progress_target:.0%} done', end='\r')

    progress_source = (source_idx + 1) / len(source_nodes)
    tqdm.write(f'Source nodes: {progress_source:.0%} done', end='\r')
                
for node in node_df['LinkID']:
    node_df.loc[node_df['LinkID'] == node, 'my_betweenness'] = G.nodes[node]['my_betweenness'] / SUM
#here change the name!!!!!!!!!!!
node_df[['LinkID', 'my_betweenness']].to_csv('betweenness_land_find_road_303_r^-1.csv', index=False)