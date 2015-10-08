import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely

__author__ = 'Andrew A Campbell'
# Tools specifically used for processing INRIX data.


def get_unique_tmc_nodes(tmc_path, out_path=None, x_coord="longitude", y_coord="latitude", ident=None):
    """
    INRIX provides a raw file called TMC_Identification.csv which contains one unique row for each TMC. A TMC is
    defined by a start and end node. This method reads that file and returns a dataframe with a unique row for each
    node. Each node contains a from_tmc and to_tmc lists as attributes.
    :param tmc_path: (str) Path to TMC_Identification.csv
    :param out_path: (str) Path to write the output dataframe as a csv.
    :param from_crs: (dict or int) CRS of TMC node coordinates. Should follow format of fiona.crs
    :param to_crs: (dict or int) EPSG code of CRS to project geometries to.
    :param x_coord: (str) Name of column containing the y-coordinates)
    :param y_coord: (str) Name of column containing the x-coordinates.
    :param ident: (str) Name of the column containing the unique ID for each point. These values are used as the index
    in the dataframe.
    :return: (geopandas.GeoDataFrame) Columns = | id | latitude | longitude | in_tmc | out_tmc |
    """
    df = pd.read_csv(tmc_path, sep=',', header=0, index_col=False)
    # Get list of unique node locations
    node_coords = dict((k, v) for k, v in enumerate(list(set(zip(df.start_latitude, df.start_longitude)
                                                             + zip(df.end_latitude, df.end_longitude)))))
    node_lookup = dict((v, k) for k, v in node_coords.items())
    # Build the digraph
    dg = nx.DiGraph()
    # Add the nodes with dummy values for tmcs
    for n, c in node_coords.items():
        dg.add_node(n, {'latitude': c[0], 'longitude': c[1], 'in_tmc': [], 'out_tmc': []})
    # Add the directed edges (TMCs are edges) and update node attributes.
    for row in df.iterrows():
        node_start = node_lookup[(row[1]['start_latitude'], row[1]['start_longitude'])]
        node_end = node_lookup[(row[1]['end_latitude'], row[1]['end_longitude'])]
        dg.add_edge(node_start, node_end, tmc=row[1]['tmc'])
        dg.node[node_start]['out_tmc'].append(row[1]['tmc'])
        dg.node[node_end]['in_tmc'].append(row[1]['tmc'])
    # Create a dataframe from the nodes
    gdf = gpd.GeoDataFrame.from_dict(dict(dg.nodes(data=True)), orient='index')
    # Add the geometry column
    gdf['geometry'] = gdf.apply(lambda x: shapely.geometry.Point(float(x[x_coord]), float(x[y_coord])), axis=1)
    # If out_path specified, write the dataframe to a csv
    if out_path:
        gdf.to_csv(out_path, sep=',', header=True, index=True, index_label='id')
    return gdf


