import fiona.collection
import geopandas as gpd
import pandas as pd
import pyproj
import shapely.geometry

import geotools

__author__ = 'Andrew A Campbell'
# These are a general set of tools for input/output of spatial data.

def points_from_polyline(path, link_name='MATCH_NAME', dir_name='MATCH_DIR'):
    """
    Reads a polyline shapefile and extracts the end nodes of each feature. The output shapefile is featureset of points
    that contains useful attributes for mapping a node's topological properties in the original network.
    :param path: (str) Path to shapefile with the polyline.
    :param link_name: (str) Name of the feature in the input polyline to use as a link name.
    :param dir_name: (str) Name of the feature in the input polyline to use for the direction of the link.
    :return: (geopapandas.GeoDataFrame) Attributes: | ID | x_coord | y_coord | degree | in_links | out_links |
    where in/out_links are lists of tuples describing the links connected to the node. Each tuple consists
    of link name and direction. This is used in matching intersection nodes during conflation.
    """
    line_gdf = gpd.GeoDataFrame.from_file(path)
    node_dict = {}
    for row in line_gdf.iterrows():
        # Get x and y
        start_x = row[1]['geometry'].boundary[0].x
        start_y = row[1]['geometry'].boundary[0].y
        end_x = row[1]['geometry'].boundary[-1].x
        end_y = row[1]['geometry'].boundary[-1].y
        # Add nodes to node_dict, or update existing entry





########################################################################################################################
# NOTE: everything below here seems to be made redundant by Geopandas
def read_shp(path,  epsg_from=None, epsg_to=None):
    """
    Reads a shapefile and returns a spatial dataframe with the appropriate Shapely objects in the geometry column.
    :param path: (str) Path to csv file to read
    :param epsg_from: (str) EPSG code for coordinate reference system that sdf is originally in.
    :param epsg_to: (str) EPSG code for coordinate reference system to project sdf geometry to.
    :return: (pandas.DataFrame)
    """

    # Get the original CRS and instantiate projections
    with fiona.open(path) as shape:
        if not epsg_from:
            p_from = pyproj.Proj(shape.crs)
        else:
            p_from = pyproj.Proj(init=epsg_from)
        p_to = pyproj.Proj(init=epsg_to)

    sdf = geotools.make_spatial_df(path)


def sdf_project(spatial_df, p_from, p_to):

    """
    Changes the CRS for a spatial dataframe. Updates the coordinates of the objects in the 'geometry' column to a
    new CRS. Normally used to project.
    WARNING: can only handle shapely geometries handled by shape_project(...)
    :param spatial_df: (pandas.DataFrame) A spatial dataframe.
    :param p_from: (pyproj.Proj) Original CRS.
    :param p_to: (pyproj.Proj) New CRS.
    :return: (pandas.DataFrame)
    """

    assert "geometry" in spatial_df.columns  # spatial_df must have column of Shapely objects
    geom = spatial_df.iloc[0].geometry.geom_type
    # possible geometries: point, linestring, poylgon, multipoint, multilinestring, multipolygon
    spatial_df["geometry"] = spatial_df["geometry"].apply(lambda x: shape_project(x, p_from, p_to))
    return spatial_df

def shape_project(geometry, p_from, p_to):
    """
    Projects a shapely object to new CRS.
    WARNING: At this point can only handle subset of shapely geometries: point, polygon. (AAC 15/10/07)
    :param geometry: (shapely object) Point, polygon
    :param p_from: (pyproj.Proj) Original CRS.
    :param p_to: (pyproj.Proj) New CRS.
    :return: (shapely object) Updated shapely object using new CRS.
    """

# TODO delete all this commented out cruft below once I am sure I don't need it for anything

# def points_gdf_from_csv(file_path, epsg_from=None, epsg_to=None, x_coord="longitude", y_coord="latitude", ident=None):
#     """
#     Creates a geodataframe. The columns contain the original attributes in the csv plus a geometry column that
#      that contains the shapely objects.
#     :param file_path: (str) Path to csv file to read
#     :param epsg_from: (str) EPSG code for coordinate reference system that csv is originally in.
#     :param epsg_to: (str) EPSG code for coordinate reference system to project points to.
#     :param x_coord: (str) Name of column containing the y-coordinates)
#     :param y_coord: (str) Name of column containing the x-coordinates.
#     :param ident (str) Name of the column containing the unique ID for each point. These values are used as the index
#     in the dataframe.
#     :return: (pd.DataFrame) Pandas dataframe with attributes and one column ('geometry') containing the shapely Points
#     as values.
#     """
#
#     # Open the file and read the lines into points
#     df = pd.read_csv(file_path, header=0, index_col=ident)
#     df['geometry'] = df.apply(lambda x: shapely.geometry.Point(float(x[x_coord]), float(x[y_coord])), axis=1)
#     # Project points, if epsg != None
#     if not epsg_to:
#         return df
#     else:
#         p_from = pyproj.Proj(init=epsg_from)
#         p_to = pyproj.Proj(init=epsg_to)
#         df['geometry'] = df['geometry'].apply(lambda p: pyproj.transform(p_from, p_to, p.x, p.y))
#     return df
#
#
# def points_from_gdf(gdf, epsg_from=None, epsg_to=None, x_coord="longitude", y_coord="latitude", ident=None):
#     """
#     Creates a geodataframe. The columns contain the original attributes in the csv plus a geometry column that
#      that contains the shapely objects.
#     :param gdf: (geopandas.GeoDataFrame)
#     :param epsg_from: (str) EPSG code for coordinate reference system that csv is originally in.
#     :param epsg_to: (str) EPSG code for coordinate reference system to project points to.
#     :param x_coord: (str) Name of column containing the y-coordinates)
#     :param y_coord: (str) Name of column containing the x-coordinates.
#     :param ident: (str) Name of the column containing the unique ID for each point. These values are used as the index
#     in the dataframe.
#     :return: (pd.DataFrame) Pandas dataframe with attributes and one column ('geometry') containing the shapely Points
#     as values.
#     """
#
#     # Open the file and read the lines into points
#     gdf['geometry'] = gdf.apply(lambda x: shapely.geometry.Point(float(x[x_coord]), float(x[y_coord])), axis=1)
#     # Project points, if epsg != None
#     if not epsg_to:
#         return gdf
#     else:
#         p_from = pyproj.Proj(init=epsg_from)
#         p_to = pyproj.Proj(init=epsg_to)
#         gdf['geometry'] = gdf['geometry'].apply(lambda p: pyproj.transform(p_from, p_to, p.x, p.y))
#     # Set the GeoDataFrame geometry and crs
#     return gdf



