__author__ = 'Andrew A Campbell'

import fiona.collection
import pandas as pd
import pyproj
import shapely.geometry

def points_from_csv(file_path, epsg_from=None, epsg_to=None, x_coord="longitude", y_coord="latitude", id="id"):
    """
    Creates a list of Shapely points from a csv file
    :param file_path: (str) Path to csv file to read
    :param epsg_from: (str) EPSG code for coordinate reference system that csv is originally in.
    :param epsg_to: (str) EPSG code for coordinate reference system to project points to.
    :param x_coord: (str) Name of column containing the y-coordinates)
    :param y_coord: (str) Name of column containing the x-coordinates.
    :param id: (str) Name of the column containing the unique ID for each point.
    :return: (df) Pandas dataframe with attributes and one column containing the shapely Points as values.
    """

    # Open the file and read the lines into points
    df = pd.read_csv(file_path, header=0)
    df['points'] = df.apply(lambda x: shapely.geometry.Point(float(x[x_coord]), float(x[y_coord])), axis=1)
    # Project points, if epsg != None
    if not epsg_to:
        return df
    else:
        p1 = pyproj.Proj(init=epsg_from)
        p2 = pyproj.Proj(init=epsg_to)
        df['points'] =  df['points'].apply(lambda p: pyproj.transform(p1, p2, p.x, p.y))
    return df


def read_shp():
    None

def match_projections():
    None