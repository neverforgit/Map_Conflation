__author__ = 'Andrew A Campbell'

import fiona.collection
import Shapely.geometry

def points_from_csv(file_path, epsg=None, lat="latitude", lon="longitude"):
    """
    Creates a list of Shapely points from a csv file
    :param file_path: (str) Path to csv file to read
    :param epsg: (str) EPSG code for coordinate reference system to project points to
    :param lat:
    :param lon:
    :return:
    """

    # Open the file and read the lines into points
    with fiona.collection(file_path) as c:


    # Project points, if epsg != None

def read_shp():


def match_projections():



