import numpy as np
import shapely


__author__ = 'Andrew A Campbell'
# Map conflation tools



########################################################################################################################
# Node-to-link projection
########################################################################################################################
#
# These tools are used to project the nodes from a straight-line networks (e.g. TMC) onto the closest points
# in the cetnerline network.

def node_link_project(node, centerlines, cutoff):
    """
    Projects the node onto the closest qualifying link in the centerlines network.
    :param node:
    :param centerlines:
    :param cutoff:
    :return:
    """
    candidate_links = get_candidate_links(node, centerlines, cutoff)  # Set of qualifying links
    distances = [shapely.node.distance(link) for link in candidate_links]  # Shortest distances to qualifying links
    link = candidate_links[np.argmin(distances)]  # Closest qualifying link
    ppoint = proj_point(node, link)  # Point on line that node is projected to
    disp_link = displacement_link(node, point)   # Displacement link connecting original node and projection point
    return disp_link

def get_candidate_links(node, centerlines, cutoff, match_attributes = ['NAME', 'DIRECTION']):
    """
    Uses a combination of spatial bounding and attribute heuristics to return a set of lines that are candidates
    for matching with the node. First, links are filtered to the subset within the circular cutoff distance of the node.
    Only features that have matching match_attributes are kept.
    :param node:
    :param centerlines:
    :param cutoff: (float) Cutoff distance
    :param match_attributes: (list) Column names of attributes in the node and centerlines shapefiles that must match.
    :return: (list) List of links
    """
