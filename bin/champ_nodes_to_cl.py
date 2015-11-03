


__author__ = 'Andrew A Campbell'

# This script projects nodes in the CHAMP network to the nearest qualifying candidate in the centerlines network
# Steps are executed here, but all the spatial analysis steps are defined in the utils.conflation.py module


##
# Step 1 - Node-Node matching (i.e. intersection matching)
##

# If champ node degree > 2:
#   find cl nodes that are close enough and have degree > 2
#   match to the cl node that has the correct MATCH attributes

##
# Step 2 - Node-Link matching (defining cut points)
##

# If champ node has degree == 2:
#   fi projection to closest qualifying cl line
#   project onto that line
#

