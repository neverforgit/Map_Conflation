import ConfigParser
import sys

import fiona.crs
import geopandas as gpd

import utils.inrix
import utils.geo_io

__author__ = 'Andrew A Campbell'


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'erROOAAARRRR!!! need to pass the path to the config file.'
        exit()

    # Process config file
    conf = ConfigParser.ConfigParser()
    conf_path = sys.argv[1]
    conf.read(conf_path)
    tmc_file = conf.get('Paths', 'tmc_file')
    out_tmc_csv = conf.get('Paths', 'out_tmc_file')
    out_shapefile = conf.get('Paths', 'out_shapefile')
    clip_file = conf.get('Paths', 'clip_file')
    tmc_from_epsg = conf.get('Params', 'tmc_from_epsg')
    to_crs = conf.get('Params', 'to_crs')

    # Convert TMC edge list (TMC_Identification.csv) to node list, write output to csv and convert CRS
    gdf_tmc = utils.inrix.get_unique_tmc_nodes(tmc_file, out_tmc_csv)
    from_crs = fiona.crs.from_epsg(tmc_from_epsg)
    to_crs = fiona.crs.from_string(to_crs)
    gdf_tmc.set_geometry('geometry', inplace=True, crs=from_crs)  # set original geometry and CRS
    gdf_tmc.to_crs(crs=to_crs, inplace=True)  # project to the new CRS


    # Clip points to clipping polygon
    gdf_clip = gpd.GeoDataFrame.from_file(clip_file)
    gdf_clip.to_crs(crs=to_crs, inplace=True)
    gdf_tmc_out = gdf_tmc[gdf_tmc.within(gdf_clip.geometry[0])]  # Only contains points within clip

    # Write the points to a shapefile
    # gdf_tmc_out["in_tmc"] = gdf_tmc_out["in_tmc"].apply(lambda x: str(x))
    # gdf_tmc_out.loc[:, "out_tmc"] = gdf_tmc_out["out_tmc"].apply(lambda x: str(x))
    gdf_tmc_out.loc[:, "in_tmc"] = gdf_tmc_out["in_tmc"].astype(str, copy=False)
    gdf_tmc_out.loc[:, "out_tmc"] = gdf_tmc_out["out_tmc"].astype(str, copy=False)
    gdf_tmc_out.to_file(out_shapefile)

    print("All done")

