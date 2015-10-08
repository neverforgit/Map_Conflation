__author__ = 'Timothy Roadrunna Brathwaite  '

import pandas as pd
import numpy as np
import networkx as nx
import shapely.geometry
import rtree

def make_chunk_indices(data, chunk_size):
    """
    data:       iterable. Contains the data to be read.
    chunk_size:  int. Should be <= len(data). Denotes the size of the chunks
                that one is to use to read data.
    ====================
    Returns:    list of tuples. Each tuple contains a starting and ending
                index. The indices denote the starting and ending positions to
                be used to read in each chunk of data
    """
    num_records = len(data)
    if chunk_size < num_records:
        # Get the number of full size chunks and the number
        # of records in the last chunk
        quotient, remainder = divmod(num_records, chunk_size)
        last_indices = [(num_records - remainder ,num_records)] if remainder != 0 else []
        full_indices = [(num * chunk_size, (num + 1) * chunk_size) for num in range(quotient)]
        return full_indices + last_indices
    else:
        # If the chunk_size is larger than or equal to the
        # number of records, read the entire dataset at once
        return (0, num_records)


def make_spatial_df(path, method='fiona', chunk_size=None, attr_filter=None):
    """
    path: a path to a shapefile
    method: One of the strings, "fiona", "pyshp", "pysal", or 'ogr'. This specifies which module has been imported and should
    be used to create the dataframe.
    chunk_size: int. only works with method=="fiona".
    attr_filter: A sql expression used to filter a shapefile. This argument only works if the method == 'ogr'
    
    ====================
    Returns a dataframe of the spatial information and attributes associated with each feature in the shapefile at "path".
    """
    import_exception = Exception("The passed method, {}, could not be imported".format(method))
    
    global shapely
    if method == "pyshp":
        try:
            import shapefile
        except:
            raise import_exception
        _shpFile = shapefile.Reader(path) #Load the shapefile into pyshp
        attrs = [] #Create an empty array to hold the column headings from the attribute table
        for field in _shpFile.fields[1:]: #Iterate over the list of fields, ignore the "DeletionFlag" field
            attrs.append(field[0]) #Add the field name to the list column headings
        cols = [("geometry", "coordinates"), ("geometry", "type"), "shapely_obj"] + attrs #Create the dataframe columns
        data = [] #Create the overall array to hold all the data
        recs = _shpFile.shapeRecords() #Create a list of all the features/records in the shapefile
        invalid_shapes = 0
        for feat in recs: #Iterate over all features in the shapefile
            newRow = [] #Create a new array to hold each row's data
            newRow.append(feat.shape.__geo_interface__["coordinates"]) #Add the feature's coordinates to the row
            newRow.append(feat.shape.__geo_interface__["type"]) #Add the feature's geometry type to the row
            try:
                newRow.append(shapely.geometry.geo.shape(feat.shape.__geo_interface__)) #Add a shapely representation of the feature
            except:
                newRow.append(np.nan)
                invalid_shapes += 1
            newRow.extend(feat.record) #Add the feature's attributes to the row
            data.append(newRow) #Add the row to the overall array of data
        if invalid_shapes > 0:
            print "This dataframe contains {} invalid shapes in the shapely_obj column.".format(invalid_shapes)
        return pd.DataFrame(data, columns = cols) #Create the dataframe from the data.
    elif method == "fiona":
        import_exception = Exception("The passed method, {}, could not be imported".format(method))
    if method == "fiona":
        # Import needed libraries
        try:
            import fiona
        except:
            raise import_exception

        # Create the fiona collection of the shapefile
        with fiona.open(path) as coll:
            # Get the column headings from the attribute table
            attrs = coll[0]["properties"].keys()
            # Create the dataframe columns
            cols = [("geometry", "coordinates"), ("geometry", "type"), "shapely_obj"] + attrs
            # Create a variable to keep track of the number of invalid shapes in the file
            invalid_shapes = 0

            # Get the chunk_indices if necessary
            if chunk_size is not None:
                chunk_indices = make_chunk_indices(coll, chunk_size)
            else:
                chunk_indices = [(0, len(coll))]

            # Create the dataframe(s) of the shapefile info
            for pos, index_pair in enumerate(chunk_indices):
                # Create the overall array to hold all the data
                data = []
                # Get starting and ending indices
                start, end = index_pair
                for index in xrange(start, end):
                    # Access the desired feature
                    feature = coll[index]

                    # Create a new array to hold each row's data
                    newRow = []
                    # Add the feature's coordinates and geometry type to the row
                    newRow.append(feature["geometry"]["coordinates"])
                    newRow.append(feature["geometry"]["type"])
                    #Add the feature's shapely representation to the row
                    try:
                        newRow.append(shapely.geometry.geo.shape(feature["geometry"]))
                    except:
                        newRow.append(np.nan)
                        invalid_shapes += 1
                    # Add the feature's attributes to the row
                    newRow.extend(feature["properties"].values())
                    # Add the row to the overall array of data
                    data.append(newRow)

                # Add the batch of data to dataframes
                if pos == 0:
                    output = pd.DataFrame(data, columns=cols)
                else:
                    output = output.append(pd.DataFrame(data, columns=cols),
                                           ignore_index=True)

            if invalid_shapes > 0:
                msg = "This dataframe contains {:,} invalid shapes in the shapely_obj column."
                print msg.format(invalid_shapes)

            return output
    elif method == "pysal":
        try:
            import pysal
        except:
            raise import_exception
        dbf = path[:-3] + "dbf" #Create the path to the database file from the path to the shapefile
        db = pysal.open(dbf, "r") #Open the database file with pysal
        attributes = pd.DataFrame(db[:, :], columns = db.header) #Create a dataframe from the database's records
        features = list(pysal.open(path, "r")) #Create a list containing all the features in the shapefile
        coords = [] #Initialize an empty list for the coordinates of each feature
        types = [] #Initialize an empty list for the geometry type of each feature
        shapely_obs = [] #Initialize an empty list for the shapely object of each feature
        invalid_shapes = 0
        for x in features: #iterate over all the features
            coords.append(x.__geo_interface__["coordinates"]) #Add the feature's coordinates to coords
            types.append(x.__geo_interface__["type"]) #Add the feature's geometry type to types
            try:
                shapely_obs.append(shapely.geometry.geo.shape(x.__geo_interface__))
            except:
                shapely_obs.append(np.nan)
                invalid_shapes += 1
        #Make a dataframe with two columns, one for the coordinates and one for the geometry type
        geoData = pd.DataFrame({("geometry", "coordinates"): coords,
                                ("geometry", "type"): types,
                                "shapely_obj": shapely_obs})
        if invalid_shapes > 0:
            print "This dataframe contains {} invalid shapes in the shapely_obj column.".format(invalid_shapes)
        return geoData.join(attributes)
    elif method == "ogr":
        try:
           import osgeo.ogr as ogr
           import shapely.wkt
        except:
            raise import_exception
        _shp = ogr.Open(path) #open the shapefile with ogr
        _lyr = _shp.GetLayer() #Access the shapefile's layer. Assumes that the shapefile has only one layer
        if attr_filter is not None:
            _lyr.SetAttributeFilter(attr_filter)
        attr_cols = _lyr.GetFeature(0).keys() #Get a list of the shapefile attribute column names
        cols = [("geometry", "coordinates"), ("geometry", "type"), "shapely_obj"] + attr_cols #List the dataframes columns
        recs = [] #Initialize an empty list for the shapefile records
        invalid_shapes = 0
        for i in range(_lyr.GetFeatureCount()): #Iterate over all of the shapefiles features
            feat = _lyr.GetFeature(i) #Get a particular feature.
            
            #Get the well-known-text representation of the geometry and use that
            #to create the shapely object
            geom = feat.GetGeometryRef()
            feat_wkt = geom.ExportToWkt()
            try:
                shapely_obj = shapely.wkt.loads(feat_wkt)
                
                
                geo_dict = shapely.geometry.mapping(shapely_obj) #Get a GeoJSON object of the feature
                coords = geo_dict['coordinates'] #Get the coordinates of the feature from the GeoJson
                geo_type = geo_dict['type'] #Get the geometry type of the coordinates from the GeoJson
            
                new_row = [] #Initialize a container for the items in the new row of the dataframe
                new_row.extend([coords, geo_type, shapely_obj])
            except:
                new_row = [] #Initialize a container for the items in the new row of the dataframe
                new_row.extend([np.nan, np.nan, np.nan])
                invalid_shapes += 1
                
            for key in feat.keys(): #Iterate over the attributes of the feature
                new_row.append(feat[key]) #Add the attributes to the list for the dataframe
            
            recs.append(new_row)
        _shp = None #close the shapefile
        if invalid_shapes > 0:
            print "This dataframe contains {} invalid shapes in the shapely_obj column.".format(invalid_shapes)
        return pd.DataFrame(recs, columns = cols) #Create the spatial dataframe
            
    else:
        raise Exception ("Invalid 'method': {} passed as argument".format(method))
        
def build_rtree(spatial_df):
    """spatial_df: A pandas dataframe of spatial objects. Should contain a
    'shapely_obj' column with the shapely representations of these spatial
    objects.
    
    ==========
    Returns a rtree index of the bounding boxes of the objects in dataframe."""
    assert "shapely_obj" in spatial_df.columns
    
    new_tree = rtree.index.Index()
    for ind, row in spatial_df.iterrows():
        new_tree.add(ind, row['shapely_obj'].bounds)
    return new_tree
    
def build_undirected_graph(spatial_df):
    """spatial_df: A pandas dataframe of linestring objects. Should contain
    the columns ('geometry', 'type'), 'shapely_obj', 'F_Node', and 'T_Node'.
    
    ==========
    Returns a networkx graph where the nodes are the 'F_Node' and 'T_Node' values
    of street_dataframe and where the edges have their 'weight' equal to the 
    length of the linestring and their 'label' equal to the dataframe index of
    the linestring."""
    cols = spatial_df.columns
    assert all([x in cols for x in [("geometry", "type"), "F_Node", "T_Node", "shapely_obj"]])
    unique_geo_types = spatial_df[("geometry", "type")].unique()
    assert "LineString" in unique_geo_types and len(unique_geo_types) == 1
    
    graph = nx.Graph()
    for ind, row in spatial_df.iterrows():
        row_shape = row['shapely_obj']
        graph.add_edges_from([(row['F_Node'], row['T_Node'],
                                    {'label': ind, 'weight': row_shape.length})])
    return graph
    
def _break_multilinestrings(spatial_df, last_orig_indice):
    """spatial_df: A pandas dataframe of geospatial objects, all of which should
    have MultiLineString geometry types.
    
    last_orig_indice: The last index value in the original dataframe from which
    these MultiLineStrings were extracted.
    
    ===========
    Returns a new dataframe containing only disaggregated records for the
    MultiLineStrings. All of the attributes have been left the same as in the 
    original MultiLineString records, except for the geometry type, geometry 
    coordinates, and shapely object. The returned dataframe contains one row for
    each component of each of the original MultiLineStrings.
    """
    cols = spatial_df.columns.tolist()
    geo_cols = [('geometry', 'coordinates'), ('geometry', 'type'), "shapely_obj"]
    assert all([x in cols for x in geo_cols])
    assert "MultiLineString" in spatial_df[('geometry', 'type')].unique() 
    assert len(spatial_df[('geometry', 'type')].unique()) == 1
    
    constant_attributes = cols[:]
    for x in geo_cols:
        if x in constant_attributes:
            constant_attributes.remove(x)
    
    new_records = []
    for ind, row in spatial_df.iterrows():
        multi_coordinates = row[('geometry', 'coordinates')]
        orig_attributes = row[constant_attributes].values.tolist()
        for coords in multi_coordinates:
            new_geo = [coords, "LineString", shapely.geometry.LineString(coords)]
            new_row = new_geo + orig_attributes
            assert isinstance(new_row, list) and len(new_row) == len(cols)
            new_records.append(new_geo + orig_attributes)
            
    new_index = range(last_orig_indice + 1, len(new_records) + last_orig_indice + 1)
    assert len(new_index) == len(new_records)
    
    return pd.DataFrame(new_records, index=new_index, columns=cols) 

def convert_multilinestrings_to_linestrings(spatial_df):
    """spatial_df: A pandas dataframe of geospatial objects.
    
    ===========
    Returns a new dataframe where any MultiLineStrings objects were broken up
    into multiple LineString records, one for each of the components in the 
    original MultiLineString records. All of the attributes have been left the 
    same as in the original MultiLineString records, except for the geometry 
    type, geometry coordinates, and shapely object.
    
    If spatial_df did not contain any MultiLineStrings then the original
    dataframe is returned."""
    cols = spatial_df.columns
    geo_cols = [('geometry', 'coordinates'), ('geometry', 'type'), "shapely_obj"]
    assert all([x in cols for x in geo_cols])
    
    
    multi_df = spatial_df[spatial_df[('geometry', 'type')] == "MultiLineString"]
    multi_labels = multi_df.index.tolist()
    if "MultiLineString" in spatial_df[('geometry', 'type')].unique():
        disaggregate_df = _break_multilinestrings(multi_df, spatial_df.index[-1])
        assert len(set(spatial_df.index).intersection(set(disaggregate_df.index))) == 0
        
        final_df = pd.concat([spatial_df, disaggregate_df])
        final_df.drop(multi_labels, inplace=True)
        return final_df
    else:
        return spatial_df
        
def write_shapefile(spatial_df, dest_path, example_path, property_schema_list=None):
    try:
        import fiona
    except:
        raise Exception("Fiona could not be imported")
    with fiona.open(example_path) as source:
        source_driver = source.driver
        source_crs = source.crs
        source_schema = source.schema
        
    assert 'shapely_obj' in spatial_df.columns
    geo_types = map(lambda x: x.geom_type, spatial_df['shapely_obj'].values)
    assert len(set(geo_types)) == 1
    
    source_schema['geometry'] = geo_types[0]
        
    if property_schema_list:
        assert isinstance(property_schema_list, list)
        assert all([isinstance(x, tuple) for x in property_schema_list])
        source_schema['properties'] = property_schema_list
        
    with fiona.open(dest_path, 'w', 
                    driver = source_driver,
                    crs = source_crs,
                    schema = source_schema) as output:
        if isinstance(source_schema['properties'], list):
            columns = [x[0] for x in source_schema['properties']]
        else:
            columns = source_schema['properties'].keys()
        
        original_columns = spatial_df.columns
        assert all([x in original_columns for x in ["shapely_obj"] + columns])
        relevant_df = spatial_df[columns]
        
        records = []
        for ind, row in relevant_df.iterrows():
            row_shape = spatial_df.at[ind, "shapely_obj"]
            rec = {}
            rec['id'] = ind
            rec['geometry'] = shapely.geometry.mapping(row_shape)
            rec['properties'] = row.to_dict()
            records.append(rec)
            
        output.writerecords(records)
        
def _create_interim_node_dict(linestring_df, name_col, ref_node_df=None):
    linestring_cols = linestring_df.columns.tolist()
    assert all([x not in linestring_cols for x in ['F_Node', 'T_Node']])
    linestring_df['F_Node'], linestring_df['T_Node'] = np.nan, np.nan
    
    if ref_node_df is not None:
        assert ("geometry", "coordinates") in ref_node_df.columns
        ref_node_coords = ref_node_df[("geometry", "coordinates")].values
        assert len(ref_node_coords) > 0

    
    number = 'number'
    names = 'names'
    
    node_points = {} #Initialize a container for the nodes
    index_extent = 0 if ref_node_df is None \
                   else np.max(ref_node_df.index.values) + 1
    
    for ind, row in linestring_df.copy().iterrows():
        row_coords = row[('geometry', 'coordinates')]
        beg_coord, end_coord = row_coords[0], row_coords[-1]
        assert all([len(x) == 2 for x in [beg_coord, end_coord]])
        row_name = row[name_col]
        for endpoint_coord in [beg_coord, end_coord]:
            if endpoint_coord not in node_points:
                node_points[endpoint_coord] = {}
    
                if ref_node_df is None:
                    node_points[endpoint_coord][number] = index_extent
                    index_extent += 1
                else:
                    filter_list = [endpoint_coord == x for x in ref_node_coords]
                    if not any(filter_list):
                        node_points[endpoint_coord][number] = index_extent
                        index_extent += 1
                    else:
                        node_points[endpoint_coord][number] = ref_node_df[filter_list].index.values[0]
    
                node_points[endpoint_coord][names] = set()
        
        if pd.notnull(row_name) and row_name not in ['', None]:
            for coords in [beg_coord, end_coord]:
                node_points[coords][names].add(row_name)
        linestring_df.loc[ind, 'F_Node'] = node_points[beg_coord][number]
        linestring_df.loc[ind, 'T_Node'] = node_points[end_coord][number]
                
    #Check for a non-uniqueness of the index
    node_indices = sorted(node_points.keys(), key=lambda x: (x[0], x[1]))
    max_pos = len(node_indices) - 1
    repeaters = set()
    i = 0
    while i < max_pos:
        if node_indices[i] == node_indices[i+1]:
            repeaters.add(node_indices[i])
        i += 1
    try:
        assert len(repeaters) == 0
    except AssertionError as e:
        print "The coordinates which were repeated are", list(repeaters)
        raise e
    
    return node_points
    
def _node_dict_to_node_df(node_dict):
    max_names = 0
    #Find the maximum number of names associated with any point
    node_info = []
    
    for key in node_dict:
        name_set = node_dict[key]['names']
        name_set_length = len(name_set)
        if name_set_length > max_names:
            max_names = name_set_length
        node_info.append([node_dict[key]['number'], key, 'Point', list(name_set)])
    
    def fill_name_fields(num_fields, list_of_names):
        standardized_info = []
        num_names = len(list_of_names)
        for i in xrange(num_fields):
            if i < num_names:
                standardized_info.append(list_of_names[i])
            else:
                standardized_info.append(np.nan)
        return standardized_info
    
    df_data = []
    for details in node_info:
        new_row = [details[0], details[1], details[2], shapely.geometry.Point(details[1])]
        new_row.extend(fill_name_fields(max_names, details[-1]))
        df_data.append(new_row)
    
    street_cols = ["street_{}".format(x) for x in range(1, max_names + 1)]
    node_cols = [('geometry', 'coordinates'), ('geometry', 'type'), 'shapely_obj'] + street_cols
    
    df_data = pd.DataFrame(df_data, columns= ['node_num'] + node_cols)
    df_data.set_index('node_num', inplace=True)
    return df_data.sort_index()

def create_node_df(linestring_df, id_col, name_col, reference_node_df=None):
    assert isinstance(linestring_df, pd.DataFrame)
    linestring_cols = linestring_df.columns.tolist()
    assert all([x in linestring_cols for x in [id_col, 'shapely_obj', name_col,
                                              ('geometry', 'coordinates')]])
    
    #Sort the linestrings by their unique identifiers
    working_df = linestring_df.sort(id_col).copy()
    
    #Create a dictionary of the points, the index number of the point,
    #and the names of the streets associated with that point
    if reference_node_df is not None:
        node_points = _create_interim_node_dict(working_df, name_col,
                                            ref_node_df = reference_node_df)
    else:
        node_points = _create_interim_node_dict(working_df, name_col)
    
    return working_df, _node_dict_to_node_df(node_points)   


def reproject_shapefile_with_gdal(input_shp_path, output_shp_path, output_epsg):
    """
    input_shp_path:     str. Absolute or relative path to the location of the
                        input shapefile which contains the shapes and that
                        should be re-projected into a new shapefile.

    output_shp_path:    str. Absolute or relative path indicating where the
                        projected shapefile should be stored.

    output_epsg:        int. The EPSG code that indicates the coordinate
                        reference system that one wants the new shapefile
                        to have.
    ====================
    Returns:            None. Writes a new shapefile to the location indicated
                        by output_shp_path.
    """
    try:
        from osgeo import ogr
        from osgeo import osr
        import os
        import time
    except:
        print "One of the required libraries for this function is missing."
        print "Ensure that osgeo.ogr and osgeo.osr are installed."
        raise Exception

    # Begin timing the entire process
    start_time = time.time()

    # Get the driver to be used to read and write the shapefiles
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # Get the file name, excluding the rest of the file path if an
    # absolute path was specified as output_shp_path and excluding
    # the '.shp' from the end of a shapefile name
    out_file_name = output_shp_path[:-4]
    folder_sep = out_file_name.rfind("\\") if out_file_name.rfind("\\") != -1 else out_file_name.rfind("/")
    if folder_sep != -1:
        out_file_name = out_file_name[folder_sep + 1:]

    # Load an object to access the shapefile that we are working with
    in_data_set = driver.Open(input_shp_path)
    in_layer = in_data_set.GetLayer()

    # Get the spatial reference of the input dataset
    in_spatial_ref = in_layer.GetSpatialRef()

    # Create the output spatial reference
    out_spatial_ref = osr.SpatialReference()
    out_spatial_ref.ImportFromEPSG(output_epsg)

    # create the Coordinate Transformation
    coord_trans = osr.CoordinateTransformation(in_spatial_ref, out_spatial_ref)

    # Map geometry type numbers to their respective well-known-binary formats
    geom_type_to_wkb = {1: ogr.wkbMultiPoint,
                        2: ogr.wkbLineString,
                        3: ogr.wkbPolygon,
                        4: ogr.wkbMultiPoint,
                        5: ogr.wkbMultiLineString,
                        6: ogr.wkbMultiPolygon,
                        7: ogr.wkbGeometryCollection}

    # create the output layer
    if os.path.exists(output_shp_path):
        driver.DeleteDataSource(output_shp_path)
    out_data_set = driver.CreateDataSource(output_shp_path)
    out_layer = out_data_set.CreateLayer(out_file_name,
                                         geom_type=geom_type_to_wkb[in_layer.GetGeomType()])

    # add fields
    in_layer_defn = in_layer.GetLayerDefn()
    for i in range(0, in_layer_defn.GetFieldCount()):
        field_defn = in_layer_defn.GetFieldDefn(i)
        out_layer.CreateField(field_defn)

    # get the output layer's feature definition
    out_layer_defn = out_layer.GetLayerDefn()

    # Count how many features are in one-tenth of the input layer
    tenth_of_data = int(0.1 * in_layer.GetFeatureCount())

    # loop through the input features
    in_feature = in_layer.GetNextFeature()
    current_feature_num = 0

    # Let users know we are beginning to reproject the individual objects
    print "Beginning to project the individual shapes"
    print "Elapsed Time = {:,.2} minutes".format((time.time() - start_time) / 60.0)

    while in_feature:
        # Provide status updates once every tenth of the data is complete
        if current_feature_num != 0 and current_feature_num % tenth_of_data == 0:
            print "Just finished re-projecting {:,} features".format(current_feature_num)
            current_elapsed_time = (time.time() - start_time) / 60.0
            print "Elapsed Time = {:,.2} minutes".format(current_elapsed_time)

        # get the input geometry
        geom = in_feature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coord_trans)
        # create a new feature
        out_feature = ogr.Feature(out_layer_defn)
        # set the geometry and attribute
        out_feature.SetGeometry(geom)
        for i in range(0, out_layer_defn.GetFieldCount()):
            out_feature.SetField(out_layer_defn.GetFieldDefn(i).GetNameRef(),
                                 in_feature.GetField(i))
        # add the feature to the shapefile
        out_layer.CreateFeature(out_feature)
        # destroy the features and get the next input feature
        out_feature.Destroy()
        in_feature.Destroy()
        in_feature = in_layer.GetNextFeature()
        # Update the feature count
        current_feature_num += 1

    # close the shapefiles
    in_data_set.Destroy()
    out_data_set.Destroy()

    # Write the .prj file that is to be associated with this shapefile
    out_spatial_ref.MorphToESRI()
    with open(output_shp_path[:-4] + ".prj", "w") as f:
        f.write(out_spatial_ref.ExportToWkt())

    # Let users know the operation is complete
    print "Finished reprojecting the shapefile"
    current_elapsed_time = (time.time() - start_time) / 60.0
    print "Total time for re-projection: {:,.2} minutes".format(current_elapsed_time)

    return None


def create_prj_file_with_gdal(path_to_shapefile, epsg_code):
    """
    path_to_shapefile:  str. A valid absolute or relative path
                        to an ESRI shapefile for which a new
                        '.prj' file should be written.
                        
    epsg_code:          int. Should specify a valid epsg code.
                        Will be used to determine the contents
                        of the output '.prj' file.
    ======================
    Returns:            None. Creates a '.prj' file at the same
                        location as path_to_shapefile with the
                        definition given by epsg_code.
    """
    try:
        from osgeo import osr
    except:
        msg = "Failed to import osgeo.osr.\nEnsure this library is installed."
        raise Exception(msg)

    try:
        assert path_to_shapefile[-4:] == ".shp"
    except AssertionError as e:
        print "Passed path_to_shapefile was:", path_to_shapefile
        print "Ensure that the path_to_shapefile ends in '.shp'."
        raise e
    
    # Get the desired projection from the epsg code
    # and format it as an ESRI projection
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromEPSG(epsg_code)
    spatial_ref.MorphToESRI()
    
    # Create the projection file path
    prj_filename = path_to_shapefile.replace(".shp", ".prj")
    
    # Write the projection file
    with open(prj_filename, 'wb') as prj_file:
        print "Writing the projection file to", prj_filename
        prj_file.write(spatial_ref.ExportToWkt())
        
    return None