"""Functions for parsing Amanzi/ATS XDMF visualization files."""
## mesh polygons functions written by Gabriel Perez (ORNL)

import sys
import os
import numpy as np
import h5py
import shapely


def get_mesh_polygons(mesh_file=None):
    """Extract mesh polygons from an ATS/Amanzi visualization file.
    
    Parameters
    ----------
    mesh_file : str
        Path to the HDF5 mesh file
        
    Returns
    -------
    list
        List of shapely Polygon objects representing mesh elements
    """
    # Read the mesh to extract the coordinates of the nodes
    with h5py.File(mesh_file, 'r') as f2:
        group = f2["/0"]
        nodes_coordinate = group['Mesh']['Nodes'][:]
        mixed_elements = group['Mesh']['MixedElements'][:]
        n_elements = group['Mesh']['ElementMap'].shape[0]

    type_element = np.zeros(n_elements)
    mesh_topology = np.zeros((n_elements, 7))
    
    for i in range(n_elements):
        type_element[i] = mixed_elements[0]
        if type_element[i] == 4:
            mesh_topology[i, 0:3] = mixed_elements[1:4].flatten()
            # Remove elements from mixed_elements
            mixed_elements = mixed_elements[4:]
        elif type_element[i] == 5:
            mesh_topology[i, 0:4] = mixed_elements[1:5].flatten()
            # Remove elements from mixed_elements
            mixed_elements = mixed_elements[5:]
        else:
            n_vert = mixed_elements[1].flatten()[0]
            mesh_topology[i, 0:n_vert] = mixed_elements[2:2+n_vert].flatten()
            # Remove elements from mixed_elements
            mixed_elements = mixed_elements[2+n_vert:]

    polygons = []
    for i in range(mesh_topology.shape[0]):
        # Extract the coordinates of the nodes that form the element
        nodes = mesh_topology[i, :]
        # Replace by -1 the zeros in column 4 and 5. This is done since there are some elements with node id equal to zero
        nodes[-4:][nodes[-4:] == 0] = -1
        # Remove the -1
        nodes = nodes[nodes != -1]
        nodes_element_coordinates = nodes_coordinate[nodes.astype(int), :]
        # Create a polygon
        polygon = shapely.geometry.Polygon(nodes_element_coordinates)
        polygons.append(polygon)
        
    return polygons
