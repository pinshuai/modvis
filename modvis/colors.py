import matplotlib
import matplotlib.colors
import matplotlib.cm
import numpy as np


def colors(ColorName = None):
    """color list lookup table. Default is flatui"""
    color_lists = {
        # color blind friendly
    'CB_color_cycle' : ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00'],
    # colors from flat UI plattee
    'flatui' : ["#9b59b6", "#3498db", "#95a5a6", 
              "#e74c3c", "#34495e", "#2ecc71"],
    # default line plot colors in matplotlib
    'matplotlib' :['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b',
                 '#e377c2','#7f7f7f','#bcbd22','#17becf']
}
    if ColorName is None:
        return color_lists['flatui']
    else:
        return color_lists[ColorName]
#
# Lists of disparate color palettes
#
enumerated_palettes = {
    1 : ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'],
    2 : ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6',
         '#6a3d9a','#ffff99','#b15928'],
    3 : ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666'],
    }

