import os

import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.stats import lognorm

from app.boundary_analysis import BoundaryAnalysis
from app.watershed.prioritization import WatershedPrioritization
from digitalarztools.pipelines.alos_palser import ALOSUtils
from digitalarztools.pipelines.srtm import SRTMUtils
from digitalarztools.raster.rio_raster import RioRaster
from digitalarztools.utils.logger import da_logger
from digitalarztools.vector.gpd_vector import GPDVector
from settings import MEDIA_DIR

from sklearn import preprocessing



if __name__ == "__main__":
    ba = BoundaryAnalysis()
    basin_gdv = ba.get_basin_gdv()
    basin_names = [
        'NARI RIVER BASIN'
        # ,'PISHIN LORA BASIN'
        # ,'HINGOL RIVER BASIN',
        # ,'HAMUN-E-MASHKHEL BASIN'
    ]
    selected_basin_gdv = basin_gdv.extract_sub_data('NAME', basin_names)
    # print(selected_basin_gdv.head())
    selected_basin_gdv.to_crs(epsg=4326)
    ws_prioritization = WatershedPrioritization(selected_basin_gdv)
    ws_prioritization.precipitation_surface(20)
    ws_prioritization.lulc_surface()
    ws_prioritization.population_surface()
    ws_prioritization.livestock_surface()
    ws_prioritization.water_streams()
    ws_prioritization.soil_moisture(8)
    ws_prioritization.combine_raster()

