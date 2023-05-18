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


def population_data(boundary: GPDVector):
    fp = "/Users/atherashraf/Documents/data/Landscan/landscan-global-2019-assets/landscan-global-2019.tif"
    raster = RioRaster(fp)
    raster.clip_raster(boundary_gdv.gdf)
    output_dir = os.path.join(MEDIA_DIR, "population")
    # raster.save_to_file(os.path.join(output_dir, "landsacn_2019.tif"))
    data = raster.get_data_array(1)  # .astype(np.float32)
    data[data == raster.get_nodata_value()] = 0
    data[data <= 30] = 0

    max_val = np.nanmax(data)
    min_val = np.nanmin(data)
    # res = data.flatten()
    c = 1 / np.log(1 + max_val)
    normalized_data = c * np.log(1 + data)
    # normalized_data = lognorm.pdf(res, params[0], loc=params[1], scale=params[2])
    # normalized_data = normalized_data.reshape(data.shape)
    raster.rio_raster_from_array(normalized_data).save_to_file(os.path.join(output_dir, "landscan_normal_2019.tif"))


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

    # district_gdv = ba.get_district_gdv()
    # res_gdv = district_gdv.spatial_join(selected_basin_gdv.get_gdf())
    # print(district_gdv.head(5))
    # gdv.add_area_col()
    # gdv= gdv.select_columns(['FID_', 'NAME', 'area'])
    # des_path = 'media/data/tables/MCARiverBasin.xlsx'
    # df = pd.read_excel(des_path, sheet_name='MCAWatershed')
    # print(df.to_latex(index=False))
    # res_gdv.to_excel(des_path)
    # df = pd.read_excel(des_path)
    # df.reset_index(drop=True, inplace=True)
    # print(df.to_latex(index=False))
    # province_path = 'media/data/boundaries/BolchistanBoundary.shp'
    # BoundaryAnalysis.get_province_extent_4326()
    # BoundaryAnalysis.district_information()
    # basin_file = 'media/data/boundaries/Basins Balochistan.shp'
    # ba.add_boundary_2_gpkg(basin_file, 'basin')

    # Alos palsar download
    basin_gdv.to_crs(4326)
    i = 0
    boundary_gdv = basin_gdv.extract_sub_data('NAME', [basin_names[i]])
    # boundary = boundary_df['geometry'].values[0]

    # img_des = ALOSUtils.download_alos_palsar(boundary, basin_names[i].lower().replace(" ", "_"), 3000)

    # SRTM download
    # bounds = boundary_df.bounds.to_dict()
    # selected_basin = basin_gdv.extract_sub_data("NAME", [basin_names[0]])
    # da_logger.debug(selected_basin.head())
    # selected_basin.to_crs(4326)
    # bounds = selected_basin.gdf.bounds.iloc[0].to_dict()
    # srtm_raster = SRTMUtils().extract_srtm_data(bounds)
    # selected_basin = selected_basin.apply_buffer(2000)
    # srtm_raster = srtm_raster.clip_raster(selected_basin.gdf,selected_basin.get_crs())
    # img_des = os.path.join(os.path.join(MEDIA_DIR, 'srtm_data'), f'{basin_names[0]}.tif')
    # srtm_raster.save_to_file(img_des)

    """
        Population
    """
    # population_data(boundary_gdv)
