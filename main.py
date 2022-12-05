import os.path
import traceback

import geopandas as gpd

from pipelines.alos_palser import AsfUtils
from pipelines.srtm import SRTMUtils
from settings import MEDIA_DIR

if __name__ == "__main__":

    boundary_path = os.path.join('media/data/boundaries/BolchistanBoundary.shp')
    boundary_df = gpd.read_file(boundary_path)
    if str(boundary_df.crs) != 'EPSG:4326':
        boundary_df.to_crs(epsg=4326, inplace=True)

    print("extent of province", boundary_df.bounds)

    # Alos palsar download
    # boundary = boundary_df['geometry'].values[0]
    # AsfUtils.download_alos_palsar(boundary.wkt)

    # SRTM download
    # bounds = boundary_df.bounds.to_dict()
    # SRTMUtils().extract_srtm_data(bounds)
    # print("done")

