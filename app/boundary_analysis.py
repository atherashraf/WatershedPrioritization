import os
import geopandas as gpd
import pandas as pd

from digitalarztools.vector.gpd_vector import GPDVector
from settings import BASE_DIR


class BoundaryAnalysis:
    def __init__(self):
        self.data_file_path = os.path.join(BASE_DIR, 'media/data/balochistan_data.gpkg')

    def get_province_gdv(self) -> 'GPDVector':
        gdv = GPDVector.from_gpkg(self.data_file_path, layer='province')
        return gdv

    def get_basin_gdv(self) -> 'GPDVector':
        gdv = GPDVector.from_gpkg(self.data_file_path, layer='basin')
        return gdv

    def get_district_gdv(self) -> 'GPDVector':
        gdv = GPDVector.from_gpkg(self.data_file_path, layer='districts')
        return gdv

    def add_boundary_2_gpkg(self, shp_file: str, layer_name: str):
        gpd_vector = GPDVector.from_shp(shp_file)
        gpd_vector.to_gpkg(self.data_file_path, layer=layer_name)

    @classmethod
    def get_province_extent_4326(cls):
        province_df = cls.get_province_df()
        if str(province_df.crs) != 'EPSG:4326':
            province_df.to_crs(epsg=4326, inplace=True)

        print("extent of province", province_df.bounds)
        return province_df.bounds

    @staticmethod
    def district_information():
        data_info = os.path.join(BASE_DIR, 'media/data/data_info.xlsx')
        df = pd.read_excel(data_info, sheet_name='district_table')
        print(df.to_latex(index=False))

    @staticmethod
    def create_district_boundary():
        data_info = os.path.join(BASE_DIR, 'media/data/temp/PakistanDistrictBoundaries.shp')
        da_gdf: GPDVector = GPDVector.from_shp(data_info)
        # da_gdf = da_gdf.extract_sub_data(col_name='UNIT', col_val='Baluchistan')
        # da_gdf.head()
        src_file = os.path.join(BASE_DIR, 'media/data/districts.xlsx')
        df = pd.read_excel(src_file)
        df['geometry'] = df['DISTRICT'].apply(lambda x: da_gdf.get_geometry('DISTRICT', x))

        gpd_vector = GPDVector.from_df(df, crs=da_gdf.get_crs())
        des_file = os.path.join(BASE_DIR, 'media/data/bol_boundaries.gpkg')
        gpd_vector.to_gpkg(des_file, layer='districts')
