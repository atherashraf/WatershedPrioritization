import datetime
import math
import os

import numpy as np
import pandas as pd
import geopandas as gpd

from digitalarztools.datasets.hydro import Hydro
from digitalarztools.datasets.lulc import LandUseLandCover
from digitalarztools.datasets.precipitation import Precipitation
from digitalarztools.datasets.soil_moisture import SoilMoisture
from digitalarztools.pipelines.fao_livestock import GWL3
from digitalarztools.pipelines.gee.core.image import GEEImage
from digitalarztools.raster.rio_raster import RioRaster
from settings import CONFIG_DIR, MEDIA_DIR
from digitalarztools.distributions.GumbleDistribution import GumbelDistribution
from digitalarztools.io.file_io import FileIO
from digitalarztools.pipelines.gee.core.auth import GEEAuth
from digitalarztools.pipelines.gee.core.region import GEERegion
from digitalarztools.raster.band_process import BandProcess
from digitalarztools.raster.rio_process import RioProcess
from digitalarztools.vector.gpd_vector import GPDVector


class WatershedPrioritization:
    def __init__(self, river_basin: GPDVector):
        self.final_raster = None
        self.river_basin = river_basin
        self. normalized_rasters = {}
        self.mca_dir = os.path.join(MEDIA_DIR, 'mca_data')


    def get_aoi(self) -> gpd.GeoDataFrame:
        aoi = self.river_basin.get_unary_union()
        gdf = gpd.GeoDataFrame(geometry=[aoi], crs=self.river_basin.get_crs())
        return gdf

    def get_gee_auth_region(self):
        gee_service_account_fp = os.path.join(CONFIG_DIR, 'goolgle-earth-engine-3bf316104d2c.json')
        gee_auth = GEEAuth.geo_init_personal('atherashraf@gmail.com', gee_service_account_fp)
        geojson = self.river_basin.to_goejson()
        region = GEERegion.from_geojson(geojson)
        return gee_auth, region

    def lulc_surface(self):
        """
        :return:
        """
        print("processing lulc probability surface...")
        lulc_dir = os.path.join(MEDIA_DIR, 'mca_data', 'lulc')
        output_dir = os.path.join(lulc_dir, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        esa_fp = os.path.join(lulc_dir, 'esa_world_cover.tif')
        if not os.path.exists(esa_fp):
            gee_auth, region = self.get_gee_auth_region()
            img: GEEImage = LandUseLandCover.esa_world_cover_using_gee(gee_auth, region)
            img.download_image(esa_fp, img_region=region)
            print("ESA World Cover data downloaded....")
        """
            Extrect landcover classes, vegetation, water (), and build up
                    
        """
        raster = RioRaster(esa_fp)
        # spatial_res_meter = raster.get_spatial_resoultion(in_meter=True)
        # count_value = BandProcess.get_value_area_data(raster.get_data_array(1), raster.get_nodata_value(), spatial_res_meter )
        raster.resample_raster_res(1000 / (110 * 1000))
        spatial_res_meter = raster.get_spatial_resoultion(in_meter=True)
        res = math.floor(spatial_res_meter[0] / 1000)
        # count_value_km = BandProcess.get_value_area_data(raster.get_data_array(1), raster.get_nodata_value(), spatial_res_meter)

        # raster.clip_raster(self.river_basin.get_unary_union(), crs=self.river_basin.get_crs())
        surfaces = {
            "builtup": {
                "pixels": [50],
                "fp": os.path.join(output_dir, f'builtup_normalized_{res}.tif')
            },
            "vegetation": {
                "pixels": [20, 30, 40],
                "fp": os.path.join(output_dir, f'vegetation_normalized_{res}.tif')
            },
            "water_bodies": {
                "pixels": [80, 90, 95],
                "fp": os.path.join(output_dir, f'water_bodies_normalized_{res}.tif')
            }
        }
        """
        creating inverse distance surfaces normalize to 0-1 
        1 mean lc pixel <1 mean far from  lc 
        """
        no_data = raster.get_nodata_value()
        # water_bodies_raster,  vegetation_raster, built_up_raster = None, None, None
        for key in surfaces:
            surface = surfaces[key]
            # print("spatial res in meter", spatial_res_meter)
            if not os.path.exists(surface["fp"]):
                data = BandProcess.create_distance_raster(raster.get_data_array(1), surface["pixels"],
                                                          spatial_res_meter[0] / 1000)
                data = BandProcess.min_max_normalization(data, no_data, is_inverse=True)
                raster = raster.rio_raster_from_array(data)
                raster.clip_raster(self.get_aoi())
                raster.save_to_file(surface["fp"])
            else:
                raster = RioRaster(surface["fp"])
            self.normalized_rasters[key] = raster


    def precipitation_surface(self, no_of_year: int):
        """
        creating probability precipitation surface for specified no of year
        :param no_of_year:
        :return:
        """
        print("processing precipitation probability surface...")
        """
            downloading CHIRPS data for specified no of years
        """
        end_date = datetime.datetime.now()
        year = end_date.year
        chirps_dir_path = os.path.join(MEDIA_DIR, 'mca_data', 'chirps')
        if not os.path.exists(chirps_dir_path):
            os.makedirs(chirps_dir_path)
        gee_auth = None
        region = None
        for i in range(no_of_year):
            year = year - 1 if i != 0 else year
            fp = os.path.join(chirps_dir_path, f'chirps_yearly_avg_{year}.tif')
            if not os.path.exists(fp):
                start_date = f'{year}-01-01'
                end_date = f'{year}-12-31' if i != 0 else end_date.strftime('%Y-%m-%d')
                if gee_auth is None:
                    gee_auth, region = self.get_gee_auth_region()

                print(start_date, end_date)
                gee_img = Precipitation.chirps_data_using_gee(gee_auth, region, start_date, end_date)

                gee_img.download_image(fp, region, scale=5500)
        """
         creating probability raster
        """
        output_dir = os.path.join(chirps_dir_path, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        """
        """
        probability_raster_fp = os.path.join(output_dir, f"rp_raster_{no_of_year}years.tif")
        if not os.path.exists(probability_raster_fp):
            """
             creating stack raster
            """
            files = FileIO.list_files_in_folder(chirps_dir_path, 'tif')

            def get_band_name(fp):
                name, ext = FileIO.get_file_name_ext(fp)
                year = name.split("_")[-1]
                return year

            raster = RioProcess.stack_bands(files, get_band_name)
            clip_polygon = self.river_basin.get_unary_union()
            raster.clip_raster(aoi=clip_polygon, crs=self.river_basin.get_crs())
            """
            calculate Gumbel distribution using max value
            """
            summaries_df = raster.get_band_summaries()
            summaries_df["year"] = pd.to_datetime(summaries_df.index)
            df = summaries_df[["year", "max"]]
            # df["year"] = pd.to_datetime(df["year"])
            df.index = df["year"]
            # print(df)
            gd = GumbelDistribution(df, event_col="max", extract_yearly_data=True)

            """
            creating average rainfall raster of all downloaded years (20)
            """

            data = raster.get_data_array()
            orig_shape = data.shape
            data = data.reshape((orig_shape[0], orig_shape[1] * orig_shape[2]))
            data = data.mean(axis=0)
            avg_raster = raster.rio_raster_from_array(data.reshape((orig_shape[1], orig_shape[2])))
            avg_raster.save_to_file(os.path.join(output_dir, f"avg_raster_{no_of_year}years.tif"))

            """
            calculate probability of avg rainfall
            """

            rp_data = gd.calculate_event_probability(data)
            rp_data = np.array(rp_data)
            # rp_data = rp_data.reshape((orig_shape[1],orig_shape[2]))
            stretched_rp_data = BandProcess.min_max_normalization(rp_data, raster.get_nodata_value())
            final_raster = raster.rio_raster_from_array(stretched_rp_data.reshape((orig_shape[1], orig_shape[2])))
            final_raster.resample_raster_res(1000 / (110 * 1000))
            final_raster.clip_raster(clip_polygon, crs=self.river_basin.get_crs())
            final_raster.save_to_file(probability_raster_fp)
            # df = pd.DataFrame({"avg_rainfall": data, "return_period": rp_data, "stretched": stretched_rp_data})
        else:
            final_raster = RioRaster(probability_raster_fp)
            # print("precipitation probability raster already available")
        self.normalized_rasters["precipitation"] = final_raster

    def population_surface(self):
        print("processing population surface")
        output_dir = os.path.join(MEDIA_DIR, 'mca_data', "population")
        # raster.save_to_file(os.path.join(output_dir, "landsacn_2019.tif"))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_fp = os.path.join(output_dir, "landscan_normalized_2019.tif")
        if not os.path.exists(out_fp):
            fp = "/Users/atherashraf/Documents/data/Landscan/landscan-global-2019-assets/landscan-global-2019.tif"
            raster = RioRaster(fp)
            raster.clip_raster(self.get_aoi())

            data = raster.get_data_array(1)  # .astype(np.float32)
            no_data = raster.get_nodata_value()
            data[data <= 30] = no_data
            normalized_data = BandProcess.logarithmic_normalization(data, no_data)
            self.norm_pop_raster = raster.rio_raster_from_array(normalized_data)
            self.norm_pop_raster.clip_raster(self.get_aoi())
            self.norm_pop_raster.save_to_file(out_fp)
        else:
            self.norm_pop_raster = RioRaster(out_fp)


    def livestock_surface(self):
        dir = os.path.join(MEDIA_DIR, 'mca_data', 'livestock')
        if not os.path.exists(dir):
            os.makedirs(dir)
        fp = os.path.join(dir, 'foa_gwl3.tif')
        norm_fp = os.path.join(dir, 'norm_fao_gwl3.tif')
        if not os.path.exists(norm_fp):
            aoi = self.get_aoi()
            bbox = list(aoi.bounds.to_numpy()[0])

            gwl3_raster: RioRaster = GWL3.download_grid(bbox, fp)
            data = BandProcess.min_max_normalization(gwl3_raster.get_data_array(1),
                                                     no_data=gwl3_raster.get_nodata_value())
            final_raster = gwl3_raster.rio_raster_from_array(data)
            final_raster.resample_raster_res(des_resolution=1000 / (110 * 1000))
            final_raster.clip_raster(aoi=aoi)
            final_raster.save_to_file(norm_fp)
        else:
            final_raster = RioRaster(norm_fp)

        self.normalized_rasters["live_stock"] = final_raster


    def water_streams(self):

        hydro_dir = os.path.join(MEDIA_DIR, 'mca_data', 'hydro')
        if not os.path.exists(hydro_dir):
            os.makedirs(hydro_dir)
        dist_fp = os.path.join(hydro_dir, 'norml_dist_raster.tif')
        if not os.path.exists(dist_fp):
            tif_fp = os.path.join(hydro_dir, 'drainage_raster.tif')
            if not os.path.exists(tif_fp):
                drain_file = os.path.join(hydro_dir, 'drainage.shp')
                gdv = GPDVector.from_shp(drain_file, srid=4326)
                aoi = self.get_aoi()
                gdv.clip_data(aoi)
                gdv.add_class_id(1)
                raster = gdv.to_raster(1000/(110*1000),value_col='cls_id')
                raster.save_to_file(tif_fp)
            else:
                raster = RioRaster(tif_fp)
            res = raster.get_spatial_resoultion(True)
            distance_data = BandProcess.create_distance_raster(raster.get_data_array(1),[1],res[0]/1000)
            data = BandProcess.min_max_normalization(distance_data, 0, is_inverse=True)
            final_raster = raster.rio_raster_from_array(data)
            final_raster.clip_raster(self.get_aoi())
            final_raster.save_to_file(dist_fp)
        else:
            final_raster = RioRaster(dist_fp)
        self.normalized_rasters['streams'] = final_raster


    def soil_moisture(self, no_of_year):
        """
        datasets url: https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_007#bands
        :return:
        """
        end_date = datetime.datetime.now()
        year = end_date.year
        smap_dir = os.path.join(MEDIA_DIR, 'mca_data', 'soil_moisture')
        if not os.path.exists(smap_dir):
            os.makedirs(smap_dir)
        gee_auth = None
        region = None
        for i in range(no_of_year):
            year = year - 1 if i != 0 else year
            fp = os.path.join(smap_dir, f'smap_yearly_avg_{year}.tif')
            if not os.path.exists(fp):
                start_date = f'{year}-01-01'
                end_date = f'{year}-12-31' if i != 0 else end_date.strftime('%Y-%m-%d')
                if gee_auth is None:
                    gee_auth, region = self.get_gee_auth_region()

                print(start_date, end_date)
                gee_img = SoilMoisture.smap_data_using_gee(gee_auth, region, start_date, end_date)
                gee_img.download_image(fp, region, scale=11000)

        """
         creating probability raster
        """
        output_dir = os.path.join(smap_dir, 'output')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        probability_raster_fp = os.path.join(output_dir, f"rp_raster_{no_of_year}years.tif")
        if not os.path.exists(probability_raster_fp):
            # creating stack raster
            files = FileIO.list_files_in_folder(smap_dir, 'tif')

            """
            stacking all avg yearly raster 
            """
            def get_band_name(fp):
                name, ext = FileIO.get_file_name_ext(fp)
                year = name.split("_")[-1]
                return year

            raster = RioProcess.stack_bands(files, get_band_name)
            clip_polygon = self.river_basin.get_unary_union()
            raster.clip_raster(aoi=clip_polygon, crs=self.river_basin.get_crs())

            """
            calculating Gumbel distribution against max value of stack
            
            """
            summaries_df = raster.get_band_summaries()
            summaries_df["year"] = pd.to_datetime(summaries_df.index)
            df = summaries_df[["year", "max"]]
            # df["year"] = pd.to_datetime(df["year"])
            df.index = df["year"]
            print(df)
            gd = GumbelDistribution(df, event_col="max", extract_yearly_data=True)

            """
             creating average soil moisture raster of all downloaded years
            """

            data = raster.get_data_array()
            orig_shape = data.shape
            data = data.reshape((orig_shape[0], orig_shape[1] * orig_shape[2]))
            data = data.mean(axis=0)
            avg_raster = raster.rio_raster_from_array(data.reshape((orig_shape[1], orig_shape[2])))
            avg_raster.save_to_file(os.path.join(output_dir, f"avg_raster_{no_of_year}years.tif"))

            """
            calculate event probability of avg soil moisture
            """
            rp_data = gd.calculate_event_probability(data)
            rp_data = np.array(rp_data)
            """
                min max stretch and resampling of final raster
            """
            stretched_rp_data = BandProcess.min_max_normalization(rp_data, raster.get_nodata_value())
            final_raster = raster.rio_raster_from_array(stretched_rp_data.reshape((orig_shape[1], orig_shape[2])))
            final_raster.resample_raster_res(1000 / (110 * 1000))
            final_raster.clip_raster(clip_polygon, crs=self.river_basin.get_crs())
            final_raster.save_to_file(probability_raster_fp)
            # df = pd.DataFrame({"avg_rainfall": data, "return_period": rp_data, "stretched": stretched_rp_data})
        else:
            final_raster = RioRaster(probability_raster_fp)
        self.normalized_rasters['soil_moisture'] = final_raster

    def combine_raster(self):
        print("combing all raster as weighted average")
        ref_raster: RioRaster
        combine_data: np.ndarray
        for idx, key in enumerate(self.normalized_rasters):
            if idx == 0:
                ref_raster = self.normalized_rasters[key]
                combine_data = np.zeros(ref_raster.get_img_resolution())
            else:
                raster: RioRaster = self.normalized_rasters[key]
                aoi = self.get_aoi()
                raster.clip_raster(aoi)
                raster.make_coincident_with(ref_raster)

                data = raster.get_data_array(1)
                combine_data = (data + combine_data)/2
        self.final_raster = ref_raster.rio_raster_from_array(combine_data)
        self.final_raster.save_to_file(os.path.join(self.mca_dir, 'combined_res.tif'))

