import datetime
import os

import numpy as np
import pandas as pd

from digitalarztools.datasets.lulc import LandUseLandCover
from digitalarztools.datasets.precipitation import Precipitation
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
        self.river_basin = river_basin

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
        # raster.resample_raster_res(1000/(110*1000))
        raster.clip_raster(self.river_basin.get_unary_union(), crs=self.river_basin.get_crs())
        vegetation_pixels = [20, 30, 40]
        built_up_pixels = [50]
        water_bodies = [80, 90, 95]
        """
        creating vegetation distance raster
        """
        vegetation_fp = os.path.join(output_dir, 'vegetation.tif')
        spatial_res_meter = raster.get_spatial_resoultion(in_meter=True)
        print("spatial res in meter", spatial_res_meter)
        if not os.path.exists(vegetation_fp):
            vegetation_data = BandProcess.create_distance_raster(raster.get_data_array(1), vegetation_pixels,
                                                                 spatial_res_meter[0] / 1000)
            vegetation_data = BandProcess.min_max_stretch(vegetation_data, raster.get_nodata_value(),
                                                          stretch_range=(0, 1), is_inverse=True)
            veg_raster = raster.rio_raster_from_array(vegetation_data)
            veg_raster.save_to_file(vegetation_fp)
        """
            creating built up distance raster
        """
        # built_up_fp = os.path.join(output_dir, 'built_up.tif')
        # if not os.path.exists(built_up_fp):
        #     built_up_data = BandProcess.get_boolean_raster(raster.get_data_array(1), built_up_pixels)
        #     bu_raster = raster.rio_raster_from_array(built_up_data)
        #     bu_raster.save_to_file(built_up_fp)

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

        probability_raster_fp = os.path.join(output_dir, f"rp_raster_{no_of_year}years.tif")
        if not os.path.exists(probability_raster_fp):
            # creating stack raster
            files = FileIO.list_files_in_folder(chirps_dir_path, 'tif')

            def get_band_name(fp):
                name, ext = FileIO.get_file_name_ext(fp)
                year = name.split("_")[-1]
                return year

            raster = RioProcess.stack_bands(files, get_band_name)
            clip_polygon = self.river_basin.get_unary_union()
            raster.clip_raster(aoi=clip_polygon, crs=self.river_basin.get_crs())

            # calculate Gumbel distribution
            summaries_df = raster.get_band_summaries()
            summaries_df["year"] = pd.to_datetime(summaries_df.index)
            df = summaries_df[["year", "max"]]
            # df["year"] = pd.to_datetime(df["year"])
            df.index = df["year"]
            print(df)
            gd = GumbelDistribution(df, event_col="max", extract_yearly_data=True)

            # creating average rainfall raster of all downloaded years (20)
            data = raster.get_data_array()
            orig_shape = data.shape
            data = data.reshape((orig_shape[0], orig_shape[1] * orig_shape[2]))
            data = data.mean(axis=0)
            avg_raster = raster.rio_raster_from_array(data.reshape((orig_shape[1], orig_shape[2])))
            avg_raster.save_to_file(os.path.join(output_dir, f"avg_raster_{no_of_year}years.tif"))

            # calculate probability of avg rainfall
            rp_data = gd.calculate_event_probability(data)
            rp_data = np.array(rp_data)
            # rp_data = rp_data.reshape((orig_shape[1],orig_shape[2]))
            stretched_rp_data = BandProcess.min_max_stretch(rp_data, raster.get_nodata_value(), (0, 1))
            rp_raster = raster.rio_raster_from_array(stretched_rp_data.reshape((orig_shape[1], orig_shape[2])))
            rp_raster.resample_raster_res(1000 / (110 * 1000))
            rp_raster.clip_raster(clip_polygon, crs=self.river_basin.get_crs())
            rp_raster.save_to_file(probability_raster_fp)
            # df = pd.DataFrame({"avg_rainfall": data, "return_period": rp_data, "stretched": stretched_rp_data})
        else:
            print("precipitation probability raster already available")
        print("dane")
