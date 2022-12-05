### https://search.asf.alaska.edu/
import os.path
from multiprocessing import Pool

import geopandas as gpd
import asf_search as asf
from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from settings import EARTHSAT_USER, EARTHSAT_PASSWORD, MEDIA_DIR


class AsfUtils:
    @staticmethod
    def get_date_range(no_of_months: int, rel_date=date.today(), is_end=True):
        if is_end:
            end_date = rel_date
            start_date = end_date + relativedelta(months=-no_of_months)
        else:
            start_date = rel_date
            end_date = start_date + relativedelta(months=+no_of_months)
        return start_date, end_date

    @classmethod
    def download_alos_palsar(cls, aoi_wkt: str):
        print("sarching for Alos palsar data")
        start_date, end_date = cls.get_date_range(18)

        options = {
            'intersectsWith': 'POLYGON((60.8274 24.712,67.3243 24.712,67.3243 30.6505,60.8274 30.6505,60.8274 24.712))',
            'platform': 'ALOS',
            'instrument': 'PALSAR',
            'processingLevel': [
                # 'RTC_LOW_RES',
                'RTC_HI_RES'
            ],
            # 'flightDirection': 'Descending',
            # 'maxResults': 250
            # 'start': start_date,
            # 'end': end_date
        }
        results = asf.geo_search(**options)
        res_df = gpd.read_file(str(results), driver='GeoJSON')

        # getting latest datasets
        res_df = res_df.sort_values('startTime', ascending=False).drop_duplicates(['geometry'])
        # res_df['startTime'] = res_df['startTime'].apply(lambda a: pd.to_datetime(a).date())
        # res_df['stopTime'] = res_df['stopTime'].apply(lambda a: pd.to_datetime(a).date())

        print('total tiles:', res_df.shape)
        urls = list(res_df['url'].values)
        session = asf.ASFSession()
        session.auth_with_creds(EARTHSAT_USER, EARTHSAT_PASSWORD)
        output_path = os.path.join(MEDIA_DIR, 'alos_palsar_data')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # res_df.to_excel(os.path.join(output_path, 'search_result.xlsx'))
        res_file = os.path.join(output_path,'search_result.gpkg')
        res_df.to_file(res_file, layer='alos_palsar_rtc_hi_res', driver="GPKG")
        for i in tqdm(range(len(urls)), desc='ALOS PALSAR Downloading'):
            asf.download_url(url=urls[i], path=output_path, session=session)
        print("\n Downloading finished")
