import pandas as pd

from app.boundary_analysis import BoundaryAnalysis

if __name__ == "__main__":
    # ba = BoundaryAnalysis()
    # basin_gdv = ba.get_basin_gdv()
    # vals = ['NARI RIVER BASIN',
    #         'PISHIN LORA BASIN',
    #         'HINGOL RIVER BASIN',
    #         'HAMUN-E-MASHKHEL BASIN']
    # selected_basin_gdv =  basin_gdv.extract_sub_data('NAME', vals)
    # district_gdv = ba.get_district_gdv()
    # res_gdv = district_gdv.spatial_join(selected_basin_gdv.get_gdf())

    # gdv.add_area_col()
    # gdv= gdv.select_columns(['FID_', 'NAME', 'area'])
    des_path = 'media/data/tables/MCARiverBasin.xlsx'
    df = pd.read_excel(des_path, sheet_name='MCAWatershed')
    print(df.to_latex(index=False))
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
    # boundary = boundary_df['geometry'].values[0]
    # AsfUtils.download_alos_palsar(boundary.wkt)

    # SRTM download
    # bounds = boundary_df.bounds.to_dict()
    # SRTMUtils().extract_srtm_data(bounds)
    # print("done")
