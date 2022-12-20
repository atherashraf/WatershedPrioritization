import os

import numpy as np
from matplotlib import pyplot as plt
from pysheds.pgrid import Grid

from digitalarztools.raster.rio_raster import RioRaster


class WatershedModel:
    def __init__(self, dem_path: str):
        raster = RioRaster(dem_path)
        is_changed = False
        if raster.get_dtype() != 'float32':
            raster.change_datatype(new_dtype=np.float32)
            is_changed = True
        if raster.get_nodata_value() is None:
            raster.add_nodata_value(0)
            is_changed = True
        if is_changed:
            os.remove(dem_path)
            raster.save_to_file(img_des=dem_path)
        # data = raster.get_data_array(band=1)
        self.grid = Grid.from_raster(dem_path, "dem")

    def execute(self):
        self.grid.dem = self.condition_dem(self.grid.dem)
        self.plot_data(self.grid.dem, "DEM", 'terrain')
        self.create_dir_acc_map()
        self.plot_data(self.grid.acc, "Direction Map", 'viridis')

    def condition_dem(self, dem):
        """
        :param dem: pyshed.View.Raster
        :return:
        """
        # Fill pits in DEM
        pit_filled_dem = self.grid.fill_pits(dem, inplace=False)
        # Fill depressions in DEM
        flooded_dem = self.grid.fill_depressions(pit_filled_dem, inplace=False)

        # Resolve flats in DEM
        inflated_dem = self.grid.resolve_flats(flooded_dem, inplace=False)
        return inflated_dem

    def plot_data(self, data, label="", cmap='terrain'):
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_alpha(0)

        plt.imshow(data, extent=self.grid.extent, cmap=cmap, zorder=1)
        plt.colorbar(label=label)  # label='Elevation (m)'
        plt.grid(zorder=0)
        # plt.title('Digital elevation map', size=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.show()

    def create_dir_acc_map(self):
        """
        Careate direction and flow accumalation map
        :return:
        """
        # Determine D8 flow directions from DEM
        # ----------------------
        # Specify directional mapping
        # N    NE    E    SE    S    SW    W    NW
        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
        # Compute flow directions
        # -------------------------------------
        self.grid.flowdir(data=self.grid.dem, out_name='dir', dirmap=dirmap)
        self.grid.accumulation(self.grid.dir, out_name="acc", dirmap=dirmap)


if __name__ == "__main__":
    dem = "media/srtm_data/nari_river_basin.tif"
    wsm = WatershedModel(dem)
    wsm.execute()
