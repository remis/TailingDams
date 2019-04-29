from tempfile import NamedTemporaryFile

import numpy as np
import gdal

import matplotlib.pyplot as plt


class GdalIO():
    def __init__(self):
        self.gdal_options = dict()

    def parse_meta_with_gdal(self, path: str):
        """to be used for parsing gdal headers and recreating them in output results
           based on https://www.gdal.org/gdal_tutorial.html
        """

        # Opening the File
        dataset = gdal.Open(path, gdal.GA_ReadOnly)

        # Getting Dataset Information
        self.gdal_options['driver'] = dataset.GetDriver()

        self.gdal_options['size'] = [dataset.RasterXSize, dataset.RasterYSize,  dataset.RasterCount]

        self.gdal_options['projection'] = dataset.GetProjection()
        geotransform = dataset.GetGeoTransform()
        self.gdal_options['geotransform'] = geotransform

        dataset = None

    def write_image(self, path, image):
        # image is self.optical_rgb.shape[0] X self.optical_rgb.shape[1] in this case
        driver = self.gdal_options['driver']
        if not driver:
            raise Exception("driver not created")
        if image.ndim == 3:
            bands = image.shape[2]
        elif image.ndim == 2:
            bands = 1
        else:
            raise Exception("Bands number incorrect")
        dst_ds = driver.Create(path, xsize=image.shape[0], ysize=image.shape[1], bands=bands, eType=gdal.GDT_Byte)

        geotransform = self.gdal_options['geotransform']
        dst_ds.SetGeoTransform(geotransform)
        projection = self.gdal_options['projection']
        dst_ds.SetProjection(projection)
        raster = image.astype(np.uint8)
        if image.ndim == 3:
            for band_ind in range(bands):
                dst_ds.GetRasterBand(band_ind + 1).WriteArray(raster[:, :, band_ind])
        elif image.ndim == 2:
            dst_ds.GetRasterBand(1).WriteArray(raster)
        dst_ds = None

    def write_surface(self, path, image):
        #todo use gdal dem

        cmap = plt.get_cmap('jet')
        rgba_img_faults = cmap(image)
        rgb_img_faults = np.delete(rgba_img_faults, 3, 2)
        rgb_img_faults=(rgb_img_faults[:, :, :3] * 255).astype(np.uint8)
        self.write_image(path, rgb_img_faults)

        im = plt.imshow(image, cmap=cmap)
        plt.colorbar(im)
        plt.savefig('{}.png'.format(path))
        plt.close('all')

