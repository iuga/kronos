from PIL import Image
import numpy as np


class Images(object):

    RESIZE_METHODS = {
        'bilinear': Image.BILINEAR,
        'nearest': Image.NEAREST,
        'lanczos': Image.LANCZOS,
        'bicubic': Image.BICUBIC
    }

    NORMALIZATION_ZERO_ONE = 'zero_one'
    NORMALIZATION_MINUS_PLUS_ONE = 'minus_plus_one'

    NORMALIZATION_METHODS = [
        NORMALIZATION_ZERO_ONE, NORMALIZATION_MINUS_PLUS_ONE
    ]

    def load(self, filename):
        """
        Load an image into PIL format
        """
        self.img = Image.open(filename)
        self.img = self.img.convert('RGB')
        return self

    def save(self, filename='/tmp/out.jpg'):
        """
        Saves this image under the given filename. The format to use is determined from the filename extension.
        """
        self.img.save(filename)
        return self

    def describe(self):
        """
        Print some useful information for debugging
        """
        print("Size: {}".format(self.img.size))
        return self

    def to_array(self, normalization=False, mean_normalized=False):
        """
        Return a NumpyArray with (height, width, channel) format.

        normalization: If False/None/Empty no normalization will be applied. Otherwise, the method should be passed.
            normalization methods:
            - zero_one: All the values will be normalized between [0, 1]
            - minus_plus_one: All the values will be normalized between [-1, 1]

        mean_normalized: Mean normalized perform a channel normalization. E.g: (123.68, 116.779, 103.939).
        """
        # Validate the normalization method
        if normalization and normalization not in self.NORMALIZATION_METHODS:
            raise ValueError("Invalid Normalization method. Valid values: {}".format(self.NORMALIZATION_METHODS.keys()))
        # Numpy array x has format (height, width, channel)
        # but original PIL image has format (width, height, channel)
        the_image_array = np.asarray(self.img, dtype='int16')

        if normalization:
            the_image_array = the_image_array.astype('float16')

        if mean_normalized:
            if len(mean_normalized) != 3:
                raise ValueError("mean_normalized should have shape 3 for (r,g,b)")
            the_image_array[:, :, 0] -= mean_normalized[0]
            the_image_array[:, :, 1] -= mean_normalized[1]
            the_image_array[:, :, 2] -= mean_normalized[2]

        if normalization == self.NORMALIZATION_ZERO_ONE:
            # All values are between 0 and 1
            the_image_array /= 255.

        if normalization == self.NORMALIZATION_MINUS_PLUS_ONE:
            # All values are between -1 and 1
            the_image_array /= 255.
            the_image_array -= 0.5
            the_image_array *= 2.

        return the_image_array

    def resize(self, width=224, height=224, method='bilinear'):
        """
        Resize this image to the given size using the defined method.
        """
        self.img = self.img.resize(size=(width, height), resample=self.RESIZE_METHODS.get(method, Image.BILINEAR))
        return self

    def central_crop(self, central_fraction=0.50):
        """
        Crop the central region of the image.
        Remove the outer parts of an image but retain the central region of the image along each dimension.
        If we specify central_fraction = 0.5, this function returns the region marked with "X" in the below diagram.

         --------
        |        |
        |  XXXX  |
        |  XXXX  |
        |        |   where "X" is the central 50% of the image.
         --------
        """
        w, h = self.img.size
        nw, nh = w * central_fraction, h * central_fraction

        left = np.ceil((w - nw) / 2.)
        top = np.ceil((h - nh) / 2.)
        right = np.floor((w + nw) / 2)
        bottom = np.floor((h + nh) / 2)

        self.img = self.img.crop((left, top, right, bottom))
        return self

    def centered_crop(self, width, height):
        """
        Crop the image to the new size keeping the content in the center.
        Remove the outer parts of an image but retain the central region of the image along each dimension.

         --------
        |        |
        |  XXXX  |
        |  XXXX  | where "X" has (width, height) size
        |        |
         --------
        """
        w, h = self.img.size
        nw, nh = width, height

        if width > w:
            width = w

        if height > h:
            height = h

        left = np.ceil((w - nw) / 2.)
        top = np.ceil((h - nh) / 2.)
        right = np.floor((w + nw) / 2)
        bottom = np.floor((h + nh) / 2)

        self.img = self.img.crop((left, top, right, bottom))
        return self

    def pad_to_square(self):
        """
        Creates a padding in the shorter side with 0 (black) until the image is squared.
        The image size will be (longer_side_size, longer_side_size, 3)
        """
        longer_side = max(self.img.size)
        horizontal_padding = (longer_side - self.img.size[0]) / 2
        vertical_padding = (longer_side - self.img.size[1]) / 2
        self.img = self.img.crop(
            (
                -horizontal_padding,
                -vertical_padding,
                self.img.size[0] + horizontal_padding,
                self.img.size[1] + vertical_padding
            )
        )
        return self
