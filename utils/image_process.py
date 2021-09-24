import torch
import numpy

class InverseNormalize :

    def __init__(self, mean, std) :
        self.mean = mean
        self.std = std

    def inverse_normalize_tensor(self, image) :
        ret_image = torch.zeros(image.shape)
        if len(image.shape) == 4:
            for i in range(3):
                ret_image[:, i, :, :] = (image[:, i, :, :]*self.std[i]) + self.mean[i]
        elif len(image.shape) == 3:
            for i in range(3):
                ret_image[i, :, :] = (image[i, :, :]*self.std[i]) + self.mean[i]
        else :
            assert (len(image.shape) <= 4) or (len(image.shape) >= 3), 'Input image''s length in wrong. Please check it.'
        return ret_image

    def inverse_normalize_ndarray(self, image) :
        ret_image = numpy.empty(image.shape)
        if len(image.shape) == 4:
            for i in range(3):
                ret_image[:, i, :, :] = (image[:, i, :, :]*self.std[i]) + self.mean[i]
        elif len(image.shape) == 3:
            for i in range(3):
                ret_image[i, :, :] = (image[i, :, :]*self.std[i]) + self.mean[i]
        else :
            assert (len(image.shape) <= 4) or (len(image.shape) >= 3), 'Input image''s length in wrong. Please check it.'
        return ret_image

    def run(self, image) :
        if type(image) == torch.Tensor : ret =  self.inverse_normalize_tensor(image)
        elif type(image) == numpy.ndarray : ret = self.inverse_normalize_ndarray(image)
        else : assert True, 'Wrong type of input image.'
        return ret