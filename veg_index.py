import numpy as np

class VegetationIndices:
    """ Class to compute vegetation indices """
    def __init__(self, red, green, blue, nir):
        """ Initialize with bands """
        self.red=red
        self.green=green
        self.blue=blue
        self.nir=nir

    def compute_ndvi(self):
        """ Computation of Normalized Difference Vegetation Index """
        numerator = (self.nir - self.red)
        denominator = (self.nir + self.red)
        
        ndvi=np.zeros_like(denominator)
        
        mask=denominator > 0
        
        ndvi[mask]=numerator[mask]/denominator[mask]

        return ndvi
    
    def compute_gndvi(self):
        """ Computation of Green Normalized Difference Vegetation Index """
        numerator = (self.nir - self.green)
        denominator = (self.nir + self.green)
        
        ndvi=np.zeros_like(denominator)
        
        mask=denominator > 0
        
        ndvi[mask]=numerator[mask]/denominator[mask]

        return ndvi
    
    def compute_evi(self):
        num=(self.nir-self.red)*2.5
        den=(self.nir+6*self.red-7.5*self.blue+1)
        den[den==0]=0.001

        evi=num/den
        evi[evi>1]=1
        evi[evi<-1]=-1

        return evi
