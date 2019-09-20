import numpy as np
from scipy import special


class AnalyticCalculator:
    
    def __init__(self, bin_edges, 
                 livetime = 1.0):
        self.flatRate = 2.4e-4+1.4e-3+3.0e-4 ## events / y / t / keV
        self.fidMass  = 12.0 ### in tons
        self.livetime = livetime  ### in years
        self.a        = 0.328 # resolutio sigma  = (a /sqrt(E ) + b )* E, where E in keV
        self.b        = 0.0008
        self.ABi214   = 0.331 # total rate events / y / t
        self.AXe136   = 0.0 # total rate / y / t
        #
        self.setBins(bin_edges)
        #
        
    def setBins(self, bin_edges):
        self.binEdges = bin_edges
    
    def getFlatBkg(self): 
        return self.flatRate*(self.binEdges[1:] - self.binEdges[0:-1])*self.fidMass*self.livetime
    
    def getBi214(self):
        mean = 2447.7  # 
        sigma = (self.a / np.sqrt(mean) + self.b)*mean
        xi_vals = (self.binEdges - mean) /(np.sqrt(2) * sigma)
        return self.ABi214*0.5*(special.erf(xi_vals[1:]) - special.erf(xi_vals[0:-1]) )*self.fidMass*self.livetime
    
    def getXe136(self):
        mean = 2457.8  # 
        sigma = (self.a / np.sqrt(mean) + self.b)*mean
        xi_vals = (self.binEdges - mean) /(np.sqrt(2) * sigma)
        # the cumulative expectation is 0.5 + 0.5*erf(xi)
        # but we are interested in event counts in bins, 
        # so bin content  = A*0.5*(erf(bin_edge_i+1) - erf(bin_edge_i))
        return self.AXe136*0.5*(special.erf(xi_vals[1:]) - special.erf(xi_vals[0:-1]) )*self.fidMass*self.livetime
    
    def getBinnedExpectation(self, 
                             AXe136 = 0.0, 
                             ABi214 = None, 
                             flatRate = None
                             ):
        if ABi214 != None: 
            self.ABi214 = ABi214
        if flatRate != None: 
            self.flatRate = flatRate
        self.AXe136 = AXe136
        ####
        expectation = np.zeros(len(self.binEdges) -1)
        expectation += self.getFlatBkg()
        expectation += self.getBi214()
        expectation += self.getXe136()
        ### All these expectations are made per year, now we need to multiply it with livetime
        self.last_expectation = np.array(expectation)
        return self.last_expectation
