import numpy as np
from scipy import special, interpolate


class AnalyticCalculator:
    
    def __init__(self, bin_edges, 
                 Xe136eff = 0.95,  
                 livetime = 1.0,
                 fidMass  = 12.0):
        self.flatRate = 2.4e-4+1.4e-3+3.0e-4 ## events / y / t / keV for 
        self.fidMass  = fidMass ### in tons
        self.livetime = livetime  ### in years
        self.a        = 0.3171 # resolution sigma  = (a /sqrt(E ) + b )* E, where E in keV
        self.b        = 0.0015
        self.ABi214   = 0.331 # total rate events / y / t
        self.AXe136   = 0.0 # total rate / y / t
        self.Xe136eff = Xe136eff
        self.QXe136   = 2457.8 # keV
        # 
        self.cont_frac =     5e-3
        self.slope     = - 2.4e-3
        
        # Parameters of the background
        self.E_208Tl_2614 = 2614.511
        self.E_44Sc       = 2656.444
        
        self.Peaks_214Bi  = np.asarray([   [2204.21, 2251.6,   2260.3,  2266.52, 2270.9, 
                                            2284.3,  2287.65,  2293.40,  2310.2, 2312.4, 
                                            2319.3 , 2325.0,   2331.3,   2348.3, 2353.5, 
                                            2361.00, 2369.0,   2376.9,   2390.8, 2405.1, 
                                            2423.27, 2444.7,   2447.86,  2459.0, 2482.8, 
                                            2505.4,  2550.7,   2562.0,   2564.0, 2604.6, 
                                            2630.9,  2662.5,   2694.8,   2699.4, 2719.3, 
                                            2769.9,  2785.9,   2827.0,   2861.1, 2880.3, 
                                            2893.6,  2921.9,   2928.6,   2934.6, 2978.9, 
                                            3000.0],
                                           [4.929,   0.0055,   0.0087,  0.0165,  0.0014, 
                                            0.0050,  0.0046,   0.306,   0.0014,  0.0086, 
                                            0.0014,  0.0017,   0.026,   0.0014,  0.00036, 
                                            0.0021,  0.0028,   0.0086,  0.00156, 0.0011, 
                                            0.0048,  0.008,    1.550,   0.00141, 0.00096, 
                                            0.0056,  0.00032,  0.00018, 0.00014, 0.00036, 
                                            0.00086, 0.000200, 0.033,   0.00282, 0.00170, 
                                            0.0225,  0.0055,   0.00218, 0.00041, 0.0101, 
                                            0.0057,  0.0134,   0.00109, 0.00046, 0.0137, 
                                            0.0089]])

        # 
        self.make_cumulative_splines()
        self.setBins(bin_edges)
        self.cache_backgrounds()
        #
        
    ### Basic shape functions
    def sigma_E(self, E):
        return (E * (self.a / np.sqrt(E) + self.b))
    ### Helper functions common to all backgeounds ###
    def continuum(self, E, E_peak):
        return 1. / (1. + np.exp((E - E_peak + 3. * self.sigma_E(E_peak))/2.)) * (np.exp(self.slope * (E - E_peak)))
    
    def gauss_no_norm(self, en, E_center):
        return np.exp( - (en - E_center)**2 / 2.0 / self.sigma_E(E_center)**2 ) 
    
    ### material backgrounds 
    def spectrum_208Tl_noscale(self, E_array):
        result = (self.gauss_no_norm(E_array,self.E_208Tl_2614) + self.cont_frac*self.continuum(E_array, self.E_208Tl_2614))
        return result
    
    def spectrum_44Sc_noscale(self, E_array):
        result = (self.gauss_no_norm(E_array, self.E_44Sc) + self.cont_frac * self.continuum(E_array, self.E_44Sc))
        return result
    
    def spectrum_214Bi_noscale(self, E_array):
        ## making 2D eneries, since i don't want to write for loop over peaks
        en_cur = E_array.repeat(self.Peaks_214Bi.shape[1]).reshape(len(E_array),self.Peaks_214Bi.shape[1])
        result = ( self.Peaks_214Bi[1, :] / self.Peaks_214Bi[1, 22]*
                          (self.gauss_no_norm(en_cur, self.Peaks_214Bi[0, :]) +  
                           self.cont_frac*self.continuum(en_cur, self.Peaks_214Bi[0, :]) 
                          )     
                 )
        return(np.sum(result, axis=1))
    ####
    ####
    def cache_backgrounds(self):
        for bkg in ["208Tl", 
                    "214Bi", 
                    "44Sc"]:
            cur_spline = getattr(self, "spectrum_%s_cumulative"%bkg) 
            cur_exp = cur_spline(self.binEdges)
            setattr(self, "hist_%s_nonorm"%bkg, 
                    cur_exp[1:] -  cur_exp[0:-1]
                   )                
    ####
    
    def make_cumulative_splines(self):
        ## making cumulative distribution to make our evalation more bin agnostic
        for bkg in ["208Tl", "214Bi", "44Sc"]:
            energies_cumulative = np.linspace(1800,3200,2801)
            cur_spectrum = getattr(self, "spectrum_%s_noscale"%bkg)(energies_cumulative)
            cumulative = np.append([0.0],
                                   np.cumsum(0.5*(cur_spectrum[1:]  + cur_spectrum[0:-1])*(
                                       energies_cumulative[1:] -energies_cumulative[0:-1]) )
                                  )
            setattr(self, "spectrum_%s_cumulative"%bkg, 
                   interpolate.interp1d(energies_cumulative, cumulative))
    ###
    def setBins(self, bin_edges):
        self.binEdges = bin_edges 
        self.cache_backgrounds()
    
    def getMaterialBkg(self, 
                            Scale214Bi,
                            Scale44Sc,
                            Scale208Tl):
        
        result  = np.zeros(len(self.binEdges)-1, dtype = float)
        result += Scale214Bi*self.hist_214Bi_nonorm
        result += Scale44Sc*self.hist_44Sc_nonorm
        result += Scale208Tl*self.hist_208Tl_nonorm
        return(result)
      
    def getIntrinsicBackground(self):
        result  = np.zeros(len(self.binEdges)-1, dtype = float)
        result += self.getFlatBackground()
        return(result)
    
    def getFlatBackground(self):
        return( self.flatRate*(self.binEdges[1:] - self.binEdges[0:-1])*self.fidMass * self.livetime )
    
    def getBinnedComponents(self, 
                            Scale214Bi, 
                            Scale44Sc, 
                            Scale208Tl, 
                            flatRate,
                            AXe136 ): 
        self.flatRate   = flatRate
        self.Scale214Bi = Scale214Bi
        self.Scale44Sc  = Scale44Sc
        self.Scale208Tl = Scale208Tl
        self.AXe136     = AXe136
        result = {}
        result['FlatBkg'] = self.getFlatBackground()
        result['214Bi']   = self.hist_214Bi_nonorm*self.Scale214Bi*self.fidMass * self.livetime
        result['44Sc']    = self.hist_44Sc_nonorm*self.Scale44Sc*self.fidMass * self.livetime
        result['208Tl']   = self.hist_208Tl_nonorm*self.Scale208Tl*self.fidMass * self.livetime
        result['Xe136_0vbb'] = self.getXe136_0vbb()
        return(result)
        
    def getBinnedExpectation(self, 
                            Scale214Bi, 
                            Scale44Sc, 
                            Scale208Tl, 
                            flatRate, 
                            AXe136):

        
        result  = np.zeros(len(self.binEdges)-1)
        components = self.getBinnedComponents(  Scale214Bi = Scale214Bi, 
                                                Scale44Sc  = Scale44Sc, 
                                                Scale208Tl = Scale208Tl, 
                                                flatRate   = flatRate,
                                                AXe136     = AXe136)
        for key in components.keys():
            result+=components[key]
        return(result)
    
    def spectrum_136Xe_0vbb(self, E_array): 
        sigma = (self.a / np.sqrt(self.QXe136) + self.b)*self.QXe136
        return (1.0 /(np.sqrt(2*np.pi) *sigma) * np.exp( - 0.5*(E_array - self.QXe136)**2 / (sigma**2 ) )  )
    
    def getXe136_0vbb(self):
        sigma = (self.a / np.sqrt(self.QXe136) + self.b)*self.QXe136
        xi_vals = (self.binEdges - self.QXe136) /(np.sqrt(2) * sigma)
        # the cumulative expectation is 0.5 + 0.5*erf(xi)
        # but we are interested in event counts in bins, 
        # so bin content  = A*0.5*(erf(bin_edge_i+1) - erf(bin_edge_i))
        return self.AXe136*0.5*(special.erf(xi_vals[1:]) - special.erf(xi_vals[0:-1]) )*self.fidMass*self.livetime*self.Xe136eff

