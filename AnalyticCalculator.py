import numpy as np
from scipy import special, interpolate, integrate


class AnalyticCalculator:
    
    def __init__(self, bin_edges, 
                 Xe136eff = 0.90,  
                 livetime = 1.0,
                 fidMass  = 5.0):
        ###
        self.fidMass  = fidMass ### in tons
        self.livetime = livetime  ### in years
        self.a        = 0.3171 # resolution sigma  = (a /sqrt(E ) + b )* E, where E in keV
        self.b        = 0.0015
        ## for intrinsic 
        self.Scale8B    =  2.36e-4
        self.Scale222Rn =  3.14e-4
        self.SlopeIndex222Rn = -4.2e-7/3.14e-4
        self.Scale137Xe =  1.42e-3
        self.SlopeIndex137Xe = -1.28e-6/1.42e-3
        ###
        self.Scale208Tl = 2.511
        self.setPeakScales(self.Scale208Tl)
        self.setContScales()
        self.Norm214Bi = 1.0
        ####
        self.AXe136   = 0.0 # total rate / y / t
        self.Xe136eff = Xe136eff
        self.QXe136   = 2457.8 # keV
        self.ContSlope = -1.2
        self.NormCont  = 1.0
        # Parameters of the background
        self.Xe136_2vbb_T_1_2_central = 2.165 #  # 10^21 y
        self.T12_136Xe_2vbb = 2.165 # y
        self.E_208Tl_2614 = 2614.511
        self.E_44Sc       = 2656.444
        self.Peaks_214Bi = np.asarray([[2204.21, 2251.6,  2260.3,  2266.52, 2270.9, 
                                        2284.3,  2287.65, 2293.40, 2310.2,  2312.4, 
                                        2319.3 , 2325.0 , 2331.3,  2348.3,  2353.5, 
                                        2361.00, 2369.0,  2376.9,  2390.8,  2405.1, 
                                        2423.27, 2444.7,  2447.86, 2459.0,  2482.8, 
                                        2505.4,  2550.7,  2562.0,  2564.0,  2604.6, 
                                        2630.9,  2662.5,  2694.8,  2699.4,  2719.3,
                                        2769.9,  2785.9,  2827.0,  2861.1,  2880.3, 
                                        2893.6,  2921.9,  2928.6,  2934.6,  2978.9, 
                                        3000.0,  3053.88, 3081.7, 3093.98, 3142.58, 
                                        3149.0, 3160.6, 3183.57],
                                       [ 4.929,   0.0055,  0.0087,  0.0165,  0.0014,
                                        0.0050,   0.0046,   0.306,  0.0014,  0.0086, 
                                        0.0014,   0.0017,   0.026,  0.0014, 0.00036, 
                                        0.0021,   0.0028,  0.0086, 0.00156,  0.0011,
                                        0.0048,    0.008,    1.550, 0.00141, 0.00096, 
                                        0.0056,  0.00032,  0.00018, 0.00014, 0.00036, 
                                        0.00086, 0.000200, 0.033, 0.00282, 0.00170, 
                                        0.0225, 0.0055, 0.00218, 0.00041, 0.0101, 
                                        0.0057, 0.0134, 0.00109, 0.00046, 0.0137, 
                                        0.0089, 0.022, 0.0052, 0.00037, 0.00118, 
                                        0.00019, 0.00047, 0.0011]])


        # 
        self.setBins(bin_edges)
        #
    #def setResolution(self, a, b):
    #    self.a = a
    #    self.b = b 
    #    self.setBins(self.binEdges) # since sigma changes background - recalculating all the background components
    ###
    def setContScales(self):
        self.ContScale_208Tl =  self.Scale208Tl*1.8e-4*(self.fidMass**0.19)
        self.ContScale_214Bi =  self.ContScale_208Tl*(4.3e-4*self.fidMass + 3.2e-3 )
        self.ContScale_44Sc  =  self.ContScale_208Tl*(1.8e-4*self.fidMass + 6.6e-3 )
    def setPeakScales(self, p_scale):
        # p scale = scale of 208Tl peak
        self.Scale208Tl =  p_scale
        self.Scale214Bi = self.Scale208Tl/(-0.25 * self.fidMass + 29.8)
        self.Scale44Sc  = self.Scale208Tl/( 0.43 * self.fidMass + 91.6) 
        self.setContScales()
    ### Basic shape functions
    def sigma_E(self, E):
        return (E * (self.a / np.sqrt(E) + self.b))
    def continuum(self, E, E_peak, slope):
        return 1 / (1 + np.exp( (E-E_peak)/5.0) )  * ( np.exp( slope * ( (E - self.QXe136 ) / E_peak ) ) )  
    
    def gauss(self, en, E_center):
        return np.exp( - (en - E_center)**2 / 2.0 / self.sigma_E(E_center)**2 ) / (self.sigma_E(E_center)*np.sqrt(2 * np.pi) )
    def linear_func(self, en, Scale_at_Qbb, sl_lin):
        return  Scale_at_Qbb *(1. + (en - self.QXe136) * sl_lin) 
    #### Material backgrounds
    # thalium 
    def spectrum_208Tl_peaks_noscale(self, en):
        return self.gauss(en, self.E_208Tl_2614)
    def spectrum_208Tl_continuum_noscale(self, en):
        return self.continuum(en,self.E_208Tl_2614, self.ContSlope)
    
    # scandium     
    def spectrum_44Sc_peaks_noscale(self, en):
        return self.gauss(en, self.E_44Sc)   
    def spectrum_44Sc_continuum_noscale(self, en):
        return self.continuum(en, self.E_44Sc, self.ContSlope)    
    
    # bismuth     
    def spectrum_214Bi_peaks_noscale(self, en):
        retFloat=False
        if (type(en)==float):
            retFloat=True
            en = np.array([en],dtype=float)        
        en_cur = en.repeat(self.Peaks_214Bi.shape[1]).reshape(len(en),self.Peaks_214Bi.shape[1])     
        result = ( self.Peaks_214Bi[1, :] / self.Peaks_214Bi[1, 22] * self.gauss(en_cur, self.Peaks_214Bi[0, :])      
                 )
        if retFloat: return(float(np.sum(result, axis=1)))
        else: return(np.sum(result, axis=1))  
    def spectrum_214Bi_continuum_noscale(self, en):
        retFloat=False
        if (type(en)==float):
            retFloat=True
            en = np.array([en],dtype=float)
        en_cur = en.repeat(self.Peaks_214Bi.shape[1]).reshape(len(en),self.Peaks_214Bi.shape[1])     
        result = ( self.Peaks_214Bi[1, :] / self.Peaks_214Bi[1, 22] * self.continuum(en_cur, self.Peaks_214Bi[0, :], self.ContSlope)      
                 )
        if retFloat: return(float(np.sum(result, axis=1)))
        else : return(np.sum(result, axis=1))
    ### combining peaks and continuum with correspoding scales
    # useful for plots but not necessarily 
    def spectrum_208Tl(self, en):
        return self.Scale208Tl*self.spectrum_208Tl_peaks_noscale(en) + self.ContScale_208Tl*self.spectrum_208Tl_continuum_noscale(en)        
    def spectrum_214Bi(self, en):
        return self.Scale214Bi*self.spectrum_214Bi_peaks_noscale(en) + self.ContScale_214Bi*self.spectrum_214Bi_continuum_noscale(en)
    def spectrum_44Sc(self, en):
        return self.Scale44Sc*self.spectrum_44Sc_peaks_noscale(en) + self.ContScale_44Sc*self.spectrum_44Sc_continuum_noscale(en)    
    ### and intrinsic backgrounds 
    def spectrum_8B(self, E_array):
        return self.linear_func(E_array,self.Scale8B , 0)
    def spectrum_222Rn(self, E_array):
        return self.linear_func(E_array,self.Scale222Rn,self.SlopeIndex222Rn )
    def spectrum_137Xe(self, E_array):
        return self.linear_func(E_array,self.Scale137Xe,self.SlopeIndex137Xe )   
    #### The most complicated background is 2vbb, 
    # which is first calculated for idealistic spectrum 
    # and then smeared according to the resolution
    def spectrum_136Xe_2vbb_PR(self, E_array):
        #
        abundance = 0.08857 # 
        Na = 6.022e2 # 1/mol
        MXe = 131.293 * 1e-6 # t / mol
        year_ton_equivalent = np.log(2)/self.Xe136_2vbb_T_1_2_central * abundance * Na / MXe
        ####
        t = E_array / 510.998
        q = self.QXe136 / 510.998
        PR_norm = 5421004.17766348 # this is the normalization or total integral of the expression below
        result =  t*( (q - t)**5)*(1. + 2.*t + (4./3.)* t **2 + (1./3.)* t**3 + t**4 / 30. ) / PR_norm * year_ton_equivalent
        if type(result) in [ np.ndarray]:
            result[(q - t) < 0.0 ] = 0.0
        else: 
            result = result if q - t > 0.0 else 0.0
        return result

    def spectrum_136Xe_2vbb(self, E_array):
        retFloat=False
        if (type(E_array)==float):
            retFloat=True
            E_array = np.array([E_array],dtype=float)
        # here I apply smearing
        
        E_sm_low  = min(self.QXe136, 
                        max(0., E_array[0] - 8.*self.sigma_E(E_array[0])) )
        E_sm_high = min(self.QXe136,  
                        E_array[-1] + 8.*self.sigma_E(E_array[-1]) ) 
        # going +- 8 sigma(e) to be sure
        nbins = int(E_sm_high - E_sm_low)*10 # making at least 0.1 keV steps
        ### values for integration
        e_vals = np.linspace(E_sm_low,E_sm_high, nbins+1)
        e_vals_centers = 0.5*(e_vals[1:] + e_vals[0:-1])
        y_vals = self.spectrum_136Xe_2vbb_PR(e_vals)
        y_vals_centers = 0.5*(y_vals[1:] + y_vals[0:-1])
        ### smearing, creating 2D array to make calculations faster and deal with vectors
        E_array_2D = E_array.repeat(len(e_vals_centers))
        E_array_2D.resize(len(E_array) ,len(e_vals_centers) ) 
        y_vals_smeared_2D = (y_vals_centers*(e_vals[1:] - e_vals[0:-1]) * 
                            np.exp( - 0.5 *(E_array_2D - e_vals_centers)**2/(self.sigma_E(e_vals_centers)) **2 )*
                            (1./ (np.sqrt(2.*np.pi)*self.sigma_E(e_vals_centers) )) 
                            )
        if retFloat: return float(np.sum(y_vals_smeared_2D, axis=1) )
        else: return np.sum(y_vals_smeared_2D, axis=1)  
    ### and spectrum for signal itself
    def spectrum_136Xe_0vbb_noscale(self, E_array): 
        sigma = (self.a / np.sqrt(self.QXe136) + self.b)*self.QXe136
        return (self.Xe136eff /(np.sqrt(2*np.pi) *sigma) * np.exp( - 0.5*(E_array - self.QXe136)**2 / (sigma**2 ) )  )
    ### now the caching of different components 
    def get_8B(self):
        result = 0.5*(self.spectrum_8B(self.binEdges[1:]) + self.spectrum_8B(self.binEdges[0:-1]) )  
        result*=(self.binEdges[1:] - self.binEdges[0:-1])
        return result*self.fidMass * self.livetime
    def get_222Rn(self):
        result = 0.5*(self.spectrum_222Rn(self.binEdges[1:]) + self.spectrum_222Rn(self.binEdges[0:-1]) )  
        result*=(self.binEdges[1:] - self.binEdges[0:-1])
        return result*self.fidMass * self.livetime
    def get_137Xe(self):
        result = 0.5*(self.spectrum_137Xe(self.binEdges[1:]) + self.spectrum_137Xe(self.binEdges[0:-1]) )  
        result*=(self.binEdges[1:] - self.binEdges[0:-1])
        return result*self.fidMass * self.livetime
    def cache_histograms(self, 
                        sources = ["208Tl_peaks_noscale",
                                       "208Tl_continuum_noscale",
                                       "214Bi_peaks_noscale",
                                       "214Bi_continuum_noscale",
                                       "44Sc_peaks_noscale",
                                       "44Sc_continuum_noscale",
                                       "136Xe_2vbb", 
                                       "136Xe_0vbb_noscale"]
                        ):
        for bkg in sources:
            cur_spectrum = getattr(self, "spectrum_%s"%bkg)
            cur_histogram = np.zeros(len(self.binEdges)-1, dtype=float)
            for i in range(0, len(cur_histogram)):
                cur_histogram[i] = integrate.quad(cur_spectrum, self.binEdges[i], self.binEdges[i+1])[0]
            setattr(self,"histogram_%s"%bkg, np.array(cur_histogram))
        return
    def setBins(self, bin_edges):
        self.binEdges = bin_edges 
        self.cache_histograms()
    def getBinnedComponents(self, 
                            AXe136 , 
                            Scale208Tl,
                            Norm214Bi,
                            NormCont,
                            ContSlope,
                            Scale137Xe, 
                            T12_136Xe_2vbb
                            ): 
        yt = self.fidMass * self.livetime
        ###
        self.setPeakScales(Scale208Tl)
        self.AXe136 = AXe136
        self.NormCont = NormCont
        self.Norm214Bi = Norm214Bi
        self.Scale137Xe = Scale137Xe
        self.T12_136Xe_2vbb = T12_136Xe_2vbb
        if self.ContSlope != ContSlope: 
            self.ContSlope = ContSlope
            self.cache_histograms( ["208Tl_continuum_noscale",
                                    "214Bi_continuum_noscale",
                                    "44Sc_continuum_noscale"] )
        ###
        result = {}
        result['137Xe']   = self.get_137Xe()
        result['222Rn']   = self.get_222Rn()
        result['8B']      = self.get_8B()
        ###
        ### materials
        result['208Tl']  = self.Scale208Tl*self.histogram_208Tl_peaks_noscale*yt
        result['208Tl'] += self.NormCont*self.ContScale_208Tl*self.histogram_208Tl_continuum_noscale*yt
        ### materials
        result['214Bi']  = self.Norm214Bi*self.Scale214Bi*self.histogram_214Bi_peaks_noscale*yt
        result['214Bi'] += self.NormCont*self.ContScale_214Bi*self.histogram_214Bi_continuum_noscale*yt
        ### materials
        result['44Sc']  = self.Scale44Sc*self.histogram_44Sc_peaks_noscale*yt
        result['44Sc'] += self.NormCont*self.ContScale_44Sc*self.histogram_44Sc_continuum_noscale*yt
        ###
        result['136Xe_2vbb'] = self.histogram_136Xe_2vbb*yt*self.Xe136_2vbb_T_1_2_central/(self.T12_136Xe_2vbb)
        result['136Xe_0vbb'] = self.AXe136*yt*self.histogram_136Xe_0vbb_noscale
        return(result)
    def getBinnedExpectation(self, 
                            AXe136 , 
                            Scale208Tl,
                            Norm214Bi,
                            NormCont, 
                            ContSlope,
                            Scale137Xe,
                            T12_136Xe_2vbb
                            ):

        
        result  = np.zeros(len(self.binEdges)-1)
        components = self.getBinnedComponents(  
                                                AXe136         = AXe136, 
                                                Scale208Tl     = Scale208Tl, 
                                                Norm214Bi      = Norm214Bi, 
                                                NormCont       = NormCont,
                                                ContSlope      = ContSlope, 
                                                Scale137Xe     = Scale137Xe,
                                                T12_136Xe_2vbb = T12_136Xe_2vbb)
        
        for key in sorted(components.keys()):
            result+=components[key]
        return(result)
    '''
    ####
    def get_136Xe_2vbb(self):
        return self.hist_136Xe_2vbb_nonorm*self.fidMass * self.livetime*self.Xe136_2vbb_T_1_2_central/(self.T12_136Xe_2vbb)
    ####
    def cache_backgrounds(self):
        for bkg in ["208Tl", 
                    "214Bi", 
                    "44Sc", 
                    "136Xe_2vbb"]:
            cur_spline = getattr(self, "spectrum_%s_cumulative"%bkg) 
            cur_exp = cur_spline(self.binEdges)
            setattr(self, "hist_%s_nonorm"%bkg, 
                    cur_exp[1:] -  cur_exp[0:-1]
                   )                
    ####
    
    def make_cumulative_splines(self):
        ## making cumulative distribution to make our evalation more bin agnostic
        for bkg in ["208Tl", "214Bi", "44Sc", "136Xe_2vbb"]:
            energies_cumulative = np.linspace(self.binEdges[0], self.binEdges[-1] , int((self.binEdges[-1]-self.binEdges[0])*20)+1 ) # 0.1 keV
            if bkg == "136Xe_2vbb": 
                cur_spectrum = getattr(self, "spectrum_%s"%bkg)(energies_cumulative)
            else:
                cur_spectrum = getattr(self, "spectrum_%s_noscale"%bkg)(energies_cumulative)
            cumulative = np.append([0.0],
                                   np.cumsum(0.5*(cur_spectrum[1:]  + cur_spectrum[0:-1])*(
                                       energies_cumulative[1:] -energies_cumulative[0:-1]) )
                                  )
            setattr(self, "spectrum_%s_cumulative"%bkg, 
                   interpolate.interp1d(energies_cumulative, cumulative, kind="cubic"))
    ###
    

        self.make_cumulative_splines()
        self.cache_backgrounds()
        #self.cache_backgrounds()
    

        
    
    

    
    def getXe136_0vbb(self):
        sigma = (self.a / np.sqrt(self.QXe136) + self.b)*self.QXe136
        xi_vals = (self.binEdges - self.QXe136) /(np.sqrt(2) * sigma)
        # the cumulative expectation is 0.5 + 0.5*erf(xi)
        # but we are interested in event counts in bins, 
        # so bin content  = A*0.5*(erf(bin_edge_i+1) - erf(bin_edge_i))
        return self.AXe136*0.5*(special.erf(xi_vals[1:]) - special.erf(xi_vals[0:-1]) )*self.fidMass*self.livetime*self.Xe136eff

    '''