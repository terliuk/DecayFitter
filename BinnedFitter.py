import iminuit
import numpy as np


class BinnedFitter:
    
    def __init__(self, loader ):
        self.loader = loader
        self.priors = {}
        self.fitparamlist = ["AXe136", 
                               ## Scales for material background
                             "Scale208Tl",
                             "Scale214Bi", 
                            ]
        self.default_fitvals = {"AXe136": [1.0, False],
                               ## Scales for material background
                                "Scale214Bi" : [ 9.278e-02, False], 
                                "Scale208Tl" : [ 2.339e+00, False], 
                                "flatRate" : [0.05, False]}
        self.default_limits = {"AXe136": (0.0, np.inf),
                               ## Scales for material background
                               "Scale208Tl" : (0.0, np.inf), 
                               "Scale214Bi" : (0.0, np.inf)
                         }
        self.default_priors = {}
        
    def FitValue(self,
                histogram, 
                fitvalues = {}, 
                errors = {}, 
                priors = {}, 
                bounds = {}, 
                ftol = 0.001, 
                minos = False, 
                verbosity = True):
        
        self.verbose = verbosity
        self.histogram = np.array(histogram)
        self.priors = self.default_priors
        self.priors.update(priors)
        self.fitvals = dict(self.default_fitvals)
        self.fitvals.update(fitvalues)
        self.limits = dict(self.default_limits)
        self.limits.update(bounds)
        errordict = {}
        for key in errors.keys():
            errordict["error_"+key] = errors[key]
        #### setting limits
        limitdict = {}
        for key in self.limits.keys():
            limitdict['limit_'+key] = self.limits[key]
        if self.verbose > 0: 
            print("-"*10 + " Boundaries " + "-"*10)
            for key in self.limits.keys():
                print(key.ljust(10) + "in  %s " %(str(self.limits[key])) )
            print("-"*30)
        ### 
        fixdict = {}
        startdict = {}
        for key in self.fitvals.keys():
            startdict[key] = self.fitvals[key][0]
            fixdict["fix_"+key] = self.fitvals[key][1]
        ## we are using Poisson LLH, so errordef must be 0.5
        
        if self.verbose > 0:
            print("-"*10 + " Prior information " + "-"*10)
            print("Variable".ljust(12) + "mean".rjust(12) + "sigma".rjust(12))
            for v in self.priors.keys():
                print(v.ljust(12) + 
                      ("%0.5f"%self.priors[v][0]).rjust(12) + 
                      ("%0.5f"%self.priors[v][1]).rjust(12) ) 
            print("-"*39)
        self.printvals = []
        if self.verbose > 1:
            print("-"*10 +" Call summary "+ "-"*10)
            pr_str = "LLH".ljust(10)
            for v in self.fitparamlist:
                if self.fitvals[v][1] == True : continue
                self.printvals.append(v)
                pr_str += " |" + ("%s"%v).rjust(10)
            print(pr_str)
                
        self.minimizer = iminuit.Minuit(self.LLH,
                                errordef = 0.5,
                                **startdict,
                                **fixdict,
                                **errordict, 
                                **limitdict)
        if ftol > 0.0: self.minimizer.tol = ftol
        self.min_result = self.minimizer.migrad()
        if minos: 
            print( "===== Minimization finshed, getting errors =======")
            self.min_errors = fitter.minimizer.minos()
        result = dict(self.minimizer.values)
        result['valid'] = self.min_result.fmin.is_valid
        result['LLH']  = self.minimizer.fval
        return result
    def SetVerbosity(self, verb):
        self.verbose = verb
        
    def LLH(self, AXe136  = 0.0, 
                  Scale208Tl = 0.0  ,                
                  Scale214Bi = 0.0, ):    
        expectation = self.loader.getBinnedExpectation(AXe136 = AXe136, 
                                                       Scale208Tl = Scale208Tl, 
                                                       Scale214Bi = Scale214Bi, 
                                                       Scale44Sc  = Scale208Tl / 100.0)
        #expectation[expectation < 0.0] = 0.0
        LLH = (- np.sum(  (self.histogram*np.log(expectation) - expectation))  )
        for key in self.priors.keys():
            LLH += 0.5*( getattr(self.loader, key) - self.priors[key][0])**2 / self.priors[key][1]
        pr_line = ""
        pr_line+= ("%0.5f"%LLH).ljust(10)
        if self.verbose > 1:
            for v in self.printvals:
                pr_line+=" |" + ("%0.5f"%getattr(self.loader, v) ).rjust(10)
            print(pr_line)
            if(np.isnan(LLH)): print(expectation)
        return LLH
