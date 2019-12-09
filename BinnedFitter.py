import iminuit
import numpy as np


class BinnedFitter:
    
    def __init__(self, loader ):
        self.loader = loader
        self.priors = {}
        self.fitparamlist = ["AXe136",  
                             "Scale208Tl",
                             "Norm214Bi",
                             "NormCont",
                             "ContSlope",
                             "Scale137Xe",
                             "T12_136Xe_2vbb",
                             "Scale8B",
                             "Scale222Rn"
                            ]
        self.default_fitvals = {"AXe136": [1.0, False],
                                "Scale208Tl" :[3.0, False],
                                "Norm214Bi" :[1.0, False],
                                "NormCont" : [1.0, False],
                                "ContSlope" : [-1.5, False],
                                "Scale137Xe" : [1.5e-3, False],
                                "T12_136Xe_2vbb" :[2.0, False],
                                "Scale8B" :  [2.36e-4, True] ,
                                "Scale222Rn" : [3.14e-4, True]
                               }
        self.default_errors = {"AXe136": 0.2,
                                "Scale208Tl" : 0.3,
                                "Norm214Bi":0.05,
                                "NormCont" : 0.05,
                                "ContSlope" : 1.0,
                                "Scale137Xe" : 3e-4,
                                "T12_136Xe_2vbb" : 0.05,
                                "Scale8B"  : 1.0e-5, 
                                "Scale222Rn" : 1.0e-5,
                                }
        self.default_limits = {"AXe136"     : (0.0, np.inf),
                               "Scale208Tl" : (0.0, np.inf),
                               "Norm214Bi"  : (0.0, 5.0),
                               "NormCont" : (0.0, 5.0),
                               "ContSlope" : (-100,100), 
                               "Scale137Xe" : (0.0, np.inf),
                               "T12_136Xe_2vbb" : (0.1,3), 
                               "Scale8B" : (0.0, np.inf),
                               "Scale222Rn" : (0.0, np.inf),
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
        self.smooth = True
        self.verbose = verbosity
        self.histogram = np.array(histogram)
        self.priors = self.default_priors
        self.priors.update(priors)
        self.fitvals = dict(self.default_fitvals)
        self.fitvals.update(fitvalues)
        self.limits = dict(self.default_limits)
        self.limits.update(bounds)
        self.errors = dict(self.default_errors)
        self.errors.update(errors)
        errordict = {}
        for key in self.errors.keys():
            errordict["error_"+key] = self.errors[key]
        #### setting limits
        limitdict = {}
        for key in self.limits.keys():
            limitdict['limit_'+key] = self.limits[key]
        if self.verbose > 0: 
            print("-"*10 + " Boundaries " + "-"*10)
            for key in self.limits.keys():
                print(key.ljust(15) + "in  %s " %(str(self.limits[key])) )
            print("-"*30)
        ### 
        fixdict = {}
        startdict = {}
        for key in self.fitvals.keys():
            startdict[key] = self.fitvals[key][0]
            fixdict["fix_"+key] = self.fitvals[key][1]
        if self.verbose > 0: 
            print("-"*10 + " Starting values: " + "-"*10)
            for key in self.fitvals.keys():
                print(key.ljust(15) + "= %0.5f , is fixed %r" %(startdict[key],fixdict["fix_"+key]) )
            print("-"*30)
        ## we are using Poisson LLH, so errordef must be 0.5
        
        if self.verbose > 0:
            print("-"*10 + " Prior information " + "-"*10)
            print("Variable".ljust(15) + "mean".rjust(12) + "sigma".rjust(12))
            for v in self.priors.keys():
                print(v.ljust(14) + 
                      ("%0.5f"%self.priors[v][0]).rjust(12) + 
                      ("%0.5f"%self.priors[v][1]).rjust(12) ) 
            print("-"*39)
        self.printvals = []
        if self.verbose > 1:
            print("-"*10 +" Call summary "+ "-"*10)
            pr_str = "LLH".ljust(12)
            for v in self.fitparamlist:
                if self.fitvals[v][1] == True : continue
                self.printvals.append(v)
                pr_str += " |" + ("%s"%v).rjust(10)
            print(pr_str)
        success = False
        n_failed = 0
        failed_LLHs = []
        while not success: 
            self.minimizer = iminuit.Minuit(self.LLH,
                                    errordef = 0.5,
                                    **startdict,
                                    **fixdict,
                                    **errordict, 
                                    **limitdict)
            self.minimizer.strategy = 1
            if ftol > 0.0: self.minimizer.tol = ftol
            self.min_result = self.minimizer.migrad()
            if not self.min_result.fmin.is_valid:
                self.minimizer.strategy = 2
                self.min_result = self.minimizer.migrad()
            if self.min_result.fmin.is_valid:
                success = True
            else: 
                self.smooth=False
                failed_LLHs.append(self.minimizer.fval)
                n_failed+=1
                if n_failed> 10:
                    print("WARNING! Minimizer failed %i attemts, continuing!"%n_failed)
                    print("list of failed LLHs: ",failed_LLHs)
                    success=True
                if not success:
                    print("WARNING! Minimizer failed for given tolerance.. trying to perturb the seed")
                    if self.verbose>0: 
                        print("Old starting values : \n", startdict)
                    for key in startdict.keys():
                        if not fixdict["fix_"+key]: 
                            startdict[key] = dict(self.minimizer.values)[key]*(0.5 + 1.0*np.random.uniform())
                        self.minimizer = iminuit.Minuit(self.LLH,
                                        errordef = 0.5,
                                        **startdict,
                                        **fixdict,
                                        **errordict, 
                                        **limitdict)
                    if self.verbose>0: print("new starting values: \n", startdict)
        if minos: 
            print( "===== Minimization finshed, getting errors =======")
            self.min_errors = fitter.minimizer.minos()
        if self.verbose > 0: 
            print( "===== Finished the minimization =======")

        result = dict(self.minimizer.values)
        result['valid'] = self.min_result.fmin.is_valid
        result['LLH']  = self.minimizer.fval
        if not self.smooth or  self.verbose > 0:
            if not self.smooth: print("There were warnings during the fit, here are the results")
            if self.verbose > 0:  print("Best fit results")
            toprint = result.keys()
            for key in toprint:
                print(key.ljust(20),  result[key],)
            print( "---"*20)

        result['fitted_histogram'] = self.getExpectation(**dict(self.minimizer.values))
        return result
    def SetVerbosity(self, verb):
        self.verbose = verb
    def getExpectation(self,AXe136, Scale208Tl,Norm214Bi,NormCont,ContSlope, Scale137Xe,T12_136Xe_2vbb, Scale8B, Scale222Rn): 
        expectation = self.loader.getBinnedExpectation(   AXe136 = AXe136, 
                                                      Scale208Tl = Scale208Tl,
                                                       Norm214Bi = Norm214Bi,
                                                        NormCont = NormCont, 
                                                       ContSlope = ContSlope,
                                                      Scale137Xe = Scale137Xe,
                                                  T12_136Xe_2vbb = T12_136Xe_2vbb,
                                                         Scale8B = Scale8B, 
                                                      Scale222Rn = Scale222Rn
                                                      )
        return(expectation)
    def LLH(self,  AXe136, Scale208Tl,Norm214Bi,NormCont,ContSlope, Scale137Xe,T12_136Xe_2vbb, Scale8B, Scale222Rn):    
        expectation = self.getExpectation(   AXe136 = AXe136, 
                                             Scale208Tl = Scale208Tl,
                                             Norm214Bi  = Norm214Bi,
                                             NormCont   = NormCont,
                                             ContSlope  = ContSlope,
                                             Scale137Xe = Scale137Xe,
                                         T12_136Xe_2vbb = T12_136Xe_2vbb,
                                             Scale8B    = Scale8B, 
                                             Scale222Rn = Scale222Rn)
        
        LLH = (- np.sum(  (self.histogram*np.log(expectation) - expectation))  )
        for key in self.priors.keys():
            LLH += 0.5* ( getattr(self.loader, key) - self.priors[key][0])**2 / (self.priors[key][1]**2)
        pr_line = ""
        pr_line+= ("%0.5f"%LLH).ljust(12)
        if self.verbose > 1:
            for v in self.printvals:
                pr_line+=" |" + ("%0.5f"%getattr(self.loader, v) ).rjust(10)
            print(pr_line)
        if(np.isnan(LLH)): 
            self.smooth=False
            print("---------  NAN enocountered --------- ")           
            print("Expectation",expectation)
        if np.sum(~(expectation > 0.0) ) >0 :
            self.smooth=False
            print("--------- invalid expectation ---------")
            print("Expectation", expectation)
            pr_str_names = "LLH".ljust(12)
            pr_str_values = ("%0.5f"%LLH).ljust(12)
            for v in self.fitparamlist:
                pr_str_names   +=" |" +("%s"%v).rjust(10)
                pr_str_values  +=" |" +("%0.5f"%getattr(self.loader, v) ).rjust(10)
            print(pr_str_names) 
            print(pr_str_values)
        return LLH
