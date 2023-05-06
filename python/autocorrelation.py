import numpy as np
import pandas
from matplotlib import pyplot as plt
import math
import os
import re
import glob
import sys

# Use formatting for grant application?
grant_application=False

class AutoCorrelation(object):
    def __init__(self,data):
        '''Base class for analysing the Autocorrelations of a time series
                
        :arg data: time-series data
        '''
        self._data = data
        self._data_c = None

    def _c(self):
        '''Compute normalised autocorrelation function in temporal domain

        Let the (empirical) covariance function be

        \gamma(k) = 1/(N-k)\sum_{j=0}^{N-k-1} (X(j) - \mu)*(X(j+k) - \mu)

        where \mu = 1/N\sum_{j=0}^{N-1} X(j) is the mean.
        
        Then this method returns [c(0),c(1),...,c(N-1)] with 
        c(k) = \gamma(k)/\gamma(0)
        '''
        if (self._data_c) is None:
            avg = np.average(self._data)
            n_array = self._data.size-np.arange(self._data.size)
            tmp = np.correlate(self._data,self._data,mode='full')
            result = tmp[tmp.size//2:]*1./n_array-avg**2
            self._data_c = result/result[0]
        return self._data_c

    def tau_int_naive(self,W):
        '''
        Compute naive integrated autocorrelation time in temporal domain
        
        This uses the 'naive' definition of the integrated autocorrelation
        time, summing up to an upper value of M.

        \tau_{int} = 1 + 2*\sum_{k=1}^{M} (1 - k/N)*c(k)

        where c(k) is the function returned by the method _c
        
        :arg M: Size of window
        
        '''
        data_c = self._c()
        n = data_c.size
        tmp = 0.0
        for k in range(1,W):
            tmp += (1.-k/n)*data_c[k]
        return 1.+2.*tmp

    def tau_int(self):
        '''Compute integrated autocorrelation time (abstract method)'''
        raise NotImplementedError()
    
    def plot(self,filename):
        '''Plot autocorrelation function and save in specified file

        :arg filename: Name of file to write to
        '''
        data_c = self._c()
        tauint, _ = self.tau_int()
        M = math.ceil(5*tauint)
        plt.clf()
        ax = plt.gca()
        ax.set_xlabel('Step $k$')
        ax.set_ylabel('$c(k)$')
        plt.plot(data_c[:M],
                 linewidth=2,
                 color='blue',
                 marker='o',
                 markersize=4,
                 markeredgewidth=2,
                 markerfacecolor='white',
                 label='data')
        X = np.arange(0,M,1.E-2)
        plt.plot(X,np.exp(-X/(0.5*(tauint+1))),
                 linewidth=2,
                 color='red',
                 label=r'$exp^{-k/\tau_{\mathrm{exp}}}$ with $\tau_{\mathrm{int}}='+('%10.3f' % tauint)+'$')
        plt.plot([0,M],[0,0],linewidth=2,color='black',linestyle='--')
        plt.legend(loc='upper right',fontsize=12)
        plt.savefig(filename,bbox_inches='tight')
        
class AutoCorrelationHeidelberger(AutoCorrelation):

    def __init__(self,data):
        '''Class for analysing the Autocorrelations of a time series using a
        the spectral method in
        
        Heidelberger, P. and Welch, P.D., 1981. "A spectral method for
        confidence interval generation and run length control in simulations."
        Communications of the ACM, 24(4), pp.233-245.
        
        :arg data: time-series data
        '''
        super().__init__(data)
        
    def _C1(self,K,degree):
        ''' The function C_1(K,d) defined in the appendix of
        Heidelberger & Welch

        :arg K: Number of points in spectral space to use
        :arg degree: Polynomial degree for fit
        '''
        N = len(self._data)
        f_n = (4.*np.arange(1,K+1)-1.)/(2.*N)
        X = np.zeros((K,degree+1))
        for k in range(K):
            for j in range(degree+1):
                X[k,j] = f_n[k]**j
        sigma2 = 0.645*(np.linalg.inv(np.transpose(X)@X)[0,0])
        return np.exp(-0.5*sigma2)

    def tau_int(self):
        '''Computes the integrated autocorrelation time'''
        # Number of spectral points
        K=50
        # Polynomial degree of fit
        degree=2
        N = len(self._data)
        I_freq = np.abs(np.fft.fft(self._data)[:N//2])**2/N
        J = np.zeros(N//4)
        for n in range(N//4):
            J[n] = np.log(0.5*(I_freq[2*n-1]+I_freq[2*n]))
        X_fit = (4.*np.arange(1,K+1)-1.)/(2.*N)
        Y_fit = J[1:K+1]+0.270

        a_fit=np.polyfit(X_fit,Y_fit,deg=degree)
        var = np.var(self._data)
        p0 = self._C1(K,degree)*np.exp(a_fit[-1])
        return p0/var, 0.0

class AutoCorrelationWolff(AutoCorrelation):

    def __init__(self,data,verbose=False):
        '''Class for analysing the Autocorrelations of a time series using a
        the automated windowing method in
        
        Wolff, U. and Alpha Collaboration, 2004. "Monte Carlo errors with less
        errors." Computer Physics Communications, 156(2), pp.143-153.

        :arg data: time-series data
        :arg verbose: print out additional information
        '''
        super().__init__(data)
        self._verbose = verbose
        
    def tau_int(self):
        '''Computes the integrated autocorrelation time and error estimate'''
        n_data = len(self._data)
        for W in range(2,n_data):
            tau = self.tau_int_naive(W)
            error_stat = np.sqrt((W+0.5*(1.-tau))/(n_data))*tau
            error_bias = np.exp(-2*W/tau)*tau
            if (error_bias <= error_stat):
                if self._verbose:
                    print ('window size = ',W)
                return tau, error_stat
        print ('ERROR: can not balance statistical error and bias')
        sys.exit(-1)

def read_data(filename):
    '''Read timeseries from file

    :arg filename: name of file to read from
    '''
    print ('Reading data from file '+filename)
    # Create a dtype with the binary data format and the desired column names
    dt = np.dtype([('qoi', 'f8')])
    data = np.fromfile(filename, dtype=dt)
    df = pandas.DataFrame.from_records(data)
    print ('...Done')
    data = np.array(df.T)[0]
    print ('Read '+str(len(data))+' entries')
    return data
        
def generate_exp_data(tau_int,n_data):
    '''Generate time series with a given integrated autocorrelation time

    :arg tau_int: Specified integrated autocorrelation time
    :arg n_data: length of time series
    '''
    alpha = (tau_int-1.0)/(tau_int+1.0)
    np.random.seed(5431227)
    X = 0.0
    for j in range(5*math.ceil(tau_int)):
        X = alpha*X + (1.-alpha)*np.random.normal()
    data = np.zeros(n_data)
    for j in range(n_data):
        X = alpha*X + (1.-alpha)*np.random.normal()
        data[j] = X
    return data

def analyse_testdata(tau_int):
    '''Analyse test data with specified autocorrelation time'''
    n_data = 100*math.ceil(tau_int)
    data = generate_exp_data(tau_int,n_data)
    plt.clf()
    autocorr_heidelberger=AutoCorrelationHeidelberger(data)
    tau_int_heidelberger, _ = autocorr_heidelberger.tau_int()
    autocorr_wolff=AutoCorrelationWolff(data,verbose=True)
    tau_int_wolff, dtau_int_wolff = autocorr_wolff.tau_int()
    tau_int = tau_int_wolff
    plt.plot(data[0:5*math.ceil(tau_int)],
             linewidth=2,
             marker='o',
             markersize=6,
             label=r'$\tau_{\mathrm{int}}^{(\mathrm{Heidelberger})} = '+('%10.2f' % tau_int_heidelberger)+r', \tau_{\mathrm{int}}^{(\mathrm{Wolff})} = '+('%10.2f' % tau_int_wolff)+r'\pm'+('%10.2f' % dtau_int_wolff)+'$')
    plt.legend(loc='upper right')
    plt.savefig('timeseries_test.pdf',bbox_inches='tight')

def analyse_file(filename,do_plot=False):
    '''Analyse data in given file'''
    data = read_data(filename)
    autocorr_wolff=AutoCorrelationWolff(data,verbose=True)
    tau_int, dtau_int = autocorr_wolff.tau_int()
    print ('tau_int = '+('%10.2f' % tau_int)+' +/- '+('%10.2f' % dtau_int))
    if (do_plot):
        plt.clf()
        plt.plot(data[0:5*math.ceil(tau_int)],
                linewidth=2,
                marker='o',
                markersize=6,
                label=r'$\tau_{\mathrm{int}} = '+('%10.2f' % tau_int)+r'\pm'+('%10.2f' % dtau_int)+'$')
        plt.legend(loc='upper right')
        plt.savefig('timeseries_test.pdf',bbox_inches='tight')

###################################################################
###                            M A I N                          ###
###################################################################
if (__name__ == '__main__'):
    if (len(sys.argv)==2):
        filename = sys.argv[1]
        analyse_file(filename,do_plot=False)
    else:
        print ('analysing test data')
        tau_int = 100
        analyse_testdata(tau_int)
