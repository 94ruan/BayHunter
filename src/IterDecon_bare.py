import numpy as np
from scipy.fft import fft,ifft
import numba 
from numba import jit
import obspy as ob
import os

@jit(nopython=True) 
def iterdecon(traces, baz, nfft, gaussF, odina_flag,
              tshift=1, itmax=200, minderr=0.01, use_bic=True, 
              dt_bare=None):
    forward_list = np.zeros((len(baz), 2, nfft), dtype=float)
    gaussF_nor = fft(ifft(gaussF).real / np.max(np.abs(ifft(gaussF).real)))
    for tr in range(len(baz)):
        stream = traces[tr, :, :nfft]
        RF_output = np.empty((2, nfft))
        # SSE_output = []
        # Bic_output = []
        half = nfft // 2
        W0 = np.zeros(nfft)
        if odina_flag==2:
            W0[:nfft] = stream[0, :]
            _transverse = [1, 2]
        elif odina_flag==1:
            W0[:nfft] = stream[2, :]
            _transverse = [0, 1]
        
        W0f = fft(W0)
        W = ifft(fft(W0)*gaussF).real
        Wf = fft(W)
        powerW = np.sum(W**2)
        
        for index, chan in enumerate(_transverse):
            # Resize and rename the numerator and denominator
            U0 = np.zeros(nfft)
            U0[:nfft] =  stream[chan]
            U = ifft(fft(U0)*gaussF).real
            powerU = np.sum(U**2)
    
            RFS = []
            # Bic = []
            Bic = np.zeros(1)
            SSE = np.zeros(itmax)
            R = U
            P0 = np.zeros(nfft)
            sumsq_i = 1
            d_error = 100*powerU+minderr
            for it in range(itmax):
                if it==0 and abs(d_error)==minderr:
                    RFS.append(P0)
                    final_index = 0
                    break
                elif abs(d_error)>minderr:
                    a = ifft(fft(R)*np.conj(Wf)).real/powerW
                    index_k = np.argmax(np.abs(a[:half]))
                    amp = a[index_k]
                    # compute predicted deconvolution
                    P0[index_k] = P0[index_k]+amp
                    RFS.append(P0)
                    P = ifft(fft(P0)*gaussF*W0f).real
                    # compute residual with filtered numerator
                    R = U-P
                    powerR = np.sum(R**2)
                    sumsq = powerR/powerU
                    if len(Bic==1):
                        Bic[0] = np.log(nfft)*np.count_nonzero(P0)*2+nfft*np.log(float(powerR/nfft))
                    else:
                        Bic = np.append(Bic, np.log(nfft)*np.count_nonzero(P0)*2+nfft*np.log(float(powerR/nfft)))
                    # Bic.append(np.log(nt_bare)*np.count_nonzero(P0)+nt_bare*np.log(float(powerR/nt_bare)))
                    SSE[it] = sumsq # scaled error
                    d_error = 100*(sumsq_i-sumsq) #change in error
                    sumsq_i = sumsq
                else: break
            
            # Select final receiver function
            if use_bic:
                # if len(Bic)!=0: final_index = np.argmin(Bic)
                # rf = ifft(fft(RFS[final_index])*gaussF).real
                if len(Bic)!=0: final_index = np.argmin(Bic[::-1])
                # rf = ifft(fft(RFS[-final_index-1])*gaussF).real
                rf = ifft(fft(RFS[-final_index-1])*gaussF_nor).real
            else :
                # rf = ifft(fft(RFS[-1])*gaussF).real
                rf = ifft(fft(RFS[-1])*gaussF_nor).real
            # Phase shift
            # rf = phaseshift(rf, nfft, dt_bare, tshift)
            shift_i = round(tshift/dt_bare)
            p = np.multiply(2*np.pi*np.arange(nfft),np.divide(shift_i,nfft))
            rf = ifft(np.multiply(fft(rf),(np.cos(p)-(1j)*np.sin(p)))).real/np.cos(2*np.pi*shift_i/nfft)
            RF_output[index, :] = rf[:nfft]
            # SSE_output.append(SSE)
            # Bic_output.append(Bic)
        forward_list[tr, :, :] = RF_output
    return forward_list
    # return [RF_output, SSE_output, Bic_output]
    # nume = [(np.sum(weight*sum_RRF[tr]*forward_list[tr][0])+\
    #          np.sum((1-weight)*sum_TRF[tr]*forward_list[tr][1]))
    #         for tr in baz_list]
    # deno_1 = [np.sum(weight*(sum_RRF[tr])**2)+ \
    #           np.sum((1-weight)*(sum_TRF[tr])**2)
    #           for tr in baz_list]   
    # deno_2 = [np.sum(weight*(forward_list[tr][0])**2)+ \
    #           np.sum((1-weight)*(forward_list[tr][1])**2)
    #           for tr in baz_list]
    # misfit = 1 - sum(nume)/(np.sqrt(sum(deno_1)*sum(deno_2)))
    # return misfit

def gauss_filter(dt, f0, nfft):
    nfft_r = int(nfft/2)
    w = 2*np.pi*(np.arange(nfft_r)*(1.0/(nfft*dt)))
    gauss = np.zeros(nfft)
    gauss[:nfft_r] = np.exp(-w**2/(2*f0)**2)
    gauss[nfft_r:] = gauss[:nfft_r][::-1]
    return gauss

def phaseshift(x,nfft,dt,tshift):
    # Add a shift to the data
    Xf = fft(x)
    shift_i = round(tshift/dt)
    p = np.multiply(2*np.pi*np.arange(nfft),np.divide(shift_i,nfft))
    # apply shift
    Xf = np.multiply(Xf,(np.cos(p)-(1j)*np.sin(p)))
    x = ifft(Xf).real/np.cos(2*np.pi*shift_i/nfft)
    return x

def stack_rf(RRF, TRF):
    '''
    RRF = input sum Stream of rf radial component
    TRF = input sum Stream of rf transverse component
    '''

    length = len(RRF)
    rf_end = len(RRF[0].data)
    for i in range(length):
        if len(RRF[i].data)<rf_end:
            rf_end = len(RRF[i].data)
    
    sum_RRF = np.zeros([73,rf_end])
    sum_TRF = np.zeros([73,rf_end])
    count = np.zeros([73,1])
    
    for i in range(length):
        baz = RRF[i].stats.sac['baz']
        list = np.linspace(0,360,145) #An degree list with the interval of 2.5
        location = max(np.argwhere(list<baz))[0] #Decide which trace rf belong to
        if np.mod(location,2):
            trace = int((location+1)/2)  #Baz = trace*5 degree
        else:
            trace = int(location/2)
        
        sum_RRF[trace,:]+=RRF[i].data[:rf_end]
        sum_TRF[trace,:]+=TRF[i].data[:rf_end]
        count[trace]+=1
    count_nonzero = np.where(count!=0, count, 1)
            
    sum_RRF=sum_RRF/count_nonzero
    sum_TRF=sum_TRF/count_nonzero
    
    # stack 0 degree and 360 degree rf together
    sum_RRF[-1,:]+=sum_RRF[0,:]
    sum_RRF[0,:]  =sum_RRF[-1,:]
    sum_TRF[-1,:]+=sum_TRF[0,:]
    sum_TRF[0,:]  =sum_TRF[-1,:]
    
    return sum_RRF, sum_TRF, np.where(count==0, count, 1)

def read_observation(sta, folder_path, timespan=None):
    '''
    `sta`: station name
    `folder_path`: path of receiver functions folder
    `timespan`: time length (in Second)
    '''
    rfs = os.listdir(folder_path/sta)
    rfs_filter = [x for x in rfs if ('SAC' in x)&('Q' in x)]
    st_r = ob.Stream()
    st_t = ob.Stream()
    for x in rfs_filter:
        try:
            st_r += ob.read(folder_path/sta/x)
            st_t += ob.read(folder_path/sta/x.replace('Q', 'T'))
        except:
            print('Reading error')
            continue
    if timespan!=None:
        timepoint = int(timespan / st_r[0].stats.sac['delta'])
    else:
        timepoint = st_r[0].stats.npts
    obsx = (st_r[0].times()+st_r[0].stats.sac['b'])[:timepoint]
    sum_r, sum_t, traceflag = stack_rf(st_r, st_t)
    obsy = np.concatenate((sum_r[:, :timepoint], sum_t[:, :timepoint]), axis=1)
    return obsx, obsy