# the following in cgs system !!
C_LIGHT          = 2.99792458E10
C_LIGHT_SI       = 2.99792458E8
AMU              = 1.6605E-24 
H_K              = 4.799243348e-11 ## 4.7995074E-11 
#BOLZMANN        = 1.3806E-16
BOLZMANN         = 1.3806488e-16
BOLTZMANN        = 1.3806488e-16
# BOLTZMANN_SI     = 1.3806E-23  
BOLTZMANN_SI     = 1.3806488e-23
STEFAN_BOLTZMANN = 5.670373e-5
SB_SI            = 5.670373e-8
CGS_TO_JY_SR     = 1e23          # erg/cm2/sr/Hz = CGS_TO_JY_SR * Jy/sr
# PLANCK         = 6.6262E-27
PLANCK           = 6.62606957e-27 
# PLANCK_SI      = 6.6262E-34
PLANCK_SI        = 6.62606957e-34
M0               = 1.99e33
MJupiter         = 1.9e30        # [g]
GRAV             = 6.67e-8
GRAV_SI          = 6.673e-11
PARSEC           = 3.0857E18
LIGHTYEAR        = 9.4607e17
LIGHTYEAR_SI     = 9.4607e15
ELECTRONVOLT     = 1.6022e-12
AU               = 149.597871e11
RSUN             = 6.955e10
RSUN_SI          = 6.955e8
DSUN             = 1.496e13  # cm
DSUN_SI          = 1.496e11  # 1.496e8 km
MSUN             = 1.9891e33
MSUN_SI          = 1.9891e30
M_EARTH          = 5.972e27
LSUN             = 3.839e33
LSUN_SI          = 3.839e26
TSUN             = 5778.0
MJUPITER         = 1.9e30
H_C2             = PLANCK/(C_LIGHT*C_LIGHT)
H_C2_GHz         = PLANCK/(C_LIGHT*C_LIGHT)*1.0e27

ARCSEC_TO_DEGREE =  (1.0/3600.0)
DEGREE_TO_RADIAN =  0.0174532925199432958
ARCMIN_TO_RADIAN =  (2.9088820e-4)
ARCSEC_TO_RADIAN =  (4.8481368e-6)
HOUR_TO_RADIAN   =  (0.261799387)
MINUTE_TO_RADIAN =  (4.3633231e-3)
SECOND_TO_RADIAN =  (7.2722052e-5)

RADIAN_TO_DEGREE =  57.2957795130823208768
RADIAN_TO_ARCMIN =  3437.746771
RADIAN_TO_ARCSEC =  206264.8063
RADIAN_TO_HOUR   =  3.819718634
RADIAN_TO_MINUTE =  229.1831181
RADIAN_TO_SECOND =  13750.98708

ARCMIN_TO_DEGREE =   (1.0/60.0)
DEGREE_TO_ARCMIN =   60.0
DEGREE_TO_ARCSEC =   3600.0
BSD              =   0.94e21    #  N(H2)=0.94e21*Av

kB = 1024.0
MB = 1024.0*kB
GB = 1024.0*MB

# Rayleigh = 1e6/(4*pi)  photons / cm2 / s / sr = 2.41e-7 erg/cm2/sr/s 
# H-alpha 6563 Angstrom => 4.5679e+14                                  
# 1 km/s = 1.5237e+9 Hz                                                
# R/(km/s) = 2.41e-7 / 1.52e9 = 1.58e-16  erg/cm2/s/sr/Hz              
RAYLEIGH_KMS_TO_JY_SR = 1.5807943647067913e-16

def planck_function(T, freq, derivatives=False):
    """
    Given a temperature T [K] and frequency freq [Hz], return the corresponding
    blackbody intensity in cgs units [erg/s/cm2/sr/Hz]. Optionally, also return derivatives
    wrt temperature. The precision set explicitly to float32.
    Uses _lots_ of memory (use pyx_planck_function from pyx.pyxPSM if memory is an issue)
    """
    # make T and freq into arrays of similar dimension
    if (0):
        T    = asarray(T, ndmin=1)
        freq = asarray(freq, ndmin=1)
    else:
        if (isscalar(T)):     T    = asarray([T,], float32)
        if (isscalar(freq)):  freq = asarray([freq,], float32)
    if    (len(T)>len(freq)):  freq = freq*ones(T.shape, float32) # freq must have been scalar !
    elif  (len(T)<len(freq)):  T    = T*ones(freq.shape, float32) # T must have been scalar !
    tmp  =  H_K*freq/T                        # argument of the exponential function
    S    =  zeros(len(tmp), float32)
    if (derivatives): dS_dT = S.copy()
    m    =  nonzero((tmp<100.0)&(tmp>1.0e-4)) # _outside_ Rayleigh-Jeans-regime (tmp>100 => result 0.0)
    if (len(m)>0):
        S[m] =  (2.0*H_C2_GHz*(freq[m]/1.0e9)**3.0/(exp(tmp[m])-1.0)).astype(float32)
        if (derivatives):
            dS_dT[m] =  (S[m] * (tmp[m]/T[m]) / (1.0-exp(-tmp[m]))).astype(float32)
    m    =  nonzero(tmp<=1.0e-4)              # _inside_ Rayleigh-Jeans-regime
    if (len(m)>0):
        S[m] = (2.0*freq[m]**2.0*BOLTZMANN*T[m]/(C_LIGHT**2.0)).astype(float32)
        if (derivatives):
            dS_dT[m] =  (S[m]/T[m]).astype(float32)
    if (derivatives):  return S, dS_dT
    return S
