
#!/usr/bin/env python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
import numpy as np
import scipy as sp
import scipy.special

def voigt(xval,params):
    norm,center,gw,lw = params
    # norm : normalization 
    # center : center of Lorentzian line
    # lw : HWFM of Lorentzian 
    # gw : sigma of the gaussian 
    z = (xval - center + 1j*lw)/(gw * np.sqrt(2.0))
    w = scipy.special.wofz(z)
    model_y = norm * (w.real)/(gw * np.sqrt(2.0*np.pi))
    return model_y

# plot init function 
plt.title("Voigt function")
x = np.arange(0,10,0.1)

amp=50

# 同じ長さで、範囲[0, 1]のランダムな数値を生成
random_values = np.random.uniform(low=10, high=amp/3, size=len(x))

FWHM_G =  0.35
FWHM_L =  0.5

sigma_G = FWHM_G / (2 * np.sqrt(2 * np.log(2))) 
sigma_L = FWHM_L / 2

print(FWHM_G,FWHM_L)

#y0 = voigt(x,[amp,2,sigma_G,sigma_L])+voigt(x,[amp/2,3,sigma_G,sigma_L])+voigt(x,[amp*3,4,sigma_G,sigma_L])+random_values
y0 = voigt(x,[amp,2,sigma_G,sigma_L])
ye = np.sqrt(y0)
plt.plot(x, y0, 'k-', label ='FWHM_G = %.3f , FWHM_L = %.3f '  %(FWHM_G, FWHM_L))
"""
y1 = voigt(x,[1,np.mean(x),1,10])
plt.plot(x, y1/np.amax(y1), '-', label = "lw = 1, gw = 10")

y2 = voigt(x,[1,np.mean(x),10,1])
plt.plot(x, y2/np.amax(y2), '-', label = "lw = 10, gw = 1")

y3 = voigt(x,[1,np.mean(x),10,10])
plt.plot(x, y3/np.amax(y3), '-', label = "lw = 10, gw = 10")
"""
plt.legend(numpoints=1, frameon=False, loc="best")
plt.grid(linestyle='dotted',alpha=0.5)
plt.savefig("voigt.png")
plt.show()

# Save the data to CSV
data = np.column_stack((x, y0 ,ye))  # Combine x and y0 into a 2D array
np.savetxt("voigt_output.csv", data, delimiter=",", header="x,y,yerr", comments='')

