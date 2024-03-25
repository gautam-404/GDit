"""Gravity Darkening (GD) module:
   Implements the Espinosa Lara & Rieutord GD model and applies it to an
   oblate spheroidal surface. Includes a process to compute the observed
   Teff and luminosity projected along the line of sight."""

__version__ = '0'
__author__ = 'Aaron Dotter'

# from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.integrate import simpson, trapz
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import RectBivariateSpline

#constants
G=6.67428e-8
sigma=5.67040e-5
f23=2.0/3.0
Lsun=3.8418e33
Msun=1.989e33
Rsun=6.96e10

#ELR11 equations
#gives the value of phi
def eq24(phi,theta,omega,rtw):
    tau = (pow(omega,2) * pow(rtw*np.cos(theta),3) )/3.0 + np.cos(theta) + np.log(np.tan(0.5*theta))
    return np.cos(phi) + np.log(np.tan(0.5*phi)) - tau

#solve for rtw given omega
def eq30(rtw,theta,omega):
    w2=omega*omega
    return (1./w2)*(1./rtw - 1.0) + 0.5*(pow(rtw*np.sin(theta),2) - 1.0)

#ratio of equatorial to polar Teff
def eq32(omega):
    w2=omega*omega
    return np.sqrt(2./(2.+w2))*pow(1.-w2, 1./12.)*np.exp(-(4./3.)*w2/pow(2+w2, 3))

def solve_ELR(omega,theta): #eq.26, 27, 28; solve the ELR11 equations
    """calculates r~, Teff_ratio, and Flux_ratio"""
    #theta is the polar angle.
    #this routine calculates values for 0 <= theta <= pi/2
    #everything else is mapped into this interval by symmetry
    # theta = 0 at the pole(s)
    # theta = pi/2 at the equator
    # -pi/2 < theta < 0: theta -> abs(theta)
    #  pi/2 > theta > pi: theta -> pi - theta
    if np.pi/2 < theta <= np.pi:
        theta = np.pi - theta
    if -np.pi/2 <= theta < 0:
        theta = abs(theta)

    if omega==0.0: #common sense
        return np.ones(3)
    
    else:
        #first we solve equation 30 for rtw
        q = root(fun=eq30,args=(theta, omega), x0=1.0)
        # rtw = asscalar(q['x'])
        rtw = q['x'].item()

        #the following are special solutions for extreme values of theta
        w2r3=pow(omega,2)*pow(rtw,3)
        
        if theta==0.0: #pole, eq. 27
            Fw = np.exp( f23 * w2r3 )

        elif theta==0.5*np.pi: #equator, eq. 28
            if omega < 1.0:
                Fw = pow(1.0 - w2r3, -f23)
            else:
                Fw = 0.0

        else: #general case for Fw
            q = root(fun=eq24,args=(theta, omega, rtw), x0=theta)
            # phi = asscalar(q['x'])
            phi = q['x'].item()
            
            Fw = pow(np.tan(phi)/np.tan(theta), 2)

        #equation 31 and similar for Fw
        term1 = pow(rtw,-4)
        term2 = pow(omega,4)*pow(rtw*np.sin(theta),2)
        term3 = -2*pow(omega*np.sin(theta),2)/rtw
        gterm = np.sqrt(term1+term2+term3)
        Flux_ratio = Fw*gterm
        Teff_ratio = pow(Flux_ratio,0.25)
        
        return rtw, Teff_ratio, Flux_ratio

#convenience functions
def Rp_div_Re(omega):
    rtw, Teff_ratio, Flux_ratio = solve_ELR(omega, theta=0)
    return rtw

def R_div_Re(omega):
    return np.cbrt(Rp_div_Re(omega))

def Re_div_R(omega):
    return 1.0/R_div_Re(omega)

def Rp_div_R(omega):
    return pow(Rp_div_Re(omega), f23)

def R_div_Rp(omega):
    return 1.0/Rp_div_R(omega)

#analytical formulas for the surface area and volume of oblate spheroid
def spheroid_surface_area(Rp,Re):
    #c=polar, a=equatorial; c/a=Rp/Re
    c_div_a=Rp / Re
    a=Re
    if c_div_a == 1.0 : #spherical
        extra = 1.0
    else:
        e=np.sqrt(1-pow(c_div_a,2))
        extra = (1-e*e)*np.arctanh(e)/e
    return 2*np.pi*a*a*(1+extra)

def spheroid_volume(Rp,Re):
    return 4*np.pi*Rp*Re*Re/3

#from Binggeli et al. (1980) the following calculates the ratio of semi-major and -minor axes
#of the ellipse resulting from the projection of an oblate spheroid along the line of sight
def beta(theta,q): #q is ratio Rp/Re; Binggeli et al. 1980
    j= pow(q*np.sin(theta),2) + pow(np.cos(theta),2)
    l=1.0
    top=j+l-np.sqrt( pow(j-l,2))
    bottom=j+l+np.sqrt( pow(j-l,2))
    return np.sqrt(top/bottom)

#the ares of the projected ellipse; inclication angle is defined differently in this case
def ellipse(Rp, Re, i): 
    b=beta(theta=np.pi/2-i,q=Rp/Re)
    return np.pi*Re*Re*b

def geometric_factors(omega, i, n_nu=50, n_phi=50, do_checks=True):
    """solves for geometric factors C_T and C_L for arbitrary omega, and inclination"""
    if omega==0: #perfect sphere
        return np.ones(2)
    
    #line of sight, inclination angle i
    # LOS = np.array([np.cos(i), 0, np.sin(i)])
    LOS = np.array([0, np.sin(i), np.cos(i)])

    # #spherical angles
    nu_array = np.linspace(-np.pi/2, np.pi/2, n_nu)
    phi_array = np.linspace(-np.pi, np.pi, n_phi)
    dnu = nu_array[1] - nu_array[0]
    dphi = phi_array[1] - phi_array[0]
    
    #find the polar radius
    Re=1.
    Rp=Rp_div_Re(omega)

    mu=np.arctanh(Rp/Re)

    a=np.sqrt(Re*Re - Rp*Rp)

    sinh_mu = np.sinh(mu)
    cosh_mu = np.cosh(mu)

    #now do area integrals
    nu_int1=np.empty(n_nu)
    nu_int2=np.empty(n_nu)
    nu_int3=np.empty(n_nu)
    nu_int4=np.empty(n_nu)

    #loop over polar angle nu
    for j,nu in enumerate(nu_array):

        theta = np.pi/2 - nu
        r,T,F = solve_ELR(omega=omega,theta=theta)
        
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)

        #scale factors
        K = np.sqrt(pow(sinh_mu,2) + pow(sin_nu,2))
        h_nu = a*K
        h_phi = a*cosh_mu*cos_nu
        
        phi_int1 = np.empty(n_phi)
        phi_int2 = np.empty(n_phi)

        #loop over azimuth angle phi
        for k,phi in enumerate(phi_array):
            e_mu_x = sinh_mu * cos_nu * np.cos(phi)
            e_mu_y = sinh_mu * cos_nu * np.sin(phi)
            e_mu_z = cosh_mu * sin_nu
            e_mu = np.array([e_mu_x, e_mu_y, e_mu_z]) / K

            dArea = h_nu * h_phi
            proj = max(0, np.dot(e_mu, LOS)) #only projected toward observer
            phi_int1[k] = dArea
            phi_int2[k] = dArea*proj

        #phi integral at constant nu
        nu_int1[j] = simpson(y=phi_int1, dx=dphi) #total surface area
        nu_int2[j] = simpson(y=phi_int2, dx=dphi) #projected area
        nu_int3[j] = F*nu_int2[j] #integral of Flux_ratio, projected
        nu_int4[j] = F*nu_int1[j] #integral of Flux_ratio, total

    #nu integrals
    surface_area = simpson(y=nu_int1, dx=dnu)
    projected_area = simpson(y=nu_int2, dx=dnu)
    projected_flux = simpson(y=nu_int3, dx=dnu)
    total_flux = simpson(y=nu_int4, dx=dnu)
    
    C_L = 4 * projected_flux / total_flux
    C_T = pow( C_L * surface_area / projected_area / 4, 0.25 )

    # C_L = 4 * simpson(y=nu_int3, dx=dnu) / total_flux
    # C_T = pow( C_L * surface_area / projected_area / 4, 0.25 )

    if do_checks:
        surface_area_ratio = surface_area / spheroid_surface_area(Rp, Re)
        surface_area_check = abs( surface_area_ratio - 1.0)
        projected_area_ratio = projected_area / ellipse(Rp, Re, i)
        projected_area_check = abs( projected_area_ratio - 1.0)

        if surface_area_check > 0.001 or projected_area_check > 0.001:
            print('Warning: surface area or projected area integration may be inaccurate')
            print(f'Try increasing n_nu or n_phi')
            print(f'Omega = {omega:.3f} and i = {np.degrees(i):.2f} degrees')
            print(f'Surface area ratio: {surface_area_ratio:.3f}') 
            print(f'Projected area ratio: {projected_area_ratio:.3f}')
        
    return C_T, C_L

#compute the coefficients C_T and C_L on an nxn matrix
def save_coefficients(n,output='GD.npz'):
    omega=np.linspace(0,1,n)
    inclination=np.linspace(0,np.pi/2,n)
    C_T=np.empty((n,n))
    C_L=np.empty((n,n))
    for k,w in enumerate(omega):
        for j,i in enumerate(inclination):
            C_T[j,k], C_L[j,k] = geometric_factors(w, i)
    np.savez(output, C_T=C_T, C_L=C_L, omega=omega, inclination=inclination)

def print_coefficients(n,output='data.txt'):
    omega=np.linspace(0,1,n)
    inclination=np.linspace(0,np.pi/2,n)
    with open(output,'w') as f:
        for i,w in enumerate(omega):
            for j,incl in enumerate(inclination):
                C_T, C_L = geometric_factors(w, incl)
                f.write('{0:.17f} {1:.17f} {2:.17f} {3:.17f}\n'.format(w,incl,C_T,C_L))

    
#returns instance of the BivariateRect
def create_interpolants(npz):
    data=np.load(npz)
    C_L=data['C_L']
    C_T=data['C_T']
    omega=data['omega']
    inclination=data['inclination']
    f_T=RectBivariateSpline(y=omega, x=inclination, z=C_T)
    f_L=RectBivariateSpline(y=omega, x=inclination, z=C_L)
    return f_T, f_L

#requires the npz datafile from create_interpolant
def plot_colormap(npz='GD.npz'):
    fs=18
    data=np.load(npz)
    C_T=data['C_T']
    C_L=data['C_L']
    C_g=pow(C_T,4)/C_L
    s=np.shape(C_T)
    plt.close(1)
    plt.figure(1,figsize=(9,8))
    plt.subplots_adjust(right=0.95)
    plt.imshow(C_L,cmap='Spectral',interpolation='bicubic',origin='lower',extent=(0,1,0,np.pi/2),aspect='auto')
    plt.yticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2], [r'$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'], fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.xlabel(r'$\omega$',fontsize=fs)
    plt.ylabel(r'Inclination',fontsize=fs)
    cbar=plt.colorbar()
    cbar.set_label(r'$\mathrm{C_L}$',size=18)
    
    plt.close(2)
    plt.figure(2,figsize=(9,8))
    plt.subplots_adjust(right=0.95)
    plt.imshow(C_T,cmap='Spectral',interpolation='bicubic',origin='lower',extent=(0,1,0,np.pi/2),aspect='auto')
    cbar=plt.colorbar()
    cbar.set_label(r'$\mathrm{C_T}$',size=18)
    plt.yticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2], [r'$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'], fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.xlabel(r'$\omega$',fontsize=fs)
    plt.ylabel(r'Inclination',fontsize=fs)

    plt.close(3)
    plt.figure(3,figsize=(9,8))
    plt.subplots_adjust(right=0.95)
    plt.imshow(C_g,cmap='Spectral',interpolation='bicubic',origin='lower',extent=(0,1,0,np.pi/2),aspect='auto')
    cbar=plt.colorbar()
    cbar.set_label(r'$\mathrm{C_g}$',size=18)
    plt.yticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2], [r'$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'], fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.xlabel(r'$\omega$',fontsize=fs)
    plt.ylabel(r'Inclination',fontsize=fs)

    
    plt.show()
