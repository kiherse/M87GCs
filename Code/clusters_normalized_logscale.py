import astropy.units as u
from astropy.io import fits,ascii
from astropy.table import Table
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate
from skimage.transform import rotate
from astropy.wcs import WCS
from matplotlib.patches import Circle
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmocean as cmo

import seaborn as sns
sns.set_palette('tab10',8)
sns.set_style('whitegrid')
[blue,orange,green,red,purple,brown,pink,gray] = palette = sns.color_palette('tab10',8)

# M87 Mireia Montes sample
m87clust = Table.read('tabla_radec.tex')
dist_nuc = np.sqrt(m87clust['xoff']**2+m87clust['yoff']**2) #arcsec

# M87 Bellini sample
bellini_clust = np.loadtxt('bellini.txt')
dist_nuc_bellini = np.sqrt((bellini_clust[:,0])**2+(bellini_clust[:,1])**2) #arcsec

# M87 image and parameters
image = 'N4486K.fit' #same size as original image but less hdu
hdu = fits.open(image)
info = hdu.info()
hdr = hdu[0].header
data = hdu[0].data

D = 16.1e3 #kpc
scale = 0.050 #arcsec/pixel
sensitivity = hdr['PHOTFLAM'] #inverse sensitivity, ergs/cm2/A/e-
zero_point = hdr['PHOTZPT'] #ST magnitude zero point

########################## CLASSIFICATION OF BELLINI GLOBULAR CLUSTERS

mask_red = bellini_clust[:,3]-bellini_clust[:,4]>0.85
dist_nuc_bellini_red = np.sqrt((bellini_clust[:,0]*mask_red)**2+(bellini_clust[:,1]*mask_red)**2) #arcsec
colour_red = []
for i in range(len(mask_red)):
	if (bellini_clust[:,3]*mask_red-bellini_clust[:,4]*mask_red)[i] > 0:
		colour_red.append((bellini_clust[:,3]*mask_red-bellini_clust[:,4]*mask_red)[i])
		
mask_blue = bellini_clust[:,3]-bellini_clust[:,4]<=0.85
dist_nuc_bellini_blue = np.sqrt((bellini_clust[:,0]*mask_blue)**2+(bellini_clust[:,1]*mask_blue)**2) #arcsec
colour_blue = []
for i in range(len(mask_blue)):
	if (bellini_clust[:,3]*mask_blue-bellini_clust[:,4]*mask_blue)[i] > 0:
		colour_blue.append((bellini_clust[:,3]*mask_blue-bellini_clust[:,4]*mask_blue)[i])
			
plt.figure()
plt.plot(figsize=(10,10))
plt.plot(bellini_clust[:,3]*mask_red-bellini_clust[:,4]*mask_red,bellini_clust[:,4]*mask_red,'.',color='red')
plt.plot(bellini_clust[:,3]*mask_blue-bellini_clust[:,4]*mask_blue,bellini_clust[:,4]*mask_blue,'.',color='blue')
plt.gca().invert_yaxis()
plt.ylim(25.5,18)
plt.xlim(0.3,1.5)
plt.xlabel(r'm$_{F606W}$-m$_{F8146W}$',fontsize=14)
plt.ylabel(R'm$_{F8146W}$',fontsize=14)
plt.savefig('CMD_Bellini_log.jpg',bbox_inches='tight')

# ~ #################### MIREIA AND BELLINI GLOBULAR CLUSTERS DISTRIBUTION

# We create the binning till 19 cause we got cluster information in a square of that area
lims = np.linspace(0, 19.2, 5)
bins,bins_b = np.zeros(len(lims)-1),np.zeros(len(lims)-1)

for i in range(len(bins)):
	bins[i] = np.sum((dist_nuc > lims[i]) & (dist_nuc<= lims[i+1]))

number_density, positions, area = np.zeros(len(bins)),np.zeros(len(bins)),np.zeros(len(bins))
for i in range(len(bins)):
	area[i] = np.pi*(lims[i+1]-lims[0])**2-np.pi*(lims[i]-lims[0])**2 #in arcsec^2
	number_density[i] = bins[i]/area[i] # num/arcsec^2
	positions[i] = (lims[i]+(lims[i+1]-lims[i])/2) # in arcsec	
uncertainties = np.sqrt(bins)/(area) # num/arcsec^2

for i in range(len(bins_b)):
	bins_b[i] = np.sum((dist_nuc_bellini > lims[i]) & (dist_nuc_bellini<= lims[i+1]))

number_density_b, positions_b, area_b = np.zeros(len(bins_b)),np.zeros(len(bins_b)),np.zeros(len(bins_b))
for i in range(len(bins_b)):
	area_b[i] = np.pi*(lims[i+1]-lims[0])**2-np.pi*(lims[i]-lims[0])**2 #in arcsec^2
	number_density_b[i] = bins_b[i]/area_b[i] # num/arcsec^2
uncertainties_b = np.sqrt(bins_b)/(area_b) # num/arcsec^2

d_0 = {'bin':lims[1:],'total':bins,'n':number_density,'s':uncertainties}
df_0 = pd.DataFrame(d_0)
df_0.to_csv('GCs_montes.csv',sep='	',index=False)

# COMPARISON BETWEEN THE HST, MIREIA AND BELLINI GCs DISTRIBUTIONS
profile = '../profile_wide/HST_profile_wide_clean.dat'
profile = ascii.read(profile)
profile.rename_columns(('col2','col3','col18','col19'),('SMA','INTENS','RSMA','mag'))

profile_mag = zero_point-2.5*np.log10(profile['INTENS']*sensitivity)
profile_mag_arcsec2 = zero_point-2.5*np.log10((profile['INTENS']*sensitivity)/(scale**2))

extrap = interpolate.interp1d(profile['SMA'][27:],profile_mag_arcsec2[27:],kind='slinear',bounds_error=False,fill_value='extrapolate')
R = np.linspace(0,450,450)
profile_HST = extrap(R)

def sersic(R,N_0,R_s,m):
	return N_0*np.exp(-(R/R_s)**(1/m))

profile_strader = Table.read('strader.tex')
R = np.linspace(0,450,450)*scale/60 #arcmin 
GC_strader = sersic(R,profile_strader['N_0'][0],profile_strader['R_s'][0],profile_strader['m'][0]) #arcmin^-2
GC_blue_strader = sersic(R,profile_strader['N_0'][1],profile_strader['R_s'][1],profile_strader['m'][1]) #arcmin^-2
GC_red_strader = sersic(R,profile_strader['N_0'][2],profile_strader['R_s'][2],profile_strader['m'][2]) #arcmin^-2

# Plot in arcmin^-2
fig,ax = plt.subplots()
ax.plot(figsize=(11,9))
ax.plot(R*60,np.log10(GC_strader/(60**2)),color='black',markersize=8,label='All GCs')
ax.plot(R*60,np.log10(GC_blue_strader/(60**2)),color=blue,markersize=8,label='Blue GCs')
ax.plot(R*60,np.log10(GC_red_strader/(60**2)),color=red,markersize=8,label='Red GCs')
ax.errorbar(positions,np.log10(number_density),yerr=(uncertainties)/(number_density),color=orange,fmt='.',markersize=10,label='Montes GCs number density')
ax.errorbar(positions,np.log10(number_density_b),yerr=(uncertainties_b)/(number_density_b),color='lightsalmon',fmt='.',markersize=10,label='Bellini GCs number density')
ax.set_xlabel('SMA [arcsec]',fontsize=14)
ax.set_ylabel(r'Log (Num clusters / arcsec$^2$)',fontsize=14)
ax.set_xscale('log')
ax.set_xlim(1,max(R*60))
ax.set_ylim(-2.1,1)
ax.legend(fontsize=12)
fig.savefig('GC_Montes_Bellini_log.jpg',bbox_inches='tight')
plt.show()
	
################################ BELLINI GLOBULAR CLUSTERS DISTRIBUTION: 5''

# We create the binning till 20 cause we got cluster information in a square of that area
lims_bellini = np.linspace(0, 160, 33) #equal areas: 33 for 5'', 17 for 10''
bins_bellini = np.zeros(len(lims_bellini)-1)
bins_bellini_blue = np.zeros(len(lims_bellini)-1)
bins_bellini_total = np.zeros(len(lims_bellini)-1)

for i in range(len(bins_bellini_total)):
	bins_bellini_total[i] = np.sum((dist_nuc_bellini > lims_bellini[i]) & (dist_nuc_bellini<= lims_bellini[i+1]))
number_density_bellini_total, positions_bellini_total, area_bellini_total = np.zeros(len(bins_bellini_total)),np.zeros(len(bins_bellini_total)),np.zeros(len(bins_bellini_total))
for i in range(len(bins_bellini_total)):
	area_bellini_total[i] = np.pi*(lims_bellini[i+1]-lims_bellini[0])**2-np.pi*(lims_bellini[i]-lims_bellini[0])**2 #in arcsec^2
	number_density_bellini_total[i] = bins_bellini_total[i]/area_bellini_total[i] # num/arcsec^2
	positions_bellini_total[i] = (lims_bellini[i]+(lims_bellini[i+1]-lims_bellini[i])/2) # in arcsec
uncertainties_bellini_total = np.sqrt(bins_bellini_total)/(area_bellini_total) # num/arcsec^2


for i in range(len(bins_bellini)):
	bins_bellini[i] = np.sum((dist_nuc_bellini_red > lims_bellini[i]) & (dist_nuc_bellini_red<= lims_bellini[i+1]))
number_density_bellini, positions_bellini, area_bellini = np.zeros(len(bins_bellini)),np.zeros(len(bins_bellini)),np.zeros(len(bins_bellini))
for i in range(len(bins_bellini)):
	area_bellini[i] = np.pi*(lims_bellini[i+1]-lims_bellini[0])**2-np.pi*(lims_bellini[i]-lims_bellini[0])**2 #in arcsec^2
	number_density_bellini[i] = bins_bellini[i]/area_bellini[i] # num/arcsec^2
	positions_bellini[i] = (lims_bellini[i]+(lims_bellini[i+1]-lims_bellini[i])/2) # in arcsec
uncertainties_bellini = np.sqrt(bins_bellini)/(area_bellini) # num/arcsec^2

for i in range(len(bins_bellini_blue)):
	bins_bellini_blue[i] = np.sum((dist_nuc_bellini_blue > lims_bellini[i]) & (dist_nuc_bellini_blue<= lims_bellini[i+1]))
number_density_bellini_blue, positions_bellini_blue, area_bellini_blue = np.zeros(len(bins_bellini_blue)),np.zeros(len(bins_bellini_blue)),np.zeros(len(bins_bellini_blue))
for i in range(len(bins_bellini_blue)):
	area_bellini_blue[i] = np.pi*(lims_bellini[i+1]-lims_bellini[0])**2-np.pi*(lims_bellini[i]-lims_bellini[0])**2 #in arcsec^2
	number_density_bellini_blue[i] = bins_bellini_blue[i]/area_bellini_blue[i] # num/arcsec^2
	positions_bellini_blue[i] = (lims_bellini[i]+(lims_bellini[i+1]-lims_bellini[i])/2) # in arcsec
uncertainties_bellini_blue = np.sqrt(bins_bellini_blue)/(area_bellini_blue) # num/arcsec^2

d = {'bin':lims_bellini[1:],'total':bins_bellini_total,'n':number_density_bellini_total,'s':uncertainties_bellini_total,'red':bins_bellini,'nred':number_density_bellini,'sred':uncertainties_bellini,'blue':bins_bellini_blue,'nblue':number_density_bellini_blue,'sblue':uncertainties_bellini_blue}
df = pd.DataFrame(d)
df.to_csv('GCs_bellini.csv',sep='	',index=False)

# COMPARISON BETWEEN THE HST AND THE BELLINI GCs DISTRIBUTIONS
profile = '../profile_wide/HST_profile_wide_clean.dat'
profile = ascii.read(profile)
profile.rename_columns(('col2','col3','col18','col19'),('SMA','INTENS','RSMA','mag'))

profile_mag = zero_point-2.5*np.log10(profile['INTENS']*sensitivity)
profile_mag_arcsec2 = zero_point-2.5*np.log10((profile['INTENS']*sensitivity)/(scale**2))

extrap = interpolate.interp1d(profile['SMA'][27:],profile_mag_arcsec2[27:],kind='slinear',bounds_error=False,fill_value='extrapolate')
R_0 = np.linspace(0,1600,1600)
profile_HST = extrap(R_0)

def sersic(R,N_0,R_s,m):
	return N_0*np.exp(-(R/R_s)**(1/m))

# Strader
profile_strader = Table.read('strader.tex')
R = np.linspace(1200,36000,36000)*scale/60 #arcmin 
GC_strader = sersic(R,profile_strader['N_0'][0],profile_strader['R_s'][0],profile_strader['m'][0]) #arcmin^-2
GC_blue_strader = sersic(R,profile_strader['N_0'][1],profile_strader['R_s'][1],profile_strader['m'][1]) #arcmin^-2
GC_red_strader = sersic(R,profile_strader['N_0'][2],profile_strader['R_s'][2],profile_strader['m'][2]) #arcmin^-2

fig,ax = plt.subplots()
ax.plot(figsize=(11,7))
ax.plot(R,np.log10(GC_strader),color='black',markersize=8,label='All GCs')
ax.plot(R,np.log10(GC_blue_strader),color=blue,markersize=8,label='Blue GCs')
ax.plot(R,np.log10(GC_red_strader),color=red,markersize=8,label='Red GCs')
ax.set_xlabel('SMA [arcmin]',fontsize=14)
ax.set_ylabel(r'Log (Num clusters / arcmin$^2$)',fontsize=14)
ax.set_xscale('log')
ax.legend(fontsize=12)
fig.savefig('GC_Strader_log.jpg',bbox_inches='tight')
plt.show()

# Strader vs Bellini: full range
profile_strader = Table.read('strader.tex')
R = np.linspace(1,36000,36000)*scale/60 #arcmin 
GC_strader = sersic(R,profile_strader['N_0'][0],profile_strader['R_s'][0],profile_strader['m'][0]) #arcmin^-2
GC_blue_strader = sersic(R,profile_strader['N_0'][1],profile_strader['R_s'][1],profile_strader['m'][1]) #arcmin^-2
GC_red_strader = sersic(R,profile_strader['N_0'][2],profile_strader['R_s'][2],profile_strader['m'][2]) #arcmin^-2

fig,ax = plt.subplots()
ax.plot(figsize=(11,9))
ax.plot(R*60,np.log10(GC_strader/(60**2)),color='black',markersize=8,label='All GCs')
ax.plot(R*60,np.log10(GC_blue_strader/(60**2)),color=blue,markersize=8,label='Blue GCs')
ax.plot(R*60,np.log10(GC_red_strader/(60**2)),color=red,markersize=8,label='Red GCs')
ax.errorbar(positions_bellini[:-16],np.log10(number_density_bellini_total[:-16]),yerr=(uncertainties_bellini_total[:-16])/(number_density_bellini_total[:-16]),color='black',fmt='.',markersize=12,label='Bellini total GCs number density')
ax.set_xlabel('SMA [arcsec]',fontsize=14)
ax.set_ylabel(r'Log (Num clusters / arcsec$^2$)',fontsize=14)
ax.set_xscale('log')
ax.set_xlim(1.0,max(R*60))
ax.legend(fontsize=12)
fig.savefig('GC_Strader_Bellini_30arcmin_log.jpg',bbox_inches='tight')
plt.show()

# Strader vs Bellini
profile_strader = Table.read('strader.tex')
R = np.linspace(0,1600,1600)*scale/60 #arcmin 
GC_strader = sersic(R,profile_strader['N_0'][0],profile_strader['R_s'][0],profile_strader['m'][0]) #arcmin^-2
GC_blue_strader = sersic(R,profile_strader['N_0'][1],profile_strader['R_s'][1],profile_strader['m'][1]) #arcmin^-2
GC_red_strader = sersic(R,profile_strader['N_0'][2],profile_strader['R_s'][2],profile_strader['m'][2]) #arcmin^-2

fig,ax = plt.subplots()
ax.plot(figsize=(11,9))
ax.plot(R*60,np.log10(GC_strader/(60**2)),color='black',markersize=8,label='All GCs')
ax.plot(R*60,np.log10(GC_blue_strader/(60**2)),color=blue,markersize=8,label='Blue GCs')
ax.plot(R*60,np.log10(GC_red_strader/(60**2)),color=red,markersize=8,label='Red GCs')
ax.errorbar(positions_bellini,np.log10(number_density_bellini_total),yerr=(uncertainties_bellini_total)/(number_density_bellini_total),color='black',fmt='.',markersize=12,label='Bellini total GCs number density')
ax.set_xlabel('SMA [arcsec]',fontsize=14)
ax.set_ylabel(r'Log (Num clusters / arcsec$^2$)',fontsize=14)
ax.set_xscale('log')
ax.set_xlim(1.0,max(R*60))
ax.set_ylim(-2.5,1)
ax.legend(fontsize=12)
fig.savefig('GC_Strader_Bellini_all_log.jpg',bbox_inches='tight')
plt.show()

# All Bellini
fig,ax = plt.subplots()
ax.plot(figsize=(11,9))
ax.errorbar(positions_bellini,np.log10(number_density_bellini_total),yerr=(uncertainties_bellini_total)/(number_density_bellini_total),color='black',fmt='.',markersize=12,label='Bellini total GCs number density')
ax.errorbar(positions_bellini,np.log10(number_density_bellini),yerr=(uncertainties_bellini)/(number_density_bellini),color='red',fmt='.',markersize=12,label='Bellini red GCs number density')
ax.errorbar(positions_bellini,np.log10(number_density_bellini_blue),yerr=(uncertainties_bellini_blue)/(number_density_bellini_blue),color='blue',fmt='.',markersize=12,label='Bellini blue GCs number density')
ax.set_xlabel('SMA [arcsec]',fontsize=14)
ax.set_ylabel(r'Log (Num clusters / arcsec^2$)',fontsize=14)
ax.set_xlim(1.0,max(R*60))
ax.set_ylim(-3.0,-0.5)
ax.legend(fontsize=12)
fig.savefig('GC_Bellini_all_log.jpg',bbox_inches='tight')
plt.show()

# Bellini vs M87
bellini_normalized  = (np.log10(number_density_bellini)-np.min(np.log10(number_density_bellini[:16]-uncertainties_bellini[:16])))/(np.max(np.log10(number_density_bellini[:16]+uncertainties_bellini[:16]))-np.min(np.log10(number_density_bellini[:16]-uncertainties_bellini[:16])))
uncertainties_bellini_normalized = ((np.log10(number_density_bellini+uncertainties_bellini)-np.min(np.log10(number_density_bellini[:16]-uncertainties_bellini[:16])))/(np.max(np.log10(number_density_bellini[:16]+uncertainties_bellini[:16]))-np.min(np.log10(number_density_bellini[:16]-uncertainties_bellini[:16])))) - bellini_normalized

M = np.log10(number_density_bellini_blue[1:16]+uncertainties_bellini_blue[1:16])
m = np.log10(number_density_bellini_blue[1:16]-uncertainties_bellini_blue[1:16])
bellini_normalized_blue  = (np.log10(number_density_bellini_blue[1:16])-np.min(m))/(np.max(M)-np.min(m))
uncertainties_bellini_blue_normalized = bellini_normalized_blue-(np.log10(number_density_bellini_blue[1:16]-uncertainties_bellini_blue[1:16])-np.min(m))/(np.max(M)-np.min(m))
HST_normalized=(profile_HST-np.min(profile_HST[50:]))/(np.max(profile_HST[50:]-np.min(profile_HST[50:])))

fig,ax = plt.subplots()
ax.plot(figsize=(11,9))
ax.errorbar(positions_bellini,bellini_normalized,yerr=uncertainties_bellini_normalized,color='red',fmt='.',markersize=12,label='Bellini red GCs number density')
ax.plot(R*60,1-HST_normalized,color=green,markersize=8,label='Surface brightness profile')
ax.set_xlabel('SMA [arcsec]',fontsize=14)
ax.set_ylabel(r'Log (Num clusters / arcsec$^2$)',fontsize=14)
ax.set_xscale('log')
ax.set_xlim(1.0,max(R*60))
ax.set_ylim(-0.2,1.3)
ax.legend(fontsize=12)
fig.savefig('GC_Bellini_red_normalized_log.jpg',bbox_inches='tight')
plt.show()

fig,ax = plt.subplots()
ax.plot(figsize=(11,9))
ax.errorbar(positions_bellini[1:16],bellini_normalized_blue,yerr=uncertainties_bellini_blue_normalized,color='blue',fmt='.',markersize=12,label='Bellini blue GCs number density')
ax.plot(R*60,1.1-HST_normalized,color=green,markersize=8,label='Surface brightness profile')
ax.set_xlabel('SMA [arcsec]',fontsize=14)
ax.set_ylabel(r'Log (Num clusters / arcsec$^2$)',fontsize=14)
ax.set_xscale('log')
ax.set_xlim(1.0,max(R*60))
ax.set_ylim(-0.2,1.3)
ax.legend(fontsize=12)
fig.savefig('GC_Bellini_blue_normalized_log.jpg',bbox_inches='tight')
plt.show()
