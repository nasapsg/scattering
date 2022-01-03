import os
import mie

source = 'GSFC'
update = 0

# List of models to run
reffs = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0] # Particle sizes [um]
files = ['../../surfaces/gsfc/data/gsfc_co2_ice.txt', '../../surfaces/gsfc/data/gsfc_nh3_ice.txt', '../../surfaces/gsfc/data/gsfc_ch4_ice.txt', '../../surfaces/gsfc/data/gsfc_hcn_crystalline.txt']
types = ['CO2Ice','Ammonia','Methane','Nitrile']
stypes = ['Hansen', 'Ice_Martonchik', 'Ice_Martonchik', 'Crystalline']
densities = [1.5, 0.86, 0.44, 0.69]
resizes = [1, 1, 1, 10]
labels = ['','','','']

# GSFC arbitrary clouds defined by optical constant
cloud= ['White','Simple']
opcs = [[0.0,1.0],[1.4,0.001]] # Real and imaginary coefficients
lmin = 0.1    # Initial wavelength [um]
lmax = 100e3  # Maximum wavelength [um]
dl   = 0.05   # Log spacing of points
for i in range(len(cloud)):
    file = 'data/gsfc_cloud_%s.opc' % cloud[i].lower()
    fw=open(file,'w'); lam=lmin
    fw.write('#REF: Simplified diffuse cloud with optical constants [%.4f+%.4fi]\n' % (opcs[i][0], opcs[i][1]))
    while lam<lmax:
        fw.write('%12.5f %12.4e %12.4e\n' % (lam, opcs[i][0], opcs[i][1]))
        lam = lam*(10.0**dl)
    #End loop file
    fw.close()
    files.append(file); types.append('Cloud'); stypes.append(cloud[i])
    densities.append(1.0); resizes.append(1)
    labels.append('n/k=%.3f+%.3fi ' % (opcs[i][0], opcs[i][1]))
# End subtypes

# Process all files
file = '../list_%s.txt' % source.lower()
if os.path.exists(file): os.remove(file)
for fi in range(len(files)):
    file = files[fi]
    type=types[fi]; subtype=stypes[fi]; rho=densities[fi]; resize=resizes[fi]; label=labels[fi];
    mie.calcmie(file,update,source,type,subtype,label,rho,resize,reffs)
#End iteration for files
