# ------------------------------------------------------------
# Mie scattering models
# Mie code based on Bohren and Huffman IDL implementation
# Implementation for NASA/PSG
# 2018-2021, Geronimo Villanueva, NASA-GSFC
# ------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt

def calcmie(file,update,source,type,subtype,label,rho,resize,reffs):
    r1=0.005  # Lower bound of size distribution
    r2=20.00  # Upper bound of size distribution
    rs=1.500  # Sigma width
    ndist=200 # Number of bins for the size distribution
    nang=20   # Number of angles for the phase function
    dtype=5   # Mie scattering model
    model = 'Mie implementation by Bohren and Huffman, 20 angles, 200 size bins (0.005 to 20 um) with sigma=1.5 and r_peak[um]=r_eff[um]/1.50833'

    # Verify if we have to perform the Mie calculations
    fbase = 'data/%s_%s_%s' % (source, type, subtype); fbase=fbase.lower(); print(fbase)
    if not os.path.exists('%s.txt' % fbase) or update:
        # Read optical constant data
        fr = open(file); ref='GSFC Scattering model'
        for k in range(10):
            line = fr.readline()
            if line[0:4]=='#REF': ref=line[5:-1]
        #End search for reference
        fdat = np.genfromtxt(file)
        ndat = len(fdat[:,0]);
        if resize>1:
            npts = int(ndat/resize)
            data = np.zeros([npts,3])
            for i in range(npts): data[i,:] = np.sum(fdat[i*resize:(i+1)*resize,0:3],axis=0)/resize
        else:
            npts = ndat
            data = fdat
        #End rebinning

        scatt = np.zeros([npts,len(reffs)*3+1])
        scatt[:,0] = data[:,0]
        for k in range(len(reffs)):
            # Calculate size distribution
            den = 1.0                # Cell number density [particles/cm3]
            rm  = reffs[k]/1.50833   # Effective particle radius [um]
            print(reffs[k])
            r1log=np.log(r1); r2log=np.log(r2); rslog=np.log(rs);
            const=np.sqrt(2.0*np.pi); constx=2.0*np.pi*1e-4; dlog=(r2log-r1log)/(1.00*ndist);
            radr=np.zeros(ndist); sized=np.zeros(ndist); dr=np.zeros(ndist)
            for i in range(ndist):
                rlog=r1log+(i*dlog)
                radr[i]=np.exp(rlog)
                alpha1=(np.log(radr[i]/rm))/rslog
                a2=(alpha1*alpha1)/2.0
                term1=den*(1.0/(radr[i]*rslog*const))*np.exp(-a2)
                sized[i]=term1
            #End loop over particles

            sum1=0; sum2=0;
            for i in range(ndist):
                if i<ndist-1: dr[i]=radr[i+1]-radr[i]
                else: dr[i]=radr[i]-radr[i-1]
                wt = np.pi*radr[i]*radr[i]*sized[i]*dr[i]
                sum1 = sum1 + wt
                sum2 = sum2 + radr[i]*wt
            #End loop over particles
            reff=sum2/sum1

            sum2=0
            for i in range(ndist):
                wt = np.pi*radr[i]*radr[i]*sized[i]*dr[i]
                sum2 = sum2 + ((radr[i]-reff)**2.0)*wt
            #End loop over particles
            veff=sum2/(sum1*reff*reff)

            # Scaling factor for extinction coefficient
            Area = np.pi*reff*reff*1e-12                    # Area [m2]
            Vol=(4.0/3.0)*np.pi*reff*reff*reff*1e-18        # Volume [m3]
            Mass=Vol*rho*1e3                                # Mass of particle [kg]
            tscl=(Area/Mass)                                # Qext to m2/kg scaler

            # Iterate across wavelength
            for i in range(npts):
                real=data[i,1]; im=data[i,2]
                if real<0: real=0
                if im<0: im=0
                refrel = np.complex(real,im)
                freq = 1e4/data[i,0]
                bext=0.0; babs=0.0; bsca=0.0; back=0.0; asym=0.0; wsum=0.0;
                for j in range(ndist):
                    x=constx*radr[j]*freq
                    if x>2950: continue
                    s1,s2,qext,qsca,qback,gfac = bhmie(x,refrel,nang)
                    #qext=1.0; qsca=1.0; qback=1.0; gfac=1.0
                    qabs=qext-qsca
                    rd2=radr[j]*radr[j]
                    weight=np.pi*rd2*sized[j]*dr[j]*1e-3
                    bext = bext + (weight*qext)      # bext [1/km]
                    babs = babs + (weight*qabs)      # babs [1/km]
                    bsca = bsca + (weight*qsca)      # bsca [1/km]
                    back = back + (weight*qback)     # back [1/km]
                    asym = asym + (weight*qsca*gfac) # asym [1/km]
                    wsum = wsum + weight
                # End size distribution loop

                scatt[i,k*3+1] = (bext/wsum)*tscl      # Extinction cross section [m2/Kg]
                scatt[i,k*3+2] = bsca / bext           # Dimensionless [0:no_scattering/only_extinction, 1:full_scattering]
                scatt[i,k*3+3] = asym / bsca           # Dimensionless [0:isotropic, 1:fully_directed]
            # End wavelength loop
        #End size loop

        # Save file
        fw = open('%s.txt' % fbase, 'w'); rstr=''
        for it in reffs: rstr = '%s %.2f' % (rstr, it)
        fw.write('#FILE:%s_%s_%s\n' % (source, type, subtype))
        fw.write('#REF:%s\n' % ref)
        fw.write('#MODEL:%s\n' % model)
        fw.write('#RADIUS:%s  ! Effective particle size radius [um]\n' % rstr)
        fw.write('#RHO:%.3f  ! Particle density [g/cm3]\n' % rho)
        fw.write('#TYPE:%d    ! 5=Scattering_gfunction, 4=Scattering_Legendre_polynomials, 0=Reflectance, 1=Optical-Constants, 2=Alpha-parameter [cm-1], 3=Cross_section[cm2/molecule]\n' % dtype)
        fw.write('#POINTS:%d  ! Number of wavelength points - Columns are: Wavelength[um] Qext[Extinction cross section, m2/kg] Omega [Single-scattering albedo, 0:no_scattering/only_extinction to 1:full_scattering] g [Asymmetry parameter, 0:isotropic to 1:fully_directed]\n' % len(scatt[:,0]))
        for j in range(len(scatt[:,0])):
            ts = "%12.6f" % scatt[j,0]
            for k in range(1,len(scatt[0,:])): ts = "%s %12.6e" % (ts,scatt[j,k])
            fw.write("%s\n" % ts)
        #End wavelength loop
        fw.close()
    else:
        scatt = np.genfromtxt('%s.txt' % fbase)
    #Endelse of generation

    # Plot the data
    pl,ax = plt.subplots(3,1, sharex=True, figsize=(6.4, 7.8))
    plt.subplots_adjust(left=0.08, right=0.99, bottom=0.1, top=0.93, hspace=0.01)
    ax[0].set_title('%s_%s_%s' % (source, type, subtype))
    plt.xscale('log'); plt.xlabel('Wavelength [um]'); ax[0].set_yscale('log');
    ax[0].set_ylabel('Ext. cross section [m2/kg]');
    ax[1].set_ylabel('Single-scattering albedo');
    ax[2].set_ylabel('Asymmetry factor [g]');
    for i in range(len(reffs)):
        ax[0].plot(scatt[:,0],scatt[:,1+i*3],label='%.2f um' % reffs[i])
        ax[1].plot(scatt[:,0],scatt[:,2+i*3],label='%.2f um' % reffs[i])
        ax[2].plot(scatt[:,0],scatt[:,3+i*3],label='%.2f um' % reffs[i])
    ax[0].legend(); ax[1].legend(); ax[0].legend(prop={'size':8}); ax[1].legend(prop={'size':8}); ax[2].legend(prop={'size':8})
    ax[0].autoscale(axis='x', tight=True); ax[0].set_ylim(bottom=1); ax[1].set_ylim([-0.05,1.05]); ax[2].set_ylim([-0.05,1.05])
    plt.tight_layout()
    plt.savefig("%s.png" % fbase)
    plt.close()

    # Add it to the list
    fl=open('../list_%s.txt' % source.lower(),'a+')
    if len(label)==0: label='r=%.2f-%.2fum' % (reffs[0],reffs[-1])
    fl.write('%s,%d_%s_%s[%s %.2f-%.2fum]\n' % (type, dtype, subtype, source, label, min(scatt[:,0]), max(scatt[:,0])))
    fl.close()
# End of calcmie() function


def bhmie(x,refrel,nang):
# This file is converted from mie.m, see http://atol.ucsd.edu/scatlib/index.htm
# Bohren and Huffman originally published the code in their book on light scattering

# Calculation based on Mie scattering theory
# input:
#      x      - size parameter = k*radius = 2pi/lambda * radius
#                   (lambda is the wavelength in the medium around the scatterers)
#      refrel - refraction index (n in complex form for example:  1.5+0.02*i;
#      nang   - number of angles for S1 and S2 function in range from 0 to pi/2
# output:
#        S1, S2 - funtion which correspond to the (complex) phase functions
#        Qext   - extinction efficiency
#        Qsca   - scattering efficiency
#        Qback  - backscatter efficiency
#        gsca   - asymmetry parameter

    nmxx=150000

    s1_1=np.zeros(nang,dtype=np.complex128)
    s1_2=np.zeros(nang,dtype=np.complex128)
    s2_1=np.zeros(nang,dtype=np.complex128)
    s2_2=np.zeros(nang,dtype=np.complex128)
    pi=np.zeros(nang)
    tau=np.zeros(nang)

    if (nang > 1000):
        print ('error: nang > mxnang=1000 in bhmie')
        return

    # Require NANG>1 in order to calculate scattering intensities
    if (nang < 2):
        nang = 2

    pii = 4.*np.arctan(1.)
    dx = x

    drefrl = refrel
    y = x*drefrl
    ymod = abs(y)


    #    Series expansion terminated after NSTOP terms
    #    Logarithmic derivatives calculated from NMX on down

    xstop = x + 4.*x**0.3333 + 2.0
    nmx = max(xstop,ymod) + 15.0
    nmx=np.fix(nmx)

    # BTD experiment 91/1/15: add one more term to series and compare resu<s
    #      NMX=AMAX1(XSTOP,YMOD)+16
    # test: compute 7001 wavelen>hs between .0001 and 1000 micron
    # for a=1.0micron SiC grain.  When NMX increased by 1, only a single
    # computed number changed (out of 4*7001) and it only changed by 1/8387
    # conclusion: we are indeed retaining enough terms in series!

    nstop = int(xstop)

    if (nmx > nmxx):
        print ( "error: nmx > nmxx=%f for |m|x=%f" % ( nmxx, ymod) )
        return

    dang = .5*pii/ (nang-1)


    amu=np.arange(0.0,nang,1)
    amu=np.cos(amu*dang)

    pi0=np.zeros(nang)
    pi1=np.ones(nang)

    # Logarithmic derivative D(J) calculated by downward recurrence
    # beginning with initial value (0.,0.) at J=NMX

    nn = int(nmx)-1
    d=np.zeros(nn+1,dtype=np.complex128)
    for n in range(0,nn):
        en = nmx - n
        d[nn-n-1] = (en/y) - (1./ (d[nn-n]+en/y))

    #*** Riccati-Bessel functions with real argument X
    #    calculated by upward recurrence

    psi0 = np.cos(dx)
    psi1 = np.sin(dx)
    chi0 = -np.sin(dx)
    chi1 = np.cos(dx)
    xi1 = psi1-chi1*1j
    qsca = 0.
    gsca = 0.
    p = -1

    for n in range(0,nstop):
        en = n+1.0
        fn = (2.*en+1.)/(en* (en+1.))

    # for given N, PSI  = psi_n        CHI  = chi_n
    #              PSI1 = psi_{n-1}    CHI1 = chi_{n-1}
    #              PSI0 = psi_{n-2}    CHI0 = chi_{n-2}
    # Calculate psi_n and chi_n
        psi = (2.*en-1.)*psi1/dx - psi0
        chi = (2.*en-1.)*chi1/dx - chi0
        xi = psi-chi*1j

    #*** Store previous values of AN and BN for use
    #    in computation of g=<cos(theta)>
        if (n > 0):
            an1 = an
            bn1 = bn

    #*** Compute AN and BN:
        an = (d[n]/drefrl+en/dx)*psi - psi1
        an = an/ ((d[n]/drefrl+en/dx)*xi-xi1)
        bn = (drefrl*d[n]+en/dx)*psi - psi1
        bn = bn/ ((drefrl*d[n]+en/dx)*xi-xi1)

    #*** Augment sums for Qsca and g=<cos(theta)>
        qsca += (2.*en+1.)* (abs(an)**2+abs(bn)**2)
        gsca += ((2.*en+1.)/ (en* (en+1.)))*( np.real(an)* np.real(bn)+np.imag(an)*np.imag(bn))

        if (n > 0):
            gsca += ((en-1.)* (en+1.)/en)*( np.real(an1)* np.real(an)+np.imag(an1)*np.imag(an)+np.real(bn1)* np.real(bn)+np.imag(bn1)*np.imag(bn))


    #*** Now calculate scattering intensity pattern
    #    First do angles from 0 to 90
        pi=0+pi1    # 0+pi1 because we want a hard copy of the values
        tau=en*amu*pi-(en+1.)*pi0
        s1_1 += fn* (an*pi+bn*tau)
        s2_1 += fn* (an*tau+bn*pi)

    #*** Now do angles greater than 90 using PI and TAU from
    #    angles less than 90.
    #    P=1 for N=1,3,...% P=-1 for N=2,4,...
    #   remember that we have to reverse the order of the elements
    #   of the second part of s1 and s2 after the calculation
        p = -p
        s1_2+= fn*p* (an*pi-bn*tau)
        s2_2+= fn*p* (bn*pi-an*tau)

        psi0 = psi1
        psi1 = psi
        chi0 = chi1
        chi1 = chi
        xi1 = psi1-chi1*1j

    #*** Compute pi_n for next value of n
    #    For each angle J, compute pi_n+1
    #    from PI = pi_n , PI0 = pi_n-1
        pi1 = ((2.*en+1.)*amu*pi- (en+1.)*pi0)/ en
        pi0 = 0+pi   # 0+pi because we want a hard copy of the values

    #*** Have summed sufficient terms.
    #    Now compute QSCA,QEXT,QBACK,and GSCA

    #   we have to reverse the order of the elements of the second part of s1 and s2
    s1=np.concatenate((s1_1,s1_2[-2::-1]))
    s2=np.concatenate((s2_1,s2_2[-2::-1]))
    gsca = 2.*gsca/qsca
    qsca = (2./ (dx*dx))*qsca
    qext = (4./ (dx*dx))* np.real(s1[0])

    # more common definition of the backscattering efficiency,
    # so that the backscattering cross section really
    # has dimension of length squared
    qback = 4*(abs(s1[2*nang-2])/dx)**2
    #qback = ((abs(s1[2*nang-2])/dx)**2 )/pii  #old form

    return s1,s2,qext,qsca,qback,gsca
#Enddef bhmie
