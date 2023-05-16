import numpy as np
import pyccl as ccl
import pytest
import time

@pytest.fixture(scope='module')
def set_up():
    t0 = time.time()
    
    # Benchmark data path
    bm_path = os.path.dirname(__file__) + 'data_NGcov/'
    
    # Setting general cosmology
    sigma8=0.8
    h=0.7
    Omega_m=0.3
    Omega_b=0.05
    Omega_c= Omega_m - Omega_b
    w0=-1.0
    wa=0.0
    n_s=0.96
    Neff=3.046
    m_nu=0
    tcmb0=2.725

    COSMO = ccl.Cosmology(sigma8=sigma8,
                          h=h,
                          Omega_c=Omega_c,
                          Omega_b=Omega_b,
                          w0=w0,
                          wa=wa,
                          n_s=n_s,
                          Neff=Neff,
                          m_nu=m_nu,
                          T_CMB=tcmb0,
                        transfer_function='boltzmann_class')

    lk_arr = np.logspace(-3,2, 100)
    z_arr = np.array([0])
    a_arr = 1/(1+z_arr)
    a_arr = np.flip(a_arr)

    mass_def = ccl.halos.MassDef200m()
    hmf = ccl.halos.MassFuncTinker08(COSMO, mass_def=mass_def)
    hbf = ccl.halos.HaloBiasTinker10(COSMO, mass_def=mass_def)
    hmc = ccl.halos.HMCalculator(COSMO, hmf, hbf, mass_def, log10M_min=9., log10M_max=17., nlog10M=900)
    prf = ccl.halos.HaloProfilePressureGNFW()
    prf.update_parameters(mass_bias=1./1.41, x_out=6.)
    
    # Getting tracers
    ell = np.logspace(1,3)

    z_lens  = np.loadtxt(f'{bm_path}/z_lens.txt' )
    nz_lens = np.loadtxt(f'{bm_path}/nz_lens.txt')

    bias = np.ones_like(z_lens)
    gcl = ccl.tracers.NumberCountsTracer(cosmo=COSMO, has_rsd=False, dndz=(z_lens, nz_lens), bias=(z_lens, bias))

    z_source  = np.loadtxt(f'{bm_path}/z_source.txt' )
    nz_source = np.loadtxt(f'{bm_path}/nz_source.txt')

    csh = ccl.tracers.WeakLensingTracer(cosmo=COSMO, dndz=(z_source, nz_source))
    
    # Getting benchmark data
    bms_tkk = {}
    bms_tkk['t1h']   = np.loadtxt(f'{bm_path}/NG_trispectrum_1h.txt')
    bms_tkk['t2h13'] = np.loadtxt(f'{bm_path}/NG_trispectrum_2h13.txt')
    bms_tkk['t2h22'] = np.loadtxt(f'{bm_path}/NG_trispectrum_2h22.txt')
    bms_tkk['t3h']   = np.loadtxt(f'{bm_path}/NG_trispectrum_3h.txt')
    bms_tkk['t4h']   = np.loadtxt(f'{bm_path}/NG_trispectrum_4h.txt')
    
    bms_tll = {}
    bms_tll['gcl'] = np.loadtxt(f'{bm_path}/NG_cov_gcl.txt')
    bms_tll['csh'] = np.loadtxt(f'{bm_path}/NG_cov_csh.txt')
    bms_tll['ggl'] = np.loadtxt(f'{bm_path}/NG_cov_ggl.txt')
    
    print('init and i/o time:', time.time() - t0)
    
    return COSMO, lk_arr, a_arr, hmf, hbf, hmc, prf, ell, gcl, csh, bms_tkk, bms_tll

# Testing 3D trispectra terms
@pytest.mark.parametrize('trispec', ['t1h',
                                     't2h13',
                                     't2h22',
                                     't3h',
                                     't4h'])
def test_tkk(set_up, trispec):
    COSMO, lk_arr, a_arr, hmf, hbf, hmc, prf, ell, gcl, csh, bms_tkk, bms_tll = set_up
    
    if trispec == 't1h':
        bm = bms_tkk[trispec]
        tkk = ccl.halos.halomod_trispectrum_1h(cosmo=COSMO, hmc=hmc, k=lk_arr, a=a_arr, prof1=prf, normprof1=True)
    if trispec == 't2h13':
        bm = bms_tkk[trispec]
        tkk = ccl.halos.halomod_trispectrum_2h_13(cosmo=COSMO, hmc=hmc, k=lk_arr, a=a_arr, prof1=prf, normprof1=True)
    if trispec == 't2h22':
        bm = bms_tkk[trispec]
        tkk = ccl.halos.halomod_trispectrum_2h_22(cosmo=COSMO, hmc=hmc, k=lk_arr, a=a_arr, prof1=prf, normprof1=True)
    if trispec == 't3h':
        bm = bms_tkk[trispec]
        tkk = ccl.halos.halomod_trispectrum_3h(cosmo=COSMO, hmc=hmc, k=lk_arr, a=a_arr, prof1=prf, normprof1=True)
    if trispec == 't4h':
        bm = bms_tkk[trispec]
        tkk = ccl.halos.halomod_trispectrum_4h(cosmo=COSMO, hmc=hmc, k=lk_arr, a=a_arr, prof1=prf, normprof1=True)

    assert tkk.flatten() = pytest.approx(bm.flatten(), rel=1e-5)
    
    
# Testing projected trispectrum
@pytest.mark.parametrize('trc', ['gcl', 
                                 'csh', 
                                 'ggl'])
def test_tll(set_up, trc):
    COSMO, lk_arr, a_arr, hmf, hbf, hmc, prf, ell, gcl, csh, bms_tkk, bms_tll = set_up

    tkk = ccl.halos.halomod_Tk3D_cNG(COSMO, hmc, prf,
                                    lk_arr=lk_arr, a_arr=a_arr,
                                    use_log=True)
    
    if trc == 'gcl':
        bm = bms_tll[trc]
        tll = ccl.angular_cl_cov_cNG(cosmo=COSMO, cltracer1=gcl, cltracer2=gcl, ell=ell, tkka=tkk, fsky=1.)
    if trc == 'csh':
        bm = bms_tll[trc]
        tll = ccl.angular_cl_cov_cNG(cosmo=COSMO, cltracer1=csh, cltracer2=csh, ell=ell, tkka=tkk, fsky=1.)
    if trc == 'ggl':
        bm = bms_tll[trc]
        tll = ccl.angular_cl_cov_cNG(cosmo=COSMO, cltracer1=gcl, cltracer2=csh, ell=ell, tkka=tkk, fsky=1.)

    assert tll.flatten() = pytest.approx(bm.flatten(), rel=1e-5)