"""Tests of ParameterGenerator() methods involving turbulence parameters."""
import psfws
import numpy as np


def test_joint_pdf():
    # iterate through some semi-random values of rho
    rho_goal = [0.2, 0.45, 0.62, 0.83]
    rho_result = []
    for rho_jv in rho_goal:
        p = psfws.ParameterGenerator(rho_jv=rho_jv, seed=23459)
        rho_result.append(np.corrcoef(p.data_gl['j_gl'],
                                      p.data_gl['speed'])[0][1])
    # check whether results are within some tolerance of desired values
    np.testing.assert_allclose(rho_result, rho_goal, rtol=.1, atol=.05,
                               err_msg='correlating marginal pdfs failed.')


def test_turbulence_draws():
    # first test with no wind correlation for ground layer
    j_gl_goal = 3.88049713e-13
    j_fa_goal = 1.93653209e-13
    p = psfws.ParameterGenerator(rho_jv=0, seed=460)
    j_fa_res, j_gl_res = p._draw_j()

    np.testing.assert_allclose([j_fa_res, j_gl_res], [j_fa_goal, j_gl_goal],
                               rtol=1.e-8,
                               err_msg='error reproducing turbulence integrals')

    # second, test with wind correlation for ground layer
    src = 'ecmwf' if len(p.data_fa['u'].iat[0]) > 50 else 'noaa'
    if src == 'noaa':
        j_gl_goal = 1.44666406e-13
        j_fa_goal = 3.30348758e-14
    elif src == 'ecmwf':
        j_gl_goal = 2.23216667e-13
        j_fa_goal = 6.40808818-13
        
    p = psfws.ParameterGenerator(rho_jv=.7, seed=2012)
    pt = p.draw_datapoint()
    j_fa_res, j_gl_res = p._draw_j(pt)

    try:
        np.testing.assert_allclose([j_fa_res, j_gl_res], [j_fa_goal, j_gl_goal],
                                   rtol=1.e-8,
                                   err_msg='error reproducing turbulence integrals')
    except AssertionError:
        # might be because of a version difference, so test result from numpy < 2 
        j_gl_goal = 7.67667532e-14
        np.testing.assert_allclose([j_fa_res, j_gl_res], [j_fa_goal, j_gl_goal],
                                   rtol=1.e-8,
                                   err_msg='error reproducing turbulence integrals')       


def test_turbulence_integration():
    # test integration of fake Cn2 that I can integrate numerically
    x0, x1 = 3, 20
    m, b = 1e-19, 2e-17
    # x/h are in km, cn2 needs meters.
    def cn2(h):
        return (h * 1000) * m + b
    def cn2_int(x0, x1):
        return 0.5 * m * 1e6 * (x1**2 - x0**2) + b * 1000 * (x1 - x0)
    h_km = np.linspace(x0, x1, 500)
    cn2_m = cn2(h_km)

    j_analytic = cn2_int(x0, x1)
    j_utils = psfws.utils.integrate_in_bins(cn2_m, h_km, [h_km[0], h_km[-1]])
    np.testing.assert_allclose(j_analytic, j_utils, atol=1e-10, rtol=1e-3,
                               err_msg='error integration cn2')

    # test sum FA weights = j integral
    p = psfws.ParameterGenerator(seed=25493867)
    pt = p.draw_datapoint()
    j, layers, edges = p.get_turbulence_integral(pt, nl=8, location='mean')
    # need to reset rng to compare
    p = psfws.ParameterGenerator(seed=25493867)
    pt = p.draw_datapoint()
    j_fa, j_gl = p._draw_j(pt)
    np.testing.assert_allclose([j_fa, j_gl], [np.sum(j[1:]), j[0]], atol=1e-12,
                               err_msg='error with turbulence weighting')
    

def test_turbulence_altitudes():
    # first make sure number of weights == number of layers == len(edges)-1
    p = psfws.ParameterGenerator(seed=85647724)
    pt = p.draw_datapoint()
    nlayers = 12
    mj, mlayers, medges = p.get_turbulence_integral(pt, nl=nlayers, location='mean')
    np.testing.assert_equal([len(mj), len(mj)], [len(mlayers), len(medges) - 1],
                            err_msg='error in number of turbulence layer/edges')

    # test com vs mean -- edges should remain the same
    cj, clayers, cedges = p.get_turbulence_integral(pt, nl=nlayers, location='com')
    np.testing.assert_equal(medges, cedges, 
                            err_msg='error with center of mass parameters')
    
    # layer locations should change!
    cn2, h = p._get_fa_cn2(pt)
    l = 2
    # not exactly the same way of calculating than package, so tolerances high
    h_l = h[(h>medges[l]) & (h<medges[l+1])]
    cn2_l = cn2[(h > medges[l]) & (h < medges[l+1])]
    np.testing.assert_allclose([mlayers[l], clayers[l]],
                               [np.mean(h_l), np.sum(h_l * cn2_l) / np.sum(cn2_l)],
                               atol=0.1, rtol=5e-2,
                               err_msg='error with layer location calculation')

    # test altitudes of edges: lowest should be at ground
    assert medges[0] == p.h0, 'error with GL altitude'
    
    # 500m <= second edge <= 1km *above ground*
    np.testing.assert_array_less([p.h0 + 0.499, medges[1]],
                                 [medges[1], p.h0 + 1.01], 
                                 err_msg='error in altitude of layer edges')


if __name__ == '__main__':
    test_joint_pdf()
    test_turbulence_draws()
    test_turbulence_integration()
    test_turbulence_altitudes()
