"""Tests of ParameterGenerator() methods involving drawing parameters."""
import psfws
import numpy as np
import pandas as pd


def dict_test_helper(test_dict, p_list):
    """Test that test_dict contains all p_list keys, check lengths all equal."""
    # select first key in p_list to use for length checking
    x = p_list[0]
    for param in p_list:
        # test that this parameter is in test_dict
        assert param in test_dict.keys(), f'{param} not in dict!'
        # test that parameter length matches that of x
        assert len(test_dict[param]) == len(test_dict[x]), \
        "Parameter lengths don't match!"


def test_init():
    """Unit tests to verify data was loaded correctly."""
    p = psfws.ParameterGenerator()

    # do the GL and FA data have the same indices? 
    np.testing.assert_array_equal(p.data_gl.index, p.data_fa.index,
                                  err_msg='Error in dates of GL and FA!')

    # do the dataframes contain the correct column names? 
    assert set(p.data_gl.columns) == set(['u', 'v', 't', 'speed', 'phi']), \
    'Error in GL data columns!'
    
    assert set(p.data_fa.columns) == set(['u', 'v', 't', 'speed', 'phi']), \
    'Error in FA data columns!'
    
    # do the FA parameter and height arrays have the same length?
    assert len(p.data_fa.iat[0,0]) == len(p.h), \
    'Mismatched length of FA, height arrays!'
    
    # making sure h0 is not in h, and FA start is in h
    np.testing.assert_equal([p.h0 in p.h, p.h[p.fa_start] in p.h],
                            [False, True],
                            err_msg='Error in altitude definitions!')

    pt = p.draw_datapoint()
    # check u,v conversion to speed
    test_speed = np.hypot(p.data_fa.at[pt,'u'], p.data_fa.at[pt,'v'])
    np.testing.assert_allclose(test_speed, p.data_fa.at[pt,'speed'],
                               err_msg='Error in wind speed conversion!')

    # test utils.to_direction() function with some analytical test points
    # one point per quadrant of the UV plane
    u = [-10,20,15,-8]
    v = [-5,-7,10,2]
    theta_true = [63.43494, 289.29004, 236.30993, 104.03624]
    theta_test = psfws.utils.to_direction(u, v)
    np.testing.assert_allclose(theta_test, theta_true, atol=.0001,
                               err_msg='Error in wind direction conversion!')


def test_params():
    """Unit tests to check the parameter outputs."""
    p = psfws.ParameterGenerator()
    pt = p.data_gl.index[0]
    
    # test get_raw_measurements()
    m_dict = p.get_measurements(pt)
    m_names = ['u', 'v', 'speed', 't', 'h', 'phi']
    dict_test_helper(m_dict, m_names)

    # check error catching in get_raw_measurements()
    # for example, a string input format should raise TypeError
    np.testing.assert_raises(TypeError, p.get_measurements, '20190510')
    # and pd.Timestamp of date not in index should raise KeyError
    np.testing.assert_raises(KeyError, p.get_measurements,
                             pd.Timestamp('19000420', tz='UTC'))

    # test get_parameters()
    p_dict = p.get_parameters(pt)
    p_names = ['h', 'u', 'v', 'speed', 't', 'phi', 'j']
    dict_test_helper(p_dict, p_names)

    
def test_interp():
    """Unit tests to test interpolation method."""
    p = psfws.ParameterGenerator()
    
    # test error catching for out of bounds values
    m_dict = p.get_measurements(p.data_gl.index[20])
    # if altitude array in units of m instead of km
    np.testing.assert_raises(ValueError, p._interpolate, m_dict, p.h * 1000)
    # if trying to interpolate to ground altitude
    np.testing.assert_raises(ValueError, p._interpolate, m_dict, [p.h0])

    # test interpolation accuracy
    m_dict = p.get_measurements(p.data_gl.index[40])
    # change one of the entries to be values of a cubic function
    def cubic(x):
        """Return cubic function (x in km)."""
        return 0.001 * x**3 + .5 * x**2 - 30
    def cubic_ddx(x):
        """Return derivative of cubic function (x in km)."""
        # divide by 1000 because d/dx is 1/m not 1/km
        return (0.003 * x**2 + x) / 1000
    m_dict['u'] = cubic(m_dict['h'])
    
    h_out = np.linspace(m_dict['h'][1], m_dict['h'][-1], 20)
    interp = p._interpolate(m_dict, h_out)
    
    np.testing.assert_allclose(interp['u'], cubic(h_out),
                               err_msg='Error in parameter interpolation!')

    np.testing.assert_allclose(interp['dudz'], cubic_ddx(h_out),
                               err_msg='Error in interpolation derivative!')


def test_coords():
    """Unit tests to check changing to GalSim coordinates."""
    # edge case: when at zenith, components for sky and earth are equal.
    params = {'u':[1], 'v':[1], 'h':[2.1], 'edges': [1,3], 'j':[1e-13]}
    p_sky = psfws.utils.convert_to_galsim(params, 90, 90)
    v_sky = np.array([p_sky['v'][0], p_sky['u'][0], 0.])
    np.testing.assert_allclose(v_sky, np.array([1.,1.,0.]))
    
    # check magnitude of vector (in 3d) is still correct after trivial rotation
    # (otherwise magnitude will change bc of vertical comp)
    params = {'u':[-10], 'v':[5], 'h':[2.1], 'edges': [1,3], 'j':[1e-13]}
    p_sky = psfws.utils.convert_to_galsim(params, 90, 90)
    v_sky = np.array([p_sky['v'][0], p_sky['u'][0], 0.])
    np.testing.assert_allclose(np.linalg.norm(v_sky), np.linalg.norm([-10, 5, 0.]))
    
    # check magnitude of vector in sky-NE plane is smaller after rotation
    params = {'u':[-10], 'v':[5], 'h':[2.1], 'edges': [1,3], 'j':[1e-13]}
    p_sky = psfws.utils.convert_to_galsim(params, 70, 70)
    v_sky = np.array([p_sky['v'][0], p_sky['u'][0], 0.])
    np.testing.assert_array_less(np.linalg.norm(v_sky), np.linalg.norm([-10, 5, 0.]))


def test_zenith():
    """Unit tests to check zenith dependence of line-of-sight and r0."""
    # altitude and seeing vs zenith?
    # check by varying zenith, d_los and j vary as expected, ie ~ sec(zenith)
    d_los, j = [], []
    for alt in [90,80,70,60]:
        params = {'u':[-10], 'v':[5], 'h':[2.1], 'edges': [1,3], 'j':[1e-13]}
        p_sky = psfws.utils.convert_to_galsim(params, alt, 60)
        d_los.append(p_sky['h'][0])
        j.append(p_sky['j'][0])

    # values of sec(zenith) for zenith = 90-alt
    sec_vals = np.array([1, 1.01542661, 1.06417777, 1.15470054])
    
    np.testing.assert_allclose(np.array(d_los), 2.1 * sec_vals)
    np.testing.assert_allclose(np.array(j), 1e-13 * sec_vals)


# if __name__ == '__main__':
#     test_init()
#     test_params()
#     test_interp()
#     test_coords()
#     test_zenith()
