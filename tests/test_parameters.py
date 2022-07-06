"""Tests of ParameterGenerator() methods involving drawing parameters."""
import psfws
import numpy as np
import pandas as pd


def dict_test_helper(thing, p_list, x):
    """Test whether all keys in p_list exist in p_list and checks lengths."""
    # this loop starts with testing x for later comparison
    np.testing.assert_equal(x in thing.keys(), True,
                                err_msg=f'{x} not in dict!')
    for param in p_list:
        if param != x:
            # test that this parameter is in dict thing
            np.testing.assert_equal(param in thing.keys(), True,
                                    err_msg=f'{param} not in dict!')
            np.testing.assert_equal(len(thing[param]), len(thing[x]),
                                    err_msg="Parameter lengths don't match!")


def test_init():
    """Unit tests to verify data was loaded correctly."""
    p = psfws.ParameterGenerator()

    # do the GL and FA data have the same indices? 
    np.testing.assert_array_equal(p.data_gl.index, p.data_fa.index,
                                  err_msg='Error in dates of GL and FA!')

    # do the dataframes contain the correct column names? 
    np.testing.assert_equal(set(p.data_gl.columns),
                            set(['u', 'v', 't', 'speed', 'phi']),
                            err_msg='Error in GL data columns!')

    np.testing.assert_equal(set(p.data_fa.columns),
                            set(['u', 'v', 't', 'speed', 'phi']),
                            err_msg='Error in FA data columns!')

    # do the FA parameter and height arrays have the same length?
    np.testing.assert_equal(len(p.data_fa.iat[0,0]), len(p.h),
                            err_msg='Mismatched length of FA, height arrays!')

    # making sure h0 is not in h, and FA start is in h
    np.testing.assert_equal([p.h0 in p.h, p.h[p.fa_start] in p.h],
                            [False, True],
                            err_msg='Error in altitude definitions!')

    pt = np.random.choice(p.data_gl.index)
    # check u,v conversion to speed
    test_speed = np.hypot(p.data_fa.at[pt,'u'], p.data_fa.at[pt,'v'])
    np.testing.assert_allclose(test_speed, p.data_fa.at[pt,'speed'],
                               err_msg='Error in wind speed conversion!')

    # test utils.to_direction() function with some analytical test points
    # one point per quadrant of the UV plane
    u = [-10,20,15,-8]
    v = [-5,-7,10,2]
    theta_true = [26.56505, 160.70995, 213.69007, 345.96376]
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
    dict_test_helper(m_dict, m_names, 'h')

    # check error catching in get_raw_measurements()
    # for example, a string input format should raise TypeError
    np.testing.assert_raises(TypeError, p.get_measurements, '20190510')
    # and pd.Timestamp of date not in index should raise KeyError
    np.testing.assert_raises(KeyError, p.get_measurements,
                             pd.Timestamp('19000420'))


    # test get_parameters()
    p_dict = p.get_parameters(pt)
    p_names = ['h', 'u', 'v', 'speed', 't', 'phi', 'j']
    dict_test_helper(p_dict, p_names, 'h')

    
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


if __name__ == '__main__':
    test_init()
    test_params()
    test_interp()
