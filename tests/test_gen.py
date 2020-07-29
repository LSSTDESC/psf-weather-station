import numpy as np
import pathlib 
import pws

## TO DO ##
# need to test wind parameters and interpolation 

def test_init():
    try:
        pgen = pws.gen_params.ParameterGenerator(pkg_home='./')
        return True
    except:
        return False

def test_load():
    pgen = pws.gen_params.ParameterGenerator(pkg_home='./')
    try: 
        outs = pgen._load_data()
        return len(outs)==3
    except:
        return False

def test_gfs_processing():
    pgen = pws.gen_params.ParameterGenerator(pkg_home='./')
    gfs_winds, *rest = pgen._load_data()
    t_speed = (gfs_winds['speed'].iloc[0] == np.hypot(gfs_winds['u'].iloc[0], 
                                                      gfs_winds['v'].iloc[0])).all()
    t_order = gfs_winds['speed'].iloc[0][10] > gfs_winds['speed'].iloc[0][0]
    return t_speed and t_order

def test_telemetry_processing():
    pgen = pws.gen_params.ParameterGenerator(pkg_home='./')
    _, cp, mask = pgen._load_data()

    try:
        len(cp['dir']) > 100
        len(cp['speed']) > 100
    except:
        return False

    masked_speed = cp['speed'].loc[mask['speed']]['vals']
    t_mask_cut = (np.max(masked_speed) < 40) 
    t_mask_nonzero = (np.min(masked_speed) > 0)

    return t_mask_cut and t_mask_nonzero

def test_matching():
    pgen = pws.gen_params.ParameterGenerator(pkg_home='./')
    try: 
        pgen._match_data()
    except:
        return False

    l_gfs = len(pgen.gfs_winds)
    t_len = (len(pgen.telemetry['speed'])==l_gfs) and (len(pgen.telemetry['dir'])==l_gfs)

    t_nan = np.isfinite(pgen.telemetry['speed']).all() and np.isfinite(pgen.telemetry['dir']).all()
           
    return t_len and t_nan

# def test_interpolation()
#     pgen = pws.gen_params.ParameterGenerator()
#     pgen._match_data()


if __name__ == '__main__':
    
    assert test_init(), "initialization didn't work!"

    assert test_load(), "loading didn't work!"

    assert test_gfs_processing(), "gfs processing didn't work!"

    assert test_telemetry_processing(), "telemetry processing didn't work!"

    assert test_matching(), "matching didn't work!"
