import numpy as np
import pathlib 
import pws

def test_loading():
    pgen = pws.gen_params.ParameterGenerator()

    try: 
        gfs_l = pgen._load_gfs()
        pgen.gfs_winds['speed']
    except:
        gfs_l = False

    try: 
        cp_l = pgen._load_telemetry()
        pgen.cp_telemetry['speed']
    except:
        cp_l = False

    return gfs_l and cp_l

def test_processing():
    pgen = pws.gen_params.ParameterGenerator()
    pgen._load_gfs()
    t_speed = (pgen.gfs_winds['speed'].iloc[0] == np.hypot(pgen.gfs_winds['u'].iloc[0], 
                                                     pgen.gfs_winds['v'].iloc[0])).all()
    t_order = pgen.gfs_winds['speed'].iloc[0][10] > pgen.gfs_winds['speed'].iloc[0][0]
    return t_speed and t_order

def test_matching():
    pgen = pws.gen_params.ParameterGenerator()
    pgen.gfs_loaded = pgen._load_gfs()
    pgen.cp_loaded = pgen._load_telemetry()

    try: 
        matched = pgen._match_data()
        pgen.cp_telemetry['matched']
    except:
        matched = False

    return matched

if __name__ == '__main__':
    
    assert test_loading(), "loading didn't work!"

    assert test_processing(), "processing didn't work!"

    assert test_matching(), "matching didn't work!"
