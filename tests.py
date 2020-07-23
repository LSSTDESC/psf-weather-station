import numpy as np
import gen_params
import pathlib 

def test_loading():
    pgen = gen_params.ParameterGenerator()

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

def test_matching():
    pgen = gen_params.ParameterGenerator()
    pgen.gfs_loaded = pgen._load_gfs()
    pgen.cp_loaded = pgen._load_telemetry()

    try: 
        matched = pgen._match_data()
        pgen.cp_telemetry['matched']
    except:
        matched = False

    return matched

if __name__ == '__main__':
    
    assert test_loading() == True, "loading didn't work!"

    # test_matching()
    assert test_matching() == True, "matching didn't work!"
