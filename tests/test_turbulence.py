import psfws
import numpy as np


def test_joint_pdf():
    # iterate through some semi-random values of rho
    rho_goal = [0.2, 0.45, 0.62, 0.83]
    rho_result = []
    for rho_jv in rho_goal:
        p = psfws.ParameterGenerator(rho_j_wind=rho_jv)
        rho_result.append(np.corrcoef(p.data_gl['j_gl'],
                                      p.data_gl['speed'])[0][1])
    # check whether results are within some tolerance of desired values
    np.testing.assert_allclose(rho_result, rho_goal, rtol=.1, atol=.05,
                               err_msg='correlating marginal pdfs failed.')


def test_turbulence_draws():
    # first test with no wind correlation for ground layer
    j_gl_goal = 3.880497137403186e-13
    j_fa_goal = 1.936532096294314e-13
    p = psfws.ParameterGenerator(location='cerro-pachon',
                                 rho_j_wind=None, seed=460)
    j_fa_res, j_gl_res = p._draw_j()

    np.testing.assert_equal([j_fa_res, j_gl_res], [j_fa_goal, j_gl_goal],
                            err_msg='error reproducing turbulence integrals')

    # second, test with wind correlation for ground layer
    j_gl_goal = 1.4466640614052723e-13
    j_fa_goal = 3.3034875838653924e-14
    p = psfws.ParameterGenerator(rho_j_wind=.7, seed=2012)
    pt = p._rng.choice(p.data_fa.index)
    j_fa_res, j_gl_res = p._draw_j(pt)

    np.testing.assert_equal([j_fa_res, j_gl_res], [j_fa_goal, j_gl_goal],
                            err_msg='error reproducing turbulence integrals')


def test_turbulence_weights():
    # first make sure number of weights == number of layers
    p = psfws.ParameterGenerator(seed=85647724)
    pt = p._rng.choice(p.data_fa.index)
    j, layers = p.get_turbulence_integral(pt)

    # come back to this test after adding new layer method!
    # np.testing.assert_equal([len(j)],[len(layers)],
    #                         err_msg='unequal number of layers and weights')

    # test sum FA weights = j integral
    p = psfws.ParameterGenerator(seed=85647724)
    pt = p._rng.choice(p.data_fa.index)
    j_fa, j_gl = p._draw_j(pt)
    np.testing.assert_allclose([np.sum(j[1:])], [j_fa])


if __name__ == '__main__':
    test_joint_pdf()
    test_turbulence_draws()
    test_turbulence_weights()
