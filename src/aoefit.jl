f_aoe_compton(x, v) = gauss_pdf(x, v.μ, v.σ) + v.B * ex_step_gauss(x, v.l, v.k, v.t, v.d)

