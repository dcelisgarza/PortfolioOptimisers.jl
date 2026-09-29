#=
The fixture of the Return Forecast evaluation, shared by `test_08x_forecast_evaluation.jl` and
`test_08x_parity_forecast_evaluation.jl`. Include it after `test06c_setup.jl`, which defines
`synthetic_asset_panel`.
=#
# The synthetic panel of issue #656, with a factor-model block fitted on a strict suffix of
# the carrier so every probe of the cut has something to cut.
#
# `planted` drives the idiosyncratic return off the composite score instead of off a
# sinusoid, so a member refitted at each date has a relation to find. Convention 4.
function evaluation_fixture(; n_observations::Integer = 60, drop::Integer = 8,
                            planted::Bool = false)
    sp = synthetic_asset_panel(; n_assets = 20, n_observations = n_observations,
                               n_industries = 4, late_listing_proba = 0.3,
                               delisting_proba = 0.3, missing_ratio = 0.08,
                               rng = StableRNG(987654321))
    rd = sp.rd
    pnl = rd.pnl
    T, N = size(pnl.amsk)
    rows = (drop + 1):T
    Tb = length(rows)
    ct_out = CrossSectionalWinsoriser()
    ct_sco = CrossSectionalStandardiser(; min_group_size = 2)
    xc = CompositeExposure(;
                           descriptors = [Passthrough(; field = "book_equity"),
                                          Passthrough(; field = "market_cap")],
                           weights = [0.4, 0.6], min_coverage = 0.5, outlier = ct_out,
                           scoring = ct_sco, group = "industry", bw = "market_cap")
    Lo = factor_exposure(OneHotExposure(; field = "industry", family = "industry"), rd)
    K = 1 + size(Lo, 3)
    Ms = Array{Float64, 3}(undef, T, N, K)
    Z = factor_exposure(xc, rd)
    Ms[:, :, 1] = Z
    for k in 1:size(Lo, 3)
        Ms[:, :, k + 1] = Lo[:, :, k]
    end
    nf = ["style"; ["ind$k" for k in 1:size(Lo, 3)]]
    fam = ["style"; fill("industry", size(Lo, 3))]
    vs = [pnl.amsk[t, i] ? 0.0004 * (1.5 + sin(0.3 * t + 0.7 * i)) : NaN
          for t in rows, i in 1:N]
    rng = StableRNG(24680)
    eps = if planted
        [if pnl.amsk[t, i] && isfinite(Z[t, i])
             0.01 * Z[t, i] + 0.003 * randn(rng)
         else
             NaN
         end
         for t in rows, i in 1:N]
    else
        [if pnl.amsk[t, i]
             0.01 * sin(0.7 * t + 0.29 * i) + 0.004 * cos(0.11 * t * i)
         else
             NaN
         end
         for t in rows, i in 1:N]
    end
    csr = CrossSectionalRegression(; f = zeros(Tb, K), eps = eps, n = fill(N, Tb))
    csfm = CrossSectionalFactorModel(; M = Ms[end, :, :], b = zeros(N), csr = csr,
                                     Ms = Ms[rows, :, :], vs = vs, nf = nf, fam = fam)
    scores = DescriptorScores(;
                              descriptors = [Passthrough(; field = "book_equity"),
                                             Passthrough(; field = "market_cap")],
                              outlier = ct_out, scoring = ct_sco, group = "industry")
    return (; rd = rd, csfm = csfm, scores = scores, rows = rows, T = T, N = N, Tb = Tb)
end
