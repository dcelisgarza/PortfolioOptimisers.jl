#=
Currency factors at parity with a stored oracle, and inside the meta-optimisers and the folds
(#1369).

The oracle under `test/assets/CrossSectionalFactorPriorCurrency*.csv.gz` holds the factor
returns, the factor mean, the factor covariance and `mu` of the same fit on the fixture below,
with and without a constrained industry family. The nested factor prior of both is a plain
`EmpiricalPrior`. The fixture draws local returns, adds the Currency Excess Return of the currency
of each asset to form the base-currency returns, and states three currencies, so the fit must
recover the local model through the derived local returns.

The `*Default*` files hold the default fit on the same fixture (#1373, #1374): the factor mean,
the factor covariance, `mu`, `sigma` and the idiosyncratic variance of the latest observation.
Its factor prior decays on the half-life of the idiosyncratic variance, and both of them measure
a second moment about zero.
=#
using Statistics, Clarabel

function ccy_fixture(; T = 120, N = 40, seed = 927_001)
    rng = StableRNG(seed)
    ind = [mod(i - 1, 3) + 1 for i in 1:N]
    codes = [mod(i, 3) + 1 for i in 1:N]
    I3 = zeros(T, N, 3)
    for i in 1:N
        I3[:, i, ind[i]] .= 1.0
    end
    z(A) = (A .- mean(A; dims = 2)) ./ std(A; dims = 2, corrected = false)
    s1 = z(randn(rng, T, N) .+ randn(rng, 1, N) .* 3)
    s2 = z(randn(rng, T, N) .+ randn(rng, 1, N) .* 3)
    mcap = exp.(randn(rng, 1, N) .* 0.8 .+ 0.05 .* cumsum(randn(rng, T, N); dims = 1))
    fi = 0.01 .* randn(rng, T, 3)
    fs = 0.004 .* randn(rng, T, 2)
    loc = zeros(T, N)
    for t in 2:T, i in 1:N
        loc[t, i] = fi[t, ind[i]] +
                    s1[t - 1, i] * fs[t, 1] +
                    s2[t - 1, i] * fs[t, 2] +
                    0.01 * randn(rng)
    end
    loc[1, :] .= 0.01 .* randn(rng, N)
    R = 0.005 .* randn(rng, T, 3)
    X = copy(loc)
    for t in 1:T, i in 1:N
        X[t, i] += R[t, codes[i]]
    end
    lv = ["EUR", "JPY", "USD"]
    pf = [NumericPanelInput(; name = "market_cap", vals = mcap),
          NumericPanelInput(; name = "style1", vals = s1),
          NumericPanelInput(; name = "style2", vals = s2),
          [NumericPanelInput(; name = "ind$k", vals = I3[:, :, k]) for k in 1:3]...,
          CategoricalPanelInput(; name = "currency",
                                vals = repeat(permutedims(lv[codes]), T))]
    pnl = asset_panel(pf; amsk = trues(T, N), emsk = trues(T, N))
    rd = ReturnsResult(; nx = ["a$i" for i in 0:(N - 1)], X = X, ne = lv, E = R, pnl = pnl)
    return (; rd, R, codes, ind)
end
function ccy_pass(field, family)
    return CompositeExposure(; descriptors = [Passthrough(; field = field)],
                             outlier = nothing, scoring = nothing, family = family)
end
function ccy_asset(name)
    return Matrix(CSV.read(joinpath(@__DIR__, "assets",
                                    "CrossSectionalFactorPriorCurrency$(name).csv.gz"),
                           DataFrame))
end

@testset "Currency factors at parity, and inside the meta-optimisers" begin
    PO = PortfolioOptimisers
    fx = ccy_fixture()
    factors = ["ind1" => ccy_pass("ind1", "industry"),
               "ind2" => ccy_pass("ind2", "industry"),
               "ind3" => ccy_pass("ind3", "industry"),
               "style1" => ccy_pass("style1", "style"),
               "style2" => ccy_pass("style2", "style"), "currency" => CurrencyExposure()]

    @testset "The fit matches the stored oracle, $(nm)" for (nm, fam) in
                                                            (("Plain", nothing),
                                                             ("Family",
                                                              ["industry" => nothing]))
        pr = prior(CrossSectionalFactorPrior(; factors = factors, families = fam,
                                             pe = EmpiricalPrior()), fx.rd)
        F = ccy_asset("$(nm)FactorReturns")
        @test size(pr.fpr.X) == size(F) == (119, 8)
        # The regression solves one weighted least squares by factorising the weighted
        # design, so the factor returns agree to machine precision in absolute terms.
        @test maximum(abs, pr.fpr.X - F) < 1e-14
        # The Currency Factors carry the observed returns, and no regression touched them.
        @test pr.fpr.X[:, 6:8] == fx.R[2:end, :]
        @test pr.fpr.mu ≈ vec(ccy_asset("$(nm)FactorMu")) rtol = 1e-12
        @test pr.fpr.sigma ≈ ccy_asset("$(nm)FactorCov") rtol = 1e-12
        @test pr.mu ≈ vec(ccy_asset("$(nm)Mu")) rtol = 1e-12
    end

    @testset "The default fit matches the stored oracle, $(nm)" for (nm, fam) in
                                                                    (("Plain", nothing),
                                                                     ("Family",
                                                                      ["industry" =>
                                                                           nothing]))
        pr = prior(CrossSectionalFactorPrior(; factors = factors, families = fam), fx.rd)
        @test pr.fpr.mu ≈ vec(ccy_asset("$(nm)DefaultFactorMu")) rtol = 1e-12
        @test pr.fpr.sigma ≈ ccy_asset("$(nm)DefaultFactorCov") rtol = 1e-12
        @test pr.mu ≈ vec(ccy_asset("$(nm)DefaultMu")) rtol = 1e-12
        # The idiosyncratic variance is a second moment about zero, because the model sets
        # the mean of the idiosyncratic return to zero (#1373).
        @test pr.rr.vs[end, :] ≈ vec(ccy_asset("$(nm)DefaultIdioVar")) rtol = 1e-12
        @test pr.sigma ≈ ccy_asset("$(nm)DefaultSigma") rtol = 1e-12
    end

    # Every cluster and every subset must estimate its own factors, so the factors here are
    # continuous styles and a market intercept, which no subset of assets leaves empty.
    pe = CrossSectionalFactorPrior(;
                                   factors = ["style1" => ccy_pass("style1", "style"),
                                              "style2" => ccy_pass("style2", "style"),
                                              "market" => ConstantExposure(),
                                              "currency" => CurrencyExposure()], minra = 3)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    inner = MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv))

    @testset "Every cluster of a NestedClustered fits its own currency factors" begin
        res = optimise(NestedClustered(; pe = pe, opti = inner, opto = EqualWeighted()),
                       fx.rd)
        @test sum(res.w) ≈ 1
        @test all(r -> r.retcode isa OptimisationSuccess, res.resi)
        for r in res.resi
            @test r.pr.rr.nf[(end - 2):end] ==
                  ["currency=EUR", "currency=JPY", "currency=USD"]
            @test r.pr.rr.fx == fx.R[2:end, :]
        end
    end

    @testset "Every subset of a SubsetResampling fits its own currency factors" begin
        res = optimise(SubsetResampling(; subset_size = 20, n_subsets = 4,
                                        rng = StableRNG(9), pe = pe, opt = inner), fx.rd)
        @test sum(res.w) ≈ 1
        @test res.pr.rr.fx == fx.R[2:end, :]
    end

    @testset "Every fold of a walk-forward reads the Exogenous Series of its own rows" begin
        cv = cross_val_predict(inner, fx.rd, IndexWalkForward(60, 20))
        @test length(cv.pred) == 3
        for (k, p) in enumerate(cv.pred)
            tr = (20 * (k - 1) + 1):(20 * (k - 1) + 60)
            @test p.res.pr.rr.fx == fx.R[tr[2:end], :]
        end
        @test cv.mrd.E == fx.R[61:120, :]
        @test cv.mrd.ne == ["EUR", "JPY", "USD"]
    end

    @testset "A cross-sectional prior of a meta-optimiser answers on its fitted rows" begin
        # The outer problem reads one row of net returns per scenario of the prior, and a
        # cross-sectional prior answers on the rows it fitted, the last ones of the data.
        pr = prior(pe, fx.rd)
        rdo = PO.outer_prior_rows(fx.rd, pr)
        @test size(rdo.X, 1) == size(pr.X, 1) == 119
        @test rdo.X == fx.rd.X[2:end, :]
        @test rdo.E == fx.R[2:end, :]
        @test PO.outer_prior_rows(fx.rd, prior(EmpiricalPrior(), fx.rd)) === fx.rd
        big = LowOrderPrior(; X = randn(StableRNG(1), 200, 40), mu = zeros(40),
                            sigma = Matrix(1.0I, 40, 40))
        @test_throws DimensionMismatch PO.outer_prior_rows(fx.rd, big)
    end
end
