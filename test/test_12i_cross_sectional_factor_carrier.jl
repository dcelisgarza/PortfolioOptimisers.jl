#=
Check that `LowOrderPrior`'s `rr` slot admits a `CrossSectionalFactorModel`, and that every
consumer of the slot answers for one exactly as it answers for a `Regression`. Issue #707.

THE FIXTURE IS A PAIR, AND THAT IS THE WHOLE METHOD. The two results are built over one `M`,
one `L` and one `b`, so a consumer that reads the slot correctly returns the SAME value for
both, and `==` rather than `isapprox` is the assertion. A consumer that reached past the
loadings into a field only `Regression` has would raise here instead.

TWO CARRIERS, BECAUSE THE FACTOR AXIS DIFFERS. A model fitted in a re-based Factor Family
carries `L` and `fcb`, and issue #649 put `fpr` on the RAW factor axis, which the re-basis
makes a linear image of a smaller one. Its `fpr.sigma` is singular by construction. The
fixture builds that case honestly — `L = M * fcb`, and the raw factor returns are the reduced
ones lifted through `fcb` — so the singularity is the fixture's arithmetic rather than a
written number. The flat carrier re-bases nothing and its factor covariance is full rank.

WHAT REFUSES AND WHAT DOES NOT. `BayesianBlackLittermanPrior` inverts `fpr.sigma`, so it
refuses a re-based model. `HighOrderFactorPriorEstimator` only projects through `rr.M`, so it
accepts one. Both are probed, because ADR 0046 asks the estimator's docstring to state which
it is, and a test is what keeps the two statements together.

`fcb` carries the family re-basis and issue #651 settles what it is. This file uses the one
rule the result states today: `L` and `fcb` are present together or absent together.

`FixedPrior` holds a THUNK, not a Result. An estimator never holds a Result in a field, and a
stub in a test file is not an exception worth making.
=#
struct FixedPrior{T} <: PortfolioOptimisers.AbstractLowOrderPriorEstimator_F
    f::T
end
function PortfolioOptimisers.prior(pe::FixedPrior, ::PortfolioOptimisers.MatNum,
                                   ::PortfolioOptimisers.Option{<:PortfolioOptimisers.MatNum} = nothing;
                                   dims::Int = 1, kwargs...)
    return pe.f()
end
function PortfolioOptimisers.factor_residual_config(::FixedPrior)
    return nothing
end

@testset "A CrossSectionalFactorModel in the prior's rr slot" begin
    PO = PortfolioOptimisers

    # 8 observations, 4 assets, 3 raw factors, 2 factors after the re-basis.
    M = [1.0 0.5 0.2
         0.3 1.2 0.4
         0.8 0.1 1.1
         0.6 0.9 0.7]
    b = [0.01, 0.02, 0.03, 0.04]
    # The third raw factor is the one the constrained family drops, and the re-basis rewrites
    # the pair it constrained against it. `L` is what `M` becomes in that basis.
    fcb = [1.0 0.0
           0.0 1.0
           -1.0 -1.0]
    L = M * fcb
    T = 8
    # Reduced factor returns, written out. The raw ones are these lifted through `fcb`, which
    # is what makes the raw factor covariance rank 2 over 3 columns.
    Fred = [0.010 -0.020
            -0.030 0.015
            0.025 0.005
            -0.010 -0.025
            0.018 0.030
            0.005 -0.012
            -0.022 0.008
            0.014 0.020]
    F_deg = Fred * transpose(fcb)
    # A third raw factor that is not a combination of the other two, for the flat carrier.
    F_ok = hcat(Fred, [0.007, -0.011, 0.019, 0.004, -0.026, 0.013, 0.009, -0.017])

    esigma = [0.0004, 0.0009, 0.0006, 0.0011]

    function carrier(Fr, rr)
        f_mu = vec(sum(Fr; dims = 1)) / T
        Fc = Fr .- transpose(f_mu)
        f_sigma = transpose(Fc) * Fc / (T - 1)
        fpr = LowOrderPrior(; X = Fr, mu = f_mu, sigma = f_sigma)
        X = Fr * transpose(M) .+ transpose(b)
        mu = M * f_mu + b
        sigma = M * f_sigma * transpose(M) + LinearAlgebra.diagm(esigma)
        return LowOrderPrior(; X = X, mu = mu, sigma = sigma, rr = rr, fpr = fpr)
    end

    csr = CrossSectionalRegression(; f = Fred, eps = zeros(T, 4), n = fill(4, T),
                                   b = zeros(T))
    reg_reb = Regression(; M = M, L = L, b = b)
    csfm_reb = CrossSectionalFactorModel(; M = M, L = L, b = b, csr = csr, fcb = fcb,
                                         esigma = esigma, fam = ["mkt", "size", "value"],
                                         lag = 1)
    reg_flat = Regression(; M = M, b = b)
    csfm_flat = CrossSectionalFactorModel(; M = M, b = b, csr = csr, esigma = esigma,
                                          fam = ["mkt", "size", "value"], lag = 1)

    pr_reg_reb = carrier(F_deg, reg_reb)
    pr_csfm_reb = carrier(F_deg, csfm_reb)
    pr_reg_flat = carrier(F_ok, reg_flat)
    pr_csfm_flat = carrier(F_ok, csfm_flat)

    @testset "The slot is bound to the loadings root" begin
        # The bound is the enforcement, so it is what the test reads.
        @test fieldtype(typeof(pr_csfm_reb), :rr) <: CrossSectionalFactorModel
        @test CrossSectionalFactorModel <: PO.AbstractLoadingsRegressionResult
        @test !(CrossSectionalFactorModel <: Regression)
        @test isa(pr_csfm_reb, LowOrderPrior)
        @test pr_csfm_reb.rr === csfm_reb
        # The positional constructor takes it too, not only the keyword one.
        @test isa(LowOrderPrior(pr_csfm_reb.X, nothing, pr_csfm_reb.mu, pr_csfm_reb.sigma,
                                nothing, nothing, nothing, nothing, nothing, csfm_reb,
                                pr_csfm_reb.fpr, nothing), LowOrderPrior)
    end

    @testset "The block invariants still read rr.M" begin
        # Both invariants of the block fire for the new member exactly as they do for a
        # `Regression`: the factor axis of `M` against `fpr`, and its asset axis against `mu`.
        wide = CrossSectionalFactorModel(; M = hcat(M, M[:, 1]), b = b)
        @test_throws DimensionMismatch LowOrderPrior(; X = pr_csfm_reb.X,
                                                     mu = pr_csfm_reb.mu,
                                                     sigma = pr_csfm_reb.sigma, rr = wide,
                                                     fpr = pr_csfm_reb.fpr)
        short = CrossSectionalFactorModel(; M = M[1:3, :], b = b[1:3])
        @test_throws DimensionMismatch LowOrderPrior(; X = pr_csfm_reb.X,
                                                     mu = pr_csfm_reb.mu,
                                                     sigma = pr_csfm_reb.sigma, rr = short,
                                                     fpr = pr_csfm_reb.fpr)
        # `fpr` still travels with `rr`, whichever member sits in the slot.
        @test_throws ArgumentError LowOrderPrior(; X = pr_csfm_reb.X, mu = pr_csfm_reb.mu,
                                                 sigma = pr_csfm_reb.sigma, rr = csfm_reb,
                                                 fpr = nothing)
    end

    @testset "has_family_rebasis states the re-basis and nothing else" begin
        # The root answers `false`, and that is an answer rather than a missing method: a
        # `Regression` re-bases no family, whether or not it carries an `L`.
        @test !PO.has_family_rebasis(reg_flat)
        @test !PO.has_family_rebasis(reg_reb)
        # The member reads the field, so an unset `L` answers `false` even though `csfm.L`
        # returns `M`.
        @test isnothing(getfield(csfm_flat, :L))
        @test csfm_flat.L == M
        @test !PO.has_family_rebasis(csfm_flat)
        @test PO.has_family_rebasis(csfm_reb)
        # The fixture's own arithmetic: a re-basis makes the raw factor covariance singular,
        # and the flat one keeps it invertible.
        @test LinearAlgebra.rank(pr_csfm_reb.fpr.sigma) == 2
        @test LinearAlgebra.rank(pr_csfm_flat.fpr.sigma) == 3
    end

    @testset "port_opt_view recurses into the model" begin
        i = [3, 1]
        v_reg = PO.port_opt_view(pr_reg_reb, i)
        v_csfm = PO.port_opt_view(pr_csfm_reb, i)
        @test isa(v_csfm.rr, CrossSectionalFactorModel)
        # The loadings answer identically, which is the whole point of the pair.
        @test v_csfm.rr.M == v_reg.rr.M == M[i, :]
        @test v_csfm.rr.L == v_reg.rr.L == L[i, :]
        @test v_csfm.rr.b == v_reg.rr.b == b[i]
        # The model's own fields are cut by its own method, and the fit comes with it.
        @test v_csfm.rr.esigma == esigma[i]
        @test v_csfm.rr.csr.eps == csr.eps[:, i]
        @test v_csfm.rr.fcb === fcb
        @test v_csfm.rr.lag === 1
        # `fpr` is a distribution over factors, so an asset index leaves it alone.
        @test v_csfm.fpr === pr_csfm_reb.fpr
        # A viewed model that re-bases still says so, and a viewed flat one still does not.
        @test PO.has_family_rebasis(v_csfm.rr)
        @test isnothing(getfield(PO.port_opt_view(pr_csfm_flat, i).rr, :L))
    end

    @testset "The five L readers answer as they do for a Regression" begin
        rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = pr_csfm_reb.X,
                           nf = ["mkt", "size", "value"], F = F_deg)

        # 1. `RegressionFeatures`, at `13_Prior/15_FeaturePrior.jl`.
        @test feature_matrix(RegressionFeatures(), pr_csfm_reb) ==
              feature_matrix(RegressionFeatures(), pr_reg_reb) ==
              L
        @test feature_matrix(RegressionFeatures(), pr_csfm_flat) ==
              feature_matrix(RegressionFeatures(), pr_reg_flat) ==
              M

        # 2. `factor_risk_contribution`, at `19_RiskMeasures/27_ExpectedRisk.jl`.
        w = [0.4, 0.1, 0.3, 0.2]
        rc_csfm = factor_risk_contribution(Variance(), w, pr_csfm_reb; rd = rd)
        rc_reg = factor_risk_contribution(Variance(), w, pr_reg_reb; rd = rd)
        @test rc_csfm == rc_reg
        @test length(rc_csfm) == size(L, 2) + 1

        # 3. `set_factor_risk_contribution_constraints!`, at
        #    `20_Optimisation/12_FactorRiskContribution.jl`.
        b1_c, rr_c = PO.set_factor_risk_contribution_constraints!(PO.JuMP.Model(),
                                                                  StepwiseRegression(), rd,
                                                                  pr_csfm_reb, true,
                                                                  nothing)
        b1_r, rr_r = PO.set_factor_risk_contribution_constraints!(PO.JuMP.Model(),
                                                                  StepwiseRegression(), rd,
                                                                  pr_reg_reb, true, nothing)
        @test b1_c == b1_r
        @test rr_c === csfm_reb
        @test rr_r === reg_reb

        # 4. The expression `set_relaxed_risk_budgeting_constraints!` forms from that `rr`,
        #    at `20_Optimisation/15_RelaxedRiskBudgeting.jl`. It is built here on the `rr`
        #    step 3 returned, so it is the same read on the same object.
        @test Matrix(LinearAlgebra.Symmetric(rr_c.L \ pr_csfm_reb.sigma * b1_c)) ==
              Matrix(LinearAlgebra.Symmetric(rr_r.L \ pr_reg_reb.sigma * b1_r))

        # 5. The factor budget axis, at `20_Optimisation/14_RiskBudgeting.jl`. The caller
        #    passes `size(rr.L, 2)`, so the reduced basis is what the names must match.
        @test size(rr_c.L, 2) == size(rr_r.L, 2) == 2
        rba = FactorRiskBudgeting(; rkb = RiskBudgetEstimator(; val = ["r1" => 0.5]),
                                  sets = UniverseSets(;
                                                      dict = Dict("nx" =>
                                                                      ["A", "B", "C", "D"],
                                                                  "nf" => ["r1", "r2"])))
        @test PO.risk_budget_universe_key(rba, size(rr_c.L, 2)) ==
              PO.risk_budget_universe_key(rba, size(rr_r.L, 2)) ==
              "nf"
        @test_throws DimensionMismatch PO.risk_budget_universe_key(rba, size(M, 2))
    end

    @testset "The M readers answer as they do for a Regression" begin
        sets = UniverseSets(;
                            dict = Dict("nx" => ["A", "B", "C", "D"],
                                        "nf" => ["mkt", "size", "value"],
                                        "cyclical" => ["size", "value"]))

        # 1. `FactorSpace`'s axis check, at
        #    `12_ConstraintGeneration/08_ExposureConstraintGeneration.jl`.
        rr_c, key_c = PO.constraint_space_basis(FactorSpace(), sets, csfm_reb)
        rr_r, key_r = PO.constraint_space_basis(FactorSpace(), sets, reg_reb)
        @test rr_c === csfm_reb
        @test rr_r === reg_reb
        @test key_c == key_r == "nf"
        # The check reads `rr.M`, the raw axis, so it is the named factors it counts and not
        # the two the re-basis left.
        short_sets = UniverseSets(;
                                  dict = Dict("nx" => ["A", "B", "C", "D"],
                                              "nf" => ["r1", "r2"]))
        @test_throws DimensionMismatch PO.constraint_space_basis(FactorSpace(), short_sets,
                                                                 csfm_reb)

        # 2. `constraint_row_term`, at
        #    `12_ConstraintGeneration/02_LinearConstraintGeneration.jl`.
        Ai = [true, false, true]
        @test PO.constraint_row_term(csfm_reb, Ai, 2.0) ==
              PO.constraint_row_term(reg_reb, Ai, 2.0) ==
              vec(sum(view(M, :, Ai); dims = 2)) * 2.0
        @test PO.constraint_row_length(csfm_reb, ["mkt", "size", "value"]) ==
              PO.constraint_row_length(reg_reb, ["mkt", "size", "value"]) ==
              4
    end

    @testset "A wrapping prior that projects accepts the model" begin
        # `HighOrderFactorPriorEstimator` reads `pr.rr.M` and never inverts the factor
        # covariance, so a re-based model is fine for it. That is the statement its own
        # docstring and `LowOrderPrior`'s make, and this is what holds them to it.
        X = pr_csfm_reb.X
        h = HighOrderFactorPriorEstimator(; pe = FixedPrior(() -> pr_csfm_reb), rsd = false)
        h_ref = HighOrderFactorPriorEstimator(; pe = FixedPrior(() -> pr_reg_reb),
                                              rsd = false)
        hp = prior(h, X, F_deg)
        hp_ref = prior(h_ref, X, F_deg)
        @test hp.kt == hp_ref.kt
        @test hp.pr.mu == hp_ref.pr.mu
    end

    @testset "A wrapping prior that inverts the factor covariance refuses a re-basis" begin
        X = pr_csfm_reb.X
        views = BlackLittermanViews(; P = [1.0 0.0 0.0], Q = [0.01])
        # The flat carrier passes: nothing about the member is refused, only the re-basis.
        ok = BayesianBlackLittermanPrior(; pe = FixedPrior(() -> pr_csfm_flat),
                                         views = views)
        ok_ref = BayesianBlackLittermanPrior(; pe = FixedPrior(() -> pr_reg_flat),
                                             views = views)
        p = prior(ok, X, F_ok)
        p_ref = prior(ok_ref, X, F_ok)
        @test p.mu == p_ref.mu
        @test p.sigma == p_ref.sigma
        @test p.fpr.sigma == p_ref.fpr.sigma

        # The re-based carrier is refused, and the message names the estimator and the way
        # out rather than surfacing a singular solve from the master equations.
        bad = BayesianBlackLittermanPrior(; pe = FixedPrior(() -> pr_csfm_reb),
                                          views = views)
        err = try
            prior(bad, X, F_deg)
            nothing
        catch e
            e
        end
        @test isa(err, ArgumentError)
        @test occursin("re-based Factor Family", err.msg)
        @test occursin("BayesianBlackLittermanPrior", err.msg)
        @test occursin("BlackLittermanPrior", err.msg)
        @test occursin("CrossSectionalFactorModel", err.msg)

        # A `Regression` carrying the same singular factor block is NOT refused, and this is
        # what the refusal is worth. The trait states a re-basis rather than a rank, so the
        # legacy route reaches the same inversion — and the inversion RAISES NOTHING. The
        # solve returns entries of order 1e18 against an input of order 1e-3, the update
        # carries on, and the posterior it reports is on the prior's own scale, so nothing
        # in the answer says it is meaningless. A rank test would need a tolerance this
        # matrix passes, which is why the refusal reads what the result states instead.
        @test !PO.has_family_rebasis(reg_reb)
        @test maximum(abs, pr_reg_reb.fpr.sigma) < 1e-2
        @test maximum(abs, pr_reg_reb.fpr.sigma \ LinearAlgebra.I) > 1e6
        legacy = BayesianBlackLittermanPrior(; pe = FixedPrior(() -> pr_reg_reb),
                                             views = views)
        p_legacy = prior(legacy, X, F_deg)
        @test isa(p_legacy, LowOrderPrior)
        @test all(isfinite, p_legacy.mu)
        @test maximum(abs, p_legacy.fpr.sigma) < 1e-2
    end
end
