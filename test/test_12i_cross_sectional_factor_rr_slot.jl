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
fixture builds that case honestly — `L` is `M` reduced through `fcb`, and the raw factor returns
are the reduced ones expanded through it — so the singularity is the fixture's arithmetic rather than a
written number. The flat carrier re-bases nothing and its factor covariance is full rank.

WHAT REFUSES AND WHAT DOES NOT. `BayesianBlackLittermanPrior` inverts `fpr.sigma`, so it
refuses a re-based model. `HighOrderFactorPriorEstimator` only projects through `rr.M`, so it
accepts one. Both are probed, because ADR 0046 asks the estimator's docstring to state which
it is, and a test is what keeps the two statements together.

`fcb` carries the family re-basis, which issue #651 settled and issue #724 built. This file uses
the one rule the result states: `L` and `fcb` are present together or absent together.

`FixedPrior` holds a THUNK, not a Result. An estimator never holds a Result in a field, and a
stub in a test file is not an exception worth making.
=#
struct FixedPrior{T} <: PortfolioOptimisers.AbstractLowOrderPriorEstimator_F
    f::T
end
function PortfolioOptimisers.prior(pe::FixedPrior, ::PortfolioOptimisers.MatNum,
                                   ::PortfolioOptimisers.Option{<:PortfolioOptimisers.MatNum} = nothing,
                                   ::PortfolioOptimisers.Option{<:PortfolioOptimisers.AssetPanel} = nothing;
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
    # `Rt` is the dense change of basis this family expresses: the first two factors are
    # retained and the third is reconstructed from them, so every ratio is one.
    Rt = [1.0 0.0
          0.0 1.0
          -1.0 -1.0]
    T = 8
    fcb = FactorFamilyBasis(; fnm = ["f"], fi = [[1, 2, 3]], di = [3], ratios = ones(T, 2),
                            K = 3)
    L = PO.reduce_loadings(fcb, M)
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
    F_deg = PO.expand_factor_returns(fcb, Fred)
    # The compact basis is the dense one, so the fixture states both and checks they agree.
    @test L == M * Rt
    @test F_deg == Fred * transpose(Rt)
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
                                pr_csfm_reb.fpr), LowOrderPrior)
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

        # 1. `RegressionPanel`, at `10_Prior/09_AssetPanelEstimators.jl`. The producer
        #    reads `pr.rr.L` for both regression results with one method.
        loadings(pr) = PO.panel_field(PO.asset_panel(RegressionPanel(), pr, rd, rd.X),
                                      "loadings").vals
        @test loadings(pr_csfm_reb) == loadings(pr_reg_reb) == L
        @test loadings(pr_csfm_flat) == loadings(pr_reg_flat) == M
        # The factor axis is named off the carrier only where it *is* the carrier's, which is
        # a `Regression` whose loadings are the raw `M`; every other case is positional.
        labels(pr) = PO.panel_field(PO.asset_panel(RegressionPanel(), pr, rd, rd.X),
                                    "loadings").labels
        @test labels(pr_reg_flat) == rd.nf
        @test labels(pr_csfm_flat) == [string(k) for k in 1:size(M, 2)]
        @test labels(pr_reg_reb) == [string(k) for k in 1:size(L, 2)]

        # 2. `factor_risk_contribution`, at `16_RiskMeasures/24_ExpectedRisk.jl`.
        w = [0.4, 0.1, 0.3, 0.2]
        rc_csfm = factor_risk_contribution(Variance(), w, pr_csfm_reb; rd = rd)
        rc_reg = factor_risk_contribution(Variance(), w, pr_reg_reb; rd = rd)
        @test rc_csfm == rc_reg
        @test length(rc_csfm) == size(L, 2) + 1

        # 3. `set_factor_risk_contribution_constraints!`, at
        #    `17_Optimisation/05_JuMP/05_FactorRiskContribution.jl`.
        b1_c, _, rr_c = PO.set_factor_risk_contribution_constraints!(PO.JuMP.Model(),
                                                                     StepwiseRegression(),
                                                                     rd, pr_csfm_reb, true,
                                                                     nothing)
        b1_r, _, rr_r = PO.set_factor_risk_contribution_constraints!(PO.JuMP.Model(),
                                                                     StepwiseRegression(),
                                                                     rd, pr_reg_reb, true,
                                                                     nothing)
        @test b1_c == b1_r
        @test rr_c === csfm_reb
        @test rr_r === reg_reb

        # 4. The expressions `set_relaxed_risk_budgeting_constraints!` forms from that `rr`,
        #    at `17_Optimisation/05_JuMP/08_RelaxedRiskBudgeting.jl`: the factor covariance
        #    under `flag = false`, and the factor marginal risks under `flag = true`. They are
        #    built here on the `rr` step 3 returned, so they are the same reads on the same
        #    object.
        @test Matrix(LinearAlgebra.Symmetric(rr_c.L \ pr_csfm_reb.sigma * b1_c)) ==
              Matrix(LinearAlgebra.Symmetric(rr_r.L \ pr_reg_reb.sigma * b1_r))
        @test transpose(b1_c) * pr_csfm_reb.sigma == transpose(b1_r) * pr_reg_reb.sigma

        # 5. The factor budget axis, at `17_Optimisation/05_JuMP/07_RiskBudgeting.jl`. The caller
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
        # #723 split the factor axis in two, so the one read that deliberately *differs*
        # is the axis key: the cross-sectional model resolves its names against `cfkey`
        # and the regression against `tfkey`. Every other `M` read still answers alike,
        # which is what the rest of this testset holds.
        sets = UniverseSets(;
                            dict = Dict("nx" => ["A", "B", "C", "D"],
                                        "nf" => ["mkt", "size", "value"],
                                        "ncf" => ["mkt", "size", "value"],
                                        "cyclical" => ["size", "value"]))

        # 1. `FactorSpace`'s axis check, at
        #    `09_ConstraintGeneration/08_ExposureConstraintGeneration.jl`.
        rr_c, key_c = PO.constraint_space_basis(FactorSpace(), sets, csfm_reb)
        rr_r, key_r = PO.constraint_space_basis(FactorSpace(), sets, reg_reb)
        @test rr_c === csfm_reb
        @test rr_r === reg_reb
        # The key follows the block that carries `M`, so a mandate written for one family
        # can never resolve against the other's names.
        @test key_c == sets.cfkey == "ncf"
        @test key_r == sets.tfkey == "nf"
        # The check reads `rr.M`, the raw axis, so it is the named factors it counts and not
        # the two the re-basis left.
        short_sets = UniverseSets(;
                                  dict = Dict("nx" => ["A", "B", "C", "D"],
                                              "nf" => ["r1", "r2"], "ncf" => ["r1", "r2"]))
        @test_throws DimensionMismatch PO.constraint_space_basis(FactorSpace(), short_sets,
                                                                 csfm_reb)
        @test_throws DimensionMismatch PO.constraint_space_basis(FactorSpace(), short_sets,
                                                                 reg_reb)
        # A sets that declares only the time-series axis is complete for a regression and
        # incomplete for the cross-sectional model, and the `KeyError` names `ncf`.
        ts_only = UniverseSets(;
                               dict = Dict("nx" => ["A", "B", "C", "D"],
                                           "nf" => ["mkt", "size", "value"]))
        @test PO.constraint_space_basis(FactorSpace(), ts_only, reg_reb)[2] == "nf"
        @test_throws KeyError PO.constraint_space_basis(FactorSpace(), ts_only, csfm_reb)
        # A name the cross-sectional axis does not carry is reported against the factor
        # universe it was searched in, under `ncf`, and not against the asset universe.
        # A row stated at `ncf` without loadings that resolves and cancels names the same
        # axis. Under a re-basis that row reads `empty_projected_row_msg` instead. #1273.
        msg = try
            linear_constraints(ExposureConstraintEstimator(;
                                                           lce = LinearConstraintEstimator(;
                                                                                           val = "valu <= 0.3"),
                                                           space = FactorSpace()), sets;
                               rr = csfm_reb, strict = true)
        catch e
            @test e isa ArgumentError
            sprint(showerror, e)
        end
        @test occursin("not in factor universe (3 factors under key `ncf`)", msg)
        @test occursin("did you mean `value`?", msg)
        msg = try
            linear_constraints("size - size <= 0.3", sets, sets.cfkey; strict = true)
        catch e
            sprint(showerror, e)
        end
        @test occursin("the universe (3 factors under key `ncf`)", msg)

        # 2. `constraint_row_term`, at
        #    `09_ConstraintGeneration/02_LinearConstraintGeneration_b.jl`.
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

#=
The two pre-fit cross-sectional factor-axis verbs (issue #724, map #643).

Issue #651 decision 5 settled that the factor names are derived rather than carried, and that a
caller reads them before any fit: one verb answers the axis and its Factor Family labels from the
`factors` Pairs and the Asset Panel's field index, and a second widens a `UniverseSets` with the
axis under `cfkey` and one plain group per family. The block stores the same answer in `nf` and
`fam`, so there is one source of truth.
=#
using Statistics, Distributions, Dates, Random
include(joinpath(@__DIR__, "test06c_setup.jl"))

@testset "The cross-sectional factor axis before any fit" begin
    PO = PortfolioOptimisers
    res = synthetic_asset_panel(; n_assets = 10, n_observations = 30, n_industries = 3,
                                rng = StableRNG(724_101))
    rd = res.rd
    # The size member keeps the default benchmark-weight field, because the last testset builds
    # a prior from these Pairs, and the prior refuses a member that reads another field.
    factors = ["market" => ConstantExposure(),
               "size" =>
                   CompositeExposure(; descriptors = [Passthrough(; field = "market_cap")]),
               "industry" => OneHotExposure(; field = "industry", family = "industry")]

    @testset "A one-hot member expands to one name per level" begin
        ax = PO.cross_sectional_factor_axis(factors, rd)
        @test ax.nf == ["market", "size", "industry=Real Estate", "industry=Software",
                        "industry=Banks"]
        @test ax.fam == ["market", "style", "industry", "industry", "industry"]
        # The axis is the Panel Field's level order, so it is the same in every fold.
        @test ax.nf[3:end] == PO.one_hot_exposure_names(last(factors[3]), rd)
        @test length(ax.nf) == length(ax.fam)
    end

    @testset "Every other member takes the name the caller wrote" begin
        one = PO.cross_sectional_factor_axis(["mkt" => ConstantExposure(; family = "m")],
                                             rd)
        @test one.nf == ["mkt"]
        @test one.fam == ["m"]
        two = PO.cross_sectional_factor_axis(["beta" => DerivedExposure(; source = "size",
                                                                        f = abs)], rd)
        @test two.nf == ["beta"]
        @test two.fam == ["style"]
    end

    @testset "The axis refuses an empty list and a repeated name" begin
        @test_throws PortfolioOptimisers.IsEmptyError PO.cross_sectional_factor_axis(Pair[],
                                                                                     rd)
        @test_throws ArgumentError PO.cross_sectional_factor_axis(["a" =>
                                                                       ConstantExposure(),
                                                                   "a" =>
                                                                       ConstantExposure()],
                                                                  rd)
    end

    @testset "The sets verb declares the axis and one group per family" begin
        sets = PO.cross_sectional_factor_sets(factors, rd)
        ax = PO.cross_sectional_factor_axis(factors, rd)
        @test sets.dict[sets.cfkey] == ax.nf
        @test sets.dict["industry"] ==
              ["industry=Real Estate", "industry=Software", "industry=Banks"]
        @test sets.dict["style"] == ["size"]
        @test sets.dict[sets.xkey] == rd.nx
        # A consumer never names the key by hand: the loadings result it holds picks it.
        csfm = CrossSectionalFactorModel(; M = ones(10, 5), b = zeros(10), nf = ax.nf,
                                         fam = ax.fam)
        @test PO.factor_axis_key(sets, csfm) == sets.cfkey
        @test sets.dict[PO.factor_axis_key(sets, csfm)] == csfm.nf
    end

    @testset "A family label that names a factor is allowed only when it is that factor" begin
        # `market` is both the name of the single factor and the label of its family, which
        # is the case the oracle permits.
        sets = PO.cross_sectional_factor_sets(factors, rd)
        @test sets.dict["market"] == ["market"]
        # A label shared with a factor of another family would answer two different lists.
        clash = ["industry" => ConstantExposure(; family = "size"),
                 "size" => CompositeExposure(;
                                             descriptors = [Passthrough(; field = "market_cap")],
                                             bw = "market_cap", family = "size")]
        @test_throws ArgumentError PO.cross_sectional_factor_sets(clash, rd)
    end

    @testset "The verb widens a declared universe" begin
        s0 = UniverseSets(; xkey = "assets",
                          dict = Dict{String, Any}("assets" => rd.nx, "nf" => ["F1", "F2"],
                                                   "sector" => ["a"]))
        s1 = PO.cross_sectional_factor_sets(factors, rd, s0)
        # Every axis the caller declared survives, and the key prefixes are the caller's.
        @test s1.xkey == "assets"
        @test s1.dict["assets"] == rd.nx
        @test s1.dict[s1.tfkey] == ["F1", "F2"]
        @test s1.dict["sector"] == ["a"]
        @test s1.dict[s1.cfkey] == PO.cross_sectional_factor_axis(factors, rd).nf
        @test s1.dict["industry"] ==
              ["industry=Real Estate", "industry=Software", "industry=Banks"]
        # The original is untouched, so the widening is a new declaration.
        @test !haskey(s0.dict, s0.cfkey)
    end

    @testset "The verb refuses to replace a group or an axis the caller declared" begin
        # A Factor Family label is a plain group name, so a caller's own group can share
        # it. Replacing it in silence would re-point every constraint written against it.
        s0 = UniverseSets(; dict = Dict{String, Any}("nx" => rd.nx, "style" => rd.nx[1:2]))
        @test_throws ArgumentError PO.cross_sectional_factor_sets(factors, rd, s0)
        msg = try
            PO.cross_sectional_factor_sets(factors, rd, s0)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("under style", msg)
        @test s0.dict["style"] == rd.nx[1:2]
        # The same holds for the axis itself under `cfkey`.
        ax = PO.cross_sectional_factor_axis(factors, rd).nf
        s2 = UniverseSets(; dict = Dict{String, Any}("nx" => rd.nx, "ncf" => reverse(ax)))
        @test_throws ArgumentError PO.cross_sectional_factor_sets(factors, rd, s2)
        # The same list is not a replacement, so it is accepted.
        s3 = UniverseSets(;
                          dict = Dict{String, Any}("nx" => rd.nx, "ncf" => ax,
                                                   "style" => ["size"]))
        s4 = PO.cross_sectional_factor_sets(factors, rd, s3)
        @test s4.dict["ncf"] == ax
        @test s4.dict["style"] == ["size"]
    end

    @testset "A new sets needs the asset names" begin
        # The asset axis is the one mandatory axis of a `UniverseSets`, and `ReturnsResult`
        # refuses every carrier that holds data without `nx`, so an empty one is what
        # reaches the refusal.
        bare = ReturnsResult()
        @test isnothing(bare.nx)
        @test_throws ArgumentError PO.cross_sectional_sets_dict(bare, nothing)
        @test_throws ArgumentError PO.cross_sectional_factor_sets(["mkt" =>
                                                                       ConstantExposure()],
                                                                  bare)
    end

    @testset "Both verbs read the axis off the estimator, and both are exported" begin
        # Issue #1059: a caller who writes a factor mandate in a pipeline step holds the
        # estimator, not its Pairs, so the estimator method forwards `pe.factors`.
        pe = CrossSectionalFactorPrior(; lambda = 1, factors = factors)
        # The names resolve unqualified, so the pipeline example can call them.
        @test :cross_sectional_factor_axis in names(PortfolioOptimisers)
        @test :cross_sectional_factor_sets in names(PortfolioOptimisers)
        ax = cross_sectional_factor_axis(pe, rd)
        @test ax == PO.cross_sectional_factor_axis(factors, rd)
        @test ax.nf == ["market", "size", "industry=Real Estate", "industry=Software",
                        "industry=Banks"]
        sets = cross_sectional_factor_sets(pe, rd)
        ref = PO.cross_sectional_factor_sets(factors, rd)
        @test sets.dict == ref.dict
        @test sets.cfkey == ref.cfkey
        @test sets.dict[sets.cfkey] == ax.nf
        # The `sets` argument is forwarded, so the estimator method widens too.
        s0 = UniverseSets(; xkey = "assets",
                          dict = Dict{String, Any}("assets" => rd.nx, "sector" => ["a"]))
        s1 = cross_sectional_factor_sets(pe, rd, s0)
        @test s1.dict == PO.cross_sectional_factor_sets(factors, rd, s0).dict
        @test s1.xkey == "assets"
        @test s1.dict["sector"] == ["a"]
        @test s1.dict[s1.cfkey] == ax.nf
    end

    @testset "The docstrings of 09_CrossSectionalFactorAxis against numbers" begin
        # `exposure_axis_names`: one name per level for a one-hot member, and the caller's
        # name for every other member. The one-hot method ignores the caller's name.
        oh = last(factors[3])
        lv = PO.one_hot_exposure_names(oh, rd)
        @test PO.exposure_axis_names("mkt", ConstantExposure(), rd) == (["mkt"], ["market"])
        @test PO.exposure_axis_names("size", last(factors[2]), rd) == (["size"], ["style"])
        @test PO.exposure_axis_names("ignored", oh, rd) ==
              (lv, fill("industry", length(lv)))
        # `cross_sectional_factor_axis`: one name per member that is not one-hot, and one
        # per level, so 1 + 1 + 3 names here.
        ax = PO.cross_sectional_factor_axis(factors, rd)
        @test length(ax.nf) == 2 + length(lv) == 5
        # The levels are the ones the Panel Field declares, so a fold over fewer
        # observations and fewer assets reads the same axis.
        @test PO.cross_sectional_factor_axis(factors, PO.port_opt_view(rd, 1:10, :)) == ax
        @test PO.cross_sectional_factor_axis(factors, PO.port_opt_view(rd, 11:30, 1:4)) ==
              ax
        # `cross_sectional_factor_sets`: a new sets takes the default key prefixes of
        # `UniverseSets`, and it equals the widening of the bare default universe.
        dflt = UniverseSets(; dict = Dict{String, Any}("nx" => rd.nx))
        ns = PO.cross_sectional_factor_sets(factors, rd)
        for k in (:xkey, :uxkey, :tfkey, :utfkey, :cfkey, :ucfkey, :nikey)
            @test getfield(ns, k) == getfield(dflt, k)
        end
        @test ns.dict == PO.cross_sectional_factor_sets(factors, rd, dflt).dict
        @test sort!(collect(keys(ns.dict))) == ["industry", "market", "ncf", "nx", "style"]
        # A widened universe keeps all seven of its key prefixes.
        own = UniverseSets(; xkey = "assets", uxkey = "uassets", tfkey = "tsf",
                           utfkey = "utsf", cfkey = "csf", ucfkey = "ucsf", nikey = "gone",
                           dict = Dict{String, Any}("assets" => rd.nx))
        wid = PO.cross_sectional_factor_sets(factors, rd, own)
        for k in (:xkey, :uxkey, :tfkey, :utfkey, :cfkey, :ucfkey, :nikey)
            @test getfield(wid, k) == getfield(own, k)
        end
        @test wid.dict["csf"] == ax.nf
        @test !haskey(wid.dict, "ncf")
        # A Factor Family label that starts with a key prefix of the universe, or equals
        # its `nikey`, would be read as an axis, so it is refused. Before the guard,
        # UniverseSets took `"ni"` as the Non-Investable Axis, and a family `"ncfx"` that
        # held every factor as a partition of the cross-sectional axis.
        fam_of(lab) = ["market" => ConstantExposure(),
                       "a" => ConstantExposure(; family = lab)]
        for lab in ("nxt", "uxx", "nfam", "uf", "ncfx", "ucfam", "ni")
            @test_throws ArgumentError PO.cross_sectional_factor_sets(fam_of(lab), rd)
        end
        @test_throws ArgumentError PO.cross_sectional_factor_sets(["a" =>
                                                                       ConstantExposure(;
                                                                                        family = "ncfx"),
                                                                   "b" =>
                                                                       ConstantExposure(;
                                                                                        family = "ncfx")],
                                                                  rd)
        msg = try
            PO.cross_sectional_factor_sets(fam_of("ni"), rd)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("Factor Family ni", msg)
        # A label that only starts with `nikey` is a plain group.
        @test PO.cross_sectional_factor_sets(fam_of("nikkei"), rd).dict["nikkei"] == ["a"]
        # The prefixes are the universe's own: a label the default refuses is a plain group
        # of a universe with other prefixes, and a label that starts with one of those is
        # refused.
        @test PO.cross_sectional_factor_sets(fam_of("ncfx"), rd, own).dict["ncfx"] == ["a"]
        @test_throws ArgumentError PO.cross_sectional_factor_sets(fam_of("csfx"), rd, own)
        @test_throws ArgumentError PO.cross_sectional_factor_sets(fam_of("gone"), rd, own)
        # `cross_sectional_sets_write!`: a new key and the same list are written, and a
        # different list is refused with its key named.
        d = Dict{String, Any}("style" => ["a"])
        @test isnothing(PO.cross_sectional_sets_write!(d, "style", ["a"]))
        @test isnothing(PO.cross_sectional_sets_write!(d, "industry", ["b", "c"]))
        @test d == Dict{String, Any}("style" => ["a"], "industry" => ["b", "c"])
        msg = try
            PO.cross_sectional_sets_write!(d, "style", ["b"])
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("under style", msg)
        @test d["style"] == ["a"]
        # `cross_sectional_sets_dict`: the method for `nothing` declares the asset axis
        # alone under `"nx"`, and the method for a universe copies its dictionary.
        @test PO.cross_sectional_sets_dict(rd, nothing) == Dict{String, Any}("nx" => rd.nx)
        cpd = PO.cross_sectional_sets_dict(rd, own)
        @test cpd == own.dict
        @test cpd !== own.dict
    end
end

#=
The rules of the two Calibration Slots of `CrossSectionalFactorPrior`: `PrecisionBlend` and
`SteinShrinkage` compute the Spanned Shrinkage `lambda`, and `ForecastCalibrationSlope` computes
the Orthogonal Forecast Scale `c`.

Each rule is pinned against a hand-coded copy of the rule that the simulation of the map measured.
The copy reads the fitted block of the prior and nothing of the library's own rule code: it splits
each forecast row by its own weighted least squares, and it takes the sample mean and the sample
covariance of the factor returns, which is what `EmpiricalPrior()` states. The measured agreement
is about 1e-16, so the pins hold at `rtol = 1e-12`.

The default of `lambda` is `PrecisionBlend()`. A test that pins a mean states `lambda = 1`.
=#
using Statistics, Distributions, Dates, Random

# The fixture of `test_12k`: a panel with no missing cell, two z-scored style fields and a
# `signal` field that no Factor Exposure reads, so a Return Forecast on it has an orthogonal part.
function cscr_case()
    PO = PortfolioOptimisers
    rd0 = synthetic_asset_panel(; n_assets = 40, n_observations = 120, n_industries = 3,
                                rng = StableRNG(725_900), late_listing_proba = 0.0,
                                delisting_proba = 0.0, missing_ratio = 0.0).rd
    function zscore(A)
        B = similar(A)
        for t in axes(A, 1)
            r = view(A, t, :)
            B[t, :] = (r .- Statistics.mean(r)) ./ Statistics.std(r; corrected = false)
        end
        return B
    end
    mcap = PO.panel_field_values(rd0, "market_cap")
    pf = Any[f for f in rd0.pnl.pf]
    push!(pf, NumericPanelField(; name = "style1", vals = zscore(log.(mcap))))
    push!(pf,
          NumericPanelField(; name = "style2",
                            vals = zscore(PO.panel_field_values(rd0, "book_equity") ./ mcap)))
    push!(pf,
          NumericPanelField(; name = "signal",
                            vals = zscore(randn(StableRNG(739_100), size(rd0.X)...))))
    rd = ReturnsResult(; nx = rd0.nx, X = rd0.X, ts = rd0.ts,
                       pnl = AssetPanel(; pf = identity.(pf), amsk = rd0.pnl.amsk,
                                        emsk = rd0.pnl.emsk))
    pass(field) = CompositeExposure(; descriptors = [Passthrough(; field = field)],
                                    outlier = nothing, scoring = nothing, family = "style")
    factors = ["market" => ConstantExposure(),
               "industry" => OneHotExposure(; field = "industry", family = "industry"),
               "style1" => pass("style1"), "style2" => pass("style2")]
    rfe = FixedWeightedReturnForecast(;
                                      scores = DescriptorScores(;
                                                                descriptors = [Passthrough(;
                                                                                           field = "signal")],
                                                                outlier = nothing,
                                                                scoring = nothing),
                                      scale = 0.02)
    return rd, factors, rfe
end

# The split of one forecast row by weighted least squares over its finite assets.
function cscr_split(alpha, L, w)
    ok = isfinite.(alpha) .& vec(all(isfinite, L; dims = 2))
    Lo = L[ok, :]
    Lw = Lo .* w[ok]
    g = (transpose(Lo) * Lw) \ (transpose(Lw) * alpha[ok])
    return g, alpha - L * g
end

# The split of every row of the forecast history, against the exposures and the weights of the row.
function cscr_history(rr)
    H = rr.rf.hist
    Tb, N = size(H)
    K = size(rr.csr.f, 2)
    G = fill(NaN, K, Tb)
    AP = fill(NaN, Tb, N)
    for t in 1:Tb
        if !(any(isfinite, view(H, t, :)))
            continue
        end
        g, ap = cscr_split(H[t, :], rr.Ms[t, :, 1:K], rr.rw[t, :])
        G[:, t] = g
        AP[t, :] = ap
    end
    return G, AP
end

# The `lambda` rules of the simulation: the precision blend over the history (`pb_raw`) and over
# the latest row (`pb_one`), and the Bayes-Stein and positive-part James-Stein weights.
function cscr_lambda(Fh, Ghist, gT)
    T, K = size(Fh)
    mu = vec(mean(Fh; dims = 1))
    S = cov(Fh)
    P = inv(Symmetric(S))
    m1 = K / T
    q = 0.0
    for t in axes(Ghist, 2)
        dt = Ghist[:, t] - mu
        q += dot(dt, P, dt)
    end
    m2p = max(q / size(Ghist, 2) - m1, 0.0)
    d = mu - gT
    BT = dot(d, P, d)
    ev = eigvals(Symmetric(S))
    ajs = (sum(ev) - 2 * maximum(ev)) / (T * dot(d, d))
    return (; pb_raw = m2p / (m1 + m2p), pb_one = max(0.0, 1 - K / (T * BT)),
            bayes_stein = 1 - (K + 2) / ((K + 2) + T * BT),
            james_stein = 1 - clamp(ajs, 0.0, 1.0))
end

function cscr_shrink(x, se, form, target)
    d = x - target
    if (iszero(d) || !isfinite(se))
        return target
    end
    w = form === :plugin ? d^2 / (d^2 + se^2) : max(zero(x), 1 - se^2 / d^2)
    return target + w * d
end

# The `c` rules of the simulation: the slope of the next row's residual on the orthogonal part,
# under the regression weights of the forecast row, with its HC0 standard error.
function cscr_c(AP, eps, rw)
    Tb, N = size(AP)
    num = 0.0
    den = 0.0
    n = 0
    for t in 1:(Tb - 1)
        hit = false
        for i in 1:N
            a, r, q = AP[t, i], eps[t + 1, i], rw[t, i]
            if !(isfinite(a) && isfinite(r))
                continue
            end
            hit = true
            q = isfinite(q) ? q : 0.0
            num += q * a * r
            den += q * a^2
        end
        n += hit
    end
    ch = den > 0 ? num / den : NaN
    meat = 0.0
    for t in 1:(Tb - 1), i in 1:N
        a, r, q = AP[t, i], eps[t + 1, i], rw[t, i]
        if !(isfinite(a) && isfinite(r))
            continue
        end
        q = isfinite(q) ? q : 0.0
        meat += (q * a * (r - ch * a))^2
    end
    se = sqrt(meat) / den
    chp = isfinite(ch) ? max(ch, 0.0) : 1.0
    return (; ch, se, n, chp, plugin = cscr_shrink(chp, se, :plugin, 1.0),
            pospart = cscr_shrink(chp, se, :pospart, 1.0))
end

@testset "Cross-sectional calibration rules: construction, defaults and the trait" begin
    PO = PortfolioOptimisers
    @test PrecisionBlend <: PO.AbstractSpannedShrinkageCalibrationAlgorithm
    @test SteinShrinkage <: PO.AbstractSpannedShrinkageCalibrationAlgorithm
    @test ForecastCalibrationSlope <: PO.AbstractOrthogonalForecastScaleCalibrationAlgorithm
    @test isa(PrecisionBlend(), PO.Num_SpanShrinkCal)
    @test !isa(ForecastCalibrationSlope(), PO.Num_SpanShrinkCal)
    @test isa(ForecastCalibrationSlope(), PO.Num_OrthFcScaleCal)
    @test isa(PrecisionBlend().err, CurrentForecastError)
    @test isa(SteinShrinkage().alg, BayesStein)
    @test isa(ForecastCalibrationSlope().wu, ThresholdWarmUp)
    @test ThresholdWarmUp().min_obs == 252
    @test isone(ForecastCalibrationSlope().target)
    @test_throws DomainError ThresholdWarmUp(; min_obs = 0)
    @test_throws DomainError ForecastCalibrationSlope(; target = -1)
    @test_throws DomainError ForecastCalibrationSlope(; target = Inf)

    # A rule reads the history exactly when the prior must make it.
    @test !PO.reads_forecast_history(PrecisionBlend())
    @test PO.reads_forecast_history(PrecisionBlend(; err = ForecastHistoryError()))
    @test !PO.reads_forecast_history(SteinShrinkage())
    @test PO.reads_forecast_history(ForecastCalibrationSlope())

    # The default of `lambda` is the rule, and the default of `c` is the number one.
    pe = CrossSectionalFactorPrior(; factors = ["market" => ConstantExposure()])
    @test isa(pe.lambda, PrecisionBlend)
    @test isone(pe.c)

    # A rule that reads the history needs a fitted forecast, and the constructor says so.
    f = ["market" => ConstantExposure()]
    @test_throws ArgumentError CrossSectionalFactorPrior(; factors = f,
                                                         c = ForecastCalibrationSlope())
    @test_throws ArgumentError CrossSectionalFactorPrior(; factors = f,
                                                         lambda = PrecisionBlend(;
                                                                                 err = ForecastHistoryError()))
    # A rule outside the prior has no fit to read.
    pr = PO.LowOrderPrior(; X = randn(StableRNG(1), 10, 2), mu = zeros(2),
                          sigma = Matrix(1.0I, 2, 2))
    @test_throws PO.IsNothingError PrecisionBlend()(:lambda, pr, nothing, nothing,
                                                    CalibrationContext())
    @test_throws PO.IsNothingError ForecastCalibrationSlope()(:c, pr, nothing, nothing,
                                                              CalibrationContext())
end

@testset "Cross-sectional calibration rules: the warm-up forms" begin
    PO = PortfolioOptimisers
    th = ThresholdWarmUp(; min_obs = 5)
    # Below the count the rule takes the target, at or above it the clipped slope.
    @test PO.forecast_scale_warm_up(th, 2.0, 0.1, 4, 1) == 1
    @test PO.forecast_scale_warm_up(th, 2.0, 0.1, 5, 1) == 2.0
    @test PO.forecast_scale_warm_up(th, -0.5, 0.1, 5, 1) == 0.0
    @test PO.forecast_scale_warm_up(th, NaN, 0.1, 50, 1) == 1
    for (wu, form) in ((PlugInWarmUp(), :plugin), (PositivePartWarmUp(), :pospart)),
        (c, se) in ((2.0, 1.0), (3.0, 1.0), (0.4, 0.3), (-0.5, 0.2), (1.2, 0.5))

        @test PO.forecast_scale_warm_up(wu, c, se, 1, 1.0) ≈
              cscr_shrink(max(c, 0.0), se, form, 1.0) rtol = 1e-15
    end
    # The two shrinkage forms take the target where the slope or its error is not finite, and
    # where the clipped slope is the target.
    for wu in (PlugInWarmUp(), PositivePartWarmUp())
        @test PO.forecast_scale_warm_up(wu, NaN, 1.0, 1, 1) == 1
        @test PO.forecast_scale_warm_up(wu, 2.0, Inf, 1, 1) == 1
        @test PO.forecast_scale_warm_up(wu, 1.0, 0.5, 1, 1) == 1
    end
    # Plug-in keeps part of every slope, and positive part drops a slope within one error.
    @test PO.forecast_scale_warm_up(PlugInWarmUp(), 2.0, 1.0, 1, 1.0) == 1.5
    @test PO.forecast_scale_warm_up(PositivePartWarmUp(), 1.5, 1.0, 1, 1.0) == 1.0
    @test PO.forecast_scale_warm_up(PositivePartWarmUp(), 3.0, 1.0, 1, 1.0) == 2.5
    # The HC0 error of the slope through the origin.
    a = [1.0, 2.0, 3.0]
    b = [1.0, 3.0, 2.0]
    q = [1.0, 0.5, 2.0]
    c = PO.forecast_calibration_slope(a, b, q)
    @test PO.forecast_calibration_slope_se(a, b, q, c) ≈
          sqrt(sum((q .* a .* (b .- c .* a)) .^ 2)) / sum(q .* a .^ 2) rtol = 1e-15
    @test isnan(PO.forecast_calibration_slope_se(zeros(2), ones(2), ones(2), NaN))
end

@testset "Cross-sectional calibration rules: each rule equals the rule of the simulation" begin
    PO = PortfolioOptimisers
    rd, factors, rfe = cscr_case()
    # Industries and styles without the market factor give a covariance of full rank, so the
    # formulas of the simulation hold as written, with `inv`.
    ffr = factors[2:end]
    fit(l, c) = prior(CrossSectionalFactorPrior(; factors = ffr, pe = EmpiricalPrior(),
                                                rfe = rfe, lambda = l, c = c), rd)
    p1 = fit(1, 1)
    rr = p1.rr
    @test rank(cov(rr.csr.f)) == size(rr.csr.f, 2)
    G, AP = cscr_history(rr)
    Tb = size(G, 2)
    gT = G[:, Tb]
    # The split of the last row is the split the prior blends.
    @test maximum(abs,
                  gT -
                  PO.cross_sectional_alpha_split(CrossSectionalLinearRegression(), rr.rf.mu,
                                                 rr.L, rr.rw[end, :]).g) < 1e-15
    lib = PO.cross_sectional_split_history((; hist = rr.rf.hist, csfm = rr,
                                            cre = CrossSectionalLinearRegression()))
    @test isapprox(lib.g, G; rtol = 1e-12)
    @test isapprox(lib.ap, AP; rtol = 1e-12)

    cols = [t for t in 1:(Tb - 1) if all(isfinite, G[:, t])]
    @test length(cols) == Tb - 1
    h = cscr_lambda(rr.csr.f, G[:, cols], gT)
    @test isapprox(fit(PrecisionBlend(), 1).rr.lambda, h.pb_one; rtol = 1e-12)
    @test isapprox(fit(PrecisionBlend(; err = ForecastHistoryError()), 1).rr.lambda,
                   h.pb_raw; rtol = 1e-12)
    @test isapprox(fit(SteinShrinkage(), 1).rr.lambda, h.bayes_stein; rtol = 1e-12)
    @test isapprox(fit(SteinShrinkage(; alg = JamesStein()), 1).rr.lambda, h.james_stein;
                   rtol = 1e-12)
    # The blend reads the resolved number: a fit that states it gives the same mean.
    pb = fit(PrecisionBlend(), 1)
    @test pb.mu == fit(pb.rr.lambda, 1).mu
    @test 0 < pb.rr.lambda < 1

    hc = cscr_c(AP, rr.csr.eps, rr.rw)
    @test hc.n == Tb - 1
    @test isapprox(fit(1, ForecastCalibrationSlope(; wu = ThresholdWarmUp(; min_obs = 10))).rr.c,
                   hc.chp; rtol = 1e-12)
    @test isapprox(fit(1, ForecastCalibrationSlope(; wu = PlugInWarmUp())).rr.c, hc.plugin;
                   rtol = 1e-12)
    @test isapprox(fit(1, ForecastCalibrationSlope(; wu = PositivePartWarmUp())).rr.c,
                   hc.pospart; rtol = 1e-12)
    # The warm-up of the default: 119 rows are below one year of daily rows, so `c` is one.
    @test Tb - 1 < 252
    @test fit(1, ForecastCalibrationSlope()).rr.c == 1
    @test fit(1, ForecastCalibrationSlope(; target = 0.5)).rr.c == 0.5
    pc = fit(1, ForecastCalibrationSlope(; wu = ThresholdWarmUp(; min_obs = 10)))
    # The block records the resolved scale, and `b` is that scale times the orthogonal part.
    @test isequal(pc.rr.b, pc.rr.c * rr.b)
end

@testset "Cross-sectional calibration rules: a singular factor covariance" begin
    PO = PortfolioOptimisers
    rd, factors, rfe = cscr_case()
    # The market exposure is the sum of the industry exposures, so the factor returns lie in a
    # subspace and their covariance is singular. The rules read its pseudo-inverse and its rank.
    p1 = prior(CrossSectionalFactorPrior(; factors = factors, pe = EmpiricalPrior(),
                                         rfe = rfe, lambda = 1), rd)
    rr = p1.rr
    F = rr.csr.f
    S = cov(F)
    K = rank(S)
    @test K == size(F, 2) - 1
    mu = vec(mean(F; dims = 1))
    g = PO.cross_sectional_alpha_split(CrossSectionalLinearRegression(), rr.rf.mu, rr.L,
                                       rr.rw[end, :]).g
    d = mu - g
    B = dot(d, pinv(S), d)
    T = size(F, 1)
    pr = prior(CrossSectionalFactorPrior(; factors = factors, pe = EmpiricalPrior(),
                                         rfe = rfe), rd)
    @test isapprox(pr.rr.lambda, max(0, 1 - K / (T * B)); rtol = 1e-10)
    bs = prior(CrossSectionalFactorPrior(; factors = factors, pe = EmpiricalPrior(),
                                         rfe = rfe, lambda = SteinShrinkage()), rd)
    @test isapprox(bs.rr.lambda, 1 - (K + 2) / ((K + 2) + T * B); rtol = 1e-10)
    # The zero eigenvalue of a singular factor covariance is round-off, and it fell on either
    # side of `m * eps` by the CI host (#1475). The threshold `sqrt(eps)` of the largest drops
    # an eigenvalue at 1e-12 of it and keeps one at 1e-7.
    cs3 = (; csfm = (; csr = (; f = zeros(1, 3))))
    for (v, k) in ((1e-12, 2), (1e-7, 3))
        p3 = LowOrderPrior(; X = ones(2, 3), mu = zeros(3), sigma = diagm([1.0, 0.5, v]))
        @test PO.spanned_shrinkage_moments(p3, cs3, zeros(3, 1)).K == k
    end

    # With no Return Forecast the default shrinks the factor mean towards zero, under the moments
    # of the default factor prior.
    q1 = prior(CrossSectionalFactorPrior(; factors = factors, lambda = 1), rd)
    Ke = size(q1.rr.csr.f, 2)
    m = q1.fpr.mu[1:Ke]
    Sd = q1.fpr.sigma[1:Ke, 1:Ke]
    Td = PO.effective_sample_size(q1.fpr, q1.fpr.w)
    Bd = dot(m, pinv(Sd), m)
    q0 = prior(CrossSectionalFactorPrior(; factors = factors), rd)
    @test isapprox(q0.rr.lambda, max(0, 1 - rank(Sd) / (Td * Bd)); rtol = 1e-10)
    @test q0.mu ==
          prior(CrossSectionalFactorPrior(; factors = factors, lambda = q0.rr.lambda), rd).mu
    # The Bodnar-Okhrin-Parolya weight divides by a gap that a zero target closes.
    @test_throws DomainError prior(CrossSectionalFactorPrior(; factors = factors,
                                                             lambda = SteinShrinkage(;
                                                                                     alg = BodnarOkhrinParolya())),
                                   rd)
end

@testset "Cross-sectional calibration rules: the history under a family re-basis and its edges" begin
    PO = PortfolioOptimisers
    rd, factors, rfe = cscr_case()
    fams = ["industry" => nothing]
    p1 = prior(CrossSectionalFactorPrior(; factors = factors, families = fams,
                                         pe = EmpiricalPrior(), rfe = rfe, lambda = 1), rd)
    rr = p1.rr
    @test !isnothing(rr.fcb)
    cs = (; hist = rr.rf.hist, csfm = rr, cre = CrossSectionalLinearRegression(),
          g = PO.cross_sectional_alpha_split(CrossSectionalLinearRegression(), rr.rf.mu,
                                             view(rr.L, :, 1:size(rr.csr.f, 2)),
                                             rr.rw[end, :]).g, ap = nothing)
    # The history splits each row on the reduced axis, so its last row is the latest split.
    sh = PO.cross_sectional_split_history(cs)
    @test size(sh.g, 1) == size(rr.csr.f, 2)
    @test isapprox(sh.g[:, end], cs.g; rtol = 1e-12)
    @test isapprox(sh.ap[end, :], rr.rf.mu - view(rr.L, :, 1:size(rr.csr.f, 2)) * cs.g;
                   rtol = 1e-12)
    ph = prior(CrossSectionalFactorPrior(; factors = factors, families = fams,
                                         pe = EmpiricalPrior(), rfe = rfe,
                                         lambda = PrecisionBlend(;
                                                                 err = ForecastHistoryError())),
               rd)
    @test 0 <= ph.rr.lambda <= 1

    # A history with no row before the latest gives the reading of the latest row alone.
    H = copy(rr.rf.hist)
    H[1:(end - 1), :] .= NaN
    cs1 = merge(cs, (; hist = H))
    @test PO.spanned_forecast_sample(ForecastHistoryError(), cs1) ==
          PO.spanned_forecast_sample(CurrentForecastError(), cs1)
    @test all(isnan, PO.cross_sectional_split_history(cs1).ap[1:(end - 1), :])
    # A context with no history is refused by the rule that reads it.
    @test_throws PO.IsNothingError PO.cross_sectional_split_history(merge(cs,
                                                                          (;
                                                                           hist = nothing)))

    # James-Stein divides by `|mu - g|^2`: where `g` is the mean, every weight is the same blend.
    W = Matrix(1.0I, 3, 3)
    ev = ones(3)
    @test PO.stein_spanned_shrinkage(JamesStein(), [0.1, 0.2, 0.3], [0.1, 0.2, 0.3], W, ev,
                                     50) == 1
    # Bodnar-Okhrin-Parolya needs more observations than the rank, and a target that is not a
    # multiple of the mean.
    @test_throws DomainError PO.stein_spanned_shrinkage(BodnarOkhrinParolya(),
                                                        [0.1, 0.2, 0.3], [0.3, 0.1, 0.2], W,
                                                        ev, 3)
    @test_throws DomainError PO.stein_spanned_shrinkage(BodnarOkhrinParolya(),
                                                        [0.1, 0.2, 0.3], zeros(3), W, ev,
                                                        50)
end

@testset "Cross-sectional calibration rules: each Stein intensity equals the library's" begin
    #=
    `ShrunkExpectedReturns` shrinks towards a target that is a multiple of the vector of ones.
    `SteinShrinkage` shrinks towards `g`, so the check sets `g` to the grand mean, where the two
    targets are one vector, and compares the blend.
    =#
    PO = PortfolioOptimisers
    rng = StableRNG(1483)
    for K in (3, 20)
        T = 60
        X = 0.01 .* randn(rng, T, K) .+ 0.001 .* transpose(randn(rng, K))
        mu = vec(mean(X; dims = 1))
        S = cov(X)
        g = fill(mean(mu), K)
        E = eigen(Symmetric(S))
        W = E.vectors ./ transpose(sqrt.(E.values))
        for alg in (BayesStein(), JamesStein())
            lam = PO.stein_spanned_shrinkage(alg, mu, g, W, E.values, T)
            libmu = vec(mean(ShrunkExpectedReturns(; alg = alg), X))
            @test isapprox(lam * mu + (1 - lam) * g, libmu; rtol = 1e-12)
        end
        # The Bodnar-Okhrin-Parolya estimate is `alpha mu + beta g`, and the rule returns the
        # clipped `alpha`.
        libmu = vec(mean(ShrunkExpectedReturns(; alg = BodnarOkhrinParolya()), X))
        ab = hcat(mu, g) \ libmu
        lam = PO.stein_spanned_shrinkage(BodnarOkhrinParolya(), mu, g, W, E.values, T)
        @test isapprox(lam, clamp(ab[1], 0, 1); rtol = 1e-10, atol = 1e-14)
    end
end
