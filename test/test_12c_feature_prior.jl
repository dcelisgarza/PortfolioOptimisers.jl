include(joinpath(@__DIR__, "test12_setup.jl"))
using Clustering, StableRNGs, LinearAlgebra

# Every field of a `LowOrderPrior` except the two feature fields. A `FeaturePrior` is a
# provably pure addition only if all of these come back identical.
# A producer emitting a time-varying `Z`. Nothing in the library ships one, but the derived
# carrier accepts the shape and a producer is handed the fold's own `X`, so it tracks the fold
# with no extra plumbing. Defined here to pin that claim rather than assert it.
struct TrailingDispersionFeatures <: PortfolioOptimisers.AbstractFeatureMatrixEstimator
    windows::Vector{Int}
end
function PortfolioOptimisers.feature_matrix(ze::TrailingDispersionFeatures,
                                            ::PortfolioOptimisers.AbstractPriorResult,
                                            X::PortfolioOptimisers.MatNum, args...;
                                            kwargs...)
    T, N = size(X)
    Z = zeros(T, N, length(ze.windows))
    for (k, w) in pairs(ze.windows), t in 1:T, i in 1:N
        Z[t, i, k] = std(view(X, max(1, t - w + 1):t, i))
    end
    Z[1, :, :] .= Z[2, :, :]
    return Z
end
const MOMENT_FIELDS = (:X, :mu, :sigma, :chol, :w, :ens, :kld, :ow, :rr, :fpr)
# Result structs are immutable, so `==` falls back to `===`, which compares the arrays they
# hold by identity — two separately computed `Regression`s never match. Recurse instead.
function eq_moment(a, b)
    if isnothing(a) || isnothing(b)
        return isnothing(a) && isnothing(b)
    end
    if isa(a, PortfolioOptimisers.AbstractResult)
        return all(f -> eq_moment(getproperty(a, f), getproperty(b, f)), propertynames(a))
    end
    return a == b
end
function same_moments(a, b)
    return all(f -> eq_moment(getproperty(a, f), getproperty(b, f)), MOMENT_FIELDS)
end

@testset "FeaturePrior is a pure addition" begin
    na = size(rd.X, 2)
    rng = StableRNG(987654321)
    Zlit = rand(rng, na, 4)
    for pe in (EmpiricalPrior(), EmpiricalPrior(; horizon = 252),
               BlackLittermanPrior(; sets = sets,
                                   views = LinearConstraintEstimator(; val = ["$(rd.nx[1]) == 0.03"])))
        base = prior(pe, rd)
        wrapped = prior(FeaturePrior(; pe = pe, ze = Zlit), rd)
        @test same_moments(base, wrapped)
        @test panel_feature_matrix(wrapped.pnl)[2] == Zlit
        # The wrapped estimator itself never grows a feature matrix.
        @test isnothing(panel_feature_matrix(base.pnl)[2])
    end
end

@testset "RegressionFeatures reads the loadings" begin
    fp = prior(FactorPrior(), rd)
    pr = prior(FeaturePrior(; pe = FactorPrior(), ze = RegressionFeatures()), rd)
    @test same_moments(fp, pr)
    # `rr.L` falls back to `rr.M` when unset, so the producer needs no branch.
    @test panel_feature_matrix(pr.pnl)[2] == fp.rr.L
    @test size(panel_feature_matrix(pr.pnl)[2], 1) == size(rd.X, 2)

    # `L` proper, not the reconstructed `M`: a dimension-reduction regression sets both, and
    # they are different matrices.
    fpd = prior(FactorPrior(; re = DimensionReductionRegression()), rd)
    prd = prior(FeaturePrior(; pe = FactorPrior(; re = DimensionReductionRegression()),
                             ze = RegressionFeatures()), rd)
    @test panel_feature_matrix(prd.pnl)[2] == fpd.rr.L
    @test size(panel_feature_matrix(prd.pnl)[2], 2) != size(fpd.rr.M, 2) ||
          panel_feature_matrix(prd.pnl)[2] != fpd.rr.M

    # A prior that carries no regression cannot produce loadings, and says so.
    @test_throws PortfolioOptimisers.IsNothingError prior(FeaturePrior(;
                                                                       pe = EmpiricalPrior(),
                                                                       ze = RegressionFeatures()),
                                                          rd)
    # `BlackLittermanPrior` forwards `rr` and the factor block `fpr` (ADR 0046), so it no
    # longer has to be wrapped from the inside: the loadings producer reads through it, and
    # both nesting orders produce the same feature matrix.
    blv = LinearConstraintEstimator(; val = ["$(rd.nx[1]) == 0.03"])
    blout = prior(FeaturePrior(;
                               pe = BlackLittermanPrior(; pe = FactorPrior(), sets = sets,
                                                        views = blv),
                               ze = RegressionFeatures()), rd)
    blin = prior(BlackLittermanPrior(;
                                     pe = FeaturePrior(; pe = FactorPrior(),
                                                       ze = RegressionFeatures()),
                                     sets = sets, views = blv), rd)
    @test panel_feature_matrix(blout.pnl)[2] == fp.rr.L
    @test panel_feature_matrix(blout.pnl)[2] == panel_feature_matrix(blin.pnl)[2]
end

@testset "Nesting order does not matter" begin
    na = size(rd.X, 2)
    rng = StableRNG(123456789)
    Zlit = rand(rng, na, 5)
    views = LinearConstraintEstimator(; val = ["$(rd.nx[1]) == 0.03"])

    outer = prior(FeaturePrior(; pe = BlackLittermanPrior(; sets = sets, views = views),
                               ze = Zlit), rd)
    inner = prior(BlackLittermanPrior(;
                                      pe = FeaturePrior(;
                                                        pe = EmpiricalPrior(;
                                                                            me = EquilibriumExpectedReturns()),
                                                        ze = Zlit), sets = sets,
                                      views = views), rd)
    @test panel_feature_matrix(outer.pnl)[2] == panel_feature_matrix(inner.pnl)[2] == Zlit
    @test same_moments(outer, inner)

    # The outermost declaration wins rather than merging.
    Zother = rand(rng, na, 2)
    nested = prior(FeaturePrior(; pe = FeaturePrior(; pe = EmpiricalPrior(), ze = Zlit),
                                ze = Zother), rd)
    @test panel_feature_matrix(nested.pnl)[2] == Zother
end

@testset "Every wrapping prior forwards Z" begin
    na = size(rd.X, 2)
    rng = StableRNG(246813579)
    Zlit = rand(rng, na, 3)
    fpe = FeaturePrior(; pe = EmpiricalPrior(), ze = Zlit)
    views = LinearConstraintEstimator(; val = ["$(rd.nx[1]) == 0.03"])
    f_views = LinearConstraintEstimator(; val = ["$(rd.nf[1]) == 0.01"])

    # Asset-space wrappers forward.
    @test panel_feature_matrix(prior(BlackLittermanPrior(; pe = fpe, sets = sets,
                                                         views = views), rd).pnl)[2] == Zlit
    @test panel_feature_matrix(prior(BayesianBlackLittermanPrior(;
                                                                 pe = FeaturePrior(;
                                                                                   pe = FactorPrior(),
                                                                                   ze = Zlit),
                                                                 sets = xfsets,
                                                                 views = f_views), rd).pnl)[2] ==
          Zlit
    @test panel_feature_matrix(prior(AugmentedBlackLittermanPrior(; a_pe = fpe,
                                                                  sets = xfsets,
                                                                  a_views = views,
                                                                  f_views = f_views), rd).pnl)[2] ==
          Zlit
    @test panel_feature_matrix(prior(EntropyPoolingPrior(; pe = fpe, sets = sets,
                                                         mu_views = LinearConstraintEstimator(;
                                                                                              val = ["$(rd.nx[1]) == 0.01"])),
                                     rd).pnl)[2] == Zlit
    @test panel_feature_matrix(prior(EntropyPoolingPrior(; pe = fpe, sets = sets,
                                                         alg = H1_EntropyPooling(),
                                                         mu_views = LinearConstraintEstimator(;
                                                                                              val = ["$(rd.nx[1]) == 0.01"])),
                                     rd).pnl)[2] == Zlit
    @test panel_feature_matrix(prior(OpinionPoolingPrior(;
                                                         pes = [EntropyPoolingPrior(;
                                                                                    pe = fpe,
                                                                                    sets = sets,
                                                                                    mu_views = LinearConstraintEstimator(;
                                                                                                                         val = ["$(rd.nx[1]) == 0.01"]))],
                                                         pe2 = fpe), rd).pnl)[2] == Zlit

    # Factor-space wrappers drop it: their wrapped prior is fit on the factors, so its
    # feature matrix would not describe the asset axis.
    f_na = size(rd.F, 2)
    f_fpe = FeaturePrior(; pe = EmpiricalPrior(), ze = rand(rng, f_na, 3))
    @test isnothing(panel_feature_matrix(prior(FactorPrior(; pe = f_fpe), rd).pnl)[2])
    @test isnothing(panel_feature_matrix(prior(FactorBlackLittermanPrior(; pe = f_fpe,
                                                                         sets = xfsets,
                                                                         views = f_views),
                                               rd).pnl)[2])

    # `HighOrderPrior` needs no edits at all — it forwards any property of its child.
    hop = prior(HighOrderPriorEstimator(; pe = fpe), rd)
    @test panel_feature_matrix(hop.pnl)[2] == Zlit
end

@testset "port_opt_view slices the feature matrix" begin
    na = size(rd.X, 2)
    nobs = size(rd.X, 1)
    rng = StableRNG(135792468)
    i = [1, 4, 7, 11]
    base = prior(EmpiricalPrior(), rd)

    # The asset axis is sliced and the feature axis never is, whatever the shape -- a square
    # derived `Z` included. The prior carrier has no squareness vocabulary: every producer
    # that builds a square feature matrix refits on the subproblem's own universe, so there
    # is no full-universe square matrix here to cut down.
    for (Z, expected) in ((rand(rng, na, 4), (idx, Z) -> Z[idx, :]),
                          (rand(rng, na, na), (idx, Z) -> Z[idx, :]),
                          (rand(rng, nobs, na, 4), (idx, Z) -> Z[:, idx, :]),
                          (rand(rng, nobs, na, na), (idx, Z) -> Z[:, idx, :]))
        pr = PortfolioOptimisers.LowOrderPrior(; X = base.X, mu = base.mu,
                                               sigma = base.sigma,
                                               pnl = feature_matrix_panel(["_z$(k)"
                                                                           for k in
                                                                               1:size(Z,
                                                                                      ndims(Z))],
                                                                          Z))
        v = PortfolioOptimisers.port_opt_view(pr, i)
        @test panel_feature_matrix(v.pnl)[2] == expected(i, Z)
        # Observations are taken whole: folds slice them before the prior is fit.
        @test size(panel_feature_matrix(v.pnl)[2], 1) == (ndims(Z) == 3 ? nobs : length(i))
        # Repeated views compose.
        v2 = PortfolioOptimisers.port_opt_view(v, [1, 3])
        @test panel_feature_matrix(v2.pnl)[2] == expected([1, 3], expected(i, Z))
    end

    # The estimator-side view: a producer is configuration and passes through, a literal
    # feature matrix is data and is sliced.
    Zlit = rand(rng, na, 4)
    pv = PortfolioOptimisers.port_opt_view(FeaturePrior(; pe = EmpiricalPrior(), ze = Zlit,
                                                        sets = sets), i)
    @test pv.ze == Zlit[i, :]
    @test pv.sets.dict[pv.sets.xkey] == rd.nx[i]
    @test PortfolioOptimisers.port_opt_view(FeaturePrior(; pe = FactorPrior(),
                                                         ze = RegressionFeatures()), i).ze ===
          RegressionFeatures()

    # And the whole chain: a viewed estimator produces a feature matrix over the subset.
    pr = prior(PortfolioOptimisers.port_opt_view(FeaturePrior(; pe = EmpiricalPrior(),
                                                              ze = Zlit), i),
               view(rd.X, :, i))
    @test panel_feature_matrix(pr.pnl)[2] == Zlit[i, :]

    # `HighOrderPrior`'s view recurses into its child.
    hv = PortfolioOptimisers.port_opt_view(prior(HighOrderPriorEstimator(;
                                                                         pe = FeaturePrior(;
                                                                                           pe = EmpiricalPrior(),
                                                                                           ze = Zlit)),
                                                 rd), i)
    @test panel_feature_matrix(hv.pnl)[2] == Zlit[i, :]
end

@testset "The prior carrier validates its feature matrix" begin
    na = size(rd.X, 2)
    nobs = size(rd.X, 1)
    rng = StableRNG(864213579)
    base = prior(EmpiricalPrior(), rd)
    lop(; kwargs...) = PortfolioOptimisers.LowOrderPrior(; X = base.X, mu = base.mu,
                                                         sigma = base.sigma, kwargs...)
    # A produced feature matrix is nameless, so the panel it enters the carrier as names its
    # columns positionally.
    lopz(Z) = lop(; pnl = feature_matrix_panel(["_z$(k)" for k in 1:size(Z, ndims(Z))], Z))

    # The carrier has no squareness flag, and the break is loud rather than silently
    # absorbed: there is no `kwargs...` on the keyword constructor to swallow it.
    @test_throws MethodError lop(; pnl = feature_matrix_panel(["_z1"], rand(rng, na, 1)),
                                 z_sq = true)
    # A square derived `Z` is an ordinary matrix here -- nothing to declare, nothing to check.
    @test size(panel_feature_matrix(lopz(rand(rng, na, na)).pnl)[2]) == (na, na)
    @test size(panel_feature_matrix(lopz(rand(rng, nobs, na, na)).pnl)[2]) == (nobs, na, na)

    # Assets-major, bound to `X`.
    @test_throws DimensionMismatch lopz(rand(rng, 4, na))
    @test_throws DimensionMismatch lopz(rand(rng, na + 1, 4))
    @test_throws DimensionMismatch lopz(rand(rng, nobs - 1, na, 4))
    @test_throws DimensionMismatch lopz(rand(rng, nobs, na + 1, 4))

    # Never imputed: a non-finite entry is rejected rather than mapped to a plausible,
    # wrong distance.
    Zbad = rand(rng, na, 4)
    Zbad[2, 3] = NaN
    @test_throws PortfolioOptimisers.IsNonFiniteError lopz(Zbad)
    Zinf = rand(rng, na, 4)
    Zinf[1, 1] = Inf
    @test_throws PortfolioOptimisers.IsNonFiniteError lopz(Zinf)
    @test_throws PortfolioOptimisers.IsEmptyError lopz(Matrix{Float64}(undef, na, 0))

    # No panel is the default, and it is not an error.
    @test isnothing(panel_feature_matrix(lop().pnl)[2])

    # A literal `ze` is checked for emptiness at construction.
    @test_throws PortfolioOptimisers.IsEmptyError FeaturePrior(; pe = EmpiricalPrior(),
                                                               ze = Matrix{Float64}(undef,
                                                                                    na, 0))
    # `sets`, when given, must match the asset axis.
    @test_throws DimensionMismatch prior(FeaturePrior(; pe = EmpiricalPrior(),
                                                      ze = rand(rng, na, 3), sets = fsets),
                                         rd)
end

@testset "A derived feature matrix reaches the distance kernel" begin
    pr = prior(FeaturePrior(; pe = FactorPrior(), ze = RegressionFeatures()), rd)
    de = FeatureDistance()

    D = distance(de, panel_feature_matrix(pr.pnl)[2])
    S, D2 = cor_and_dist(de, panel_feature_matrix(pr.pnl)[2])
    @test D == D2
    @test size(D) == (size(rd.X, 2), size(rd.X, 2))
    @test issymmetric(D)
    @test all(iszero, diag(D))
    @test all(isfinite, D)
    @test all(isfinite, S)

    # The whole point: loadings cluster the universe differently from returns correlations.
    Dr = distance(Distance(; alg = CanonicalDistance()), PortfolioOptimisersCovariance(),
                  rd.X)
    hf = Clustering.hclust(D; linkage = :ward)
    hc = Clustering.hclust(Dr; linkage = :ward)
    @test hf.merges != hc.merges
    @test Clustering.cutree(hf; k = 3) != Clustering.cutree(hc; k = 3)

    # A cluster subset of the prior yields the distance over exactly that subset.
    i = [1, 4, 7, 11]
    @test distance(de,
                   panel_feature_matrix(PortfolioOptimisers.port_opt_view(pr, i).pnl)[2]) ==
          D[i, i]
end

@testset "A time-varying literal cannot follow an observation fold, and says so" begin
    rng = StableRNG(987654321)
    na = length(rd.nx)
    nobs = size(rd.X, 1)
    Zlit = rand(rng, nobs, na, 2)

    # It is only a *changed* observation count that fails. Construction, an asset view and a
    # fit on the full sample all succeed, which is why cross-validation is where it surfaces.
    pe = FeaturePrior(; pe = EmpiricalPrior(), ze = Zlit)
    @test size(panel_feature_matrix(prior(pe, rd).pnl)[2]) == (nobs, na, 2)
    @test size(PortfolioOptimisers.port_opt_view(pe, [1, 2, 3]).ze) == (nobs, 3, 2)

    # The estimator-side view leaves the observation axis alone — that is what creates the
    # mismatch, since the returns lose every row outside the fold.
    err = try
        prior(pe, rd.X[1:(nobs - 10), :])
        nothing
    catch e
        e
    end
    @test isa(err, DimensionMismatch)
    msg = sprint(showerror, err)
    # The message must name the carrier and all three remedies, producer first: the shared
    # `LowOrderPrior` message can only name the axis convention.
    @test occursin("FeaturePrior.ze", msg)
    @test occursin("producer", msg)
    @test occursin("z_src = :data", msg)
    @test occursin("static", msg)

    # A producer has no such problem: it is handed the fold's own `X`, so a time-varying `Z`
    # tracks the fold. The derived carrier supports the shape; nothing shipped emits it.
    pe_prod = FeaturePrior(; pe = EmpiricalPrior(),
                           ze = TrailingDispersionFeatures([5, 21]))
    @test size(panel_feature_matrix(prior(pe_prod, rd).pnl)[2]) == (nobs, na, 2)
    @test size(panel_feature_matrix(prior(pe_prod, rd.X[1:(nobs - 10), :]).pnl)[2]) ==
          (nobs - 10, na, 2)

    # And the same matrix on the data carrier is sliced with the returns rather than refused.
    rdz = ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts,
                        pnl = feature_matrix_panel(["a", "b"], Zlit))
    v = PortfolioOptimisers.port_opt_view(rdz, 1:(nobs - 10), [1, 2, 3])
    @test size(panel_feature_matrix(v.pnl)[2]) == (nobs - 10, 3, 2)
end

@testset "Both standalone priors forward the residual config of the estimator they wrap" begin
    # `FeaturePrior` adds only `Z`, so the residual block is the wrapped `pe`'s.
    frc = PortfolioOptimisers.factor_residual_config
    @test frc(FeaturePrior(; pe = FactorPrior(), ze = RegressionFeatures())) ==
          frc(FactorPrior())
    @test isnothing(frc(FeaturePrior(; pe = EmpiricalPrior(), ze = RegressionFeatures())))

    # `OpinionPoolingPrior`'s opinions contribute observation weights alone, so every moment
    # of the result -- and hence the residual block -- comes from the refit `pe2`.
    ep = EntropyPoolingPrior()
    @test frc(OpinionPoolingPrior(; pes = [ep], pe2 = FactorPrior())) == frc(FactorPrior())
    @test isnothing(frc(OpinionPoolingPrior(; pes = [ep])))
end
