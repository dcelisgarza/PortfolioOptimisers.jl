#=
Unit tests for `forward_prior` and the `reconstruct_prior` it is built on
(ADR 0046). These pin the composition rule's mechanics — forwarding is the
default, drops are explicit, the three field bindings are enforced, and the
carrier's own validation runs on every reconstruction — without going through any
estimator. What the construction sites actually forward is pinned separately, in
`test_12f_forward_prior.jl`.
=#
# Imported explicitly rather than inherited from whichever suite file ran first, so this
# file also runs standalone.
using Test, PortfolioOptimisers, StableRNGs, Logging, LinearAlgebra, StatsBase, Random

const PO = PortfolioOptimisers

# Small synthetic priors, built directly rather than fitted: these tests are about
# what survives a forward, not about any estimator's numbers.
function plain_prior()
    return LowOrderPrior(; X = [0.01 0.02; 0.03 0.04], mu = [0.02, 0.03],
                         sigma = [0.0004 0.0002; 0.0002 0.0003])
end
function chol_prior()
    return LowOrderPrior(; X = [0.01 0.02; 0.03 0.04], mu = [0.02, 0.03],
                         sigma = [0.0004 0.0002; 0.0002 0.0003],
                         chol = [0.02 0.01; 0.0 0.01415])
end
function weighted_prior()
    return LowOrderPrior(; X = [0.01 0.02; 0.03 0.04], mu = [0.02, 0.03],
                         sigma = [0.0004 0.0002; 0.0002 0.0003],
                         w = StatsBase.pweights([0.4, 0.6]), ens = 1.92, kld = 0.11,
                         ow = [0.25, 0.75])
end
function factor_prior()
    rng = StableRNG(987654321)
    X = randn(rng, 64, 4) / 100
    F = randn(rng, 64, 2) / 100
    # Random data has no significant factor, so the stepwise search warns; the
    # loadings it settles on are irrelevant here, only that `rr` is populated.
    return Logging.with_logger(Logging.NullLogger()) do
        return prior(FactorPrior(), X, F)
    end
end

@testset "forward_prior: a pure forward is the identity" begin
    pr = weighted_prior()
    # Forwarding is the default that costs nothing: no overrides means no work and
    # no reconstruction at all.
    @test PO.forward_prior(pr) === pr
    @test PO.forward_prior(pr;) === pr
end

@testset "forward_prior: unnamed fields are forwarded untouched" begin
    pr = weighted_prior()
    fpr = PO.forward_prior(pr; mu = [0.05, 0.06])
    @test fpr isa LowOrderPrior
    @test fpr.mu == [0.05, 0.06]
    # Every other field is the very same object, not a copy or a recomputation.
    for sym in (:X, :o_X, :sigma, :chol, :w, :ens, :kld, :ow, :rr, :fpr)
        @test getfield(fpr, sym) === getfield(pr, sym)
    end
    # In particular, the three fields whose silent loss motivated ADR 0046.
    @test fpr.w === pr.w
    @test fpr.ens === pr.ens
    @test fpr.kld === pr.kld
end

@testset "forward_prior: a drop is spelled, and only fields can be named" begin
    pr = chol_prior()
    dropped = PO.forward_prior(pr; mu = [0.05, 0.06], chol = nothing)
    @test isnothing(dropped.chol)
    @test dropped.sigma === pr.sigma

    # A name that is not a field of the carrier is refused, with the field list in
    # the message.
    @test_throws ArgumentError PO.forward_prior(pr; bogus = 1)
    err = try
        PO.forward_prior(pr; bogus = 1)
    catch e
        e
    end
    @test occursin("bogus", sprint(showerror, err))
    @test occursin("LowOrderPrior", sprint(showerror, err))
end

@testset "forward_prior: sigma binds chol" begin
    pr = chol_prior()
    sigma2 = [0.0009 0.0001; 0.0001 0.0004]

    # Naming `sigma` without naming `chol` is refused: `chol` takes precedence over
    # `sigma` at every consumer, so the forward would silently keep the old factor.
    @test_throws PO.ConflictingArgumentError PO.forward_prior(pr; sigma = sigma2)
    err = try
        PO.forward_prior(pr; sigma = sigma2)
    catch e
        e
    end
    @test occursin("chol", sprint(showerror, err))
    @test occursin("sigma", sprint(showerror, err))

    # Both remedies are accepted: drop it, or supply a factor rebuilt from the new
    # covariance.
    @test isnothing(PO.forward_prior(pr; sigma = sigma2, chol = nothing).chol)
    chol2 = Matrix(LinearAlgebra.cholesky(sigma2).U)
    @test PO.forward_prior(pr; sigma = sigma2, chol = chol2).chol === chol2

    # The binding is inert when there is nothing stale to carry.
    bare = plain_prior()
    @test isnothing(bare.chol)
    @test PO.forward_prior(bare; sigma = sigma2).sigma === sigma2

    # Naming `chol` alone is always fine — dropping it never states anything false.
    @test isnothing(PO.forward_prior(pr; chol = nothing).chol)
end

@testset "forward_prior: w binds its diagnostics" begin
    pr = weighted_prior()
    w2 = StatsBase.pweights([0.5, 0.5])

    # Diagnostics follow their weights: `ens`, `kld` and `ow` describe the `w` they
    # were computed with, so a new `w` cannot travel with the old three.
    @test_throws PO.ConflictingArgumentError PO.forward_prior(pr; w = w2)
    err = try
        PO.forward_prior(pr; w = w2)
    catch e
        e
    end
    msg = sprint(showerror, err)
    @test occursin("ens", msg)
    @test occursin("kld", msg)
    @test occursin("ow", msg)

    # A partial patch still names whichever of the three is left behind.
    err = try
        PO.forward_prior(pr; w = w2, ens = 2.0)
    catch e
        e
    end
    msg = sprint(showerror, err)
    @test !occursin("ens", msg)
    @test occursin("kld", msg)
    @test occursin("ow", msg)

    # Naming all three resolves it, whether dropped or recomputed.
    fpr = PO.forward_prior(pr; w = w2, ens = nothing, kld = nothing, ow = nothing)
    @test fpr.w === w2
    @test isnothing(fpr.ens)
    @test isnothing(fpr.kld)
    @test isnothing(fpr.ow)
    fpr = PO.forward_prior(pr; w = w2, ens = 2.0, kld = 0.2, ow = [0.5, 0.5])
    @test fpr.ens == 2.0

    # Inert when the carrier has no diagnostics to go stale.
    bare = LowOrderPrior(; X = [0.01 0.02; 0.03 0.04], mu = [0.02, 0.03],
                         sigma = [0.0004 0.0002; 0.0002 0.0003],
                         w = StatsBase.pweights([0.4, 0.6]))
    @test PO.forward_prior(bare; w = w2).w === w2

    # Changing the diagnostics without changing `w` is not a violation.
    @test PO.forward_prior(pr; ens = nothing).w === pr.w
end

@testset "forward_prior: reconstruction runs the carrier's validation" begin
    pr = factor_prior()
    @test !isnothing(pr.rr)

    #=
    A fitted factor prior carries an original, so `rr` binds `o_X` on it and that binding
    is reported before the carrier is rebuilt. A carrier may hold `rr` *without* an
    original — the requirement runs one way only — and that is the shape that still pins
    the all-or-none factor block. Both are worth keeping: the first is the new rule, the
    second is the constructor's, and the one must not hide the other.
    =#
    bare = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, chol = pr.chol,
                         rr = pr.rr, fpr = pr.fpr)
    @test isnothing(bare.o_X)

    # The factor block is all-or-none, so dropping one half of it throws — the
    # point of routing through the carrier's own constructor.
    @test_throws ArgumentError PO.forward_prior(bare; rr = nothing)
    @test_throws ArgumentError PO.forward_prior(bare; fpr = nothing)
    @test_throws ArgumentError PO.forward_prior(pr; fpr = nothing)

    #=
    `rr` binds `o_X`. The block records the projection that produced the reconstruction, so
    a carrier cannot keep the original after dropping what explains it. Dropping the block
    therefore drops the original with it, and saying so is the caller's job.
    =#
    @test !isnothing(pr.o_X)
    for patch in ((; rr = nothing), (; rr = nothing, fpr = nothing))
        @test_throws PO.ConflictingArgumentError PO.forward_prior(pr; patch...)
        err = try
            PO.forward_prior(pr; patch...)
        catch e
            e
        end
        @test occursin("o_X", sprint(showerror, err))
        @test occursin("rr", sprint(showerror, err))
    end

    # Dropping the whole block at once is valid, and the original goes with it.
    dropped = PO.forward_prior(pr; rr = nothing, fpr = nothing, o_X = nothing)
    @test isnothing(dropped.rr)
    @test isnothing(dropped.fpr)
    @test isnothing(dropped.o_X)
    @test dropped.original_X === dropped.X
    # The binding is inert on a carrier with no original to orphan.
    @test isnothing(PO.forward_prior(bare; rr = nothing, fpr = nothing).rr)

    # The flat `f_*` names read the nested block but are not fields of the carrier, so
    # they cannot be patched — the same refusal `mu` gets on a `HighOrderPrior`.
    for sym in (:f_mu, :f_sigma, :f_w)
        err = try
            PO.forward_prior(pr; NamedTuple{(sym,)}((nothing,))...)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(string(sym), sprint(showerror, err))
        @test occursin("cannot be forwarded", sprint(showerror, err))
    end

    # The asset and factor blocks must describe the same observations — a check
    # nothing enforced while the factor moments were flat fields.
    short_fpr = LowOrderPrior(; X = pr.fpr.X[1:32, :], mu = pr.fpr.mu, sigma = pr.fpr.sigma)
    @test_throws DimensionMismatch PO.forward_prior(pr; fpr = short_fpr)

    # Dimension checks fire on the reconstructed carrier, not on the original.
    bare = plain_prior()
    @test_throws DimensionMismatch PO.forward_prior(bare; mu = [0.1, 0.2, 0.3])
    @test_throws DimensionMismatch PO.forward_prior(bare;
                                                    w = StatsBase.pweights([0.2, 0.3, 0.5]))
    @test_throws PortfolioOptimisers.IsEmptyError PO.forward_prior(bare; mu = Float64[],
                                                                   sigma = Matrix{Float64}(undef,
                                                                                           0,
                                                                                           0),
                                                                   X = Matrix{Float64}(undef,
                                                                                       0,
                                                                                       0))
    # A prior result carries no feature data at all, so `pnl` is not a field to patch.
    @test_throws ArgumentError PO.forward_prior(bare; pnl = nothing)
end

@testset "forward_prior: HighOrderPrior forwards through the same rule" begin
    rng = StableRNG(123456789)
    X = randn(rng, 64, 3) / 100
    hop = prior(HighOrderPriorEstimator(), X)
    @test hop isa HighOrderPrior

    # Pure forward is the identity here too.
    @test PO.forward_prior(hop) === hop

    # `kt`/`L2`/`S2` are all-or-none, so the drop is spelled in full.
    dropped = PO.forward_prior(hop; kt = nothing, D2 = nothing, L2 = nothing, S2 = nothing)
    @test isnothing(dropped.kt)
    @test dropped.sk === hop.sk

    # `mu` is a forwarded property of `HighOrderPrior`, not a field of it: setting it
    # has to go through the field that holds it.
    @test_throws ArgumentError PO.forward_prior(hop; mu = fill(0.01, 3))
    nested = PO.forward_prior(hop; pr = PO.forward_prior(hop.pr; mu = fill(0.01, 3)))
    @test nested.mu == fill(0.01, 3)
    @test nested.kt === hop.kt

    # The bindings are inert on a carrier that owns neither field.
    @test_throws ArgumentError PO.forward_prior(hop; sigma = hop.sigma)

    # `pr` binds `fpr` on a `HighOrderPrior`, but not through a `forward_prior` binding:
    # the carrier's own `fpr.pr === pr.fpr` invariant catches a patched `pr` that leaves
    # the nested factor block pointing at the old factor distribution.
    fhop = Logging.with_logger(Logging.NullLogger()) do
        rng = StableRNG(987654321)
        return prior(HighOrderFactorPriorEstimator(), randn(rng, 64, 4) / 100,
                     randn(rng, 64, 2) / 100)
    end
    @test fhop.fpr.pr === fhop.pr.fpr
    strayed = PO.forward_prior(fhop.fpr;
                               pr = PO.forward_prior(fhop.pr.fpr; mu = fill(0.01, 2)))
    @test_throws PO.ConflictingArgumentError PO.forward_prior(fhop; fpr = strayed)
    # Naming both keeps it consistent, and the invariant survives the round trip.
    inner = PO.forward_prior(fhop.pr; mu = fill(0.01, 4))
    patched = PO.forward_prior(fhop; pr = inner, fpr = fhop.fpr)
    @test patched.mu == fill(0.01, 4)
    @test patched.fpr.pr === patched.pr.fpr
end

@testset "reconstruct_prior: the field list is derived, not written out" begin
    pr = weighted_prior()

    # `prior_field_values` reads the carrier's own fields, in declaration order, and
    # nothing a `@forward_properties` block layers on top. This is what keeps
    # `forward_prior` from needing an edit when a carrier gains a field.
    vals = PO.prior_field_values(pr)
    @test keys(vals) === fieldnames(LowOrderPrior)
    @test all(sym -> vals[sym] === getfield(pr, sym), keys(vals))

    hop = prior(HighOrderPriorEstimator(), randn(StableRNG(1), 64, 3) / 100)
    @test keys(PO.prior_field_values(hop)) === fieldnames(HighOrderPrior)
    @test !(:mu in keys(PO.prior_field_values(hop)))
    @test :mu in propertynames(hop)

    # Reconstruction is keyword-based, so the patch order is irrelevant.
    a = PO.reconstruct_prior(pr, (; mu = [0.7, 0.8], ens = nothing))
    b = PO.reconstruct_prior(pr, (; ens = nothing, mu = [0.7, 0.8]))
    @test a.mu == b.mu == [0.7, 0.8]
    @test isnothing(a.ens) && isnothing(b.ens)
    @test a.X === b.X === pr.X

    # It runs the carrier's validation, but it is not the rule-bearing entry point:
    # only `forward_prior` enforces the three bindings.
    @test_throws DimensionMismatch PO.reconstruct_prior(pr, (; mu = [0.7, 0.8, 0.9]))
    @test PO.reconstruct_prior(pr, (; w = StatsBase.pweights([0.5, 0.5]))).ens === pr.ens
end

@testset "forward_prior: an inconsistent forward throws as the constructor would" begin
    #=
    `reconstruct_prior` routes through the carrier's ordinary keyword constructor, so the
    docstring claims that a patch leaving the carrier inconsistent "throws exactly as a
    hand-written constructor call would". Exactly means the same exception type and the
    same message: a caller that reads the message of a hand-written call must be able to
    read the message of a forward. The claim is checked by building each inconsistency
    twice and comparing the two exceptions.
    =#
    function raised(f)
        return try
            f()
            nothing
        catch e
            e
        end
    end
    hand(pr, patch) = LowOrderPrior(; merge(PO.prior_field_values(pr), patch)...)

    pr = weighted_prior()
    for patch in ((; mu = [0.02, 0.03, 0.04]), (; ow = Float64[]),
                  (; sigma = [0.0004 0.0002; 0.0002 0.0003; 0.0 0.0]), (; X = zeros(0, 0)))
        by_forward = raised(() -> PO.forward_prior(pr; patch...))
        by_hand = raised(() -> hand(pr, patch))
        @test !isnothing(by_forward)
        @test typeof(by_forward) === typeof(by_hand)
        @test sprint(showerror, by_forward) == sprint(showerror, by_hand)
    end

    # The same holds on the other carrier, whose constructor validates different shapes.
    hop = HighOrderPrior(; pr = pr, kt = Matrix(1.0I, 4, 4), L2 = PO.elimination_matrix(2),
                         S2 = PO.summation_matrix(2))
    for patch in ((; kt = Matrix(1.0I, 3, 3)), (; S2 = nothing))
        by_forward = raised(() -> PO.forward_prior(hop; patch...))
        by_hand = raised(() -> HighOrderPrior(; merge(PO.prior_field_values(hop), patch)...))
        @test !isnothing(by_forward)
        @test typeof(by_forward) === typeof(by_hand)
        @test sprint(showerror, by_forward) == sprint(showerror, by_hand)
    end
end

#=
Regression tests for the composition rule at the *estimator* level (ADR 0046).

`test_12f_forward_prior.jl` pins the `forward_prior` mechanics in isolation; this
file pins what the construction sites actually forward, and the three silent
defects #181 found by reading them. Every defect test carries its own
falsification witness: a carrier rebuilt exactly as the pre-fix construction site
built it, asserted to show the old behaviour. That is what makes "this test fails
before the fix" checkable from here rather than from a git archaeology session.
=#
include(joinpath(@__DIR__, "test12_setup.jl"))
using StableRNGs, LinearAlgebra, InteractiveUtils

const PO = PortfolioOptimisers

# An asset-side entropy pooling prior whose posterior weights are far enough from
# uniform to be visible: `ens ≈ 499` against `T = 1008`.
const ep_asset = EntropyPoolingPrior(; pe = EmpiricalPrior(), sets = sets,
                                     mu_views = LinearConstraintEstimator(;
                                                                          val = ["$(rd.nx[1]) == 0.02",
                                                                                 "$(rd.nx[2]) == -0.01"]))
# The factor-side counterpart: `ens ≈ 341`. Factor returns are far less dispersed
# than asset returns, so it takes three views to move the weights this far.
const ep_factor = EntropyPoolingPrior(; pe = EmpiricalPrior(), sets = fsets,
                                      mu_views = LinearConstraintEstimator(;
                                                                           val = ["$(rd.nf[1]) == 0.004",
                                                                                  "$(rd.nf[2]) == -0.003",
                                                                                  "$(rd.nf[3]) == 0.003"]))
const bl_views = LinearConstraintEstimator(; val = ["$(rd.nx[1]) == 0.03"])
const f_views = LinearConstraintEstimator(; val = ["$(rd.nf[1]) == 0.004"])

@testset "Defect 1: pooled observation weights reach the risk-measure layer" begin
    pooled = prior(ep_asset, rd)
    pr = prior(BlackLittermanPrior(; pe = ep_asset, sets = sets, views = bl_views), rd)

    # Black-Litterman never touches the observation axis, so the pooled weights
    # still describe exactly the rows of the returned `X` — and they survive.
    @test pr.X === pooled.X
    @test pr.w == pooled.w
    @test pr.ens == pooled.ens
    @test pr.kld == pooled.kld
    @test pr.ens < size(rd.X, 1)

    # ADR 0012's `@pprop w` selection is what carries them into a risk measure.
    rm = ConditionalValueatRisk()
    @test !isnothing(factory(rm, pr).w)

    # And the weights change the number, so the optimisation is no longer running
    # unweighted: the same returns priced against the unweighted empirical prior
    # give a materially different risk.
    wt = fill(inv(size(pr.X, 2)), size(pr.X, 2))
    r_weighted = expected_risk(factory(rm, pr), wt, pr.X)
    r_flat = expected_risk(factory(rm, prior(EmpiricalPrior(), rd)), wt, pr.X)
    @test !isapprox(r_weighted, r_flat; rtol = 1e-2)

    # The pre-fix carrier: `BlackLittermanPrior` forwarded `X`/`mu`/`sigma` and
    # nothing else, so the weights never reached the selection machinery at all.
    old = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma)
    @test isnothing(old.w)
    @test isnothing(factory(rm, old).w)
    @test isapprox(expected_risk(factory(rm, old), wt, pr.X), r_flat)
end

@testset "Defect 2: the uncertainty set is sized off the count of the moments, not T" begin
    pe = BlackLittermanPrior(; pe = ep_asset, sets = sets, views = bl_views)
    pr = prior(pe, rd)
    T = size(rd.X, 1)

    # `choose_scaling_parameter` reads the count of the moment in `pr.eoc` and falls back
    # to `size(pr.X, 1)`. The pooled weights bind it, and it is their Kish count (#1893).
    ue = NormalUncertaintySet(; pe = pe, alg = EllipsoidalUncertaintySetAlgorithm())
    T_mu = PO.moment_observation_count(pr, 1)
    @test PO.choose_scaling_parameter(ue, pr, 1) == T_mu
    @test T_mu ≈ sum(pr.w)^2 / sum(abs2, pr.w)
    @test T_mu < T

    # The set scales as `1/T`, so pinning `ens = T` recovers exactly the set the
    # pre-fix carrier produced, and the two differ by the factor that was lost.
    pinned = NormalUncertaintySet(; pe = pe, alg = EllipsoidalUncertaintySetAlgorithm(),
                                  ens = float(T))
    @test isapprox(tr(mu_ucs(ue, rd).sigma) / tr(mu_ucs(pinned, rd).sigma), T / T_mu)

    # The pre-fix carrier: with the counts dropped, the fallback sized the set off a
    # sample count ~2x too large.
    old = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma)
    @test old.eoc == (nothing, nothing)
    @test PO.choose_scaling_parameter(ue, old, 1) == T
end

@testset "Defect 3: the same defect through FactorPrior over a pooled factor prior" begin
    f_pooled = prior(ep_factor, rd.F)
    pe = FactorPrior(; pe = ep_factor)
    pr = prior(pe, rd)
    T = size(rd.X, 1)

    # `w = f_prior.w` was already forwarded — it is the only weighting in existence
    # here, and `posterior_X = F*M' + b'` has exactly `F`'s rows. What was missing
    # is the provenance that travels with it.
    @test pr.w == f_pooled.w
    @test pr.ens == f_pooled.ens
    @test pr.eoc == f_pooled.eoc
    @test pr.kld == f_pooled.kld
    T_mu = PO.moment_observation_count(pr, 1)
    @test T_mu < T

    ue = NormalUncertaintySet(; pe = pe, alg = EllipsoidalUncertaintySetAlgorithm())
    pinned = NormalUncertaintySet(; pe = pe, alg = EllipsoidalUncertaintySetAlgorithm(),
                                  ens = float(T))
    @test PO.choose_scaling_parameter(ue, pr, 1) == T_mu
    @test isapprox(tr(mu_ucs(ue, rd).sigma) / tr(mu_ucs(pinned, rd).sigma), T / T_mu)

    # The pre-fix carrier: `w` present, diagnostics dropped, so every uncertainty
    # set built on a factor prior over a pooled factor prior was silently too small.
    old = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, chol = pr.chol, w = pr.w,
                        rr = pr.rr, fpr = pr.fpr)
    @test isnothing(old.ens)
    @test old.eoc == (nothing, nothing)
    @test isnothing(old.kld)
    @test PO.choose_scaling_parameter(ue, old, 1) == T
end

@testset "Behaviour change: high order factor moments behind Black-Litterman" begin
    pe = BlackLittermanPrior(; pe = FactorPrior(), sets = sets, views = bl_views)
    pr = prior(pe, rd)

    # `rr` is structural — the regression of `X` on `F`, over data Black-Litterman
    # does not modify — so it is forwarded, and the factor block travels with it.
    @test !isnothing(pr.rr)
    @test !isnothing(pr.fpr)

    # This threw `IsNothingError` before the fix. It now projects the higher
    # co-moments through `rr.M` while `mu`/`sigma` carry the views.
    hop = prior(HighOrderFactorPriorEstimator(; pe = pe), rd)
    @test hop isa HighOrderPrior
    @test all(isfinite, hop.kt)
    @test all(isfinite, hop.sk)
    @test !isnothing(hop.fpr)

    # The pre-fix carrier, and the message it produced.
    old = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma)
    @test isnothing(old.rr)
    @test_throws PO.IsNothingError PO.assert_prior_regression(old, :pe)

    # `assert_prior_regression` serves two kinds of consumer. The default `lead` is the
    # wrapping-estimator one and must stay estimator-framed — it is the half that explains
    # why the *field type* did not catch this, which is only true of an estimator field.
    # The tail is shared with the plotting entry points (see `test_25_plotting.jl`), so it
    # is asserted from the constant rather than restated here.
    err = try
        PO.assert_prior_regression(old, :pe)
    catch e
        e
    end
    @test occursin("this estimator projects factor moments through the regression loadings",
                   err.msg)
    @test occursin("`pe` accepts estimators that use factor returns only optionally",
                   err.msg)
    @test occursin(PO.prior_regression_remedy, err.msg)
    @test endswith(err.msg, PO.prior_regression_remedy)
end

@testset "Every construction site forwards the w bundle" begin
    a_pooled = prior(ep_asset, rd)
    f_pooled = prior(ep_factor, rd.F)

    # `FactorPrior` — the plain lift. No prior result carries feature data at all: the
    # Feature Matrix is derived from the Asset Panel on the data carrier, or built by a
    # producer on the distance, so there is nothing here to forward or to drop.
    fp_lift = prior(FactorPrior(; pe = ep_factor), rd)
    @test !hasproperty(fp_lift, :pnl)
    @test fp_lift.w == f_pooled.w
    @test fp_lift.ens == f_pooled.ens
    @test !isnothing(fp_lift.chol)           # rebuilt on the asset axis, not forwarded

    # `BayesianBlackLittermanPrior` — forwards the bundle; `chol` is the only drop.
    bbl = prior(BayesianBlackLittermanPrior(; pe = FactorPrior(; pe = ep_factor),
                                            sets = xfsets, views = f_views), rd)
    @test bbl.w == f_pooled.w
    @test bbl.ens == f_pooled.ens
    @test isnothing(bbl.chol)
    @test !isnothing(bbl.fpr)

    # `FactorBlackLittermanPrior` — keeps `w = f_prior.w` and now its diagnostics.
    # Its factor block is *modified*, not passed through, and it builds its own
    # `chol` from the posterior factor covariance rather than dropping one.
    fbl = prior(FactorBlackLittermanPrior(; pe = ep_factor, sets = xfsets, views = f_views),
                rd)
    @test fbl.w == f_pooled.w
    @test fbl.ens == f_pooled.ens
    @test fbl.kld == f_pooled.kld
    @test !isnothing(fbl.chol)
    @test fbl.fpr.w == f_pooled.w
    @test fbl.fpr.mu != f_pooled.mu          # the views landed on the factor block
    @test isnothing(fbl.fpr.chol)            # superseded by the posterior covariance

    # `AugmentedBlackLittermanPrior` — symmetric, and the two weightings stay
    # distinguishable: the asset slot is `a_prior`'s, the factor block is `f_prior`.
    abl = prior(AugmentedBlackLittermanPrior(; a_pe = ep_asset, f_pe = ep_factor,
                                             sets = afsets, a_views = bl_views,
                                             f_views = f_views), rd)
    @test abl.w == a_pooled.w
    @test abl.ens == a_pooled.ens
    @test abl.kld == a_pooled.kld
    @test isnothing(abl.chol)
    @test abl.fpr.w == f_pooled.w
    @test abl.fpr.ens == f_pooled.ens
    @test abl.ens != abl.fpr.ens
end

@testset "The factor block reported is the posterior one where a posterior exists" begin
    # Whether `mu == rr.M * fpr.mu + rr.b` holds is a per-member property of the
    # Black-Litterman family, not a uniform caveat. Each member is pinned here with
    # the reason it lands where it does, so a future change to any of them has to
    # move an assertion rather than quietly widening or narrowing the claim.
    identity_gap(pr) = maximum(abs, pr.mu .- (pr.rr.M * pr.fpr.mu + pr.rr.b))

    # `FactorBlackLittermanPrior` — `posterior_mu` *is* `M * f_posterior_mu + b`,
    # so the identity is exact by construction rather than by cancellation.
    fbl = prior(FactorBlackLittermanPrior(; pe = ep_factor, sets = xfsets, views = f_views),
                rd)
    @test identity_gap(fbl) == 0

    # `BayesianBlackLittermanPrior` — the views land on the factors and reach the
    # assets through the loadings, so reporting `mu_hat`/`inv(sigma_hat)` makes the
    # carrier internally consistent. Forwarding the *prior* block, as it used to,
    # left the two halves describing different distributions.
    #
    # The shared `f_views` cannot be used here: it asks for `nf[1] == 0.004`, which is
    # exactly what `ep_factor` already pools to, so the Black-Litterman update moves the
    # factor block by 1.4e-10 and the pre-fix carrier looks consistent by accident. A
    # falsification witness needs a view the prior *disagrees* with.
    contrary_f_views = LinearConstraintEstimator(; val = ["$(rd.nf[1]) == 0.05"])
    bbl_pe = BayesianBlackLittermanPrior(; pe = FactorPrior(; pe = ep_factor),
                                         sets = xfsets, views = contrary_f_views)
    bbl = prior(bbl_pe, rd)
    @test identity_gap(bbl) < 1e-12

    # The falsification witness: the pre-fix carrier, rebuilt by forwarding the
    # wrapped prior's factor block instead of the posterior one.
    wrapped = prior(bbl_pe.pe, rd)
    old = PO.forward_prior(bbl; fpr = wrapped.fpr)
    @test identity_gap(old) > 1e-3
    @test bbl.fpr.mu != wrapped.fpr.mu

    # The views moved the factor block, and only the moments: the observation axis
    # is untouched, so the factor prior's weighting and its diagnostics survive.
    @test bbl.fpr.w == wrapped.fpr.w
    @test bbl.fpr.ens == wrapped.fpr.ens
    @test isnothing(bbl.fpr.chol)
    @test issymmetric(bbl.fpr.sigma)

    # `f_mp` processes the factor block, `mp` the asset block. Naming a processing
    # estimator that would visibly alter a covariance is what proves the field is
    # wired to the factor side rather than ignored.
    bbl_dn = prior(BayesianBlackLittermanPrior(; pe = FactorPrior(; pe = ep_factor),
                                               f_mp = MatrixProcessing(; dn = Denoise()),
                                               sets = xfsets, views = contrary_f_views), rd)
    @test bbl_dn.fpr.sigma != bbl.fpr.sigma
    @test bbl_dn.sigma == bbl.sigma
    # `mp` processes the asset covariance after the posterior mean is formed, as in the
    # other three members. So a processing estimator that changes the covariance moves
    # neither the mean nor the identity. R8 of #1817 found the mean read the processed one.
    bbl_amp = prior(BayesianBlackLittermanPrior(; pe = FactorPrior(; pe = ep_factor),
                                                mp = MatrixProcessing(; dn = Denoise()),
                                                sets = xfsets, views = contrary_f_views),
                    rd)
    @test bbl_amp.sigma != bbl.sigma
    @test bbl_amp.mu == bbl.mu
    @test identity_gap(bbl_amp) < 1e-12

    # `AugmentedBlackLittermanPrior` — both halves are posterior, and the identity
    # still fails, so the warning it carries is about the model rather than about a
    # discarded value. Reporting the posterior half is nonetheless a large
    # improvement over reporting the prior one.
    abl_pe = AugmentedBlackLittermanPrior(; a_pe = ep_asset, f_pe = ep_factor,
                                          sets = afsets, a_views = bl_views,
                                          f_views = f_views)
    abl = prior(abl_pe, rd)
    f_prior = prior(ep_factor, rd.F)
    @test abl.fpr.mu != f_prior.mu
    @test identity_gap(abl) < identity_gap(PO.forward_prior(abl; fpr = f_prior))

    # The cause is idiosyncratic variance, not the asset views. Isolate it on a pair of
    # *unweighted* priors, where least squares with an intercept reproduces the mean and
    # the two halves therefore start out consistent to machine precision: whatever gap
    # the augmented posterior shows is opened by the update alone. (`ep_asset` and
    # `ep_factor` are two different reweightings, so their means do not satisfy the
    # identity to begin with — the fixture above carries both causes at once.)
    rr = regression(abl_pe.re, rd.X, rd.F)
    a_emp = prior(EmpiricalPrior(), rd.X)
    f_emp = prior(EmpiricalPrior(), rd.F)
    @test maximum(abs, a_emp.mu .- (rr.M * f_emp.mu + rr.b)) < 1e-12

    # `BlackLittermanPrior` — asset views only, so no posterior factor distribution
    # is ever computed and the block it forwards is the wrapped prior's, unchanged.
    bl = prior(BlackLittermanPrior(; pe = FactorPrior(; pe = ep_factor), sets = sets,
                                   views = bl_views), rd)
    bl_wrapped = prior(FactorPrior(; pe = ep_factor), rd)
    @test bl.fpr.mu == bl_wrapped.fpr.mu
    @test identity_gap(bl) > 1e-6
end

@testset "`fpr` is the idiomatic read and the flat names are frozen at 13" begin
    pr = prior(FactorPrior(; pe = ep_factor), rd)
    hop = prior(HighOrderFactorPriorEstimator(; pe = FactorPrior(; pe = ep_factor)), rd)

    # Every frozen flat name agrees with the read it stands for. This is the whole
    # compatibility guarantee, so it is asserted name by name rather than sampled.
    low_flat = (:f_mu => :mu, :f_sigma => :sigma, :f_w => :w, :f_ens => :ens,
                :f_kld => :kld, :f_ow => :ow)
    for (flat, nested) in low_flat
        @test getproperty(pr, flat) === getproperty(pr.fpr, nested)
    end
    high_flat = (:f_kt => :kt, :f_sk => :sk, :f_V => :V, :f_D2 => :D2, :f_L2 => :L2,
                 :f_S2 => :S2, :f_skmp => :skmp)
    for (flat, nested) in high_flat
        @test getproperty(hop, flat) === getproperty(hop.fpr, nested)
    end
    @test length(low_flat) + length(high_flat) == 13

    # The set is frozen: no flat name is ever added. Catching a fourteenth here is
    # the point — the freeze is a decision (ADR 0046), not an accident of which
    # names happened to be written.
    flat_names(T) = filter(n -> startswith(string(n), "f_"), propertynames(T))
    @test Set(flat_names(pr)) == Set(first.(low_flat))
    @test Set(flat_names(hop)) == Set(first.(low_flat)) ∪ Set(first.(high_flat))

    # And it is partial, which is *why* it cannot be the idiomatic read: the factor
    # returns matrix has no flat spelling, so `fpr` is the only way to reach it.
    @test !isnothing(pr.fpr.X)
    @test :f_X ∉ propertynames(pr)

    # The two reads differ exactly where the block is absent — the flat name is
    # total, the nested read is partial. That is what the guard exists to settle.
    plain = prior(EmpiricalPrior(), rd)
    @test isnothing(plain.fpr)
    @test isnothing(plain.f_mu)
    @test_throws Exception plain.fpr.mu
end

#=
The per-type declaration of ADR 0046's 2026-08-17 amendment. `factor_residual_config`
has no default: a silent `nothing` fallback cannot separate "this estimator adds no
residual block" from "the author forgot the method", and the second reading drops a
residual block the covariance really carries.
=#
struct UndeclaredResidualPrior <: PO.AbstractLowOrderPriorEstimator_F end
struct WrongShapeResidualPrior <: PO.AbstractLowOrderPriorEstimator_F end
function PO.factor_residual_config(::WrongShapeResidualPrior)
    return (; ve = SimpleVariance())
end

@testset "`factor_residual_config` is declared per type, with no default" begin
    # The census. Every concrete prior estimator the package ships declares beside its own
    # definition, so a new estimator that forgets the method fails here rather than silently
    # reporting "no residual block". `UndeclaredResidualPrior` above is the falsification
    # witness: it is the shape of that forgotten method. The census is scoped to
    # `PortfolioOptimisers` so the witnesses — and any estimator another test file defines —
    # stay out of it.
    tn(T) = Base.typename(Base.unwrap_unionall(T))
    function concrete_estimators(T, acc = Any[])
        for S in InteractiveUtils.subtypes(T)
            isabstracttype(S) ? concrete_estimators(S, acc) : push!(acc, S)
        end
        return acc
    end
    declared = Set(tn(m.sig.parameters[2]) for m in methods(PO.factor_residual_config))
    census = filter(T -> parentmodule(T) === PO,
                    concrete_estimators(PO.AbstractPriorEstimator))
    # `LowDimensionEnsemblePrior` (online portfolio selection) is the thirteenth.
    @test length(census) == 13
    for T in census
        @test tn(T) in declared
    end

    # There is no default: an undeclared type throws, and the message names it.
    @test_throws ArgumentError PO.factor_residual_config(UndeclaredResidualPrior())
    msg = try
        PO.factor_residual_config(UndeclaredResidualPrior())
    catch e
        sprint(showerror, e)
    end
    @test occursin("UndeclaredResidualPrior", msg)

    # The two shapes the contract admits, and the one it does not. The shape is checked
    # before the consumer reads `ve`/`pdm`/`rsd`, so a wrong declaration names itself
    # instead of surfacing as a `FieldError` inside the cokurtosis correction.
    fp = FactorPrior()
    @test isnothing(PO.assert_factor_residual_config(fp, nothing))
    @test isnothing(PO.assert_factor_residual_config(fp, PO.factor_residual_config(fp)))
    @test_throws ArgumentError PO.assert_factor_residual_config(WrongShapeResidualPrior(),
                                                                PO.factor_residual_config(WrongShapeResidualPrior()))

    # The two owners report the block; every wrapper forwards it, so a wrapped
    # `FactorPrior` is still reachable through any depth of wrapping.
    cfg = PO.factor_residual_config(fp)
    @test (cfg.ve, cfg.pdm, cfg.rsd) === (fp.ve, fp.mp.pdm, fp.rsd)
    @test PO.factor_residual_config(HighOrderPriorEstimator(; pe = fp)) === cfg

    # And the estimators that add none say so explicitly, rather than by omission.
    @test isnothing(PO.factor_residual_config(EmpiricalPrior()))
    @test isnothing(PO.factor_residual_config(HighOrderPriorEstimator()))

    # A pooling estimator is a wrapper for this purpose. `OpinionPoolingPrior` takes every
    # moment of its result from the refit `pe2` — the pooled `pes` contribute observation
    # weights alone — so `pe2` is the single wrapped estimator, and the count of pooled
    # opinions does not change that. `OpinionPoolingPrior` sits inside
    # `HighOrderFactorPriorEstimator.pe`'s bound, so a `nothing` here left a residual block
    # in the covariance the cokurtosis correction is defined on.
    op1 = OpinionPoolingPrior(; pes = [EntropyPoolingPrior()], pe2 = fp)
    op2 = OpinionPoolingPrior(; pes = [EntropyPoolingPrior(), EntropyPoolingPrior()],
                              pe2 = EntropyPoolingPrior(; pe = fp))
    op3 = OpinionPoolingPrior(; pes = [EntropyPoolingPrior()])
    @test PO.factor_residual_config(op1) === cfg
    @test PO.factor_residual_config(op2) === cfg
    @test isnothing(PO.factor_residual_config(op3))
end
