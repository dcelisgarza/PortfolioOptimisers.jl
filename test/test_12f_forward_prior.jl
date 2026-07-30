#=
Unit tests for `forward_prior` and the `reconstruct_prior` it is built on
(ADR 0046). These pin the composition rule's mechanics — forwarding is the
default, drops are explicit, the two field bindings are enforced, and the
carrier's own validation runs on every reconstruction — without going through any
estimator. What the construction sites actually forward is pinned separately, in
`test_12g_forwarding_rule.jl`.
=#
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
    for sym in (:X, :sigma, :chol, :w, :ens, :kld, :ow, :rr, :fpr, :Z)
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

    # The factor block is all-or-none, so dropping one half of it throws — the
    # point of routing through the carrier's own constructor.
    @test_throws ArgumentError PO.forward_prior(pr; rr = nothing)
    @test_throws ArgumentError PO.forward_prior(pr; fpr = nothing)
    # Dropping the whole block at once is valid.
    dropped = PO.forward_prior(pr; rr = nothing, fpr = nothing)
    @test isnothing(dropped.rr)
    @test isnothing(dropped.fpr)

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
    # The feature matrix keeps its assets-major check.
    @test_throws DimensionMismatch PO.forward_prior(bare; Z = [1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test PO.forward_prior(bare; Z = [1.0 2.0; 3.0 4.0]).Z == [1.0 2.0; 3.0 4.0]
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
    # only `forward_prior` enforces the two bindings.
    @test_throws DimensionMismatch PO.reconstruct_prior(pr, (; mu = [0.7, 0.8, 0.9]))
    @test PO.reconstruct_prior(pr, (; w = StatsBase.pweights([0.5, 0.5]))).ens === pr.ens
end
