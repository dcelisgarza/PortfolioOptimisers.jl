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
using StableRNGs, LinearAlgebra

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

    # The pre-fix carrier: `BlackLittermanPrior` forwarded `X`/`mu`/`sigma`/`Z` and
    # nothing else, so the weights never reached the selection machinery at all.
    old = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, Z = pr.Z)
    @test isnothing(old.w)
    @test isnothing(factory(rm, old).w)
    @test isapprox(expected_risk(factory(rm, old), wt, pr.X), r_flat)
end

@testset "Defect 2: the uncertainty set is sized off ens, not T" begin
    pe = BlackLittermanPrior(; pe = ep_asset, sets = sets, views = bl_views)
    pr = prior(pe, rd)
    T = size(rd.X, 1)

    # `choose_scaling_parameter` prefers `pr.ens` and falls back to `size(pr.X, 1)`.
    ue = NormalUncertaintySet(; pe = pe, alg = EllipsoidalUncertaintySetAlgorithm())
    @test PO.choose_scaling_parameter(ue, pr) == pr.ens
    @test pr.ens < T

    # The set scales as `1/T`, so pinning `ens = T` recovers exactly the set the
    # pre-fix carrier produced, and the two differ by the factor that was lost.
    pinned = NormalUncertaintySet(; pe = pe, alg = EllipsoidalUncertaintySetAlgorithm(),
                                  ens = float(T))
    @test isapprox(tr(mu_ucs(ue, rd).sigma) / tr(mu_ucs(pinned, rd).sigma), T / pr.ens)

    # The pre-fix carrier: with `ens` dropped, the fallback sized the set off a
    # sample count ~2x too large.
    old = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, Z = pr.Z)
    @test isnothing(old.ens)
    @test PO.choose_scaling_parameter(ue, old) == T
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
    @test pr.kld == f_pooled.kld
    @test pr.ens < T

    ue = NormalUncertaintySet(; pe = pe, alg = EllipsoidalUncertaintySetAlgorithm())
    pinned = NormalUncertaintySet(; pe = pe, alg = EllipsoidalUncertaintySetAlgorithm(),
                                  ens = float(T))
    @test PO.choose_scaling_parameter(ue, pr) == pr.ens
    @test isapprox(tr(mu_ucs(ue, rd).sigma) / tr(mu_ucs(pinned, rd).sigma), T / pr.ens)

    # The pre-fix carrier: `w` present, diagnostics dropped, so every uncertainty
    # set built on a factor prior over a pooled factor prior was silently too small.
    old = LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, chol = pr.chol, w = pr.w,
                        rr = pr.rr, fpr = pr.fpr)
    @test isnothing(old.ens)
    @test isnothing(old.kld)
    @test PO.choose_scaling_parameter(ue, old) == T
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

    # `FeaturePrior` — `Z` is the single deviation; everything else forwards.
    rng = StableRNG(987654321)
    Zlit = rand(rng, size(rd.X, 2), 4)
    fp = prior(FeaturePrior(; pe = ep_asset, ze = Zlit), rd)
    @test fp.Z == Zlit
    @test fp.w == a_pooled.w
    @test fp.ens == a_pooled.ens
    @test fp.kld == a_pooled.kld
    @test fp.chol == a_pooled.chol
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
