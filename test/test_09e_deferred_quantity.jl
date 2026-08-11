#=
A risk-measure slot may hold a **Deferred Quantity** — the Estimator that computes the
value — instead of the value itself. The Estimator resolves in `factory`, against the
optimisation's own prior, so the struct that reaches a kernel always holds a plain value.

The point of the feature is fold stability: `port_opt_view` runs *before* `factory`, so a
stated matrix crosses a view as the whole universe's answer sliced, while a Deferred
Quantity crosses unresolved and computes on the subset.
=#
const PO = PortfolioOptimisers

# Counts the fits, so "one estimator, one fit" is asserted rather than assumed.
mutable struct CountingPriorEstimator{T} <: PO.AbstractPriorEstimator
    pe::T
    n::Int
end
function PortfolioOptimisers.prior(c::CountingPriorEstimator, X, F = nothing; kwargs...)
    c.n += 1
    return prior(c.pe, X, F; kwargs...)
end

@testset "Deferred Quantity: the slot aliases" begin
    # The dynamic half answers "is this slot deferred?" for all four quantity families and
    # for a prior estimator, which computes every quantity at once.
    @test isa(SimpleExpectedReturns(), PO.DeferredQuantity)
    @test isa(PortfolioOptimisersCovariance(), PO.DeferredQuantity)
    @test isa(Coskewness(), PO.DeferredQuantity)
    @test isa(Cokurtosis(), PO.DeferredQuantity)
    @test isa(EmpiricalPrior(), PO.DeferredQuantity)

    # A value is never a Deferred Quantity, and neither is a centring strategy.
    @test !isa(nothing, PO.DeferredQuantity)
    @test !isa([0.1, 0.2], PO.DeferredQuantity)
    @test !isa([1.0 0.0; 0.0 1.0], PO.DeferredQuantity)
    @test !isa(MedianCentering(), PO.DeferredQuantity)
    @test !isa(MeanCentering(), PO.DeferredQuantity)

    # Each field bound admits its own value shape and its own estimator family, plus a
    # prior estimator, which is the only occupant that can reach the factor returns.
    @test isa(0.1, PO.MuSlot) && isa([0.1, 0.2], PO.MuSlot)
    @test isa(SimpleExpectedReturns(), PO.MuSlot) && isa(EmpiricalPrior(), PO.MuSlot)
    @test !isa(PortfolioOptimisersCovariance(), PO.MuSlot)

    @test isa([1.0 0.0; 0.0 1.0], PO.SigmaSlot)
    @test isa(PortfolioOptimisersCovariance(), PO.SigmaSlot) &&
          isa(EmpiricalPrior(), PO.SigmaSlot)
    @test !isa(SimpleExpectedReturns(), PO.SigmaSlot)

    @test isa(Cokurtosis(), PO.KtSlot) && isa(EmpiricalPrior(), PO.KtSlot)
    @test !isa(Coskewness(), PO.KtSlot)

    @test isa(Coskewness(), PO.SkSlot) && isa(EmpiricalPrior(), PO.SkSlot)
    @test !isa(Cokurtosis(), PO.SkSlot)

    # `MedianAbsoluteDeviation.mu` is the widened slot plus its two centring strategies.
    @test isa(MedianCentering(), PO.MedAbsDevMu) && isa(MeanCentering(), PO.MedAbsDevMu)
    @test isa(MedianExpectedReturns(), PO.MedAbsDevMu) && isa([0.1, 0.2], PO.MedAbsDevMu)
    @test !isa(nothing, PO.MedAbsDevMu)
end

@testset "Deferred Quantity: the resolution kernel" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    pr = prior(EmpiricalPrior(), X)

    # An estimator whose answer differs from the prior's, so "resolved" and "took the
    # prior's field" cannot be confused.
    ce = PortfolioOptimisersCovariance(;
                                       ce = Covariance(;
                                                       ce = GeneralCovariance(;
                                                                              ce = StatsBase.SimpleCovariance(;
                                                                                                              corrected = false))))
    me = MedianExpectedReturns()

    # The estimator runs on the prior's own returns and gives its own numbers.
    @test PO.resolve_slot(ce, :sigma, pr) ≈ cov(ce, pr.X)
    @test !isapprox(PO.resolve_slot(ce, :sigma, pr), pr.sigma)
    @test PO.resolve_slot(me, :mu, pr) ≈ vec(mean(me, pr.X))
    @test !isapprox(PO.resolve_slot(me, :mu, pr), pr.mu)

    # A prior estimator computes every quantity at once; `key` picks the one wanted.
    @test PO.resolve_slot(EmpiricalPrior(), :sigma, pr) ≈ pr.sigma
    @test PO.resolve_slot(EmpiricalPrior(), :mu, pr) ≈ pr.mu

    # Everything that is not a Deferred Quantity comes back untouched, so the ordinary
    # prior fallback still applies on top of the kernel.
    @test PO.resolve_slot(nothing, :mu, pr) === nothing
    @test PO.resolve_slot(pr.mu, :mu, pr) === pr.mu
    @test PO.resolve_slot(MedianCentering(), :mu, pr) === MedianCentering()

    # The two families that need a matrix-processing estimator to name their second half
    # arrive with the high-order measures; until then the kernel refuses them loudly.
    @test_throws MethodError PO.fit_deferred_quantity(Coskewness(), pr)
    @test_throws MethodError PO.fit_deferred_quantity(Cokurtosis(), pr)
end

@testset "Deferred Quantity: observation weights are threaded, not invented" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    wp = pweights(range(1, 2; length = 200) ./ sum(range(1, 2; length = 200)))
    wo = pweights(range(2, 1; length = 200) ./ sum(range(2, 1; length = 200)))

    unweighted = prior(EmpiricalPrior(), X)
    weighted = PO.LowOrderPrior(; X = X, mu = unweighted.mu, sigma = unweighted.sigma,
                                w = wp)

    # An unweighted prior leaves the estimator's own weights alone.
    r = LowOrderMoment(; mu = SimpleExpectedReturns(; w = wo))
    @test factory(r, unweighted).mu ≈ vec(mean(SimpleExpectedReturns(; w = wo), X))

    # A weighted prior replaces them.
    @test factory(r, weighted).mu ≈ vec(mean(SimpleExpectedReturns(; w = wp), X))
    @test !isapprox(factory(r, weighted).mu, factory(r, unweighted).mu)

    # The replacement reaches the estimators nested inside a prior estimator too.
    @test factory(LowOrderMoment(;
                                 mu = EmpiricalPrior(;
                                                     me = SimpleExpectedReturns(; w = wo))),
                  weighted).mu ≈ vec(mean(SimpleExpectedReturns(; w = wp), X))
end

@testset "Deferred Quantity: every widened slot resolves" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    pr = prior(EmpiricalPrior(), X)
    ce = PortfolioOptimisersCovariance(;
                                       ce = Covariance(;
                                                       ce = GeneralCovariance(;
                                                                              ce = StatsBase.SimpleCovariance(;
                                                                                                              corrected = false))))
    me = MedianExpectedReturns()
    sigma = cov(ce, X)
    mu = vec(mean(me, X))

    @test factory(Variance(; sigma = ce), pr).sigma ≈ sigma
    @test factory(StandardDeviation(; sigma = ce), pr).sigma ≈ sigma
    @test factory(UncertaintySetVariance(; sigma = ce), pr, nothing).sigma ≈ sigma
    @test factory(LowOrderMoment(; mu = me), pr).mu ≈ mu
    @test factory(HighOrderMoment(; mu = me), pr).mu ≈ mu
    @test factory(ThirdCentralMoment(; mu = me), pr).mu ≈ mu
    @test factory(MedianAbsoluteDeviation(; mu = me), pr).mu ≈ mu

    # None of them silently took the prior's own field instead.
    @test !isapprox(sigma, pr.sigma)
    @test !isapprox(mu, pr.mu)

    # The unstated and stated states are untouched by the third one.
    @test factory(Variance(), pr).sigma === pr.sigma
    @test factory(Variance(; sigma = sigma), pr).sigma === sigma
    @test factory(LowOrderMoment(), pr).mu === pr.mu
    @test factory(LowOrderMoment(; mu = mu), pr).mu === mu

    # `MedianAbsoluteDeviation.mu` is never *filled* by the prior — only an occupant of the
    # field resolves. A bare measure keeps median-centring inside an optimiser.
    @test factory(MedianAbsoluteDeviation(), pr).mu === MedianCentering()
    @test factory(MedianAbsoluteDeviation(; mu = MeanCentering()), pr).mu ===
          MeanCentering()
end

@testset "Deferred Quantity: `sigma` binds `chol`" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    base = prior(EmpiricalPrior(), X)
    # A prior that carries a factorisation, so "kept the prior's `chol`" is visible.
    pr = PO.LowOrderPrior(; X = X, mu = base.mu, sigma = base.sigma,
                          chol = Matrix(LinearAlgebra.cholesky(base.sigma).U))
    ce = PortfolioOptimisersCovariance(;
                                       ce = Covariance(;
                                                       ce = GeneralCovariance(;
                                                                              ce = StatsBase.SimpleCovariance(;
                                                                                                              corrected = false))))
    sigma = base.sigma .* 2
    chol = Matrix(LinearAlgebra.cholesky(sigma).U)

    for T in (Variance, StandardDeviation)
        # Both stated: both kept, and the stated factor is never rebuilt. Under a factor
        # prior it is sparse and special, so a rebuild would throw that structure away.
        f = factory(T(; sigma = sigma, chol = chol), pr)
        @test f.sigma === sigma
        @test f.chol === chol

        # A stated `sigma` with no factor keeps no factor: the prior's would factorise a
        # different matrix, and the kernel derives the right one downstream.
        f = factory(T(; sigma = sigma), pr)
        @test f.sigma === sigma
        @test isnothing(f.chol)

        # A deferred `sigma` supplies the pair, and the stated factor is discarded.
        f = factory(T(; sigma = ce, chol = chol), pr)
        @test f.sigma ≈ cov(ce, X)
        @test isnothing(f.chol)

        # Neither stated: the prior supplies both, as before.
        f = factory(T(), pr)
        @test f.sigma === pr.sigma
        @test f.chol === pr.chol

        # A factor stated on its own is refused at construction. The prior would supply a
        # covariance matrix the caller never saw, and pair it with the caller's factor.
        @test_throws ArgumentError T(; chol = chol)
        err = try
            T(; chol = chol)
        catch e
            e
        end
        @test occursin("chol", sprint(showerror, err))
        @test occursin("sigma", sprint(showerror, err))
    end
end

@testset "Deferred Quantity: factors reach a slot through a prior estimator" begin
    rng = StableRNG(987654321)
    F = randn(rng, 200, 3)
    X = F * randn(rng, 3, 5) .+ 0.1 .* randn(rng, 200, 5)
    pr = prior(FactorPrior(), X, F)

    # The factor returns are the carrier's factor block. No moment estimator takes them.
    @test PO.deferred_factors(pr) === pr.fpr.X
    @test isnothing(PO.deferred_factors(prior(EmpiricalPrior(), X)))

    # A prior estimator in the slot is refit on the caller's own returns and factors, and
    # supplies the sparse factorisation that belongs with the covariance it produced. The
    # returns are `pr.original_X`, not `pr.X`: `FactorPrior` overwrote `X` with the
    # reconstruction, and a factor prior cannot regress that against the same factors.
    ref = prior(FactorPrior(), pr.original_X, pr.fpr.X)
    f = factory(Variance(; sigma = FactorPrior()), pr)
    @test f.sigma ≈ ref.sigma
    @test f.chol ≈ ref.chol
    @test !isnothing(f.chol)

    # The reconstruction is a different matrix, so the refusal to use it is observable.
    @test !isapprox(f.sigma, prior(FactorPrior(), pr.X, pr.fpr.X).sigma)
end

@testset "Deferred Quantity: the fit is on the caller's returns" begin
    rng = StableRNG(987654321)
    T, N, Nf = 200, 8, 3
    F = randn(rng, T, Nf) .* 0.01
    X = F * transpose(randn(rng, N, Nf) .* 0.5) .+ randn(rng, T, N) .* 0.008
    pr = prior(FactorPrior(), X, F)
    ce = SimpleCovariance(; corrected = false)

    # The carrier says which of the two matrices it holds.
    @test pr.X !== X
    @test pr.o_X === X
    @test pr.original_X === X

    # The kernel fits on the caller's matrix. The reconstruction spans only the factors, so
    # a fit there is singular whenever there are more assets than factors — this is the
    # whole reason the field exists, and it is a rank difference rather than a small one.
    fitted = PO.fit_deferred_quantity(ce, pr)
    @test fitted ≈ cov(ce, X)
    @test rank(fitted) == N
    @test rank(cov(ce, pr.X)) == Nf
    @test !isapprox(fitted, cov(ce, pr.X))

    # The mean method reads the same matrix.
    @test PO.fit_deferred_quantity(SimpleExpectedReturns(), pr) ≈
          vec(mean(SimpleExpectedReturns(), X))

    # A resolved slot therefore carries the honest covariance, through both entry points.
    @test factory(Variance(; sigma = ce), pr).sigma ≈ cov(ce, X)

    # Off a factor route the read is the identity, so nothing moved for anyone else.
    ep = prior(EmpiricalPrior(), X)
    @test isnothing(ep.o_X)
    @test ep.original_X === ep.X
    @test PO.fit_deferred_quantity(ce, ep) ≈ cov(ce, X)
end

@testset "Deferred Quantity: the original crosses a view sliced" begin
    rng = StableRNG(987654321)
    T, N, Nf = 200, 8, 3
    F = randn(rng, T, Nf) .* 0.01
    X = F * transpose(randn(rng, N, Nf) .* 0.5) .+ randn(rng, T, N) .* 0.008
    pr = prior(FactorPrior(), X, F)
    i = [1, 3, 5, 7]
    v = PO.port_opt_view(pr, i)
    ce = SimpleCovariance(; corrected = false)

    # `o_X` is assets-major over the same observations as `X`, so it takes the same cut.
    @test v.o_X == X[:, i]
    @test v.original_X == X[:, i]
    @test v.o_X !== v.X

    # And the subset refit is full rank, where a refit on the sliced reconstruction would
    # still be rank `Nf`.
    @test PO.fit_deferred_quantity(ce, v) ≈ cov(ce, X[:, i])
    @test rank(PO.fit_deferred_quantity(ce, v)) == length(i)
    @test rank(cov(ce, v.X)) == Nf
end

@testset "Deferred Quantity: a view is crossed unresolved and refits on the subset" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 20)
    i = [1, 4, 7, 11, 15]
    # Denoising is a function of the whole universe, so a full-universe fit sliced and a
    # subset refit genuinely disagree — the whole argument for the feature.
    ce = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; dn = Denoise()))

    for r in (Variance(; sigma = ce), StandardDeviation(; sigma = ce),
              UncertaintySetVariance(; sigma = ce),
              LowOrderMoment(; mu = MedianExpectedReturns()),
              HighOrderMoment(; mu = MedianExpectedReturns()),
              ThirdCentralMoment(; mu = MedianExpectedReturns()),
              MedianAbsoluteDeviation(; mu = MedianExpectedReturns()))
        # The estimator is the identical object on the other side of the view: nothing was
        # sliced, because there are no numbers yet to slice.
        v = PO.port_opt_view(r, i)
        slot = if isa(r, Union{<:Variance, <:StandardDeviation, <:UncertaintySetVariance})
            :sigma
        else
            :mu
        end
        @test getproperty(v, slot) === getproperty(r, slot)
    end

    # And it computes on the subset, not on the whole universe restricted to it.
    prsub = prior(EmpiricalPrior(), X[:, i])
    resolved = factory(PO.port_opt_view(Variance(; sigma = ce), i), prsub).sigma
    @test resolved ≈ cov(ce, X[:, i])
    @test !isapprox(resolved, cov(ce, X)[i, i])

    # A stated matrix does the opposite, which is the choice the caller is making.
    stated = factory(PO.port_opt_view(Variance(; sigma = cov(ce, X)), i), prsub).sigma
    @test stated ≈ cov(ce, X)[i, i]
end

@testset "Deferred Quantity: `MedianAbsoluteDeviation` keeps two resolution points" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    pr = prior(EmpiricalPrior(), X)
    w = fill(inv(5), 5)

    # The median is not linear, so `median(w'X) != w' * median(X)`. A centring strategy and
    # a median estimator are different quantities, not two spellings of one.
    deferred = factory(MedianAbsoluteDeviation(; mu = MedianExpectedReturns()), pr)(w, X)
    centred = MedianAbsoluteDeviation(; mu = MedianCentering())(w, X)
    @test deferred != centred

    # `MeanCentering()` centres net of fees, a resolved vector is gross. With fees the two
    # differ; with no fees they agree by linearity.
    fees = Fees(; l = 0.02)
    resolved = factory(MedianAbsoluteDeviation(; mu = SimpleExpectedReturns()), pr)
    @test resolved(w, X, fees) !=
          MedianAbsoluteDeviation(; mu = MeanCentering())(w, X, fees)
    @test resolved(w, X) ≈ MedianAbsoluteDeviation(; mu = MeanCentering())(w, X)

    # Neither interface the widening touches needs a new method. `weight_independent_target`
    # is correct before and after resolution: an unresolved estimator resolves to a
    # per-asset vector, which is weight-dependent, and so is the vector it becomes.
    @test !PO.weight_independent_target(MedianExpectedReturns())
    @test !PO.weight_independent_target(resolved.mu)
    @test PO.weight_independent_target(MedianCentering())
    @test !PO.supports_precomputed_returns(LowOrderMoment(; mu = MedianExpectedReturns()))

    # `nothing_scalar_array_view` already covers every estimator family, including the
    # covariance one, which is a `StatsBase.CovarianceEstimator` and not an
    # `AbstractEstimator`.
    for e in
        (SimpleExpectedReturns(), MedianExpectedReturns(), PortfolioOptimisersCovariance(),
         Coskewness(), Cokurtosis(), EmpiricalPrior(), MedianCentering())
        @test PO.nothing_scalar_array_view(e, 1:3) === e
        @test PO.port_opt_view(e, 1:3) == e
    end
end

@testset "Deferred Quantity: a Subset Resampling run" begin
    using Clarabel, JuMP
    rng = StableRNG(987654321)
    X = randn(rng, 200, 10) ./ 100
    rd = ReturnsResult(; X = X, nx = string.(1:10))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    ce = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; dn = Denoise()))

    # The inner optimiser refits its own prior on each subset, so the Deferred Quantity
    # crosses the view and computes there. A stated full-universe matrix is sliced instead,
    # so the two runs must disagree.
    inner(r) = MeanRisk(; r = r, opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv))
    outer(r) = SubsetResampling(; subset_size = 5, n_subsets = 8, rng = StableRNG(123),
                                pe = EmpiricalPrior(), opt = inner(r))

    deferred = optimise(outer(Variance(; sigma = ce)), rd)
    stated = optimise(outer(Variance(; sigma = cov(ce, X))), rd)
    @test isapprox(sum(deferred.w), 1)
    @test isapprox(sum(stated.w), 1)
    @test !isapprox(deferred.w, stated.w)
end

#=
The fan-out. A measure with **two or more independently deferrable slots** takes a `pe`
instead of widening each slot: one prior estimator, one fit, every unstated slot filled.

A measure with exactly one deferrable slot takes no `pe`. It widens that slot, and a derived
companion — `chol` with `sigma` — travels with it out of the same fit. Counting *slots*
rather than *deferrable* slots would have given `Variance` a `pe` that says what
`sigma = <prior estimator>` already says.
=#
@testset "Deferred Quantity: only two independent slots earn a `pe`" begin
    # Four take the fan-out.
    for T in (DistributionValueatRisk, Kurtosis, Skewness, VarianceSkewKurtosis)
        @test :pe in fieldnames(T)
    end

    # Three do not: their second slot is derived, so it can never hold an estimator, and it
    # is already supplied by the fit that resolves its source.
    for T in (Variance, StandardDeviation, NegativeSkewness)
        @test :pe ∉ fieldnames(T)
    end
    @test_throws MethodError Variance(; pe = EmpiricalPrior())
    @test_throws MethodError StandardDeviation(; pe = EmpiricalPrior())
end

@testset "Deferred Quantity: the fan-out fills every unstated slot from one fit" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    lop = prior(EmpiricalPrior(), X)
    hop = prior(HighOrderPriorEstimator(), X)

    # `mu` and `sigma` are independent quantities, so one fit supplies both.
    d = factory(DistributionValueatRisk(; pe = EmpiricalPrior()), lop)
    @test d.mu ≈ lop.mu
    @test d.sigma ≈ lop.sigma
    @test isnothing(d.pe)

    # `mu` and `kt` likewise, off one high-order result.
    k = factory(Kurtosis(; pe = HighOrderPriorEstimator()), hop)
    @test k.mu ≈ hop.mu
    @test k.kt ≈ hop.kt
    @test isnothing(k.pe)

    # And `mu` with `sk`.
    s = factory(Skewness(; pe = HighOrderPriorEstimator()), hop)
    @test s.mu ≈ hop.mu
    @test s.sk ≈ hop.sk
    @test isnothing(s.pe)

    # The container fans out into all three children, five slots in total.
    v = PO.resolve_deferred_quantities(VarianceSkewKurtosis(;
                                                            pe = HighOrderPriorEstimator()),
                                       hop)
    @test v.vr.sigma ≈ hop.sigma
    @test v.sk.sk ≈ hop.sk
    @test v.sk.mu ≈ hop.mu
    @test v.kt.kt ≈ hop.kt
    @test v.kt.mu ≈ hop.mu
    @test isnothing(v.pe)
    @test isnothing(v.sk.pe) && isnothing(v.kt.pe)
end

@testset "Deferred Quantity: a stated slot wins over the fan-out" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    lop = prior(EmpiricalPrior(), X)
    hop = prior(HighOrderPriorEstimator(), X)
    mu = fill(0.5, 5)
    ce = SimpleCovariance(; corrected = false)

    # A stated value keeps its place, and the `pe` fills only what is left.
    d = factory(DistributionValueatRisk(; mu = mu, pe = EmpiricalPrior()), lop)
    @test d.mu === mu
    @test d.sigma ≈ lop.sigma

    k = factory(Kurtosis(; mu = mu, pe = HighOrderPriorEstimator()), hop)
    @test k.mu === mu
    @test k.kt ≈ hop.kt

    s = factory(Skewness(; mu = mu, pe = HighOrderPriorEstimator()), hop)
    @test s.mu === mu
    @test s.sk ≈ hop.sk

    # A child of the container is no different: a Deferred Quantity on a child resolves
    # first and keeps its answer, and the container's `pe` fills the rest. The deferred
    # spelling and the stated one are treated alike — neither is refused.
    v = PO.resolve_deferred_quantities(VarianceSkewKurtosis(;
                                                            pe = HighOrderPriorEstimator(),
                                                            vr = Variance(; sigma = ce)),
                                       hop)
    @test v.vr.sigma ≈ cov(ce, X)
    @test !isapprox(v.vr.sigma, hop.sigma)
    @test v.kt.kt ≈ hop.kt
    @test v.sk.sk ≈ hop.sk

    v = PO.resolve_deferred_quantities(VarianceSkewKurtosis(;
                                                            pe = HighOrderPriorEstimator(),
                                                            vr = Variance(;
                                                                          sigma = cov(ce,
                                                                                      X))),
                                       hop)
    @test v.vr.sigma ≈ cov(ce, X)
    @test v.kt.kt ≈ hop.kt
end

@testset "Deferred Quantity: the fan-out runs the estimator once per measure" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    hop = prior(HighOrderPriorEstimator(), X)

    c = CountingPriorEstimator(HighOrderPriorEstimator(), 0)
    k = factory(Kurtosis(; pe = c), hop)
    @test k.mu ≈ hop.mu && k.kt ≈ hop.kt
    @test c.n == 1

    # Five slots across three children, still one fit. The container resolves the whole
    # measure, not one field at a time, which is what makes this expressible.
    c = CountingPriorEstimator(HighOrderPriorEstimator(), 0)
    v = PO.resolve_deferred_quantities(VarianceSkewKurtosis(; pe = c), hop)
    @test v.kt.kt ≈ hop.kt && v.sk.sk ≈ hop.sk && v.vr.sigma ≈ hop.sigma
    @test c.n == 1

    # Resolution clears `pe`, so a second pass over an already-resolved measure refits
    # nothing. Both entry points may run on the same measure.
    @test PO.resolve_deferred_quantities(v, hop) == v
    @test c.n == 1
end

@testset "Deferred Quantity: the fan-out must be able to supply the quantity" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    hop = prior(HighOrderPriorEstimator(), X)

    # A low-order prior estimator computes no `kt`, and says so by name rather than
    # surfacing a bare `getproperty` failure several frames down.
    err = try
        factory(Kurtosis(; pe = EmpiricalPrior()), hop)
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("kt", sprint(showerror, err))
    @test_throws ArgumentError factory(Skewness(; pe = EmpiricalPrior()), hop)

    # `mu` and `sigma` live on a low-order result, so `DistributionValueatRisk` is content.
    @test factory(DistributionValueatRisk(; pe = EmpiricalPrior()), hop).mu ≈ hop.mu
end

@testset "Deferred Quantity: `DistributionValueatRisk` binds `chol` to `sigma`" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    base = prior(EmpiricalPrior(), X)
    pr = PO.LowOrderPrior(; X = X, mu = base.mu, sigma = base.sigma,
                          chol = Matrix(LinearAlgebra.cholesky(base.sigma).U))
    sigma = base.sigma .* 2
    chol = Matrix(LinearAlgebra.cholesky(sigma).U)
    ce = SimpleCovariance(; corrected = false)

    # The same six rows `Variance` and `StandardDeviation` already answer. `chol` used to be
    # `@pprop`-tagged here, so a stated `sigma` was paired with the prior's factorisation —
    # a factor of a matrix the caller never saw.
    f = factory(DistributionValueatRisk(; sigma = sigma, chol = chol), pr)
    @test f.sigma === sigma
    @test f.chol === chol

    f = factory(DistributionValueatRisk(; sigma = sigma), pr)
    @test f.sigma === sigma
    @test isnothing(f.chol)

    f = factory(DistributionValueatRisk(; sigma = ce, chol = chol), pr)
    @test f.sigma ≈ cov(ce, X)
    @test isnothing(f.chol)

    f = factory(DistributionValueatRisk(), pr)
    @test f.sigma === pr.sigma
    @test f.chol === pr.chol

    @test_throws ArgumentError DistributionValueatRisk(; chol = chol)

    # The fan-out supplies the pair, and a stated `sigma` beside it keeps no factor.
    f = factory(DistributionValueatRisk(; pe = EmpiricalPrior()), pr)
    @test f.sigma ≈ base.sigma
    f = factory(DistributionValueatRisk(; sigma = sigma, pe = EmpiricalPrior()), pr)
    @test f.sigma === sigma
    @test isnothing(f.chol)
end

@testset "Deferred Quantity: the fan-out reaches the `JuMP` resolution point" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    lop = prior(EmpiricalPrior(), X)

    # The `JuMP` builders never call `factory`, so `set_risk_constraints!` resolves. The
    # quantities live one level below the wrapper, on `alg`.
    r = PO.resolve_deferred_quantities(ValueatRisk(;
                                                   alg = DistributionValueatRisk(;
                                                                                 pe = EmpiricalPrior())),
                                       lop)
    @test r.alg.mu ≈ lop.mu
    @test r.alg.sigma ≈ lop.sigma
    @test isnothing(r.alg.pe)

    r = PO.resolve_deferred_quantities(ValueatRiskRange(;
                                                        alg = DistributionValueatRisk(;
                                                                                      pe = EmpiricalPrior())),
                                       lop)
    @test r.alg.mu ≈ lop.mu
    @test r.alg.sigma ≈ lop.sigma

    # `MIPValueatRisk` defers nothing, so the recursion is inert for it.
    @test PO.resolve_deferred_quantities(ValueatRisk(), lop).alg === MIPValueatRisk()
end

@testset "Deferred Quantity: the fan-out crosses a view and fits the subset" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    i = [1, 3, 5]

    # `port_opt_view` runs before `factory`, so the estimator crosses unsliced and computes
    # on the subset — the whole point of the feature, now on the fan-out slot.
    for r in (DistributionValueatRisk(; pe = EmpiricalPrior()),
              Kurtosis(; pe = HighOrderPriorEstimator()),
              Skewness(; pe = HighOrderPriorEstimator()),
              VarianceSkewKurtosis(; pe = HighOrderPriorEstimator()))
        @test PO.port_opt_view(r, i).pe === r.pe
    end

    sub = prior(EmpiricalPrior(), X[:, i])
    d = factory(PO.port_opt_view(DistributionValueatRisk(; pe = EmpiricalPrior()), i), sub)
    @test size(d.sigma) == (3, 3)
    @test d.sigma ≈ sub.sigma
    @test d.mu ≈ sub.mu

    subh = prior(HighOrderPriorEstimator(), X[:, i])
    k = factory(PO.port_opt_view(Kurtosis(; pe = HighOrderPriorEstimator()), i), subh)
    @test size(k.kt) == (9, 9)
    @test k.kt ≈ subh.kt
end
