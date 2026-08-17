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
mutable struct _test_CountingPriorEstimator{T} <: PO.AbstractPriorEstimator
    pe::T
    n::Int
end
function PortfolioOptimisers.prior(c::_test_CountingPriorEstimator, X, F = nothing;
                                   kwargs...)
    c.n += 1
    return prior(c.pe, X, F; kwargs...)
end

# Declares a deferrable slot and writes no resolver, which is the half of ADR 0051's pair that
# the derived recursion cannot supply. Nothing in `src/` may look like this.
struct _test_DeclaresWithoutResolving{T}
    mu::T
end
PortfolioOptimisers.deferred_slots(x::_test_DeclaresWithoutResolving) = (; mu = x.mu)

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

    # The two high-order families. A cokurtosis estimator gives its tensor. A coskewness
    # estimator gives the pair `(sk, V)` plus the processor that built `V`, because `V` is
    # derived from `sk` and never travels on its own.
    @test PO.fit_deferred_quantity(Cokurtosis(), pr) ≈ cokurtosis(Cokurtosis(), pr.X)
    fitted = PO.fit_deferred_quantity(Coskewness(), pr)
    sk, V = coskewness(Coskewness(), pr.X)
    @test PO.deferred_quantity(fitted, :sk) ≈ sk
    @test PO.deferred_derived_quantity(fitted, :V) ≈ V
    @test PO.deferred_derived_quantity(fitted, :skmp) === Coskewness().mp
    @test PO.resolve_slot(Coskewness(), :sk, pr) ≈ sk

    # A key the fit does not carry is refused by name rather than several frames down.
    @test_throws ArgumentError PO.deferred_quantity(fitted, :kt)
    # A *derived* key it does not carry is `nothing`, so the consumer keeps its fallback.
    @test isnothing(PO.deferred_derived_quantity(fitted, :mu))
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

        # A deferred `sigma` supplies the pair, so a factor stated beside it is refused at
        # construction rather than silently dropped. Same rule as `NegativeSkewness`'s
        # `sk`/`V` pair.
        @test_throws ArgumentError T(; sigma = ce, chol = chol)

        # A deferred `sigma` on its own resolves, and the fit supplies whatever factor it
        # has. A covariance estimator has none, so the kernel derives it downstream.
        f = factory(T(; sigma = ce), pr)
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

@testset "Deferred Quantity: every derived slot follows one rule" begin
    # `chol` is derived from `sigma`, `V` from `sk`. Both are refused in the same two
    # states, so a caller learns the rule once. Neither state was constructible before the
    # source slots widened, so neither refusal breaks an existing caller.
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    base = prior(EmpiricalPrior(), X)
    chol = Matrix(LinearAlgebra.cholesky(base.sigma).U)
    sk, V = coskewness(Coskewness(), X)

    # The last entry is the error type for a derived slot stated with no source at all.
    # `NegativeSkewness` refuses that state through its own both-or-neither rule, which
    # predates this feature and keeps its own error type.
    pairs = ((Variance, :sigma, :chol, PortfolioOptimisersCovariance(), base.sigma, chol,
              ArgumentError),
             (StandardDeviation, :sigma, :chol, PortfolioOptimisersCovariance(), base.sigma,
              chol, ArgumentError),
             (DistributionValueatRisk, :sigma, :chol, PortfolioOptimisersCovariance(),
              base.sigma, chol, ArgumentError),
             (NegativeSkewness, :sk, :V, Coskewness(), sk, V, PO.IsNothingError))

    for (T, sname, dname, dq, svalue, dvalue, alone_err) in pairs
        # 1. The derived slot stated with no source at all.
        @test_throws alone_err T(; (dname => dvalue,)...)

        # 2. The derived slot stated beside a Deferred Quantity in the source.
        @test_throws ArgumentError T(; (sname => dq, dname => dvalue)...)
        err = try
            T(; (sname => dq, dname => dvalue)...)
        catch e
            e
        end
        @test occursin("Deferred Quantity", sprint(showerror, err))
        @test occursin(string(dname), sprint(showerror, err))
        @test occursin(string(sname), sprint(showerror, err))

        # Both stated as values is the one way to state the derived slot, and it is kept.
        @test getproperty(T(; (sname => svalue, dname => dvalue)...), dname) === dvalue

        # A deferred source on its own is always fine.
        @test isa(getproperty(T(; (sname => dq,)...), sname), PO.DeferredQuantity)
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

    c = _test_CountingPriorEstimator(HighOrderPriorEstimator(), 0)
    k = factory(Kurtosis(; pe = c), hop)
    @test k.mu ≈ hop.mu && k.kt ≈ hop.kt
    @test c.n == 1

    # Five slots across three children, still one fit. The container resolves the whole
    # measure, not one field at a time, which is what makes this expressible.
    c = _test_CountingPriorEstimator(HighOrderPriorEstimator(), 0)
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

    @test_throws ArgumentError DistributionValueatRisk(; sigma = ce, chol = chol)

    f = factory(DistributionValueatRisk(; sigma = ce), pr)
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

@testset "Deferred Quantity: a higher moment carries the centre it was taken about" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    lop = prior(EmpiricalPrior(), X)
    me = MedianExpectedReturns()
    centre = vec(mean(me, X))

    # A higher moment is a moment *about* a centre, so the tensor and the centre are one
    # pair of quantities out of one object: `mu` comes from the co-moment estimator's own
    # `me`, and the tensor is built about exactly that vector.
    k = factory(Kurtosis(; kt = Cokurtosis(; me = me)), lop)
    @test k.kt ≈ cokurtosis(Cokurtosis(; me = me), X)
    @test k.mu ≈ centre

    s = factory(Skewness(; sk = Coskewness(; me = me)), lop)
    @test s.sk ≈ first(coskewness(Coskewness(; me = me), X))
    @test s.mu ≈ centre

    # Neither silently took the prior's own field instead.
    @test !isapprox(centre, lop.mu)

    # A stated `mu` wins, and is threaded as `mean =` all the same, so the tensor is
    # centred on it rather than on the estimator's own choice.
    mu = fill(0.05, 5)
    k = factory(Kurtosis(; kt = Cokurtosis(), mu = mu), lop)
    @test k.mu === mu
    @test k.kt ≈ cokurtosis(Cokurtosis(), X; mean = transpose(mu))
    @test !isapprox(k.kt, cokurtosis(Cokurtosis(), X))

    s = factory(Skewness(; sk = Coskewness(), mu = mu), lop)
    @test s.mu === mu
    @test s.sk ≈ first(coskewness(Coskewness(), X; mean = transpose(mu)))

    # A scalar and a `VecScalar` centre reach the fit in the shape the estimator wants.
    @test PO.centring_target(0.05) === 0.05
    @test PO.centring_target(VecScalar(; v = mu, s = 0.01)) ≈ transpose(mu .+ 0.01)
    @test isnothing(PO.centring_target(nothing))
    @test factory(Kurtosis(; kt = Cokurtosis(), mu = 0.05), lop).kt ≈
          cokurtosis(Cokurtosis(), X; mean = 0.05)

    # An estimator that names no `me` centres itself, and the slot stays on its fallback.
    @test isnothing(PO.deferred_centre(SimpleExpectedReturns(), lop))
end

@testset "Deferred Quantity: a prior estimator in `kt`/`sk` centres itself" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    lop = prior(EmpiricalPrior(), X)
    hop = prior(HighOrderPriorEstimator(), X)

    # A prior estimator computes its own `mu` and its own tensor about that `mu`, so the
    # centre is read *back* off the result rather than pushed into it.
    k = factory(Kurtosis(; kt = HighOrderPriorEstimator()), lop)
    @test k.kt ≈ hop.kt
    @test k.mu ≈ hop.mu

    s = factory(Skewness(; sk = HighOrderPriorEstimator()), lop)
    @test s.sk ≈ hop.sk
    @test s.mu ≈ hop.mu

    # It has no channel to take a centre, so a stated `mu` still wins as the measure's
    # centring target and the tensor keeps the prior's own.
    mu = fill(0.05, 5)
    k = factory(Kurtosis(; kt = HighOrderPriorEstimator(), mu = mu), lop)
    @test k.mu === mu
    @test k.kt ≈ hop.kt

    # A low-order prior estimator computes no `kt`, and says so by name.
    err = try
        factory(Kurtosis(; kt = EmpiricalPrior()), lop)
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("kt", sprint(showerror, err))
end

@testset "Deferred Quantity: a deferred high-order slot wins over `pe`" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    hop = prior(HighOrderPriorEstimator(), X)
    me = MedianExpectedReturns()
    centre = vec(mean(me, X))

    # The map's precedence rule one level down: a stated slot wins over the fan-out, and a
    # deferred slot is stated. It brings its centre with it, so `pe` fills neither.
    k = factory(Kurtosis(; kt = Cokurtosis(; me = me), pe = HighOrderPriorEstimator()), hop)
    @test k.kt ≈ cokurtosis(Cokurtosis(; me = me), X)
    @test k.mu ≈ centre
    @test !isapprox(k.mu, hop.mu)
    @test isnothing(k.pe)

    s = factory(Skewness(; sk = Coskewness(; me = me), pe = HighOrderPriorEstimator()), hop)
    @test s.sk ≈ first(coskewness(Coskewness(; me = me), X))
    @test s.mu ≈ centre

    # `pe` still fills the half the caller left unstated.
    k = factory(Kurtosis(; mu = MedianExpectedReturns(), pe = HighOrderPriorEstimator()),
                hop)
    @test k.mu ≈ centre
    @test k.kt ≈ hop.kt

    # Nothing deferred leaves the measure untouched, so every existing fallback stands.
    for r in (Kurtosis(), Skewness(), NegativeSkewness())
        @test PO.resolve_deferred_quantities(r, hop) === r
    end
end

@testset "Deferred Quantity: `sk` binds `V`, and the fit's own `mp` builds it" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    lop = prior(EmpiricalPrior(), X)
    hop = prior(HighOrderPriorEstimator(), X)
    me = MedianExpectedReturns()
    sk, V = coskewness(Coskewness(; me = me), X)

    # `V = negative_spectral_coskewness(sk, X, mp)` is never a function of `sk` alone, so
    # building it always names a processor. The one that built the `V` the fit hands back
    # is the one recorded, and the measure's own is discarded.
    mp = MatrixProcessing(; dn = Denoise())
    @test !isapprox(PO.negative_spectral_coskewness(sk, X, mp), V)

    r = factory(NegativeSkewness(; sk = Coskewness(; me = me), mp = mp), lop)
    @test r.sk ≈ sk
    @test r.V ≈ V
    @test r.mp == Coskewness().mp
    @test r.mp !== mp

    r = factory(NegativeSkewness(; sk = Coskewness(; me = me, mp = mp)), lop)
    @test r.V ≈ PO.negative_spectral_coskewness(sk, X, mp)
    @test r.mp === mp

    # A prior estimator supplies the same triple, and `HighOrderPrior` already carries the
    # processor as `skmp` for exactly this reason.
    r = factory(NegativeSkewness(; sk = HighOrderPriorEstimator(), mp = mp), hop)
    @test r.sk ≈ hop.sk
    @test r.V ≈ hop.V
    @test r.mp === hop.skmp

    # The processor is read off the estimator by a per-type method, because the
    # `CoskewnessEstimator` interface does not require an `mp` field.
    @test PO.coskewness_processor(Coskewness(; mp = mp)) === mp

    # Coskewness needs only a returns matrix, so it resolves against a low-order prior,
    # where the measure had no fallback at all.
    @test factory(NegativeSkewness(; sk = Coskewness()), lop).V ≈
          last(coskewness(Coskewness(), X))
end

@testset "Deferred Quantity: `V` never defers and never travels alone" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    sk, V = coskewness(Coskewness(), X)

    # `V` is derived from `sk`, so the fit supplies the pair and a stated `V` beside a
    # deferred `sk` would describe a coskewness matrix the caller never saw. `sigma`/`chol`
    # is refused in the same state, by the same helper.
    @test_throws ArgumentError NegativeSkewness(; sk = Coskewness(), V = V)

    # The existing both-or-neither rule on the stated pair is untouched, error type and all.
    @test_throws PO.IsNothingError NegativeSkewness(; V = V)
    @test_throws PO.IsNothingError NegativeSkewness(; sk = sk)
    @test NegativeSkewness(; sk = sk, V = V).V === V
    @test isnothing(NegativeSkewness().V)
end

@testset "Deferred Quantity: the high-order measures reach the `JuMP` resolution point" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    lop = prior(EmpiricalPrior(), X)
    me = MedianExpectedReturns()
    centre = vec(mean(me, X))

    # The `JuMP` builders never call `factory`, so `set_risk_constraints!` resolves. Each
    # of the three is reached directly rather than through a wrapper.
    k = PO.resolve_deferred_quantities(Kurtosis(; kt = Cokurtosis(; me = me)), lop)
    @test k.kt ≈ cokurtosis(Cokurtosis(; me = me), X)
    @test k.mu ≈ centre

    s = PO.resolve_deferred_quantities(Skewness(; sk = Coskewness(; me = me)), lop)
    @test s.mu ≈ centre

    n = PO.resolve_deferred_quantities(NegativeSkewness(; sk = Coskewness(; me = me)), lop)
    @test n.V ≈ last(coskewness(Coskewness(; me = me), X))
end

@testset "Deferred Quantity: a high-order slot crosses a view and fits the subset" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    i = [1, 3, 5]
    me = MedianExpectedReturns()

    # `port_opt_view` runs before `factory`, so the estimator crosses unresolved and
    # computes on the subset rather than being the whole universe's answer sliced.
    @test isa(PO.port_opt_view(Kurtosis(; kt = Cokurtosis()), i, X).kt, Cokurtosis)
    @test isa(PO.port_opt_view(Skewness(; sk = Coskewness()), i, X).sk, Coskewness)
    @test isa(PO.port_opt_view(NegativeSkewness(; sk = Coskewness()), i, X).sk, Coskewness)

    sub = prior(EmpiricalPrior(), view(X, :, i))
    k = factory(PO.port_opt_view(Kurtosis(; kt = Cokurtosis(; me = me)), i, X), sub)
    @test size(k.kt) == (9, 9)
    @test k.kt ≈ cokurtosis(Cokurtosis(; me = me), X[:, i])
    @test k.mu ≈ vec(mean(me, X[:, i]))

    sk_sub, V_sub = coskewness(Coskewness(; me = me), X[:, i])
    n = factory(PO.port_opt_view(NegativeSkewness(; sk = Coskewness(; me = me)), i, X), sub)
    @test size(n.sk) == (3, 9)
    @test n.sk ≈ sk_sub
    @test n.V ≈ V_sub

    # A view keeps the measure's settings, so a `VarianceSkewKurtosis` child that was
    # silenced stays silent across a subset.
    s = PO.port_opt_view(Skewness(;
                                  settings = MaxRiskMeasureSettings(; scale = 3.0, lb = 0.1,
                                                                    rke = false)), i, X)
    @test s.settings.scale == 3.0
    @test s.settings.lb == 0.1
    @test s.settings.rke === false
end

@testset "Deferred Quantity: a `VarianceSkewKurtosis` child defers one level down" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    hop = prior(HighOrderPriorEstimator(), X)
    me = MedianExpectedReturns()

    # A deferred child slot resolves first and wins over the container's `pe`, which then
    # fills only what is left. One container, two fits, five slots.
    r = PO.resolve_deferred_quantities(VarianceSkewKurtosis(;
                                                            sk = Skewness(;
                                                                          sk = Coskewness(;
                                                                                          me = me)),
                                                            pe = HighOrderPriorEstimator()),
                                       hop)
    @test r.sk.sk ≈ first(coskewness(Coskewness(; me = me), X))
    @test r.sk.mu ≈ vec(mean(me, X))
    @test !isapprox(r.sk.mu, hop.mu)
    @test r.kt.kt ≈ hop.kt
    @test r.kt.mu ≈ hop.mu
    @test r.vr.sigma ≈ hop.sigma
    @test isnothing(r.pe)

    # The children keep the `rke = false` the container forced on them.
    @test r.vr.settings.rke === false
    @test r.sk.settings.rke === false
    @test r.kt.settings.rke === false
end

@testset "Deferred Quantity: a resolved quantity lifts the `HighOrderPrior` gate" begin
    using Clarabel, JuMP
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    rd = ReturnsResult(; X = X, nx = string.(1:5))
    lop = prior(EmpiricalPrior(), X)
    hop = prior(HighOrderPriorEstimator(), X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))

    # The vectorisation matrices were the only other thing the kurtosis kernel took from
    # the prior, and they are a pure function of the asset count. So the rebuild is not an
    # approximation of the prior's: it is the same three matrices.
    D2, L2, S2 = PO.dup_elim_sum_selector(lop, 5)
    @test D2 == hop.D2
    @test L2 == hop.L2
    @test S2 == hop.S2
    @test PO.dup_elim_sum_selector(hop, 5) === (hop.D2, hop.L2, hop.S2)

    # A `HighOrderPrior` carries the high-order quantities and a `LowOrderPrior` none, and
    # the read answers `nothing` rather than throwing either way.
    @test PO.prior_high_order_quantity(hop, :kt) === hop.kt
    @test isnothing(PO.prior_high_order_quantity(lop, :kt))
    @test isnothing(PO.prior_high_order_quantity(lop, :V))

    # A caller who has told the measure how to build its own tensor has met the
    # requirement, so the same problem solves under either prior and gives one answer.
    low(r) = optimise(MeanRisk(; r = r,
                               opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv)), rd)
    high(r) = optimise(MeanRisk(; r = r,
                                opt = JuMPOptimiser(; pe = HighOrderPriorEstimator(),
                                                    slv = slv)), rd)
    for (deferred, bare) in ((Kurtosis(; kt = Cokurtosis()), Kurtosis()),
                             (Kurtosis(; kt = Cokurtosis(), N = 3), Kurtosis(; N = 3)),
                             (Kurtosis(; pe = HighOrderPriorEstimator()), Kurtosis()),
                             (NegativeSkewness(; sk = Coskewness()), NegativeSkewness()),
                             (NegativeSkewness(; sk = HighOrderPriorEstimator()), NegativeSkewness()),
                             (VarianceSkewKurtosis(; sk = Skewness(; sk = Coskewness()),
                                                   kt = Kurtosis(; kt = Cokurtosis())), VarianceSkewKurtosis()),
                             (VarianceSkewKurtosis(; pe = HighOrderPriorEstimator()), VarianceSkewKurtosis()))
        a = low(deferred)
        b = high(bare)
        @test isa(a.retcode, OptimisationSuccess)
        @test isa(b.retcode, OptimisationSuccess)
        @test isapprox(a.w, b.w; rtol = 5e-6)
    end

    # The bare measure under a low-order prior still refuses, and the message names the
    # three ways out rather than only the prior it wanted.
    for (r, quantity, est) in ((Kurtosis(), "kt", "CokurtosisEstimator"),
                               (NegativeSkewness(), "sk", "CoskewnessEstimator"),
                               (VarianceSkewKurtosis(), "sk", "CoskewnessEstimator"))
        err = try
            low(r)
        catch e
            e
        end
        @test isa(err, ArgumentError)
        msg = sprint(showerror, err)
        @test occursin("needs a `$quantity`", msg)
        @test occursin("LowOrderPrior", msg)
        @test occursin(est, msg)
        @test occursin("AbstractPriorEstimator", msg)
    end

    # A container refuses on whichever tensor is missing, so a `kt` stated alone is not
    # enough and the message says which one it still needs.
    err = try
        low(VarianceSkewKurtosis(; kt = Kurtosis(; kt = Cokurtosis())))
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("needs a `sk`", sprint(showerror, err))

    # A `HighOrderPrior` fitted with only one of the two tensors leaves the vectorisation
    # matrices `nothing`, so the rebuild is what makes the other measure buildable there.
    part = optimise(MeanRisk(; r = Kurtosis(; kt = Cokurtosis()),
                             opt = JuMPOptimiser(;
                                                 pe = HighOrderPriorEstimator(;
                                                                              kte = nothing),
                                                 slv = slv)), rd)
    @test isa(part.retcode, OptimisationSuccess)
    @test isapprox(part.w, low(Kurtosis(; kt = Cokurtosis())).w; rtol = 5e-6)
end

#=
`ArithmeticReturn.mu` is the last widened slot, and it is not a risk-measure slot. It sits
on the middle rung of the #277 ladder: the set's carried centre wins, then `rt.mu`, then
`pr.mu` (ADR 0050). A Deferred Quantity adds a **state** to the `rt.mu` rung, not a rung.
An uncertainty set must bound the quantity it was calibrated on, so letting a Deferred
Quantity outrank the carried centre would reintroduce the defect ADR 0050 fixed.
=#
@testset "Deferred Quantity: `ArithmeticReturn.mu` admits an Estimator, not a Result" begin
    @test isa(0.1, PO.ArithRetMu) && isa([0.1, 0.2], PO.ArithRetMu)
    @test isa(SimpleExpectedReturns(), PO.ArithRetMu)
    @test isa(MedianExpectedReturns(), PO.ArithRetMu)
    @test isa(EmpiricalPrior(), PO.ArithRetMu)
    @test !isa(PortfolioOptimisersCovariance(), PO.ArithRetMu)

    # Narrower than `MuSlot` by a `VecScalar`. The return expression is `dot_scalar(mu, w)`,
    # which takes a number or a vector, and a `VecScalar` is an `AbstractResult`, which an
    # Estimator must not hold.
    vs = PO.VecScalar(; v = [0.1, 0.2], s = 0.3)
    @test isa(vs, PO.MuSlot)
    @test !isa(vs, PO.ArithRetMu)
    @test_throws TypeError ArithmeticReturn(; mu = vs)
end

@testset "Deferred Quantity: `ArithmeticReturn.mu` resolves in `factory`" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    lop = prior(EmpiricalPrior(), X)
    me = MedianExpectedReturns()
    mu_me = vec(mean(me, X))
    rt = ArithmeticReturn(; mu = me)
    slv = Solver(; name = :none, solver = nothing)
    ucs = L1UncertaintySet(; eps = 0.05)

    # The seam is the identity on a slot that holds no Deferred Quantity.
    for x in (ArithmeticReturn(), ArithmeticReturn(; mu = lop.mu))
        @test PO.resolve_deferred_quantities(x, lop) === x
    end
    @test PO.resolve_deferred_quantities(rt, lop).mu ≈ mu_me
    # A prior estimator supplies the vector out of a whole fit.
    rtp = ArithmeticReturn(; mu = EmpiricalPrior())
    @test PO.resolve_deferred_quantities(rtp, lop).mu ≈ lop.mu

    # All three prior-carrying `factory` methods resolve; the fourth has no prior in hand,
    # so the Deferred Quantity travels on unchanged.
    @test factory(rt, lop).mu ≈ mu_me
    @test factory(rt, lop, slv).mu ≈ mu_me
    @test factory(rt, ucs, lop).mu ≈ mu_me
    @test factory(rt, ucs).mu === me

    # `lb` and `ucs` are untouched by the resolution.
    rtb = ArithmeticReturn(; mu = me, settings = JuMPReturnsSettings(; lb = 0.001),
                           ucs = ucs)
    @test factory(rtb, lop).settings.lb == 0.001
    @test factory(rtb, lop).ucs === ucs

    # The value-level twin of the `ret` expression applies the same ladder, and the prior is
    # in hand there too.
    w = fill(0.2, 5)
    @test expected_return(rt, w, lop) ≈
          expected_return(ArithmeticReturn(; mu = mu_me), w, lop)
end

@testset "Deferred Quantity: `ArithmeticReturn.mu` crosses a view, drops its bounds" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    i = [1, 3, 5]
    me = MedianExpectedReturns()
    rt = ArithmeticReturn(; mu = me, settings = JuMPReturnsSettings(; lb = 0.001))

    # `port_opt_view` runs before `factory`, so the Estimator crosses unsliced and then
    # computes on the subset.
    @test PO.port_opt_view(rt, i).mu === me
    sub = prior(EmpiricalPrior(), X[:, i])
    @test factory(PO.port_opt_view(rt, i), sub).mu ≈ vec(mean(me, X[:, i]))

    # The frontier sub-problem strips the bounds and forwards the slot, which its own
    # prior-carrying `factory` then resolves.
    @test PO.no_bounds_returns_estimator(rt).mu === me
    @test isnothing(PO.no_bounds_returns_estimator(rt).settings.lb)
    # `flag = false` strips the uncertainty set, and the characteristic stays put. It used
    # to be dropped too, which silently re-centred the corner solve on the prior's own
    # vector — the ADR 0050 defect class, and with several terms it collapsed every corner
    # onto the same one.
    @test PO.no_bounds_returns_estimator(rt, false).mu === me
    @test isnothing(PO.no_bounds_returns_estimator(rt, false).ucs)
end

@testset "Deferred Quantity: `ArithmeticReturn.mu` keeps the #277 ladder intact" begin
    using Clarabel, JuMP
    rng = StableRNG(987654321)
    X = randn(rng, 200, 10) ./ 100
    rd = ReturnsResult(; X = X, nx = string.(1:10))
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = Dict("verbose" => false))
    pr = prior(EmpiricalPrior(), X)
    me = MedianExpectedReturns()
    mu_me = vec(mean(me, X))
    # The median is not linear, so `mu_me` is a different vector from `pr.mu`.
    function w_of(rt)
        return optimise(MeanRisk(; r = Variance(), obj = MaximumUtility(),
                                 opt = JuMPOptimiser(; pe = pr, slv = slv, ret = rt)), rd).w
    end

    # With no set, the Deferred Quantity gives what its own vector gives, and it outranks
    # the prior.
    w_def = w_of(ArithmeticReturn(; mu = me))
    @test isapprox(w_def, w_of(ArithmeticReturn(; mu = mu_me)))
    @test !isapprox(w_def, w_of(ArithmeticReturn()))

    # A set that carries its centre outranks the `rt.mu` rung, whichever state that rung is
    # in. `pr.mu` is sorted descending, so the reversed vector is a different ranking.
    carried = L1UncertaintySet(; eps = 1e-4, mu = reverse(pr.mu))
    bare = L1UncertaintySet(; eps = 1e-4)
    w_carried = w_of(ArithmeticReturn(; ucs = carried))
    @test isapprox(w_of(ArithmeticReturn(; ucs = carried, mu = me)), w_carried)
    @test isapprox(w_of(ArithmeticReturn(; ucs = carried, mu = mu_me)), w_carried)
    # The carried centre is doing work, so the assertion above is not vacuous.
    @test !isapprox(w_carried, w_of(ArithmeticReturn(; ucs = bare)))

    # A set that carries none falls back to the `rt.mu` rung, and the Deferred Quantity is
    # that rung.
    w_bare_def = w_of(ArithmeticReturn(; ucs = bare, mu = me))
    @test isapprox(w_bare_def, w_of(ArithmeticReturn(; ucs = bare, mu = mu_me)))
    @test !isapprox(w_bare_def, w_of(ArithmeticReturn(; ucs = bare)))
end

@testset "Deferred Quantity: `UncertaintySetVariance` needs no rule of its own" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    rd = ReturnsResult(; X = X, nx = string.(1:5))
    lop = prior(EmpiricalPrior(), X)
    ce = PortfolioOptimisersCovariance()
    w = fill(0.2, 5)

    # The same ladder on the covariance axis. The slot widened with the kernel, and the
    # set's own centre still wins at the point of use, so no rule is added here.
    r = PO.resolve_deferred_quantities(UncertaintySetVariance(; sigma = ce), lop)
    @test r.sigma ≈ cov(ce, X)
    @test factory(UncertaintySetVariance(; sigma = ce), lop).sigma ≈ cov(ce, X)

    ell = sigma_ucs(NormalUncertaintySet(; alg = EllipsoidalUncertaintySetAlgorithm()), rd)
    @test !isnothing(ell.val)
    @test PO.ucs_variance(ell, r.sigma, w) ≈ PO.ucs_variance(ell, lop.sigma .* 3, w)

    # The box covariance route builds its worst case from the bounds alone, so `val` is
    # populated and never read, and the resolved `sigma` is inert there too.
    box = sigma_ucs(NormalUncertaintySet(; alg = BoxUncertaintySetAlgorithm()), rd)
    @test !isnothing(box.val)
    @test PO.ucs_variance(box, r.sigma, w) == PO.ucs_variance(box, lop.sigma .* 3, w)
end

#=
The value-level seam. `expected_risk` takes either a prior result or a bare returns matrix,
and the two are not interchangeable. Given the prior it resolves the measure through
`factory`, so the number it reports is the one the optimiser optimises. Given the matrix it
cannot: that call has no `pr.w` to thread and no factor returns to reach, so resolving there
would use a different rule than the settled one. It refuses instead — #248's shape, the
consumer resolves and the kernel refuses.
=#
@testset "Deferred Quantity: the value-level seam resolves with a prior" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    w = fill(0.2, 5)
    pr = prior(EmpiricalPrior(), X)
    hop = prior(HighOrderPriorEstimator(), X)
    ce = PortfolioOptimisersCovariance(;
                                       ce = Covariance(;
                                                       ce = GeneralCovariance(;
                                                                              ce = StatsBase.SimpleCovariance(;
                                                                                                              corrected = false))))
    me = MedianExpectedReturns()

    # The whole family, one measure per widened slot. Each equals the resolved equivalent,
    # so the seam resolves rather than falling back to the prior's own field.
    for (r, p) in ((Variance(; sigma = ce), pr), (StandardDeviation(; sigma = ce), pr),
                   (UncertaintySetVariance(; sigma = ce), pr), (LowOrderMoment(; mu = me), pr),
                   (HighOrderMoment(; mu = me), pr), (ThirdCentralMoment(; mu = me), pr),
                   (MedianAbsoluteDeviation(; mu = me), pr), (Kurtosis(; kt = Cokurtosis()), pr),
                   (Kurtosis(; mu = me), hop), (Kurtosis(; pe = HighOrderPriorEstimator()), pr),
                   (Skewness(; sk = Coskewness()), pr), (Skewness(; mu = me), hop),
                   (Skewness(; pe = HighOrderPriorEstimator()), pr),
                   (NegativeSkewness(; sk = Coskewness()), pr),
                   (VarianceSkewKurtosis(; pe = HighOrderPriorEstimator()), pr),
                   (VarianceSkewKurtosis(; vr = Variance(; sigma = ce),
                                         sk = Skewness(; sk = Coskewness()),
                                         kt = Kurtosis(; kt = Cokurtosis())), pr),
                   (ValueatRisk(;
                                alg = PortfolioOptimisers.DistributionValueatRisk(; mu = me, sigma = ce)), pr),
                   (ValueatRiskRange(;
                                     alg = PortfolioOptimisers.DistributionValueatRisk(;
                                                                                       pe = EmpiricalPrior())),
                    pr))
        @test expected_risk(r, w, p) ≈ expected_risk(factory(r, p), w, p.X)
    end

    # A bare `Variance` reaches the kernel with `sigma === nothing` unless the seam runs
    # `factory`, so the prior fallback is what makes the unstated state work at all.
    @test expected_risk(Variance(), w, pr) ≈ dot(w, pr.sigma, w)
    @test expected_risk(Variance(; sigma = ce), w, pr) ≈ dot(w, cov(ce, X), w)
    @test !isapprox(expected_risk(Variance(), w, pr),
                    expected_risk(Variance(; sigma = ce), w, pr))

    # `factory` also fills an unstated slot, so the seam and the optimiser now report one
    # number where they used to report two. With fees the two centres genuinely differ:
    # `mu === nothing` centres on the mean of the net portfolio series, and `pr.mu` is
    # gross. The prior arm follows the optimiser; the matrix arm keeps the old centring,
    # because there is no prior there to fill from.
    fees = Fees(; l = 0.01)
    @test expected_risk(LowOrderMoment(), w, pr, fees) ==
          expected_risk(factory(LowOrderMoment(), pr), w, X, fees)
    @test expected_risk(LowOrderMoment(), w, pr, fees) !=
          expected_risk(LowOrderMoment(), w, X, fees)

    # A `ReturnsResult` carries no moments, so it only unwraps its `X` and the measure is
    # left as it is.
    rd = ReturnsResult(; X = X, nx = string.(1:5))
    @test expected_risk(ConditionalValueatRisk(), w, rd) ==
          expected_risk(ConditionalValueatRisk(), w, X)

    # A vector of weight vectors resolves once and maps, rather than resolving per vector.
    ws = [w, reverse(collect(range(0.1, 0.3; length = 5)))]
    @test expected_risk(Variance(; sigma = ce), ws, pr) ==
          [expected_risk(Variance(; sigma = ce), wi, pr) for wi in ws]
end

@testset "Deferred Quantity: the value-level seam refuses without one" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    w = fill(0.2, 5)
    ce = PortfolioOptimisersCovariance()
    me = MedianExpectedReturns()

    # Sweep the family: every widened slot has a refusal path, and the message names the
    # slot that is unresolved rather than the frame that tripped over it.
    for (r, owner, slot) in ((Variance(; sigma = ce), "Variance", "sigma"),
                             (StandardDeviation(; sigma = ce), "StandardDeviation", "sigma"),
                             (UncertaintySetVariance(; sigma = ce), "UncertaintySetVariance", "sigma"),
                             (LowOrderMoment(; mu = me), "LowOrderMoment", "mu"),
                             (HighOrderMoment(; mu = me), "HighOrderMoment", "mu"),
                             (ThirdCentralMoment(; mu = me), "ThirdCentralMoment", "mu"),
                             (MedianAbsoluteDeviation(; mu = me), "MedianAbsoluteDeviation", "mu"),
                             (Kurtosis(; kt = Cokurtosis()), "Kurtosis", "kt"),
                             (Kurtosis(; mu = me), "Kurtosis", "mu"),
                             (Kurtosis(; pe = HighOrderPriorEstimator()), "Kurtosis", "pe"),
                             (Skewness(; sk = Coskewness()), "Skewness", "sk"),
                             (Skewness(; mu = me), "Skewness", "mu"),
                             (Skewness(; pe = HighOrderPriorEstimator()), "Skewness", "pe"),
                             (NegativeSkewness(; sk = Coskewness()), "NegativeSkewness", "sk"),
                             (VarianceSkewKurtosis(; pe = HighOrderPriorEstimator()), "VarianceSkewKurtosis", "pe"),
                             (ValueatRisk(; alg = PortfolioOptimisers.DistributionValueatRisk(; mu = me)),
                              "DistributionValueatRisk", "mu"),
                             (ValueatRiskRange(; alg = PortfolioOptimisers.DistributionValueatRisk(; sigma = ce)),
                              "DistributionValueatRisk", "sigma"),
                             (ValueatRisk(;
                                          alg = PortfolioOptimisers.DistributionValueatRisk(;
                                                                                            pe = EmpiricalPrior())),
                              "DistributionValueatRisk", "pe"))
        err = try
            expected_risk(r, w, X)
        catch e
            e
        end
        @test isa(err, ArgumentError)
        msg = sprint(showerror, err)
        @test occursin("$owner.$slot", msg)
        @test occursin("Deferred Quantity", msg)
        @test occursin("factory(r, pr)", msg)
    end

    # A container is covered by its children's declarations, so a deferred child is named by
    # the child that holds it.
    err = try
        expected_risk(VarianceSkewKurtosis(; vr = Variance(; sigma = ce)), w, X)
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("`Variance.sigma`", sprint(showerror, err))

    # The refusal reaches every value-level consumer, because they all funnel through the
    # same entry.
    @test_throws ArgumentError risk_contribution(Variance(; sigma = ce), w, X)
    @test_throws ArgumentError PortfolioOptimisers.rolling_window_measure(Variance(;
                                                                                   sigma = ce),
                                                                          w, X, nothing, 50)
    @test_throws ArgumentError expected_risk(RiskRatio(; r1 = Variance(; sigma = ce),
                                                       r2 = ConditionalValueatRisk()), w, X)

    # A slot that holds an Estimator by design is not a Deferred Quantity, so a measure that
    # carries one is evaluated rather than refused.
    @test isa(expected_risk(LowOrderMoment(; alg = SecondMoment(; ve = SimpleVariance())),
                            w, X), Number)
    @test isa(expected_risk(Skewness(; ve = SimpleVariance()), w, X), Number)

    # `ve` is the falsification case: a variance estimator **is** a Deferred Quantity by
    # type, so only the per-type declaration keeps it from being refused.
    @test isa(SimpleVariance(), PortfolioOptimisers.DeferredQuantity)
    @test !haskey(PortfolioOptimisers.deferred_slots(Skewness()), :ve)
    @test isa(expected_risk(UncertaintySetVariance(; sigma = cov(X)), w, X), Number)
    @test isa(expected_risk(MedianAbsoluteDeviation(), w, X), Number)
    @test isa(expected_risk(MedianAbsoluteDeviation(; mu = MeanCentering()), w, X), Number)
end

@testset "Deferred Quantity: the value-level seam fits once, not once per difference" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    w = fill(0.2, 5)
    pr = prior(EmpiricalPrior(), X)

    # `risk_contribution` evaluates the measure `2N` times. Resolving inside that loop would
    # refit the covariance once per finite difference, so the seam resolves before it.
    c = _test_CountingPriorEstimator(EmpiricalPrior(), 0)
    rc = risk_contribution(Variance(; sigma = c), w, pr)
    @test c.n == 1
    @test rc ≈ risk_contribution(factory(Variance(; sigma = EmpiricalPrior()), pr), w, X)

    c = _test_CountingPriorEstimator(EmpiricalPrior(), 0)
    expected_risk(Variance(; sigma = c), w, pr)
    @test c.n == 1

    c = _test_CountingPriorEstimator(EmpiricalPrior(), 0)
    expected_risk(Variance(; sigma = c), [w, w, w], pr)
    @test c.n == 1

    # A factor decomposition goes through `risk_contribution`, so it inherits the same rule.
    rd = ReturnsResult(; X = X, nx = string.(1:5), F = X[:, 1:2], nf = string.(1:2))
    c = _test_CountingPriorEstimator(EmpiricalPrior(), 0)
    frc = factor_risk_contribution(Variance(; sigma = c), w, pr; rd = rd)
    @test c.n == 1
    @test frc ≈
          factor_risk_contribution(factory(Variance(; sigma = EmpiricalPrior()), pr), w, X;
                                   rd = rd)
end

@testset "Deferred Quantity: a `VarianceSkewKurtosis` `pe` reaches the `factory` path" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    hop = prior(HighOrderPriorEstimator(), X)

    # The container holds no quantity of its own, so `@fprop` alone reached the children and
    # left `pe` standing — the measure said one thing and computed another. `factory` now
    # fans it out first, in the same order the `JuMP` path already used.
    f = factory(VarianceSkewKurtosis(; pe = HighOrderPriorEstimator()), hop)
    @test isnothing(f.pe)
    @test f.vr.sigma ≈ hop.sigma
    @test f.sk.sk ≈ hop.sk
    @test f.sk.mu ≈ hop.mu
    @test f.kt.kt ≈ hop.kt
    @test f.kt.mu ≈ hop.mu

    # The children keep the `rke = false` the container forced on them.
    @test f.vr.settings.rke === false
    @test f.sk.settings.rke === false
    @test f.kt.settings.rke === false

    # It agrees with the `JuMP` path's resolution point, which is the whole point of adding
    # the second one.
    g = PortfolioOptimisers.resolve_deferred_quantities(VarianceSkewKurtosis(;
                                                                             pe = HighOrderPriorEstimator()),
                                                        hop)
    @test f.vr.sigma ≈ g.vr.sigma
    @test f.sk.sk ≈ g.sk.sk
    @test f.kt.kt ≈ g.kt.kt
end

@testset "Deferred Quantity: container recursion is derived from the declaration" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    lop = prior(EmpiricalPrior(), X)
    ce = PortfolioOptimisersCovariance()
    me = SimpleExpectedReturns()
    R = PO.resolve_deferred_quantities

    # A container declares its children in `deferred_slots` and writes no forwarding method.
    # `ValueatRisk` and `ValueatRiskRange` had one and lost it, so this is the regression.
    v = R(ValueatRisk(; alg = DistributionValueatRisk(; sigma = ce)), lop)
    @test v.alg.sigma ≈ lop.sigma
    vr = R(ValueatRiskRange(; alg = DistributionValueatRisk(; sigma = ce)), lop)
    @test vr.alg.sigma ≈ lop.sigma

    # The four containers that never wrote one now resolve the same way.
    t = R(RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = fill(0.2, 5)),
                                  r = Variance(; sigma = ce)), lop)
    @test t.r.sigma ≈ lop.sigma
    # The rebuild is positional, so the inner constructor re-applies the container's own rule
    # that the tracked measure carries no risk expression of its own.
    @test t.r.settings.rke === false

    g = R(GenericValueatRiskRange(;
                                  loss = ValueatRisk(;
                                                     alg = DistributionValueatRisk(;
                                                                                   sigma = ce)),
                                  gain = ValueatRisk(;
                                                     alg = DistributionValueatRisk(;
                                                                                   mu = me))),
          lop)
    @test g.loss.alg.sigma ≈ lop.sigma
    @test g.gain.alg.mu ≈ lop.mu

    rr = R(RiskRatio(; r1 = Variance(; sigma = ce), r2 = LowOrderMoment(; mu = me)), lop)
    @test rr.r1.sigma ≈ lop.sigma
    @test rr.r2.mu ≈ lop.mu

    er = R(ExpectedReturn(; rt = ArithmeticReturn(; mu = me)), lop)
    @test er.rt.mu ≈ lop.mu

    mrr = R(MeanReturnRiskRatio(; rk = Variance(; sigma = ce)), lop)
    @test mrr.rk.sigma ≈ lop.sigma

    # A vector slot resolves element by element, which is the rule `factory_child` already
    # applies on the other path.
    nrr = R(NonOptimisationRiskRatio(; r1 = [Variance(; sigma = ce), StandardDeviation()],
                                     r2 = [LowOrderMoment(; mu = me)]), lop)
    @test nrr.r1[1].sigma ≈ lop.sigma
    @test isnothing(nrr.r1[2].sigma)
    @test nrr.r2[1].mu ≈ lop.mu

    errr = R(ExpectedReturnRiskRatio(; rt = [ArithmeticReturn(; mu = me)],
                                     rk = [Variance(; sigma = ce),
                                           LowOrderMoment(; mu = me)]), lop)
    @test errr.rt[1].mu ≈ lop.mu
    @test errr.rk[1].sigma ≈ lop.sigma
    @test errr.rk[2].mu ≈ lop.mu

    # A container whose children resolved to themselves is returned unchanged, so the common
    # case allocates nothing.
    for r in (ValueatRisk(), ValueatRiskRange(), RiskRatio(), GenericValueatRiskRange(),
              ExpectedReturn(), MeanReturnRiskRatio(),
              RiskTrackingRiskMeasure(; tr = WeightsTracking(; w = fill(0.2, 5))))
        @test R(r, lop) === r
    end
end

@testset "Deferred Quantity: the derived recursion refuses a declaration with no resolver" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    lop = prior(EmpiricalPrior(), X)

    # The derivation carries container recursion alone. A type that declares a quantity slot
    # and writes no resolver would let the Estimator reach a model builder and be multiplied
    # as though it were a matrix, so the derived method refuses it by name.
    err = try
        PO.resolve_deferred_quantities(_test_DeclaresWithoutResolving(SimpleExpectedReturns()),
                                       lop)
    catch e
        e
    end
    @test isa(err, ArgumentError)
    msg = sprint(showerror, err)
    @test occursin("`_test_DeclaresWithoutResolving.mu`", msg)
    @test occursin("`SimpleExpectedReturns`", msg)
    @test occursin("declares no `resolve_deferred_quantities` method", msg)

    # A declared slot holding a plain value is not refused.
    @test PO.resolve_deferred_quantities(_test_DeclaresWithoutResolving(lop.mu), lop).mu ===
          lop.mu
end

@testset "Deferred Quantity: every resolver has a declaration beside it" begin
    # ADR 0051 pairs the two declarations. The naming half may stand alone — that is a
    # container, whose recursion is derived — but the resolving half never may, or a widened
    # slot resolves where a prior is in hand and passes unnoticed where one is not.
    base(T) = Base.typename(Base.unwrap_unionall(T)).wrapper
    function firstargs(f)
        out = Set{Any}()
        for m in methods(f)
            T = Base.unwrap_unionall(m.sig).parameters[2]
            T === Any && continue
            push!(out, base(T))
        end
        return out
    end
    resolvers = firstargs(PO.resolve_deferred_quantities)
    declarations = firstargs(PO.deferred_slots)
    @test isempty(setdiff(resolvers, declarations))

    # `ArithmeticReturn` was the one violation, and it is reachable now: `ExpectedReturn.rt`
    # carries it into the value-level check.
    @test ArithmeticReturn in resolvers
    @test ArithmeticReturn in declarations

    # The six containers declare and resolve nothing of their own.
    for T in (RiskTrackingRiskMeasure, GenericValueatRiskRange, RiskRatio,
              NonOptimisationRiskRatio, ExpectedReturn, ExpectedReturnRiskRatio)
        @test T in declarations
        @test !(T in resolvers)
    end
end

@testset "Deferred Quantity: the value-level check recurses into a vector slot" begin
    rng = StableRNG(987654321)
    X = randn(rng, 200, 5)
    w = fill(0.2, 5)
    ce = PortfolioOptimisersCovariance()

    # A slot that holds one child was recursed into and a slot that held a vector of them was
    # not, so the refusal landed several frames down instead of naming the slot.
    err = try
        PO.assert_resolved_slots(NonOptimisationRiskRatio(;
                                                          r1 = [StandardDeviation(;
                                                                                  sigma = ce)]))
    catch e
        e
    end
    @test isa(err, ArgumentError)
    @test occursin("`StandardDeviation.sigma`", sprint(showerror, err))

    @test_throws ArgumentError PO.assert_resolved_slots(ExpectedReturnRiskRatio(;
                                                                                rk = [ConditionalValueatRisk(),
                                                                                      Variance(;
                                                                                               sigma = ce)]))
    @test_throws ArgumentError PO.assert_resolved_slots(ExpectedReturn(;
                                                                       rt = [ArithmeticReturn(;
                                                                                              mu = SimpleExpectedReturns())]))
    # A vector of measures that state nothing passes the check and evaluates. The two axes
    # hold measures a bare returns matrix can evaluate; `StandardDeviation()` is not one,
    # because it needs a `sigma` that this entry point does not carry.
    @test isnothing(PO.assert_resolved_slots(NonOptimisationRiskRatio(;
                                                                      r1 = [StandardDeviation()])))
    @test isa(expected_risk(NonOptimisationRiskRatio(; r1 = [ConditionalValueatRisk()],
                                                     r2 = [ConditionalValueatRisk(;
                                                                                  alpha = 0.1)]),
                            w, X), Number)
end

@testset "Deferred Quantity: a container that holds a declaring child declares it" begin
    # `deferred_slots` cannot be derived by walking field values — ADR 0051's falsification
    # witness is `SimpleVariance`, which is a Deferred Quantity by type and stands
    # legitimately in `Skewness.ve`. So the declaration is per type, and this is the gate that
    # catches the container that forgot one: it reads the field's own type bound off the
    # positional constructor and asks whether that bound admits any type that declares slots.
    # Six containers were missing before the recursion was derived.
    tr = WeightsTracking(; w = fill(0.2, 5))
    function instantiate(T)
        for f in (() -> T(), () -> T(; tr = tr), () -> T(; w = fill(0.2, 5)))
            try
                return f()
            catch
            end
        end
        return nothing
    end
    base(T) = Base.typename(Base.unwrap_unionall(T)).wrapper
    declaring = Any[]
    for m in methods(PO.deferred_slots)
        T = Base.unwrap_unionall(m.sig).parameters[2]
        T === Any && continue
        x = instantiate(base(T))
        isnothing(x) || push!(declaring, x)
    end
    @test length(declaring) == 22

    # The inner constructor states one bound per field, in field order.
    function field_bounds(T)
        n = fieldcount(Base.unwrap_unionall(T))
        for m in methods(T)
            ps = Base.unwrap_unionall(m.sig).parameters
            if length(ps) == n + 1 && !Base.isvarargtype(last(ps))
                return ps[2:end]
            end
        end
        return nothing
    end

    pool = vcat(PO.traverse_concrete_subtypes(PO.AbstractBaseRiskMeasure),
                PO.traverse_concrete_subtypes(PO.JuMPReturnsEstimator))
    undeclared = String[]
    unbounded = String[]
    for T in pool
        bounds = field_bounds(T)
        x = instantiate(T)
        @test !isnothing(bounds)
        @test !isnothing(x)
        declared = keys(PO.deferred_slots(x))
        for (f, b) in zip(fieldnames(T), bounds)
            if b === Any
                # An unbounded field says nothing about what it admits, so the gate cannot
                # read it. It is listed rather than skipped silently.
                push!(unbounded, string(nameof(T), ".", f))
            elseif !(f in declared) && any(d -> isa(d, b), declaring)
                push!(undeclared, string(nameof(T), ".", f))
            end
        end
    end
    @test isempty(undeclared)
    # One field in the two families is unbounded, and every sibling writes
    # `settings::RiskMeasureSettings`. A new one must be bounded or added here deliberately.
    @test unbounded == ["RelativisticDrawdownatRisk.settings"]
end
