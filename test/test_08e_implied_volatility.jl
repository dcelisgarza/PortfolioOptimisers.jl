#=
`ImpliedVolatility` had no behavioural test, and all three of its defects survived because
of that. `test_08d_dims_guard.jl` reaches the type, but its guard runs before the estimator
computes anything, so a `cov` that throws still passed there.

The three defects, each of which a testset below now pins:

  1. `realised_vol` reshaped its window into a `ws × chunk × N` array and asked
     `Statistics.std(::AbstractVarianceEstimator, ...)` for it. No method takes a 3-D array,
     so `cov(ImpliedVolatility(), X; iv = IV)` -- the type's own default algorithm -- raised
     a `MethodError` for every input.

  2. `cov` and `cor` called `predict_realised_vols(ce.alg, X, iv, ivpa)`, where the method
     declares `(alg, iv, X, ivpa)`. The two matrices were swapped, so the premium algorithm
     returned the last row of the *returns* as a volatility, sign and all.

  3. `cov` called `cov2cor!` on a matrix that was already a correlation, where the
     documented `Sigma = diag(sigma) rho diag(sigma)` needs `cor2cov!`. The returned
     "covariance" carried a unit diagonal whatever the implied volatilities were.

Three kinds of check run here, and the first two are what catch a drift no invariant sees:

  - REFERENCE VALUES, as the rest of the covariance family does it: nine configurations
    against `assets/ImpliedVolatility.csv.gz`, one flattened covariance per column.

  - AN INDEPENDENT FIT. The log-linear regression is re-derived here from the normal
    equations through `\`, not through `GLM`, and the two must agree. This is the assertion
    the reference implementation makes of itself -- rebuild the prediction from the
    coefficients, the last implied volatility and the last window's realised volatility --
    so one solver's bug cannot hide behind the other's.

  - INVARIANTS: the documented `Sigma = diag(sigma) rho diag(sigma)`, a correlation that
    does not move when the volatilities do, and the window alignment of `realised_vol`
    against `implied_vol`.

THE IMPLIED VOLATILITY MATRIX IS SYNTHETIC AND DETERMINISTIC. The library ships no option
data, so `make_implied_vol` below derives one from the returns themselves: a 20-day trailing
realised volatility, annualised, lifted by a fixed 1.15 volatility risk premium and
perturbed by a seeded log-normal shock. The result runs from about 9 % to about 120 %
annualised, and the predicted realised volatilities it produces sit between 0.0093 and
0.049 daily, which is the range the reference implementation's own fixture reaches. THE
RECIPE MUST NOT CHANGE without regenerating `assets/ImpliedVolatility.csv.gz`.
=#
using PortfolioOptimisers, Statistics, StatsBase

# Two custom variance estimators, to pin both sides of the `obs_weights_view` fallback. They
# live at top level because a `@testset` body becomes a function and cannot host a `struct`.
# `OptedOut` implements no method, so its full-sample weights meet a window and raise, which
# is what every weighted estimator did before the verb existed. `OptedIn` implements one.
struct OptedOutVariance{T} <: PortfolioOptimisers.AbstractVarianceEstimator
    w::T
end
struct OptedInVariance{T} <: PortfolioOptimisers.AbstractVarianceEstimator
    w::T
end
function Statistics.std(ve::Union{OptedOutVariance, OptedInVariance},
                        X::PortfolioOptimisers.MatNum; dims::Int = 1, kwargs...)
    return Statistics.std(X, ve.w, dims; corrected = false)
end
function PortfolioOptimisers.obs_weights_view(ve::OptedInVariance, i)
    return OptedInVariance(PortfolioOptimisers.nothing_scalar_array_getindex(ve.w, i))
end

@testset "Implied volatility" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, StableRNGs, StatsBase,
          Statistics, LinearAlgebra

    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end],
                           TimeArray(CSV.File(joinpath(@__DIR__, "./assets/Factors.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    X = rd.X
    T, N = size(X)

    # A 20-day trailing realised volatility, annualised, times a 1.15 volatility risk
    # premium, times a seeded log-normal shock. The leading rows that have no full window
    # take the first full window's value, so no row is built from a single observation --
    # the `log` of such a value is an outlier that would dominate the regression.
    function make_implied_vol(X; w = 20, prem = 1.15, noise = 0.10, seed = 987654321)
        rng = StableRNG(seed)
        T, N = size(X)
        roll = similar(X)
        for t in w:T
            roll[t, :] = vec(Statistics.std(view(X, (t - w + 1):t, :); dims = 1))
        end
        for t in 1:(w - 1)
            roll[t, :] = roll[w, :]
        end
        return (roll * sqrt(252) * prem) .* exp.(noise * randn(rng, T, N))
    end

    iv = make_implied_vol(X)
    ew = eweights(1:T, inv(T); scale = true)
    # `Weights` refuses bias correction, and `SimpleVariance` corrects by default, so a
    # weighted case needs `AnalyticWeights` unless it also passes `corrected = false`. That
    # rule is `StatsBase`'s and it holds over the whole sample too, not only over a window.
    aw = AnalyticWeights(collect(ew))
    ivpav = collect(range(1.05, 1.35; length = N))

    @testset "Reference values" begin
        # One entry per column of `assets/ImpliedVolatility.csv.gz`, in order.
        ces = [(ImpliedVolatility(), (;)),
               (ImpliedVolatility(; alg = ImpliedVolatilityRegression(; ws = 30)), (;)),
               (ImpliedVolatility(; alg = ImpliedVolatilityRegression(; ws = 25), af = 63),
                (;)),
               (ImpliedVolatility(;
                                  alg = ImpliedVolatilityRegression(;
                                                                    ve = SimpleVariance(;
                                                                                        corrected = false))),
                (;)), (ImpliedVolatility(; ce = SpearmanCovariance()), (;)),
               (ImpliedVolatility(;
                                  mp = MatrixProcessing(;
                                                        dn = Denoise(;
                                                                     alg = SpectralDenoise()))),
                (;)),
               (ImpliedVolatility(; alg = ImpliedVolatilityPremium()), (; ivpa = 1.2)),
               (ImpliedVolatility(; alg = ImpliedVolatilityPremium(), af = 63),
                (; ivpa = ivpav)),
               (ImpliedVolatility(; alg = ImpliedVolatilityPremium(),
                                  ce = KendallCovariance()), (; ivpa = ivpav))]
        df = CSV.read(joinpath(@__DIR__, "./assets/ImpliedVolatility.csv.gz"), DataFrame)
        @test length(ces) == size(df, 2)
        for (i, (ce, kwargs)) in pairs(ces)
            sigma = cov(ce, X; iv = iv, kwargs...)
            rho = cor(ce, X; iv = iv, kwargs...)
            # A covariance whose base correlation is recoverable from it.
            @test isapprox(StatsBase.cov2cor(copy(sigma)), rho)
            # Every configuration returns a usable covariance, never a near-singular one.
            @test isposdef(sigma)
            success = isapprox(vec(sigma), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(sigma), df[!, i])
            end
            @test success
        end
    end

    @testset "The regression reproduces an independent least-squares fit" begin
        # The estimator fits through `GLM`. This rebuilds the same model from the normal
        # equations and predicts from the coefficients by hand.
        function hand_predict(ve, X, iv, ws, af)
            T, N = size(X)
            chunk = div(T, ws)
            offset = T - chunk * ws
            rv = reduce(vcat,
                        [Statistics.std(ve,
                                        view(X,
                                             (offset + (c - 1) * ws + 1):(offset + c * ws),
                                             :); dims = 1) for c in 1:chunk])
            ivw = (iv / sqrt(af))[(T - (chunk - 1) * ws):ws:T, :]
            lrv, liv = log.(rv), log.(ivw)
            out = Vector{Float64}(undef, N)
            for i in 1:N
                D = [ones(chunk - 1) liv[1:(chunk - 1), i] lrv[1:(chunk - 1), i]]
                b = D \ lrv[2:chunk, i]
                out[i] = exp(b[1] + b[2] * liv[chunk, i] + b[3] * lrv[chunk, i])
            end
            return out
        end

        for (ve, ws, af) in
            [(SimpleVariance(), 20, 252), (SimpleVariance(; corrected = false), 30, 252),
             (SimpleVariance(), 25, 63)]
            alg = ImpliedVolatilityRegression(; ve = ve, ws = ws)
            got = PortfolioOptimisers.predict_realised_vols(alg, iv / sqrt(af), X, nothing)
            @test isapprox(got, hand_predict(ve, X, iv, ws, af))
            # The prediction is the diagonal of the covariance the estimator returns.
            sigma = cov(ImpliedVolatility(; alg = alg, af = af), X; iv = iv)
            @test isapprox(sqrt.(diag(sigma)), got)
        end
    end

    @testset "ImpliedVolatility reproduces its documented covariance" begin
        rho = cor(Covariance(), X)
        for (alg, kwargs) in [(ImpliedVolatilityRegression(), (;)),
                              (ImpliedVolatilityPremium(), (; ivpa = 1.2))]
            ce = ImpliedVolatility(; alg = alg)
            sigma = cov(ce, X; iv = iv, kwargs...)
            sd = PortfolioOptimisers.predict_realised_vols(alg, iv / sqrt(ce.af), X,
                                                           get(kwargs, :ivpa, nothing))
            # The diagonal is the predicted realised volatility, never a unit.
            @test isapprox(sqrt.(diag(sigma)), sd)
            @test all(>(0), sd)
            # Sigma = diag(sigma) rho diag(sigma).
            @test isapprox(sigma, Diagonal(sd) * rho * Diagonal(sd))
            # A correlation does not depend on the volatilities that scale it.
            @test isapprox(cor(ce, X; iv = iv, kwargs...), rho)
        end
    end

    @testset "ImpliedVolatilityPremium divides the latest implied volatility" begin
        ce = ImpliedVolatility(; alg = ImpliedVolatilityPremium())
        for ivpa in (1.2, ivpav)
            sd = sqrt.(diag(cov(ce, X; iv = iv, ivpa = ivpa)))
            @test isapprox(sd, vec(iv[end, :]) ./ sqrt(ce.af) ./ ivpa)
        end
        # `af` rescales the whole diagonal by sqrt(af).
        ce2 = ImpliedVolatility(; alg = ImpliedVolatilityPremium(), af = 63)
        @test isapprox(sqrt.(diag(cov(ce2, X; iv = iv, ivpa = 1.2))) * sqrt(63),
                       sqrt.(diag(cov(ce, X; iv = iv, ivpa = 1.2))) * sqrt(252))
    end

    @testset "realised_vol and implied_vol read the same windows" begin
        for ws in (20, 30, 25)
            chunk = div(T, ws)
            offset = T - chunk * ws
            rv = PortfolioOptimisers.realised_vol(SimpleVariance(), X, ws)
            ivw = PortfolioOptimisers.implied_vol(iv, ws)
            @test size(rv) == (chunk, N) == size(ivw)
            for c in 1:chunk
                rows = (offset + (c - 1) * ws + 1):(offset + c * ws)
                @test isapprox(rv[c, :], vec(std(view(X, rows, :); dims = 1)))
                # Window `c` ends at the row `implied_vol` samples for the same window.
                @test ivw[c, :] == iv[last(rows), :]
            end
            # A window count that does not divide `T` drops the OLDEST rows, never the
            # newest: the last window always ends at the last observation.
            @test ivw[chunk, :] == iv[T, :]
        end
    end

    @testset "ImpliedVolatility rejects what it cannot compute" begin
        # A window of 2 or fewer observations carries no usable standard deviation.
        @test_throws DomainError ImpliedVolatilityRegression(; ws = 2)
        # The regression needs more than two windows: two of them leave one training row.
        @test_throws DomainError cov(ImpliedVolatility(;
                                                       alg = ImpliedVolatilityRegression(;
                                                                                         ws = 126)),
                                     X; iv = iv)
        # Three windows is the smallest the fit accepts, and it stays finite there. This is
        # the reference implementation's own floor of three folds.
        @test all(isfinite,
                  diag(cov(ImpliedVolatility(;
                                             alg = ImpliedVolatilityRegression(; ws = 84)),
                           X; iv = iv)))
        # The premium algorithm has no default factor.
        @test_throws ArgumentError cov(ImpliedVolatility(;
                                                         alg = ImpliedVolatilityPremium()),
                                       X; iv = iv)
        # The implied volatility series itself is mandatory.
        @test_throws UndefKeywordError cov(ImpliedVolatility(), X)
        # A custom variance estimator that carries observation weights and implements no
        # `obs_weights_view` method keeps its full-sample weights against a window. This is
        # the documented fallback, and it is what every weighted estimator did before the
        # verb existed.
        @test_throws DimensionMismatch PortfolioOptimisers.realised_vol(OptedOutVariance(aw),
                                                                        X, 20)
    end

    #=
    `realised_vol` used to hand each block the estimator untouched. An observation weight is
    one value per row of the WHOLE sample, so a weighted estimator met a block of `ws` rows
    with a vector of `T` weights and raised -- `ArgumentError` when only the dispersion was
    weighted, `DimensionMismatch` when the mean was too. The field is typed
    `AbstractVarianceEstimator` and nothing rejected such a value, so an estimator that can
    never answer was constructible and only failed at the call.

    `obs_weights_view` slices every weights field to the block. It is deliberately NOT
    `factory`: `factory` writes one incoming value into every `@wprop`-tagged field, which
    would give a weighted dispersion to an estimator that only asked for a weighted mean.
    =#
    @testset "obs_weights_view slices each weights field on its own" begin
        rows = 21:40

        # A weighted dispersion stays a weighted dispersion, and the mean stays unweighted.
        v = PortfolioOptimisers.obs_weights_view(SimpleVariance(; w = aw), rows)
        @test length(v.w) == length(rows)
        @test v.w == aw[rows]
        @test isnothing(v.me.w)

        # A weighted mean stays a weighted mean, and the dispersion stays unweighted. This
        # is the case `factory` cannot express, so the next two lines are the reason the
        # verb exists.
        v = PortfolioOptimisers.obs_weights_view(SimpleVariance(;
                                                                me = SimpleExpectedReturns(;
                                                                                           w = aw)),
                                                 rows)
        @test isnothing(v.w)
        @test length(v.me.w) == length(rows)
        @test !isnothing(PortfolioOptimisers.factory(SimpleVariance(;
                                                                    me = SimpleExpectedReturns(;
                                                                                               w = aw)),
                                                     aw).w)

        # Both weighted, and `corrected` carried through.
        v = PortfolioOptimisers.obs_weights_view(SimpleVariance(;
                                                                me = SimpleExpectedReturns(;
                                                                                           w = aw),
                                                                corrected = false, w = aw),
                                                 rows)
        @test length(v.w) == length(v.me.w) == length(rows)
        @test v.corrected == false

        # Neither weighted: nothing to slice, and nothing gained.
        v = PortfolioOptimisers.obs_weights_view(SimpleVariance(), rows)
        @test isnothing(v.w)
        @test isnothing(v.me.w)

        # A `nothing` mean estimator survives the recursion.
        v = PortfolioOptimisers.obs_weights_view(SimpleVariance(; me = nothing, w = aw),
                                                 rows)
        @test isnothing(v.me)
        @test length(v.w) == length(rows)

        # The wrapper recurses, and `window` is not an observation index, so it is carried
        # through untouched.
        v = PortfolioOptimisers.obs_weights_view(WindowedVariance(;
                                                                  ve = SimpleVariance(;
                                                                                      w = aw),
                                                                  w = aw, window = 15),
                                                 rows)
        @test length(v.w) == length(v.ve.w) == length(rows)
        @test v.window == 15

        # The fallback returns the estimator unchanged, so an estimator that carries no
        # weights and one that implements nothing both behave as they did before.
        c = Covariance()
        @test PortfolioOptimisers.obs_weights_view(c, rows) === c
        o = OptedOutVariance(aw)
        @test PortfolioOptimisers.obs_weights_view(o, rows) === o
    end

    @testset "realised_vol weighs each block by that block's weights" begin
        ws = 20
        chunk = div(T, ws)
        offset = T - chunk * ws

        # Every weighted shape now answers, and each block matches a hand-built weighted
        # standard deviation over that block's own rows and that block's own weights.
        for ve in (SimpleVariance(; w = aw),
                   SimpleVariance(; me = SimpleExpectedReturns(; w = aw), w = aw),
                   SimpleVariance(; me = SimpleExpectedReturns(; w = ew), corrected = false, w = ew), OptedInVariance(aw))
            rv = PortfolioOptimisers.realised_vol(ve, X, ws)
            @test size(rv) == (chunk, N)
            for c in 1:chunk
                rows = (offset + (c - 1) * ws + 1):(offset + c * ws)
                sliced = PortfolioOptimisers.obs_weights_view(ve, rows)
                @test isapprox(rv[c, :], vec(std(sliced, view(X, rows, :); dims = 1)))
                # The block's weights are that block's slice, never the whole sample's.
                @test length(PortfolioOptimisers.obs_weights_view(ve, rows).w) == ws
            end
        end

        # The unweighted path does not move: it is the same estimator, returned unchanged.
        @test isapprox(PortfolioOptimisers.realised_vol(SimpleVariance(), X, ws),
                       PortfolioOptimisers.realised_vol(PortfolioOptimisers.obs_weights_view(SimpleVariance(),
                                                                                             1:T),
                                                        X, ws))
    end

    @testset "ImpliedVolatility accepts a weighted variance estimator" begin
        base = cov(ImpliedVolatility(), X; iv = iv)
        for ve in (SimpleVariance(; w = aw),
                   SimpleVariance(; me = SimpleExpectedReturns(; w = aw), w = aw),
                   SimpleVariance(; me = SimpleExpectedReturns(; w = ew), corrected = false, w = ew), WindowedVariance(; w = aw, window = 15))
            ce = ImpliedVolatility(; alg = ImpliedVolatilityRegression(; ve = ve))
            sigma = cov(ce, X; iv = iv)
            @test all(isfinite, sigma)
            @test isposdef(sigma)
            @test all(>(0), sqrt.(diag(sigma)))
            # The weights change the answer, so they are not being quietly dropped.
            @test !isapprox(sigma, base)
            # The correlation is untouched by the volatility model either way.
            @test isapprox(cor(ce, X; iv = iv), cor(Covariance(), X))
        end
    end
end
