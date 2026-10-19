#=
Inference on the Sharpe ratio: the Probabilistic Sharpe Ratio, the Minimum Track Record
Length, the expected maximum Sharpe ratio of a search, the Deflated Sharpe Ratio, and the
implied number of independent trials.

The pinned values of the panel below were computed independently in double precision from the
closed forms, with the same sample moments: the Sharpe ratio reads the corrected standard
deviation, and the skewness and the kurtosis read the biased central moments. Every value
agrees to about 1e-15.

The numerical examples of the two sources are pinned too:

  - `sharpe_stderr`, section 3: a monthly record of two years with SR = 0.458, skewness
    -2.448 and kurtosis 10.164 gives PSR(0) = 0.913, 0.982 under normal moments, and 0.953
    with three years.
  - `bailey2014deflated`, section "A numerical example": 100 trials with an annualised
    variance of 0.5 give SR_0 = 0.1132 per day (1.7894 annualised), and a selected series of
    1250 days with SR = 2.5 annualised, skewness -3 and kurtosis 10 gives DSR = 0.9004.
=#

# Deterministic data, so the panel is the same in any language.
function sr_panel()
    t = collect(1.0:500.0)
    x = 0.0008 .+ 0.01 .* (exp.(0.5 .* sin.(0.7 .* t .^ 1.5)) .- 1)
    y = 0.0003 .+ 0.012 .* cos.(1.3 .* t) .+ 0.004 .* sin.(0.11 .* t) .^ 3
    X = hcat(x, y, copy(x), copy(y), fill(0.001, 500))
    # Column 3 misses every third observation, column 4 keeps two observations, and
    # column 5 is constant.
    X[1:3:end, 3] .= NaN
    X[3:end, 4] .= NaN
    srs = 0.02 .+ 0.05 .* sin.((1:60) .* 0.9)
    return x, y, X, srs
end

@testset "expected_max_sharpe_ratio: the approximation and its floor" begin
    @test isapprox(expected_max_sharpe_ratio(2, 0.5, 0.0), 0.25987767214029694;
                   rtol = 1e-13)
    @test isapprox(expected_max_sharpe_ratio(10, 0.5, 0.3), 1.087299150672875; rtol = 1e-13)
    @test isapprox(expected_max_sharpe_ratio(25.7, 0.5), 1.004419945164259; rtol = 1e-13)
    @test isapprox(expected_max_sharpe_ratio(1000, 0.05, -0.1), 0.06275607568263616;
                   rtol = 1e-12)
    @test isapprox(expected_max_sharpe_ratio(1.5, 1.0), 0.21588797092618767; rtol = 1e-13)
    # The example of the source.
    @test isapprox(expected_max_sharpe_ratio(100, 1.0), 2.5306; atol = 5e-5)
    @test isapprox(expected_max_sharpe_ratio(100, sqrt(0.5 / 250)), 0.1132; atol = 5e-5)
    @test isapprox(expected_max_sharpe_ratio(100, sqrt(0.5 / 250)) * sqrt(250), 1.7894;
                   atol = 5e-5)
    # One trial, or trials with no dispersion, leave no selection to correct for.
    @test expected_max_sharpe_ratio(1, 0.5) == 0
    @test expected_max_sharpe_ratio(50, 0.0) == 0
    @test expected_max_sharpe_ratio(1, 0.5, 0.2) == 0.2
    @test expected_max_sharpe_ratio(50, 0.0, 0.2) == 0.2
    # The benchmark grows with the trials, scales with the dispersion and shifts with the mean.
    @test expected_max_sharpe_ratio(100, 0.5) > expected_max_sharpe_ratio(10, 0.5)
    @test isapprox(expected_max_sharpe_ratio(10, 1.0) / 2,
                   expected_max_sharpe_ratio(10, 0.5))
    @test isapprox(expected_max_sharpe_ratio(10, 0.5, 0.3) -
                   expected_max_sharpe_ratio(10, 0.5), 0.3)
    @test expected_max_sharpe_ratio(25, 0.5) <
          expected_max_sharpe_ratio(25.7, 0.5) <
          expected_max_sharpe_ratio(26, 0.5)
    @test expected_max_sharpe_ratio(10.0, 0.5) == expected_max_sharpe_ratio(10, 0.5)
    #=
    The maximum of N >= 1 draws is at least one draw, so its expectation is at least the
    mean. The approximation falls below the mean for N < 1.2836: at N = 1.1 it gives
    -0.3176 sigma, and the exact expectation of the maximum of 1.1 standard normal draws,
    the integral of x N Phi(x)^(N - 1) phi(x), is +0.0848. The floor answers 0, which is
    nearer the truth. Above the crossing the floor does not act.
    =#
    for n in (1.01, 1.1, 1.2, 1.28)
        @test expected_max_sharpe_ratio(n, 1.0, 0.1) == 0.1
    end
    @test expected_max_sharpe_ratio(1.29, 1.0) > 0
    # The types follow the arguments.
    @test expected_max_sharpe_ratio(10, 0.5f0) isa Float32
    @test expected_max_sharpe_ratio(big(10), big(0.5)) isa BigFloat
    @test_throws DomainError expected_max_sharpe_ratio(0, 0.5)
    @test_throws DomainError expected_max_sharpe_ratio(0.5, 0.5)
    @test_throws DomainError expected_max_sharpe_ratio(Inf, 0.5)
    @test_throws DomainError expected_max_sharpe_ratio(NaN, 0.5)
    @test_throws DomainError expected_max_sharpe_ratio(10, -0.1)
    @test_throws DomainError expected_max_sharpe_ratio(10, NaN)
    @test_throws DomainError expected_max_sharpe_ratio(10, Inf)
    @test_throws PortfolioOptimisers.IsNonFiniteError expected_max_sharpe_ratio(10, 0.5,
                                                                                NaN)
end

@testset "probabilistic_sharpe_ratio: the pinned panel" begin
    x, y, X, srs = sr_panel()
    pins = Dict((0.0, 0.0) => [1.0, 0.7951196594394141, 0.9999999996952144],
                (0.05, 0.0) =>
                    [0.9999999999996273, 0.38495975967961193, 0.9999999128061138],
                (0.0, 0.0005) =>
                    [0.9999998215083528, 0.3275220081716762, 0.9997283490346746],
                (-0.02, 0.0002) =>
                    [0.9999999999999671, 0.7771742228846319, 0.9999999768922784])
    for ((b, r), v) in pins
        psr = probabilistic_sharpe_ratio(X; bsr = b, rf = r)
        @test length(psr) == 5
        @test isapprox(psr[1:3], v; rtol = 1e-12)
        # Two observations and a constant series have no defined ratio.
        @test all(isnan, psr[4:5])
        # The matrix method is the vector method on each column without its `NaN` entries.
        for j in 1:5
            c = filter(!isnan, X[:, j])
            @test isequal(psr[j], probabilistic_sharpe_ratio(c; bsr = b, rf = r))
        end
        @test psr[1] == probabilistic_sharpe_ratio(x; bsr = b, rf = r)
    end
    # The risk-free rate enters through the excess mean.
    @test isapprox(probabilistic_sharpe_ratio(y; rf = 0.0005),
                   probabilistic_sharpe_ratio(y .- 0.0005))
    # A higher benchmark is harder to clear.
    @test probabilistic_sharpe_ratio(y; bsr = 0.05) < probabilistic_sharpe_ratio(y)
    # A zero Sharpe ratio is a valid input, and it meets a zero benchmark with probability 1/2.
    @test probabilistic_sharpe_ratio([-0.01, 0.0, 0.01]) == 0.5
end

@testset "probabilistic_sharpe_ratio: the examples of the sources" begin
    #=
    The PSR of the first source, from its moments. `sharpe_ratio_variance_factor` takes the
    EXCESS kurtosis, so the raw 10.164 enters as 7.164.
    =#
    Phi(z) = PortfolioOptimisers.Distributions.cdf(PortfolioOptimisers.Distributions.Normal(),
                                                   z)
    psr(sr, n, g3, g4) = Phi(sr * sqrt(n - 1) /
                             sqrt(PortfolioOptimisers.sharpe_ratio_variance_factor(sr, g3,
                                                                                   g4 - 3)))
    @test isapprox(psr(0.458, 24, -2.448, 10.164), 0.913; atol = 5e-4)
    @test isapprox(psr(0.458, 24, 0.0, 3.0), 0.982; atol = 5e-4)
    @test isapprox(psr(0.458, 36, -2.448, 10.164), 0.953; atol = 1e-3)
    #=
    The example of the second source on a series. Skewness -3 and kurtosis 10 lie on the
    bound kurtosis = skewness^2 + 1, which only a distribution on two points reaches, so
    the series has two values: 105 of one and 1145 of the other come within 0.004 of both
    moments. Standardised and shifted, its Sharpe ratio is 2.5 / sqrt(250) per day.
    =#
    base = vcat(zeros(105), ones(1145))
    two = (base .- mean(base)) ./ std(base) .+ 2.5 / sqrt(250)
    @test isapprox(skewness(two), -3; atol = 1e-3)
    @test isapprox(kurtosis(two) + 3, 10; atol = 1e-2)
    sr0 = expected_max_sharpe_ratio(100, sqrt(0.5 / 250))
    @test isapprox(probabilistic_sharpe_ratio(two; bsr = sr0), 0.900405293333958;
                   rtol = 1e-12)
    @test isapprox(probabilistic_sharpe_ratio(two; bsr = sr0), 0.9004; atol = 5e-5)
end

@testset "probabilistic_sharpe_ratio: an undefined ratio is NaN" begin
    for x in (Float64[], [0.01], [0.01, 0.02], [NaN, NaN, NaN], [NaN, 0.01, NaN],
              [0.01, NaN, 0.02], [0.0, 0.0, 0.0], fill(0.001, 250), fill(1e-7, 250),
              fill(100.0, 250))
        @test isnan(probabilistic_sharpe_ratio(x))
    end
    for n in 0:2
        @test all(isnan, probabilistic_sharpe_ratio(zeros(n, 3)))
    end
    # A matrix with no columns gives an empty vector of the type of a value, not of `Any`.
    @test probabilistic_sharpe_ratio(zeros(5, 0)) isa Vector{Float64}
    @test isempty(probabilistic_sharpe_ratio(zeros(5, 0)))
    @test minimum_track_record_length(zeros(Float32, 5, 0); alpha = 0.05f0) isa
          Vector{Float32}
    @test deflated_sharpe_ratio(zeros(5, 0), [0.1, 0.2]) isa Vector{Float64}
    # A series with a very small but real dispersion keeps its value.
    quiet = 1e-8 .* sin.(1:250) .+ 1e-9
    @test 0 <= probabilistic_sharpe_ratio(quiet) <= 1
    @test_throws PortfolioOptimisers.IsNonFiniteError probabilistic_sharpe_ratio(quiet;
                                                                                 bsr = Inf)
    @test_throws PortfolioOptimisers.IsNonFiniteError probabilistic_sharpe_ratio(quiet;
                                                                                 rf = NaN)
end

@testset "probabilistic_sharpe_ratio: the type follows the data" begin
    _, y, _, _ = sr_panel()
    @test probabilistic_sharpe_ratio(y) isa Float64
    @test probabilistic_sharpe_ratio(Float32.(y)) isa Float32
    @test isapprox(probabilistic_sharpe_ratio(Float32.(y)), probabilistic_sharpe_ratio(y);
                   rtol = 1e-5)
    @test probabilistic_sharpe_ratio(big.(y)) isa BigFloat
    @test isapprox(probabilistic_sharpe_ratio(big.(y)), probabilistic_sharpe_ratio(y);
                   rtol = 1e-12)
    # Integer data widens through the square root of the standard deviation.
    @test probabilistic_sharpe_ratio(round.(Int, 1000 .* y)) isa Float64
end

@testset "minimum_track_record_length" begin
    x, y, X, _ = sr_panel()
    PO = PortfolioOptimisers
    # At the length, the Probabilistic Sharpe Ratio of the same moments is 1 - alpha.
    for (b, a) in ((0.0, 0.05), (0.01, 0.05), (0.02, 0.1), (0.0, 0.01))
        n = minimum_track_record_length(y; bsr = b, alpha = a)
        _, sr, v = PO.sharpe_ratio_moments(y, 0)
        z = (sr - b) * sqrt(n - 1) / sqrt(v)
        @test isapprox(PO.Distributions.cdf(PO.Distributions.Normal(), z), 1 - a)
    end
    #=
    The example of the first source, under daily normal returns: an annualised Sharpe
    ratio of 2 needs 2.73 years to exceed 1 at 95 %. Skewness 0 and kurtosis 3 make the
    variance factor 1 + SR^2 / 2.
    =#
    sr, b = 2 / sqrt(252), 1 / sqrt(252)
    v = PO.sharpe_ratio_variance_factor(sr, 0, 0)
    z = PO.Distributions.quantile(PO.Distributions.Normal(), 0.95)
    @test isapprox((1 + v * (z / (sr - b))^2) / 252, 2.73; atol = 5e-3)
    # A Sharpe ratio at or below the benchmark never clears it.
    _, sry, _ = PO.sharpe_ratio_moments(y, 0)
    @test minimum_track_record_length(y; bsr = sry) == Inf
    @test minimum_track_record_length(y; bsr = 0.5) == Inf
    # The length rises with the benchmark and with the confidence.
    @test minimum_track_record_length(y; bsr = 0.01) > minimum_track_record_length(y)
    @test minimum_track_record_length(y; alpha = 0.01) > minimum_track_record_length(y)
    # The risk-free rate enters through the excess mean.
    @test isapprox(minimum_track_record_length(y; rf = 0.0001),
                   minimum_track_record_length(y .- 0.0001))
    # The matrix method is the vector method on each column.
    n = minimum_track_record_length(X)
    @test length(n) == 5
    @test n[1] == minimum_track_record_length(x)
    @test n[3] == minimum_track_record_length(filter(!isnan, X[:, 3]))
    @test all(isnan, n[4:5])
    @test isnan(minimum_track_record_length(Float64[]))
    @test minimum_track_record_length(Float32.(y); alpha = 0.05f0) isa Float32
    @test_throws DomainError minimum_track_record_length(y; alpha = 0)
    @test_throws DomainError minimum_track_record_length(y; alpha = 0.5)
    @test_throws DomainError minimum_track_record_length(y; alpha = 0.7)
    @test_throws PO.IsNonFiniteError minimum_track_record_length(y; bsr = NaN)
    @test_throws PO.IsNonFiniteError minimum_track_record_length(y; rf = Inf)
end

@testset "deflated_sharpe_ratio" begin
    x, y, X, srs = sr_panel()
    pins = Dict(0.0 => [0.9999999999150345, 0.1488923431523469, 0.9999976642150865],
                0.0005 => [0.9992285680767888, 0.01037771783142452, 0.9702887559569111])
    for (r, v) in pins
        dsr = deflated_sharpe_ratio(X, srs; rf = r)
        @test isapprox(dsr[1:3], v; rtol = 1e-12)
        @test all(isnan, dsr[4:5])
        # It is the Probabilistic Sharpe Ratio against the expected maximum of the search.
        sr0 = expected_max_sharpe_ratio(length(srs), std(srs))
        @test isequal(dsr, probabilistic_sharpe_ratio(X; bsr = sr0, rf = r))
        @test dsr[2] == deflated_sharpe_ratio(y, srs; rf = r)
    end
    # One trial, or trials with no dispersion, leave the Probabilistic Sharpe Ratio.
    @test isapprox(deflated_sharpe_ratio(y, [0.3]), 0.7951196594394144; rtol = 1e-12)
    @test deflated_sharpe_ratio(y, [0.3]) == probabilistic_sharpe_ratio(y)
    @test deflated_sharpe_ratio(y, [0.3, 0.3]) == probabilistic_sharpe_ratio(y)
    # A `NaN` trial is dropped.
    @test deflated_sharpe_ratio(y, vcat(srs, NaN)) == deflated_sharpe_ratio(y, srs)
    # Fewer independent trials lower the benchmark and raise the ratio.
    @test deflated_sharpe_ratio(y, srs; ntrials = 10) > deflated_sharpe_ratio(y, srs)
    @test deflated_sharpe_ratio(y, srs; ntrials = 60) == deflated_sharpe_ratio(y, srs)
    @test deflated_sharpe_ratio(y, srs; ntrials = 1) == probabilistic_sharpe_ratio(y)
    @test deflated_sharpe_ratio(Float32.(y), Float32.(srs)) isa Float32
    @test_throws PortfolioOptimisers.IsEmptyError deflated_sharpe_ratio(y, Float64[])
    @test_throws PortfolioOptimisers.IsEmptyError deflated_sharpe_ratio(y, [NaN])
    @test_throws PortfolioOptimisers.IsNonFiniteError deflated_sharpe_ratio(y, [Inf])
    @test_throws PortfolioOptimisers.IsNonFiniteError deflated_sharpe_ratio(y,
                                                                            [0.05, 0.1,
                                                                             -Inf])
    @test_throws DomainError deflated_sharpe_ratio(y, srs; ntrials = 0.5)
end

@testset "deflated_sharpe_ratio: the winner of a search with no skill" begin
    rng = StableRNG(0)
    trials = 0.01 .* randn(rng, 1000, 40)
    srs = vec(mean(trials; dims = 1) ./ std(trials; dims = 1))
    winner = trials[:, argmax(srs)]
    dsr = deflated_sharpe_ratio(winner, srs)
    @test 0 <= dsr < 0.95
    # It looks significant when the search behind it is ignored.
    @test deflated_sharpe_ratio(winner, [maximum(srs)]) > dsr
end

@testset "implied_independent_trials" begin
    @test implied_independent_trials(10, 0.0) == 10
    @test implied_independent_trials(10, 1.0) == 1
    @test implied_independent_trials(10, 0.25) == 7.75
    @test implied_independent_trials(1, 0.3) == 1
    # The least mean correlation of a correlation matrix gives M + 1.
    @test isapprox(implied_independent_trials(5, -0.249), 5.996)
    # The matrix method reads the mean of the entries off the diagonal.
    C = [1.0 0.2 0.4; 0.2 1.0 0.6; 0.4 0.6 1.0]
    @test isapprox(implied_independent_trials(C), 0.4 + 0.6 * 3)
    @test implied_independent_trials(ones(1, 1)) == 1
    @test_throws DomainError implied_independent_trials(0, 0.1)
    @test_throws DomainError implied_independent_trials(10, 1.1)
    @test_throws DomainError implied_independent_trials(10, -0.2)
    # The source takes a positive definite matrix, so the bound itself is refused, even from
    # a singular correlation matrix that reaches it.
    @test_throws DomainError implied_independent_trials(2, -1.0)
    @test_throws DomainError implied_independent_trials([1.0 -1.0; -1.0 1.0])
    @test_throws DomainError implied_independent_trials(zeros(0, 0))
    @test_throws DimensionMismatch implied_independent_trials(ones(2, 3))
end
