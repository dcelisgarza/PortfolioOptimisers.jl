#=
The plain exponentially weighted family answers a gapped panel the way the oracle does.

`ExpWeightedExpectedReturns`, `ExpWeightedVariance` and `ExpWeightedCovariance` are ports of the
oracle's three plain exponentially weighted members. Each seeds its recursion at
zero, freezes a moment of an asset on its holiday, resets on an inactive period, and divides out the
damping that the cold start costs. The covariance takes the oracle's step, so its raw state is
the oracle's, but it divides each pair by the weight that the pair holds, where the oracle
divides by the per-asset congruence (ADR 0181, amendment of 2026-09-29, #1420). The two agree
where every asset has the same history. On this fixture asset 4 lists late, so its covariances
are the oracle's times `sqrt(W_ii W_jj) / W_ij`, 1.0607, and every other entry is equal. The
oracle is the external implementation itself: `oracle_returns` is an exactly representable
fixture that both languages build bit for bit, so no file is exchanged, and the literals below were
measured by fitting the oracle on it at `half_life = 10`. They pin `centring = PreCentred()`,
the oracle's default, and `ZeroStartCentring()`, the oracle's estimated location, which
starts at zero and is not divided by its weight. The library's default estimated location is
pinned against the hand references at the end of this file (#1507, ADR 0190).

The fixture is 60 observations of 4 assets, and asset 4 lists at observation 31. The measured
parity is exact on every masked path and about `1e-19` on a complete window, where the oracle
takes a matrix-multiply fast path and the port takes the row recursion. Measured on 2026-10-07,
cell by cell: the variances, the standard deviations and the raw state are bit-equal, the means
differ by maxrel 1.4e-16 (maxabs 8.7e-19) and the covariances by 2.2e-16. Every oracle check
takes `rtol = 1e-14`: the measure is one ulp, and the margin is for a host that orders the sums
of the recursion otherwise.

Two families of testset sit beside the parity. The first pins the structural identities the census
of the oracle states, and each is checked in plain Julia rather than against a stored number: the
division of each pair by its weight, the congruence where the histories agree, the positive
semidefiniteness of the raw state, the holiday identity on the raw state, the warm-up mask and the
equal-history identity. The second pins the seam of ADR 0117: a mask-aware estimator overrides the
reduce-and-expand root and answers a young asset that the Coverage Universe drops.
=#
using Test, PortfolioOptimisers, Statistics, LinearAlgebra, StableRNGs
include(joinpath(@__DIR__, "parity_harness.jl"))

# The oracle's answers on the fixture below, at `half_life = 10`.
const EW_MU_MIN1 = [0.001292041816555475, 0.002114893063360849, 0.0033233997239623943,
                    0.006285777102402957]
const EW_MU_MIN40 = [0.001292041816555475, 0.002114893063360849, 0.0033233997239623943, NaN]
const EW_MU_DELIST = [0.001292041816555475, NaN, 0.0033233997239623943,
                      0.006285777102402957]
const EW_VAR_CENTRED = [0.0003153593429894155, 0.0003460166110502017,
                        0.00047822388797051364, 0.000565460111939217]
const EW_VAR_UNCENTRED = [0.0003362325423834515, 0.0003661325253794454,
                          0.0005008949140988661, 0.0005689916618128689]
const EW_VAR_MIN40 = [0.0003153593429894155, 0.0003460166110502017, 0.00047822388797051364,
                      NaN]
const EW_VAR_DELIST = [0.0003153593429894155, NaN, 0.00047822388797051364,
                       0.000565460111939217]
const EW_COV_CENTRED = [0.0003153593429894154 -6.82903394296283e-06 -0.00020955298107695888 -0.0001574566933195339;
                        -6.82903394296283e-06 0.0003460166110502016 3.277154854967503e-05 -0.0002078671138989788;
                        -0.00020955298107695888 3.277154854967503e-05 0.00047822388797051353 1.2140794128293598e-05;
                        -0.0001574566933195339 -0.0002078671138989788 1.2140794128293598e-05 0.0005654601119392171]
const EW_COV_UNCENTRED = [0.00033623254238345147 -1.0202070579665484e-05 -0.00022912357990465665 -0.00017683625902148656;
                          -1.0202070579665484e-05 0.0003661325253794453 2.7708276324016633e-05 -0.00023600959967264636;
                          -0.00022912357990465665 2.7708276324016633e-05 0.0005008949140988659 -7.767021272719638e-06;
                          -0.00017683625902148656 -0.00023600959967264636 -7.767021272719638e-06 0.000568991661812869]
const EW_COV_RAW_STATE = [0.00031043185325520587 -6.722330287604037e-06 -0.00020627871574763144 -0.00014613203796115705;
                           -6.722330287604037e-06 0.0003406101015025423 3.225949310358637e-05 -0.00019291682264353322;
                           -0.00020627871574763144 3.225949310358637e-05 0.0004707516397209744 1.1267599687451917e-05;
                           -0.00014613203796115705 -0.00019291682264353322 1.1267599687451917e-05 0.0004947775979468148]

# The fixture. Every entry is an exactly rounded product of exactly representable doubles, so the
# oracle builds the same matrix with no file exchange.
const EW_BASE = [0.012, -0.005, 0.031, -0.018, 0.007, 0.024, -0.011, 0.002, -0.027, 0.015]
const EW_T = 60
const EW_N = 4
const EW_LIST_T = 30
const EW_DECAY = exp2(-inv(10.0))

function oracle_returns()
    X = Matrix{Float64}(undef, EW_T, EW_N)
    for t in 0:(EW_T - 1), i in 0:(EW_N - 1)
        X[t + 1, i + 1] = EW_BASE[(t + 2i) % 10 + 1] * (1.0 + i / 10.0) +
                          (t * 1e-5) * (i - 2)
    end
    return X
end

function oracle_active_mask()
    amsk = trues(EW_T, EW_N)
    amsk[1:EW_LIST_T, EW_N] .= false
    return amsk
end

function gapped(X, amsk)
    Xg = copy(X)
    Xg[.!amsk] .= NaN
    return Xg
end

function oracle_panel(amsk)
    pf = [PortfolioOptimisers.NumericPanelField(; name = "mcap", vals = ones(size(amsk)...),
                                                omsk = trues(size(amsk)...))]
    return AssetPanel(; pf = pf, amsk = amsk, emsk = copy(amsk))
end

@testset "Exponentially weighted moments: parity with the oracle" begin
    X = oracle_returns()
    amsk = oracle_active_mask()
    Xg = gapped(X, amsk)

    me = ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1)
    @test parity_compare(mean(me, Xg; active_mask = amsk), EW_MU_MIN1; rtol = 1e-14,
                         name = "mu").ok

    me40 = ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 40)
    mu40 = mean(me40, Xg; active_mask = amsk)
    @test isnan(mu40[EW_N])
    @test parity_compare(view(mu40, 1:(EW_N - 1)), view(EW_MU_MIN40, 1:(EW_N - 1));
                         rtol = 1e-14, name = "mu min40").ok

    # With no mask every asset is active, so the leading `NaN` reads as a holiday and the
    # recursion freezes. The gap and the inactive period coincide here, so the two agree.
    @test isequal(mean(me, Xg), mean(me, Xg; active_mask = amsk))

    vc = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centring = PreCentred())
    @test parity_compare(var(vc, Xg; active_mask = amsk), EW_VAR_CENTRED; rtol = 1e-14,
                         name = "var").ok
    @test parity_compare(std(vc, Xg; active_mask = amsk), sqrt.(EW_VAR_CENTRED);
                         rtol = 1e-14, name = "std").ok

    # The oracle's estimated location starts at zero and is not divided by its weight. It is
    # `ZeroStartCentring()` since #1507 (ADR 0190). The default estimated location of the
    # library is pinned against its hand reference at the end of this file.
    vu = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1,
                             centring = ZeroStartCentring())
    @test parity_compare(var(vu, Xg; active_mask = amsk), EW_VAR_UNCENTRED; rtol = 1e-14,
                         name = "var zero start").ok

    v40 = var(ExpWeightedVariance(; decay = EW_DECAY, min_obs = 40,
                                  centring = PreCentred()), Xg; active_mask = amsk)
    @test isnan(v40[EW_N])
    @test parity_compare(view(v40, 1:(EW_N - 1)), view(EW_VAR_MIN40, 1:(EW_N - 1));
                         rtol = 1e-14, name = "var min40").ok

    # The oracle divides by the per-asset congruence, and the port divides each pair by the
    # weight it holds (ADR 0181, 2026-09-29). Asset 4 lists at 31, so its pairs hold less weight
    # than the congruence assumes, and the port's covariance is the oracle's times
    # `sqrt(W_ii W_jj) / W_ij`: measured to maxrel 2.2e-16 cell by cell.
    for (centring, lit) in
        ((PreCentred(), EW_COV_CENTRED), (ZeroStartCentring(), EW_COV_UNCENTRED))
        ce = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centring = centring)
        W = partial_fit!(ce, Xg; active_mask = amsk).cache.weight
        w = sqrt.(diag(W))
        ratio = (w * transpose(w)) ./ W
        @test parity_compare(cov(ce, Xg; active_mask = amsk), lit .* ratio; rtol = 1e-14,
                             name = "cov $(nameof(typeof(centring)))").ok
        @test all(isapprox.(ratio[1:3, 1:3], 1; rtol = 1e-14))
        @test all(isapprox.(ratio[1:3, 4], 1.0606601717798214; rtol = 1e-12))
    end
    cc = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centring = PreCentred())

    # The delisting: asset 2 leaves at observation 46, so its state resets and it is blanked.
    amsk_d = copy(amsk)
    amsk_d[46:end, 2] .= false
    Xd = gapped(X, amsk_d)
    mu_d = mean(me, Xd; active_mask = amsk_d)
    @test isnan(mu_d[2])
    @test parity_compare(mu_d[[1, 3, 4]], EW_MU_DELIST[[1, 3, 4]]; rtol = 1e-14,
                         name = "mu delist").ok
    var_d = var(vc, Xd; active_mask = amsk_d)
    @test isnan(var_d[2])
    @test parity_compare(var_d[[1, 3, 4]], EW_VAR_DELIST[[1, 3, 4]]; rtol = 1e-14,
                         name = "var delist").ok
    cov_d = cov(cc, Xd; active_mask = amsk_d)
    @test all(isnan, view(cov_d, 2, :))
    @test all(isnan, view(cov_d, :, 2))
end

@testset "Exponentially weighted moments: the structural identities" begin
    X = oracle_returns()
    amsk = oracle_active_mask()
    Xg = gapped(X, amsk)

    cc = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centring = PreCentred())
    fitted = partial_fit!(cc, Xg; active_mask = amsk)
    S = fitted.cache.covariance
    W = fitted.cache.weight
    n = fitted.cache.obs_count
    sigma = cov(cc, Xg; active_mask = amsk)

    # The raw state matches the oracle's own, which is what the holiday identity compares.
    @test parity_compare(S, EW_COV_RAW_STATE; rtol = 1e-14, name = "raw state").ok

    # 1. Each pair is divided by the weight it holds, and the weight of a variance is the
    #    per-asset correction `1 - λ^n`. No repair runs on this fixture.
    @test isapprox(diag(W), 1 .- EW_DECAY .^ n; rtol = 1e-14)
    @test isapprox((S ./ W + transpose(S ./ W)) / 2, sigma; atol = 1e-18)

    # 2. Where the histories agree the division is the congruence `D S D`, which moves no
    #    correlation. On the pairs of asset 4, which lists late, the congruence would divide by
    #    more than the weight, so the port's correlation is larger in magnitude.
    D = Diagonal(inv.(sqrt.(1 .- EW_DECAY .^ n)))
    @test isapprox((D * S * D)[1:3, 1:3], sigma[1:3, 1:3]; atol = 1e-18)
    cor_raw = Symmetric(S) ./ sqrt.(diag(S) * transpose(diag(S)))
    cor_out = Symmetric(sigma) ./ sqrt.(diag(sigma) * transpose(diag(sigma)))
    @test all(abs.(cor_out[1:3, 4]) .> abs.(cor_raw[1:3, 4]))

    # 3. Without a holiday the raw state is a sum of positive semidefinite products, and on
    #    this fixture the report is positive definite too.
    @test minimum(eigvals(Symmetric(S))) > 0
    @test minimum(eigvals(Symmetric(sigma))) > 0

    # 4. The holiday identity, on the raw state. One holiday at the last row of asset 1 leaves
    #    its variance, each of its covariances and their weights exactly where a fit that stops
    #    one observation earlier leaves them, and its count does not rise: a holiday carries no
    #    information about the entries of its asset.
    Xh = copy(Xg)
    Xh[end, 1] = NaN
    held = partial_fit!(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                                              centring = PreCentred()), Xh;
                        active_mask = amsk)
    short = partial_fit!(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                                               centring = PreCentred()),
                         view(Xg, 1:(EW_T - 1), :);
                         active_mask = view(amsk, 1:(EW_T - 1), :))
    @test held.cache.covariance[1, :] == short.cache.covariance[1, :]
    @test held.cache.weight[1, :] == short.cache.weight[1, :]
    @test held.cache.obs_count[1] == EW_T - 1
    @test held.cache.obs_count[2] == EW_T

    # 5. The warm-up mask. Asset 4 carries 30 observations, so a threshold above it blanks the
    #    whole row and column and leaves every other entry alone.
    warm = cov(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 31,
                                     centring = PreCentred()), Xg; active_mask = amsk)
    @test all(isnan, view(warm, EW_N, :))
    @test all(isnan, view(warm, :, EW_N))
    @test all(isfinite, view(warm, 1:(EW_N - 1), 1:(EW_N - 1)))

    # 6. The equal-history identity. With no gap every count is the same, so the correction is a
    #    scalar and the answer is the ordinary adjusted exponentially weighted moment.
    w = EW_DECAY .^ ((EW_T - 1):-1:0)
    sw = sum(w)
    @test isapprox(mean(ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1), X),
                   transpose(w) * X / sw |> vec; rtol = 1e-12)
    @test isapprox(var(ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1,
                                           centring = PreCentred()), X),
                   transpose(w) * (X .^ 2) / sw |> vec; rtol = 1e-12)
    Xw = X .* sqrt.(w)
    @test isapprox(cov(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                                             centring = PreCentred()), X),
                   transpose(Xw) * Xw / sw; rtol = 1e-12)

    # 7. The mean's correction is the first power, and the covariance's is the square root. The
    #    two are not interchangeable, so the mean is re-derived here in plain Julia.
    fm = partial_fit!(ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1), Xg;
                      active_mask = amsk)
    @test isapprox(fm.cache.mu ./ (1 .- EW_DECAY .^ fm.cache.obs_count),
                   mean(ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1), Xg;
                        active_mask = amsk); atol = 1e-18)
end

@testset "Exponentially weighted moments: the Asset Panel seam" begin
    X = oracle_returns()
    amsk = oracle_active_mask()
    Xg = gapped(X, amsk)
    pnl = oracle_panel(amsk)

    # The Coverage Universe drops the young asset, so a plain estimator blanks it.
    @test PortfolioOptimisers.coverage_mask(Xg, pnl; dims = 1) == [true, true, true, false]
    @test isnan(mean(SimpleExpectedReturns(), Xg, pnl)[EW_N])

    # A mask-aware estimator overrides the root, reads the panel's own mask, and answers it.
    me = ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1)
    @test isequal(mean(me, Xg, pnl), mean(me, Xg; active_mask = amsk))
    @test isfinite(mean(me, Xg, pnl)[EW_N])

    vc = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centring = PreCentred())
    @test isequal(var(vc, Xg, pnl), var(vc, Xg; active_mask = amsk))
    @test isequal(std(vc, Xg, pnl), std(vc, Xg; active_mask = amsk))
    @test isfinite(var(vc, Xg, pnl)[EW_N])

    cc = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centring = PreCentred())
    @test isequal(cov(cc, Xg, pnl), cov(cc, Xg; active_mask = amsk))
    @test isequal(cor(cc, Xg, pnl), cor(cc, Xg; active_mask = amsk))
    @test isfinite(cov(cc, Xg, pnl)[EW_N, EW_N])

    # A static panel carries no mask, so it is the unmasked path.
    @test isequal(mean(me, Xg, nothing), mean(me, Xg))

    # The regime-adjusted pair is mask-aware too, and it owed the same override.
    ra = RegimeAdjustedExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1)
    @test isequal(cov(ra, Xg, pnl),
                  cov(ra, Xg; estimation_mask = pnl.emsk, active_mask = amsk))
    @test isequal(cor(ra, Xg, pnl),
                  cor(ra, Xg; estimation_mask = pnl.emsk, active_mask = amsk))
    @test isequal(var(ra, Xg, pnl),
                  var(ra, Xg; estimation_mask = pnl.emsk, active_mask = amsk))
    @test isequal(std(ra, Xg, pnl),
                  std(ra, Xg; estimation_mask = pnl.emsk, active_mask = amsk))
    @test isfinite(cov(ra, Xg, pnl)[EW_N, EW_N])
end

@testset "Exponentially weighted moments: the incremental fit" begin
    X = oracle_returns()
    amsk = oracle_active_mask()
    Xg = gapped(X, amsk)

    me = ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1)
    one_row_at_a_time = foldl((c, i) -> partial_fit!(c, view(Xg, i, :);
                                                     active_mask = view(amsk, i, :)),
                              axes(Xg, 1); init = me)
    @test isequal(mean(one_row_at_a_time), mean(me, Xg; active_mask = amsk))

    vc = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centring = PreCentred())
    two_blocks = partial_fit!(partial_fit!(vc, view(Xg, 1:20, :);
                                           active_mask = view(amsk, 1:20, :)),
                              view(Xg, 21:EW_T, :); active_mask = view(amsk, 21:EW_T, :))
    @test isequal(var(two_blocks), var(vc, Xg; active_mask = amsk))
    @test isequal(std(two_blocks), std(vc, Xg; active_mask = amsk))
    @test isequal(var(vc, two_blocks.cache), var(two_blocks))

    cc = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centring = PreCentred())
    fitted = partial_fit!(cc, Xg; active_mask = amsk)
    @test isequal(cov(fitted), cov(cc, Xg; active_mask = amsk))
    @test isequal(cor(fitted), cor(cc, Xg; active_mask = amsk))
    @test isequal(cov(cc, fitted.cache), cov(fitted))

    # A copy shares no array, so a fold on one leaves the other alone.
    original = fitted.cache
    duplicate = copy(original)
    duplicate.covariance[1, 1] = -1.0
    @test original.covariance[1, 1] != duplicate.covariance[1, 1]
    @test copy(one_row_at_a_time.cache).mu == one_row_at_a_time.cache.mu
    vstate = two_blocks.cache
    @test copy(vstate).variance == vstate.variance
end

@testset "Exponentially weighted moments: the refusals and the sibling identity" begin
    X = oracle_returns()
    amsk = oracle_active_mask()
    Xg = gapped(X, amsk)

    # An estimator that has been given no observation carries no state to read.
    @test_throws ArgumentError mean(ExpWeightedExpectedReturns())
    @test_throws ArgumentError var(ExpWeightedVariance())
    @test_throws ArgumentError std(ExpWeightedVariance())
    @test_throws ArgumentError cov(ExpWeightedCovariance())
    @test_throws ArgumentError cor(ExpWeightedCovariance())

    # A variance estimator resolves no cross-asset structure.
    @test_throws MethodError cov(ExpWeightedVariance(), X)

    # The masks are checked against the sample.
    @test_throws DimensionMismatch mean(ExpWeightedExpectedReturns(), X;
                                        active_mask = trues(2, 2))
    @test_throws DimensionMismatch var(ExpWeightedVariance(), X; active_mask = trues(2, 2))
    @test_throws DimensionMismatch cov(ExpWeightedCovariance(), X;
                                       active_mask = trues(2, 2))
    @test_throws DomainError mean(ExpWeightedExpectedReturns(), X; dims = 3)

    # A constructor refuses a decay or a warm-up that is not positive.
    @test_throws DomainError ExpWeightedExpectedReturns(; decay = 0.0)
    @test_throws DomainError ExpWeightedVariance(; min_obs = 0)
    @test_throws DomainError ExpWeightedCovariance(; decay = -1.0)

    # A decay of one or more is refused too. At one the weight `1 - decay` of a new
    # observation is zero and the default warm-up is `round(Int, Inf)`; above one the weight
    # is negative, so a variance goes negative. The regime-adjusted pair derives
    # `regime_min_obs` from `decay` too, so the test states it.
    for decay in (1.0, 1.5)
        for E in (ExpWeightedExpectedReturns, ExpWeightedVariance, ExpWeightedCovariance)
            @test_throws DomainError E(; decay = decay, min_obs = 1)
        end
        for E in (RegimeAdjustedExpWeightedVariance, RegimeAdjustedExpWeightedCovariance)
            @test_throws DomainError E(; decay = decay, min_obs = 1, regime_min_obs = 1)
        end
    end

    # #1355: the keyword defaults read `decay` before the inner constructor checks it, so
    # each reads it through `decay_half_life`, which checks it first. The refusal must be
    # the check's own `DomainError`: at one the old default raised `InexactError`, and below
    # zero `log2` raised a `DomainError` that did not name `decay`.
    refuses(f, sym) =
        try
            f()
            false
        catch e
            e isa DomainError && occursin("0 < $sym < 1 must hold", e.msg)
        end
    for decay in (1.0, 1.5, 0.0, -0.5)
        for E in (ExpWeightedExpectedReturns, ExpWeightedVariance, ExpWeightedCovariance,
                  RegimeAdjustedExpWeightedVariance, RegimeAdjustedExpWeightedCovariance)
            @test refuses(() -> E(; decay = decay), :decay)
        end
        # `regime_decay` and `regime_min_obs` read `decay` too, where `min_obs` is given.
        for E in (RegimeAdjustedExpWeightedVariance, RegimeAdjustedExpWeightedCovariance)
            @test refuses(() -> E(; decay = decay, min_obs = 1), :decay)
        end
        @test refuses(() -> PortfolioOptimisers.decay_half_life(decay), :decay)
        @test refuses(() -> PortfolioOptimisers.decay_half_life(decay, :cor_decay),
                      :cor_decay)
    end
    # `cor_decay` is a decay factor too. At one its weight `1 - cor_decay` is zero, and above
    # one it is negative, so the constructor refuses both, and so does the default `min_obs`.
    for cor_decay in (1.0, 1.5, 0.0)
        @test refuses(() -> RegimeAdjustedExpWeightedCovariance(; decay = 0.9,
                                                                cor_decay = cor_decay),
                      :cor_decay)
        @test refuses(() -> RegimeAdjustedExpWeightedCovariance(; decay = 0.9,
                                                                cor_decay = cor_decay,
                                                                min_obs = 1), :cor_decay)
    end
    # The defaults the half-life states are the ones the constructors held before #1355.
    @test ExpWeightedCovariance().min_obs == 40
    @test RegimeAdjustedExpWeightedVariance().regime_min_obs == 20
    @test RegimeAdjustedExpWeightedCovariance().regime_decay == exp2(-inv(40.0))^2
    @test RegimeAdjustedExpWeightedCovariance(; decay = 0.9, cor_decay = 0.99).min_obs ==
          round(Int, PortfolioOptimisers.decay_half_life(0.99))
    @test PortfolioOptimisers.decay_half_life(0.5) == 1
    @test all(h -> round(Int, PortfolioOptimisers.decay_half_life(exp2(-inv(h)))) == h,
              1:500)

    # A state does not merge, because it does not record whether an asset reset.
    a = partial_fit!(ExpWeightedVariance(; decay = EW_DECAY), view(Xg, 1:20, :)).cache
    b = partial_fit!(ExpWeightedVariance(; decay = EW_DECAY), view(Xg, 21:40, :)).cache
    @test_throws ArgumentError PortfolioOptimisers.merge_states(a, b)
    ma = partial_fit!(ExpWeightedExpectedReturns(; decay = EW_DECAY), view(Xg, 1:20, :)).cache
    mb = partial_fit!(ExpWeightedExpectedReturns(; decay = EW_DECAY), view(Xg, 21:40, :)).cache
    @test_throws ArgumentError PortfolioOptimisers.merge_states(ma, mb)
    ca = partial_fit!(ExpWeightedCovariance(; decay = EW_DECAY), view(Xg, 1:20, :)).cache
    cb = partial_fit!(ExpWeightedCovariance(; decay = EW_DECAY), view(Xg, 21:40, :)).cache
    @test_throws ArgumentError PortfolioOptimisers.merge_states(ca, cb)

    # The regime-adjusted variance with no regime is this estimator, so the two cannot drift.
    plain = var(ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1,
                                    centring = PreCentred()), Xg; active_mask = amsk)
    sibling = var(RegimeAdjustedExpWeightedVariance(; decay = EW_DECAY, min_obs = 1,
                                                    centring = PreCentred(),
                                                    regime_method = nothing), Xg;
                  active_mask = amsk)
    @test isequal(plain, sibling)

    # `dims = 2` reads the transpose of the same sample.
    @test isequal(mean(ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1),
                       transpose(Xg); dims = 2, active_mask = transpose(amsk)),
                  mean(ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1), Xg;
                       active_mask = amsk))
    @test isequal(cov(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                                            centring = PreCentred()), transpose(Xg);
                      dims = 2, active_mask = transpose(amsk)),
                  cov(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                                            centring = PreCentred()), Xg;
                      active_mask = amsk))
end

@testset "Exponentially weighted moments: the edges of the recursion" begin
    X = oracle_returns()
    amsk = oracle_active_mask()
    Xg = gapped(X, amsk)

    # A whole observation with no valid asset advances nothing, so the answer is the answer of
    # the sample with that row removed.
    blank = copy(amsk)
    blank[10, :] .= false
    Xb = gapped(X, blank)
    # Every asset is inactive at observation 10, so every state resets there. The row itself
    # contributes nothing: the recursion leaves the observation untouched.
    me = ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1)
    @test all(isfinite, mean(me, Xb; active_mask = blank))
    vc = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centring = PreCentred())
    @test all(isfinite, var(vc, Xb; active_mask = blank))
    cc = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centring = PreCentred())
    @test all(isfinite, diag(cov(cc, Xb; active_mask = blank)))

    # An asset that is inactive at the last observation loses its location as well as its
    # moment, so an uncentred fit reports `NaN` for both.
    amsk_end = copy(amsk)
    amsk_end[end, 2] = false
    Xe = gapped(X, amsk_end)
    vu = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1,
                             centring = EstimatedCentring())
    @test isnan(var(vu, Xe; active_mask = amsk_end)[2])
    cu = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                               centring = EstimatedCentring())
    @test all(isnan, view(cov(cu, Xe; active_mask = amsk_end), 2, :))

    # The covariance continues from a state the estimator already holds.
    two_blocks = partial_fit!(partial_fit!(cc, view(Xg, 1:20, :);
                                           active_mask = view(amsk, 1:20, :)),
                              view(Xg, 21:EW_T, :); active_mask = view(amsk, 21:EW_T, :))
    @test isequal(cov(two_blocks), cov(cc, Xg; active_mask = amsk))

    # A state of the wrong width is refused rather than silently reshaped.
    narrow = partial_fit!(ExpWeightedCovariance(; decay = EW_DECAY), view(Xg, :, 1:2);
                          active_mask = view(amsk, :, 1:2))
    @test_throws DimensionMismatch partial_fit!(narrow, Xg; active_mask = amsk)
    narrow_v = partial_fit!(ExpWeightedVariance(; decay = EW_DECAY), view(Xg, :, 1:2);
                            active_mask = view(amsk, :, 1:2))
    @test_throws DimensionMismatch partial_fit!(narrow_v, Xg; active_mask = amsk)
    narrow_m = partial_fit!(ExpWeightedExpectedReturns(; decay = EW_DECAY),
                            view(Xg, :, 1:2); active_mask = view(amsk, :, 1:2))
    @test_throws DimensionMismatch partial_fit!(narrow_m, Xg; active_mask = amsk)

    # `dims = 2` reads the transpose, for the variance as for its two siblings.
    @test isequal(var(vc, transpose(Xg); dims = 2, active_mask = transpose(amsk)),
                  var(vc, Xg; active_mask = amsk))

    # One observation at a time, with no mask: a non-finite return reads as a holiday.
    one_at_a_time = foldl((c, i) -> partial_fit!(c, view(Xg, i, :)), axes(Xg, 1); init = me)
    @test isequal(mean(one_at_a_time), mean(me, Xg))
end

@testset "Exponentially weighted moments: the pass without a callback" begin
    X = oracle_returns()
    amsk = oracle_active_mask()
    Xg = gapped(X, amsk)

    # Issue #877. The callback of `exp_weighted_pass!` is what makes a point-in-time series one
    # forward pass, and every other caller wants the last cache alone. The method that takes no
    # callback runs the same recursion, so the two caches read the same moment.
    for est in (ExpWeightedExpectedReturns(; decay = EW_DECAY),
                ExpWeightedVariance(; decay = EW_DECAY),
                ExpWeightedCovariance(; decay = EW_DECAY))
        with_f = PortfolioOptimisers.exp_weighted_pass!((args...) -> nothing, est, Xg, 1,
                                                        amsk)
        without_f = PortfolioOptimisers.exp_weighted_pass!(est, Xg, 1, amsk)
        @test isequal(PortfolioOptimisers.exp_weighted_moment(with_f, est),
                      PortfolioOptimisers.exp_weighted_moment(without_f, est))
    end
end

@testset "The gap fill value is keyed by the estimator" begin
    # Issue #925. A consumer that holds a gapped sample and an arbitrary covariance estimator
    # asks the estimator what a gapped cell is worth to it. A plain moment estimator takes the
    # fallback zero and never sees the gap; a gap-aware one answers `NaN`, which leaves the gap
    # where it is and asks for the mask that explains it. Adding a gap-aware estimator is that
    # one method, so the trait is checked here rather than at the one consumer.
    @test iszero(PortfolioOptimisers.gap_fill_value(PortfolioOptimisersCovariance()))
    @test iszero(PortfolioOptimisers.gap_fill_value(Covariance()))
    @test isnan(PortfolioOptimisers.gap_fill_value(ExpWeightedCovariance()))
    @test isnan(PortfolioOptimisers.gap_fill_value(RegimeAdjustedExpWeightedCovariance()))

    # The two answers are the two routes: a finite one is written over the gap, and a non-finite
    # one is not, so `isfinite` is the whole branch the consumer takes.
    @test isfinite(PortfolioOptimisers.gap_fill_value(PortfolioOptimisersCovariance()))
    @test !isfinite(PortfolioOptimisers.gap_fill_value(ExpWeightedCovariance()))

    # An estimator that answers a non-finite value owns the masked verb the consumer then calls.
    for ce in (ExpWeightedCovariance(), RegimeAdjustedExpWeightedCovariance())
        m = which(Statistics.cov, Tuple{typeof(ce), Matrix{Float64}})
        @test m.sig.parameters[2] === typeof(ce).name.wrapper
        @test :active_mask in Base.kwarg_decl(m)
    end

    # A covariance estimator nests, so the answer recurses. A composite that forwards the
    # sample and its keywords untouched answers what it wraps; one that reads the sample
    # itself before it delegates keeps the fallback, because it refuses the gap on its own
    # account and no answer of its inner estimator changes that. `ProcessedCovariance` and
    # `DenoiseCovariance` are constructors that build a `PortfolioOptimisersCovariance`, so
    # they take the recursion of the type they build.
    ew = ExpWeightedCovariance(; centring = PreCentred())
    for nest in (PortfolioOptimisersCovariance(; ce = ew), ProcessedCovariance(; ce = ew),
                 DenoiseCovariance(; ce = ew), CorrelationCovariance(; ce = ew),
                 PortfolioOptimisersCovariance(; ce = CorrelationCovariance(; ce = ew)))
        @test isnan(PortfolioOptimisers.gap_fill_value(nest))
    end
    for nest in (Covariance(; ce = ew), GeneralCovariance(; ce = ew),
                 PortfolioOptimisersCovariance(; ce = Covariance(; ce = ew)),
                 PortfolioOptimisersCovariance(), Covariance())
        @test iszero(PortfolioOptimisers.gap_fill_value(nest))
    end

    # The trait is the estimator's own promise, so it holds on a gapped sample: every nest
    # that answers `NaN` answers the masked verb, and every nest that answers zero refuses it.
    T, N = 60, 4
    rng = StableRNG(925_002)
    Xg = 0.7 .* randn(rng, T) .+ 0.7 .* randn(rng, T, N)
    amsk = trues(T, N)
    amsk[41:T, 4] .= false
    Xg[41:T, 4] .= NaN
    for nest in
        (ew, PortfolioOptimisersCovariance(; ce = ew), ProcessedCovariance(; ce = ew),
         CorrelationCovariance(; ce = ew))
        @test size(cov(nest, Xg; dims = 1, active_mask = amsk)) == (N, N)
        @test size(cor(nest, Xg; dims = 1, active_mask = amsk)) == (N, N)
    end
    for nest in (Covariance(; ce = ew), GeneralCovariance(; ce = ew),
                 PortfolioOptimisersCovariance(; ce = Covariance(; ce = ew)))
        @test_throws PortfolioOptimisers.IsNonFiniteError cov(nest, Xg; dims = 1,
                                                              active_mask = amsk)
    end
end

@testset "Exponentially weighted expected returns: the weighted mean it states" begin
    # Issue #889. The docstring states the answer as a weighted mean of the valid returns
    # after the last reset, with weights proportional to `decay^k`. This checks that statement
    # against a direct sum, over a listing, a reset, a holiday and a return that is not finite.
    rng = StableRNG(889)
    T, N = 50, 3
    Xw = randn(rng, T, N) / 100
    aw = trues(T, N)
    aw[1:12, 2] .= false
    aw[20:25, 3] .= false
    Xw[.!aw] .= NaN
    Xw[30, 1] = NaN
    Xw[31, 1] = Inf
    lam = 0.93
    function ew_direct_mean(x, a, lam)
        reset = findlast(t -> t > 1 && !a[t] && a[t - 1], eachindex(a))
        first_t = isnothing(reset) ? 1 : reset
        r = [x[t] for t in first_t:length(x) if a[t] && isfinite(x[t])]
        wt = lam .^ ((length(r) - 1):-1:0)
        return sum(wt .* r) / sum(wt)
    end
    mu = mean(ExpWeightedExpectedReturns(; decay = lam, min_obs = 1), Xw; active_mask = aw)
    @test isapprox(mu, [ew_direct_mean(view(Xw, :, i), view(aw, :, i), lam) for i in 1:N];
                   rtol = 1e-14)
    # Asset 3 has 25 valid returns after it returns, so a warm-up of 30 blanks it alone.
    mu30 = mean(ExpWeightedExpectedReturns(; decay = lam, min_obs = 30), Xw;
                active_mask = aw)
    @test isnan(mu30[3]) && isequal(mu30[1:2], mu[1:2])

    # The state of two blocks with no reset is `decay^n_b * S_a + S_b`.
    Y = randn(rng, 40, 2) / 100
    Y[25, 1] = NaN
    sa = partial_fit!(ExpWeightedExpectedReturns(; decay = lam), view(Y, 1:20, :)).cache
    sb = partial_fit!(ExpWeightedExpectedReturns(; decay = lam), view(Y, 21:40, :)).cache
    sw = partial_fit!(ExpWeightedExpectedReturns(; decay = lam), Y).cache
    @test isapprox(lam .^ sb.obs_count .* sa.mu .+ sb.mu, sw.mu; atol = 1e-17)

    # The default warm-up is the half-life, rounded, and round-off does not move it.
    @test all(h -> ExpWeightedExpectedReturns(; decay = exp2(-inv(h))).min_obs == h, 1:1000)
end

@testset "Exponentially weighted expected returns: the number type and the correction" begin
    # Issue #889. The state was seeded with `eltype(X)`, so an integer sample threw
    # `InexactError` at the first step. A mean divides, so the state now holds the type of a
    # division: an integer sample lands in a float, and a `Float32` sample stays `Float32`.
    me = ExpWeightedExpectedReturns(; decay = 0.9, min_obs = 1)
    Xi = [1 -2; -1 3; 2 -1; 0 1]
    @test isequal(mean(me, Xi), mean(me, float.(Xi)))
    @test eltype(mean(me, Float32.(Xi ./ 100))) === Float32

    # With one observation the mean is that observation. The correction divided by
    # `max(1 - decay^n, eps(T))`, which cut the answer when `1 - decay` fell below `eps(T)`:
    # 0.0083886 for 0.01 on `Float32` data at `decay = 1 - 1e-7`, and half the answer at
    # `decay = prevfloat(1.0)`. With `0 < decay < 1`, `1 - decay^n` is positive, and the
    # division needs no floor.
    @test mean(ExpWeightedExpectedReturns(; decay = 1 - 1e-7, min_obs = 1),
               Float32[0.01 -0.02]) == Float32[0.01, -0.02]
    @test mean(ExpWeightedExpectedReturns(; decay = prevfloat(1.0), min_obs = 1),
               [0.01 -0.02]) == [0.01, -0.02]
end

# The normalised weights of the location of asset `i` before the observation `t`, one per valid
# observation of `V` before `t`, written out from the definition with no recursion.
function ew_location_weights(V, t, λ)
    prior = [s for s in V if s < t]
    w = [λ^count(u -> s < u < t, prior) for s in prior]
    return Dict(zip(prior, w ./ sum(w)))
end

# The closed form that the docstring of `ExpWeightedCovariance` states, entry by entry and in
# BigFloat: the valid set of each asset, the normalised location of the returns before each one,
# the factor `1 + c_ij` of each product, and the mean of the two exponents of a pair, each
# counted on the clock of its own asset. Under `PreCentred` the deviation is the return.
function ew_cov_closed_form(X, amsk, λ, estimated, min_obs)
    T, N = size(X)
    λ = big(λ)
    active = isnothing(amsk) ? trues(T, N) : amsk
    V = Vector{Vector{Int}}(undef, N)
    for i in 1:N
        s = findlast(!, view(active, :, i))
        s = isnothing(s) ? 1 : s + 1
        V[i] = [t for t in s:T if isfinite(X[t, i]) && active[t, i]]
    end
    # The observations that give a deviation: every one but the first under the estimate.
    D = estimated ? [v[2:end] for v in V] : V
    a = [Dict(t => ew_location_weights(V[i], t, λ) for t in D[i]) for i in 1:N]
    e = fill(big(NaN), T, N)
    for i in 1:N, t in D[i]
        m = estimated ? sum(w * big(X[s, i]) for (s, w) in a[i][t]) : big(0)
        e[t, i] = big(X[t, i]) - m
    end
    factor(i, j, t) =
        if estimated
            1 + sum((w * get(a[j][t], s, big(0)) for (s, w) in a[i][t]))
        else
            big(1)
        end
    n = length.(V)
    Σ = fill(big(NaN), N, N)
    # Each pair ages on its common observations and is divided by the weight it holds
    # (ADR 0181, amendment of 2026-09-29).
    for i in 1:N, j in 1:N
        common = intersect(D[i], D[j])
        c(t) = count(>(t), common)
        S = (1 - λ) * sum((λ^c(t) * e[t, i] * e[t, j] / factor(i, j, t) for t in common);
                          init = big(0))
        Σ[i, j] = isempty(common) ? big(0) : S / (1 - λ^length(common))
    end
    bad = [n[i] < min_obs || n[i] < 1 + estimated || !active[T, i] for i in 1:N]
    Σ[bad, :] .= NaN
    Σ[:, bad] .= NaN
    return Σ
end

@testset "ExpWeightedCovariance against its closed form" begin
    # A late listing, a delisting and relisting, a delisting at the end, three holidays of an
    # active asset and an infinite return.
    X = oracle_returns()[:, 1:EW_N]
    X = hcat(X, X[:, 1] .* 0.5 .+ X[:, 3] .* 0.25)
    amsk = trues(EW_T, EW_N + 1)
    amsk[1:12, 2] .= false
    amsk[20:27, 3] .= false
    amsk[55:EW_T, 5] .= false
    X[.!amsk] .= NaN
    X[[5, 17, 33], 1] .= NaN
    X[40, 4] = Inf
    for (centring, estimated) in ((PreCentred(), false), (EstimatedCentring(), true)),
        mask in (amsk, nothing)

        ce = ExpWeightedCovariance(; decay = 0.93, min_obs = 3, centring = centring)
        sigma = cov(ce, X; active_mask = mask)
        ref = Float64.(ew_cov_closed_form(X, mask, 0.93, estimated, 3))
        # The report repairs a block that the division leaves indefinite.
        f = findall(isfinite, diag(ref))
        ref[f, f] = PortfolioOptimisers.restore_psd!(ref[f, f])
        @test isequal(isnan.(sigma), isnan.(ref))
        @test isapprox(filter(isfinite, sigma), filter(isfinite, ref); rtol = 1e-13)
    end

    # The first return of an asset gives no deviation under the estimated location, so one
    # observation is not ready. Two observations give the sample covariance of the two: the
    # location of one return is that return, and its factor is `1 + 1 / n_eff = 2`.
    x1 = [0.01 -0.02 0.03]
    @test all(isnan, cov(ExpWeightedCovariance(; decay = 0.9, min_obs = 1), x1))
    @test cov(ExpWeightedCovariance(; decay = 0.9, min_obs = 1, centring = PreCentred()),
              x1) ≈ transpose(x1) * x1
    x2 = [0.01 -0.02 0.03; -0.005 0.01 0.02]
    @test cov(ExpWeightedCovariance(; decay = 0.9, min_obs = 1), x2) ≈ cov(x2)

    # The correlation is the correlation of the state divided by its weights.
    ce = ExpWeightedCovariance(; decay = 0.9, min_obs = 1)
    Xc = oracle_returns()
    st = partial_fit!(ce, Xc).cache
    P = st.covariance ./ st.weight
    @test isapprox(cor(ce, Xc), P ./ sqrt.(diag(P) * transpose(diag(P))); atol = 1e-14)

    # Issue #1343. Two equal assets and five holidays of the second, while the first returns
    # zero. The pair holds its covariance and the variance of asset 1 falls, so the entries
    # imply a correlation of 2.50, and the matrix they form is indefinite. The report restores
    # the nearest positive semidefinite correlation, which is one, and keeps the variances.
    r = [0.02, -0.01, 0.015, -0.02, 0.01, 0.03, -0.025, 0.02]
    Xh = vcat(hcat(r, r), [zeros(5) fill(NaN, 5)])
    ch = ExpWeightedCovariance(; decay = 0.7, min_obs = 1, centring = PreCentred())
    sh = cov(ch, Xh)
    sth = partial_fit!(ch, Xh).cache
    Ph = sth.covariance ./ sth.weight
    @test isapprox(Ph[1, 2] / sqrt(Ph[1, 1] * Ph[2, 2]), 2.50054196358793; rtol = 1e-12)
    @test isapprox(diag(sh), diag(Ph); rtol = 1e-14)
    @test isapprox(sh[1, 2] / sqrt(sh[1, 1] * sh[2, 2]), 1; rtol = 1e-14)
    @test minimum(eigvals(Symmetric(sh))) > -1e-14 * maximum(abs, sh)
    @test isapprox(cor(ch, Xh)[1, 2], 1; rtol = 1e-14)

    # The report is positive semidefinite for every pattern of holidays, resets and listings.
    # The division by the weight of each pair need not be: on these panels the state divided by
    # its weights reaches a smallest eigenvalue far below zero, and the repair restores it.
    # A block whose correlation has a Cholesky factor is returned as it is, with no eigen
    # decomposition, and the eigenvalue test that follows it would keep that block too (#1612).
    rng = StableRNG(1343)
    worst = Inf
    worst_raw = Inf
    n_chol = 0
    n_fixed = 0
    for _ in 1:300
        T, N = rand(rng, 5:40), rand(rng, 2:6)
        Xr = randn(rng, T, N) / 100
        Xr[rand(rng, T, N) .< rand(rng) / 2] .= NaN
        ar = rand(rng, T, N) .> 0.05
        λr = 0.01 + 0.98 * rand(rng)
        for centring in (PreCentred(), EstimatedCentring()), mask in (ar, nothing)
            er = ExpWeightedCovariance(; decay = λr, min_obs = 1, centring = centring)
            Sr = cov(er, Xr; active_mask = mask)
            f = findall(isfinite, diag(Sr))
            length(f) < 2 && continue
            B = Sr[f, f]
            m = maximum(abs, B)
            iszero(m) && continue
            worst = min(worst, minimum(eigvals(Symmetric(B))) / m)
            st = partial_fit!(er, Xr; active_mask = mask).cache
            Pr = PortfolioOptimisers.pair_weighted_block(st.covariance, st.weight, f)
            worst_raw = min(worst_raw,
                            minimum(eigvals(Symmetric((Pr + transpose(Pr)) / 2))) / m)
            Ps = (Pr + transpose(Pr)) / 2
            p = findall(>(0), diag(Ps))
            isempty(p) && continue
            sp = sqrt.(diag(Ps)[p])
            Rp = Ps[p, p] ./ (sp .* transpose(sp))
            if issuccess(cholesky(Symmetric(Rp); check = false))
                n_chol += 1
                vals = eigvals(Symmetric(Rp))
                @test vals[1] >= -length(p) * eps() * maximum(abs, vals)
                Pc = copy(Ps)
                @test PortfolioOptimisers.restore_psd!(Pc) === Pc && Pc == Ps
            else
                n_fixed += PortfolioOptimisers.restore_psd!(copy(Ps)) != Ps
            end
        end
    end
    @test worst > -1e-14
    @test worst_raw < -0.01
    @test n_chol > 100 && n_fixed > 100

    # The fold S = λ^{n_b} S_a + S_b is exact under `PreCentred` over a complete block and not
    # under the estimated location, which is why a merge is refused.
    fold_gap(centring) = begin
        est = ExpWeightedCovariance(; decay = 0.9, centring = centring)
        a = partial_fit!(est, view(Xc, 1:12, :)).cache.covariance
        b = partial_fit!(est, view(Xc, 13:EW_T, :)).cache.covariance
        full = partial_fit!(est, Xc).cache.covariance
        maximum(abs, 0.9^(EW_T - 12) * a + b - full) / maximum(abs, full)
    end
    @test fold_gap(PreCentred()) < 1e-14
    @test fold_gap(EstimatedCentring()) > 1e-4

    # A second block writes the state of the first estimator in place, so the two share it.
    c1 = partial_fit!(ExpWeightedCovariance(; decay = 0.9), view(Xc, 1:20, :))
    c2 = partial_fit!(c1, view(Xc, 21:EW_T, :))
    @test c1.cache === c2.cache

    # The state takes the type of a quotient of two returns: an integer sample gives a
    # floating-point estimate, and a `Float32` sample keeps `Float32`.
    Xi = [1 2 3; 4 -1 2; 0 3 -2; 5 1 1; -3 2 4]
    ci = ExpWeightedCovariance(; decay = 0.8, min_obs = 1)
    @test isequal(cov(ci, Xi), cov(ci, float.(Xi)))
    @test eltype(cov(ExpWeightedCovariance(; decay = 0.8f0, min_obs = 1), Float32.(Xi))) ===
          Float32

    # A panel whose mask does not have the size of the sample is refused.
    amskp = trues(EW_T - 5, EW_N)
    @test_throws DimensionMismatch cov(ExpWeightedCovariance(), Xc, oracle_panel(amskp))
end

# The mathematics of `ExpWeightedVariance`, written out one asset at a time: the valid returns
# since the last reset, the normalised location of the returns before each one, its factor
# `1 + sum(a^2)` read from the weights themselves, and the corrected weighted sum of the terms.
# Under `PreCentred` each return is its own term.
function ewvar_reference(X, amsk, decay, min_obs, estimated)
    T, N = size(X)
    out = fill(NaN, T, N)
    lag = estimated ? 1 : 0
    for i in 1:N
        hist = Float64[]
        for t in 1:T
            act = isnothing(amsk) || amsk[t, i]
            if !act
                empty!(hist)
            elseif isfinite(X[t, i])
                push!(hist, X[t, i])
            end
            n = length(hist)
            K = n - lag
            if (!act || n < min_obs || K < 1)
                continue
            end
            S = 0.0
            for k in (lag + 1):n
                w = decay .^ ((k - 2):-1:0)
                a = w ./ sum(w)
                e2 = if estimated
                    (hist[k] - sum(a .* hist[1:(k - 1)]))^2 / (1 + sum(abs2, a))
                else
                    hist[k]^2
                end
                S += (1 - decay) * decay^(n - k) * e2
            end
            out[t, i] = S / (1 - decay^K)
        end
    end
    return out
end

@testset "The docstrings of 02_ExpWeightedVariance.jl against numbers" begin
    # Issue #890. The recursion equals its # Mathematical definition on random panels with
    # holidays, infinite returns and resets, through every verb that reads it.
    rng = StableRNG(890)
    for _ in 1:100
        T = rand(rng, 3:30)
        N = rand(rng, 1:4)
        X = 0.02 .* randn(rng, T, N) .+ 0.001
        X[rand(rng, T, N) .< 0.15] .= NaN
        X[rand(rng, T, N) .< 0.03] .= Inf
        amsk = rand(rng, Bool) ? nothing : rand(rng, T, N) .> 0.12
        decay = rand(rng, (0.5, 0.9, exp2(-inv(40.0))))
        min_obs = rand(rng, 1:4)
        estimated = rand(rng, Bool)
        ce = ExpWeightedVariance(; decay = decay, min_obs = min_obs,
                                 centring = estimated ? EstimatedCentring() : PreCentred())
        ref = ewvar_reference(X, amsk, decay, min_obs, estimated)
        vs = PortfolioOptimisers.variance_series(ce, X; active_mask = amsk)
        @test isequal(isnan.(vs), isnan.(ref))
        @test isapprox(filter(isfinite, vs), filter(isfinite, ref); rtol = 1e-12)
        @test isequal(var(ce, X; active_mask = amsk), vs[end, :])
        @test isequal(permutedims(PortfolioOptimisers.variance_series(ce, permutedims(X);
                                                                      dims = 2,
                                                                      active_mask = if isnothing(amsk)
                                                                          nothing
                                                                      else
                                                                          permutedims(amsk)
                                                                      end)), vs)
        k = rand(rng, 0:T)
        head = partial_fit!(ce, X[1:k, :];
                            active_mask = isnothing(amsk) ? nothing : amsk[1:k, :])
        both = partial_fit!(head, X[(k + 1):end, :];
                            active_mask = isnothing(amsk) ? nothing : amsk[(k + 1):end, :])
        @test isequal(var(both), vs[end, :])
    end

    # The estimated location on the oracle's fixture, against the hand reference.
    Xo, ao = oracle_returns(), oracle_active_mask()
    Xog = gapped(Xo, ao)
    @test isapprox(var(ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1), Xog;
                       active_mask = ao),
                   ewvar_reference(Xog, ao, EW_DECAY, 1, true)[end, :]; rtol = 1e-12)

    # The location is divided by the sum of its weights, so it is the mean of
    # `ExpWeightedExpectedReturns` (#1507).
    X = 0.02 .* randn(rng, 25, 3) .+ 0.01
    X[1:7, 2] .= NaN
    ce = ExpWeightedVariance(; decay = 0.9, min_obs = 1)
    mu = mean(ExpWeightedExpectedReturns(; decay = 0.9, min_obs = 1), X)
    @test partial_fit!(ce, X).cache.location ≈ mu rtol = 1e-14

    # A reset leaves the asset with no location, `NaN`.
    amsk = trues(size(X))
    amsk[end, 3] = false
    state = partial_fit!(ce, X; active_mask = amsk).cache
    @test isnan(state.location[3]) &&
          iszero(state.variance[3]) &&
          iszero(state.obs_count[3])

    # One observation at a time folds to the matrix fit, and the volatility of a held state is
    # the square root of its variance. The recursion folds in every configuration.
    Xf = 0.02 .* randn(rng, 12, 3)
    amskf = trues(size(Xf))
    amskf[4:6, 2] .= false
    ce = ExpWeightedVariance(; decay = 0.9, min_obs = 2)
    one_by_one = foldl((c, t) -> partial_fit!(c, view(Xf, t, :);
                                              active_mask = view(amskf, t, :)), axes(Xf, 1);
                       init = ce)
    @test isequal(var(one_by_one), var(ce, Xf; active_mask = amskf))
    @test isequal(std(ce, one_by_one.cache), sqrt.(var(ce, one_by_one.cache)))
    no_mask = foldl((c, t) -> partial_fit!(c, view(Xf, t, :)), axes(Xf, 1); init = ce)
    @test isequal(var(no_mask), var(ce, Xf))
    @test PortfolioOptimisers.supports_partial_fit(ce)

    # The cold-start correction divides by `1 - decay^n` with no floor. A floor at `eps` of the
    # element type halved the one-observation answer at `decay = prevfloat(1.0)`, and cut it
    # to 0.84 of itself on `Float32` data at `decay = 1 - 1e-7`.
    @test var(ExpWeightedVariance(; decay = prevfloat(1.0), min_obs = 1,
                                  centring = PreCentred()), fill(0.01, 1, 1)) ≈ [1e-4]
    @test var(ExpWeightedVariance(; decay = 1 - 1e-7, min_obs = 1, centring = PreCentred()),
              fill(0.01f0, 1, 1)) ≈ [1.0f-4] rtol = 1e-5
    # Under the estimated location one return gives no term, and two give the sample variance
    # of the two, whatever the decay.
    @test isnan(only(var(ExpWeightedVariance(; decay = 0.9, min_obs = 1), fill(0.01, 1, 1))))
    for decay in (0.5, 0.9, prevfloat(1.0))
        @test only(var(ExpWeightedVariance(; decay = decay, min_obs = 1), [0.01; -0.02;;])) ≈
              var([0.01, -0.02])
    end

    # An integer panel gets a floating-point state, and a `Float32` panel keeps `Float32`.
    Xi = [1 2; 3 -1; 0 2; 2 1; -1 0]
    ce = ExpWeightedVariance(; decay = 0.9, min_obs = 2)
    @test var(ce, Xi) == var(ce, float.(Xi))
    @test isequal(PortfolioOptimisers.variance_series(ce, Xi),
                  PortfolioOptimisers.variance_series(ce, float.(Xi)))
    @test eltype(var(ce, Float32.(Xi))) === Float32
    @test eltype(PortfolioOptimisers.variance_series(ce, Float32.(Xi))) === Float32

    # Under `PreCentred` the effective count is Kish's count of the weights `decay^k` over the
    # finite rows, and it is also the divisor, because the weights sum to one. Under the
    # estimated location the terms start at the second row, so the divisor is Kish's count of
    # one row fewer, and the location spends one observation, as the mean of a sample variance
    # does: with equal weights the pair is `(n, n - 1)`.
    X[3:9, 1] .= NaN
    cnt = PortfolioOptimisers.variance_count(ExpWeightedVariance(; decay = 0.9,
                                                                 centring = PreCentred()),
                                             X)
    cne = PortfolioOptimisers.variance_count(ExpWeightedVariance(; decay = 0.9), X)
    kish(k) = (a = 0.9 .^ (0:(k - 1)); a = a / sum(a); sum(a)^2 / sum(abs2, a))
    for i in axes(X, 2)
        k = count(isfinite, view(X, :, i))
        @test cnt.n[i] ≈ kish(k)
        @test cne.m[i] ≈ kish(k - 1)
    end
    @test cnt.m == cnt.n
    @test cne.n == cne.m .+ 1
    λ = exp2(-inv(40.0))
    @test PortfolioOptimisers.exp_weighted_variance_count(λ, ones(10_000, 1), PreCentred()).n[1] ≈
          (1 + λ) / (1 - λ)
    @test round((1 + λ) / (1 - λ)) == 115
end

#=
#1507. The estimated location is unbiased from the second valid return.

For returns independent in time with mean `μ` and variance `σ²`, the deviation from the normalised
location of the returns before it has the variance `σ² (1 + 1 / n_eff)`, and the deviations of a
pair have the covariance `σ_ij (1 + c_ij)`, with `c_ij` the overlap of the two locations. Each
term divided by its factor has the mean `σ²`, so every exponentially weighted mean of the terms is
unbiased, at every row and on any pattern of holidays. A simulation on #1507 checked the three
formulas (the scalar factor, the pair factor and the lagged HAC term) against Monte Carlo means:
the z-scores had an RMS of 0.97 and 1.02 over 245 and 4390 cells. Each test below compares the
mean over many independent samples with the truth, within four standard errors of that mean.
`PreCentred()` is the contrast: its mean is too large by `μ²`, or by `μ_i μ_j` for a pair.
=#
function within_se(est::AbstractMatrix, truth; k = 4)
    m = vec(mean(est; dims = 2))
    se = vec(std(est; dims = 2)) ./ sqrt(size(est, 2))
    return all(abs.(m .- truth) .< k .* se)
end

@testset "The estimated location is unbiased from the second return" begin
    rng = StableRNG(1507)
    μ, M = 0.3, 20_000
    for (hl, T) in ((10, 20), (40, 60))
        λ = 2.0^(-1 / hl)
        X = μ .+ randn(rng, T, M)
        vs = PortfolioOptimisers.variance_series(ExpWeightedVariance(; decay = λ,
                                                                     min_obs = 1), X)
        @test all(isnan, vs[1, :])
        @test within_se(vs[2:end, :], 1.0)
        vp = PortfolioOptimisers.variance_series(ExpWeightedVariance(; decay = λ,
                                                                     min_obs = 1,
                                                                     centring = PreCentred()),
                                                 X)
        @test within_se(vp, 1 + μ^2)
        @test !within_se(vp, 1.0)
        # The regime-adjusted variance with no regime is the same estimate.
        rv = RegimeAdjustedExpWeightedVariance(; decay = λ, min_obs = 1,
                                               regime_method = nothing)
        @test isequal(PortfolioOptimisers.variance_series(rv, X), vs)
    end
end

@testset "A pair with different histories is unbiased under its own factor" begin
    # The holiday fixture of the simulation of #1507: asset 1 is valid at every row, asset 2 has
    # a holiday every fifth row, and asset 3 lists at row 12 with holidays at 18, 19 and 27.
    T, R = 40, 20_000
    λ = 2.0^(-1 / 10)
    μ = [0.3, -0.5, 1.0]
    s = [1.0, 1.5, 0.7]
    C = [1.0 0.6 -0.3; 0.6 1.0 0.2; -0.3 0.2 1.0] .* (s * transpose(s))
    L = cholesky(C).L
    V = trues(T, 3)
    V[5:5:T, 2] .= false
    V[1:11, 3] .= false
    V[[18, 19, 27], 3] .= false
    rng = StableRNG(1508)
    est = Dict(c => zeros(6, R) for c in (:estimated, :pre, :hac))
    pairs = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
    ce = ExpWeightedCovariance(; decay = λ, min_obs = 1)
    cp = ExpWeightedCovariance(; decay = λ, min_obs = 1, centring = PreCentred())
    ch = RegimeAdjustedExpWeightedCovariance(; decay = λ, min_obs = 1, hac_lags = 2,
                                             regime_method = nothing)
    for r in 1:R
        X = transpose(μ .+ L * randn(rng, 3, T))
        X = ifelse.(V, X, NaN)
        Se = PortfolioOptimisers.exp_weighted_moment(partial_fit!(ce, X).cache, ce;
                                                     repair = false)
        Sp = PortfolioOptimisers.exp_weighted_moment(partial_fit!(cp, X).cache, cp;
                                                     repair = false)
        Sh = PortfolioOptimisers.regime_adjusted_covariance(partial_fit!(ch, X).cache, ch;
                                                            repair = false)
        for (k, (i, j)) in enumerate(pairs)
            est[:estimated][k, r] = Se[i, j]
            est[:pre][k, r] = Sp[i, j]
            est[:hac][k, r] = Sh[i, j]
        end
    end
    truth = [C[i, j] for (i, j) in pairs]
    @test within_se(est[:estimated], truth)
    @test within_se(est[:hac], truth)
    @test within_se(est[:pre], truth .+ [μ[i] * μ[j] for (i, j) in pairs])
    @test !within_se(est[:pre], truth)
end

@testset "The HAC variance is unbiased under the estimated location" begin
    # Under HAC each term adds the lagged products of the deviations, and a lagged product of
    # two estimated deviations has a non-zero mean. The factor reads it, so the recursion is
    # unbiased for returns independent in time, at one, two and five lags, with holidays. The
    # test reads the raw state: the report floors a negative HAC variance at zero, which is the
    # floor of `NoHacFloor` and adds a positive bias to a young estimate.
    rng = StableRNG(1509)
    μ, M, T = 0.3, 20_000, 30
    λ = 2.0^(-1 / 10)
    X = μ .+ randn(rng, T, M)
    X[rand(rng, T, M) .< 0.1] .= NaN
    for (L, centring, truth) in
        ((1, EstimatedCentring(), 1.0), (2, EstimatedCentring(), 1.0),
         (5, EstimatedCentring(), 1.0), (2, PreCentred(), 1.0))
        rv = RegimeAdjustedExpWeightedVariance(; decay = λ, min_obs = 1, hac_lags = L,
                                               regime_method = nothing, centring = centring)
        lag = PortfolioOptimisers.centring_lag(centring)
        raw = fill(NaN, T, M)
        PortfolioOptimisers.regime_adjusted_variance_pass!(rv, X, 1, nothing, nothing
                                                           ) do t, c
            K = c.obs_count .- lag
            raw[t, :] = ifelse.(K .>= 1, c.variance ./ (1 .- λ .^ K), NaN)
            return nothing
        end
        z = map(2:T) do t
            r = filter(isfinite, view(raw, t, :))
            return (mean(r) - truth) / (std(r) / sqrt(length(r)))
        end
        # `PreCentred` is unbiased only for a mean of zero, so it is the contrast here.
        @test (maximum(abs, z) < 4) == isa(centring, EstimatedCentring)
    end
end

@testset "The estimated location folds exactly, one block at a time" begin
    # The online state carries the location, its count and the overlap of each pair, so three
    # blocks fold to the fit over every row, on a panel with a listing, a reset and holidays.
    rng = StableRNG(1510)
    T, N = 45, 4
    X = 0.02 .* randn(rng, T, N) .+ 0.005
    amsk = trues(T, N)
    amsk[1:9, 2] .= false
    amsk[20:24, 3] .= false
    X[.!amsk] .= NaN
    X[[7, 13, 31], 1] .= NaN
    blocks = (1:11, 12:30, 31:T)
    for est in (ExpWeightedVariance(; decay = 0.9, min_obs = 2),
                ExpWeightedCovariance(; decay = 0.9, min_obs = 2),
                RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 2, hac_lags = 2,
                                                  regime_min_obs = 2),
                RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2, hac_lags = 2,
                                                    regime_min_obs = 2))
        folded = foldl((e, b) -> partial_fit!(e, view(X, b, :);
                                              active_mask = view(amsk, b, :)), blocks;
                       init = est)
        whole = partial_fit!(est, X; active_mask = amsk)
        v = if est isa Union{ExpWeightedVariance, RegimeAdjustedExpWeightedVariance}
            var
        else
            cov
        end
        @test isequal(v(folded), v(whole))
    end
end
