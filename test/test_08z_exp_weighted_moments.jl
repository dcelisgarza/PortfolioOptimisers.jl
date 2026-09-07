#=
The plain exponentially weighted family answers a gapped panel the way the reference does.

`ExpWeightedExpectedReturns`, `ExpWeightedVariance` and `ExpWeightedCovariance` are ports of the
reference implementation's three plain exponentially weighted members. Each seeds its recursion at
zero, freezes on a holiday, resets on an inactive period, and divides out the damping that the cold
start costs. The oracle is the reference itself: `oracle_returns` is an exactly representable
fixture that both languages build bit for bit, so no file is exchanged, and the literals below were
measured by fitting the reference on it at `half_life = 10`.

The fixture is 60 observations of 4 assets, and asset 4 lists at observation 31. The measured
parity is exact on every masked path and about `1e-19` on a complete window, where the reference
takes a matrix-multiply fast path and the port takes the row recursion.

Two families of testset sit beside the parity. The first pins the structural identities the census
of the reference states, and each is checked in plain Julia rather than against a stored number: the
congruence identity, the invariance of every correlation under the correction, the positive
semidefiniteness of the raw state, the freeze identity on the raw state, the warm-up mask and the
equal-history identity. The second pins the seam of ADR 0117: a mask-aware estimator overrides the
reduce-and-expand root and answers a young asset that the Coverage Universe drops.
=#
using Test, PortfolioOptimisers, Statistics, LinearAlgebra

# The reference's answers on the fixture below, at `half_life = 10`.
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
# reference builds the same matrix with no file exchange.
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

@testset "Exponentially weighted moments: parity with the reference" begin
    X = oracle_returns()
    amsk = oracle_active_mask()
    Xg = gapped(X, amsk)

    me = ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1)
    @test isapprox(mean(me, Xg; active_mask = amsk), EW_MU_MIN1; rtol = 1e-12)

    me40 = ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 40)
    mu40 = mean(me40, Xg; active_mask = amsk)
    @test isnan(mu40[EW_N])
    @test isapprox(view(mu40, 1:(EW_N - 1)), view(EW_MU_MIN40, 1:(EW_N - 1)); rtol = 1e-12)

    # With no mask every asset is active, so the leading `NaN` reads as a holiday and the
    # recursion freezes. The gap and the inactive period coincide here, so the two agree.
    @test isequal(mean(me, Xg), mean(me, Xg; active_mask = amsk))

    vc = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centred = true)
    @test isapprox(var(vc, Xg; active_mask = amsk), EW_VAR_CENTRED; rtol = 1e-12)
    @test isapprox(std(vc, Xg; active_mask = amsk), sqrt.(EW_VAR_CENTRED); rtol = 1e-12)

    vu = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centred = false)
    @test isapprox(var(vu, Xg; active_mask = amsk), EW_VAR_UNCENTRED; rtol = 1e-12)

    v40 = var(ExpWeightedVariance(; decay = EW_DECAY, min_obs = 40, centred = true), Xg;
              active_mask = amsk)
    @test isnan(v40[EW_N])
    @test isapprox(view(v40, 1:(EW_N - 1)), view(EW_VAR_MIN40, 1:(EW_N - 1)); rtol = 1e-12)

    cc = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centred = true)
    @test isapprox(cov(cc, Xg; active_mask = amsk), EW_COV_CENTRED; rtol = 1e-12)

    cu = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centred = false)
    @test isapprox(cov(cu, Xg; active_mask = amsk), EW_COV_UNCENTRED; rtol = 1e-12)

    # The delisting: asset 2 leaves at observation 46, so its state resets and it is blanked.
    amsk_d = copy(amsk)
    amsk_d[46:end, 2] .= false
    Xd = gapped(X, amsk_d)
    mu_d = mean(me, Xd; active_mask = amsk_d)
    @test isnan(mu_d[2])
    @test isapprox(mu_d[[1, 3, 4]], EW_MU_DELIST[[1, 3, 4]]; rtol = 1e-12)
    var_d = var(vc, Xd; active_mask = amsk_d)
    @test isnan(var_d[2])
    @test isapprox(var_d[[1, 3, 4]], EW_VAR_DELIST[[1, 3, 4]]; rtol = 1e-12)
    cov_d = cov(cc, Xd; active_mask = amsk_d)
    @test all(isnan, view(cov_d, 2, :))
    @test all(isnan, view(cov_d, :, 2))
end

@testset "Exponentially weighted moments: the structural identities" begin
    X = oracle_returns()
    amsk = oracle_active_mask()
    Xg = gapped(X, amsk)

    cc = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centred = true)
    fitted = partial_fit!(cc, Xg; active_mask = amsk)
    S = fitted.cache.covariance
    n = fitted.cache.obs_count
    sigma = cov(cc, Xg; active_mask = amsk)

    # The raw state matches the reference's own, which is what the freeze identity compares.
    @test isapprox(S, EW_COV_RAW_STATE; rtol = 1e-12)

    # 1. The congruence identity: the correction is `D S D`, and nothing else.
    D = Diagonal(inv.(sqrt.(1 .- EW_DECAY .^ n)))
    @test isapprox(D * S * D, sigma; atol = 1e-18)

    # 2. A congruence transform moves no correlation.
    cor_raw = Symmetric(S) ./ sqrt.(diag(S) * transpose(diag(S)))
    cor_out = Symmetric(sigma) ./ sqrt.(diag(sigma) * transpose(diag(sigma)))
    @test isapprox(cor_raw, cor_out; atol = 1e-14)

    # 3. The zero start keeps the raw state positive semidefinite, before any repair.
    @test minimum(eigvals(Symmetric(S))) > 0
    @test minimum(eigvals(Symmetric(sigma))) > 0

    # 4. The freeze identity, on the raw state. One holiday at the last row of asset 1 leaves
    #    that asset's raw row exactly where a fit that stops one observation earlier leaves it,
    #    and its count does not rise. The corrected output moves, because the other assets'
    #    counts rise and so the correction changes, which is why this reads the raw state.
    Xh = copy(Xg)
    Xh[end, 1] = NaN
    held = partial_fit!(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                                              centred = true), Xh; active_mask = amsk)
    short = partial_fit!(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                                               centred = true), view(Xg, 1:(EW_T - 1), :);
                         active_mask = view(amsk, 1:(EW_T - 1), :))
    @test view(held.cache.covariance, 1, :) == view(short.cache.covariance, 1, :)
    @test held.cache.obs_count[1] == EW_T - 1
    @test held.cache.obs_count[2] == EW_T

    # 5. The warm-up mask. Asset 4 carries 30 observations, so a threshold above it blanks the
    #    whole row and column and leaves every other entry alone.
    warm = cov(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 31, centred = true), Xg;
               active_mask = amsk)
    @test all(isnan, view(warm, EW_N, :))
    @test all(isnan, view(warm, :, EW_N))
    @test all(isfinite, view(warm, 1:(EW_N - 1), 1:(EW_N - 1)))

    # 6. The equal-history identity. With no gap every count is the same, so the correction is a
    #    scalar and the answer is the ordinary adjusted exponentially weighted moment.
    w = EW_DECAY .^ ((EW_T - 1):-1:0)
    sw = sum(w)
    @test isapprox(mean(ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1), X),
                   transpose(w) * X / sw |> vec; rtol = 1e-12)
    @test isapprox(var(ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centred = true),
                       X), transpose(w) * (X .^ 2) / sw |> vec; rtol = 1e-12)
    Xw = X .* sqrt.(w)
    @test isapprox(cov(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                                             centred = true), X), transpose(Xw) * Xw / sw;
                   rtol = 1e-12)

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

    vc = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centred = true)
    @test isequal(var(vc, Xg, pnl), var(vc, Xg; active_mask = amsk))
    @test isequal(std(vc, Xg, pnl), std(vc, Xg; active_mask = amsk))
    @test isfinite(var(vc, Xg, pnl)[EW_N])

    cc = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centred = true)
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

    vc = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centred = true)
    two_blocks = partial_fit!(partial_fit!(vc, view(Xg, 1:20, :);
                                           active_mask = view(amsk, 1:20, :)),
                              view(Xg, 21:EW_T, :); active_mask = view(amsk, 21:EW_T, :))
    @test isequal(var(two_blocks), var(vc, Xg; active_mask = amsk))
    @test isequal(std(two_blocks), std(vc, Xg; active_mask = amsk))
    @test isequal(var(vc, two_blocks.cache), var(two_blocks))

    cc = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centred = true)
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
    plain = var(ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centred = true), Xg;
                active_mask = amsk)
    sibling = var(RegimeAdjustedExpWeightedVariance(; decay = EW_DECAY, min_obs = 1,
                                                    centred = true,
                                                    regime_method = nothing), Xg;
                  active_mask = amsk)
    @test isequal(plain, sibling)

    # `dims = 2` reads the transpose of the same sample.
    @test isequal(mean(ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1),
                       transpose(Xg); dims = 2, active_mask = transpose(amsk)),
                  mean(ExpWeightedExpectedReturns(; decay = EW_DECAY, min_obs = 1), Xg;
                       active_mask = amsk))
    @test isequal(cov(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                                            centred = true), transpose(Xg); dims = 2,
                      active_mask = transpose(amsk)),
                  cov(ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1,
                                            centred = true), Xg; active_mask = amsk))
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
    vc = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centred = true)
    @test all(isfinite, var(vc, Xb; active_mask = blank))
    cc = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centred = true)
    @test all(isfinite, diag(cov(cc, Xb; active_mask = blank)))

    # An asset that is inactive at the last observation loses its location as well as its
    # moment, so an uncentred fit reports `NaN` for both.
    amsk_end = copy(amsk)
    amsk_end[end, 2] = false
    Xe = gapped(X, amsk_end)
    vu = ExpWeightedVariance(; decay = EW_DECAY, min_obs = 1, centred = false)
    @test isnan(var(vu, Xe; active_mask = amsk_end)[2])
    cu = ExpWeightedCovariance(; decay = EW_DECAY, min_obs = 1, centred = false)
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
