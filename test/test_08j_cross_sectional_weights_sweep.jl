const PO = PortfolioOptimisers

# The sweep of `src/05_Moments/32_CrossSectionalFactorModel/02_CrossSectionalWeights.jl`.
# Each testset states one formula of the file with an oracle that the library does not share:
# the three equations of `BlendedInverseVarianceWeights` written out per observation over the
# sets that its `Where:` list names, with the expanding sample variance of `Statistics`.
# `test_08j_base_regression.jl` holds the refusals, the infinite variances and the timing of
# the lag, so this file does not repeat them.

const CSWS_EPS, CSWS_MCAP, CSWS_MASK = let rng = StableRNG(698)
    T, N = 9, 41
    eps = randn(rng, T, N) .* exp.(randn(rng, 1, N))
    mcap = exp.(10 .+ 2 .* randn(rng, T, N))
    mask = rand(rng, T, N) .> 0.15
    # One eligible pair with a zero capitalisation, which leaves the estimation universe.
    mcap[5, findfirst(mask[5, :])] = 0.0
    eps, mcap, mask
end

# The second-pass weights of the `# Mathematical definition`. `M` is the eligible set of the
# observation, `V` the eligible assets with a variance estimate, and `E` the assets of `V`
# with a positive capitalisation weight. The sample variance needs two residuals, so `V` is
# empty before the third observation.
function csws_oracle(eps, mcap, mask, p, lambda, ratio, wins)
    T, N = size(eps)
    W = zeros(T, N)
    for t in 1:T
        M = findall(mask[t, :])
        if isempty(M)
            continue
        end
        g = [mcap[t, i]^p for i in M]
        g ./= sum(g)
        V = t >= 3 ? M : Int[]
        E = [i for i in V if mcap[t, i]^p > 0]
        u = if isempty(E)
            g
        else
            iv = [inv(var(eps[1:(t - 1), i])) for i in V]
            lo, hi = quantile([inv(var(eps[1:(t - 1), i])) for i in E], collect(wins))
            c = clamp.(iv, lo, hi)
            uv = min.(c, ratio * median(c))
            uu = zeros(length(M))
            uu[indexin(V, M)] = uv ./ sum(uv)
            uu
        end
        W[t, M] = lambda .* u .+ (1 - lambda) .* g
    end
    return W
end

@testset "Cross-sectional weights sweep" begin
    @testset "The blend states its three equations" begin
        cases = [(0.5, 0.6, 20.0, (0.025, 0.975)), (0.5, 1.0, 1.5, (0.1, 0.9)),
                 (0.0, 0.3, 3.0, (0.0, 1.0)), (1.0, 0.8, 2.0, (0.2, 0.7))]
        for (p, lambda, ratio, wins) in cases
            alg = BlendedInverseVarianceWeights(; p = p, lambda = lambda, ratio = ratio,
                                                wins = wins)
            W0 = PO.cs_weights_initial(alg, CSWS_MCAP, CSWS_MASK)
            W1 = PO.cs_weights_refine(alg, W0, CSWS_EPS, SimpleVariance(), CSWS_MASK)
            @test isapprox(W1,
                           csws_oracle(CSWS_EPS, CSWS_MCAP, CSWS_MASK, p, lambda, ratio,
                                       wins); rtol = 1e-12)
            # A single zero capitalisation leaves the row finite, and both components sum to
            # one, so every row of eligible weights does too.
            @test all(isfinite, W1)
            @test all(W1[.!CSWS_MASK] .== 0)
            @test all(t -> sum(W1[t, :]) ≈ 1, axes(W1, 1))
        end
        # The two bounds bind on this panel, so the oracle reads them.
        tight = csws_oracle(CSWS_EPS, CSWS_MCAP, CSWS_MASK, 0.5, 1.0, 1.5, (0.1, 0.9))
        loose = csws_oracle(CSWS_EPS, CSWS_MCAP, CSWS_MASK, 0.5, 1.0, 1e9, (0.0, 1.0))
        @test !isapprox(tight[3:end, :], loose[3:end, :]; rtol = 1e-3)
        # An observation with no variance estimate takes the capitalisation weights alone.
        alg = BlendedInverseVarianceWeights(; lambda = 1.0)
        W0 = PO.cs_weights_initial(alg, CSWS_MCAP, CSWS_MASK)
        W1 = PO.cs_weights_refine(alg, W0, CSWS_EPS, SimpleVariance(), CSWS_MASK)
        @test W1[1:2, :] ≈ W0[1:2, :] ./ sum(W0[1:2, :]; dims = 2)
    end

    @testset "A lagged inverse of integer variances holds NaN" begin
        IV = PO.cross_sectional_lagged_inverse([1 2; 4 8], trues(2, 2))
        @test eltype(IV) == Float64
        @test all(isnan, IV[1, :])
        @test IV[2, :] == [1.0, 0.5]
    end

    @testset "A row of zero capitalisations returns NaN for its eligible pairs" begin
        alg = BlendedInverseVarianceWeights(; lambda = 0.5)
        mask = [true true false; true true true; true true true]
        W0 = PO.cs_weights_initial(alg, [0.0 0.0 1.0; 4.0 9.0 1.0; 1.0 1.0 1.0], mask)
        W1 = PO.cs_weights_refine(alg, W0, [1.0 2.0 0.5; 3.0 6.0 1.0; 2.0 4.0 3.0],
                                  SimpleVariance(), mask)
        @test all(isnan, W1[1, 1:2])
        @test W1[1, 3] == 0
        @test all(isfinite, W1[2:3, :])
    end

    @testset "A power of zero takes the type of the power" begin
        mask = [true false true]
        @test PO.cross_sectional_cap_weights(0, nothing, mask) == [1 0 1]
        @test eltype(PO.cross_sectional_cap_weights(0, nothing, mask)) == Int
        @test eltype(PO.cross_sectional_cap_weights(0.0f0, nothing, mask)) == Float32
    end
end
