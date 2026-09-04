#=
Check `src/08_Moments/42_FactorExposures/04_EWMeanDescriptors.jl` and
`05_EWVolatilityDescriptors.jl` against the contract their docstrings state, and against the
reference implementation. Issue #718, map #643.

FOUR CONVENTIONS SHAPE THE PROBES, and the first three are `test_08s_descriptors.jl`'s.

1. A BLANK NEVER REACHES A CARRIER, so a Descriptor reads its Panel Fields back through the
   observed-mask column and never sees a fill value as data.

2. AN INACTIVE CELL IS `NaN`, whatever its Panel Fields hold.

3. A NON-POSITIVE DENOMINATOR IS `NaN`, NEVER AN ERROR.

4. A GAP HOLDS THE RECURSION. An asset that is listed but carries no value at an observation
   neither advances its state nor resets it, and the observation does not count toward the
   warm-up. Only [`EWVolatility`](@ref) resets, and only where the asset turns inactive. Every
   hand recursion below writes that rule out separately from the code under test.

The two recursions the census pins are written out as literals: `EWMomentum(half_life = 5,
skip = 0)` and `EWVolatility(half_life = 5)`. The last testset diffs two cases against numbers
the reference implementation printed on the synthetic panel of `test06c_setup.jl`.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))

# A small hand panel. Every field takes a forward fill, so each earns an observed-mask column
# and a raw `NaN` reads back as `NaN` through the mask rather than as the fill value.
function ew_hand_panel(fields::AbstractVector{<:Pair{String, <:AbstractMatrix}},
                       X::AbstractMatrix{<:Real};
                       amsk::AbstractMatrix{Bool} = trues(size(X)...),
                       emsk::AbstractMatrix{Bool} = amsk)
    inputs = [NumericPanelInput(; name = n, vals = v, alg = ForwardPanelFill(; val = 0.0))
              for (n, v) in fields]
    res = asset_panel(inputs; amsk = amsk, emsk = emsk)
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:size(X, 2)], X = X, res...)
end

# Two matrices agree when they hold `NaN` in the same cells and are close everywhere else.
function ew_same(A::AbstractMatrix, B::AbstractMatrix; atol::Real = 0, rtol::Real = 1e-12)
    if !(size(A) == size(B))
        return false
    end
    return all(CartesianIndices(A)) do k
        a, b = A[k], B[k]
        return isnan(a) ? isnan(b) : (!isnan(b) && isapprox(a, b; atol = atol, rtol = rtol))
    end
end

# The exponentially weighted mean recursion, written out by hand. A cell that is not finite
# holds the state and does not count toward the warm-up.
function ew_hand_mean(R::AbstractMatrix{<:Real}, decay::Real, min_obs::Integer)
    T, N = size(R)
    S = fill(NaN, T, N)
    for i in 1:N
        s, n = 0.0, 0
        for t in 1:T
            if isfinite(R[t, i])
                s = decay * s + (1 - decay) * R[t, i]
                n += 1
            end
            if n >= min_obs
                S[t, i] = s
            end
        end
    end
    return S
end

# The exponentially weighted volatility, written out by hand: uncentred, bias corrected by
# `1 / (1 - λ^n)`, and reset where the asset turns inactive. `mar` of `nothing` is the
# two-sided form.
function ew_hand_volatility(X::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                            decay::Real, min_obs::Integer; mar = nothing)
    T, N = size(X)
    D = fill(NaN, T, N)
    act, v, n = trues(N), zeros(N), zeros(Int, N)
    for t in 1:T, i in 1:N
        if act[i] && !amsk[t, i]
            v[i], n[i] = 0.0, 0
        end
        act[i] = amsk[t, i]
        x = X[t, i]
        if amsk[t, i] && isfinite(x)
            c = isnothing(mar) ? x : min(x - mar, 0.0)
            v[i] = decay * v[i] + (1 - decay) * c^2
            n[i] += 1
        end
        if n[i] >= min_obs
            D[t, i] = sqrt(v[i] / (1 - decay^n[i]))
        end
    end
    return D
end

# Read one raw field and its observed mask back off the carrier by column name.
function ew_raw_field(rd::ReturnsResult, name::AbstractString)
    Z = rd.Z
    col = findfirst(==(name), rd.nz)
    ocol = findfirst(==(name * "::observed"), rd.nz)
    V = Matrix{Float64}(Z[:, :, col])
    if !isnothing(ocol)
        V[iszero.(Z[:, :, ocol])] .= NaN
    end
    return V
end

@testset "Exponentially weighted Descriptor constructors" begin
    @testset "Every named constructor fixes its census fields and half-life" begin
        mom = EWMomentum()
        @test isa(mom, EWMean)
        @test mom.decay ≈ exp2(-inv(87.0))
        @test mom.min_obs == 87
        @test mom.skip == 21
        @test mom.exponentiate == false

        trn = EWShareTurnover()
        @test isa(trn, EWVolumeRatio)
        @test trn.num == "adj_volume"
        @test trn.den == "adj_shares_outstanding"
        @test trn.decay ≈ exp2(-inv(21.0))
        @test trn.min_obs == 21

        ami = EWAmihudIlliquidity()
        @test isa(ami, EWVolumeRatio)
        @test isnothing(ami.num)
        @test ami.den == ["adj_close", "adj_volume"]
        @test ami.decay ≈ exp2(-inv(63.0))
        @test ami.min_obs == 63

        dtc = DaysToCover()
        @test isa(dtc, DaysToCover)
        @test dtc.num == "short_interest"
        @test dtc.den == "adj_volume"
        @test dtc.decay ≈ exp2(-inv(21.0))
        @test dtc.min_obs == 21

        vol = EWVolatility()
        @test isa(vol, EWVolatility)
        @test vol.ce.decay ≈ exp2(-inv(40.0))
        @test vol.ce.min_obs == 40
        @test vol.ce.centred
        @test isnothing(vol.ce.regime_method)
        @test isa(vol.alg, FullMoment)
        @test iszero(vol.mar)

        dvol = EWDownsideVolatility()
        @test isa(dvol, EWVolatility)
        @test dvol.ce.decay ≈ exp2(-inv(40.0))
        @test isa(dvol.alg, SemiMoment)
        @test iszero(dvol.mar)
    end
    @testset "A keyword override renames a field, and half_life fixes the two it converts" begin
        @test EWShareTurnover(; num = "vol", den = "shr").num == "vol"
        @test EWShareTurnover(; num = "vol", den = "shr").den == "shr"
        @test DaysToCover(; num = "si", den = "vol").num == "si"
        @test EWAmihudIlliquidity(; den = ["px", "vol"]).den == ["px", "vol"]
        @test EWMomentum(; half_life = 4).decay ≈ exp2(-inv(4.0))
        @test EWMomentum(; half_life = 4).min_obs == 4
        # A value passed for a converted field is used as it stands.
        @test EWMomentum(; half_life = 4, min_obs = 9).min_obs == 9
        @test DaysToCover(; half_life = 4, decay = 0.25).decay == 0.25
        @test EWVolatility(; half_life = 7).ce.min_obs == 7
        @test EWDownsideVolatility(; half_life = 7, mar = 0.01).mar == 0.01
    end
    @testset "The half-life conversions" begin
        @test PortfolioOptimisers.half_life_decay(1.0) == 0.5
        @test PortfolioOptimisers.half_life_decay(40.0) ≈ exp2(-inv(40.0))
        @test PortfolioOptimisers.half_life_min_obs(40.0) == 40
        @test PortfolioOptimisers.half_life_min_obs(40.5) == 41
        @test PortfolioOptimisers.half_life_min_obs(0.25) == 1
        @test_throws Exception PortfolioOptimisers.half_life_decay(0.0)
        @test_throws Exception PortfolioOptimisers.half_life_min_obs(-1.0)
    end
    @testset "The constructors refuse what the docstrings say they refuse" begin
        @test_throws DomainError EWMean(; decay = 0.0, min_obs = 1)
        @test_throws DomainError EWMean(; decay = 1.0, min_obs = 1)
        @test_throws DomainError EWMean(; decay = 0.5, min_obs = 1, skip = -1)
        @test_throws Exception EWMean(; decay = 0.5, min_obs = 0)
        @test_throws DomainError EWVolumeRatio(; num = "a", den = "b", decay = 1.5,
                                               min_obs = 1)
        @test_throws Exception EWVolumeRatio(; num = "", den = "b", decay = 0.5,
                                             min_obs = 1)
        @test_throws Exception EWVolumeRatio(; num = "a", den = String[], decay = 0.5,
                                             min_obs = 1)
        @test_throws Exception EWVolumeRatio(; num = "a", den = ["b", ""], decay = 0.5,
                                             min_obs = 1)
        @test_throws Exception EWVolumeRatio(; num = Pair{String, Float64}[], den = "b",
                                             decay = 0.5, min_obs = 1)
        @test_throws Exception DaysToCover(; num = "", den = "b")
        @test_throws Exception EWVolatility(; mar = NaN)
    end
    @testset "A Descriptor over the returns still refuses a carrier with no Asset Panel" begin
        rd = ReturnsResult(; nx = ["A", "B"], X = [0.01 0.02; 0.03 0.04])
        @test_throws Exception descriptor(EWMomentum(; half_life = 2, skip = 0), rd)
        @test_throws Exception descriptor(EWVolatility(; half_life = 2), rd)
    end
end

@testset "Exponentially weighted means on a hand panel" begin
    X = [0.10 0.20; -0.05 NaN; 0.04 0.03; NaN -0.02; 0.01 0.06]
    vol = [10.0 20.0; 30.0 40.0; 50.0 60.0; 70.0 80.0; 90.0 100.0]
    shr = [100.0 200.0; 100.0 200.0; 100.0 400.0; 100.0 400.0; 100.0 400.0]
    px = [5.0 6.0; 5.5 6.5; 6.0 7.0; 6.5 7.5; 7.0 8.0]
    si = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0; 9.0 10.0]
    rd = ew_hand_panel(["adj_volume" => copy(vol), "adj_shares_outstanding" => copy(shr),
                        "adj_close" => copy(px), "short_interest" => copy(si)], X)
    lam, mo = 0.5, 2
    @testset "EWMean equals the hand recursion over log1p, and a NaN holds the state" begin
        D = descriptor(EWMean(; decay = lam, min_obs = mo), rd)
        @test ew_same(D, ew_hand_mean(log1p.(X), lam, mo))
        # The second asset misses observation two, so its warm-up ends one row later.
        @test isnan(D[2, 2])
        @test !isnan(D[2, 1])
    end
    @testset "A skip delays the input, and the first skip rows read nothing" begin
        for skip in (1, 2, 3)
            R = fill(NaN, size(X))
            for i in axes(X, 2), t in (skip + 1):size(X, 1)
                R[t, i] = log1p(X[t - skip, i])
            end
            D = descriptor(EWMean(; decay = lam, min_obs = 1, skip = skip), rd)
            @test ew_same(D, ew_hand_mean(R, lam, 1))
            @test all(isnan, view(D, 1:skip, :))
        end
    end
    @testset "exponentiate returns the same signal in return units" begin
        de = EWMean(; decay = lam, min_obs = 1)
        D = descriptor(de, rd)
        E = descriptor(EWMean(; decay = lam, min_obs = 1, exponentiate = true), rd)
        @test ew_same(E, expm1.(D))
    end
    @testset "A return at or below -1 raises, and NaN does not" begin
        bad = ew_hand_panel(["adj_volume" => [1.0 2.0; 3.0 4.0]], [0.1 0.2; -1.0 0.3])
        @test_throws DomainError descriptor(EWMean(; decay = lam, min_obs = 1), bad)
        ok = ew_hand_panel(["adj_volume" => [1.0 2.0; 3.0 4.0]], [0.1 0.2; NaN 0.3])
        @test size(descriptor(EWMean(; decay = lam, min_obs = 1), ok)) == (2, 2)
    end
    @testset "EWVolumeRatio equals the recursion over the ratio of its two sides" begin
        D = descriptor(EWVolumeRatio(; num = "adj_volume", den = "adj_shares_outstanding",
                                     decay = lam, min_obs = mo), rd)
        @test ew_same(D, ew_hand_mean(vol ./ shr, lam, mo))
    end
    @testset "A nothing numerator reads the absolute returns, and a product denominator multiplies" begin
        D = descriptor(EWVolumeRatio(; num = nothing, den = ["adj_close", "adj_volume"],
                                     decay = lam, min_obs = 1), rd)
        @test ew_same(D, ew_hand_mean(abs.(X) ./ (px .* vol), lam, 1))
    end
    @testset "A combination side sums its terms with their coefficients" begin
        D = descriptor(EWVolumeRatio(; num = ["adj_volume" => 1, "short_interest" => -1],
                                     den = "adj_shares_outstanding", decay = lam,
                                     min_obs = 1), rd)
        @test ew_same(D, ew_hand_mean((vol .- si) ./ shr, lam, 1))
    end
    @testset "A denominator that is not strictly positive is NaN, and holds the state" begin
        zvol = copy(vol)
        zvol[2, 1] = 0.0
        nvol = copy(shr)
        nvol[3, 2] = -1.0
        rd2 = ew_hand_panel(["adj_volume" => zvol, "adj_shares_outstanding" => nvol], X)
        D = descriptor(EWVolumeRatio(; num = "adj_volume", den = "adj_shares_outstanding",
                                     decay = lam, min_obs = 1), rd2)
        R = [nvol[k] > 0 ? zvol[k] / nvol[k] : NaN for k in CartesianIndices(zvol)]
        @test ew_same(D, ew_hand_mean(R, lam, 1))
        @test D[3, 2] == D[2, 2]
    end
    @testset "DaysToCover smooths the denominator alone, and only a positive value advances" begin
        zvol = copy(vol)
        zvol[3, 1] = 0.0
        rd2 = ew_hand_panel(["adj_volume" => zvol, "short_interest" => copy(si)], X)
        D = descriptor(DaysToCover(; decay = lam, min_obs = mo), rd2)
        V = [v > 0 ? v : NaN for v in zvol]
        @test ew_same(D, si ./ ew_hand_mean(V, lam, mo))
        # The zero volume neither advances the state nor counts toward the warm-up.
        @test D[4, 1] ≈ si[4, 1] / ew_hand_mean(V, lam, mo)[4, 1]
    end
    @testset "An inactive cell is NaN in every archetype" begin
        amsk = trues(size(X))
        amsk[4, 1] = false
        rd2 = ew_hand_panel(["adj_volume" => copy(vol),
                             "adj_shares_outstanding" => copy(shr),
                             "short_interest" => copy(si)], ifelse.(amsk, X, NaN);
                            amsk = amsk)
        for de in (EWMean(; decay = lam, min_obs = 1),
                   EWVolumeRatio(; num = "adj_volume", den = "adj_shares_outstanding",
                                 decay = lam, min_obs = 1),
                   DaysToCover(; decay = lam, min_obs = 1))
            @test isnan(descriptor(de, rd2)[4, 1])
        end
    end
    @testset "The estimator holds no data: two carriers, one estimator" begin
        de = EWVolumeRatio(; num = "adj_volume", den = "adj_shares_outstanding",
                           decay = lam, min_obs = 1)
        D1 = descriptor(de, rd)
        rd2 = ew_hand_panel(["adj_volume" => 2 .* vol,
                             "adj_shares_outstanding" => copy(shr)], X)
        D2 = descriptor(de, rd2)
        @test ew_same(D2, 2 .* D1)
        @test ew_same(D1, descriptor(de, rd))
    end
end

@testset "Exponentially weighted volatility on a hand panel" begin
    X = [0.10 0.20; -0.05 NaN; 0.04 0.03; -0.02 -0.06; 0.01 0.05]
    mcap = ones(size(X))
    rd = ew_hand_panel(["market_cap" => copy(mcap)], X)
    @testset "EWVolatility equals sqrt(S / (1 - lambda^n)) over the raw returns" begin
        for hl in (1.0, 2.0, 5.0)
            de = EWVolatility(; half_life = hl)
            lam, mo = exp2(-inv(hl)), max(1, ceil(Int, hl))
            @test ew_same(descriptor(de, rd),
                          ew_hand_volatility(X, trues(size(X)), lam, mo); rtol = 1e-10)
        end
    end
    @testset "EWDownsideVolatility clips every return above the target to zero" begin
        for mar in (0.0, 0.02)
            de = EWDownsideVolatility(; half_life = 2, mar = mar)
            @test ew_same(descriptor(de, rd),
                          ew_hand_volatility(X, trues(size(X)), exp2(-inv(2.0)), 2;
                                             mar = mar); rtol = 1e-10)
        end
    end
    @testset "The transform is the alg slot, and it reads the target" begin
        @test PortfolioOptimisers.ew_volatility_input(FullMoment(), X, 0.5) === X
        @test PortfolioOptimisers.ew_volatility_input(SemiMoment(), [0.1 -0.2], 0.0) ==
              [0.0 -0.2]
        @test PortfolioOptimisers.ew_volatility_input(SemiMoment(), [0.1 -0.2], 0.2) ==
              [-0.1 -0.4]
    end
    @testset "An asset that turns inactive restarts its recursion" begin
        amsk = trues(size(X))
        amsk[3, 2] = false
        rd2 = ew_hand_panel(["market_cap" => copy(mcap)], ifelse.(amsk, X, NaN);
                            amsk = amsk)
        de = EWVolatility(; half_life = 1)
        D = descriptor(de, rd2)
        @test ew_same(D, ew_hand_volatility(ifelse.(amsk, X, NaN), amsk, 0.5, 1);
                      rtol = 1e-10)
        @test isnan(D[3, 2])
        # The observation after the gap is the first of a fresh recursion, so it equals the
        # volatility of one observation.
        @test D[4, 2] ≈ abs(X[4, 2])
    end
    @testset "A caller may pass any variance estimator in the ce slot" begin
        de = EWVolatility(; ce = SimpleVariance(; corrected = false))
        D = descriptor(de, rd)
        @test size(D) == size(X)
        # The expanding-window fallback refits, so row `t` is the sample deviation of the
        # first `t` observations rather than a recursion.
        @test iszero(D[1, 1])
        @test D[3, 1] ≈ std(view(X, 1:3, 1); corrected = false)
    end
end

@testset "The two recursions the census pins" begin
    X = [0.010 0.020; -0.005 0.030; 0.004 -0.010; 0.007 0.002; -0.003 0.006; 0.002 -0.004;
         0.001 0.005; -0.002 0.003]
    rd = ew_hand_panel(["market_cap" => ones(size(X))], X)
    lam = exp2(-inv(5.0))
    @testset "EWMomentum(half_life = 5, skip = 0) is the recursion over log1p" begin
        S = zeros(2)
        expected = fill(NaN, size(X))
        for t in axes(X, 1), i in axes(X, 2)
            S[i] = lam * S[i] + (1 - lam) * log1p(X[t, i])
            if t >= 5
                expected[t, i] = S[i]
            end
        end
        @test ew_same(descriptor(EWMomentum(; half_life = 5, skip = 0), rd), expected)
    end
    @testset "EWVolatility(half_life = 5) is sqrt(var / (1 - lambda^n))" begin
        V = zeros(2)
        expected = fill(NaN, size(X))
        for t in axes(X, 1), i in axes(X, 2)
            V[i] = lam * V[i] + (1 - lam) * X[t, i]^2
            if t >= 5
                expected[t, i] = sqrt(V[i] / (1 - lam^t))
            end
        end
        @test ew_same(descriptor(EWVolatility(; half_life = 5), rd), expected; rtol = 1e-10)
    end
end

@testset "Every exponentially weighted Descriptor runs on the synthetic panel" begin
    res = synthetic_asset_panel(; n_assets = 12, n_observations = 300, n_industries = 3,
                                rng = StableRNGs.StableRNG(718))
    rd = res.rd
    amsk = rd.pnl.amsk
    des = ("EWMomentum" => EWMomentum(; half_life = 5, skip = 3),
           "EWShareTurnover" => EWShareTurnover(; half_life = 5),
           "EWAmihudIlliquidity" => EWAmihudIlliquidity(; half_life = 5),
           "DaysToCover" => DaysToCover(; half_life = 5),
           "EWVolatility" => EWVolatility(; half_life = 5),
           "EWDownsideVolatility" => EWDownsideVolatility(; half_life = 5))
    @testset "Shape, the inactive fill, and finite-or-NaN everywhere" begin
        for (name, de) in des
            D = descriptor(de, rd)
            @test size(D) == size(rd.X)
            @test all(isnan, D[.!amsk])
            @test all(x -> isnan(x) || isfinite(x), D)
            @test any(isfinite, D)
        end
    end
    @testset "Every one equals its census formula, written from rd.nz and rd.Z" begin
        lam, mo = exp2(-inv(5.0)), 5
        X = rd.X
        vol = ew_raw_field(rd, "adj_volume")
        shr = ew_raw_field(rd, "adj_shares_outstanding")
        px = ew_raw_field(rd, "adj_close")
        si = ew_raw_field(rd, "short_interest")

        R = fill(NaN, size(X))
        for i in axes(X, 2), t in 4:size(X, 1)
            R[t, i] = log1p(X[t - 3, i])
        end
        mom = ew_hand_mean(R, lam, mo)
        mom[.!amsk] .= NaN
        @test ew_same(descriptor(EWMomentum(; half_life = 5, skip = 3), rd), mom)

        trn = ew_hand_mean([b > 0 ? a / b : NaN for (a, b) in zip(vol, shr)], lam, mo)
        trn[.!amsk] .= NaN
        @test ew_same(descriptor(EWShareTurnover(; half_life = 5), rd), trn)

        amt = px .* vol
        ami = ew_hand_mean([b > 0 ? a / b : NaN for (a, b) in zip(abs.(X), amt)], lam, mo)
        ami[.!amsk] .= NaN
        @test ew_same(descriptor(EWAmihudIlliquidity(; half_life = 5), rd), ami)

        V = ew_hand_mean([v > 0 ? v : NaN for v in vol], lam, mo)
        dtc = [b > 0 ? a / b : NaN for (a, b) in zip(si, V)]
        dtc[.!amsk] .= NaN
        @test ew_same(descriptor(DaysToCover(; half_life = 5), rd), dtc)

        @test ew_same(descriptor(EWVolatility(; half_life = 5), rd),
                      ew_hand_volatility(X, amsk, lam, mo); rtol = 1e-10)
        @test ew_same(descriptor(EWDownsideVolatility(; half_life = 5), rd),
                      ew_hand_volatility(X, amsk, lam, mo; mar = 0.0); rtol = 1e-10)
    end
    @testset "An asset view of the carrier gives the same Descriptor as a slice of the whole" begin
        idx = [2, 5, 9]
        rdv = PortfolioOptimisers.port_opt_view(rd, idx)
        for (name, de) in des
            @test ew_same(descriptor(de, rdv), descriptor(de, rd)[:, idx]; rtol = 1e-10)
        end
    end
    #=
    The reference implementation was run on this exact panel, all thirteen cases of the two
    files. Every one agreed: no cell differed in whether it is `NaN`, and the largest relative
    difference over the finite cells was 4.3e-15, which is floating point noise. Four cases
    agreed to the last bit.

    Two rows of two cases are stored here as literals, so a regression is caught without the
    reference. Each carries the skip, the exponentiate flag and a non-zero minimum acceptable
    return, which the hand recursions above exercise least. Asset two is delisted before the
    last observation, so its literal is `NaN`.
    =#
    @testset "Two cases against numbers the reference implementation printed" begin
        mom_250 = [-0.0062617311055179368, -0.0017340648519905755, -0.0014142152400465399,
                   -0.0066824380942661545, -0.003862538174442892, -0.0059118968616883802,
                   -0.0011177723894685181, -0.0030445356884902497, -0.0018186579722233619,
                   -0.0037452755978652427, -0.0025812232095826274, -0.0011207232842962492]
        mom_300 = [0.0026801832529124402, NaN, 0.0013041330563512584,
                   -0.00034676038877819767, 0.0021258058284273546, 0.0066697029825081421,
                   0.0001141682515419082, 0.0038148303524405849, -0.0035315191667812159,
                   0.0050263759405964302, -0.0050916758636027565, 0.00049559608707931669]
        D = descriptor(EWMomentum(; half_life = 5, skip = 3, exponentiate = true), rd)
        @test ew_same(reshape(view(D, 250, :), 1, :), reshape(mom_250, 1, :); rtol = 1e-12)
        @test ew_same(reshape(view(D, 300, :), 1, :), reshape(mom_300, 1, :); rtol = 1e-12)

        dvol_250 = [0.014740415979782184, 0.010935460514462909, 0.0078246811492014238,
                    0.011856636498505527, 0.011270136212473032, 0.016393160826384069,
                    0.0088465900408822348, 0.015734551356596065, 0.0082358432719374357,
                    0.013898674790251199, 0.0068745744622555538, 0.0088197316727841851]
        dvol_300 = [0.012527657490412247, NaN, 0.0074375526678808193, 0.010477183167212674,
                    0.0098730045114479772, 0.015458198225516948, 0.0070970854066652106,
                    0.014025551244518937, 0.0089478645922382478, 0.013239202598403822,
                    0.0069770549263295598, 0.008116416489161369]
        V = descriptor(EWDownsideVolatility(; half_life = 40, mar = 0.001), rd)
        @test ew_same(reshape(view(V, 250, :), 1, :), reshape(dvol_250, 1, :); rtol = 1e-12)
        @test ew_same(reshape(view(V, 300, :), 1, :), reshape(dvol_300, 1, :); rtol = 1e-12)
    end
end
