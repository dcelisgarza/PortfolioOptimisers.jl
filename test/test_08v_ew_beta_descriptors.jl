#=
Check `src/05_Moments/32_CrossSectionalFactorModel/04_FactorExposures/06_EWBetaDescriptors.jl`, the residual half of
`05_EWVolatilityDescriptors.jl`, and the market-return builder and beta recursion they share
in `01_Base_Descriptor.jl`, against the contract their docstrings state and against the
reference implementation. Issue #719, map #643.

FIVE CONVENTIONS SHAPE THE PROBES, and the first three are `test_08u_ew_descriptors.jl`'s.

1. AN INACTIVE CELL IS `NaN`, whatever the Panel Fields hold.

2. A GAP HOLDS THE RECURSION. An asset that is listed but carries no return at an
   observation neither advances its state nor resets it, and the observation does not count
   toward its warm-up.

3. AN INACTIVE CELL IS ALSO A GAP. The reference implementation's own container refuses a
   return outside the active mask, so no member ever advances a state there. The library
   masks the returns through `ew_active_returns` before the recursion starts, which is the
   same rule stated on this side.

4. THE MARKET STATE IS ONE STATE FOR THE WHOLE PANEL, and it advances at every observation
   whatever any single asset does. Only [`EWMacroSensitivity`](@ref) freezes it, and only
   where the reference return is not finite, because a partial beta needs both series.

5. ONLY THE RESIDUAL VOLATILITY RESETS. Its beta and its variance both restart where an
   asset turns inactive. The three beta Descriptors freeze instead, so an asset that
   re-enters the universe resumes from the state it left.

THE STORED CASES ARE THE REFERENCE IMPLEMENTATION'S OWN OUTPUT. `assets/EWMarketBeta.csv.gz`
is `EWMarketBeta(half_life = 5, group = "industry", min_group_size = 2,
bounds = (0.1, 0.9))`, and `assets/EWMacroSensitivity.csv.gz` is
`EWMacroSensitivity(half_life = 5, agg_obs = 3)`. Both were produced on the panel the last
testset rebuilds. They were chosen for the two paths the hand recursions below exercise
least: the James-Stein shrinkage under a clamped weight, and the bivariate partial beta on
an aggregated clock. All thirteen cases of the diff agreed with no difference in the `NaN`
pattern and a worst relative difference of 2.5e-13 over the finite cells.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))

# A small hand panel carrying a market capitalisation beside the returns, and optionally an
# industry classification. Every numeric field takes a forward fill, so each earns an
# observed-mask column and a raw `NaN` reads back as `NaN` rather than as the fill value.
function ewb_hand_panel(X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real};
                        amsk::AbstractMatrix{Bool} = trues(size(X)...),
                        emsk::AbstractMatrix{Bool} = amsk,
                        industry::Union{Nothing, <:AbstractMatrix{<:AbstractString}} = nothing)
    inputs = Any[NumericPanelInput(; name = "market_cap", vals = W,
                                   alg = ForwardPanelFill(; val = 0.0))]
    if !isnothing(industry)
        push!(inputs,
              CategoricalPanelInput(; name = "industry", vals = industry,
                                    levels = sort(unique(vec(industry)))))
    end
    res = asset_panel(identity.(inputs); amsk = amsk, emsk = emsk)
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:size(X, 2)], X = X, pnl = res)
end

# Two matrices agree when they hold `NaN` in the same cells and are close everywhere else.
function ewb_same(A::AbstractMatrix, B::AbstractMatrix; atol::Real = 0, rtol::Real = 1e-12)
    if !(size(A) == size(B))
        return false
    end
    return all(CartesianIndices(A)) do k
        a, b = A[k], B[k]
        return isnan(a) ? isnan(b) : (!isnan(b) && isapprox(a, b; atol = atol, rtol = rtol))
    end
end

# The capitalisation-weighted market return, written out by hand. Only an estimable pair
# whose return and whose weight are both finite enters.
function ewb_hand_market(X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real},
                         emsk::AbstractMatrix{Bool})
    T, N = size(X)
    rm = zeros(T)
    for t in 1:T
        s, w = 0.0, 0.0
        for i in 1:N
            if emsk[t, i] && isfinite(X[t, i]) && isfinite(W[t, i])
                s += W[t, i] * X[t, i]
                w += W[t, i]
            end
        end
        rm[t] = s / w
    end
    return rm
end

# The exponentially weighted market beta, written out by hand. Deviations are taken from the
# means of the previous step. A cell outside the active mask, or whose return is not finite,
# advances nothing; the beta of that cell is the one the asset last carried. `reset` says
# whether an asset that turns inactive restarts.
function ewb_hand_beta(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                       amsk::AbstractMatrix{Bool}, decay::Real, min_obs::Integer,
                       min_val::Real; reset::Bool = false)
    T, N = size(X)
    B = fill(NaN, T, N)
    b, mu, cv, n, act = fill(NaN, N), zeros(N), zeros(N), zeros(Int, N), trues(N)
    mu_m, var_m = 0.0, 0.0
    for t in 1:T
        if reset
            for i in 1:N
                if act[i] && !amsk[t, i]
                    mu[i], cv[i], n[i] = 0.0, 0.0, 0
                end
                act[i] = amsk[t, i]
            end
        end
        dm = rm[t] - mu_m
        mu_m = decay * mu_m + (1 - decay) * rm[t]
        var_m = decay * var_m + (1 - decay) * dm * dm
        for i in 1:N
            if amsk[t, i] && isfinite(X[t, i])
                d = X[t, i] - mu[i]
                mu[i] = decay * mu[i] + (1 - decay) * X[t, i]
                cv[i] = decay * cv[i] + (1 - decay) * d * dm
                n[i] += 1
                if t >= min_obs && n[i] >= min_obs
                    b[i] = cv[i] / (var_m + min_val)
                end
            end
            B[t, i] = b[i]
        end
    end
    return B
end

# The exponentially weighted downside beta, written out by hand. The market's shortfall
# advances at every observation, and a ready asset reads it whether or not its own return is
# finite at that observation.
function ewb_hand_downside(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                           amsk::AbstractMatrix{Bool}, decay::Real, min_obs::Integer,
                           mar::Real, min_val::Real)
    T, N = size(X)
    B = fill(NaN, T, N)
    cd, n, vd = zeros(N), zeros(Int, N), 0.0
    for t in 1:T
        dm = min(rm[t] - mar, 0.0)
        vd = decay * vd + (1 - decay) * dm * dm
        for i in 1:N
            if amsk[t, i] && isfinite(X[t, i])
                cd[i] = decay * cd[i] + (1 - decay) * min(X[t, i] - mar, 0.0) * dm
                n[i] += 1
            end
            if t >= min_obs && n[i] >= min_obs
                B[t, i] = cd[i] / (vd + min_val)
            end
        end
    end
    return B
end

# The exponentially weighted partial beta on a reference series, written out by hand through
# the Frisch-Waugh closed form. A reference return that is not finite freezes the whole
# state, market included.
function ewb_hand_macro(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                        rf::AbstractVector{<:Real}, amsk::AbstractMatrix{Bool}, decay::Real,
                        min_obs::Integer, min_val::Real)
    T, N = size(X)
    B = fill(NaN, T, N)
    b, mu, cam, caf, n = fill(NaN, N), zeros(N), zeros(N), zeros(N), zeros(Int, N)
    mu_m, mu_f, var_m, var_f, cov_mf, c = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for t in 1:T
        if isfinite(rf[t])
            c += 1
            dm, df = rm[t] - mu_m, rf[t] - mu_f
            mu_m = decay * mu_m + (1 - decay) * rm[t]
            mu_f = decay * mu_f + (1 - decay) * rf[t]
            var_m = decay * var_m + (1 - decay) * dm * dm
            var_f = decay * var_f + (1 - decay) * df * df
            cov_mf = decay * cov_mf + (1 - decay) * dm * df
            vm = var_m + min_val
            vf = var_f - cov_mf * cov_mf / vm + min_val
            for i in 1:N
                if amsk[t, i] && isfinite(X[t, i])
                    d = X[t, i] - mu[i]
                    mu[i] = decay * mu[i] + (1 - decay) * X[t, i]
                    cam[i] = decay * cam[i] + (1 - decay) * d * dm
                    caf[i] = decay * caf[i] + (1 - decay) * d * df
                    n[i] += 1
                    if c >= min_obs && n[i] >= min_obs
                        b[i] = (caf[i] - cam[i] * cov_mf / vm) / vf
                    end
                end
            end
        end
        B[t, :] = b
    end
    return B
end

# The exponentially weighted volatility of the market-model residual, written out by hand:
# uncentred, bias corrected by `1 / (1 - λ^n)`, and reset where the asset turns inactive.
function ewb_hand_residual(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                           B::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                           decay::Real, min_obs::Integer; mar = nothing)
    T, N = size(X)
    D = fill(NaN, T, N)
    v, n, act = zeros(N), zeros(Int, N), trues(N)
    for t in 1:T, i in 1:N
        if act[i] && !amsk[t, i]
            v[i], n[i] = 0.0, 0
        end
        act[i] = amsk[t, i]
        if amsk[t, i] && isfinite(X[t, i])
            e = X[t, i] - B[t, i] * rm[t]
            c = isnothing(mar) ? e : min(e - mar, 0.0)
            v[i] = decay * v[i] + (1 - decay) * c * c
            n[i] += 1
        end
        if n[i] >= min_obs
            D[t, i] = sqrt(v[i] / (1 - decay^n[i]))
        end
    end
    return D
end

# The James-Stein shrinkage of one cross-section, written out from its closed form. The
# estimation set is every asset whose beta is not `NaN`, whose group is set and whose weight
# is finite and strictly positive. A group below `mgs` takes the prior of the whole set.
function ewb_hand_shrink(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                         L::AbstractVector{<:Integer}, w::AbstractVector{<:Real},
                         mgs::Integer, bounds::Tuple{<:Real, <:Real}, min_val::Real)
    ix = [i
          for i in eachindex(b)
          if !isnan(b[i]) &&
                 L[i] != PortfolioOptimisers.CS_MISSING_GROUP &&
                 isfinite(w[i]) &&
                 w[i] > 0]
    function prior(jx)
        W = sum(w[j] for j in jx)
        mu = sum(w[j] * b[j] for j in jx) / W
        tau2 = sum(w[j] * (b[j] - mu)^2 for j in jx) / W -
               sum(bev[j] for j in jx) / length(jx)
        return mu, max(tau2, 0.0)
    end
    s = copy(b)
    for i in ix
        gx = [j for j in ix if L[j] == L[i]]
        mu, tau2 = length(gx) < mgs ? prior(ix) : prior(gx)
        q = clamp(tau2 / (tau2 + bev[i] + min_val), bounds...)
        s[i] = q * b[i] + (1 - q) * mu
    end
    return s
end

@testset "EW beta descriptors: the market return" begin
    X = [0.10 0.20 -0.05
         -0.10 0.00 0.03
         0.05 0.05 0.01
         0.02 -0.03 0.04]
    W = [1.0 3.0 2.0
         2.0 2.0 2.0
         5.0 1.0 4.0
         1.0 1.0 1.0]
    emsk = [true true false
            true true true
            false true true
            true true true]
    rd = ewb_hand_panel(X, W; amsk = trues(4, 3), emsk = emsk)
    rm = PortfolioOptimisers.market_return_series(rd, "market_cap")
    @testset "It is the capitalisation-weighted mean over the estimation universe" begin
        @test rm ≈ ewb_hand_market(X, W, emsk)
        @test length(rm) == 4
        @test rm[1] ≈ (1.0 * 0.10 + 3.0 * 0.20) / 4.0
        @test rm[3] ≈ (1.0 * 0.05 + 4.0 * 0.01) / 5.0
    end
    @testset "A missing return and a missing weight both leave the estimate" begin
        Xn = copy(X)
        Xn[2, 1] = NaN
        rdn = ewb_hand_panel(Xn, W; amsk = trues(4, 3), emsk = emsk)
        @test PortfolioOptimisers.market_return_series(rdn, "market_cap")[2] ≈
              (2.0 * 0.00 + 2.0 * 0.03) / 4.0
        Wn = copy(W)
        Wn[2, 3] = NaN
        rdw = ewb_hand_panel(Xn, Wn; amsk = trues(4, 3), emsk = emsk)
        @test PortfolioOptimisers.market_return_series(rdw, "market_cap")[2] ≈ 0.0
    end
    @testset "An observation with no estimable weight raises" begin
        emz = copy(emsk)
        emz[3, :] .= false
        rdz = ewb_hand_panel(X, W; amsk = trues(4, 3), emsk = emz)
        @test_throws ArgumentError PortfolioOptimisers.market_return_series(rdz,
                                                                            "market_cap")
        @test_throws ArgumentError descriptor(EWMarketBeta(; half_life = 1), rdz)
    end
    @testset "A carrier with no Asset Panel raises" begin
        rdp = ReturnsResult(; nx = ["A1", "A2", "A3"], X = X)
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.market_return_series(rdp,
                                                                                                 "market_cap")
    end
end

@testset "EW beta descriptors: EWBeta against its hand recursion" begin
    X = [0.10 0.20 -0.05
         -0.10 0.00 0.03
         0.05 NaN 0.01
         0.02 -0.03 0.04
         -0.04 0.06 -0.02
         0.03 0.01 0.05]
    W = [1.0 3.0 2.0
         2.0 2.0 2.0
         5.0 1.0 4.0
         1.0 1.0 1.0
         2.0 3.0 1.0
         1.0 2.0 3.0]
    amsk = trues(6, 3)
    amsk[1, 3] = false
    amsk[5, 1] = false
    emsk = copy(amsk)
    rd = ewb_hand_panel(X, W; amsk = amsk, emsk = emsk)
    rm = ewb_hand_market(X, W, emsk)
    Xm = ifelse.(amsk, X, NaN)
    decay, min_obs, min_val = exp2(-inv(2.0)), 2, 1e-12
    @testset "The beta is the hand recursion, masked by the active mask" begin
        D = descriptor(EWMarketBeta(; half_life = 2), rd)
        B = ewb_hand_beta(Xm, rm, amsk, decay, min_obs, min_val)
        @test ewb_same(D, ifelse.(amsk, B, NaN))
        @test all(k -> !amsk[k] ? isnan(D[k]) : true, CartesianIndices(D))
    end
    @testset "A gap holds the beta rather than resetting it" begin
        D = descriptor(EWMarketBeta(; half_life = 2), rd)
        # Asset 2 has no return at observation 3, so it carries observation 2's beta there.
        @test D[3, 2] == D[2, 2]
        # Asset 1 leaves the universe at observation 5 and comes back at observation 6, so
        # its beta at observation 6 continues the state it left rather than restarting.
        @test isnan(D[5, 1])
        @test isfinite(D[6, 1])
    end
    @testset "The warm-up answers NaN before an asset has min_obs of its own" begin
        D = descriptor(EWMarketBeta(; half_life = 4), rd)
        @test all(isnan, D[1:3, :])
        @test isfinite(D[4, 1])
    end
    @testset "The recursion reads the decay and the warm-up, not the half-life" begin
        a = descriptor(EWBeta(; decay = decay, min_obs = min_obs), rd)
        b = descriptor(EWMarketBeta(; half_life = 2), rd)
        @test isequal(a, b)
        c = descriptor(EWBeta(; decay = decay, min_obs = 4), rd)
        @test all(isnan, c[1:3, :])
    end
end

@testset "EW beta descriptors: the aggregated clock" begin
    X = [0.10 0.20
         -0.10 0.00
         0.05 0.03
         0.02 -0.03
         -0.04 0.06
         0.03 0.01
         0.04 -0.01]
    W = ones(7, 2) .+ 0.5
    rd = ewb_hand_panel(X, W)
    rm = ewb_hand_market(X, W, trues(7, 2))
    D = descriptor(EWMarketBeta(; half_life = 2, agg_obs = 3), rd)
    @testset "It aggregates, then spreads the windows over their observations" begin
        Xa = PortfolioOptimisers.ew_agg_series(X, 3)
        rma = PortfolioOptimisers.ew_agg_vector(rm, 3)
        @test size(Xa) == (2, 2)
        @test Xa[1, 1] ≈ (0.10 - 0.10 + 0.05) / 3
        B = ewb_hand_beta(Xa, rma, trues(2, 2), exp2(-inv(2.0)), 2, 1e-12)
        @test all(isnan, D[1:5, :])
        @test isequal(D[6, :], B[2, :])
        @test isequal(D[7, :], B[2, :])
    end
    @testset "A window whose entries are all missing aggregates to NaN" begin
        A = [1.0 NaN; 3.0 NaN; 5.0 NaN]
        @test PortfolioOptimisers.ew_agg_series(A, 3) ≈ [3.0 NaN] nans = true
    end
    @testset "A tail shorter than one window is dropped" begin
        @test size(PortfolioOptimisers.ew_agg_series(X, 4)) == (1, 2)
        @test PortfolioOptimisers.ew_agg_vector([1.0, 3.0, 5.0], 2) ≈ [2.0]
    end
end

@testset "EW beta descriptors: the James-Stein shrinkage" begin
    @testset "A shrunk beta lies between its raw value and the group mean" begin
        b = [0.6, 1.4, 1.0, 2.0]
        bev = [0.01, 0.01, 0.01, 0.01]
        L = [1, 1, 1, 1]
        w = ones(4)
        s = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 1, (0.0, 1.0), 1e-12)
        m = sum(b) / 4
        for i in eachindex(b)
            @test min(b[i], m) <= s[i] <= max(b[i], m)
        end
        @test sum(s) ≈ sum(b)
    end
    @testset "The bounds clamp the weight the raw beta keeps" begin
        # Twenty precise betas and one noisy one. The precise assets want the whole of
        # their raw value and the noisy one wants almost none of it, so a clamp of
        # `(0.25, 0.75)` binds at both ends at once.
        b = vcat(fill(0.6, 10), fill(1.4, 10), 2.0)
        bev = vcat(fill(1e-9, 20), 1.0)
        L = ones(Int, 21)
        w = ones(21)
        m = sum(b) / 21
        s = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 1, (0.25, 0.75), 1e-12)
        @test s[1] ≈ 0.75 * b[1] + 0.25 * m
        @test s[21] ≈ 0.25 * b[21] + 0.75 * m
        # With no clamp the precise asset keeps almost the whole of its raw beta and the
        # noisy one moves almost the whole way to the mean, so it is the bounds that set
        # the two weights above and not the data.
        u = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 1, (0.0, 1.0), 1e-12)
        @test abs(u[1] - b[1]) < abs(s[1] - b[1])
        @test abs(u[21] - m) < abs(s[21] - m)
    end
    @testset "An asset outside the estimation set keeps its raw beta" begin
        b = [0.6, 1.4, 2.0, 3.0]
        bev = fill(0.01, 4)
        L = [1, 1, PortfolioOptimisers.CS_MISSING_GROUP, 1]
        w = [1.0, 1.0, 1.0, 0.0]
        s = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 1, (0.0, 1.0), 1e-12)
        @test s[3] == b[3]
        @test s[4] == b[4]
        @test s[1] != b[1]
        # A NaN beta is outside the set too, and it stays NaN.
        @test isnan(PortfolioOptimisers.ew_beta_shrink([NaN], [0.01], [1], [1.0], 1,
                                                       (0.0, 1.0), 1e-12)[1])
    end
    @testset "A group below min_group_size falls back on the whole cross-section" begin
        b = [0.5, 0.7, 1.5, 1.7, 3.0]
        bev = fill(0.01, 5)
        L = [1, 1, 2, 2, 3]
        w = ones(5)
        small = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 3, (0.0, 1.0), 1e-12)
        whole = PortfolioOptimisers.ew_beta_shrink(b, bev, ones(Int, 5), w, 1, (0.0, 1.0),
                                                   1e-12)
        @test small ≈ whole
        grouped = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 2, (0.0, 1.0), 1e-12)
        @test !(grouped ≈ whole)
        # The lone asset of group 3 is below min_group_size either way, so it takes the
        # whole cross-section's mean in both runs.
        @test grouped[5] ≈ whole[5]
    end
    @testset "No estimable asset leaves every beta raw" begin
        b = [0.6, 1.4]
        s = PortfolioOptimisers.ew_beta_shrink(b, [0.01, 0.01],
                                               fill(PortfolioOptimisers.CS_MISSING_GROUP,
                                                    2), ones(2), 1, (0.0, 1.0), 1e-12)
        @test s == b
    end
    @testset "The Descriptor shrinks only after the warm-up, and only with a group" begin
        X = [0.10 0.20 -0.05
             -0.10 0.00 0.03
             0.05 0.02 0.01
             0.02 -0.03 0.04
             -0.04 0.06 -0.02]
        W = fill(2.0, 5, 3)
        ind = repeat(["a" "a" "b"], 5, 1)
        rd = ewb_hand_panel(X, W; industry = ind)
        raw = descriptor(EWMarketBeta(; half_life = 2), rd)
        shr = descriptor(EWMarketBeta(; half_life = 2, group = "industry",
                                      min_group_size = 2), rd)
        @test isequal(raw[1, :], shr[1, :])
        @test !isequal(raw[3, :], shr[3, :])
        @test isequal(isnan.(raw), isnan.(shr))
    end
    @testset "The residual variance reads the previous observation's beta" begin
        X = [0.10 0.20; -0.10 0.00; 0.05 0.02; 0.02 -0.03]
        rm = [0.15, -0.05, 0.035, -0.005]
        B, _ = PortfolioOptimisers.ew_beta_series(X, rm, 0.5, 1, 1e-12)
        Vr = PortfolioOptimisers.ew_beta_residual_variance(X, rm, B, 0.5, 1)
        @test all(iszero, Vr[1, :])
        for t in 2:4, i in 1:2
            e = X[t, i] - B[t - 1, i] * rm[t]
            @test Vr[t, i] ≈ 0.5 * Vr[t - 1, i] + 0.5 * e * e
        end
    end
    @testset "The half-life is recovered from the decay exactly enough" begin
        for hl in (1.0, 5.0, 40.0, 60.0, 87.0)
            @test PortfolioOptimisers.decay_half_life(PortfolioOptimisers.half_life_decay(hl)) ≈
                  hl
        end
    end
end

@testset "EW beta descriptors: EWMacroSensitivity" begin
    X = [0.10 0.20 -0.05
         -0.10 0.00 0.03
         0.05 0.02 0.01
         0.02 -0.03 0.04
         -0.04 0.06 -0.02
         0.03 0.01 0.05]
    W = fill(2.0, 6, 3)
    amsk = trues(6, 3)
    amsk[1, 3] = false
    rd = ewb_hand_panel(X, W; amsk = amsk, emsk = amsk)
    rm = ewb_hand_market(X, W, amsk)
    ref = [0.02, -0.01, 0.03, 0.00, NaN, 0.01]
    Xm = ifelse.(amsk, X, NaN)
    @testset "It is the Frisch-Waugh partial beta of the hand recursion" begin
        D = descriptor(EWMacroSensitivity(; half_life = 2), rd; ref = ref)
        B = ewb_hand_macro(Xm, rm, ref, amsk, exp2(-inv(2.0)), 2, 1e-12)
        @test ewb_same(D, ifelse.(amsk, B, NaN))
    end
    @testset "A reference return that is not finite freezes the whole state" begin
        D = descriptor(EWMacroSensitivity(; half_life = 2), rd; ref = ref)
        @test isequal(D[5, :], D[4, :])
        # The freeze reaches the market too: a run whose fifth reference return is present
        # differs at every later observation.
        rf = copy(ref)
        rf[5] = 0.005
        Dn = descriptor(EWMacroSensitivity(; half_life = 2), rd; ref = rf)
        @test !isequal(Dn[6, :], D[6, :])
    end
    @testset "The reference series is required, and it is one entry per observation" begin
        @test_throws PortfolioOptimisers.IsNothingError descriptor(EWMacroSensitivity(;
                                                                                      half_life = 2),
                                                                   rd)
        @test_throws DimensionMismatch descriptor(EWMacroSensitivity(; half_life = 2), rd;
                                                  ref = ref[1:3])
    end
    @testset "The aggregated clock aggregates the reference series too" begin
        D = descriptor(EWMacroSensitivity(; half_life = 2, agg_obs = 3), rd; ref = ref)
        Xa = PortfolioOptimisers.ew_agg_series(Xm, 3)
        rma = PortfolioOptimisers.ew_agg_vector(rm, 3)
        rfa = PortfolioOptimisers.ew_agg_vector(ref, 3)
        @test rfa[2] ≈ (0.00 + 0.01) / 2
        B = ewb_hand_macro(Xa, rma, rfa, trues(2, 3), exp2(-inv(2.0)), 2, 1e-12)
        @test isequal(D[6, :], ifelse.(amsk[6, :], B[2, :], NaN))
        @test all(isnan, D[1:2, :])
    end
end

# Returns driven by the market and by one macro series, with the macro series in the Exogenous
# Series of the returns data under the name "FX" (#1365, ADR 0184).
function ewb_macro_fixture(; T::Integer = 120, N::Integer = 10, fx = nothing)
    rng = StableRNGs.StableRNG(1365)
    m = 0.01 .* randn(rng, T)
    f = isnothing(fx) ? 0.005 .* randn(rng, T) : fx
    bm = collect(range(0.6, 1.4; length = N))
    bf = collect(range(-1.0, 1.0; length = N))
    X = m .* transpose(bm) .+ ifelse.(isfinite.(f), f, 0.0) .* transpose(bf) .+
        0.002 .* randn(rng, T, N)
    W = 1.0 .+ rand(rng, T, N)
    pnl = asset_panel([NumericPanelInput(; name = "market_cap", vals = W)];
                      amsk = trues(T, N), emsk = trues(T, N))
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:N], X = X, ne = ["SPX", "FX"],
                         E = hcat(m, f), pnl = pnl)
end

@testset "EW beta descriptors: EWMacroSensitivity reads the Exogenous Series (#1365)" begin
    PO = PortfolioOptimisers
    rd = ewb_macro_fixture()
    ref = rd.E[:, 2]
    @testset "The named column gives the partial beta of the keyword" begin
        for agg_obs in (1, 3)
            de = EWMacroSensitivity(; series = "FX", half_life = 5, agg_obs = agg_obs)
            dr = EWMacroSensitivity(; half_life = 5, agg_obs = agg_obs)
            @test isequal(descriptor(de, rd), descriptor(dr, rd; ref = ref))
        end
        # The column is found by its name, not by its place.
        rds = ReturnsResult(; nx = rd.nx, X = rd.X, ne = ["FX", "SPX"], E = rd.E[:, [2, 1]],
                            pnl = rd.pnl)
        @test isequal(descriptor(EWMacroSensitivity(; series = "FX", half_life = 5), rds),
                      descriptor(EWMacroSensitivity(; series = "FX", half_life = 5), rd))
    end
    @testset "A gap in the warm-up is accepted, and the recursion skips it" begin
        # The warm-up of a half-life of 5 is more than four observations, and three of its
        # values are missing.
        de = EWMacroSensitivity(; series = "FX", half_life = 5)
        f = copy(ref)
        f[[1, 2, 4]] .= NaN
        rdg = ewb_macro_fixture(; fx = f)
        @test isequal(descriptor(de, rdg),
                      descriptor(EWMacroSensitivity(; half_life = 5), rdg; ref = f))
        # With agg_obs = 3, a window of the warm-up averages its finite values.
        da = EWMacroSensitivity(; series = "FX", half_life = 5, agg_obs = 3)
        @test isequal(descriptor(da, rdg),
                      descriptor(EWMacroSensitivity(; half_life = 5, agg_obs = 3), rdg;
                                 ref = f))
        # With agg_obs = 7, the 17 complete windows end at observation 119, so the recursion
        # does not read observation 120, and a gap there is accepted.
        f = copy(ref)
        f[120] = NaN
        rdt = ewb_macro_fixture(; fx = f)
        dt = EWMacroSensitivity(; series = "FX", half_life = 5, agg_obs = 7)
        @test isequal(descriptor(dt, rdt),
                      descriptor(EWMacroSensitivity(; half_life = 5, agg_obs = 7), rdt;
                                 ref = f))
    end
    @testset "The refusals" begin
        de = EWMacroSensitivity(; series = "FX", half_life = 5)
        # Both sources of the series.
        @test_throws ArgumentError descriptor(de, rd; ref = ref)
        # Neither source.
        @test_throws PO.IsNothingError descriptor(EWMacroSensitivity(; half_life = 5), rd)
        # No Exogenous Series.
        rd0 = ReturnsResult(; nx = rd.nx, X = rd.X, pnl = rd.pnl)
        @test_throws PO.IsNothingError descriptor(de, rd0)
        # A name the Exogenous Series does not hold is named in the message.
        err = try
            descriptor(EWMacroSensitivity(; series = "JPY", half_life = 5), rd)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("\"JPY\"", err.msg)
        # A value that is not finite after the warm-up, at observation 50.
        f = copy(ref)
        f[50] = NaN
        rdn = ewb_macro_fixture(; fx = f)
        err = try
            descriptor(de, rdn)
        catch e
            e
        end
        @test err isa PO.IsNonFiniteError
        @test occursin("\"FX\"", err.msg)
        @test occursin("observation 50", err.msg)
        # With agg_obs = 3, the window of observations 49 to 51 still holds two finite
        # values, and the recursion reads their mean, so the gap is accepted.
        da = EWMacroSensitivity(; series = "FX", half_life = 5, agg_obs = 3)
        @test isequal(descriptor(da, rdn),
                      descriptor(EWMacroSensitivity(; half_life = 5, agg_obs = 3), rdn;
                                 ref = f))
        # A window with no finite value is refused, and the message names the window.
        f3 = copy(ref)
        f3[49:51] .= NaN
        err = try
            descriptor(da, ewb_macro_fixture(; fx = f3))
        catch e
            e
        end
        @test err isa PO.IsNonFiniteError
        @test occursin("observations 49 to 51", err.msg)
        # The keyword keeps the rule of the recursion: the state holds its value there.
        D = descriptor(EWMacroSensitivity(; half_life = 5), rdn; ref = f)
        @test isequal(D[50, :], D[49, :])
        # An infinite value is refused on both paths, in the warm-up too, because it is
        # not a gap.
        fi = copy(ref)
        fi[2] = -Inf
        rdi = ewb_macro_fixture(; fx = fi)
        @test_throws PO.IsNonFiniteError descriptor(de, rdi)
        @test_throws PO.IsNonFiniteError descriptor(EWMacroSensitivity(; half_life = 5),
                                                    rdi; ref = fi)
        # An empty name is refused by the constructor.
        err = try
            EWMacroSensitivity(; series = "")
        catch e
            e
        end
        @test err isa PO.IsEmptyError
        @test occursin("Exogenous Series", err.msg)
    end
    @testset "An observation view reads the rows of E that it keeps" begin
        de = EWMacroSensitivity(; series = "FX", half_life = 5)
        rows, cols = 21:100, [1, 3, 4, 8]
        rdv = PO.port_opt_view(rd, rows, cols)
        @test isequal(descriptor(de, rdv),
                      descriptor(EWMacroSensitivity(; half_life = 5), rdv; ref = ref[rows]))
    end
    @testset "Inside CrossSectionalFactorPrior" begin
        de = EWMacroSensitivity(; series = "FX", half_life = 5)
        xe = CompositeExposure(; descriptors = [de], outlier = nothing, scoring = nothing,
                               family = "macro")
        # The exposure is the Descriptor of a direct call with the same series. A direct call
        # of the member reads the benchmark weights from the panel, where the prior writes them.
        pnl = rd.pnl
        rdb = ReturnsResult(; nx = rd.nx, X = rd.X, ne = rd.ne, E = rd.E,
                            pnl = AssetPanel(;
                                             pf = [pnl.pf...,
                                                   NumericPanelField(;
                                                                     name = "benchmark_weights",
                                                                     vals = ones(size(rd.X)))],
                                             amsk = pnl.amsk, emsk = pnl.emsk))
        L = factor_exposure(xe, rdb)
        @test isequal(L, descriptor(EWMacroSensitivity(; half_life = 5), rd; ref = ref))
        for f in (xe, ObservedExposure(; xe = xe, series = "FX"))
            pe = CrossSectionalFactorPrior(;
                                           factors = ["market" => ConstantExposure(),
                                                      "fx" => f], minra = 3, bp = 0,
                                           wa = MarketCapWeights(; p = 0))
            pr = prior(pe, rd)
            k = findfirst(==("fx"), pr.rr.nf)
            @test !isnothing(k)
            @test all(isfinite, pr.mu)
            @test all(isfinite, pr.sigma)
            @test pr.rr.M[:, k] ≈ L[end, :]
        end
    end
end

@testset "EW beta descriptors: EWDownsideBeta" begin
    X = [0.10 0.20 -0.05
         -0.10 -0.02 0.03
         0.05 NaN 0.01
         0.02 -0.03 0.04
         -0.04 0.06 -0.02]
    W = [1.0 3.0 2.0
         2.0 2.0 2.0
         5.0 1.0 4.0
         1.0 1.0 1.0
         2.0 3.0 1.0]
    amsk = trues(5, 3)
    amsk[1, 3] = false
    rd = ewb_hand_panel(X, W; amsk = amsk, emsk = amsk)
    rm = ewb_hand_market(X, W, amsk)
    Xm = ifelse.(amsk, X, NaN)
    @testset "It is the lower partial co-moment of the hand recursion" begin
        for mar in (0.0, 0.001, -0.01)
            D = descriptor(EWDownsideBeta(; half_life = 2, mar = mar), rd)
            B = ewb_hand_downside(Xm, rm, amsk, exp2(-inv(2.0)), 2, mar, 1e-12)
            @test ewb_same(D, ifelse.(amsk, B, NaN))
        end
    end
    @testset "A ready asset reads the market's shortfall through its own gap" begin
        D = descriptor(EWDownsideBeta(; half_life = 2), rd)
        # Asset 2 has no return at observation 3, and the market's downside variance still
        # advances there, so its beta moves even though its co-moment holds its value.
        @test isfinite(D[3, 2])
        @test D[3, 2] != D[2, 2]
    end
    @testset "A market that never falls short leaves every beta at zero" begin
        Xp = fill(0.01, 4, 2)
        rdp = ewb_hand_panel(Xp, ones(4, 2))
        D = descriptor(EWDownsideBeta(; half_life = 2, mar = 0.0), rdp)
        @test all(iszero, D[2:end, :])
    end
end

@testset "EW beta descriptors: EWResidualVolatility" begin
    X = [0.10 0.20 -0.05
         -0.10 0.00 0.03
         0.05 NaN 0.01
         0.02 -0.03 0.04
         -0.04 0.06 -0.02
         0.03 0.01 0.05
         0.01 -0.02 0.02]
    W = fill(2.0, 7, 3)
    amsk = trues(7, 3)
    amsk[4, 1] = false
    rd = ewb_hand_panel(X, W; amsk = amsk, emsk = amsk)
    rm = ewb_hand_market(X, W, amsk)
    Xm = ifelse.(amsk, X, NaN)
    bdecay = exp2(-inv(3.0))
    vdecay = exp2(-inv(2.0))
    B = ewb_hand_beta(Xm, rm, amsk, bdecay, 1, 1e-12; reset = true)
    @testset "It is the volatility of the hand residuals" begin
        D = descriptor(EWResidualVolatility(; half_life = 2, beta_half_life = 3), rd)
        E = ewb_hand_residual(Xm, rm, B, amsk, vdecay, 3)
        @test ewb_same(D, ifelse.(amsk, E, NaN); rtol = 1e-10)
    end
    @testset "The downside form clips every excess above the target" begin
        for mar in (0.0, 0.001)
            D = descriptor(EWResidualDownsideVolatility(; half_life = 2, beta_half_life = 3,
                                                        mar = mar), rd)
            E = ewb_hand_residual(Xm, rm, B, amsk, vdecay, 3; mar = mar)
            @test ewb_same(D, ifelse.(amsk, E, NaN); rtol = 1e-10)
        end
    end
    @testset "An asset that turns inactive restarts both recursions" begin
        D = descriptor(EWResidualVolatility(; half_life = 2, beta_half_life = 3), rd)
        @test isnan(D[4, 1])
        # Asset 1 restarts at observation 5, so it needs three more valid returns before it
        # answers again, and it answers at observation 7.
        @test isnan(D[5, 1])
        @test isnan(D[6, 1])
        @test isfinite(D[7, 1])
    end
    @testset "The warm-up is the longer of the two half-lives" begin
        a = EWResidualVolatility(; half_life = 2, beta_half_life = 9)
        b = EWResidualVolatility(; half_life = 9, beta_half_life = 2)
        @test a.ce.min_obs == 9
        @test b.ce.min_obs == 9
        @test a.ce.decay ≈ exp2(-inv(2.0))
        @test b.ce.decay ≈ exp2(-inv(9.0))
        @test PortfolioOptimisers.ew_variance_estimator(5.0).min_obs == 5
        @test PortfolioOptimisers.ew_variance_estimator(5.0, 8.0).min_obs == 8
    end
    @testset "A residual exists only where the asset is active and its return is finite" begin
        E = PortfolioOptimisers.ew_residual_returns([0.1 0.2; NaN 0.0], [0.1, -0.05],
                                                    [1.0 2.0; 1.0 2.0],
                                                    [true false; true true])
        @test isnan(E[1, 2])
        @test isnan(E[2, 1])
        @test E[1, 1] ≈ 0.0
        @test E[2, 2] ≈ 0.1
        # Integer arguments hold the `NaN` of a missing cell in their floating point type.
        Ei = PortfolioOptimisers.ew_residual_returns([1 2; 3 4], [1, 2], [1 1; 2 2],
                                                     [true false; true true])
        @test isequal(Ei, [0.0 NaN; -1.0 0.0])
    end
    @testset "At the first valid observation the residual is zero up to the floor" begin
        # The beta includes observation 1, so β = (1 - λ) x r_m / ((1 - λ) r_m² + min_val) and
        # the residual is x min_val / ((1 - λ) r_m² + min_val). A warm-up of one observation
        # makes the Descriptor its absolute value there. A large floor keeps the subtraction
        # well conditioned.
        mv = 1e-4
        de = EWResidualVolatility(; half_life = 2, beta_half_life = 3, min_val = mv,
                                  ce = PortfolioOptimisers.ew_variance_estimator(2.0, 1.0))
        D = descriptor(de, rd)
        expected = abs.(Xm[1, :] .* mv ./ ((1 - bdecay) * rm[1]^2 + mv))
        @test isapprox(D[1, :], expected; rtol = 1e-8)
    end
    @testset "A refusal names the half-life the caller passed" begin
        @test_throws r"beta_half_life" EWResidualVolatility(; beta_half_life = Inf)
        @test_throws r"beta_half_life" EWResidualDownsideVolatility(; beta_half_life = 0)
        @test_throws r"half_life" EWResidualVolatility(; half_life = -1)
    end
end

@testset "EW beta descriptors: constructors and their refusals" begin
    @testset "Every archetype refuses a decay outside the open unit interval" begin
        for f in
            (d -> EWBeta(; decay = d, min_obs = 2), d -> EWMacroSensitivity(; decay = d),
             d -> EWDownsideBeta(; decay = d), d -> EWResidualVolatility(; beta_decay = d))
            @test_throws DomainError f(0.0)
            @test_throws DomainError f(1.0)
            @test_throws DomainError f(-0.5)
        end
    end
    @testset "Every archetype refuses a warm-up, a group size or a floor below one" begin
        @test_throws DomainError EWBeta(; decay = 0.5, min_obs = 0)
        @test_throws DomainError EWBeta(; decay = 0.5, min_obs = 2, min_group_size = 0)
        @test_throws DomainError EWBeta(; decay = 0.5, min_obs = 2, min_val = 0.0)
        @test_throws DomainError EWMacroSensitivity(; min_obs = 0)
        @test_throws DomainError EWDownsideBeta(; min_obs = 0)
        @test_throws DomainError EWResidualVolatility(; min_val = 0.0)
    end
    @testset "The aggregation period is at least one observation" begin
        @test_throws DomainError EWBeta(; decay = 0.5, min_obs = 2, agg_obs = 0)
        @test_throws DomainError EWMacroSensitivity(; agg_obs = -1)
        @test EWBeta(; decay = 0.5, min_obs = 2, agg_obs = 1).agg_obs == 1
    end
    @testset "The shrinkage bounds are ordered and lie in the unit interval" begin
        for b in ((-0.1, 1.0), (0.0, 1.1), (0.7, 0.3))
            @test_throws DomainError EWBeta(; decay = 0.5, min_obs = 2, bounds = b)
        end
        @test EWBeta(; decay = 0.5, min_obs = 2, bounds = (0.1, 0.9)).bounds == (0.1, 0.9)
    end
    @testset "A minimum acceptable return must be finite" begin
        @test_throws DomainError EWDownsideBeta(; mar = NaN)
        @test_throws DomainError EWResidualVolatility(; mar = Inf)
    end
    @testset "The capitalisation Panel Field is named, and its name is not empty" begin
        @test_throws PortfolioOptimisers.IsEmptyError EWBeta(; decay = 0.5, min_obs = 2,
                                                             mcap = "")
        @test EWMarketBeta(; mcap = "cap").mcap == "cap"
        @test EWDownsideBeta(; mcap = "cap").mcap == "cap"
        @test EWMacroSensitivity(; mcap = "cap").mcap == "cap"
        @test EWResidualVolatility(; mcap = "cap").mcap == "cap"
    end
    @testset "A named constructor fixes the defaults its half-life converts to" begin
        @test EWMarketBeta().decay ≈ exp2(-inv(60.0))
        @test EWMarketBeta().min_obs == 60
        @test EWMacroSensitivity().decay ≈ exp2(-inv(60.0))
        @test EWDownsideBeta().decay ≈ exp2(-inv(60.0))
        @test EWDownsideBeta().mar == 0.0
        @test EWResidualVolatility().ce.decay ≈ exp2(-inv(40.0))
        @test EWResidualVolatility().ce.min_obs == 60
        @test EWResidualVolatility().beta_decay ≈ exp2(-inv(60.0))
        @test isa(EWResidualVolatility().alg, FullMoment)
        @test isa(EWResidualDownsideVolatility().alg, SemiMoment)
        # A value passed for a converted field is used as it stands.
        @test EWMarketBeta(; half_life = 2, decay = 0.9).decay == 0.9
        @test EWMarketBeta(; half_life = 2, min_obs = 7).min_obs == 7
    end
    @testset "Every member is a Descriptor Estimator" begin
        for de in
            (EWMarketBeta(), EWMacroSensitivity(), EWDownsideBeta(), EWResidualVolatility(),
             EWResidualDownsideVolatility())
            @test isa(de, PortfolioOptimisers.AbstractDescriptorEstimator)
        end
    end
end

@testset "EW beta descriptors: the stored reference cases" begin
    res = synthetic_asset_panel(; n_assets = 12, n_observations = 300, n_industries = 3,
                                rng = StableRNGs.StableRNG(719))
    rd0 = res.rd
    pnl = rd0.pnl
    amsk = Matrix{Bool}(pnl.amsk)
    T, N = size(amsk)
    # The same forty gaps the reference implementation was driven on. The generator leaves
    # none inside the active mask, and a gap is what holds a recursion.
    Xg = Matrix{Float64}(rd0.X)
    let g = StableRNGs.StableRNG(7192), holes = 0
        while holes < 40
            t = rand(g, 1:T)
            i = rand(g, 1:N)
            if amsk[t, i] && isfinite(Xg[t, i])
                Xg[t, i] = NaN
                holes += 1
            end
        end
    end
    rd = ReturnsResult(; nx = rd0.nx, X = Xg, nf = rd0.nf, F = rd0.F, nb = rd0.nb,
                       B = rd0.B, ts = rd0.ts, iv = rd0.iv, pnl = rd0.pnl)
    ref = 0.01 .* randn(StableRNGs.StableRNG(7191), T)
    ref[7] = NaN
    ref[8] = NaN
    ref[123] = NaN
    @testset "Shape, the inactive fill, and finite-or-NaN everywhere" begin
        des = ("EWMarketBeta" => EWMarketBeta(; half_life = 5),
               "EWDownsideBeta" => EWDownsideBeta(; half_life = 5),
               "EWResidualVolatility" =>
                   EWResidualVolatility(; half_life = 5, beta_half_life = 8),
               "EWResidualDownsideVolatility" =>
                   EWResidualDownsideVolatility(; half_life = 5, beta_half_life = 8))
        for (name, de) in des
            D = descriptor(de, rd)
            @test size(D) == (T, N)
            @test all(k -> amsk[k] || isnan(D[k]), CartesianIndices(D))
            @test all(d -> isfinite(d) || isnan(d), D)
        end
        Dm = descriptor(EWMacroSensitivity(; half_life = 5), rd; ref = ref)
        @test size(Dm) == (T, N)
        @test all(k -> amsk[k] || isnan(Dm[k]), CartesianIndices(Dm))
    end
    @testset "The shrunk beta matches the stored case cell by cell" begin
        D = descriptor(EWMarketBeta(; half_life = 5, group = "industry", min_group_size = 2,
                                    bounds = (0.1, 0.9)), rd)
        E = Matrix(CSV.read(joinpath(@__DIR__, "assets/EWMarketBeta.csv.gz"), DataFrame))
        @test size(D) == size(E)
        @test isequal(isnan.(D), isnan.(E))
        @test D[isfinite.(E)] ≈ E[isfinite.(E)]
    end
    @testset "The aggregated macro sensitivity matches the stored case cell by cell" begin
        D = descriptor(EWMacroSensitivity(; half_life = 5, agg_obs = 3), rd; ref = ref)
        E = Matrix(CSV.read(joinpath(@__DIR__, "assets/EWMacroSensitivity.csv.gz"),
                            DataFrame))
        @test size(D) == size(E)
        @test isequal(isnan.(D), isnan.(E))
        @test D[isfinite.(E)] ≈ E[isfinite.(E)]
        # The same series named in the Exogenous Series gives the same case (#1365). Its
        # gap at observation 123 falls in a window that holds two finite values.
        rde = ReturnsResult(; nx = rd.nx, X = rd.X, ne = ["FX"], E = reshape(ref, :, 1),
                            pnl = rd.pnl)
        Ds = descriptor(EWMacroSensitivity(; series = "FX", half_life = 5, agg_obs = 3),
                        rde)
        @test isequal(isnan.(Ds), isnan.(E))
        @test Ds[isfinite.(E)] ≈ E[isfinite.(E)]
    end
end

@testset "EW beta descriptors: the shrinkage is the closed form, written out" begin
    @testset "Unequal weights, a small group, a missing group and a NaN beta" begin
        b = [0.6, 1.4, 1.1, 0.9, 1.8, 0.7, 1.2, NaN]
        bev = [0.02, 0.05, 0.01, 0.03, 0.04, 0.02, 0.06, 0.01]
        L = [1, 1, 1, 2, 2, 3, PortfolioOptimisers.CS_MISSING_GROUP, 1]
        w = [3.0, 1.0, 2.0, 5.0, 1.0, 2.0, 4.0, 1.0]
        for bounds in ((0.0, 1.0), (0.2, 0.7))
            s = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 2, bounds, 1e-12)
            @test ewb_same(s[:, :], ewb_hand_shrink(b, bev, L, w, 2, bounds, 1e-12)[:, :];
                           rtol = 1e-14)
        end
        # The asset with no group keeps its raw beta, and the NaN beta stays NaN.
        s = PortfolioOptimisers.ew_beta_shrink(b, bev, L, w, 2, (0.0, 1.0), 1e-12)
        @test s[7] == b[7]
        @test isnan(s[8])
    end
    @testset "The grouped Descriptor on an aggregated clock is the hand pipeline" begin
        X = [0.10 0.20 -0.05 0.04
             -0.10 0.00 0.03 -0.02
             0.05 NaN 0.01 0.06
             0.02 -0.03 0.04 -0.01
             -0.04 0.06 -0.02 0.03
             0.03 0.01 0.05 -0.04
             0.06 -0.02 0.02 0.01
             -0.01 0.04 -0.03 0.05]
        W = [1.0 3.0 2.0 1.5
             2.0 2.0 2.0 1.0
             5.0 1.0 4.0 2.0
             1.0 1.0 1.0 3.0
             2.0 3.0 1.0 1.0
             1.0 2.0 3.0 2.0
             3.0 1.0 2.0 1.0
             2.0 2.0 1.0 4.0]
        ind = repeat(["a" "a" "b" "b"], 8, 1)
        rd = ewb_hand_panel(X, W; industry = ind)
        D = descriptor(EWMarketBeta(; half_life = 2, agg_obs = 2, group = "industry",
                                    min_group_size = 2, bounds = (0.1, 0.9)), rd)
        decay, min_obs, min_val, en = exp2(-inv(2.0)), 2, 1e-12, 4.0
        # Each window is the mean of the finite entries of its two observations.
        rm = ewb_hand_market(X, W, trues(8, 4))
        Xa = [sum(filter(isfinite, X[(2k - 1):(2k), i])) /
              count(isfinite, X[(2k - 1):(2k), i]) for k in 1:4, i in 1:4]
        rma = [(rm[2k - 1] + rm[2k]) / 2 for k in 1:4]
        Ba = ewb_hand_beta(Xa, rma, trues(4, 4), decay, min_obs, min_val)
        # The market variance, and the residual variance against the previous window's beta.
        Vm, Vr = zeros(4), zeros(4, 4)
        let mu = 0.0, v = 0.0, r = zeros(4)
            for k in 1:4
                d = rma[k] - mu
                mu = decay * mu + (1 - decay) * rma[k]
                v = decay * v + (1 - decay) * d * d
                Vm[k] = v
                if k - 1 >= min_obs
                    for i in 1:4
                        e = Xa[k, i] - Ba[k - 1, i] * rma[k]
                        r[i] = decay * r[i] + (1 - decay) * e * e
                    end
                end
                Vr[k, :] = r
            end
        end
        E = fill(NaN, 8, 4)
        s = fill(NaN, 4)
        for t in 1:8
            k = div(t, 2)
            if iseven(t) && k >= min_obs
                s = ewb_hand_shrink(Ba[k, :], Vr[k, :] ./ (en * (Vm[k] + min_val)),
                                    [1, 1, 2, 2], W[t, :], 2, (0.1, 0.9), min_val)
            end
            if k >= min_obs
                E[t, :] = s
            elseif k >= 1
                E[t, :] = Ba[k, :]
            end
        end
        @test ewb_same(D, E; rtol = 1e-12)
        @test all(isfinite, D[4:8, :])
        # The shrinkage moved every beta of the last window.
        @test all(D[8, :] .!= Ba[4, :])
    end
end

@testset "EW beta descriptors: the number type comes from the data" begin
    Xi = [1 2
          -1 0
          1 1
          2 -1
          0 3]
    W = [1.0 2.0
         3.0 4.0
         5.0 6.0
         1.0 1.0
         2.0 1.0]
    amsk = trues(5, 2)
    amsk[3, 2] = false
    rdi = ewb_hand_panel(Xi, W; amsk = amsk)
    rdf = ewb_hand_panel(Float64.(Xi), W; amsk = amsk)
    @testset "Integer returns give the Descriptor of the same returns as floats" begin
        @test isequal(descriptor(EWMarketBeta(; half_life = 1), rdi),
                      descriptor(EWMarketBeta(; half_life = 1), rdf))
        @test isequal(descriptor(EWMarketBeta(; half_life = 1, agg_obs = 2), rdi),
                      descriptor(EWMarketBeta(; half_life = 1, agg_obs = 2), rdf))
        @test isequal(descriptor(EWDownsideBeta(; half_life = 1), rdi),
                      descriptor(EWDownsideBeta(; half_life = 1), rdf))
        @test isequal(descriptor(EWMacroSensitivity(; half_life = 1, agg_obs = 2), rdi;
                                 ref = [1, 0, -1, 2, 1]),
                      descriptor(EWMacroSensitivity(; half_life = 1, agg_obs = 2), rdf;
                                 ref = [1.0, 0.0, -1.0, 2.0, 1.0]))
    end
    @testset "The helpers float an integer and keep every other type" begin
        A = PortfolioOptimisers.ew_agg_series([1 2; 3 4], 2)
        @test A == [2.0 3.0]
        @test eltype(A) == Float64
        R = PortfolioOptimisers.ew_agg_series([1//2 1//3; 1//4 1//5], 2)
        @test R == [3//8 4//15]
        @test eltype(R) == Rational{Int}
        @test eltype(PortfolioOptimisers.ew_agg_series(Float32[1 2; 3 4], 2)) == Float32
        @test isequal(PortfolioOptimisers.ew_beta_expand([1 2; 3 4], 3, 2),
                      [NaN NaN; 1.0 2.0; 1.0 2.0])
        @test PortfolioOptimisers.ew_beta_expand([1//2 1//3; 1//4 1//5], 2, 1) ==
              [1//2 1//3; 1//4 1//5]
        B = PortfolioOptimisers.ew_downside_beta_series([1 2; -1 0], [1, -1], 0.5, 1, 0,
                                                        1e-12)
        @test eltype(B) == Float64
        @test B ≈
              PortfolioOptimisers.ew_downside_beta_series([1.0 2.0; -1.0 0.0], [1.0, -1.0],
                                                          0.5, 1, 0.0, 1e-12) nans = true
        M = PortfolioOptimisers.ew_macro_sensitivity_series([1 2; -1 0; 2 1], [1, -1, 0],
                                                            [1, 0, 2], 0.5, 1, 1e-12)
        @test isequal(M,
                      PortfolioOptimisers.ew_macro_sensitivity_series([1.0 2.0; -1.0 0.0;
                                                                       2.0 1.0],
                                                                      [1.0, -1.0, 0.0],
                                                                      [1.0, 0.0, 2.0], 0.5,
                                                                      1, 1e-12))
        Bi, Vi = PortfolioOptimisers.ew_beta_series([1 2; -1 0; 2 1], [1, -1, 0], 0.5, 1,
                                                    1e-12)
        Bf, Vf = PortfolioOptimisers.ew_beta_series([1.0 2.0; -1.0 0.0; 2.0 1.0],
                                                    [1.0, -1.0, 0.0], 0.5, 1, 1e-12)
        @test isequal(Bi, Bf)
        @test isequal(Vi, Vf)
        V = PortfolioOptimisers.ew_beta_residual_variance([1 2; -1 0; 2 1], [1, -1, 0],
                                                          [1 2; 1 1; 1 1], 0.5, 1)
        @test V[2, :] ≈ [0.5 * (-1 - 1 * -1)^2, 0.5 * (0 - 2 * -1)^2]
        @test eltype(V) == Float64
    end
end

# The beta series from its closed form in BigFloat: each mean, variance and covariance is a sum
# over the valid observations of its asset since the last restart, so it shares no loop with
# the recursion it checks. A restart happens where an asset turns inactive, and a cell that
# writes no new beta holds the last one.
function ewb_closed_beta(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                         amsk::AbstractMatrix{Bool}, decay::Real, min_obs::Integer,
                         min_val::Real)
    lam = BigFloat(decay)
    T, N = size(X)
    r = BigFloat.(rm)
    # `mum[s]` is the market mean after `s - 1` observations, the mean a deviation at `s` uses.
    mum = [(1 - lam) * sum((lam^(s - 1 - u) * r[u] for u in 1:(s - 1)); init = zero(lam))
           for s in 1:T]
    Vm = [(1 - lam) * sum(lam^(t - s) * (r[s] - mum[s])^2 for s in 1:t) for t in 1:T]
    B = fill(BigFloat(NaN), T, N)
    for i in 1:N
        start = 1
        b = BigFloat(NaN)
        for t in 1:T
            if t > 1 && amsk[t - 1, i] && !amsk[t, i]
                start = t
            end
            v = [s for s in start:t if amsk[s, i] && isfinite(X[s, i])]
            k = length(v)
            if k > 0 && last(v) == t && t >= min_obs && k >= min_obs
                x = BigFloat.(X[v, i])
                mu = [(1 - lam) *
                      sum((lam^(j - 1 - l) * x[l] for l in 1:(j - 1)); init = zero(lam))
                      for j in 1:k]
                C = (1 - lam) *
                    sum(lam^(k - j) * (x[j] - mu[j]) * (r[v[j]] - mum[v[j]]) for j in 1:k)
                b = C / (Vm[t] + min_val)
            end
            B[t, i] = b
        end
    end
    return B, Vm
end

@testset "The beta series is the closed form of its recursion (#726)" begin
    # Asset 1 has a gap at observation 3, and asset 2 turns inactive at observation 4. The
    # warm-up of two observations delays the first beta and the first beta after the restart.
    X = [0.10 0.20
         -0.10 0.00
         NaN 0.05
         0.02 -0.03
         -0.04 0.06
         0.03 0.01]
    rm = [0.15, -0.05, 0.035, -0.005, 0.01, 0.02]
    amsk = [true true; true true; true true; true false; true true; true true]
    decay, min_obs, min_val = 0.6, 2, 1e-4
    B, Vm = PortfolioOptimisers.ew_beta_series(X, rm, decay, min_obs, min_val, amsk)
    Bc, Vc = ewb_closed_beta(X, rm, amsk, decay, min_obs, min_val)
    @test ewb_same(B, Float64.(Bc); rtol = 1e-12)
    @test Vm ≈ Float64.(Vc) rtol = 1e-12
    @test all(isnan, B[1, :])
    # A gap holds the beta, and a restart does not clear it: the first valid return after the
    # restart counts one observation, below the warm-up, so the old beta holds one more step.
    @test B[3, 1] == B[2, 1]
    @test B[5, 2] == B[3, 2]
    @test B[6, 2] != B[5, 2]
    # With no mask, no state restarts: the inactive cell is an ordinary finite return.
    B0, _ = PortfolioOptimisers.ew_beta_series(X, rm, decay, min_obs, min_val)
    Bc0, _ = ewb_closed_beta(X, rm, trues(size(X)), decay, min_obs, min_val)
    @test ewb_same(B0, Float64.(Bc0); rtol = 1e-12)
end

@testset "A negative capitalisation enters the market return as it stands (#726)" begin
    X = [0.10 0.20 -0.05; 0.02 -0.03 0.04]
    W = [1.0 -0.5 2.0; 1.0 1.0 1.0]
    rd = ewb_hand_panel(X, W)
    rm = PortfolioOptimisers.market_return_series(rd, "market_cap")
    @test rm[1] ≈ (1.0 * 0.10 - 0.5 * 0.20 + 2.0 * -0.05) / 2.5
    # A total weight of zero divides no mean, even where every pair is finite.
    Wz = [1.0 -1.0 0.0; 1.0 1.0 1.0]
    @test_throws ArgumentError PortfolioOptimisers.market_return_series(ewb_hand_panel(X,
                                                                                       Wz),
                                                                        "market_cap")
end
