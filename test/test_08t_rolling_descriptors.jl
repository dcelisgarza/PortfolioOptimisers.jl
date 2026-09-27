#=
Check `src/05_Moments/32_CrossSectionalFactorModel/04_FactorExposures/07_RollingDescriptors.jl` against the contract its
docstrings state, and against the reference implementation's own rolling descriptors.
Issue #720, map #643.

FIVE CONVENTIONS SHAPE THE PROBES.

1. RETURNS ARE NOT A PANEL FIELD. A rolling Descriptor reads `rd.X`, and the active mask of
   the Asset Panel beside it. It reads no column of the feature matrix.

2. A WINDOW THAT IS NOT WHOLLY ACTIVE IS `NaN`. One rule covers three cases: the warm-up at
   the start of the sample, an asset that lists late, and a gap inside a listing. The hand
   panels below carry a return on an inactive cell on purpose, so a Descriptor that forgets
   the active mask fails.

3. A MISSING RETURN IS NOT A LOSS. Inside an active window, `RollingLogReturn` counts a
   missing return as a zero contribution, and `RollingMax` ignores it. A window whose
   returns are all missing is `NaN` for `RollingMax` and zero for `RollingLogReturn`.

4. A RETURN AT OR BELOW `-1` IS A REFUSAL, NOT A `NaN`. The logarithm is undefined there, so
   `RollingLogReturn` raises on the whole matrix. `RollingMax` takes no logarithm and needs
   no such rule.

5. AN INFINITE RETURN IS A REFUSAL, FOR BOTH ESTIMATORS. The window sum is one difference of
   two cumulative sums, so one infinite return would make every later window `NaN`, also a
   window that does not hold it. The reference implementation refuses it too (#741).

The last two testsets run the estimators on the synthetic panel of `test06c_setup.jl` and
compare them with the reference implementation. The parity is pinned twice: once as a
re-derivation written out by hand in plain Julia, and once as the reference's own output at
five observations, stored as a literal.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))

# A small hand panel of returns. The Asset Panel needs one Panel Field, and no rolling
# Descriptor reads it, so a constant one carries the masks and nothing else.
function rolling_hand_panel(X::AbstractMatrix{<:Real};
                            amsk::AbstractMatrix{Bool} = trues(size(X)...),
                            emsk::AbstractMatrix{Bool} = amsk)
    T, N = size(X)
    res = asset_panel([NumericPanelInput(; name = "market_cap", vals = ones(T, N))];
                      amsk = amsk, emsk = emsk)
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:N], X = X, pnl = res)
end

# The window sum of log returns, written out by hand: a missing return contributes zero, and
# the value is `NaN` unless every observation of the window is active.
function rolling_expected_sum(X::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                              window::Integer, skip::Integer)
    T, N = size(X)
    E = fill(NaN, T, N)
    for i in 1:N, t in (skip + window):T
        rows = (t - skip - window + 1):(t - skip)
        if all(k -> amsk[k, i], rows)
            E[t, i] = sum(k -> isnan(X[k, i]) ? 0.0 : log1p(X[k, i]), rows)
        end
    end
    E[.!amsk] .= NaN
    return E
end

# The rolling maximum, written out by hand: a missing return is ignored, and the value is
# `NaN` unless every observation of the window is active and one return exists.
function rolling_expected_max(X::AbstractMatrix{<:Real}, amsk::AbstractMatrix{Bool},
                              window::Integer)
    T, N = size(X)
    E = fill(NaN, T, N)
    for i in 1:N, t in window:T
        rows = (t - window + 1):t
        vals = [X[k, i] for k in rows if !isnan(X[k, i])]
        if all(k -> amsk[k, i], rows) && !isempty(vals)
            E[t, i] = maximum(vals)
        end
    end
    E[.!amsk] .= NaN
    return E
end

# Compare a log-return Descriptor with the sum written out by hand. The NaN pattern must
# agree exactly. The values agree only to the last bit, because the estimator differences one
# cumulative sum where the re-derivation adds the window up directly, and the two orders of
# summation round differently. `RollingMax` takes no sum, so its probes compare exactly.
function rolling_agrees(D::AbstractMatrix{<:Real}, E::AbstractMatrix{<:Real})
    if !(isequal(isnan.(D), isnan.(E)))
        return false
    end
    return all(k -> isapprox(D[k], E[k]; rtol = 1e-13, atol = 1e-15), findall(.!isnan.(D)))
end

@testset "Rolling Descriptor constructors and their refusals" begin
    @testset "The three named Descriptors fix the census defaults" begin
        @test isa(RollingMomentum(), RollingLogReturn)
        @test isa(Reversal(), RollingLogReturn)
        @test isa(MaxReturn(), RollingMax)
        for de in (RollingMomentum(), Reversal(), MaxReturn())
            @test isa(de, PortfolioOptimisers.AbstractDescriptorEstimator)
        end
        @test RollingMomentum().window == 252
        @test RollingMomentum().skip == 21
        @test RollingMomentum().sign == 1
        @test RollingMomentum().exponentiate == false
        @test Reversal().window == 21
        @test Reversal().skip == 0
        @test Reversal().sign == -1
        @test Reversal().exponentiate == false
        @test MaxReturn().window == 21
    end
    @testset "Every keyword of a named Descriptor overrides its default" begin
        de = RollingMomentum(; window = 126, skip = 5, sign = -1, exponentiate = true)
        @test de.window == 126
        @test de.skip == 5
        @test de.sign == -1
        @test de.exponentiate
        rv = Reversal(; window = 5, skip = 2, sign = 1, exponentiate = true)
        @test rv.window == 5
        @test rv.skip == 2
        @test rv.sign == 1
        @test rv.exponentiate
        @test MaxReturn(; window = 5).window == 5
    end
    @testset "A window, a skip and a sign outside their domain are refused" begin
        @test_throws DomainError RollingLogReturn(; window = 0)
        @test_throws DomainError RollingLogReturn(; window = -3)
        @test_throws DomainError RollingLogReturn(; window = 2, skip = -1)
        @test_throws DomainError RollingLogReturn(; window = 2, sign = 2)
        @test_throws DomainError RollingLogReturn(; window = 2, sign = 0)
        @test_throws DomainError RollingLogReturn(; window = 2, sign = -0.5)
        @test_throws DomainError RollingMax(; window = 0)
        @test_throws DomainError RollingMax(; window = -1)
        # The two the sign accepts, in both the integer and the floating point spelling.
        @test PortfolioOptimisers.assert_rolling_sign(1) === nothing
        @test PortfolioOptimisers.assert_rolling_sign(-1.0) === nothing
    end
    @testset "A carrier without returns or without an Asset Panel is refused" begin
        @test_throws PortfolioOptimisers.IsNothingError descriptor(Reversal(; window = 1),
                                                                   ReturnsResult())
        @test_throws PortfolioOptimisers.IsNothingError descriptor(MaxReturn(; window = 1),
                                                                   ReturnsResult())
        rd = ReturnsResult(; nx = ["A", "B"], X = [0.1 0.1; 0.2 0.2])
        @test_throws PortfolioOptimisers.IsNothingError descriptor(Reversal(; window = 1),
                                                                   rd)
        @test_throws PortfolioOptimisers.IsNothingError descriptor(MaxReturn(; window = 1),
                                                                   rd)
    end
end

@testset "RollingLogReturn on a hand panel" begin
    X = [0.10 -0.05; 0.20 0.10; -0.10 0.05; 0.05 NaN; 0.02 0.03]
    T, N = size(X)
    @testset "The window sum equals the sum of log1p written out by hand" begin
        for (window, skip) in ((1, 0), (2, 0), (3, 0), (2, 1), (1, 3))
            de = RollingLogReturn(; window = window, skip = skip)
            rd = rolling_hand_panel(X)
            @test rolling_agrees(descriptor(de, rd),
                                 rolling_expected_sum(X, trues(T, N), window, skip))
        end
    end
    @testset "The skip moves the window back and lengthens the warm-up" begin
        rd = rolling_hand_panel(X)
        plain = descriptor(RollingLogReturn(; window = 2, skip = 0), rd)
        moved = descriptor(RollingLogReturn(; window = 2, skip = 1), rd)
        @test all(isnan, moved[1:2, :])
        @test isequal(moved[3:T, :], plain[2:(T - 1), :])
        # A skip that leaves no room for a whole window gives a Descriptor of NaN.
        @test all(isnan, descriptor(RollingLogReturn(; window = 2, skip = 4), rd))
    end
    @testset "The sign negates the sum, and Reversal is the negated momentum" begin
        rd = rolling_hand_panel(X)
        up = descriptor(RollingLogReturn(; window = 2, sign = 1), rd)
        down = descriptor(RollingLogReturn(; window = 2, sign = -1), rd)
        @test isequal(down, -up)
        @test isequal(descriptor(Reversal(; window = 2), rd), down)
        @test isequal(descriptor(RollingMomentum(; window = 2, skip = 0), rd), up)
    end
    @testset "The exponentiate flag reads the simple return of the window" begin
        rd = rolling_hand_panel(X)
        S = descriptor(RollingLogReturn(; window = 2), rd)
        E = descriptor(RollingLogReturn(; window = 2, exponentiate = true), rd)
        @test isequal(E, expm1.(S))
        # Two returns compound, so the simple return of the window is their product.
        @test E[2, 1] ≈ (1 + X[1, 1]) * (1 + X[2, 1]) - 1
        # The sign is applied before the exponentiation, as the reference does.
        R = descriptor(Reversal(; window = 2, exponentiate = true), rd)
        @test isequal(R, expm1.(-S))
    end
    @testset "An active observation with a missing return contributes zero" begin
        rd = rolling_hand_panel(X)
        D = descriptor(RollingLogReturn(; window = 2), rd)
        @test D[4, 2] ≈ log1p(X[3, 2])
        @test D[5, 2] ≈ log1p(X[5, 2])
        # A window whose returns are all missing sums to zero, not to NaN.
        Y = [0.1 NaN; 0.2 NaN; 0.3 NaN]
        @test descriptor(RollingLogReturn(; window = 2), rolling_hand_panel(Y))[3, 2] == 0
    end
    @testset "A window that is not wholly active is NaN across a delisting" begin
        amsk = trues(T, N)
        amsk[3, 2] = false
        rd = rolling_hand_panel(X; amsk = amsk)
        D = descriptor(RollingLogReturn(; window = 2), rd)
        @test rolling_agrees(D, rolling_expected_sum(X, amsk, 2, 0))
        # The inactive observation itself, and the one after it, both lose their window.
        @test isnan(D[3, 2])
        @test isnan(D[4, 2])
        @test !isnan(D[5, 2])
        # A late listing loses the whole warm-up of the window.
        late = trues(T, N)
        late[1:2, 1] .= false
        Dl = descriptor(RollingLogReturn(; window = 2), rolling_hand_panel(X; amsk = late))
        @test all(isnan, Dl[1:3, 1])
        @test !isnan(Dl[4, 1])
    end
    @testset "A return at or below -1 is refused" begin
        for bad in (-1.0, -1.5)
            Y = [0.1 0.1; 0.2 bad; 0.3 0.3]
            @test_throws DomainError descriptor(Reversal(; window = 1),
                                                rolling_hand_panel(Y))
        end
        # The refusal reads the whole matrix, so a bad return outside every window still
        # raises, and an inactive cell does not exempt it.
        Y = [-1.0 0.1; 0.2 0.2; 0.3 0.3]
        amsk = trues(3, 2)
        amsk[1, 1] = false
        @test_throws DomainError descriptor(Reversal(; window = 1),
                                            rolling_hand_panel(Y; amsk = amsk))
        # RollingMax takes no logarithm, so it answers where RollingLogReturn refuses.
        @test descriptor(RollingMax(; window = 1),
                         rolling_hand_panel([0.1 0.1; 0.2 -1.0; 0.3 0.3]))[2, 2] == -1
    end
    @testset "With a skip, an inactive current observation is NaN" begin
        # The window of observation 4 with a skip of 2 is rows 1 and 2, which are active, so
        # only the active mask of observation 4 itself makes the value NaN.
        amsk = trues(T, N)
        amsk[4, 2] = false
        D = descriptor(RollingLogReturn(; window = 2, skip = 2),
                       rolling_hand_panel(X; amsk = amsk))
        @test isnan(D[4, 2])
        @test D[4, 1] ≈ log1p(X[1, 1]) + log1p(X[2, 1])
    end
end

@testset "The rolling Descriptors refuse an infinite return (#741)" begin
    # Before the refusal, one infinite return in row 2 made rows 4 and 5 of RollingLogReturn
    # NaN, though neither window holds row 2: the difference of two infinite cumulative sums
    # is NaN.
    for bad in (Inf, -Inf)
        Y = [0.1 0.1; bad 0.2; 0.3 0.3; 0.1 0.1; 0.2 0.2]
        for de in (RollingLogReturn(; window = 2), Reversal(; window = 2),
                   RollingMax(; window = 2), MaxReturn(; window = 2))
            @test_throws DomainError descriptor(de, rolling_hand_panel(Y))
        end
    end
    # The refusal reads the whole matrix, so an inactive cell does not exempt it.
    Y = [Inf 0.1; 0.2 0.2; 0.3 0.3]
    amsk = trues(3, 2)
    amsk[1, 1] = false
    @test_throws DomainError descriptor(RollingMax(; window = 1),
                                        rolling_hand_panel(Y; amsk = amsk))
    msg = try
        descriptor(MaxReturn(; window = 1), rolling_hand_panel([0.1 0.1; 0.2 Inf]))
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("observation 2 for asset 2", msg)
end

@testset "The rolling Descriptors derive their number type from the returns (#741)" begin
    # An integer panel raised InexactError: Int64(NaN) at the first NaN of the warm-up. It
    # is now floated, and gives the answer of the same panel in Float64 bit for bit.
    Xi = [0 1; 2 0; 1 3; 4 2; 0 1]
    Xf = Float64.(Xi)
    for de in
        (RollingMax(; window = 2), MaxReturn(; window = 3), RollingLogReturn(; window = 2),
         Reversal(; window = 1))
        Di = descriptor(de, rolling_hand_panel(Xi))
        @test eltype(Di) == Float64
        @test isequal(Di, descriptor(de, rolling_hand_panel(Xf)))
    end
    # A Float32 panel keeps a Float32 Descriptor.
    X32 = Float32[0.1 0.2; 0.05 -0.1; 0.3 0.0]
    for de in (RollingLogReturn(; window = 2), RollingMax(; window = 2))
        @test eltype(descriptor(de, rolling_hand_panel(X32))) == Float32
    end
end

@testset "The named Descriptors agree with the reference implementation on a panel with gaps" begin
    #=
    The reference implementation, installed from disk at v1.2.4, ran on this 8 x 3 panel. Asset
    1 misses the return of row 4 inside an active listing. Asset 2 is inactive in rows 3 and 4,
    a gap inside its listing. Asset 3 lists late, at row 3, and misses the return of row 6. The
    reference refuses a finite value on an inactive cell, so those cells hold NaN here.
    `mom_w2_s2` pins the rule on the current observation: row 4 of asset 2 has an active
    window, rows 1 and 2, and it is NaN because observation 4 is inactive.
    =#
    X = [0.010 -0.020 0.030
         0.025 0.015 -0.010
         -0.030 0.040 0.020
         NaN -0.010 0.005
         0.015 0.020 -0.025
         0.040 -0.035 NaN
         -0.012 0.018 0.012
         0.022 0.030 -0.008]
    amsk = trues(8, 3)
    amsk[3:4, 2] .= false
    amsk[1:2, 3] .= false
    X[.!amsk] .= NaN
    rd = rolling_hand_panel(X; amsk = amsk)
    ref_mom_w3_s2 = [NaN NaN NaN
                     NaN NaN NaN
                     NaN NaN NaN
                     NaN NaN NaN
                     0.004183735958831036 NaN NaN
                     -0.005766594894337047 NaN NaN
                     -0.015570594990957892 NaN -0.0005276391770710903
                     0.05410932564703195 NaN -0.020330266473250803]
    ref_mom_w2_s2 = [NaN NaN NaN
                     NaN NaN NaN
                     NaN NaN NaN
                     0.03464294344353958 NaN NaN
                     -0.005766594894337047 NaN NaN
                     -0.030459207484708546 NaN 0.024790168807218786
                     0.014888612493750654 NaN -0.020330266473250803
                     0.05410932564703195 -0.015824550346971413 -0.025317807984289876]
    ref_mom_w2_s0_exp = [NaN NaN NaN
                         0.03525 -0.005300000000000001 NaN
                         -0.0057500000000000025 NaN NaN
                         -0.03 NaN 0.0251
                         0.015 NaN -0.020125
                         0.0556 -0.015700000000000002 -0.025
                         0.02752 -0.017630000000000003 0.012
                         0.009736000000000007 0.04854 0.0039039999999999995]
    ref_rev_w2 = [NaN NaN NaN
                  -0.03464294344353958 0.005314094823768795 NaN
                  0.005766594894337047 NaN NaN
                  0.030459207484708546 NaN -0.024790168807218786
                  -0.014888612493750654 NaN 0.020330266473250803
                  -0.05410932564703195 0.015824550346971413 0.025317807984289876
                  -0.02714813191901206 0.017787259514820127 -0.011928570865273802
                  -0.00968891054724346 -0.0473987203698754 -0.003896399168009542]
    ref_max_w3 = [NaN NaN NaN
                  NaN NaN NaN
                  0.025 NaN NaN
                  0.025 NaN NaN
                  0.015 NaN 0.02
                  0.04 NaN 0.005
                  0.04 0.02 0.012
                  0.04 0.03 0.012]
    cases = ((RollingMomentum(; window = 3, skip = 2), ref_mom_w3_s2),
             (RollingMomentum(; window = 2, skip = 2), ref_mom_w2_s2),
             (RollingMomentum(; window = 2, skip = 0, exponentiate = true),
              ref_mom_w2_s0_exp), (Reversal(; window = 2), ref_rev_w2),
             (MaxReturn(; window = 3), ref_max_w3))
    for (de, ref) in cases
        @test rolling_agrees(descriptor(de, rd), ref)
    end
    @test isequal(descriptor(MaxReturn(; window = 3), rd), ref_max_w3)
    @test isnan(ref_mom_w2_s2[4, 2])
end

@testset "RollingMax on a hand panel" begin
    X = [0.10 -0.05; 0.20 0.10; -0.10 0.05; 0.05 NaN; 0.02 0.03]
    T, N = size(X)
    @testset "The maximum equals the one written out by hand" begin
        for window in (1, 2, 3, 5)
            rd = rolling_hand_panel(X)
            @test isequal(descriptor(RollingMax(; window = window), rd),
                          rolling_expected_max(X, trues(T, N), window))
        end
    end
    @testset "A missing return is ignored, and an all-missing window is NaN" begin
        rd = rolling_hand_panel(X)
        D = descriptor(RollingMax(; window = 2), rd)
        @test D[4, 2] == X[3, 2]
        Y = [0.1 NaN; 0.2 NaN; 0.3 NaN]
        @test isnan(descriptor(RollingMax(; window = 2), rolling_hand_panel(Y))[3, 2])
    end
    @testset "A window that is not wholly active is NaN, and so is the warm-up" begin
        amsk = trues(T, N)
        amsk[3, 2] = false
        rd = rolling_hand_panel(X; amsk = amsk)
        D = descriptor(RollingMax(; window = 2), rd)
        @test isequal(D, rolling_expected_max(X, amsk, 2))
        @test isnan(D[3, 2])
        @test isnan(D[4, 2])
        @test all(isnan, D[1, :])
        # A window of one has no warm-up at all.
        @test !any(isnan, descriptor(RollingMax(; window = 1), rolling_hand_panel(X))[1, :])
    end
    @testset "A window of one is the return itself, which the reference refuses" begin
        # The reference implementation raises for `window <= 1`. The port answers, which
        # adds a mode rather than removes one. Restore the refusal by raising the bound of
        # `assert_gt0(window, :window)` in the inner constructor of `RollingMax`.
        rd = rolling_hand_panel(X)
        D = descriptor(RollingMax(; window = 1), rd)
        @test isequal(D, X)
        @test D[1, 1] == X[1, 1]
        @test isnan(D[4, 2])
    end
end

@testset "Every rolling Descriptor runs on the synthetic panel" begin
    rng = StableRNG(717)
    rd = synthetic_asset_panel(; n_assets = 12, n_observations = 300, n_industries = 3,
                               rng = rng).rd
    amsk = rd.pnl.amsk
    T, N = size(amsk)
    X = Matrix{Float64}(rd.X)
    @testset "Shape, the inactive fill, and finite-or-NaN everywhere" begin
        for de in (RollingMomentum(; window = 63, skip = 21), Reversal(), MaxReturn(),
                   RollingMax(; window = 5))
            D = descriptor(de, rd)
            @test size(D) == (T, N)
            @test all(isnan, D[.!amsk])
            @test all(x -> isnan(x) || isfinite(x), D)
            @test count(!isnan, D) > 0
        end
    end
    @testset "Each estimator equals the formula written out by hand" begin
        for (window, skip) in ((63, 21), (21, 0), (252, 21), (1, 0))
            de = RollingLogReturn(; window = window, skip = skip)
            @test rolling_agrees(descriptor(de, rd),
                                 rolling_expected_sum(X, amsk, window, skip))
        end
        @test rolling_agrees(descriptor(Reversal(), rd),
                             -rolling_expected_sum(X, amsk, 21, 0))
        for window in (5, 21)
            @test isequal(descriptor(RollingMax(; window = window), rd),
                          rolling_expected_max(X, amsk, window))
        end
    end
    @testset "A window of 252 leaves only the last observations readable" begin
        D = descriptor(RollingMomentum(; window = 252, skip = 21), rd)
        @test all(isnan, D[1:272, :])
        @test count(!isnan, D) == 303
    end
    @testset "An asset view of the carrier gives the same Descriptor as a slice" begin
        v = PortfolioOptimisers.port_opt_view(rd, [2, 5, 9])
        for de in (RollingMomentum(; window = 63, skip = 21), Reversal(), MaxReturn())
            @test isequal(descriptor(de, v), descriptor(de, rd)[:, [2, 5, 9]])
        end
    end
end

@testset "The reference implementation's own output, pinned" begin
    #=
    The reference implementation was driven on the same synthetic panel, rebuilt as its own
    panel container with the same active and estimation masks, and its output at
    observations 296 to 300 is stored below. Asset 8 delists at observation 295, so its
    column pins the inactive rule as well as the arithmetic.

    Every NaN pattern agrees on all seven cases that were run. `RollingMax` agrees to the
    last bit. `RollingLogReturn` agrees to 5.6e-17 in absolute value, which is the residue
    of a one-bit difference in `log1p` itself on 9 of the 3474 returns of the panel; the
    two cumulative sums are taken in the same order.
    =#
    reference_reversal_21 = [0.005157908733073507 0.028093695361967097 0.12535624416725344 0.026813586525631483 0.0343297254340056 0.0840484614301415 0.09754260674030271 NaN 0.13459419284319432 0.0634559227890284 0.1386150341782934 0.030867361361560597
                                   -0.006743816405957448 0.013834529481491276 0.11600843205164058 0.02436438201512081 0.03047133581177322 0.07996858781623134 0.08743823363933584 NaN 0.13363775359043986 0.05438198569666042 0.12405686794558396 0.035775406692541395
                                   0.0006776615676298559 0.030254850755471074 0.11219040135996705 0.01522123064251657 0.02705239073655294 0.08545514958512845 0.07628570799226408 NaN 0.11405493566344696 0.05991986944125616 0.11630365270684817 0.05049959289708941
                                   -0.02950449992547463 0.0019573302318608987 0.08019697436427603 -0.04584546850027882 0.022572876148533413 0.05568418435993244 0.05963487774185494 NaN 0.08728377563883488 0.06670118720530782 0.06762213916522058 0.0507463797793371
                                   -0.044203394546773106 -0.00048265885253762075 0.09359272043606551 0.0049821566375432 0.026374438266559413 0.05894600925303006 0.045928205707354064 NaN 0.09270316512248242 0.07705538088004404 0.09992704661917473 0.06624252677832965]
    reference_max_return_21 = [0.047040239006114946 0.019845745986494884 0.015747046220745013 0.05305034898441107 0.023668269945923172 0.031145013351726942 0.03526007018851601 NaN 0.022958162687840997 0.03231014345231089 0.017469906096132946 0.014427781564077926
                                     0.047040239006114946 0.019845745986494884 0.015747046220745013 0.05305034898441107 0.023668269945923172 0.031145013351726942 0.03526007018851601 NaN 0.022958162687840997 0.03231014345231089 0.017469906096132946 0.014427781564077926
                                     0.047040239006114946 0.019845745986494884 0.015747046220745013 0.05305034898441107 0.023668269945923172 0.031145013351726942 0.03526007018851601 NaN 0.022958162687840997 0.03231014345231089 0.017469906096132946 0.014427781564077926
                                     0.047040239006114946 0.019845745986494884 0.018654034668595016 0.05305034898441107 0.023668269945923172 0.031145013351726942 0.03526007018851601 NaN 0.022958162687840997 0.03231014345231089 0.028353674929423155 0.014427781564077926
                                     0.047040239006114946 0.019845745986494884 0.018654034668595016 0.05305034898441107 0.023668269945923172 0.031145013351726942 0.03526007018851601 NaN 0.022958162687840997 0.03231014345231089 0.028353674929423155 0.014427781564077926]
    rng = StableRNG(717)
    rd = synthetic_asset_panel(; n_assets = 12, n_observations = 300, n_industries = 3,
                               rng = rng).rd
    rows = 296:300
    rev = descriptor(Reversal(), rd)[rows, :]
    mx = descriptor(MaxReturn(), rd)[rows, :]
    @test isequal(isnan.(rev), isnan.(reference_reversal_21))
    @test isequal(isnan.(mx), isnan.(reference_max_return_21))
    ok = .!isnan.(reference_reversal_21)
    @test isapprox(rev[ok], reference_reversal_21[ok]; rtol = 1e-14)
    @test maximum(abs, rev[ok] - reference_reversal_21[ok]) < 1e-16
    okm = .!isnan.(reference_max_return_21)
    # The window selects one return of the history, and that return is itself the output of a
    # product whose reduction order is the BLAS thread count of the machine. A CI runner and a
    # developer machine therefore differ in the last bit, so the pin is one ulp wide, as the
    # `rev` pin above is.
    @test isapprox(mx[okm], reference_max_return_21[okm]; rtol = 1e-14)
    @test maximum(abs, mx[okm] - reference_max_return_21[okm]) < 1e-16
end
