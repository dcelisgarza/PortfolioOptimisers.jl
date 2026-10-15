#=
The regression group of the cross-sectional diagnostics, decided by #709 and built by #798.

The testset `the stored oracle` compares with literals that the oracle gave on the very
arrays this file builds. Rebuild them by an export of the same fixture and a run of the
oracle's factor model block on it. Each comparison is cell by cell (`parity_compare`), and
states the worst value measured beside its tolerance.
=#
using Statistics
include(joinpath(@__DIR__, "parity_harness.jl"))

@testset "Cross-sectional regression diagnostics" begin
    # The fixture the oracle was measured on. Every mutation below drives one branch, so a
    # change to any of them invalidates the literals of `the stored oracle`.
    rng = StableRNG(987654321)
    T, N, K = 8, 6, 3
    Ms = randn(rng, T, N, K)
    f = 0.02 * randn(rng, T, K)
    eps = 0.01 * randn(rng, T, N)
    rw = abs.(randn(rng, T, N)) .+ 0.1
    bw = fill(1 / N, T, N)
    # A pair whose weight is zero is excluded.
    rw[3, 2] = 0.0
    rw[6, 5] = 0.0
    # An observation with fewer eligible assets than factors has no degrees of freedom.
    rw[8, 3] = 0.0
    rw[8, 4] = 0.0
    rw[8, 5] = 0.0
    rw[8, 6] = 0.0
    # A pair whose residual is not finite leaves the mask.
    eps[4, 3] = NaN
    # A pair whose exposure is not finite leaves the mask.
    Ms[2, 4, 1] = NaN
    # An observation whose factor return is not finite has no t-statistic.
    f[5, 2] = NaN
    # An observation whose design is collinear identifies one coefficient of three.
    Ms[6, :, 3] = Ms[6, :, 1]

    csr = CrossSectionalRegression(; f = f, eps = eps, n = fill(N, T))
    csfm = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr, Ms = Ms,
                                     rw = rw, bw = bw, lag = 1)

    @testset "the stored oracle" begin
        ref_t = [3.069044320420541 -5.281244570976589 9.690084569128215;
                 -0.3009740870778322 -0.25525639680758366 0.4581283746731422;
                 0.5071171529394058 -1.3892133124364714 -0.2597881748342029;
                 NaN NaN NaN;
                 2.3398424449301682 -2.5324783294991913 -1.8821440946792465;
                 0.6639470271883212 -14.363795117368108 7.300135695412292;
                 NaN NaN NaN]
        ref_vif = [1.2599251318909874 1.2727980646620327 1.1208405179884529;
                   3.10281260008445 2.588807016496487 1.3503993754421315;
                   3.097217651220898 2.253293121577358 2.483251074452251;
                   1.0862919838510936 2.055747720555408 2.0167833289680854;
                   3.482053032817249 1.5630479431601327 2.6122984907892386;
                   0.3308291918946691 1.3233167675786761 0.3308291918946689;
                   NaN NaN NaN]
        ref_rate = [0.4, 0.6, 0.4]
        ref_r2 = [0.9817378235244757, -0.3840824898176942, 0.7349775519611599, NaN,
                  0.7785383767983827, 0.9910410261099509, 0.9482920371476079]
        ref_adj = [0.9543445588111892, NaN, -0.06008979215536048, NaN, 0.11415350719353068,
                   0.9776025652748773, NaN]
        ref_aic = [-57.14936045686335, -27.58559514891217, -39.04463396286469, NaN,
                   -38.63887948333526, -52.65578599059639, NaN]
        ref_bic = [-57.774082049179185, -29.426712065552497, -40.216320225562384, NaN,
                   -39.81056574603296, -53.28050758291223, NaN]

        # Cell by cell (`parity_compare`): the pattern of the absent answers must agree
        # exactly, and each finite cell lies within `rtol` of its own size. The worst
        # value measured is stated at each check. The tolerance `1e-14` is about ten times
        # the worst one, which is the round-off of a Gram solve of a different order.
        agrees(a, b, name; rtol = 1e-14) = parity_compare(a, b; rtol = rtol, name = name).ok

        # Observation 6 is collinear: its third column repeats the first, so the design
        # identifies the second coefficient alone and has rank two (#1421). The oracle's
        # answers there read the pseudo-inverse. Its t-statistics of the pair are ratios of
        # an arbitrary split, its factors of the pair fall below one, and its variance and
        # scores charge three regressors. Every other observation has full rank and agrees.
        fr = [1:5; 7]
        t = cs_regression_t_stats(csfm).X
        vif = exposure_vif(csfm).X
        # Measured maxrel 7.6e-16, 1.7e-15 and 1.2e-15.
        @test agrees(t[fr, :], ref_t[fr, :], "t")
        @test agrees(vif[fr, :], ref_vif[fr, :], "vif")
        @test agrees(cs_regression_r2(csfm), ref_r2, "r2")
        # The three scores are bit-equal, with either count of regressors.
        for (verb, ref) in
            ((cs_regression_adjusted_r2, ref_adj), (cs_regression_aic, ref_aic),
             (cs_regression_bic, ref_bic))
            @test agrees(verb(csfm)[fr], ref[fr], "$(verb)")
            # The oracle's count of regressors is one keyword away.
            @test agrees(verb(csfm; k = K), ref, "$(verb) k")
        end
        # The pair has no t-statistic and an infinite factor. The second coefficient keeps
        # its factor, and its residual variance divides by `n - 2 = 4` rather than
        # `n - 3 = 3`, so its t-statistic is the oracle's times `sqrt(4 / 3)`. Each of the
        # five checks below is bit-equal.
        @test all(isnan, t[6, [1, 3]]) && all(==(Inf), vif[6, [1, 3]])
        @test agrees([vif[6, 2]], [ref_vif[6, 2]], "vif 6")
        @test agrees([t[6, 2]], [ref_t[6, 2] * sqrt(4 / 3)], "t 6")
        # A score charges two regressors: the AIC less 2, the BIC less `ln 6`, and the
        # adjusted score over `n - 3` rather than `n - 4`.
        @test agrees([cs_regression_aic(csfm)[6]], [ref_aic[6] - 2], "aic 6")
        @test agrees([cs_regression_bic(csfm)[6]], [ref_bic[6] - log(6)], "bic 6")
        @test agrees([cs_regression_adjusted_r2(csfm)[6]], [1 - (1 - ref_r2[6]) * 5 / 3],
                     "adjusted r2 6")
        # The rate of the pair loses observation 6, where the oracle counted 2 of 5 and
        # 2 of 5. The second factor keeps it, at the oracle's 3 of 5.
        @test cs_regression_t_stat_exceedance_rate(csfm).X == [2 / 4, 3 / 5, 1 / 4]
        @test ref_rate == [2 / 5, 3 / 5, 2 / 5]

        # The condition number agrees everywhere the design has full rank. The oracle
        # answers `1.9e17` at the collinear observation where this answers `Inf`: the
        # smallest singular value of a rank-deficient matrix rounds either to zero or to a
        # value near the machine epsilon, which is a property of the decomposition and not
        # of the design. Both answers say that the design is singular.
        ref_cond = [2.9908189507547056, 12.318608475111041, 19.287897328200575,
                    6.588259138577632, 13.607428734322603]
        kappa = exposure_condition_number(csfm)
        @test length(kappa) == 7
        # Measured maxrel 1.2e-15: two singular value decompositions.
        @test agrees(kappa[1:5], ref_cond, "condition number")
        @test kappa[6] > 1e12
        @test isnan(kappa[7])
    end

    @testset "the answers carry the lagged observation axis" begin
        @test size(cs_regression_t_stats(csfm).X) == (T - 1, K)
        @test size(exposure_vif(csfm).X) == (T - 1, K)
        @test length(exposure_condition_number(csfm)) == T - 1
        @test length(cs_regression_t_stat_exceedance_rate(csfm).X) == K
        for verb in (cs_regression_r2, cs_regression_adjusted_r2, cs_regression_aic,
                     cs_regression_bic)
            @test length(verb(csfm)) == T - 1
        end
    end

    @testset "the standard error comes from a direct solve" begin
        rng2 = StableRNG(11223344)
        Tl, Nl, Kl = 2, 7, 3
        B = randn(rng2, Tl, Nl, Kl)
        fl = 0.05 * randn(rng2, Tl, Kl)
        epsl = 0.02 * randn(rng2, Tl, Nl)
        t = cs_regression_t_stats(B, fl, epsl)
        for tt in 1:Tl
            A = B[tt, :, :]
            G = transpose(A) * A
            rss = sum(abs2, epsl[tt, :])
            s2 = rss / (Nl - Kl)
            se = sqrt.(s2 .* LinearAlgebra.diag(inv(G)))
            @test isapprox(t[tt, :], fl[tt, :] ./ se; rtol = 1e-10)
        end
        # The Gram history a caller already holds is the same answer.
        @test isapprox(cs_regression_t_stats(B, fl, epsl; G = cs_gram(B)), t; rtol = 1e-12)
        @test_throws DimensionMismatch cs_regression_t_stats(B, fl, epsl;
                                                             G = cs_gram(B[:, :, 1:2]))
        @test_throws DimensionMismatch cs_regression_t_stats(B, fl[:, 1:2], epsl)
    end

    @testset "the Gram kernel and the weights" begin
        B = reshape([1.0, 1.0, 0.0, 1.0], 1, 2, 2)
        @test cs_gram(B)[1, :, :] == [2.0 1.0; 1.0 1.0]
        # A zero weight excludes the pair, so the second asset leaves the design.
        @test cs_gram(B, [1.0 0.0])[1, :, :] == [1.0 0.0; 0.0 0.0]
        # A weight scales the pair.
        @test cs_gram(B, [1.0 4.0])[1, :, :] == [5.0 4.0; 4.0 4.0]
        # An exposure that is not finite excludes the pair.
        Bn = reshape([1.0, NaN, 0.0, 1.0], 1, 2, 2)
        @test cs_gram(Bn)[1, :, :] == [1.0 0.0; 0.0 0.0]
        @test_throws PortfolioOptimisers.IsEmptyError cs_gram(zeros(0, 0, 0))
        @test_throws DimensionMismatch cs_gram(B, [1.0 1.0 1.0])
    end

    @testset "the variance inflation factor and the condition number" begin
        # An orthonormal design inflates nothing and is perfectly conditioned.
        Bo = reshape([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 1, 3, 2)
        @test exposure_vif(cs_gram(Bo)) == [1.0 1.0]
        @test exposure_condition_number(cs_gram(Bo)) == [1.0]
        # The methods that read a design also gate on the degrees of freedom, which a
        # third asset supplies.
        @test exposure_vif(Bo, nothing) == [1.0 1.0]
        @test exposure_condition_number(Bo, nothing) == [1.0]
        # Two assets and two factors leave no degrees of freedom, so there is no answer.
        Bt = reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2)
        @test all(isnan, exposure_vif(Bt, nothing))
        @test all(isnan, exposure_condition_number(Bt, nothing))

        # A nearly duplicated column inflates the variance of both members of the pair.
        rng3 = StableRNG(55667788)
        Bd = randn(rng3, 4, 30, 3)
        Bd[:, :, 3] = Bd[:, :, 1] + 0.01 * randn(rng3, 4, 30)
        vif = exposure_vif(Bd, nothing)
        @test all(vif[:, 1] .> 100)
        @test all(vif[:, 3] .> 100)
        @test all(vif[:, 2] .< 2)
        @test all(exposure_condition_number(Bd, nothing) .> 1e3)

        # An exactly duplicated column is singular. Each member of the pair is the other,
        # so `R_k^2 = 1` and its factor is `Inf` (#1421). The third column is identified,
        # and its factor is finite and at least one.
        Bs = copy(Bd)
        Bs[:, :, 3] = Bs[:, :, 1]
        vs = exposure_vif(Bs, nothing)
        @test all(==(Inf), vs[:, [1, 3]])
        @test all(v -> isfinite(v) && v >= 1, vs[:, 2])

        # An observation with no degrees of freedom has no answer.
        w = ones(4, 30)
        w[2, 4:end] .= 0.0
        @test all(isnan, exposure_vif(Bd, w)[2, :])
        @test isnan(exposure_condition_number(Bd, w)[2])
    end

    @testset "the exceedance rate" begin
        @test cs_regression_t_stat_exceedance_rate([3.0 1.0; 1.0 1.0; NaN 1.0]) ==
              [0.5, 0.0]
        @test cs_regression_t_stat_exceedance_rate([3.0 1.0; 1.0 1.0]; threshold = 0.5) ==
              [1.0, 1.0]
        # A factor with no finite t-statistic answers zero rather than raising.
        @test cs_regression_t_stat_exceedance_rate([NaN NaN;]) == [0.0, 0.0]
        @test_throws PortfolioOptimisers.IsEmptyError cs_regression_t_stat_exceedance_rate(zeros(0,
                                                                                                 0))
    end

    @testset "a signal factor and a noise factor" begin
        rng4 = StableRNG(31415926)
        Ts, Ns = 300, 40
        B = randn(rng4, Ts, Ns, 2)
        noise = 0.01 * randn(rng4, Ts, Ns)
        Rt = Matrix{Float64}(undef, Ts, Ns)
        for i in 1:Ns, t in 1:Ts
            Rt[t, i] = 0.03 * B[t, i, 1] + noise[t, i]
        end
        f2 = Matrix{Float64}(undef, Ts, 2)
        eps2 = Matrix{Float64}(undef, Ts, Ns)
        B1 = B[:, :, 1:1]
        f1 = Matrix{Float64}(undef, Ts, 1)
        eps1 = Matrix{Float64}(undef, Ts, Ns)
        for t in 1:Ts
            A = B[t, :, :]
            y = Rt[t, :]
            b2 = A \ y
            f2[t, :] = b2
            eps2[t, :] = y - A * b2
            A1 = B1[t, :, :]
            b1 = A1 \ y
            f1[t, :] = b1
            eps1[t, :] = y - A1 * b1
        end

        t2 = cs_regression_t_stats(B, f2, eps2)
        rate = cs_regression_t_stat_exceedance_rate(t2)
        @test Statistics.median(abs.(t2[:, 1])) > 3
        @test Statistics.median(abs.(t2[:, 2])) < 2
        @test rate[1] > 0.8
        @test rate[2] < 0.2

        # The adjusted score never exceeds the score it adjusts.
        r2 = cs_regression_r2(B, f2, eps2)
        adj = cs_regression_adjusted_r2(B, f2, eps2)
        @test all(adj .<= r2 .+ 1e-12)

        # Adding the noise factor cannot lower the score, and both criteria charge for it.
        r2_1 = cs_regression_r2(B1, f1, eps1)
        @test minimum(r2 - r2_1) >= -1e-10
        @test Statistics.mean(cs_regression_aic(B1, f1, eps1)) <
              Statistics.mean(cs_regression_aic(B, f2, eps2))
        @test Statistics.mean(cs_regression_bic(B1, f1, eps1)) <
              Statistics.mean(cs_regression_bic(B, f2, eps2))
    end

    @testset "a re-based block answers on the reduced axis" begin
        rng5 = StableRNG(19283746)
        Tr, Nr, Kr = 6, 5, 3
        Msr = randn(rng5, Tr, Nr, Kr)
        # A fit under a re-basis estimates its factor returns on the reduced axis, so the
        # block states them there.
        fr = 0.02 * randn(rng5, Tr, Kr - 1)
        epsr = 0.01 * randn(rng5, Tr, Nr)
        rwr = abs.(randn(rng5, Tr, Nr)) .+ 0.1
        # One constrained family holds factors 1 and 2, and drops the second of them.
        fcb = FactorFamilyBasis(; fnm = ["industry"], fi = [[1, 2]], di = [2],
                                ratios = reshape(collect(range(0.4, 0.9; length = Tr)), Tr,
                                                 1), K = Kr)
        csr_r = CrossSectionalRegression(; f = fr, eps = epsr, n = fill(Nr, Tr))
        L = PortfolioOptimisers.reduce_loadings(fcb, Msr[Tr, :, :])
        blk = CrossSectionalFactorModel(; M = Msr[Tr, :, :], L = L, b = zeros(Nr),
                                        csr = csr_r, Ms = Msr, rw = rwr, fcb = fcb, lag = 1,
                                        nf = ["value", "size", "momentum"],
                                        fam = ["industry", "industry", "style"])
        Kred = PortfolioOptimisers.reduced_factor_count(fcb)
        @test Kred == 2
        @test size(cs_regression_t_stats(blk).X) == (Tr - 1, Kred)
        @test size(exposure_vif(blk).X) == (Tr - 1, Kred)
        @test length(cs_regression_t_stat_exceedance_rate(blk).X) == Kred
        @test length(exposure_condition_number(blk)) == Tr - 1
        @test length(cs_regression_r2(blk)) == Tr - 1
        # The factor names of the answer's axis are the reduced ones.
        @test PortfolioOptimisers.cs_diagnostic_factor_names(blk) == ["value", "momentum"]
        # Each answer carries the names and the family labels of the reduced axis, and a
        # view selects its factors by position, by name and by family.
        t = cs_regression_t_stats(blk)
        @test t.nf == ["value", "momentum"] && t.fam == ["industry", "style"]
        @test t.dims == (2,)
        for r in (t, exposure_vif(blk), cs_regression_t_stat_exceedance_rate(blk))
            @test r.nf == ["value", "momentum"] && r.fam == ["industry", "style"]
            by_family = PortfolioOptimisers.port_opt_view(r, LabelGroup("style"))
            by_name = PortfolioOptimisers.port_opt_view(r, ["momentum"])
            by_index = PortfolioOptimisers.port_opt_view(r, 2)
            for v in (by_family, by_name, by_index)
                @test v.nf == ["momentum"] && v.fam == ["style"]
                @test isequal(v.X, selectdim(r.X, ndims(r.X), 2:2))
            end
            @test isequal(PortfolioOptimisers.port_opt_view(r, :).X, r.X)
        end
        # The rate over a view of the t-statistics is the view of the rate.
        rv = cs_regression_t_stat_exceedance_rate(PortfolioOptimisers.port_opt_view(t,
                                                                                    LabelGroup("industry")))
        @test rv.nf == ["value"]
        @test rv.X == cs_regression_t_stat_exceedance_rate(blk).X[1:1]
        # A name or a family the axis does not hold is refused, and the dropped factor of
        # the re-basis is not on the axis.
        @test_throws ArgumentError PortfolioOptimisers.port_opt_view(t, ["size"])
        @test_throws ArgumentError PortfolioOptimisers.port_opt_view(t,
                                                                     LabelGroup("sector"))
        @test isnothing(PortfolioOptimisers.cs_diagnostic_factor_names(csfm))
        @test PortfolioOptimisers.cs_diagnostic_factor_names(CrossSectionalFactorModel(;
                                                                                       M = Msr[Tr,
                                                                                               :,
                                                                                               :],
                                                                                       b = zeros(Nr),
                                                                                       nf = ["a",
                                                                                             "b",
                                                                                             "c"])) ==
              ["a", "b", "c"]
    end

    @testset "a block that carries no history is refused" begin
        no_ms = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr)
        no_csr = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), Ms = Ms)
        for verb in (cs_regression_t_stats, exposure_vif, exposure_condition_number,
                     cs_regression_r2, cs_regression_adjusted_r2, cs_regression_aic,
                     cs_regression_bic, cs_regression_t_stat_exceedance_rate)
            @test_throws PortfolioOptimisers.IsNothingError verb(no_ms)
            @test_throws PortfolioOptimisers.IsNothingError verb(no_csr)
        end
        err = try
            cs_regression_t_stats(no_ms)
        catch e
            e
        end
        @test occursin("Ms", err.msg)
        err2 = try
            cs_regression_t_stats(no_csr)
        catch e
            e
        end
        @test occursin("csr", err2.msg)
        # A lag that consumes the whole history leaves nothing to answer on.
        short = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr,
                                          Ms = Ms, lag = T)
        @test_throws DimensionMismatch cs_regression_t_stats(short)
    end

    @testset "the lag defaults to zero" begin
        rng6 = StableRNG(65432109)
        Tn, Nn, Kn = 5, 4, 2
        Msn = randn(rng6, Tn, Nn, Kn)
        csrn = CrossSectionalRegression(; f = 0.02 * randn(rng6, Tn, Kn),
                                        eps = 0.01 * randn(rng6, Tn, Nn), n = fill(Nn, Tn))
        blk = CrossSectionalFactorModel(; M = Msn[Tn, :, :], b = zeros(Nn), csr = csrn,
                                        Ms = Msn)
        @test size(cs_regression_t_stats(blk).X) == (Tn, Kn)
        @test isapprox(cs_regression_t_stats(blk).X,
                       cs_regression_t_stats(Msn, csrn.f, csrn.eps); rtol = 1e-12)
    end

    @testset "the verbs agree with a weighted least-squares fit written out" begin
        rng7 = StableRNG(24681357)
        Tw, Nw, Kw = 4, 12, 3
        B = randn(rng7, Tw, Nw, Kw)
        B[:, :, 1] .= 1.0
        W = rand(rng7, Tw, Nw)
        # Each mask branch: a zero weight, a NaN weight, a NaN exposure, a NaN residual.
        W[1, 3] = 0.0
        W[2, 5] = NaN
        B[3, 4, 2] = NaN
        fw = randn(rng7, Tw, Kw)
        ew = 0.2 * randn(rng7, Tw, Nw)
        ew[4, 6] = NaN
        vif = exposure_vif(B, W)
        kappa = exposure_condition_number(B, W)
        ts = cs_regression_t_stats(B, fw, ew, W)
        r2 = cs_regression_r2(B, fw, ew, W)
        adj = cs_regression_adjusted_r2(B, fw, ew, W)
        aic = cs_regression_aic(B, fw, ew, W)
        bic = cs_regression_bic(B, fw, ew, W)
        for t in 1:Tw
            # The Gram diagnostics mask on the weight and the exposures.
            ok = [W[t, i] > 0 && all(isfinite, B[t, i, :]) for i in 1:Nw]
            X = B[t, ok, :]
            G = transpose(X) * LinearAlgebra.Diagonal(W[t, ok]) * X
            @test isapprox(vif[t, :], LinearAlgebra.diag(G) .* LinearAlgebra.diag(inv(G));
                           rtol = 1e-12)
            @test isapprox(kappa[t], LinearAlgebra.cond(G); rtol = 1e-12)
            # The t-statistic also masks on the residual.
            ok2 = ok .& isfinite.(ew[t, :])
            X2 = B[t, ok2, :]
            u2 = W[t, ok2]
            G2 = transpose(X2) * LinearAlgebra.Diagonal(u2) * X2
            s2 = sum(u2 .* ew[t, ok2] .^ 2) / (count(ok2) - Kw)
            se = sqrt.(s2 .* LinearAlgebra.diag(inv(G2)))
            @test isapprox(ts[t, :], fw[t, :] ./ se; rtol = 1e-12)
            # The scores mask on the reconstructed return and normalise the weights.
            r = [sum(B[t, i, :] .* fw[t, :]) + ew[t, i] for i in 1:Nw]
            ok3 = (W[t, :] .> 0) .& isfinite.(r)
            q = W[t, ok3] ./ sum(W[t, ok3])
            n = count(ok3)
            rss = sum(q .* ew[t, ok3] .^ 2)
            m = sum(q .* r[ok3])
            R2 = 1 - rss / sum(q .* (r[ok3] .- m) .^ 2)
            @test isapprox(r2[t], R2; rtol = 1e-12)
            @test isapprox(adj[t], 1 - (1 - R2) * (n - 1) / (n - Kw - 1); rtol = 1e-12)
            @test isapprox(aic[t], n * log(rss) + 2 * Kw; rtol = 1e-12)
            @test isapprox(bic[t], n * log(rss) + Kw * log(n); rtol = 1e-12)
        end
        # A perfect fit has a residual sum of squares of zero.
        Bp = reshape([1.0, 1.0, 1.0, -1.0, 1.0, 0.0], 1, 3, 2)
        @test cs_regression_aic(Bp, [1.0 0.0], zeros(1, 3); k = 1) == [-Inf]
        @test cs_regression_bic(Bp, [1.0 0.0], zeros(1, 3); k = 1) == [-Inf]
    end

    @testset "a cross-section of equal returns has no coefficient of determination" begin
        # The weighted mean of equal returns rounds away from their common value for most
        # weights, so the total sum of squares must not be read off the rounded mean. Sizes
        # of three to seven assets reach the rounding that four assets can miss.
        rng8 = StableRNG(97531864)
        for n in 3:7, _ in 1:200
            B = randn(rng8, 1, n, 1)
            e = fill(randn(rng8), 1, n)
            @test isnan(cs_regression_r2(B, zeros(1, 1), e, rand(rng8, 1, n))[1])
        end
    end

    @testset "a collinear design states what it does not identify (#1421)" begin
        # An intercept beside a one-hot block is exactly collinear, and the null vector
        # `(1, -1, -1, -1)` touches every column. No coefficient is identified, so every
        # factor is `Inf`: the diagonal of the pseudo-inverse gave values below one, which
        # no `1 / (1 - R_k^2)` reaches.
        Nc = 9
        Bc = zeros(1, Nc, 4)
        Bc[1, :, 1] .= 1.0
        for i in 1:Nc
            Bc[1, i, 2 + mod(i, 3)] = 1.0
        end
        @test all(==(Inf), exposure_vif(Bc, nothing))
        @test all(==(Inf), PortfolioOptimisers.cs_gram_inverse_diagonal(cs_gram(Bc)))
        @test exposure_condition_number(Bc, nothing)[1] > 1e12

        # A design whose third column repeats the first, and whose fourth column is zero,
        # against the same design with both columns removed. The fit is the minimum-norm
        # solution. It identifies the second coefficient alone, and its residuals span
        # `n - 2` dimensions, so the second t-statistic and the scores equal those of the
        # reduced fit. The other coefficients have no t-statistic, the repeated pair has
        # an infinite factor, and the zero column has no factor.
        rngc = StableRNG(24681357)
        Tc, Nd = 5, 12
        Br = randn(rngc, Tc, Nd, 2)
        Bx = cat(Br, Br[:, :, 1:1], zeros(Tc, Nd, 1); dims = 3)
        wc = rand(rngc, Tc, Nd) .+ 0.5
        fx = zeros(Tc, 4)
        ex = zeros(Tc, Nd)
        fr = zeros(Tc, 2)
        for t in 1:Tc
            y = Br[t, :, :] * [0.3, -0.2] + 0.1 * randn(rngc, Nd)
            sw = sqrt.(wc[t, :])
            fx[t, :] = pinv(sw .* Bx[t, :, :]) * (sw .* y)
            ex[t, :] = y - Bx[t, :, :] * fx[t, :]
            fr[t, :] = (sw .* Br[t, :, :]) \ (sw .* y)
        end
        @test fr ≈ fx[:, 1:2] .* [2 1] rtol = 1e-12
        tx = cs_regression_t_stats(Bx, fx, ex, wc)
        tr = cs_regression_t_stats(Br, fr, ex, wc)
        @test isapprox(tx[:, 2], tr[:, 2]; rtol = 1e-10)
        @test all(isnan, tx[:, [1, 3, 4]])
        vx = exposure_vif(Bx, wc)
        @test all(==(Inf), vx[:, [1, 3]]) && all(isnan, vx[:, 4])
        @test all(v -> isfinite(v) && v >= 1, vx[:, 2])
        @test PortfolioOptimisers.cs_score_regressors(nothing, Bx,
                                                      PortfolioOptimisers.cs_regression_score_parts(Bx,
                                                                                                    fx,
                                                                                                    ex,
                                                                                                    wc)[4]) ==
              fill(2, Tc)
        for verb in (cs_regression_adjusted_r2, cs_regression_aic, cs_regression_bic)
            @test isapprox(verb(Bx, fx, ex, wc), verb(Br, fr, ex, wc); rtol = 1e-12)
            # A count the caller states holds on every observation.
            @test isapprox(verb(Bx, fx, ex, wc; k = 4), verb(Br, fr, ex, wc; k = 4);
                           rtol = 1e-12)
            @test !isapprox(verb(Bx, fx, ex, wc; k = 4), verb(Bx, fx, ex, wc))
        end
    end

    @testset "the numeric types" begin
        rng9 = StableRNG(13572468)
        Tn, Nn, Kn = 3, 7, 2
        Bi = rand(rng9, -3:3, Tn, Nn, Kn)
        Bi[:, :, 1] .= 1
        fi = rand(rng9, -3:3, Tn, Kn)
        ei = rand(rng9, -3:3, Tn, Nn)
        wi = rand(rng9, 1:3, Tn, Nn)
        Bf, ff, ef, wf = float.(Bi), float.(fi), float.(ei), float.(wi)
        # An integer history keeps an exact Gram history, and an integer weight scales it.
        @test cs_gram(Bi, wi) isa Array{Int, 3}
        @test cs_gram(Bi, wi) == cs_gram(Bf, wf)
        # Every other verb answers in `Float64`, and agrees with the float history.
        @test exposure_vif(Bi, wi) ≈ exposure_vif(Bf, wf)
        @test exposure_condition_number(Bi, wi) ≈ exposure_condition_number(Bf, wf)
        @test cs_regression_t_stats(Bi, fi, ei, wi) ≈ cs_regression_t_stats(Bf, ff, ef, wf)
        for verb in (cs_regression_r2, cs_regression_adjusted_r2, cs_regression_aic,
                     cs_regression_bic)
            @test verb(Bi, fi, ei, wi) isa Vector{Float64}
            @test verb(Bi, fi, ei, wi) ≈ verb(Bf, ff, ef, wf)
        end
        @test cs_regression_t_stat_exceedance_rate([3 1; 1 1]) == [0.5, 0.0]
        # A `Rational` history keeps its Gram history and its scores exact. The
        # decomposition and the logarithm answer in `Float64`.
        Br, fr, er = Rational{Int}.(Bi), Rational{Int}.(fi), ei .// 7
        @test cs_gram(Br) isa Array{Rational{Int}, 3}
        @test cs_regression_r2(Br, fr, er) isa Vector{Rational{Int}}
        @test cs_regression_r2(Br, fr, er) ≈ cs_regression_r2(Bf, ff, ei ./ 7)
        @test exposure_vif(Br, nothing) ≈ exposure_vif(Bf, nothing)
        @test cs_regression_aic(Br, fr, er) ≈ cs_regression_aic(Bf, ff, ei ./ 7)
        @test cs_regression_t_stat_exceedance_rate([3//1 1; 1 1]) == [1//2, 0]
        # A `Float32` history stays `Float32`.
        B32, f32, e32, w32 = Float32.(Bf), Float32.(ff), Float32.(ef), Float32.(wf)
        @test exposure_vif(B32, w32) isa Matrix{Float32}
        @test exposure_condition_number(B32, w32) isa Vector{Float32}
        @test cs_regression_t_stats(B32, f32, e32, w32) isa Matrix{Float32}
        @test cs_regression_r2(B32, f32, e32, w32) isa Vector{Float32}
        @test cs_regression_aic(B32, f32, e32, w32) isa Vector{Float32}
    end
end

@testset "An observed factor leaves the design of the regression (#1367)" begin
    PO = PortfolioOptimisers
    rng = StableRNG(1_367_08)
    T, N = 8, 6
    Ms = randn(rng, T, N, 3)
    Ms[:, :, 3] .= 1.0
    f = 0.02 * randn(rng, T, 2)
    fx = 0.01 * randn(rng, T, 1)
    eps = 0.01 * randn(rng, T, N)
    rw = abs.(randn(rng, T, N)) .+ 0.1
    csr = CrossSectionalRegression(; f = f, eps = eps, n = fill(N, T))
    blk = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr, Ms = Ms,
                                    rw = rw, lag = 1, nf = ["value", "size", "usd"],
                                    fam = ["style", "style", "currency"], fx = fx)
    # The same block without the observed factor is the design the regression ran on.
    est = CrossSectionalFactorModel(; M = Ms[T, :, 1:2], b = zeros(N), csr = csr,
                                    Ms = Ms[:, :, 1:2], rw = rw, lag = 1,
                                    nf = ["value", "size"])
    @test cs_regression_t_stats(blk).X == cs_regression_t_stats(est).X
    @test isequal(exposure_vif(blk).X, exposure_vif(est).X)
    # The answer drops the observed factor from its names and from its families.
    @test cs_regression_t_stats(blk).nf == ["value", "size"]
    @test exposure_vif(blk).fam == ["style", "style"]
    @test isequal(cs_regression_r2(blk), cs_regression_r2(est))
    @test isequal(cs_regression_bic(blk), cs_regression_bic(est))
    @test PO.cs_diagnostic_factor_names(blk) == ["value", "size"]
    # The loadings axis still names the observed factor.
    @test PO.cs_diagnostic_factor_names(blk.fcb, blk.nf) == ["value", "size", "usd"]
end

#=
The exposure group of the cross-sectional diagnostics, decided by #709 and built by #799.

The testset `the stored oracle` compares with literals that the oracle gave on the very
arrays this file builds. Rebuild them by an export of the same fixture and a run of the
oracle's factor model block on it. Each comparison is cell by cell (`parity_compare`), and
states the worst value measured beside its tolerance.
=#
using Statistics

@testset "Cross-sectional exposure diagnostics" begin
    # The fixture the oracle was measured on. Every mutation below drives one branch, so a
    # change to any of them invalidates the literals of `the stored oracle`.
    rng = StableRNG(24680)
    T, N, K = 10, 6, 4
    Ms = randn(rng, T, N, K)
    f = 0.02 * randn(rng, T, K)
    eps = 0.01 * randn(rng, T, N)
    rw = abs.(randn(rng, T, N)) .+ 0.1
    bw = fill(1 / N, T, N)
    vs = abs.(randn(rng, T, N)) .+ 0.5
    # The third factor is the global intercept. Its cross-section is constant, so every
    # pair that holds it is degenerate and reads `0` by convention.
    Ms[:, :, 3] .= 1.0
    # The fourth factor is finite on two assets alone, so no pair that holds it ever
    # reaches the three common assets an estimate needs, and every such pair reads `NaN`.
    Ms[:, 3:6, 4] .= NaN
    # An asset that carries no weight leaves the universe of that observation.
    rw[3, 2] = 0.0
    bw[7, 4] = 0.0
    # A missing exposure costs its own pair and not the observation.
    Ms[2, 4, 1] = NaN
    # An observation at which a factor keeps two finite assets is insufficient there alone.
    Ms[5, 1, 2] = NaN
    Ms[5, 2, 2] = NaN
    Ms[5, 3, 2] = NaN
    Ms[5, 4, 2] = NaN

    csr = CrossSectionalRegression(; f = f, eps = eps, n = fill(N, T))
    csfm = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr, Ms = Ms,
                                     vs = vs, rw = rw, bw = bw, lag = 1)
    # The fourth factor is not finite on four of the six assets, so the reconstructed asset
    # return of those assets is not finite either, and the information coefficient of every
    # factor would be `NaN`. It is read on the three dense factors of the same fixture.
    csr3 = CrossSectionalRegression(; f = f[:, 1:3], eps = eps, n = fill(N, T))
    csfm3 = CrossSectionalFactorModel(; M = Ms[T, :, 1:3], b = zeros(N), csr = csr3,
                                      Ms = Ms[:, :, 1:3], vs = vs, rw = rw, bw = bw,
                                      lag = 1)

    # Cell by cell: the pattern of the absent answers agrees exactly, and each finite cell
    # lies within `rtol` of its own size. The worst value measured against the oracle is
    # 6.3e-15, and all but one check measure 6.6e-16 or less, so `1e-14` is about ten
    # times the round-off of a weighted mean of another order.
    function exposure_agrees(a, b, label = ""; rtol = 1e-14)
        return parity_compare(a, b; rtol = rtol, name = label).ok
    end

    @testset "the stored oracle" begin
        ref_corr_bw = [ 1.0 -0.11910774310012738 0.0 NaN;
                       -0.11910774310012745 1.0 0.0 NaN;
                       0.0 0.0 1.0 NaN;
                       NaN NaN NaN 1.0]
        ref_corr_id = [ 1.0 -0.06318259646864377 0.0 NaN;
                       -0.06318259646864377 1.0 0.0 NaN;
                       0.0 0.0 1.0 NaN;
                       NaN NaN NaN 1.0]
        ref_corr_rw = [ 1.0 -0.12423074669435247 0.0 NaN;
                       -0.12423074669435247 1.0 0.0 NaN;
                       0.0 0.0 1.0 NaN;
                       NaN NaN NaN 1.0]
        ref_ic_spearman = [ 0.2571428571428571 0.8285714285714286 0.08571428571428572;
                           1.0 0.4 -0.6;
                           0.2 -0.6571428571428571 -0.08571428571428572;
                           1.0 0.14285714285714285 -0.42857142857142855;
                           NaN NaN NaN;
                           0.2 0.7142857142857143 -0.08571428571428572;
                           -0.8857142857142857 0.14285714285714285 -0.14285714285714285;
                           -0.6571428571428571 0.02857142857142857 -0.14285714285714285;
                           -0.6571428571428571 0.3142857142857143 -0.4857142857142857]
        ref_ic_pearson = [ 0.017660125172323477 0.8554536111961 NaN;
                          0.9704452550730513 0.3235269729843106 NaN;
                          0.6358545350022913 -0.7769061129248623 NaN;
                          0.987622896128322 0.021171816164635107 NaN;
                          NaN NaN NaN;
                          -0.14622469419159947 0.5839881757529729 NaN;
                          -0.7618984163053351 0.6376018710833449 NaN;
                          -0.4310263691775938 0.12206610944522596 NaN;
                          -0.6113039548526288 0.5481220484702234 NaN]
        ref_ic_h2 = [ 0.14285714285714285 0.8285714285714286 -0.6571428571428571;
                     0.2 0.9428571428571428 -0.3142857142857143;
                     -0.14285714285714285 -0.2 -0.42857142857142855;
                     1.0 0.14285714285714285 -0.42857142857142855;
                     -0.14285714285714285 NaN -0.2571428571428571;
                     0.6571428571428571 -0.6 -0.37142857142857144;
                     -0.7714285714285715 0.37142857142857144 -0.02857142857142857;
                     -0.2 0.2571428571428571 -0.37142857142857144]
        ref_mean_ic = [0.057142857142857204, 0.2392857142857143, -0.23571428571428568]
        ref_std_ic = [0.733695466730492, 0.4584006929917067, 0.23843513154179433]
        ref_ic_ir = [0.07788361756887816, 0.5220012053734906, -0.9885887377002087]
        # The oracle's exposure summary reads `nanmean(ic > 0)`, and the comparison turns
        # a `NaN` into `false` before the mean sees it, so it counts the one silenced date of
        # nine as a miss: 5/9, 7/9, 1/9. Its evaluation summary drops a `NaN`, as every
        # summary of this library does, so the port reads the eight dates that carried a
        # coefficient and this is a deliberate divergence, not a missed one (2026-09-14).
        ref_hit_rate = [0.5555555555555556, 0.7777777777777778, 0.1111111111111111]
        port_hit_rate = [0.625, 0.875, 0.125]
        ref_stability = [ -0.5461756157848291 0.13772753623689454 NaN NaN;
                         -0.4619931721037595 0.7008326890567194 NaN NaN;
                         -0.8047640404112031 NaN NaN NaN;
                         -0.6946808326022188 0.01790866776878912 NaN NaN;
                         0.0779317550879532 NaN NaN NaN;
                         0.42596729822258317 0.004471339771113092 NaN NaN;
                         -0.5135794708889005 -0.03310005030782944 NaN NaN;
                         -0.5065206111895351 0.9103284212148111 NaN NaN]
        ref_dispersion = [ 0.8727358662984228 0.6268674463921016 2.220446049250313e-16 0.5606120216910173;
                          0.7351065923556201 0.8648558927554777 2.220446049250313e-16 1.0423815034982549;
                          0.7448846636964156 0.6002230796625391 2.220446049250313e-16 1.1554390259263865;
                          1.7166240612882144 0.8779819496662131 2.220446049250313e-16 0.05788537538701428;
                          0.9003385783471725 0.04365284049156737 2.220446049250313e-16 0.019149574064470987;
                          1.4050487363573019 0.9902060412128743 2.220446049250313e-16 0.3669213565103253;
                          0.7763831805507717 0.543290572945996 0.0 0.2348554998332349;
                          1.089040170020986 0.4291012199829881 2.220446049250313e-16 0.18499707402029192;
                          0.36515357013408956 0.4882494000681807 2.220446049250313e-16 0.2326031925443415;
                          0.6332267854890944 0.3970126344357655 2.220446049250313e-16 0.01635061891197087]
        ref_coverage = [0.9833333333333334, 0.9333333333333332, 1.0, 0.32]

        # Measured maxrel 3.5e-16, 6.6e-16 and 3.4e-16.
        @test exposure_agrees(exposure_correlation(csfm).X, ref_corr_bw)
        @test exposure_agrees(exposure_correlation(csfm; weighting = IdentityMetric()).X,
                              ref_corr_id)
        @test exposure_agrees(exposure_correlation(csfm;
                                                   weighting = RegressionWeightMetric()).X,
                              ref_corr_rw)
        # The third factor is the constant intercept, so every asset of it is in one tie. The
        # oracle breaks a tie by its sort, which on this fixture is the order of the asset
        # axis, and it scores a rank coefficient there. `ties = :ordinal` reproduces it.
        # Every rank coefficient below is bit-equal.
        @test exposure_agrees(exposure_ic(csfm3; ties = :ordinal).X, ref_ic_spearman)
        # Measured maxrel 6.3e-15 on the cell `0.0177`, maxscaled 9.3e-16: a correlation
        # near zero is a cancellation of the centred cross-products, so its error scales
        # with the largest cell.
        @test exposure_agrees(exposure_ic(csfm3; rank = false).X, ref_ic_pearson,
                              "linear ic"; rtol = 5e-14)
        @test exposure_agrees(exposure_ic(csfm3; horizon = 2, ties = :ordinal).X, ref_ic_h2)
        # Under the default `ties = :average`, a constant exposure has constant ranks, so it
        # has no rank coefficient, as it has no Pearson one. The two factors with no tie keep
        # the oracle's numbers.
        ica = exposure_ic(csfm3).X
        @test all(isnan, ica[:, 3])
        @test exposure_agrees(ica[:, 1:2], ref_ic_spearman[:, 1:2])
        ica2 = exposure_ic(csfm3; horizon = 2).X
        @test all(isnan, ica2[:, 3])
        @test exposure_agrees(ica2[:, 1:2], ref_ic_h2[:, 1:2])
        @test isequal(exposure_ic_summary(csfm3; mt = OverlapMeanTest()).mean_ic[1:2],
                      exposure_ic_summary(csfm3; ties = :ordinal, mt = OverlapMeanTest()).mean_ic[1:2])
        @test isnan(exposure_ic_summary(csfm3; mt = OverlapMeanTest()).mean_ic[3])
        # The summary is bit-equal, and the scaled hit rate measures maxrel 1.4e-16.
        s = exposure_ic_summary(csfm3; ties = :ordinal, mt = OverlapMeanTest())
        @test exposure_agrees(s.mean_ic, ref_mean_ic)
        @test exposure_agrees(s.std_ic, ref_std_ic)
        @test exposure_agrees(s.ic_ir, ref_ic_ir)
        @test exposure_agrees(s.hit_rate, port_hit_rate)
        @test count(isfinite, exposure_ic(csfm3).X[:, 1]) == 8
        @test size(exposure_ic(csfm3).X, 1) == 9
        @test exposure_agrees(port_hit_rate .* (8 / 9), ref_hit_rate)
        # Measured maxrel 4.2e-16.
        @test exposure_agrees(exposure_stability(csfm; step = 2).X, ref_stability)
        # The intercept is constant, so its dispersion is exactly zero. The oracle reads the
        # round-off `2.2e-16` of a mean that is not clamped to the range of the values, and
        # this library clamps it (#842, as in #1045): a Better answer, which `a constant
        # cross-section has no spread at any scale` pins. The other three columns agree
        # with the oracle, cell by cell: measured maxrel 1.5e-16.
        disp = copy(ref_dispersion)
        disp[:, 3] .= 0
        @test exposure_agrees(exposure_dispersion(csfm).X, disp, "dispersion")
        # The coverage reads the regression weights by default, so its universe is the
        # estimation universe of the fit, as the oracle's summary reads it. The benchmark
        # universe is one keyword away, and it gives a different answer here. Bit-equal.
        @test exposure_agrees(exposure_coverage(csfm).X, ref_coverage)
        @test exposure_coverage(csfm).X ==
              exposure_coverage(csfm; weighting = RegressionWeightMetric()).X
        @test exposure_coverage(csfm; weighting = BenchmarkWeightMetric()).X[4] == 0.34
    end

    @testset "the answers carry the axes the verbs state" begin
        @test size(exposure_correlation(csfm).X) == (K, K)
        @test size(exposure_ic(csfm3).X) == (T - 1, 3)
        @test size(exposure_ic(csfm3; horizon = 3).X) == (T - 3, 3)
        @test size(exposure_stability(csfm; step = 4).X) == (T - 4, K)
        @test size(exposure_dispersion(csfm).X) == (T, K)
        @test length(exposure_coverage(csfm).X) == K
        s3 = exposure_ic_summary(csfm3; mt = OverlapMeanTest())
        for v in (s3.mean_ic, s3.std_ic, s3.ic_ir, s3.t_stat, s3.hit_rate)
            @test length(v) == 3
        end
    end

    @testset "the block summary reads the overlap of its forward window" begin
        # `exposure_ic` scores every observation, so a window of `h` observations overlaps
        # its `h - 1` neighbours and the block method hands `h - 1` lags to the kernel. At
        # the default window there is no overlap and the bare method's default agrees.
        ic2 = exposure_ic(csfm3; horizon = 2).X
        s2 = exposure_ic_summary(csfm3; horizon = 2, mt = OverlapMeanTest())
        # The constant third factor has no rank coefficient, so its column is `NaN`, and
        # the comparisons read `isequal`.
        @test isequal(s2.t_stat,
                      exposure_ic_summary(ic2; lags = 1, mt = OverlapMeanTest()).t_stat)
        @test !isequal(s2.t_stat, exposure_ic_summary(ic2; mt = OverlapMeanTest()).t_stat)
        @test isequal(s2.mean_ic, exposure_ic_summary(ic2; mt = OverlapMeanTest()).mean_ic)
        @test isequal(s2.ic_ir, exposure_ic_summary(ic2; mt = OverlapMeanTest()).ic_ir)
        @test isequal(exposure_ic_summary(csfm3; mt = OverlapMeanTest()).t_stat,
                      exposure_ic_summary(exposure_ic(csfm3).X; mt = OverlapMeanTest()).t_stat)
        @test_throws DomainError exposure_ic_summary(ic2; lags = -1, mt = OverlapMeanTest())
    end

    @testset "the correlation of a constant exposure and of a sparse one" begin
        # The third factor is constant across the assets, so its correlation against any
        # other is `0` by convention. The fourth is never finite on three common assets,
        # so its correlation is `NaN`. The diagonal is `1` in both cases.
        C = exposure_correlation(csfm).X
        @test all(isone, LinearAlgebra.diag(C))
        @test C[1, 3] == C[3, 1] == 0
        @test C[2, 3] == C[3, 2] == 0
        @test all(isnan, C[1:3, 4])
        @test all(isnan, C[4, 1:3])
        @test exposure_agrees(C, Matrix(transpose(C)))
    end

    @testset "the information coefficient of a perfect forecast is one" begin
        rng2 = StableRNG(13571113)
        Ti, Ni = 5, 6
        R = randn(rng2, Ti, Ni)
        B = Array{Float64, 3}(undef, Ti, Ni, 1)
        # The exposure of an observation is the return of the observation that follows it,
        # so the exposure forecasts the return without error.
        for t in 1:(Ti - 1)
            B[t, :, 1] = R[t + 1, :]
        end
        B[Ti, :, 1] = R[1, :]
        @test isapprox(exposure_ic(B, R)[:, 1], ones(Ti - 1); rtol = 1e-12)
        @test isapprox(exposure_ic(B, R; rank = false)[:, 1], ones(Ti - 1); rtol = 1e-12)
        # The forward mean return of a horizon of one observation is the next return.
        @test PortfolioOptimisers.exposure_forward_mean_return(R, 1) == R[2:Ti, :]
        # A window with no finite return has no mean.
        Rn = copy(R)
        Rn[2, :] .= NaN
        @test all(isnan, PortfolioOptimisers.exposure_forward_mean_return(Rn, 1)[1, :])
        @test_throws DimensionMismatch PortfolioOptimisers.exposure_forward_mean_return(R,
                                                                                        Ti)
    end

    @testset "the stability of a constant exposure and of a reshuffled one" begin
        rng3 = StableRNG(97531)
        Ts, Ns = 4, 8
        B = Array{Float64, 3}(undef, Ts, Ns, 3)
        col = randn(rng3, Ns)
        for t in 1:Ts
            # The first factor keeps its ordering, so its stability is `1`.
            B[t, :, 1] = col
            # The second factor is constant across the assets, so it has no correlation and
            # its stability is `NaN`. The summary of #709 is what reads such a factor as
            # `1`.
            B[t, :, 2] .= 2.0
            # The third factor is drawn afresh at each observation, so its stability is the
            # correlation of two independent cross-sections.
            B[t, :, 3] = randn(rng3, Ns)
        end
        S = exposure_stability(B; step = 1)
        @test size(S) == (Ts - 1, 3)
        @test all(isapprox(1.0), S[:, 1])
        @test all(isnan, S[:, 2])
        @test all(abs.(S[:, 3]) .< 1)
        @test_throws DomainError exposure_stability(B; step = 0)
        @test_throws DimensionMismatch exposure_stability(B; step = Ts)
    end

    @testset "the dispersion of a standardised exposure is one" begin
        rng4 = StableRNG(2468013)
        Td, Nd = 3, 7
        B = Array{Float64, 3}(undef, Td, Nd, 2)
        for t in 1:Td
            x = randn(rng4, Nd)
            # A cross-section of zero mean and unit population standard deviation has a
            # dispersion of exactly one under equal weights.
            B[t, :, 1] = (x .- Statistics.mean(x)) ./ Statistics.std(x; corrected = false)
            B[t, :, 2] = 3.0 * B[t, :, 1]
        end
        D = exposure_dispersion(B)
        @test isapprox(D[:, 1], ones(Td); rtol = 1e-12)
        @test isapprox(D[:, 2], fill(3.0, Td); rtol = 1e-12)
        # A factor whose whole cross-section is absent has no dispersion.
        Bn = copy(B)
        Bn[1, :, 1] .= NaN
        @test isnan(exposure_dispersion(Bn)[1, 1])
        @test !isnan(exposure_dispersion(Bn)[1, 2])
    end

    @testset "the coverage counts the universe of each observation" begin
        B = Array{Float64, 3}(undef, 2, 4, 2)
        B[:, :, 1] .= 1.0
        B[:, :, 2] .= 1.0
        # One asset of the universe carries no exposure, so the coverage is `1 - 1/n`.
        B[1, 2, 2] = NaN
        @test exposure_coverage(B) == [1.0, 1 - 1 / 8]
        # An asset outside the universe counts in neither the numerator nor the
        # denominator, so a missing exposure there costs no coverage.
        w = ones(2, 4)
        w[1, 2] = 0.0
        @test exposure_coverage(B, w) == [1.0, 1.0]
        # An observation whose universe is empty contributes nothing.
        w0 = ones(2, 4)
        w0[1, :] .= 0.0
        @test exposure_coverage(B, w0) == [0.5, 0.5]
    end

    @testset "a weighting member reads the field it names" begin
        # Each member reads one field of the block over the whole observation axis.
        @test PortfolioOptimisers.cs_diagnostic_weights(BenchmarkWeightMetric(), csfm) ===
              bw
        @test PortfolioOptimisers.cs_diagnostic_weights(RegressionWeightMetric(), csfm) ===
              rw
        @test PortfolioOptimisers.cs_diagnostic_weights(InverseIdiosyncraticVarianceMetric(),
                                                        csfm) == 1 ./ vs
        @test isnothing(PortfolioOptimisers.cs_diagnostic_weights(IdentityMetric(), csfm))
        # The answer under a member is the answer under the history it names.
        @test exposure_agrees(exposure_dispersion(csfm;
                                                  weighting = RegressionWeightMetric()).X,
                              exposure_dispersion(Ms, rw))
        @test exposure_agrees(exposure_stability(csfm; step = 2,
                                                 weighting = IdentityMetric()).X,
                              exposure_stability(Ms; step = 2))
        # A block that carries no such field refuses, and the refusal names the field.
        bare = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr,
                                         Ms = Ms, lag = 1)
        for metric in (BenchmarkWeightMetric(), RegressionWeightMetric(),
                       InverseIdiosyncraticVarianceMetric())
            @test_throws PortfolioOptimisers.IsNothingError exposure_correlation(bare;
                                                                                 weighting = metric)
        end
        @test_throws PortfolioOptimisers.IsNothingError exposure_stability(bare; step = 2)
        @test_throws PortfolioOptimisers.IsNothingError exposure_dispersion(bare)
        for metric in (BenchmarkWeightMetric(), InverseIdiosyncraticVarianceMetric())
            @test_throws PortfolioOptimisers.IsNothingError exposure_coverage(bare;
                                                                              weighting = metric)
        end
        # The coverage alone reads an absent regression weight history as a fit on every
        # asset, so its universe is every asset, and it refuses nothing.
        @test isnothing(PortfolioOptimisers.exposure_coverage_weights(RegressionWeightMetric(),
                                                                      bare))
        @test exposure_coverage(bare).X ==
              exposure_coverage(bare; weighting = IdentityMetric()).X ==
              exposure_coverage(Ms)
    end

    @testset "a block that carries no history refuses by name" begin
        empty_block = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N))
        @test_throws PortfolioOptimisers.IsNothingError exposure_correlation(empty_block;
                                                                             weighting = IdentityMetric())
        @test_throws PortfolioOptimisers.IsNothingError exposure_ic(empty_block)
        no_fit = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), Ms = Ms,
                                           lag = 1)
        @test_throws PortfolioOptimisers.IsNothingError exposure_ic(no_fit)
        @test_throws PortfolioOptimisers.IsNothingError exposure_ic_summary(no_fit;
                                                                            mt = OverlapMeanTest())
        # The exposure history and the fit must carry the same axes.
        short = CrossSectionalRegression(; f = f[1:(T - 1), :], eps = eps[1:(T - 1), :],
                                         n = fill(N, T - 1))
        mismatched = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = short,
                                               Ms = Ms, lag = 1)
        @test_throws DimensionMismatch exposure_ic(mismatched)
        long_lag = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr,
                                             Ms = Ms, lag = T)
        @test_throws DimensionMismatch exposure_ic(long_lag)
    end

    @testset "the level-1 verbs check their arguments" begin
        B = randn(StableRNG(5), 3, 4, 2)
        @test_throws DimensionMismatch exposure_dispersion(B, ones(2, 4))
        @test_throws DomainError exposure_dispersion(B, fill(-1.0, 3, 4))
        @test_throws PortfolioOptimisers.IsEmptyError exposure_dispersion(zeros(0, 0, 0))
        @test_throws DimensionMismatch exposure_ic(B, randn(StableRNG(6), 2, 4))
        @test_throws DomainError exposure_ic(B, randn(StableRNG(6), 3, 4); horizon = 0)
        @test_throws DimensionMismatch exposure_ic(B, randn(StableRNG(6), 3, 4);
                                                   horizon = 3)
        @test_throws PortfolioOptimisers.IsEmptyError exposure_ic_summary(zeros(0, 0);
                                                                          mt = OverlapMeanTest())
        # A history of ones is what an absent weight history resolves to.
        @test PortfolioOptimisers.exposure_weights(B, nothing) == ones(3, 4)
        @test PortfolioOptimisers.exposure_weights(B, ones(3, 4)) == ones(3, 4)
    end

    @testset "the correlation kernels" begin
        # The weighted correlation of two cross-sections, and the cases with no answer.
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 4.0, 6.0, 8.0]
        u = ones(4)
        @test PortfolioOptimisers.cs_weighted_correlation(a, b, u) ≈ 1.0
        @test PortfolioOptimisers.cs_weighted_correlation(a, -b, u) ≈ -1.0
        # Fewer than three assets enter, so there is no answer.
        @test isnan(PortfolioOptimisers.cs_weighted_correlation(a, b, [1.0, 1.0, 0.0, 0.0]))
        # One cross-section is constant, so the denominator vanishes.
        @test isnan(PortfolioOptimisers.cs_weighted_correlation(a, ones(4), u))
        # A weight that is not finite excludes the asset.
        @test isnan(PortfolioOptimisers.cs_weighted_correlation(a, b, [1.0, 1.0, NaN, NaN]))
        # The rank correlation reads the order and not the level, so a monotone map of one
        # cross-section leaves it unchanged.
        @test PortfolioOptimisers.cs_spearman_correlation(a, exp.(b)) ≈ 1.0
        @test PortfolioOptimisers.cs_spearman_correlation(a, [1.0, 2.0, NaN, 4.0]) ≈ 1.0
        # Under the default, equal values share the mean of their positions, so a constant
        # cross-section has constant ranks and no rank correlation (#1332).
        ra = PortfolioOptimisers.cs_ranks([1.0, 1.0, 1.0], fill(true, 3), :average)
        @test ra == [2.0, 2.0, 2.0]
        @test isnan(PortfolioOptimisers.cs_spearman_correlation(ones(5), collect(1.0:5.0)))
        @test isnan(PortfolioOptimisers.cs_spearman_correlation(ones(5),
                                                                collect(5.0:-1:1.0)))
        # Under `:ordinal`, a tie takes the order of the asset axis, so the ranks are a
        # permutation, and the asset order sets the answer of a constant cross-section.
        ro = PortfolioOptimisers.cs_ranks([1.0, 1.0, 1.0], fill(true, 3), :ordinal)
        @test ro == [1.0, 2.0, 3.0]
        @test PortfolioOptimisers.cs_spearman_correlation(ones(5), collect(1.0:5.0);
                                                          ties = :ordinal) ≈ 1.0
        @test PortfolioOptimisers.cs_spearman_correlation(ones(5), collect(5.0:-1:1.0);
                                                          ties = :ordinal) ≈ -1.0
        # Two targets that differ only inside the ties of the first cross-section get one
        # answer under the default, and two under `:ordinal`.
        tb = [1.0, 1.0, 2.0, 2.0]
        for (yb, ordv) in (([1.0, 2.0, 3.0, 4.0], 1.0), ([2.0, 1.0, 4.0, 3.0], 0.6))
            @test PortfolioOptimisers.cs_spearman_correlation(tb, yb) ≈ 4 / sqrt(20)
            @test PortfolioOptimisers.cs_spearman_correlation(tb, yb; ties = :ordinal) ≈
                  ordv
        end
        # A run of two and a run of three, with a masked entry sorted to the end.
        rm = PortfolioOptimisers.cs_ranks([2.0, 1.0, 2.0, Inf, 2.0, 1.0],
                                          [true, true, true, false, true, true], :average)
        @test rm[[1, 2, 3, 5, 6]] == [4.0, 1.5, 4.0, 4.0, 1.5]
        @test isnan(rm[4])
        # The two rules give the same ranks to a cross-section with no tie.
        xr = randn(StableRNG(1332), 20)
        @test PortfolioOptimisers.cs_ranks(xr, trues(20), :average) ==
              PortfolioOptimisers.cs_ranks(xr, trues(20), :ordinal)
        for tr in (:average, :ordinal)
            rn = PortfolioOptimisers.cs_ranks([1.0, 2.0, Inf], [true, true, false], tr)
            @test rn[1:2] == [1.0, 2.0]
            @test isnan(rn[3])
        end
        @test_throws PortfolioOptimisers.ConflictingArgumentError PortfolioOptimisers.cs_ranks([1.0],
                                                                                               [true],
                                                                                               :dense)
        @test_throws PortfolioOptimisers.ConflictingArgumentError PortfolioOptimisers.cs_spearman_correlation(a,
                                                                                                              b;
                                                                                                              ties = :min)
    end

    @testset "a re-based block answers on the reduced axis when it is asked to" begin
        rng6 = StableRNG(864213579)
        Tr, Nr, Kr = 6, 5, 3
        Msr = randn(rng6, Tr, Nr, Kr)
        # A fit under a re-basis estimates its factor returns on the reduced axis.
        fr = 0.02 * randn(rng6, Tr, Kr - 1)
        epsr = 0.01 * randn(rng6, Tr, Nr)
        rwr = abs.(randn(rng6, Tr, Nr)) .+ 0.1
        fcb = FactorFamilyBasis(; fnm = ["industry"], fi = [[1, 2]], di = [2],
                                ratios = reshape(collect(range(0.4, 0.9; length = Tr)), Tr,
                                                 1), K = Kr)
        csr_r = CrossSectionalRegression(; f = fr, eps = epsr, n = fill(Nr, Tr))
        L = PortfolioOptimisers.reduce_loadings(fcb, Msr[Tr, :, :])
        blk = CrossSectionalFactorModel(; M = Msr[Tr, :, :], L = L, b = zeros(Nr),
                                        csr = csr_r, Ms = Msr, rw = rwr, fcb = fcb, lag = 1,
                                        nf = ["tech", "bank", "value"],
                                        fam = ["industry", "industry", "style"])
        Kred = PortfolioOptimisers.reduced_factor_count(fcb)
        @test Kred == 2
        # The exposure group answers on the raw axis, because it reads the exposure history
        # and never the design of the fit.
        @test size(exposure_correlation(blk; weighting = IdentityMetric()).X) == (Kr, Kr)
        @test size(exposure_dispersion(blk; weighting = IdentityMetric()).X) == (Tr, Kr)
        @test size(exposure_ic(blk).X) == (Tr - 1, Kr)
        # `reduced` is the one mode that maps the exposures through the re-basis first.
        @test size(exposure_ic(blk; reduced = true).X) == (Tr - 1, Kred)
        @test size(exposure_ic_summary(blk; reduced = true, mt = OverlapMeanTest()).mean_ic) ==
              (Kred,)
        # A block that carries no re-basis has one axis, whatever `reduced` states.
        @test exposure_agrees(exposure_ic(csfm3; reduced = true).X, exposure_ic(csfm3).X)

        # Each answer carries the names and the families of its own axis: the raw axis, or
        # the reduced axis, which drops the second industry.
        PO = PortfolioOptimisers
        raw = (exposure_correlation(blk; weighting = IdentityMetric()),
               exposure_dispersion(blk; weighting = IdentityMetric()), exposure_ic(blk),
               exposure_stability(blk; step = 2, weighting = IdentityMetric()),
               exposure_coverage(blk; weighting = IdentityMetric()))
        for r in raw
            @test r.nf == ["tech", "bank", "value"]
            @test r.fam == ["industry", "industry", "style"]
            v = PO.port_opt_view(r, LabelGroup("industry"))
            @test v.nf == ["tech", "bank"]
            @test isequal(PO.port_opt_view(r, ["value"]).X,
                          PO.port_opt_view(r, LabelGroup("style")).X)
        end
        red = exposure_ic(blk; reduced = true)
        @test red.nf == ["tech", "value"] && red.fam == ["industry", "style"]
        @test isequal(PO.port_opt_view(red, LabelGroup("style")).X, red.X[:, 2:2])
        @test_throws ArgumentError PO.port_opt_view(red, ["bank"])
        # The correlation selects the rows and the columns of a family together.
        C = raw[1]
        @test C.dims == (1, 2)
        Ci = PO.port_opt_view(C, LabelGroup("industry"))
        @test isequal(Ci.X, C.X[1:2, 1:2])
        @test isequal(PO.port_opt_view(C, ["value", "tech"]).X, C.X[[3, 1], [3, 1]])
        # The summary of a view is the view of the summary, and it keeps the labels.
        s = exposure_ic_summary(blk; reduced = true, mt = OverlapMeanTest())
        @test s isa ExposureICSummaryResult
        @test s.nf == ["tech", "value"] && s.fam == ["industry", "style"]
        sv = PO.port_opt_view(s, LabelGroup("style"))
        sr = exposure_ic_summary(PO.port_opt_view(red, LabelGroup("style"));
                                 mt = OverlapMeanTest())
        for f in
            (:mean_ic, :std_ic, :ic_ir, :t_stat, :p_value, :mean_ci_lower, :mean_ci_upper,
             :hit_rate, :nf, :fam)
            @test isequal(getfield(sv, f), getfield(s, f)[2:2])
            @test isequal(getfield(sr, f), getfield(s, f)[2:2])
        end
        # The summary reads a series, not a matrix of factor pairs.
        @test_throws ArgumentError exposure_ic_summary(C; mt = OverlapMeanTest())
    end

    @testset "a diagnostic answer checks its axis, and a view refuses what it cannot find" begin
        PO = PortfolioOptimisers
        X = [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test FactorDiagnosticResult(; X = X).dims == (2,)
        @test_throws DomainError FactorDiagnosticResult(; X = X, dims = ())
        @test_throws DomainError FactorDiagnosticResult(; X = X, dims = (3,))
        @test_throws DomainError FactorDiagnosticResult(; X = X, dims = (2, 2))
        @test_throws DimensionMismatch FactorDiagnosticResult(; X = X, dims = (1, 2))
        @test_throws DimensionMismatch FactorDiagnosticResult(; X = X, nf = ["a", "b"])
        @test_throws ArgumentError FactorDiagnosticResult(; X = X, nf = ["a", "b", "a"])
        @test_throws DimensionMismatch FactorDiagnosticResult(; X = X, fam = ["f"])
        # A view by name needs names, and a view by family needs families.
        bare = FactorDiagnosticResult(; X = X)
        @test_throws ArgumentError PO.port_opt_view(bare, ["a"])
        @test_throws ArgumentError PO.port_opt_view(bare, LabelGroup("style"))
        @test isequal(PO.port_opt_view(bare, [3, 1]).X, X[:, [3, 1]])
        @test isnothing(PO.port_opt_view(bare, 2:3).nf)
        named = FactorDiagnosticResult(; X = X, nf = ["value", "size", "tech"],
                                       fam = ["style", "style", "industry"])
        err = try
            PO.port_opt_view(named, LabelGroup("styl"))
        catch e
            e
        end
        @test occursin("did you mean `style`", err.msg)
        @test_throws ArgumentError PO.port_opt_view(named, ["valu"])
        # A view of a view selects on the axis the first view kept.
        vv = PO.port_opt_view(PO.port_opt_view(named, LabelGroup("style")), ["size"])
        @test vv.X == X[:, 2:2] && vv.fam == ["style"]
    end

    @testset "an asset enters only on a finite positive weight" begin
        rng7 = StableRNG(842)
        Bw = randn(rng7, 1, 4, 2)
        # Two assets of positive weight are fewer than the three a correlation needs, and a
        # zero weight does not make up the count.
        @test isnan(exposure_correlation(Bw, [1.0 1.0 0.0 0.0])[1, 2])
        # A `NaN` or an infinite weight excludes its own asset and not the observation.
        for bad in (NaN, Inf)
            wb = [1.0 1.0 1.0 bad]
            @test exposure_correlation(Bw, wb) ≈ exposure_correlation(Bw[:, 1:3, :])
            @test exposure_dispersion(Bw, wb) ≈ exposure_dispersion(Bw[:, 1:3, :])
        end
        # The universe of the coverage reads the same rule: the asset of infinite weight
        # leaves the universe, so the three finite exposures cover all of it.
        Bc = copy(Bw)
        Bc[1, 4, :] .= NaN
        @test exposure_coverage(Bc, [1.0 1.0 1.0 Inf]) == [1.0, 1.0]
        @test exposure_coverage(Bc) == [0.75, 0.75]
    end

    @testset "a constant cross-section has no spread at any scale" begin
        # The mean of equal values rounds. Without the clamp, the correlation below came out
        # `0.0`, because the round-off of the constant side times a spread of `1e6` cleared
        # the tolerance.
        @test isnan(PortfolioOptimisers.cs_weighted_correlation([0.1, 0.1, 0.1],
                                                                [1e6, 3e6, 2e6], ones(3)))
        rng8 = StableRNG(2842)
        for x in (0.1, 0.3, 0.7), n in (3, 5, 7)
            @test iszero(exposure_dispersion(fill(x, 1, n, 1), rand(rng8, 1, n))[1, 1])
        end
        # The intercept of the fixture is constant, so its dispersion is exactly zero, where
        # the oracle reads the round-off `2.2e-16`.
        @test all(iszero, exposure_dispersion(csfm).X[:, 3])
    end

    @testset "the verbs agree with the textbook estimators" begin
        rng9 = StableRNG(1842)
        Tn, Nn, Kn, H = 30, 12, 3, 3
        Bn = randn(rng9, Tn, Nn, Kn)
        Bn[rand(rng9, Tn, Nn, Kn) .< 0.1] .= NaN
        Rn = 0.02 * randn(rng9, Tn, Nn)
        Rn[rand(rng9, Tn, Nn) .< 0.05] .= NaN
        wn = rand(rng9, Tn, Nn)
        wn[rand(rng9, Tn, Nn) .< 0.1] .= 0.0
        function wcor(x, z, u)
            m = (u .> 0) .& isfinite.(x) .& isfinite.(z)
            return count(m) < 3 ? NaN : cor(hcat(x[m], z[m]), Weights(u[m]), 1)[1, 2]
        end
        function srank(x, z)
            m = isfinite.(x) .& isfinite.(z)
            return count(m) < 3 ? NaN : corspearman(x[m], z[m])
        end
        y = PortfolioOptimisers.exposure_forward_mean_return(Rn, H)
        yref = [mean(filter(isfinite, Rn[(t + 1):(t + H), i]))
                for t in 1:(Tn - H), i in 1:Nn]
        @test y ≈ yref
        Cref = [if k == l
                    1.0
                else
                    mean(filter(isfinite,
                                [wcor(Bn[t, :, k], Bn[t, :, l], wn[t, :]) for t in 1:Tn]))
                end
                for k in 1:Kn, l in 1:Kn]
        @test exposure_correlation(Bn, wn) ≈ Cref
        @test exposure_ic(Bn, Rn, wn; horizon = H, rank = false) ≈
              [wcor(Bn[t, :, k], y[t, :], wn[t, :]) for t in 1:(Tn - H), k in 1:Kn]
        @test exposure_ic(Bn, Rn, wn; horizon = H) ≈
              [srank(Bn[t, :, k], y[t, :]) for t in 1:(Tn - H), k in 1:Kn]
        @test exposure_stability(Bn, wn; step = 4) ≈
              [wcor(Bn[t, :, k], Bn[t + 4, :, k], wn[t, :]) for t in 1:(Tn - 4), k in 1:Kn]
        Dref = [(m = isfinite.(Bn[t, :, k]) .& (wn[t, :] .> 0);
                 std(Bn[t, m, k], Weights(wn[t, m]); corrected = false))
                for t in 1:Tn, k in 1:Kn]
        @test exposure_dispersion(Bn, wn) ≈ Dref
        @test exposure_coverage(Bn, wn) ≈
              [mean(count(isfinite.(Bn[t, :, k]) .& (wn[t, :] .> 0)) / count(wn[t, :] .> 0)
                    for t in 1:Tn) for k in 1:Kn]
        # The t-statistic over two lags, written out from its definition.
        ic = exposure_ic(Bn, Rn, wn; horizon = H)
        # The sums below read every row, which holds because no row of this fixture is `NaN`.
        @test !any(isnan, ic)
        s2 = exposure_ic_summary(ic; lags = 2, mt = OverlapMeanTest())
        for k in 1:Kn
            c = ic[:, k]
            m = mean(c)
            g = sum((c[t] - m) * (c[t + j] - m) for j in 1:2 for t in 1:(length(c) - j))
            # The divisor `n - 2L - 1` removes the bias of the centring at order `L = 2`.
            @test s2.t_stat[k] ≈
                  m / sqrt((sum(abs2, c .- m) + 2 * g) / (length(c) - 5)) * sqrt(length(c))
            @test s2.std_ic[k] ≈ std(c)
            @test s2.hit_rate[k] == mean(c .> 0)
        end
    end

    @testset "the verbs derive their types from the data" begin
        rng10 = StableRNG(3842)
        Bi = rand(rng10, -3:3, 8, 5, 2)
        Ri = rand(rng10, -3:3, 8, 5)
        Bf = float.(Bi)
        Rf = float.(Ri)
        # An integer panel answers in `Float64`, with the numbers of the same panel in floats.
        @test exposure_correlation(Bi) ≈ exposure_correlation(Bf)
        @test isequal(exposure_ic(Bi, Ri), exposure_ic(Bf, Rf))
        @test isequal(exposure_ic(Bi, Ri; rank = false), exposure_ic(Bf, Rf; rank = false))
        @test isequal(exposure_stability(Bi; step = 2), exposure_stability(Bf; step = 2))
        @test exposure_dispersion(Bi) ≈ exposure_dispersion(Bf)
        @test exposure_coverage(Bi) == exposure_coverage(Bf)
        @test PortfolioOptimisers.cs_ranks([3, 1, 1, 2], trues(4), :average) ==
              [4.0, 1.5, 1.5, 3.0]
        ici = rand(rng10, -1:1, 6, 2)
        @test isequal(exposure_ic_summary(ici; lags = 1, mt = OverlapMeanTest()),
                      exposure_ic_summary(float.(ici); lags = 1, mt = OverlapMeanTest()))
        # A `Rational` panel keeps its means exact, and a square root answers in `Float64`.
        Rr = Rational{Int}.(Ri)
        @test PortfolioOptimisers.exposure_forward_mean_return(Rr, 2) isa
              Matrix{Rational{Int}}
        @test exposure_coverage(Rational{Int}.(Bi)) isa Vector{Rational{Int}}
        sr = exposure_ic_summary(Rational{Int}.(ici); lags = 1, mt = OverlapMeanTest())
        @test sr.mean_ic isa Vector{Rational{Int}}
        @test sr.hit_rate isa Vector{Rational{Int}}
        @test sr.std_ic ≈
              exposure_ic_summary(float.(ici); lags = 1, mt = OverlapMeanTest()).std_ic
        @test exposure_correlation(Rational{Int}.(Bi)) ≈ exposure_correlation(Bf)
        # A `Float32` panel stays in `Float32`.
        B32 = Float32.(Bf)
        R32 = Float32.(Rf)
        @test eltype(exposure_correlation(B32)) == Float32
        @test eltype(exposure_ic(B32, R32)) == Float32
        @test eltype(exposure_dispersion(B32)) == Float32
        @test eltype(exposure_ic_summary(exposure_ic(B32, R32); mt = OverlapMeanTest()).t_stat) ==
              Float32
    end
end

@testset "The information coefficient reconstructs the returns the observed factors carry (#1367)" begin
    PO = PortfolioOptimisers
    rng = StableRNG(1_367_09)
    T, N = 8, 6
    Ms = randn(rng, T, N, 3)
    f = 0.02 * randn(rng, T, 2)
    fx = 0.01 * randn(rng, T, 1)
    eps = 0.01 * randn(rng, T, N)
    csr = CrossSectionalRegression(; f = f, eps = eps, n = fill(N, T))
    blk = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr, Ms = Ms,
                                    lag = 1, fx = fx)
    B, R, _ = PO.exposure_ic_data(blk, nothing, false)
    F = hcat(f, fx)
    for t in 2:T
        @test R[t, :] ≈ Ms[t - 1, :, :] * F[t, :] + eps[t, :]
    end
    @test size(exposure_ic(blk).X) == (T - 1, 3)
    # The returns the reconstruction gives back, observed part included, score every factor
    # as the reconstruction does (#1524).
    rd = ReturnsResult(; nx = string.(1:N), X = R)
    @test isequal(exposure_ic(blk, rd).X, exposure_ic(blk).X)
end

@testset "The information coefficient over the returns data scores each factor alone (#1524)" begin
    PO = PortfolioOptimisers
    rng = StableRNG(1_524)
    T, N, K = 10, 6, 4
    Ms = randn(rng, T, N, K)
    f = 0.02 * randn(rng, T, K)
    eps = 0.01 * randn(rng, T, N)
    rw = abs.(randn(rng, T, N)) .+ 0.1
    # The fourth factor is finite on two assets alone.
    Ms[:, 3:6, 4] .= NaN
    csr = CrossSectionalRegression(; f = f, eps = eps, n = fill(N, T))
    csfm = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr, Ms = Ms,
                                     rw = rw, lag = 1)
    csr3 = CrossSectionalRegression(; f = f[:, 1:3], eps = eps, n = fill(N, T))
    csfm3 = CrossSectionalFactorModel(; M = Ms[T, :, 1:3], b = zeros(N), csr = csr3,
                                      Ms = Ms[:, :, 1:3], rw = rw, lag = 1)
    # The reconstruction reads the sparse factor, so four assets of six lose their return,
    # and no factor keeps the three assets a coefficient needs.
    @test all(isnan, exposure_ic(csfm).X)
    # The returns of the three dense factors. Their first row has no lagged exposure, and
    # no score reads it, because a score reads the returns after its observation.
    _, R3, _ = PO.exposure_ic_data(csfm3, nothing, false)
    rd3 = ReturnsResult(; nx = string.(1:N), X = R3)
    ic = exposure_ic(csfm, rd3)
    @test size(ic.X) == (T - 1, K)
    @test ic.nf == csfm.nf
    @test isequal(ic.X[:, 1:3], exposure_ic(csfm3).X)
    @test all(isfinite, ic.X[:, 1:2])
    @test isequal(exposure_ic(csfm, rd3; rank = false).X[:, 1:3],
                  exposure_ic(csfm3; rank = false).X)
    # Two assets are too few for a coefficient of the sparse factor itself.
    @test all(isnan, ic.X[:, 4])
    # On the dense block, the returns the reconstruction gives back score as it does.
    @test isequal(exposure_ic(csfm3, rd3).X, exposure_ic(csfm3).X)
    # The summary derives its lags from the window on both forms.
    @test isequal(exposure_ic_summary(csfm, rd3; horizon = 2, mt = OverlapMeanTest()).t_stat[1:3],
                  exposure_ic_summary(csfm3; horizon = 2, mt = OverlapMeanTest()).t_stat)

    # A block that records the position of each row reads those rows of the returns data.
    pos = CrossSectionalFactorModel(; M = Ms[T, :, 1:3], b = zeros(N), csr = csr3,
                                    Ms = Ms[:, :, 1:3], rw = rw, lag = 1,
                                    idx = collect(3:(T + 2)))
    rdp = ReturnsResult(; nx = string.(1:N), X = vcat(randn(rng, 2, N), R3))
    @test isequal(exposure_ic(pos, rdp).X, exposure_ic(csfm3).X)
    # The returns data form reads no fit.
    no_fit = CrossSectionalFactorModel(; M = Ms[T, :, 1:3], b = zeros(N),
                                       Ms = Ms[:, :, 1:3], rw = rw, lag = 1)
    @test_throws PO.IsNothingError exposure_ic(no_fit)
    @test isequal(exposure_ic(no_fit, rd3).X, exposure_ic(csfm3).X)

    # The reconstruction starts at row `lag` of the block. The returns data has every row,
    # so it scores the rows before that too.
    lag2 = CrossSectionalFactorModel(; M = Ms[T, :, 1:3], b = zeros(N), csr = csr3,
                                     Ms = Ms[:, :, 1:3], rw = rw, lag = 2)
    R = 0.01 * randn(rng, T, N)
    rdr = ReturnsResult(; nx = string.(1:N), X = R)
    @test size(exposure_ic(lag2).X) == (T - 2, 3)
    @test isequal(exposure_ic(lag2, rdr).X, exposure_ic(Ms[:, :, 1:3], R))
    @test isequal(exposure_ic(lag2, rdr; rank = false, horizon = 2).X,
                  exposure_ic(Ms[:, :, 1:3], R, rw; rank = false, horizon = 2))

    # The refusals name what is missing.
    @test_throws PO.IsNothingError exposure_ic(csfm,
                                               ReturnsResult(; nf = ["a"],
                                                             F = randn(rng, T, 1)))
    @test_throws PO.IsNothingError exposure_ic(CrossSectionalFactorModel(; M = Ms[T, :, :],
                                                                         b = zeros(N)), rdr)
    @test_throws DimensionMismatch exposure_ic(csfm,
                                               ReturnsResult(; nx = string.(1:N),
                                                             X = R[1:(T - 1), :]))
    @test_throws DimensionMismatch PO.exposure_ic_returns(1:3, R, T)
    @test_throws BoundsError PO.exposure_ic_returns(collect(2:(T + 1)), R, T)
end

#=
Check `src/05_Moments/32_CrossSectionalFactorModel/04_FactorExposures/08_Base_Exposure.jl`, `09_CompositeExposure.jl`,
`10_DerivedExposure.jl`, `11_OneHotExposure.jl` and `12_ConstantExposure.jl` against the
contract their docstrings state, and against the stored oracle. Issue #721.

FOUR CONVENTIONS SHAPE THE PROBES.

1. AN INACTIVE CELL IS `NaN`, on every member. The Descriptors carry the convention into
   the composite and the derived member; the one-hot member and the constant member write
   it themselves. The hand panels below carry a value on an inactive cell on purpose.

2. THE COMBINATION IS FINITE-AWARE. A Descriptor that is `NaN` on a cell contributes
   neither its score nor its weight there, so the surviving weight is the coverage of the
   cell. `min_coverage` is a threshold on that weight, not on a count of Descriptors.

3. A BENCHMARK WEIGHT IS A SELECTOR. `cross_sectional_transform` refuses a `NaN` weight,
   so the read zeroes every unobserved and inactive cell, which puts it outside the
   estimation set of its observation rather than inside it with an unknown weight.

4. THE STORED CASES ARE THE OUTPUT OF THE ORACLE.
   `assets/CompositeExposure.csv.gz` and `assets/DerivedExposure.csv.gz` hold the oracle's
   composite and derived members, driven on the synthetic panel the last testset rebuilds,
   with the same two raw Descriptors, the same weights, the same coverage threshold, the same
   two transforms and the same grouping. They compare cell by cell with `parity_compare` of
   the harness of #1376 (#1381). The other two members agree to the last bit, so they are
   checked here against their closed forms, and `test_12u` stores the one-hot block.

ONE DELIBERATE DEVIATION. The oracle's constant exposure returns one on every cell, including a
cell outside the active universe. The library writes `NaN` there, because the root's contract
makes every Factor Exposure `NaN` on an inactive cell. The regression drops those cells on both
sides, so no fitted quantity moves.
=#
include(joinpath(@__DIR__, "test06c_setup.jl"))

# A small hand panel carrying the benchmark weights beside the fields. Every numeric field
# takes a forward fill, so each earns an observed-mask column and a raw `NaN` reads back as
# `NaN` rather than as the fill value.
function exposure_hand_panel(fields::AbstractVector{<:Pair{String, <:AbstractMatrix}};
                             amsk::AbstractMatrix{Bool} = trues(size(fields[1][2])...),
                             emsk::AbstractMatrix{Bool} = amsk,
                             bw::Union{Nothing, <:AbstractMatrix} = nothing,
                             sectors::Union{Nothing, <:AbstractMatrix} = nothing,
                             sector_alg::PortfolioOptimisers.AbstractPanelFillAlgorithm = NoPanelFill())
    inputs = Any[NumericPanelInput(; name = n, vals = v,
                                   alg = ForwardPanelFill(; val = 0.0))
                 for (n, v) in fields]
    if !isnothing(bw)
        push!(inputs,
              NumericPanelInput(; name = "benchmark_weights", vals = bw,
                                alg = ForwardPanelFill(; val = 0.0)))
    end
    if !isnothing(sectors)
        push!(inputs,
              CategoricalPanelInput(; name = "sector", vals = sectors, alg = sector_alg))
    end
    res = asset_panel(identity.(inputs); amsk = amsk, emsk = emsk)
    T, N = size(amsk)
    return ReturnsResult(; nx = ["A" * string(i) for i in 1:N], X = zeros(T, N), pnl = res)
end

@testset "Exposure constructors and their refusals" begin
    @testset "Every member carries the oracle's own defaults" begin
        xc = CompositeExposure(; descriptors = [Passthrough(; field = "a")])
        @test isa(xc.outlier, CrossSectionalWinsoriser)
        @test isa(xc.scoring, CrossSectionalStandardiser)
        @test isnothing(xc.weights)
        @test isnothing(xc.group)
        @test iszero(xc.min_coverage)
        @test xc.bw == "benchmark_weights"
        @test xc.family == "style"
        xd = DerivedExposure(; source = "size", f = abs2)
        @test isnothing(xd.outlier)
        @test isa(xd.scoring, CrossSectionalStandardiser)
        @test xd.bw == "benchmark_weights"
        @test xd.family == "style"
        @test ConstantExposure().family == "market"
        @test OneHotExposure(; field = "sector", family = "sector").family == "sector"
    end
    @testset "Every member is an Exposure Estimator" begin
        for xe in (CompositeExposure(; descriptors = [Passthrough(; field = "a")]),
                   DerivedExposure(; source = "size", f = abs2),
                   OneHotExposure(; field = "sector", family = "sector"), ConstantExposure())
            @test isa(xe, PortfolioOptimisers.AbstractExposureEstimator)
            @test isa(xe, PortfolioOptimisers.AbstractEstimator)
        end
    end
    @testset "The constructors refuse what the docstrings say they refuse" begin
        @test_throws PortfolioOptimisers.IsEmptyError CompositeExposure(;
                                                                        descriptors = PortfolioOptimisers.AbstractDescriptorEstimator[])
        @test_throws DimensionMismatch CompositeExposure(;
                                                         descriptors = [Passthrough(;
                                                                                    field = "a")],
                                                         weights = [0.5, 0.5])
        @test_throws DomainError CompositeExposure(;
                                                   descriptors = [Passthrough(;
                                                                              field = "a"),
                                                                  Passthrough(;
                                                                              field = "b")],
                                                   weights = [-0.5, 1.5])
        @test_throws DomainError CompositeExposure(;
                                                   descriptors = [Passthrough(;
                                                                              field = "a"),
                                                                  Passthrough(;
                                                                              field = "b")],
                                                   weights = [0.5, 0.4])
        @test_throws DomainError CompositeExposure(;
                                                   descriptors = [Passthrough(;
                                                                              field = "a")],
                                                   min_coverage = 1.5)
        @test_throws DomainError CompositeExposure(;
                                                   descriptors = [Passthrough(;
                                                                              field = "a")],
                                                   min_coverage = NaN)
        @test_throws PortfolioOptimisers.IsEmptyError CompositeExposure(;
                                                                        descriptors = [Passthrough(;
                                                                                                   field = "a")],
                                                                        family = "")
        @test_throws PortfolioOptimisers.IsEmptyError CompositeExposure(;
                                                                        descriptors = [Passthrough(;
                                                                                                   field = "a")],
                                                                        bw = "")
        @test_throws PortfolioOptimisers.IsEmptyError CompositeExposure(;
                                                                        descriptors = [Passthrough(;
                                                                                                   field = "a")],
                                                                        group = "")
        @test_throws PortfolioOptimisers.IsEmptyError DerivedExposure(; source = "",
                                                                      f = abs2)
        @test_throws PortfolioOptimisers.IsEmptyError DerivedExposure(; source = "s",
                                                                      f = abs2, family = "")
        @test_throws PortfolioOptimisers.IsEmptyError OneHotExposure(; field = "",
                                                                     family = "sector")
        @test_throws PortfolioOptimisers.IsEmptyError OneHotExposure(; field = "sector",
                                                                     family = "")
        @test_throws PortfolioOptimisers.IsEmptyError ConstantExposure(; family = "")
    end
    @testset "The equal-weight composite is the mean of its Descriptors" begin
        @test PortfolioOptimisers.composite_weights(nothing, 4) == fill(0.25, 4)
        @test PortfolioOptimisers.composite_weights([0.4, 0.6], 2) == [0.4, 0.6]
    end
end

@testset "The shared reads of the exposure base" begin
    rd = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]]; amsk = [true true; true false],
                             bw = [1.0 2.0; NaN 4.0])
    @testset "A benchmark weight is zero where it is unobserved or inactive" begin
        # The blank at observation 2 of asset A1 is active, so its zero is named (#1520). The
        # inactive cell of asset A2 is not.
        W = @test_logs((:warn,
                        r"not finite on 1 active cell\(s\) of the Asset Panel, first at observation 2 for the assets \[\"A1\"\]"),
                       PortfolioOptimisers.exposure_benchmark_weights(rd,
                                                                      "benchmark_weights"))
        @test W == [1.0 2.0; 0.0 0.0]
        @test all(isfinite, W)
        @test_throws r"not finite on 1 active cell" PortfolioOptimisers.exposure_benchmark_weights(rd,
                                                                                                   "benchmark_weights";
                                                                                                   strict = true)
        # A weight that is zero on an inactive cell alone stays silent.
        ri = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]];
                                 amsk = [true true; true false], bw = [1.0 2.0; 3.0 NaN])
        Wi = @test_logs PortfolioOptimisers.exposure_benchmark_weights(ri,
                                                                       "benchmark_weights";
                                                                       strict = true)
        @test Wi == [1.0 2.0; 3.0 0.0]
        @test PortfolioOptimisers.exposure_weight_fill!([1.0 NaN; Inf 4.0], ri.pnl) ==
              [(2, 1), (1, 2)]
    end
    @testset "A member that names no grouping field partitions nothing" begin
        @test isnothing(PortfolioOptimisers.exposure_group_labels(rd, nothing))
    end
    @testset "A nothing transform slot is the identity" begin
        X = [1.0 2.0; 3.0 4.0]
        @test PortfolioOptimisers.exposure_transform(nothing, X, nothing, nothing) === X
    end
    @testset "The active fill covers both ranks, and checks the shape" begin
        L2 = [1.0 2.0; 3.0 4.0]
        PortfolioOptimisers.exposure_active_fill!(L2, rd.pnl)
        @test isequal(L2, [1.0 2.0; 3.0 NaN])
        L3 = ones(2, 2, 3)
        PortfolioOptimisers.exposure_active_fill!(L3, rd.pnl)
        @test all(isnan, L3[2, 2, :])
        @test all(isone, L3[1, :, :])
        @test_throws DimensionMismatch PortfolioOptimisers.exposure_active_fill!(ones(3, 3),
                                                                                 rd.pnl)
        @test_throws DimensionMismatch PortfolioOptimisers.exposure_active_fill!(ones(3, 3,
                                                                                      2),
                                                                                 rd.pnl)
    end
end

@testset "The composite on a hand panel" begin
    a = [1.0 2.0; 3.0 4.0]
    b = [5.0 6.0; 7.0 8.0]
    rd = exposure_hand_panel(["a" => a, "b" => b]; bw = ones(2, 2))
    des = [Passthrough(; field = "a"), Passthrough(; field = "b")]
    @testset "Fixed weights combine the Descriptors when both slots are nothing" begin
        xe = CompositeExposure(; descriptors = des, weights = [0.25, 0.75],
                               outlier = nothing, scoring = nothing)
        @test factor_exposure(xe, rd) ≈ 0.25 .* a .+ 0.75 .* b
    end
    @testset "The equal-weight default is the plain mean" begin
        xe = CompositeExposure(; descriptors = des, outlier = nothing, scoring = nothing)
        @test factor_exposure(xe, rd) ≈ 0.5 .* a .+ 0.5 .* b
    end
    @testset "A single Descriptor takes no second scoring pass" begin
        xe = CompositeExposure(; descriptors = [Passthrough(; field = "a")],
                               outlier = nothing, scoring = nothing)
        @test factor_exposure(xe, rd) ≈ a
    end
    @testset "The coverage threshold is on weight, not on a count" begin
        # The second Descriptor is blank on the first cell, so that cell carries a
        # surviving weight of 0.4, which is the weight of the Descriptor that remains.
        bn = copy(b)
        bn[1, 1] = NaN
        rdn = exposure_hand_panel(["a" => a, "b" => bn]; bw = ones(2, 2))
        xhi = CompositeExposure(; descriptors = des, weights = [0.4, 0.6],
                                min_coverage = 0.5, outlier = nothing, scoring = nothing)
        L = factor_exposure(xhi, rdn)
        @test isnan(L[1, 1])
        @test L[2, 1] ≈ 0.4 * a[2, 1] + 0.6 * bn[2, 1]
        xlo = CompositeExposure(; descriptors = des, weights = [0.4, 0.6],
                                min_coverage = 0.0, outlier = nothing, scoring = nothing)
        @test factor_exposure(xlo, rdn)[1, 1] ≈ a[1, 1]
        # A threshold at exactly the surviving weight admits the cell.
        xeq = CompositeExposure(; descriptors = des, weights = [0.4, 0.6],
                                min_coverage = 0.4, outlier = nothing, scoring = nothing)
        @test factor_exposure(xeq, rdn)[1, 1] ≈ a[1, 1]
    end
    @testset "A cell no Descriptor reaches is NaN, whatever the threshold" begin
        an = copy(a)
        bn = copy(b)
        an[1, 2] = NaN
        bn[1, 2] = NaN
        rdn = exposure_hand_panel(["a" => an, "b" => bn]; bw = ones(2, 2))
        xe = CompositeExposure(; descriptors = des, min_coverage = 0.0, outlier = nothing,
                               scoring = nothing)
        @test isnan(factor_exposure(xe, rdn)[1, 2])
    end
    @testset "An inactive cell is NaN, whatever the Panel Fields hold" begin
        rdi = exposure_hand_panel(["a" => a, "b" => b]; amsk = [true true; true false],
                                  bw = ones(2, 2))
        xe = CompositeExposure(; descriptors = des, outlier = nothing, scoring = nothing)
        L = factor_exposure(xe, rdi)
        @test isnan(L[2, 2])
        @test L[1, 1] ≈ 0.5 * (a[1, 1] + b[1, 1])
    end
    @testset "Both slots run on every Descriptor, and the composite is scored again" begin
        ct = CrossSectionalStandardiser(; min_group_size = 1)
        xe = CompositeExposure(; descriptors = des, outlier = nothing, scoring = ct)
        Sa = cross_sectional_transform(ct, a; w = ones(2, 2))
        Sb = cross_sectional_transform(ct, b; w = ones(2, 2))
        expected = cross_sectional_transform(ct, 0.5 .* Sa .+ 0.5 .* Sb; w = ones(2, 2))
        @test factor_exposure(xe, rd) ≈ expected
    end
    @testset "The grouping field reaches the transforms" begin
        rdg = exposure_hand_panel(["a" => [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0]];
                                  bw = ones(2, 4),
                                  sectors = ["t" "t" "b" "b"; "t" "t" "b" "b"])
        ct = CrossSectionalStandardiser(; min_group_size = 2)
        xe = CompositeExposure(; descriptors = [Passthrough(; field = "a")],
                               outlier = nothing, scoring = ct, group = "sector")
        G = cross_sectional_groups(rdg.pnl, "sector")
        expected = cross_sectional_transform(ct,
                                             PortfolioOptimisers.panel_field_values(rdg,
                                                                                    "a");
                                             w = ones(2, 4), groups = G)
        @test factor_exposure(xe, rdg) ≈ expected
    end
    @testset "The estimator holds no data: two carriers, one estimator" begin
        xe = CompositeExposure(; descriptors = des, outlier = nothing, scoring = nothing)
        rd2 = exposure_hand_panel(["a" => 2 .* a, "b" => 2 .* b]; bw = ones(2, 2))
        @test factor_exposure(xe, rd) ≈ 0.5 .* (a .+ b)
        @test factor_exposure(xe, rd2) ≈ a .+ b
    end
end

@testset "The derived exposure on a hand panel" begin
    rd = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]]; bw = ones(2, 2))
    xs = [1.0 2.0; 3.0 4.0]
    @testset "The function runs, then the two slots" begin
        xe = DerivedExposure(; source = "size", f = x -> x .^ 2, scoring = nothing)
        @test factor_exposure(xe, rd, xs) ≈ xs .^ 2
        ct = CrossSectionalStandardiser(; min_group_size = 1)
        xs2 = DerivedExposure(; source = "size", f = x -> x .^ 2, scoring = ct)
        @test factor_exposure(xs2, rd, xs) ≈
              cross_sectional_transform(ct, xs .^ 2; w = ones(2, 2))
    end
    @testset "An inactive cell is NaN whatever f returns there" begin
        amsk = [true false; true true]
        rdm = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]]; bw = ones(2, 2), amsk = amsk)
        xe = DerivedExposure(; source = "size", f = x -> zero(x), scoring = nothing)
        L = factor_exposure(xe, rdm, [1.0 NaN; 3.0 4.0])
        @test isnan(L[1, 2])
        @test L[[1, 2, 4]] == [0.0, 0.0, 0.0]
    end
    @testset "The two-argument method refuses, naming the source" begin
        xe = DerivedExposure(; source = "size", f = abs2)
        @test_throws ArgumentError factor_exposure(xe, rd)
    end
    @testset "A source of the wrong shape, and a function that changes it, are refused" begin
        xe = DerivedExposure(; source = "size", f = x -> x .^ 2, scoring = nothing)
        @test_throws DimensionMismatch factor_exposure(xe, rd, ones(3, 3))
        xf = DerivedExposure(; source = "size", f = x -> x[:, 1:1], scoring = nothing)
        @test_throws DimensionMismatch factor_exposure(xf, rd, xs)
    end
end

@testset "The one-hot exposure on a hand panel" begin
    rd = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]]; amsk = [true true; true false],
                             sectors = ["tech" "banks"; "tech" "tech"])
    xe = OneHotExposure(; field = "sector", family = "sector")
    @testset "One factor per level, named by the expansion form" begin
        @test PortfolioOptimisers.one_hot_exposure_names(xe, rd) ==
              ["sector=banks", "sector=tech"]
        L = factor_exposure(xe, rd)
        @test size(L) == (2, 2, 2)
        @test L[1, 1, :] == [0.0, 1.0]
        @test L[1, 2, :] == [1.0, 0.0]
        @test L[2, 1, :] == [0.0, 1.0]
    end
    @testset "An inactive cell is NaN across every level" begin
        @test all(isnan, factor_exposure(xe, rd)[2, 2, :])
    end
    @testset "Integer returns take a float exposure, which holds the inactive NaN" begin
        # The levels used to take the element type of the returns, so integer returns threw
        # `InexactError: Int64(NaN)` at the first inactive cell.
        rdi = ReturnsResult(; nx = rd.nx, X = zeros(Int, 2, 2), pnl = rd.pnl)
        @test isequal(factor_exposure(xe, rdi), factor_exposure(xe, rd))
        rdf = ReturnsResult(; nx = rd.nx, X = zeros(Float32, 2, 2), pnl = rd.pnl)
        @test eltype(factor_exposure(xe, rdf)) == Float32
    end
    @testset "An asset cannot set no level" begin
        # A categorical Panel Field stores one code per cell, so the level-less cell the
        # one-hot layout could express is now unrepresentable: the code is refused rather
        # than read back as a row of zeros.
        rdb = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]];
                                  sectors = ["tech" "banks"; "tech" "tech"])
        f = PortfolioOptimisers.panel_field(rdb.pnl, "sector")
        @test all(c -> 1 <= c <= length(f.levels), f.codes)
        @test_throws DomainError CategoricalPanelField(; name = "sector", levels = f.levels,
                                                       codes = zeros(Int, size(f.codes)))
    end
    @testset "A level the fill wrote is NaN, not a classification the asset carried" begin
        # The field can blank, so it earns an observed-mask column. The forward fill
        # resolves the blank to a level, and the read must undo that resolution.
        rdf = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]];
                                  sectors = ["tech" missing; "tech" "banks"],
                                  sector_alg = ForwardPanelFill(; val = "tech"))
        L = factor_exposure(xe, rdf)
        @test all(isnan, L[1, 2, :])
        @test L[1, 1, :] == [0.0, 1.0]
        @test L[2, 2, :] == [1.0, 0.0]
    end
    @testset "A numeric Panel Field, and a carrier with no Asset Panel, are refused" begin
        @test_throws ArgumentError factor_exposure(OneHotExposure(; field = "a",
                                                                  family = "f"), rd)
        rdn = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2))
        @test_throws PortfolioOptimisers.IsNothingError factor_exposure(xe, rdn)
    end
end

@testset "The constant exposure on a hand panel" begin
    rd = exposure_hand_panel(["a" => [1.0 2.0; 3.0 4.0]]; amsk = [true true; true false])
    @testset "The ones column, NaN where the asset is not listed" begin
        L = factor_exposure(ConstantExposure(), rd)
        @test isequal(L, [1.0 1.0; 1.0 NaN])
    end
    @testset "Integer returns take a float exposure, which holds the inactive NaN" begin
        # The ones used to take the element type of the returns, so integer returns threw
        # `InexactError: Int64(NaN)` at the first inactive cell.
        rdi = ReturnsResult(; nx = rd.nx, X = zeros(Int, 2, 2), pnl = rd.pnl)
        @test isequal(factor_exposure(ConstantExposure(), rdi), [1.0 1.0; 1.0 NaN])
        rdf = ReturnsResult(; nx = rd.nx, X = zeros(Float32, 2, 2), pnl = rd.pnl)
        @test eltype(factor_exposure(ConstantExposure(), rdf)) == Float32
    end
    @testset "A carrier with no Asset Panel is refused" begin
        rdn = ReturnsResult(; nx = ["A1", "A2"], X = zeros(2, 2))
        @test_throws PortfolioOptimisers.IsNothingError factor_exposure(ConstantExposure(),
                                                                        rdn)
    end
end

@testset "The members reproduce the stored oracle" begin
    sp = synthetic_asset_panel(; n_assets = 20, n_observations = 60, n_industries = 4,
                               late_listing_proba = 0.3, delisting_proba = 0.3,
                               missing_ratio = 0.08, rng = StableRNG(987654321))
    rd = sp.rd
    ct_out = CrossSectionalWinsoriser()
    ct_sco = CrossSectionalStandardiser(; min_group_size = 2)
    xc = CompositeExposure(;
                           descriptors = [Passthrough(; field = "book_equity"),
                                          Passthrough(; field = "market_cap")],
                           weights = [0.4, 0.6], min_coverage = 0.5, outlier = ct_out,
                           scoring = ct_sco, group = "industry", bw = "market_cap")
    Lc = factor_exposure(xc, rd)
    xd = DerivedExposure(; source = "style", f = x -> x .^ 3, scoring = ct_sco,
                         group = "industry", bw = "market_cap")
    Ld = factor_exposure(xd, rd, Lc)
    @testset "The composite matches the stored case cell by cell" begin
        E = Matrix(CSV.read(joinpath(@__DIR__, "assets/CompositeExposure.csv.gz"),
                            DataFrame))
        @test size(Lc) == size(E)
        # Measured maxrel 3.9e-14 cell by cell, and the NaN pattern is equal.
        @test parity_compare(Lc, E; name = "composite").ok
    end
    @testset "The derived exposure matches the stored case cell by cell" begin
        E = Matrix(CSV.read(joinpath(@__DIR__, "assets/DerivedExposure.csv.gz"), DataFrame))
        @test size(Ld) == size(E)
        # Measured maxrel 4.4e-14 cell by cell, and the NaN pattern is equal.
        @test parity_compare(Ld, E; name = "derived").ok
    end
    @testset "The one-hot block is the classification, and the constant is the ones" begin
        Lo = factor_exposure(OneHotExposure(; field = "industry", family = "industry"), rd)
        G = cross_sectional_groups(rd.pnl, "industry")
        amsk = rd.pnl.amsk
        @test size(Lo) == (size(amsk)..., 4)
        for k in CartesianIndices(amsk)
            if amsk[k]
                @test Lo[k, :] == [l == G[k] ? 1.0 : 0.0 for l in 1:4]
            else
                @test all(isnan, Lo[k, :])
            end
        end
        Lk = factor_exposure(ConstantExposure(), rd)
        @test isequal(isnan.(Lk), .!amsk)
        @test all(isone, Lk[amsk])
    end
end

#=
The idiosyncratic group of the cross-sectional diagnostics, decided by #709 and built by #800.

The testset `the stored oracle` compares with literals that the oracle gave on the very
arrays this file builds. Rebuild them by an export of the same fixture and a run of the
oracle's factor model block on it. Each comparison is cell by cell (`parity_compare`), and
states the worst value measured beside its tolerance.
=#
using Statistics

@testset "Cross-sectional idiosyncratic diagnostics" begin
    # The fixture the oracle was measured on. Every mutation below drives one branch, so a
    # change to any of them invalidates the literals of `the stored oracle`. The order
    # matters: `eps` is drawn from the unmutated `vs`, so a mutation of `vs` that moved
    # above the draw would change every entry of both arrays.
    rng = StableRNG(13579)
    T, N = 12, 8
    vs = abs.(randn(rng, T, N)) .+ 0.5
    eps = sqrt.(vs) .* randn(rng, T, N)
    # A standardised return of ten drives the tail rate above zero.
    eps[3, 2] = 10 * sqrt(vs[3, 2])
    # A predicted variance of zero leaves the asset with no standardised return.
    vs[2, 3] = 0.0
    # A negative variance estimate is clamped to zero, so it reads the same way.
    vs[4, 5] = -0.25
    # An idiosyncratic return that is not finite costs its own asset alone.
    eps[6, 1] = NaN
    # A variance that is not finite costs the asset in every series of the group.
    vs[7, 4:8] .= NaN
    # A cross-section that is constant has no kurtosis and no skewness.
    eps[8, :] .= 0.0
    # An observation with one finite asset answers no moment at all.
    vs[10, 2:8] .= 0.0
    # An observation with no finite asset answers nothing, not even the tail rate.
    vs[11, :] .= 0.0
    # An observation with three finite assets answers the skewness and not the kurtosis.
    vs[12, 4:8] .= 0.0

    csr = CrossSectionalRegression(; f = 0.02 * randn(rng, T, 3), eps = eps, n = fill(N, T))
    csfm = CrossSectionalFactorModel(; M = randn(rng, N, 3), b = zeros(N), csr = csr,
                                     vs = vs, lag = 1)

    # Cell by cell: the pattern of the absent answers agrees exactly, and each finite cell
    # lies within `rtol` of its own size.
    # Against the oracle, the calibration measures maxrel 1.7e-16 and the rest is
    # bit-equal, except the two higher moments, which state their own tolerance.
    function idio_agrees(a, b, label = ""; rtol = 1e-14)
        return parity_compare(a, b; rtol = rtol, name = label).ok
    end

    @testset "the stored oracle" begin
        ref_cal = [0.6576024139953216, 0.9566856355619364, 3.5819946565338796,
                   1.0519333239956812, 0.9497821692436044, 1.2977713493444354,
                   0.6959235710960231, 0.0, 0.8998539717557806, NaN, NaN,
                   0.5846367760017855]
        ref_tr3 = [0.0, 0.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, 0.0]
        ref_tr2 = [0.0, 0.0, 0.125, 0.14285714285714285, 0.0, 0.0, 0.0, 0.0, 0.125, 0.0,
                   NaN, 0.0]
        ref_kur = [-0.15254705014793474, -0.8040388814186846, 5.876859489839838,
                   0.184494728400335, -0.7683124676705245, -2.3086251082994793, NaN, NaN,
                   1.408003196921393, NaN, NaN, NaN]
        ref_skw = [-0.13260664664663818, -0.5295931972573044, 2.2683821115641236,
                   0.6761337170660473, -0.387749898137298, 0.26469296344660576,
                   0.03572329744216094, NaN, 0.6348304054154105, NaN, NaN,
                   1.0182452423606356]
        # The oracle ranks a cross-section with an unstable sort, so a tie block takes an
        # order the sort chose and not one a rule states. Observation ten carries seven
        # predicted volatilities that are exactly zero, and the oracle answers
        # `0.21428571428571427` there. Under `ties = :ordinal`, `cs_ranks` breaks a tie by the
        # order of the asset axis, which is the deterministic reading, and the oracle
        # reproduces this series to the last bit once its own ranks are made stable. The
        # entry below is the library's answer under that rule, and it is the only one of the
        # ninety-eight this file asserts that parts from the oracle as it runs.
        ref_ic = [-0.09523809523809523, -0.5714285714285714, -0.047619047619047616,
                  -0.3333333333333333, -0.07142857142857142, -0.023809523809523808, NaN,
                  0.2857142857142857, -0.8095238095238095, 0.19047619047619047,
                  -0.19047619047619047]
        ref_rd = [-0.35714285714285715, -0.6071428571428571, -0.30952380952380953,
                  -0.10714285714285714, -0.17857142857142858, -0.11904761904761904, NaN,
                  0.19047619047619047, -0.9047619047619048, NaN, NaN]

        # The oracle divides each return by the volatility of its own observation, which
        # `ahead = false` states (#1387).
        @test idio_agrees(idio_calibration(eps, vs; ahead = false), ref_cal)
        @test idio_agrees(idio_tail_rate(eps, vs; ahead = false), ref_tr3)
        @test idio_agrees(idio_tail_rate(eps, vs; threshold = 2, ahead = false), ref_tr2)
        # Measured maxrel 1.9e-14 and 3.6e-15, maxscaled 4.9e-16 and 2.1e-16. An excess
        # kurtosis subtracts 3 from a ratio of moments, and a skewness near zero cancels its
        # centred cubes, so a small cell carries the round-off of the largest one.
        @test idio_agrees(idio_kurtosis(eps, vs; ahead = false), ref_kur, "kurtosis";
                          rtol = 1e-13)
        @test idio_agrees(idio_skewness(eps, vs; ahead = false), ref_skw, "skewness";
                          rtol = 5e-14)
        @test idio_agrees(idio_vol_ic(eps, vs; ties = :ordinal), ref_ic)
        @test idio_agrees(idio_vol_residual_dependence(eps, vs; ties = :ordinal), ref_rd)
        # Under the default `ties = :average`, the two observations whose predictions tie
        # move. Observation eleven predicts zero for every asset, so its ranks are constant
        # and it has no coefficient. Observation ten gives its seven zeros their mean rank,
        # which `tiedrank` states. No other entry has a tie, so each keeps the oracle's
        # number. The residual dependence divides by the zeros, so it has no tie to move.
        ica = idio_vol_ic(eps, vs)
        sig = PortfolioOptimisers.idio_predicted_volatility(vs)
        @test idio_agrees(ica[1:9], ref_ic[1:9])
        @test ica[10] ≈ cor(tiedrank(sig[10, :]), tiedrank(abs.(eps[11, :])))
        @test isnan(ica[11])
        @test idio_agrees(idio_vol_residual_dependence(eps, vs), ref_rd)
        @test isequal(idio_vol_ic(csfm), ica)
        @test isequal(idio_vol_ic(csfm; ties = :ordinal),
                      idio_vol_ic(eps, vs; ties = :ordinal))

        # The oracle's own summary, under its own key names. Measured maxrel 1.0e-15.
        s = idio_calibration_summary(eps, vs; ahead = false)
        @test idio_agrees([s.mean_cs_std, s.median_cs_std, s.mean_kurtosis, s.mean_skewness,
                           s.mean_tail_rate],
                          [1.0676183867528448, 0.9248180704996924, 0.4908334153749917,
                           0.4275619994726381, 0.011363636363636364], "summary")
    end

    @testset "the level-0 kernel" begin
        z = standardised_idio_returns(eps, vs; ahead = false)
        @test size(z) == (T, N)
        # A zero variance, a negative variance and a return that is not finite each read
        # `NaN`, and a variance that is not finite reads `NaN` too.
        @test isnan(z[2, 3])
        @test isnan(z[4, 5])
        @test isnan(z[6, 1])
        @test all(isnan, z[7, 4:8])
        @test all(isnan, z[11, :])
        # Every other entry is the return over the predicted volatility.
        @test z[1, 1] ≈ eps[1, 1] / sqrt(vs[1, 1])
        @test z[3, 2] ≈ 10.0
        # The predicted volatility clamps a negative variance to zero.
        s = PortfolioOptimisers.idio_predicted_volatility(vs)
        @test s[4, 5] == 0.0
        @test s[1, 1] ≈ sqrt(vs[1, 1])
        @test isnan(s[7, 4])
        # The four `z`-verbs answer the same over `z` as over the two histories.
        @test idio_agrees(idio_calibration(z), idio_calibration(eps, vs; ahead = false))
        @test idio_agrees(idio_tail_rate(z), idio_tail_rate(eps, vs; ahead = false))
        @test idio_agrees(idio_kurtosis(z), idio_kurtosis(eps, vs; ahead = false))
        @test idio_agrees(idio_skewness(z), idio_skewness(eps, vs; ahead = false))
        # The default divides the return of row `t` by the volatility of row `t - 1`, so the
        # first row has no forecast and every other row is the same-row kernel of the
        # shifted variance history.
        za = standardised_idio_returns(eps, vs)
        @test all(isnan, za[1, :])
        @test isequal(za[2:end, :],
                      standardised_idio_returns(eps[2:end, :], vs[1:(end - 1), :];
                                                ahead = false))
        @test za[4, 2] ≈ eps[4, 2] / sqrt(vs[3, 2])
        # The zero variance of row 2 costs the return of row 3, not of row 2.
        @test isfinite(za[2, 3]) && isnan(za[3, 3])
        @test idio_agrees(idio_calibration(za), idio_calibration(eps, vs))
        @test idio_agrees(idio_tail_rate(za), idio_tail_rate(eps, vs))
        @test idio_agrees(idio_kurtosis(za), idio_kurtosis(eps, vs))
        @test idio_agrees(idio_skewness(za), idio_skewness(eps, vs))
        @test idio_calibration_summary(eps, vs) ==
              idio_calibration_summary(eps[2:end, :], vs[1:(end - 1), :]; ahead = false)
    end

    @testset "a variance that has read its own return bounds the standardised return" begin
        # An exponentially weighted variance puts the weight `1 - λ` on the latest squared
        # return, so `v_t ≥ (1 - λ) ε_t²` and the same-row ratio can never pass
        # `1 / sqrt(1 - λ)`, however large the shock. The variance of the previous row has not
        # read the shock, so the default reports it at its full size (#1387).
        lam = 0.9
        e = [0.01 0.01; -0.01 0.01; 0.01 -0.01; 0.5 0.01; 0.01 0.01]
        v = similar(e)
        v[1, :] .= 1e-4
        for t in 2:size(e, 1)
            v[t, :] = lam .* v[t - 1, :] .+ (1 - lam) .* e[t, :] .^ 2
        end
        zs = standardised_idio_returns(e, v; ahead = false)
        za = standardised_idio_returns(e, v)
        @test maximum(abs, filter(isfinite, zs)) <= 1 / sqrt(1 - lam)
        @test za[4, 1] ≈ 0.5 / sqrt(v[3, 1])
        @test za[4, 1] > 45
        @test idio_tail_rate(e, v; threshold = 4, ahead = false)[4] == 0.0
        @test idio_tail_rate(e, v; threshold = 4)[4] == 0.5
    end

    @testset "the level-2 methods read the block" begin
        @test idio_agrees(idio_calibration(csfm), idio_calibration(eps, vs))
        @test idio_agrees(idio_tail_rate(csfm), idio_tail_rate(eps, vs))
        @test idio_agrees(idio_tail_rate(csfm; threshold = 2),
                          idio_tail_rate(eps, vs; threshold = 2))
        @test idio_agrees(idio_kurtosis(csfm), idio_kurtosis(eps, vs))
        @test idio_agrees(idio_skewness(csfm), idio_skewness(eps, vs))
        @test idio_agrees(idio_vol_ic(csfm), idio_vol_ic(eps, vs))
        @test idio_agrees(idio_vol_residual_dependence(csfm),
                          idio_vol_residual_dependence(eps, vs))
        @test idio_calibration_summary(csfm) == idio_calibration_summary(eps, vs)
        @test idio_calibration_summary(csfm; threshold = 2) ==
              idio_calibration_summary(eps, vs; threshold = 2)
        for f in (idio_calibration, idio_kurtosis, idio_skewness, idio_tail_rate)
            @test idio_agrees(f(csfm; ahead = false), f(eps, vs; ahead = false))
        end
        @test idio_calibration_summary(csfm; threshold = 2, ahead = false) ==
              idio_calibration_summary(eps, vs; threshold = 2, ahead = false)
    end

    @testset "the analytic cases" begin
        # A panel whose predicted variance is the true one, constant in time and ranking
        # the assets. The standardised returns are then standard normal by construction.
        rng2 = StableRNG(2468)
        T2, N2 = 400, 80
        vs2 = repeat(collect(range(0.5, 4.0; length = N2))', T2, 1)
        eps2 = sqrt.(vs2) .* randn(rng2, T2, N2)
        s2 = idio_calibration_summary(eps2, vs2)
        # The calibration is near one, the tail rate near the Gaussian reference, and the
        # kurtosis and the skewness near zero.
        @test isapprox(s2.mean_cs_std, 1.0; atol = 0.02)
        @test isapprox(s2.median_cs_std, 1.0; atol = 0.02)
        # The Gaussian reference of a threshold of three, `2Φ(-3)`.
        @test isapprox(s2.mean_tail_rate, 0.002699796063260207; atol = 0.0015)
        @test isapprox(s2.mean_kurtosis, 0.0; atol = 0.15)
        @test isapprox(s2.mean_skewness, 0.0; atol = 0.08)
        # The volatility ranks the truth, so the information coefficient is high, and the
        # standardised move no longer depends on it, so the residual dependence is near zero.
        @test mean(idio_vol_ic(eps2, vs2)) > 0.2
        @test isapprox(mean(idio_vol_residual_dependence(eps2, vs2)), 0.0; atol = 0.03)
        # A scaled variance history moves the calibration by the inverse scale.
        @test idio_agrees(idio_calibration(eps2, 4 .* vs2),
                          idio_calibration(eps2, vs2) ./ 2)
        # It moves neither dependence series: a positive scale is monotone, so it leaves
        # every rank of both cross-sections where it was.
        @test idio_vol_ic(eps2, 4 .* vs2) == idio_vol_ic(eps2, vs2)
        @test idio_vol_residual_dependence(eps2, 4 .* vs2) ==
              idio_vol_residual_dependence(eps2, vs2)
        # The five entries of the summary are the means and the median of the series. The
        # summary sums the entries in order and `Statistics.mean` sums them pairwise, so
        # the two agree to the last bits of the answer and not bit for bit. The first row
        # has no forecast, so every series is `NaN` there and the summary skips it.
        z2 = standardised_idio_returns(eps2, vs2)
        @test all(isnan, z2[1, :])
        @test s2.mean_cs_std ≈ mean(idio_calibration(z2)[2:end]) rtol = 1e-12
        @test s2.median_cs_std == median(idio_calibration(z2)[2:end])
        @test s2.mean_kurtosis ≈ mean(idio_kurtosis(z2)[2:end]) rtol = 1e-12
        @test s2.mean_skewness ≈ mean(idio_skewness(z2)[2:end]) rtol = 1e-12
        @test s2.mean_tail_rate ≈ mean(idio_tail_rate(z2)[2:end]) rtol = 1e-12
    end

    @testset "an aggregate over a series that never answered" begin
        # Every predicted variance is zero, so no observation carries a standardised
        # return, every series is `NaN` throughout, and every aggregate answers `NaN`.
        eps0 = ones(3, 4)
        vs0 = zeros(3, 4)
        @test all(isnan, standardised_idio_returns(eps0, vs0))
        @test all(isnan, idio_calibration(eps0, vs0))
        @test all(isnan, idio_tail_rate(eps0, vs0))
        s0 = idio_calibration_summary(eps0, vs0)
        @test all(isnan,
                  (s0.mean_cs_std, s0.median_cs_std, s0.mean_kurtosis, s0.mean_skewness,
                   s0.mean_tail_rate))
        @test isnan(PortfolioOptimisers.idio_nan_mean([NaN, NaN]))
        @test isnan(PortfolioOptimisers.idio_nan_median([NaN, NaN]))
        @test PortfolioOptimisers.idio_nan_mean([1.0, NaN, 3.0]) == 2.0
        @test PortfolioOptimisers.idio_nan_median([1.0, NaN, 3.0, 5.0]) == 3.0
        # The kernel of the moments answers `NaN` at an observation with no finite asset.
        m = PortfolioOptimisers.idio_row_moments(standardised_idio_returns(eps0, vs0), 1)
        @test m.n == 0
        @test all(isnan, (m.m2, m.m3, m.m4))
    end

    @testset "a constant cross-section, an infinite entry and an integer input" begin
        # A cross-section of equal non-zero values has no skewness and no kurtosis. Its mean
        # carries round-off, so a centred moment computed from it is often a tiny positive
        # number and not zero: before the exact test, 4195 of 20000 such rows of 3 to 7
        # assets gave a finite skewness, up to 2.45. The counts avoid 4, whose mean is exact.
        for nc in (3, 5, 6, 7), x in (0.1, -0.7, 1.3, 2.9)
            zc = fill(x, 2, nc)
            @test all(isnan, idio_skewness(zc))
            @test nc < 4 || all(isnan, idio_kurtosis(zc))
            @test idio_calibration(zc) == zeros(2)
            mc = PortfolioOptimisers.idio_row_moments(zc, 1)
            @test (mc.n, mc.m2, mc.m3, mc.m4) == (nc, 0.0, 0.0, 0.0)
        end
        # An infinite entry enters neither the count nor the sum of the tail rate, so the
        # rate over the finite entries 0.5, 4 and -5 is two in three, never one.
        @test idio_tail_rate([Inf 0.5 4.0 NaN -5.0]) == [2 / 3]
        # An integer history answers in floating point rather than raising `InexactError`.
        e_int = [1 -2 3 0; 3 4 -1 2; 0 1 2 -3]
        v_int = [1 4 1 0; 1 0 4 1; 1 1 1 1]
        z_int = standardised_idio_returns(e_int, v_int; ahead = false)
        @test eltype(z_int) == Float64
        @test isequal(z_int, [1.0 -1.0 3.0 NaN; 3.0 NaN -0.5 2.0; 0.0 1.0 2.0 -3.0])
        @test idio_calibration([1 2 3; 4 5 7]) ≈ [1.0, std([4, 5, 7])]
        # A Float32 history keeps its element type through every series.
        rng32 = StableRNG(7)
        e32 = randn(rng32, Float32, 6, 9)
        v32 = rand(rng32, Float32, 6, 9) .+ 0.5f0
        for f in
            (idio_calibration, idio_tail_rate, idio_kurtosis, idio_skewness, idio_vol_ic,
             idio_vol_residual_dependence)
            @test eltype(f(e32, v32)) == Float32
        end
        @test idio_calibration_summary(e32, v32).mean_cs_std isa Float32
        # Each series takes its type from its own operation, never from `float`. A square
        # root is inexact, so a `Rational` history answers in the type `sqrt` lands in. A
        # rate and the moment ratio of the kurtosis only divide, so an exact `z` stays exact.
        er = Rational{Int}.(round.(Int, 8 .* e32)) .// 8
        vr = Rational{Int}.(round.(Int, 8 .* v32)) .// 8
        @test eltype(standardised_idio_returns(er, vr)) == typeof(sqrt(1 // 1))
        @test eltype(idio_vol_ic(er, vr)) == typeof(sqrt(1 // 1))
        zr = [1//2 -3//2 2//1 1//4 -1//1; 3//1 -1//3 1//5 2//1 -2//1]
        @test idio_tail_rate(zr; threshold = 2) == [0 // 1, 1 // 5]
        @test eltype(idio_kurtosis(zr)) == Rational{Int}
        @test eltype(idio_skewness(zr)) == typeof(sqrt(1 // 1))
        @test PortfolioOptimisers.idio_nan_mean([1 // 2, 1 // 3]) === 5 // 12
    end

    @testset "the refusals" begin
        @test_throws PortfolioOptimisers.IsEmptyError standardised_idio_returns(zeros(0, 0),
                                                                                zeros(0, 0))
        @test_throws DimensionMismatch standardised_idio_returns(eps, vs[:, 1:4])
        @test_throws PortfolioOptimisers.IsEmptyError idio_vol_ic(zeros(0, 0), zeros(0, 0))
        @test_throws DimensionMismatch idio_vol_ic(eps, vs[:, 1:4])
        # The two dependence series read two observations at a time, so one is not enough.
        @test_throws DimensionMismatch idio_vol_ic(eps[1:1, :], vs[1:1, :])
        @test_throws DimensionMismatch idio_vol_residual_dependence(eps[1:1, :], vs[1:1, :])
        # A block that carries no cross-sectional fit, and one that carries no variance
        # history, each name the field the caller must populate.
        no_csr = CrossSectionalFactorModel(; M = randn(rng, N, 3), b = zeros(N), vs = vs,
                                           lag = 1)
        no_vs = CrossSectionalFactorModel(; M = randn(rng, N, 3), b = zeros(N), csr = csr,
                                          lag = 1)
        for f in
            (idio_calibration, idio_tail_rate, idio_kurtosis, idio_skewness, idio_vol_ic,
             idio_vol_residual_dependence, idio_calibration_summary)
            @test_throws PortfolioOptimisers.IsNothingError f(no_csr)
            @test_throws PortfolioOptimisers.IsNothingError f(no_vs)
        end
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.idio_diagnostic_data(nothing,
                                                                                                 nothing)
    end
end

#=
The summary of the cross-sectional diagnostics, decided by #709 and built by #801.

The two testsets `the stored oracle` compare with literals that the oracle gave on the very
arrays this file builds. Rebuild them by an export of the same fixtures and a run of the
oracle's factor model block on them. Each comparison is cell by cell (`parity_compare`).

The first fixture is the one `test_08r_cs_regression_diagnostics.jl` uses, whose design is
well conditioned on every observation but one, so its Gram columns carry values. The second is the one
`test_08r_cs_regression_diagnostics.jl` uses, whose third factor is a constant intercept, so it
drives the patch that reads a constant exposure as perfectly stable.
=#
using Statistics

@testset "Factor model summary" begin
    # Cell by cell: the pattern of the absent answers agrees exactly, and each finite cell
    # lies within `rtol` of its own size. Against the oracle, every column of both fixtures
    # measures maxrel 4.3e-16 or less, and the rates and the coverage are bit-equal. The
    # tolerance `1e-14` is about twenty times the worst value: the
    # round-off of a mean and a deviation that sum in another order.
    function summary_agrees(a, b, label = ""; rtol = 1e-14)
        return parity_compare(a, b; rtol = rtol, name = label).ok
    end

    # The first fixture. Every mutation drives one branch of the regression group, so a
    # change to any of them invalidates the literals of `the stored oracle`.
    rngA = StableRNG(987654321)
    TA, NA, KA = 8, 6, 3
    MsA = randn(rngA, TA, NA, KA)
    fA = 0.02 * randn(rngA, TA, KA)
    epsA = 0.01 * randn(rngA, TA, NA)
    rwA = abs.(randn(rngA, TA, NA)) .+ 0.1
    bwA = fill(1 / NA, TA, NA)
    rwA[3, 2] = 0.0
    rwA[6, 5] = 0.0
    rwA[8, 3] = 0.0
    rwA[8, 4] = 0.0
    rwA[8, 5] = 0.0
    rwA[8, 6] = 0.0
    epsA[4, 3] = NaN
    MsA[2, 4, 1] = NaN
    # The factor return of one observation is absent, which costs that observation its
    # t-statistic and costs the whole series its autocorrelation.
    fA[5, 2] = NaN
    MsA[6, :, 3] = MsA[6, :, 1]

    csrA = CrossSectionalRegression(; f = fA, eps = epsA, n = fill(NA, TA))
    csfmA = CrossSectionalFactorModel(; M = MsA[TA, :, :], b = zeros(NA), csr = csrA,
                                      Ms = MsA, rw = rwA, bw = bwA, lag = 1)

    @testset "the stored oracle, the well-conditioned fixture" begin
        ref_ann_return = [1.7688020592276463, -4.704611043276863, 0.06102501991895286]
        ref_ann_vol = [0.2500971159903252, 0.30449966488269103, 0.2801176875443828]
        ref_sharpe = [7.07246084075624, -15.450299576156583, 0.21785493252468693]
        ref_autocorr = [-0.5425646747475837, NaN, -0.39620089475439907]
        # Observation 6 is collinear, and it identifies the second coefficient alone
        # (#1421). The oracle's three columns that read the t-statistics and the factors
        # count its pseudo-inverse answers there. Ours drop the t-statistics of the pair
        # (`0.6639470271883212` and `7.300135695412292` in the oracle), scale the second
        # (`14.363795117368108`) by `sqrt(4 / 3)` for a variance over `n - 2`, and give the
        # pair an infinite factor. `test_08r_cs_regression_diagnostics.jl` pins each row.
        ref_mean_abs_t = [(5 * 1.3761850065112538 - 0.6639470271883212) / 4,
                          4.764397545417589 + 14.363795117368108 * (sqrt(4 / 3) - 1) / 5,
                          (5 * 3.9180561817454196 - 7.300135695412292) / 4]
        ref_t_rate = [2 / 4, 3 / 5, 1 / 4]
        ref_mean_vif = [Inf, 1.8428351056716823, Inf]
        ref_stability = [-0.4257338194977316, 0.11924492275222029, 0.1814524979202049]
        ref_coverage = [0.9791666666666667, 1.0, 1.0]

        fs = factor_model_summary(csfmA; ppy = 252, step = 3)
        @test summary_agrees(fs.ann_return, ref_ann_return)
        @test summary_agrees(fs.ann_volatility, ref_ann_vol)
        @test summary_agrees(fs.sharpe, ref_sharpe)
        @test summary_agrees(fs.autocorr, ref_autocorr)
        @test summary_agrees(fs.mean_abs_t, ref_mean_abs_t)
        @test summary_agrees(fs.t_rate, ref_t_rate)
        @test summary_agrees(fs.mean_vif, ref_mean_vif)
        @test summary_agrees(fs.stability, ref_stability)
        @test summary_agrees(fs.coverage, ref_coverage)
        @test fs.ppy == 252

        # `ppy = 1` reports the per-period numbers, which is the #708 default.
        ref_ann_return_1 = [0.007019055790585898, -0.018669091441574852,
                            0.00024216277745616214]
        ref_ann_vol_1 = [0.015754637441068472, 0.0191816758948121, 0.01764575569221267]
        ref_sharpe_1 = [0.4455231557591381, -0.973277389522785, 0.013723570794025676]
        fs1 = factor_model_summary(csfmA; ppy = 1, step = 3)
        @test summary_agrees(fs1.ann_return, ref_ann_return_1)
        @test summary_agrees(fs1.ann_volatility, ref_ann_vol_1)
        @test summary_agrees(fs1.sharpe, ref_sharpe_1)
        # Only the first three columns carry the annualisation.
        @test summary_agrees(fs1.autocorr, ref_autocorr)
        @test summary_agrees(fs1.mean_abs_t, ref_mean_abs_t)
        @test summary_agrees(fs1.stability, ref_stability)
        @test summary_agrees(fs1.coverage, ref_coverage)

        # The stability reads the weight history the caller names, and the coverage does
        # not: the oracle wires its universe to the regression weights alone.
        ref_stability_id = [-0.4257338194977314, 0.11924492275222023, 0.18145249792020487]
        ref_stability_rw = [-0.25229769514274375, 0.0032308107079838436,
                            0.26713295230303086]
        fs_id = factor_model_summary(csfmA; ppy = 252, step = 3,
                                     weighting = IdentityMetric())
        fs_rw = factor_model_summary(csfmA; ppy = 252, step = 3,
                                     weighting = RegressionWeightMetric())
        @test summary_agrees(fs_id.stability, ref_stability_id)
        @test summary_agrees(fs_rw.stability, ref_stability_rw)
        @test summary_agrees(fs_id.coverage, ref_coverage)
        @test summary_agrees(fs_rw.coverage, ref_coverage)

        # A history with no more observations than `step` has no stability series at all.
        fs21 = factor_model_summary(csfmA; ppy = 252, step = 21)
        @test all(isnan, fs21.stability)
        @test summary_agrees(fs21.coverage, ref_coverage)

        # The threshold steers the exceedance rate alone.
        # The oracle's `[2 / 5, 4 / 5, 3 / 5]`, less observation 6 on the pair.
        ref_t_rate_1 = [2 / 4, 4 / 5, 2 / 4]
        fst = factor_model_summary(csfmA; ppy = 252, step = 3, threshold = 1)
        @test summary_agrees(fst.t_rate, ref_t_rate_1)
        @test summary_agrees(fst.mean_abs_t, ref_mean_abs_t)
    end

    # The second fixture. Its third factor is a constant intercept and its fourth is finite
    # on two assets alone.
    rngB = StableRNG(24680)
    TB, NB, KB = 10, 6, 4
    MsB = randn(rngB, TB, NB, KB)
    fB = 0.02 * randn(rngB, TB, KB)
    epsB = 0.01 * randn(rngB, TB, NB)
    rwB = abs.(randn(rngB, TB, NB)) .+ 0.1
    bwB = fill(1 / NB, TB, NB)
    vsB = abs.(randn(rngB, TB, NB)) .+ 0.5
    MsB[:, :, 3] .= 1.0
    MsB[:, 3:6, 4] .= NaN
    rwB[3, 2] = 0.0
    bwB[7, 4] = 0.0
    MsB[2, 4, 1] = NaN
    MsB[5, 1, 2] = NaN
    MsB[5, 2, 2] = NaN
    MsB[5, 3, 2] = NaN
    MsB[5, 4, 2] = NaN

    csrB = CrossSectionalRegression(; f = fB, eps = epsB, n = fill(NB, TB))
    csfmB = CrossSectionalFactorModel(; M = MsB[TB, :, :], b = zeros(NB), csr = csrB,
                                      Ms = MsB, vs = vsB, rw = rwB, bw = bwB, lag = 1)

    @testset "the stored oracle, the constant-exposure fixture" begin
        ref_ann_return = [-0.4907810529445153, 1.7641281876800439, 3.287865325954306,
                          2.112811248657294]
        ref_ann_vol = [0.33405767428380584, 0.3481792051822995, 0.27711562516097665,
                       0.3038473320095795]
        ref_sharpe = [-1.4691506608752887, 5.06672472514946, 11.86459740061349,
                      6.953529045931142]
        ref_autocorr = [-0.1731707888686296, -0.11429924268878187, -0.16733266080439935,
                        -0.009766919079024838]
        ref_t_rate = [0.0, 0.0, 0.0, 0.0]
        ref_stability = [-0.15454019826790003, -0.041871561766556646, 1.0, NaN]
        ref_coverage = [0.9833333333333334, 0.9333333333333332, 1.0, 0.32]

        fs = factor_model_summary(csfmB; ppy = 252, step = 3)
        @test summary_agrees(fs.ann_return, ref_ann_return)
        @test summary_agrees(fs.ann_volatility, ref_ann_vol)
        @test summary_agrees(fs.sharpe, ref_sharpe)
        @test summary_agrees(fs.autocorr, ref_autocorr)
        @test summary_agrees(fs.t_rate, ref_t_rate)
        @test summary_agrees(fs.stability, ref_stability)
        @test summary_agrees(fs.coverage, ref_coverage)
        # No cross-section of this fixture carries enough eligible assets for a Gram
        # answer, so the two Gram averages are absent at every factor.
        @test all(isnan, fs.mean_abs_t)
        @test all(isnan, fs.mean_vif)

        # A constant exposure keeps its `1` when the history is too short for a series.
        ref_stability_21 = [NaN, NaN, 1.0, NaN]
        fs21 = factor_model_summary(csfmB; ppy = 252, step = 21)
        @test summary_agrees(fs21.stability, ref_stability_21)
    end

    @testset "the factor return columns are a performance summary" begin
        # The first and the third series carry no absent observation, so the summary's
        # first three columns are `performance_summary` on the series itself.
        fs = factor_model_summary(csfmA; ppy = 252, step = 3)
        for k in (1, 3)
            ps = performance_summary(fA[:, k]; ppy = 252)
            @test isapprox(fs.ann_return[k], ps.ann_return)
            @test isapprox(fs.ann_volatility[k], ps.ann_volatility)
            @test isapprox(fs.sharpe[k], ps.sharpe)
        end
        # The second series carries an absent observation. The summary drops it and
        # `performance_summary` does not, so only the summary answers there.
        ps2 = performance_summary(fA[:, 2]; ppy = 252)
        @test isnan(ps2.ann_return)
        @test isfinite(fs.ann_return[2])
    end

    @testset "the Gram columns are the means of the level-2 series" begin
        fs = factor_model_summary(csfmA; ppy = 252, step = 3)
        t = cs_regression_t_stats(csfmA).X
        vif = exposure_vif(csfmA).X
        rate = cs_regression_t_stat_exceedance_rate(csfmA; threshold = 2).X
        for k in 1:KA
            at = filter(!isnan, abs.(t[:, k]))
            av = filter(!isnan, vif[:, k])
            @test isapprox(fs.mean_abs_t[k], Statistics.mean(at))
            @test isapprox(fs.mean_vif[k], Statistics.mean(av))
        end
        @test fs.t_rate == rate
        # The stability column is the MEDIAN of the series, and not its mean.
        S = exposure_stability(csfmA; step = 3).X
        for k in 1:KA
            @test isapprox(fs.stability[k], Statistics.median(filter(!isnan, S[:, k])))
        end
        # The standalone verb reads the regression weights by default, as the summary
        # does, so the two give one coverage.
        @test fs.coverage == exposure_coverage(csfmA).X
        @test fs.coverage ==
              exposure_coverage(csfmA; weighting = RegressionWeightMetric()).X
    end

    @testset "a block with no regression weights covers every asset" begin
        # The block was fitted on every asset, so the universe of the coverage is every
        # asset, in the summary and in the standalone verb.
        blk = CrossSectionalFactorModel(; M = MsA[TA, :, :], b = zeros(NA), csr = csrA,
                                        Ms = MsA, bw = bwA, lag = 1)
        fs = factor_model_summary(blk; ppy = 252, step = 3)
        @test fs.coverage == exposure_coverage(blk).X
        @test fs.coverage == exposure_coverage(MsA)
        @test fs.coverage == exposure_coverage(blk; weighting = IdentityMetric()).X
    end

    @testset "a block with no exposure history carries five absent columns" begin
        blk = CrossSectionalFactorModel(; M = MsA[TA, :, :], b = zeros(NA), csr = csrA,
                                        rw = rwA, bw = bwA, lag = 1)
        fs = factor_model_summary(blk; ppy = 252, step = 3)
        @test isnothing(fs.mean_abs_t)
        @test isnothing(fs.t_rate)
        @test isnothing(fs.mean_vif)
        @test isnothing(fs.stability)
        @test isnothing(fs.coverage)
        # The four factor return columns are unchanged by the absence.
        full = factor_model_summary(csfmA; ppy = 252, step = 3)
        @test summary_agrees(fs.ann_return, full.ann_return)
        @test summary_agrees(fs.ann_volatility, full.ann_volatility)
        @test summary_agrees(fs.sharpe, full.sharpe)
        @test summary_agrees(fs.autocorr, full.autocorr)
        # A view keeps an absent column absent, and a block that names no factor is
        # selected by position.
        fv = PortfolioOptimisers.port_opt_view(fs, 1:2)
        @test isnothing(fv.mean_abs_t) && isnothing(fv.nf) && isnothing(fv.fam)
        @test fv.ann_return == fs.ann_return[1:2]
    end

    @testset "a re-based block writes NaN on the factor the basis dropped" begin
        rng5 = StableRNG(19283746)
        Tr, Nr, Kr = 6, 5, 3
        Msr = randn(rng5, Tr, Nr, Kr)
        # A fit under a re-basis estimates its factor returns on the reduced axis.
        fr = 0.02 * randn(rng5, Tr, Kr - 1)
        epsr = 0.01 * randn(rng5, Tr, Nr)
        rwr = abs.(randn(rng5, Tr, Nr)) .+ 0.1
        # One constrained family holds factors 1 and 2, and drops the second of them.
        fcb = FactorFamilyBasis(; fnm = ["industry"], fi = [[1, 2]], di = [2],
                                ratios = reshape(collect(range(0.4, 0.9; length = Tr)), Tr,
                                                 1), K = Kr)
        csr_r = CrossSectionalRegression(; f = fr, eps = epsr, n = fill(Nr, Tr))
        L = PortfolioOptimisers.reduce_loadings(fcb, Msr[Tr, :, :])
        blk = CrossSectionalFactorModel(; M = Msr[Tr, :, :], L = L, b = zeros(Nr),
                                        csr = csr_r, Ms = Msr, rw = rwr, fcb = fcb, lag = 1,
                                        nf = ["value", "size", "momentum"],
                                        fam = ["industry", "industry", "style"])
        fs = factor_model_summary(blk; ppy = 1, step = 2,
                                  weighting = RegressionWeightMetric())
        # The answer is on the raw axis, which is wider than the reduced one.
        @test length(fs.ann_return) == Kr
        # It carries the names and the families of the raw axis, and a view selects every
        # column of the factors it keeps, the dropped member included.
        @test fs.nf == ["value", "size", "momentum"]
        @test fs.fam == ["industry", "industry", "style"]
        fi = PortfolioOptimisers.port_opt_view(fs, LabelGroup("industry"))
        @test fi.ppy == fs.ppy
        for f in fieldnames(FactorSummaryResult)
            f == :ppy && continue
            @test isequal(getfield(fi, f), getfield(fs, f)[1:2])
            @test isequal(getfield(PortfolioOptimisers.port_opt_view(fs, ["momentum"]), f),
                          getfield(fs, f)[3:3])
        end
        @test_throws ArgumentError PortfolioOptimisers.port_opt_view(fs, ["beta"])
        @test length(fs.mean_abs_t) == Kr
        # "size" is the dropped member, so it carries no Gram answer and no return series.
        @test isnan(fs.ann_return[2])
        @test isnan(fs.autocorr[2])
        @test fs.ann_return[[1, 3]] ≈ vec(Statistics.mean(fr; dims = 1))
        @test isnan(fs.mean_abs_t[2])
        @test isnan(fs.t_rate[2])
        @test isnan(fs.mean_vif[2])
        # The retained members carry the reduced answers, in the reduced order.
        t = cs_regression_t_stats(blk).X
        vif = exposure_vif(blk).X
        rate = cs_regression_t_stat_exceedance_rate(blk).X
        @test size(t, 2) == 2
        for (raw, red) in ((1, 1), (3, 2))
            @test isapprox(fs.mean_abs_t[raw],
                           Statistics.mean(filter(!isnan, abs.(t[:, red]))))
            @test isapprox(fs.mean_vif[raw], Statistics.mean(filter(!isnan, vif[:, red])))
            @test fs.t_rate[raw] == rate[red]
        end
        # The exposure columns read the raw axis, so they carry an answer at every factor.
        @test !any(isnothing, (fs.stability, fs.coverage))
        @test length(fs.coverage) == Kr
    end
    @testset "a re-based block that carries fr states the returns of the dropped factor (#1422)" begin
        # The constraint of the family fixes the return of the dropped factor, so a block that
        # carries the raw-axis history states its four return statistics.
        rng6 = StableRNG(56473829)
        Tr, Nr, Kr = 6, 5, 3
        Msr = randn(rng6, Tr, Nr, Kr)
        fred = 0.02 * randn(rng6, Tr, Kr - 1)
        fcb = FactorFamilyBasis(; fnm = ["industry"], fi = [[1, 2]], di = [2],
                                ratios = reshape(collect(range(0.4, 0.9; length = Tr)), Tr,
                                                 1), K = Kr)
        fraw = PortfolioOptimisers.expand_factor_returns(fcb, fred)
        csr_r = CrossSectionalRegression(; f = fred, eps = 0.01 * randn(rng6, Tr, Nr),
                                         n = fill(Nr, Tr))
        kw = (; M = Msr[Tr, :, :],
              L = PortfolioOptimisers.reduce_loadings(fcb, Msr[Tr, :, :]), b = zeros(Nr),
              csr = csr_r, Ms = Msr, rw = abs.(randn(rng6, Tr, Nr)) .+ 0.1, fcb = fcb,
              lag = 1, nf = ["value", "size", "momentum"])
        blk = CrossSectionalFactorModel(; kw..., fr = fraw)
        fs = factor_model_summary(blk; ppy = 1, step = 2,
                                  weighting = RegressionWeightMetric())
        @test fs.ann_return ≈ vec(Statistics.mean(fraw; dims = 1)) rtol = 1e-14
        @test isequal(fs.autocorr, PortfolioOptimisers.factor_summary_autocorrelation(fraw))
        @test all(isfinite, fs.ann_volatility) && all(isfinite, fs.sharpe)
        # The Gram columns still read the reduced axis, where the dropped factor has none.
        @test isnan(fs.mean_abs_t[2]) && isnan(fs.t_rate[2]) && isnan(fs.mean_vif[2])
        # The same block without `fr` states no return of the dropped factor, and agrees on
        # every other column.
        fs0 = factor_model_summary(CrossSectionalFactorModel(; kw...); ppy = 1, step = 2,
                                   weighting = RegressionWeightMetric())
        @test isnan(fs0.ann_return[2])
        for cn in fieldnames(typeof(fs))
            cn in (:ann_return, :ann_volatility, :sharpe, :autocorr) && continue
            @test isequal(getfield(fs, cn), getfield(fs0, cn))
        end
    end

    @testset "the refusals" begin
        # A block with no cross-sectional fit carries no factor return history.
        blk = CrossSectionalFactorModel(; M = MsA[TA, :, :], b = zeros(NA), Ms = MsA,
                                        rw = rwA, bw = bwA, lag = 1)
        @test_throws PortfolioOptimisers.IsNothingError factor_model_summary(blk)
        # A non-positive annualisation factor has no meaning.
        @test_throws DomainError factor_model_summary(csfmA; ppy = 0)
        @test_throws DomainError factor_model_summary(csfmA; ppy = -252)
        # A re-based block that names no factor cannot be joined back to the raw axis.
        rng6 = StableRNG(5647382)
        Tn, Nn, Kn = 6, 5, 3
        Msn = randn(rng6, Tn, Nn, Kn)
        fn = 0.02 * randn(rng6, Tn, Kn)
        epsn = 0.01 * randn(rng6, Tn, Nn)
        rwn = abs.(randn(rng6, Tn, Nn)) .+ 0.1
        fcbn = FactorFamilyBasis(; fnm = ["industry"], fi = [[1, 2]], di = [2],
                                 ratios = reshape(collect(range(0.4, 0.9; length = Tn)), Tn,
                                                  1), K = Kn)
        csr_n = CrossSectionalRegression(; f = fn, eps = epsn, n = fill(Nn, Tn))
        Ln = PortfolioOptimisers.reduce_loadings(fcbn, Msn[Tn, :, :])
        blkn = CrossSectionalFactorModel(; M = Msn[Tn, :, :], L = Ln, b = zeros(Nn),
                                         csr = csr_n, Ms = Msn, rw = rwn, fcb = fcbn,
                                         lag = 1)
        @test_throws PortfolioOptimisers.IsNothingError factor_model_summary(blkn; step = 2,
                                                                             weighting = RegressionWeightMetric())
    end

    @testset "the kernels of the summary" begin
        # An empty column has no mean and no median, and neither reduction throws.
        A = [NaN 1.0; NaN 2.0; NaN 4.0]
        @test isnan(PortfolioOptimisers.factor_summary_column_mean(A, 1))
        @test isnan(PortfolioOptimisers.factor_summary_column_median(A, 1))
        @test PortfolioOptimisers.factor_summary_column_mean(A, 2) ≈ 7 / 3
        @test PortfolioOptimisers.factor_summary_column_median(A, 2) == 2.0
        @test PortfolioOptimisers.factor_summary_finite_column(A, 1) == Float64[]
        # A ratio has no value on a zero, an absent or an overflowing denominator.
        @test isnan(PortfolioOptimisers.factor_summary_ratio(1.0, 0.0))
        @test isnan(PortfolioOptimisers.factor_summary_ratio(1.0, NaN))
        @test isnan(PortfolioOptimisers.factor_summary_ratio(Inf, 1.0))
        @test PortfolioOptimisers.factor_summary_ratio(3.0, 1.5) == 2.0
        # A series of one observation has no autocorrelation.
        @test all(isnan,
                  PortfolioOptimisers.factor_summary_autocorrelation(reshape([1.0], 1, 1)))
        # A series of one observation has no volatility either, and so no Sharpe ratio.
        r, v, s = PortfolioOptimisers.factor_summary_return_stats(reshape([0.5], 1, 1), 4)
        @test r == [2.0]
        @test isnan(v[1])
        @test isnan(s[1])
        # A cross-section with no finite exposure has no variance, and a factor with no
        # finite cross-section anywhere is not constant.
        E = fill(NaN, 2, 3, 1)
        @test isnan(PortfolioOptimisers.factor_summary_exposure_variance(E, 1, 1))
        @test PortfolioOptimisers.factor_summary_constant_exposures(E) == falses(1)
        @test PortfolioOptimisers.factor_summary_constant_exposures(ones(2, 3, 1)) ==
              trues(1)
        # A statistic of the reduced axis maps onto the raw axis, and a dropped factor
        # takes a `NaN`.
        @test summary_agrees(PortfolioOptimisers.factor_summary_mapped([5.0, 6.0],
                                                                       [2, 0, 1]),
                             [6.0, NaN, 5.0])
    end

    @testset "the kernels against their definitions" begin
        # The autocorrelation is the Pearson correlation of the two halves, each centred on
        # its own mean, so `Statistics.cor` of the halves is an independent oracle.
        ac = PortfolioOptimisers.factor_summary_autocorrelation(fA)
        for k in (1, 3)
            @test isapprox(ac[k], Statistics.cor(fA[1:(end - 1), k], fA[2:end, k]))
        end
        # The cross-sectional variance is uncorrected, over the exposures that are not NaN.
        for t in 1:TA, k in 1:KA
            b = filter(!isnan, MsA[t, :, k])
            @test isapprox(PortfolioOptimisers.factor_summary_exposure_variance(MsA, t, k),
                           Statistics.var(b; corrected = false))
        end
        # An infinite entry is present: it stays, and the mean of its column is infinite.
        A = [1.0 0.0; Inf 0.0; NaN 0.0]
        @test PortfolioOptimisers.factor_summary_finite_column(A, 1) == [1.0, Inf]
        @test PortfolioOptimisers.factor_summary_column_mean(A, 1) == Inf
    end

    @testset "a constant series and a constant cross-section" begin
        # The rounded mean of equal values can differ from their common value, which left a
        # volatility of about 1e-17 and a Sharpe ratio of about 1e17, an autocorrelation of
        # +-1, and a variance of about 1e-34. Lengths 3 to 7 and values that do not divide
        # exactly are the cases that exposed it.
        for T in 2:9, c in (0.1, 0.01, 0.3, -0.07, 1e-3, 0.123456789, 2.5)
            r, v, s = PortfolioOptimisers.factor_summary_return_stats(fill(c, T, 1), 252)
            @test iszero(v[1])
            @test isnan(s[1])
            @test isnan(PortfolioOptimisers.factor_summary_autocorrelation(fill(c, T, 1))[1])
        end
        for N in 3:7, c in (0.1, 0.3, 1e8, 0.123456789)
            @test iszero(PortfolioOptimisers.factor_summary_exposure_variance(fill(c, 1, N,
                                                                                   1), 1,
                                                                              1))
        end
        # A constant half alone leaves the coefficient without a value.
        @test isnan(PortfolioOptimisers.factor_summary_autocorrelation(reshape([0.3, 0.3,
                                                                                0.3, 0.5],
                                                                               4, 1))[1])
    end

    @testset "an integer history" begin
        # An integer series holds neither a mean nor a NaN, so it computes in floating point
        # and gives the answers of the same series in Float64.
        fi = [1 2; 3 5; 4 1; 2 2]
        ri, vi, si = PortfolioOptimisers.factor_summary_return_stats(fi, 12)
        rf, vf, sf = PortfolioOptimisers.factor_summary_return_stats(Float64.(fi), 12)
        @test ri == rf && vi == vf && si == sf
        @test eltype(ri) == Float64
        @test PortfolioOptimisers.factor_summary_autocorrelation(fi) ==
              PortfolioOptimisers.factor_summary_autocorrelation(Float64.(fi))
        Mi = reshape([1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], 2, 3, 2)
        @test PortfolioOptimisers.factor_summary_exposure_variance(Mi, 1, 1) ≈ 2 / 9
        @test PortfolioOptimisers.factor_summary_mapped([1, 2], [2, 0]) isa Vector{Float64}
        # A Float32 history stays in Float32.
        r32, v32, s32 = PortfolioOptimisers.factor_summary_return_stats(Float32.(fi), 12)
        @test eltype(r32) == eltype(v32) == eltype(s32) == Float32
        @test eltype(PortfolioOptimisers.factor_summary_autocorrelation(Float32.(fi))) ==
              Float32
    end
end

@testset "A summary of an observed factor states its returns and no regression column (#1367)" begin
    rng = StableRNG(1_367_10)
    T, N = 30, 6
    Ms = randn(rng, T, N, 3)
    f = 0.02 * randn(rng, T, 2)
    fx = 0.01 * randn(rng, T, 1)
    csr = CrossSectionalRegression(; f = f, eps = 0.01 * randn(rng, T, N), n = fill(N, T))
    rw = abs.(randn(rng, T, N)) .+ 0.1
    blk = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr, Ms = Ms,
                                    rw = rw, bw = rw, lag = 1,
                                    nf = ["value", "size", "usd"], fx = fx)
    fs = factor_model_summary(blk; step = 2, weighting = RegressionWeightMetric())
    @test length(fs.ann_return) == 3
    @test fs.ann_return ≈ vec(Statistics.mean(hcat(f, fx); dims = 1))
    @test all(isfinite, fs.ann_volatility)
    @test isnan(fs.mean_abs_t[3]) && isnan(fs.t_rate[3]) && isnan(fs.mean_vif[3])
    @test all(isfinite, fs.mean_abs_t[1:2])
    @test length(fs.coverage) == 3
end

@testset "The idiosyncratic group leaves out the variance of a Leverage-One Pair (#1571)" begin
    # Asset 1 is alone in its level on rows 3 and 4 and has a sibling again from row 5. The
    # variance of the marked rows read only zero residuals, so it is not a forecast, and the
    # first unmarked row must not divide by it, as the forecast fit does not.
    rng = StableRNG(1_571_08)
    T, N = 8, 5
    eps = 0.01 * randn(rng, T, N)
    vs = 1e-4 .* (1 .+ rand(rng, T, N))
    h1 = falses(T, N)
    h1[3:4, 1] .= true
    eps[3:4, 1] .= 0
    vs[3:4, 1] .= [1e-12, 1e-20]
    csr = CrossSectionalRegression(; f = zeros(T, 1), eps = eps, n = fill(N, T), h1 = h1)
    blk = CrossSectionalFactorModel(; M = ones(N, 1), b = zeros(N), csr = csr, vs = vs,
                                    lag = 1)
    e, v = PortfolioOptimisers.idio_diagnostic_data(blk)
    @test isequal(v, PortfolioOptimisers.forecast_idiosyncratic_variances(blk))
    @test all(isnan, e[3:4, 1]) && all(isnan, v[3:4, 1])
    z = standardised_idio_returns(e, v)
    @test all(isnan, z[3:5, 1])
    @test z[6, 1] ≈ eps[6, 1] / sqrt(vs[5, 1])
    em = copy(eps)
    vm = copy(vs)
    em[h1] .= NaN
    vm[h1] .= NaN
    for ahead in (true, false)
        @test isequal(idio_calibration(blk; ahead = ahead),
                      idio_calibration(em, vm; ahead = ahead))
    end
    @test isequal(idio_vol_ic(blk), idio_vol_ic(em, vm))
end

@testset "The regression diagnostics read the rank and the inverse of the weighted design (#1719)" begin
    PO = PortfolioOptimisers
    seq = PO.FLoops.SequentialEx()
    T, N = 3, 40
    Z0 = randn(StableRNG(1), T, N, 3)
    X = randn(StableRNG(3), T, N) ./ 100
    W = ones(T, N)
    noise = randn(StableRNG(2), T, N)
    cre = CrossSectionalLinearRegression(; ex = seq)
    # A Gram matrix squares the condition number of the design, so from about 1e8 its rank
    # test drops a direction that the fit kept. Measured before the fix at `d = 1e-9`: the
    # t-statistics `[NaN, 0.397, NaN]`, the VIF `[Inf, 1.0005, Inf]`. The diagnostics now
    # read the design at the rank of the fit, and stay finite and continuous over the band.
    tref = nothing
    for d in (1e-5, 1e-7, 1e-9, 1e-11)
        Z = copy(Z0)
        Z[:, :, 3] = Z0[:, :, 1] .+ d .* noise
        r = cross_sectional_regression(cre, Z, X, W)
        t = cs_regression_t_stats(Z, r.f, r.eps, W)
        @test all(isfinite, t)
        tref = something(tref, t)
        @test t≈tref rtol=1e-3
        vif = exposure_vif(Z, W)
        @test all(x -> 1 <= x < Inf, vif)
        D = Z[1, :, :] .* sqrt.(W[1, :])
        @test exposure_condition_number(Z, W)[1]≈LinearAlgebra.cond(D)^2 rtol=1e-6
        @test PO.cs_score_regressors(nothing, Z, W) == fill(3, T)
        # The Gram route stays for a caller that holds the Gram history alone. Its precision
        # is that of a matrix whose condition number is the square: measured 4e-6 at 1e-5.
        if d >= 1e-5
            tolg = 10 * LinearAlgebra.cond(D)^2 * eps()
            @test isapprox(cs_regression_t_stats(Z, r.f, r.eps, W; G = cs_gram(Z, W)), t;
                           rtol = tolg)
            @test isapprox(exposure_vif(cs_gram(Z, W)), vif; rtol = tolg)
        elseif d <= 1e-9
            @test any(isnan, cs_regression_t_stats(Z, r.f, r.eps, W; G = cs_gram(Z, W)))
        end
    end
    @test_throws DimensionMismatch cs_regression_t_stats(Z0, zeros(T, 3), zeros(T, N), W;
                                                         G = zeros(T, 2, 2))
    # Under `DependentColumnDrop()` an exactly dependent factor takes a return of zero by the
    # rule. The block reads the mask of the fit: the dropped factor has no t-statistic, and
    # the kept factors have those of the fit on the kept columns. Without the mask the
    # pseudo-inverse of the whole design leaves the group unidentified.
    Z = copy(Z0)
    Z[:, :, 3] = Z0[:, :, 1]
    r = cross_sectional_regression(CrossSectionalLinearRegression(;
                                                                  alg = DependentColumnDrop(),
                                                                  ex = seq), Z, X, W)
    @test r.dc == repeat([false false true], T)
    blk = CrossSectionalFactorModel(; M = Z[T, :, :], b = zeros(N), csr = r, Ms = Z, rw = W,
                                    bw = fill(1 / N, T, N))
    t = cs_regression_t_stats(blk).X
    r2 = cross_sectional_regression(cre, Z[:, :, 1:2], X, W)
    @test t[:, 1:2] == cs_regression_t_stats(Z[:, :, 1:2], r2.f, r2.eps, W)
    @test all(isnan, t[:, 3])
    @test all(isnan, cs_regression_t_stats(Z, r.f, r.eps, W)[:, [1, 3]])
    # The design keeps its own columns for the VIF, which describes the exposures.
    @test all(isinf, exposure_vif(blk).X[:, [1, 3]])
    # A mask that drops nothing, or no mask, leaves the design and the changes alone.
    @test PO.cs_dropped_design(Z, (), falses(T, 3)).B === Z
    @test PO.cs_dropped_design(Z, (), nothing).B === Z
    # The selection composes with the change of an Unseen Member.
    P = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 -0.5 0.0]
    dd = PO.cs_dropped_design(Z, [1 => P, 2 => P], BitMatrix([1 0 0; 0 0 0; 0 0 0]))
    @test only(p for p in dd.P if first(p) == 1) ==
          (1 => P * LinearAlgebra.Diagonal([0, 1, 1]))
    @test only(p for p in dd.P if first(p) == 2) == (2 => P)
    @test all(iszero, dd.B[1, :, 1]) && dd.B[2, :, 1] == Z[2, :, 1]
end
