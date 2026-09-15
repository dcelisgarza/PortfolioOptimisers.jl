#=
The regression group of the cross-sectional diagnostics, decided by #709 and built by #798.

The oracle is the reference implementation. The block of `# the reference oracle` below was
built from the very arrays this file builds, run in the reference implementation's own
environment, and its answers are written out as literals. Rebuild them by generating the
same fixture, exporting it, and running the reference's factor model block on it.
=#
using Statistics

@testset "Cross-sectional regression diagnostics" begin
    # The fixture the reference oracle was measured on. Every mutation below drives one
    # branch, so a change to any of them invalidates the literals in `# the reference
    # oracle`.
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
    # An observation whose design is collinear needs the pseudo-inverse.
    Ms[6, :, 3] = Ms[6, :, 1]

    csr = CrossSectionalRegression(; f = f, eps = eps, n = fill(N, T))
    csfm = CrossSectionalFactorModel(; M = Ms[T, :, :], b = zeros(N), csr = csr, Ms = Ms,
                                     rw = rw, bw = bw, lag = 1)

    @testset "the reference oracle" begin
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

        # A `NaN` compares unequal to itself, so the pattern of the absent answers is
        # asserted separately from the values.
        function agrees(a, b; rtol = 1e-10)
            if isnan.(a) != isnan.(b)
                return false
            end
            m = .!isnan.(a)
            return isapprox(a[m], b[m]; rtol = rtol)
        end

        @test agrees(cs_regression_t_stats(csfm), ref_t)
        @test agrees(exposure_vif(csfm), ref_vif)
        @test agrees(cs_regression_t_stat_exceedance_rate(csfm), ref_rate)
        @test agrees(cs_regression_r2(csfm), ref_r2)
        @test agrees(cs_regression_adjusted_r2(csfm), ref_adj)
        @test agrees(cs_regression_aic(csfm), ref_aic)
        @test agrees(cs_regression_bic(csfm), ref_bic)

        # The condition number agrees everywhere the design has full rank. The reference
        # answers `1.9e17` at the collinear observation where this answers `Inf`: the
        # smallest singular value of a rank-deficient matrix rounds either to zero or to a
        # value near the machine epsilon, which is a property of the decomposition and not
        # of the design. Both answers say that the design is singular.
        ref_cond = [2.9908189507547056, 12.318608475111041, 19.287897328200575,
                    6.588259138577632, 13.607428734322603]
        kappa = exposure_condition_number(csfm)
        @test length(kappa) == 7
        @test isapprox(kappa[1:5], ref_cond; rtol = 1e-10)
        @test kappa[6] > 1e12
        @test isnan(kappa[7])
    end

    @testset "the answers carry the lagged observation axis" begin
        @test size(cs_regression_t_stats(csfm)) == (T - 1, K)
        @test size(exposure_vif(csfm)) == (T - 1, K)
        @test length(exposure_condition_number(csfm)) == T - 1
        @test length(cs_regression_t_stat_exceedance_rate(csfm)) == K
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

        # An exactly duplicated column is singular. The pseudo-inverse answers it rather
        # than raising, which is what the reference implementation does.
        Bs = copy(Bd)
        Bs[:, :, 3] = Bs[:, :, 1]
        @test all(isfinite, exposure_vif(Bs, nothing))

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
        fr = 0.02 * randn(rng5, Tr, Kr)
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
                                        nf = ["value", "size", "momentum"])
        Kred = PortfolioOptimisers.reduced_factor_count(fcb)
        @test Kred == 2
        @test size(cs_regression_t_stats(blk)) == (Tr - 1, Kred)
        @test size(exposure_vif(blk)) == (Tr - 1, Kred)
        @test length(cs_regression_t_stat_exceedance_rate(blk)) == Kred
        @test length(exposure_condition_number(blk)) == Tr - 1
        @test length(cs_regression_r2(blk)) == Tr - 1
        # The factor names of the answer's axis are the reduced ones.
        @test PortfolioOptimisers.cs_diagnostic_factor_names(blk) == ["value", "momentum"]
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
        @test size(cs_regression_t_stats(blk)) == (Tn, Kn)
        @test isapprox(cs_regression_t_stats(blk),
                       cs_regression_t_stats(Msn, csrn.f, csrn.eps); rtol = 1e-12)
    end
end
