#=
Issue #777 builds `OrthogonalUncertaintySet`, the one member of the prior arm of the ucs
triple that #776 opened. It reads the factor model of the optimisation's own prior and
confines both of its sets to the directions the loadings do not span.

The literals below are the reference implementation's own output on one fixed synthetic
case, taken by driving it in a Python environment built for the purpose. They are the
oracle of the port, and they are stored as the two invariant products `L * L'` and `Q * Q'`
rather than as `L` and `Q` themselves: a singular vector and a QR factor each carry an
arbitrary sign, so the maps differ between the two implementations where the subspaces they
span do not.

`The geometry is the orthogonal complement of the weighted span` re-derives the same
subspace in plain Julia by a different route -- a pseudo-inverse projector rather than a
singular value decomposition and an eigendecomposition -- so no testset here compares the
file against itself.
=#
# The synthetic point-in-time Asset Panel of the last testset.
include(joinpath(@__DIR__, "test06c_setup.jl"))
@testset "Orthogonal uncertainty sets" begin
    using PortfolioOptimisers, Test, StableRNGs, Random, Clarabel, Statistics,
          LinearAlgebra, Distributions
    using PortfolioOptimisers: orthogonal_factor_span, orthogonality_weights,
                               orthogonal_scaling

    # The reference implementation's fixed case: six assets, two factors.
    B = [1.3554269999999999 -1.0825450000000001;
         0.145787 0.265851;
         0.36465999999999998 1.1712560000000001;
         -0.82563299999999995 -0.99081699999999995;
         0.18280199999999999 0.93871899999999997;
         -0.040494000000000002 -1.178922]
    D = [0.0047879999999999997, 0.0010809999999999999, 0.0068830000000000002,
         0.0060359999999999997, 0.0057340000000000004, 0.0042620000000000002]
    RW = [1.529793, 4.9889279999999996, 1.4463459999999999, 4.5231320000000004,
          2.6551070000000001, 1.9526269999999999]
    BW = [0.24634600000000001, 0.91622199999999998, 0.79029000000000005,
          0.17810000000000001, 0.98294599999999999, 0.72601099999999996]
    T777 = 12
    N777 = size(B, 1)

    # `rr` and `fpr` travel together on a `LowOrderPrior`, so the factor block comes with a
    # factor-axis prior. Neither the fit nor the sets read `fpr`, and the factor covariance
    # is the one the reference's case used.
    function prior777(Bm, dv; rw = nothing, bw = nothing, esigma = dv)
        K = size(Bm, 2)
        rr = CrossSectionalFactorModel(; M = Bm, b = zeros(size(Bm, 1)), esigma = esigma,
                                       rw = if isnothing(rw)
                                           nothing
                                       else
                                           repeat(transpose(rw), T777, 1)
                                       end, bw = if isnothing(bw)
                                           nothing
                                       else
                                           repeat(transpose(bw), T777, 1)
                                       end)
        sigma = Bm * Matrix(0.01I, K, K) * transpose(Bm) + Diagonal(dv)
        sigma = (sigma + transpose(sigma)) / 2
        fpr = LowOrderPrior(; X = zeros(T777, K), mu = zeros(K),
                            sigma = Matrix(0.01I, K, K))
        return LowOrderPrior(; X = zeros(T777, size(Bm, 1)), mu = collect(dv),
                             sigma = sigma, rr = rr, fpr = fpr)
    end
    pr777 = prior777(B, D; rw = RW, bw = BW)

    @testset "The constructor states its defaults and checks its two numbers" begin
        ue = OrthogonalUncertaintySet()
        @test ue.q == 0.05
        @test isa(ue.method, ChiSqKUncertaintyAlgorithm)
        @test isa(ue.scaling, IdentityScaling)
        @test ue.kappa == 1.0
        @test isa(ue.metric, InverseIdiosyncraticVarianceMetric)
        @test isa(ue, PortfolioOptimisers.AbstractPriorUncertaintySetEstimator)
        @test isa(ue, PortfolioOptimisers.AbstractUncertaintySetEstimator)
        @test_throws DomainError OrthogonalUncertaintySet(; q = 0.0)
        @test_throws DomainError OrthogonalUncertaintySet(; q = 1.0)
        @test_throws DomainError OrthogonalUncertaintySet(; q = -0.1)
        @test_throws DomainError OrthogonalUncertaintySet(; kappa = -1.0)
        @test_throws DomainError OrthogonalUncertaintySet(; kappa = Inf)
        # The two roots are unexported, as every marker family in this directory is.
        @test !isdefined(Main, :AbstractOrthogonalityMetric)
        @test !isdefined(Main, :AbstractOrthogonalScaling)
        for m in (BenchmarkWeightMetric(), RegressionWeightMetric(),
                  InverseIdiosyncraticVarianceMetric(), IdentityMetric())
            @test isa(m, PortfolioOptimisers.AbstractOrthogonalityMetric)
        end
        for s in (IdentityScaling(), IdiosyncraticVarianceScaling())
            @test isa(s, PortfolioOptimisers.AbstractOrthogonalScaling)
        end
    end

    @testset "The port reproduces the reference implementation" begin
        # `L * L'` and `Q * Q'` of the reference's own fit on the case above.
        LLt_inv_idio_identity = [0.11358862792337392 -0.10042660828263195 0.027492390721669847 0.17947653852694018 0.072346980909459874 -0.22890828469670357;
                                 -0.10042660828263195 0.64267960183902639 -0.20473902747961212 0.28777903056481913 -0.17685759609006041 0.25207851647259433;
                                 0.027492390721669847 -0.20473902747961212 0.87426181348351339 0.15512341386912093 -0.1134306090674099 0.17413907500422537;
                                 0.17947653852694018 0.28777903056481913 0.15512341386912093 0.75690094003391728 0.12841529094340137 -0.16858098414934339;
                                 0.072346980909459874 -0.17685759609006041 -0.1134306090674099 0.12841529094340137 0.89510459148278188 0.16739975157171977;
                                 -0.22890828469670357 0.25207851647259433 0.17413907500422537 -0.16858098414934339 0.16739975157171977 0.71746442523738652]
        QQt_inv_idio_identity = [0.89078081350665184 0.055118983751157077 -0.02226162263028908 -0.21559733039239479 -0.070734461486436001 0.2055993693304588;
                                 0.055118983751157077 0.096869151554831584 0.14031574634835692 -0.18407121530671566 0.11076416594493715 -0.13641112574439468;
                                 -0.02226162263028908 0.14031574634835692 0.21537856135745034 -0.25345807626151506 0.17633392561521688 -0.23123991524941082;
                                 -0.21559733039239479 -0.18407121530671566 -0.25345807626151506 0.36407328716308057 -0.19322034276141561 0.22267557181133285;
                                 -0.070734461486436001 0.11076416594493715 0.17633392561521688 -0.19322034276141561 0.14747066285548627 -0.20005811099149259;
                                 0.2055993693304588 -0.13641112574439468 -0.23123991524941082 0.22267557181133285 -0.20005811099149259 0.28542752356249995]
        C_inv_idio_identity = [0.06919537556802477, 0.032878564445547188,
                               0.08296384754819415, 0.077691698398220127,
                               0.075723180070570204, 0.065283994975797857]
        LLt_regression_identity = [0.30080297322610933 0.019244022163526317 0.11550659406979061 0.22410027804297478 0.23575127011582075 -0.30134505617588436;
                                   0.019244022163526317 0.95266548965719799 -0.055090403976061825 0.17701173206839041 -0.077167970508965081 0.066337172925963098;
                                   0.11550659406979061 -0.055090403976061825 0.92334354793508044 0.16614360690862387 -0.11733642590136217 0.11415769609612639;
                                   0.22410027804297478 0.17701173206839041 0.16614360690862387 0.21126523248132045 0.20105625865119336 -0.13057882239368263;
                                   0.23575127011582075 -0.077167970508965081 -0.11733642590136217 0.20105625865119336 0.81377875293262247 0.18925612589354068;
                                   -0.30134505617588436 0.066337172925963098 0.11415769609612639 -0.13057882239368263 0.18925612589354068 0.7981440037676687]
        QQt_regression_identity = [0.80059937842392592 0.011143066805821673 -0.087932697443029728 -0.24518326395031032 -0.1565782179594199 0.25913725168919227;
                                   0.011143066805821673 0.033102516807556874 0.066643337261675442 -0.13883534777379966 0.066020916445751504 -0.061702695642419571;
                                   -0.087932697443029728 0.066643337261675442 0.14945522637024888 -0.25202320804313155 0.15768077625328938 -0.16299062368030323;
                                   -0.24518326395031032 -0.13883534777379966 -0.25202320804313155 0.63171143565722865 -0.23236932567429081 0.18907884624578186;
                                   -0.1565782179594199 0.066020916445751504 0.15768077625328938 -0.23236932567429081 0.17179554162680849 -0.18586985511941134;
                                   0.25913725168919227 -0.061702695642419571 -0.16299062368030323 0.18907884624578186 -0.18586985511941134 0.21333590111423104]
        C_regression_identity = [0.80850677841008056, 0.44770957426660374,
                                 0.83150315339084691, 0.47019755767297416,
                                 0.61370404551577984, 0.71563299323537544]
        LLt_inv_idio_idiovar = [0.00052564878629827332 -0.00017067096926816412 0.00015371721257909223 0.0011334294439515624 0.00038420372223206778 -0.00093205507746635472;
                                -0.00017067096926816412 0.0017333608397734357 -0.00081591269310580313 0.00089844812483163747 -0.00050262617046211518 0.0003480198240005325;
                                0.00015371721257909223 -0.00081591269310580313 0.0056581064537803757 0.0013934599539521619 -0.00097161442700997066 0.0012277298375759171;
                                0.0011334294439515624 0.00089844812483163747 0.0013934599539521619 0.0040830818024392155 0.0010115451512762597 -0.0010947776534619641;
                                0.00038420372223206778 -0.00050262617046211518 -0.00097161442700997066 0.0010115451512762597 0.0049605532722933757 0.00097695094346104917;
                                -0.00093205507746635472 0.0003480198240005325 0.0012277298375759171 -0.0010947776534619641 0.00097695094346104917 0.0030544090896761642]
        # The reference's radius at `q = 0.05` and a rank of four.
        radius_ref = 3.0802157451680481

        for (metric, scaling, LLt, QQt, C) in
            ((InverseIdiosyncraticVarianceMetric(), IdentityScaling(),
              LLt_inv_idio_identity, QQt_inv_idio_identity, C_inv_idio_identity),
             (RegressionWeightMetric(), IdentityScaling(), LLt_regression_identity,
              QQt_regression_identity, C_regression_identity),
             (InverseIdiosyncraticVarianceMetric(), IdiosyncraticVarianceScaling(),
              LLt_inv_idio_idiovar, QQt_inv_idio_identity, C_inv_idio_identity))
            ue = OrthogonalUncertaintySet(; metric = metric, scaling = scaling)
            mu_set, sigma_set = ucs(ue, pr777)
            @test isa(mu_set, NormBallUncertaintySet)
            @test isa(sigma_set, CompactCovarianceUncertaintySet)
            @test isapprox(mu_set.kappa, radius_ref; rtol = 1e-12)
            @test isapprox(mu_set.L * transpose(mu_set.L), LLt; atol = 1e-10)
            @test isapprox(sigma_set.Q * transpose(sigma_set.Q), QQt; atol = 1e-12)
            @test isapprox(collect(sigma_set.C), C; rtol = 1e-12)
            @test mu_set.p == 2
            @test isa(mu_set.class, MuUncertaintySetClass)
            @test size(mu_set.L) == (N777, 4)
            @test size(sigma_set.Q) == (N777, 2)
            # The two single-axis verbs agree with the pair.
            @test isapprox(mu_ucs(ue, pr777).L, mu_set.L)
            @test isapprox(collect(sigma_ucs(ue, pr777).C), collect(sigma_set.C))
        end

        # The identity metric leaves the compact set's metric a vector of ones.
        ident = sigma_ucs(OrthogonalUncertaintySet(; metric = IdentityMetric()), pr777)
        @test isapprox(collect(ident.C), ones(N777))
        # The benchmark metric reads the other history, and it is a third geometry.
        bench = sigma_ucs(OrthogonalUncertaintySet(; metric = BenchmarkWeightMetric()),
                          pr777)
        @test isapprox(collect(bench.C), inv.(sqrt.(BW)); rtol = 1e-12)

        # The centre and the nominal covariance are the prior's own, copied at the fit.
        m777, s777 = ucs(OrthogonalUncertaintySet(), pr777)
        @test isapprox(m777.val, pr777.mu)
        @test isapprox(s777.val, pr777.sigma)
        @test s777.kappa == 1.0
    end

    @testset "The geometry is the orthogonal complement of the weighted span" begin
        # An independent derivation: the projector onto the weighted span through a
        # pseudo-inverse rather than through a singular value decomposition, and the
        # complement read as a null space rather than as an eigendecomposition.
        for (metric, w) in ((InverseIdiosyncraticVarianceMetric(), inv.(D)),
                            (RegressionWeightMetric(), RW), (BenchmarkWeightMetric(), BW),
                            (IdentityMetric(), ones(N777)))
            ue = OrthogonalUncertaintySet(; metric = metric)
            mu_set, sigma_set = ucs(ue, pr777)
            W = Diagonal(w)
            # The mean set lives in the `W`-orthogonal complement of the loadings.
            @test isapprox(transpose(mu_set.L) * W * B, zeros(4, 2); atol = 1e-10)
            # The compact set's basis spans the weighted loadings exactly, so its
            # projector is the pseudo-inverse projector of `W^{1/2} B`.
            Bw = sqrt.(w) .* B
            @test isapprox(sigma_set.Q * transpose(sigma_set.Q), Bw * pinv(Bw);
                           atol = 1e-10)
            # `C` is the inverse square root of the metric, entry by entry.
            @test isapprox(collect(sigma_set.C), inv.(sqrt.(w)); rtol = 1e-12)
            # The rank of the mean set is the whole complement, and the two ranks add up.
            @test size(mu_set.L, 2) + size(sigma_set.Q, 2) == N777
        end
        # The weights themselves are the latest row of the history, not an average.
        rr777 = pr777.rr
        @test isapprox(collect(orthogonality_weights(RegressionWeightMetric(), rr777)), RW)
        @test isapprox(collect(orthogonality_weights(BenchmarkWeightMetric(), rr777)), BW)
        @test isapprox(orthogonality_weights(InverseIdiosyncraticVarianceMetric(), rr777),
                       inv.(D))
        @test isnothing(orthogonality_weights(IdentityMetric(), rr777))
    end

    @testset "The rank of the complement is counted, not cut at a tolerance" begin
        # The two ranks add to the number of assets by construction, because the projector
        # `I - Q * Q'` has rank `N - size(Q, 2)` and the metric scaling is invertible. The
        # fit counts that rank rather than cutting the eigenvalues of `A' * A` at a
        # tolerance: on the case above, the largest eigenvalue a tolerance must cut clears
        # `max(N * eps, N * maximum(abs, ev) * eps)` by a factor of three under the identity
        # metric and by a factor of four under the regression one, so a cut states a
        # complement one dimension too wide when the reduction order of the machine moves
        # the eigenvalue by that much.
        #
        # A second column that repeats the first to the last bit leaves the loadings of
        # numerical rank one, so the span is one column and the complement is `N - 1`. The
        # identity must hold there as well.
        Bd = hcat(view(B, :, 1), view(B, :, 1) .* (1 + 1e-14))
        prd = prior777(Bd, D; rw = RW, bw = BW)
        for (pr, w_span) in ((pr777, 2), (prd, 1))
            for metric in (InverseIdiosyncraticVarianceMetric(), RegressionWeightMetric(),
                           BenchmarkWeightMetric(), IdentityMetric())
                mu_set, sigma_set = ucs(OrthogonalUncertaintySet(; metric = metric), pr)
                @test size(sigma_set.Q, 2) == w_span
                @test size(mu_set.L, 2) == N777 - w_span
                @test size(mu_set.L, 2) + size(sigma_set.Q, 2) == N777
                # The map is orthonormal up to the scaling, so it spans a subspace of that
                # rank and does not carry a dependent direction.
                @test LinearAlgebra.rank(mu_set.L) == N777 - w_span
            end
        end
    end

    @testset "A factor-aligned portfolio pays nothing on either axis" begin
        for (metric, w) in ((InverseIdiosyncraticVarianceMetric(), inv.(D)),
                            (RegressionWeightMetric(), RW), (IdentityMetric(), ones(N777)))
            ue = OrthogonalUncertaintySet(; metric = metric)
            mu_set, sigma_set = ucs(ue, pr777)
            # `W B e_1` is inside the span the metric spares.
            wp = Diagonal(w) * view(B, :, 1)
            wp ./= norm(wp)
            @test isapprox(norm(transpose(mu_set.L) * wp), 0.0; atol = 1e-10)
            exposure = collect(sigma_set.C) .* wp
            residual = exposure - sigma_set.Q * (transpose(sigma_set.Q) * exposure)
            @test isapprox(norm(residual), 0.0; atol = 1e-10)
        end
    end

    @testset "A residual portfolio pays a positive compact penalty" begin
        ue = OrthogonalUncertaintySet()
        mu_set, sigma_set = ucs(ue, pr777)
        # A portfolio built out of the orthogonal directions themselves.
        wr = mu_set.L * ones(size(mu_set.L, 2))
        wr ./= norm(wr)
        exposure = collect(sigma_set.C) .* wr
        residual = exposure - sigma_set.Q * (transpose(sigma_set.Q) * exposure)
        @test sum(abs2, residual) > 1e-8
        # And it pays on the mean axis too, which is the same subspace.
        @test norm(transpose(mu_set.L) * wr) > 1e-8
    end

    @testset "The metric and the scaling each change the geometry" begin
        base = mu_ucs(OrthogonalUncertaintySet(), pr777)
        reg = mu_ucs(OrthogonalUncertaintySet(; metric = RegressionWeightMetric()), pr777)
        idv = mu_ucs(OrthogonalUncertaintySet(; scaling = IdiosyncraticVarianceScaling()),
                     pr777)
        @test !isapprox(base.L * transpose(base.L), reg.L * transpose(reg.L))
        @test !isapprox(base.L * transpose(base.L), idv.L * transpose(idv.L))
        # The scaling moves the shape and not the subspace: both maps span the same space.
        @test isapprox(transpose(idv.L) * Diagonal(inv.(D)) * B, zeros(4, 2); atol = 1e-10)
        # `Λ` on the identity scaling is the identity of the rank.
        G = Matrix(qr(base.L).Q)[:, 1:size(base.L, 2)]
        @test isapprox(orthogonal_scaling(IdentityScaling(), G, pr777.rr),
                       Matrix(1.0I, 4, 4))
        # And on the idiosyncratic scaling it is the projected idiosyncratic covariance.
        @test isapprox(orthogonal_scaling(IdiosyncraticVarianceScaling(), G, pr777.rr),
                       transpose(G) * Diagonal(D) * G)
    end

    @testset "A full-rank factor model leaves no orthogonal direction" begin
        rng = StableRNG(777001)
        Bf = randn(rng, 3, 3)
        Df = rand(rng, 3) .* 0.009 .+ 0.001
        prf = prior777(Bf, Df)
        mu_set, sigma_set = ucs(OrthogonalUncertaintySet(), prf)
        @test mu_set.kappa == 0.0
        @test size(mu_set.L) == (3, 1)
        @test all(iszero, mu_set.L)
        @test size(sigma_set.Q) == (3, 3)
        # A rank-zero map leaves the nominal mean, which is what a zero radius means.
        @test isapprox(mu_set.val, prf.mu)
    end

    @testset "The radius is the chi-square quantile at the rank" begin
        for q in (0.5, 0.1, 0.05, 0.01)
            ue = OrthogonalUncertaintySet(; q = q)
            mu_set = mu_ucs(ue, pr777)
            r = size(mu_set.L, 2)
            @test r == 4
            @test isapprox(mu_set.kappa, sqrt(cquantile(Chisq(r), q)); rtol = 1e-12)
        end
        # A smaller `q` is the more demanding one, so the radius rises as `q` falls.
        ks = [mu_ucs(OrthogonalUncertaintySet(; q = q), pr777).kappa
              for q in (0.5, 0.1, 0.05, 0.01)]
        @test issorted(ks)
        # The two algorithms that read no sample, and a stated number, all come free.
        @test isapprox(mu_ucs(OrthogonalUncertaintySet(; q = 0.05,
                                                       method = GeneralKUncertaintyAlgorithm()),
                              pr777).kappa, sqrt(0.95 / 0.05); rtol = 1e-12)
        @test mu_ucs(OrthogonalUncertaintySet(; method = 2.5), pr777).kappa == 2.5
    end

    @testset "kappa scales the compact penalty and zero leaves the nominal variance" begin
        rng = StableRNG(777002)
        wr = randn(rng, N777)
        wr ./= sum(wr)
        function penalty(kappa)
            s = sigma_ucs(OrthogonalUncertaintySet(; kappa = kappa), pr777)
            exposure = collect(s.C) .* wr
            residual = exposure - s.Q * (transpose(s.Q) * exposure)
            return s.kappa * sum(abs2, residual)
        end
        @test isapprox(penalty(3.0), 3.0 * penalty(1.0); rtol = 1e-12)
        @test penalty(0.0) == 0.0
        # A zero radius leaves the worst-case variance at the nominal one.
        s0 = sigma_ucs(OrthogonalUncertaintySet(; kappa = 0.0), pr777)
        @test isapprox(s0.val, pr777.sigma)
        @test s0.kappa == 0.0
    end

    @testset "The refusals" begin
        # A prior with no factor block.
        pr_none = LowOrderPrior(; X = zeros(T777, N777), mu = zeros(N777),
                                sigma = Matrix(1.0I, N777, N777))
        for verb in (ucs, mu_ucs, sigma_ucs)
            err = try
                verb(OrthogonalUncertaintySet(), pr_none)
                nothing
            catch e
                e
            end
            @test isa(err, PortfolioOptimisers.IsNothingError)
            @test occursin("OrthogonalUncertaintySet", err.msg)
            @test occursin("pr.rr", err.msg)
            @test occursin("Got", err.msg)
        end

        # A time-series block carries no cross-sectional weight history.
        fpr2 = LowOrderPrior(; X = zeros(T777, 2), mu = zeros(2),
                             sigma = Matrix(0.01I, 2, 2))
        pr_ts = LowOrderPrior(; X = zeros(T777, N777), mu = zeros(N777),
                              sigma = Matrix(1.0I, N777, N777),
                              rr = Regression(; M = B, b = zeros(N777)), fpr = fpr2)
        for (metric, field) in
            ((BenchmarkWeightMetric(), "bw"), (RegressionWeightMetric(), "rw"))
            err = try
                ucs(OrthogonalUncertaintySet(; metric = metric), pr_ts)
                nothing
            catch e
                e
            end
            @test isa(err, PortfolioOptimisers.IsNothingError)
            @test occursin(field, err.msg)
            @test occursin("Regression", err.msg)
        end
        # The same block carries no `esigma` either, so the default metric refuses.
        @test_throws PortfolioOptimisers.IsNothingError ucs(OrthogonalUncertaintySet(),
                                                            pr_ts)

        # A cross-sectional block whose history is unset names the field it lacks.
        pr_nobw = prior777(B, D; rw = RW)
        err = try
            ucs(OrthogonalUncertaintySet(; metric = BenchmarkWeightMetric()), pr_nobw)
            nothing
        catch e
            e
        end
        @test isa(err, PortfolioOptimisers.IsNothingError)
        @test occursin("bw", err.msg)

        # A weight of zero excluded its asset from the fit, and it leaves the metric
        # singular, so it refuses rather than dividing by zero.
        rw0 = copy(RW)
        rw0[2] = 0.0
        pr_zero = prior777(B, D; rw = rw0)
        @test_throws DomainError ucs(OrthogonalUncertaintySet(;
                                                              metric = RegressionWeightMetric()),
                                     pr_zero)
        # And so does a non-positive idiosyncratic variance.
        d0 = copy(D)
        d0[3] = 0.0
        pr_d0 = prior777(B, D; esigma = d0)
        @test_throws DomainError ucs(OrthogonalUncertaintySet(), pr_d0)

        # The sampled radius has no sample to read, so it refuses by name.
        err = try
            mu_ucs(OrthogonalUncertaintySet(; method = NormalKUncertaintyAlgorithm()),
                   pr777)
            nothing
        catch e
            e
        end
        @test isa(err, ArgumentError)
        @test occursin("NormalKUncertaintyAlgorithm", err.msg)
        @test occursin("ChiSqKUncertaintyAlgorithm", err.msg)
        # The covariance axis reads no radius algorithm, so it answers.
        @test isa(sigma_ucs(OrthogonalUncertaintySet(;
                                                     method = NormalKUncertaintyAlgorithm()),
                            pr777), CompactCovarianceUncertaintySet)
    end

    @testset "The pre-fit, the three-argument form and a Pipeline step reach the same set" begin
        ue = OrthogonalUncertaintySet()
        rd777 = ReturnsResult(; X = zeros(T777, N777), nx = string.("A", 1:N777))
        base = mu_ucs(ue, pr777)

        # The near-optimal head pre-fits from the returns alone, so the estimator travels
        # unchanged and each corner solve fits it in its own builder.
        r = UncertaintySetVariance(; ucs = ue)
        @test PortfolioOptimisers.ucs_risk_measure(r, rd777) === r

        # The three-argument form drops the returns and reads the prior.
        @test isapprox(mu_ucs(ue, rd777, pr777).L, base.L)
        @test isapprox(collect(sigma_ucs(ue, rd777, pr777).C),
                       collect(sigma_ucs(ue, pr777).C))

        # A Pipeline context carrying the prior alone runs, because the returns are never
        # read.
        ctx = PortfolioOptimisers.PipelineContext(; prior = pr777)
        step_mu, _ = PortfolioOptimisers.run_uncertainty_step(ue, :mu, ctx)
        @test isapprox(step_mu.L, base.L)
        step_both, ctx2 = PortfolioOptimisers.run_uncertainty_step(ue, :both, ctx)
        @test isapprox(step_both.mu.L, base.L)
        @test isa(step_both.sigma, CompactCovarianceUncertaintySet)
        @test isapprox(ctx2.uncertainty.mu.L, base.L)

        # An estimator crosses a cluster view unchanged, and the cluster's own prior view
        # refits it.
        @test PortfolioOptimisers.port_opt_view(ue, [1, 2]) === ue
    end

    @testset "A MeanRisk with both sets in its two slots solves" begin
        slv777 = Solver(; name = :clarabel777, solver = Clarabel.Optimizer,
                        check_sol = (; allow_local = true, allow_almost = true),
                        settings = Dict("verbose" => false))
        rng = StableRNG(777003)
        Nm = 8
        Km = 3
        Bm = randn(rng, Nm, Km)
        Dm = rand(rng, Nm) .* 0.009 .+ 0.001
        Fm = randn(rng, 400, Km) .* 0.1
        Xm = Fm * transpose(Bm) + randn(rng, 400, Nm) .* transpose(sqrt.(Dm))
        rdm = ReturnsResult(; X = Xm, nx = string.("A", 1:Nm), F = Fm,
                            nf = string.("F", 1:Km))

        # The time-series route: `FactorPrior` with a residual block fills `esigma`, so the
        # default metric reads it.
        # A minimum-variance objective is flat near its optimum, so the parity pair below
        # runs on a tightened solver: adding the compact block's free variables moves the
        # solver's iterates and it stops elsewhere in the near-optimal set. At the default
        # tolerances two identical programmes land about `1e-4` apart.
        slv_parity = Solver(; name = :clarabel777p, solver = Clarabel.Optimizer,
                            check_sol = (; allow_local = true, allow_almost = true),
                            settings = Dict("verbose" => false, "tol_gap_abs" => 1e-12,
                                            "tol_gap_rel" => 1e-12, "tol_feas" => 1e-12))
        function solve777(pe, ucs_est, r; slv = slv777)
            opt = JuMPOptimiser(; pe = pe, slv = slv, bgt = 1.0,
                                wb = WeightBounds(; lb = 0.0, ub = 1.0),
                                ret = ArithmeticReturn(; ucs = ucs_est))
            return optimise(MeanRisk(; r = r, obj = MinimumRisk(), opt = opt), rdm)
        end
        ue777 = OrthogonalUncertaintySet(; kappa = 4.0)
        res = solve777(FactorPrior(; rsd = true), ue777,
                       UncertaintySetVariance(; ucs = ue777))
        @test isa(res.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test isapprox(sum(res.w), 1.0; rtol = 1e-6)
        @test all(isfinite, res.w)
        # The penalty moves the answer: the nominal variance gives other weights.
        base = solve777(FactorPrior(; rsd = true), nothing, Variance())
        @test isa(base.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test !isapprox(res.w, base.w; atol = 1e-4)
        # A zero radius on the covariance axis reproduces the nominal variance exactly.
        ue0 = OrthogonalUncertaintySet(; kappa = 0.0)
        res0 = solve777(FactorPrior(; rsd = true), nothing,
                        UncertaintySetVariance(; ucs = ue0); slv = slv_parity)
        base_p = solve777(FactorPrior(; rsd = true), nothing, Variance(); slv = slv_parity)
        @test isa(res0.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test isapprox(res0.w, base_p.w; rtol = 1e-5, atol = 1e-6)

        # The cross-sectional route: a hand-built block passed through as a prior result.
        prm = prior(FactorPrior(; rsd = true), rdm)
        csm = CrossSectionalFactorModel(; M = prm.rr.M, b = prm.rr.b,
                                        esigma = collect(PortfolioOptimisers.idiosyncratic_variances(prm.rr)),
                                        rw = repeat(transpose(inv.(collect(PortfolioOptimisers.idiosyncratic_variances(prm.rr)))),
                                                    size(Xm, 1), 1))
        prcs = LowOrderPrior(; X = prm.X, mu = prm.mu, sigma = prm.sigma, rr = csm,
                             fpr = prm.fpr)
        rescs = solve777(prcs, ue777, UncertaintySetVariance(; ucs = ue777))
        @test isa(rescs.retcode, PortfolioOptimisers.OptimisationSuccess)
        @test isapprox(sum(rescs.w), 1.0; rtol = 1e-6)
        # The history above was built as the inverse idiosyncratic variances, so the
        # regression metric and the default read one vector and give one geometry. The two
        # members differ by which field they read, and by nothing else.
        uereg = OrthogonalUncertaintySet(; kappa = 4.0, metric = RegressionWeightMetric())
        @test isapprox(collect(sigma_ucs(uereg, prcs).C), collect(sigma_ucs(ue777, prcs).C);
                       rtol = 1e-10)
    end

    #=
    Issue #928 gives the covariance radius a family of rules, so that the one radius of this
    estimator that was a bare number can be sized from the sample and the span. The mean
    radius was already sized this way, through `method::Num_UcSK` and `k_norm_ball`; this is
    the covariance axis's counterpart, through `kappa::Num_CptRad` and `k_compact`.

    The probes below re-derive each rule's number in plain Julia from the formula in its own
    docstring, rather than storing a literal: both rules are closed forms over quantities the
    prior result and the span already carry, so a literal would only restate the arithmetic.
    =#
    @testset "Issue 928: the covariance radius is sized by a rule" begin
        using PortfolioOptimisers: k_compact, compact_reference_weights, parse_lens
        using Accessors
        # A time-series block built by hand. It records the degrees of freedom of an OLS
        # residual over `K` factors and an intercept, and no divisor, so the rule reads the
        # divisor as the degrees of freedom: the variance is the residual sum of squares
        # over `T - K - 1`. Issue 1334 moved this count from the rule onto the block.
        rng928 = StableRNG(928928928)
        T928, N928, K928 = 260, 7, 3
        F928 = randn(rng928, T928, K928) * 0.01
        B928 = randn(rng928, N928, K928)
        X928 = F928 * transpose(B928) + randn(rng928, T928, N928) * 0.02
        d928 = vec(var(X928 - F928 * transpose(B928); dims = 1))
        rr928 = Regression(; M = B928, b = zeros(N928), esigma = d928,
                           edof = fill(T928 - K928 - 1, N928))
        fpr928 = LowOrderPrior(; X = F928, mu = vec(mean(F928; dims = 1)),
                               sigma = cov(F928))
        pr928 = LowOrderPrior(; X = X928, mu = vec(mean(X928; dims = 1)), sigma = cov(X928),
                              rr = rr928, fpr = fpr928)
        # The relative inflation the rule is built on, stated here from the formula alone.
        rho928(q, dof) = dof / quantile(Chisq(dof), q) - 1

        @testset "A stated number still passes through untouched" begin
            # The whole point of the widening is that it costs the existing caller nothing.
            for k in (0.0, 1.0, 2.5, 100.0)
                @test sigma_ucs(OrthogonalUncertaintySet(; kappa = k), pr928).kappa == k
            end
            # `k_compact` on a number ignores every other argument, which is what makes the
            # resolution free on the stated path.
            @test k_compact(3.25) == 3.25
            @test k_compact(3.25, 0.05, IdentityMetric(), pr928, rr928, ones(N928),
                            zeros(N928, 1), nothing) == 3.25
        end

        @testset "ResidualInflation is the relative inflation under the default metric" begin
            #=
            `W = D^{-1}` leaves a bare projector inside the operator norm, whose norm is 1,
            so the metric-generic formula collapses to `rho` itself. This is the identity the
            rule's docstring claims, and it is why the number is dimensionless there.
            =#
            ue = OrthogonalUncertaintySet(; kappa = ResidualInflation(), q = 0.05)
            k = sigma_ucs(ue, pr928).kappa
            @test isapprox(k, rho928(0.05, T928 - K928 - 1); rtol = 1e-12)
            # A recorded divisor enters the numerator in place of the degrees of freedom:
            # a variance over `T - 1` needs a larger inflation than one over `T - K - 1`.
            rrm = Regression(; M = B928, b = zeros(N928), esigma = d928,
                             edof = fill(T928 - K928 - 1, N928),
                             ediv = fill(T928 - 1, N928))
            prm = LowOrderPrior(; X = X928, mu = vec(mean(X928; dims = 1)),
                                sigma = cov(X928), rr = rrm, fpr = fpr928)
            km = sigma_ucs(ue, prm).kappa
            nu = T928 - K928 - 1
            @test isapprox(km, (T928 - 1) / quantile(Chisq(nu), 0.05) - 1; rtol = 1e-12)
            @test km > k
        end

        @testset "ResidualInflation carries variance units where the metric needs them" begin
            #=
            Under `IdentityMetric` the penalty matrix is dimensionless, so the radius must
            carry the variance itself. The generic formula does that without a method of its
            own: the same operator norm is `lambda_max(P D P)` there.
            =#
            ue = OrthogonalUncertaintySet(; kappa = ResidualInflation(), q = 0.05,
                                          metric = IdentityMetric())
            _, w_sqrt, Q = orthogonal_factor_span(ue, pr928)
            @test isnothing(w_sqrt)
            P = I - Q * transpose(Q)
            expected = rho928(0.05, T928 - K928 - 1) * opnorm(sqrt.(d928) .* P)^2
            @test isapprox(sigma_ucs(ue, pr928).kappa, expected; rtol = 1e-12)
            # The two metrics give two different numbers, which is the whole reason the rule
            # is sized in family rather than through the calibration channel.
            @test !isapprox(sigma_ucs(ue, pr928).kappa,
                            sigma_ucs(OrthogonalUncertaintySet(;
                                                               kappa = ResidualInflation()),
                                      pr928).kappa; rtol = 1e-3)
        end

        @testset "The rule's own q overrides the estimator's, and a tighter q widens it" begin
            base = sigma_ucs(OrthogonalUncertaintySet(; kappa = ResidualInflation(),
                                                      q = 0.05), pr928).kappa
            tight = sigma_ucs(OrthogonalUncertaintySet(;
                                                       kappa = ResidualInflation(;
                                                                                 q = 0.01),
                                                       q = 0.05), pr928).kappa
            @test tight > base
            @test isapprox(tight, rho928(0.01, T928 - K928 - 1); rtol = 1e-12)
            # Stating the owner's own level on the rule is the same number as reading it.
            @test isapprox(sigma_ucs(OrthogonalUncertaintySet(;
                                                              kappa = ResidualInflation(;
                                                                                        q = 0.05),
                                                              q = 0.20), pr928).kappa, base;
                           rtol = 1e-12)
        end

        @testset "A stated dof overrides the block, and a block with no count refuses" begin
            k = sigma_ucs(OrthogonalUncertaintySet(;
                                                   kappa = ResidualInflation(; dof = 120)),
                          pr928).kappa
            @test isapprox(k, rho928(0.05, 120); rtol = 1e-12)
            # The cross-sectional block of `prior777` is built by hand and records no count,
            # so the rule refuses it by name rather than guess at the fit behind it.
            err = try
                sigma_ucs(OrthogonalUncertaintySet(; kappa = ResidualInflation()),
                          prior777(B, D))
                nothing
            catch e
                e
            end
            @test isa(err, PortfolioOptimisers.IsNothingError)
            @test occursin("dof", sprint(showerror, err))
            # A stated count serves it, and one on the block serves it the same way.
            kcs = sigma_ucs(OrthogonalUncertaintySet(;
                                                     kappa = ResidualInflation(; dof = 9.5)),
                            prior777(B, D)).kappa
            @test isapprox(kcs, rho928(0.05, 9.5); rtol = 1e-10)
        end

        @testset "ResidualInflation refuses a block with no idiosyncratic variances" begin
            # The rule reads `D`, and a block whose prior added no residual block carries
            # none. The refusal is `idiosyncratic_variances`'s own, so the message names the
            # field and the fit that left it unset.
            bare = Regression(; M = B928, b = zeros(N928))
            prbare = LowOrderPrior(; X = X928, mu = vec(mean(X928; dims = 1)),
                                   sigma = cov(X928), rr = bare, fpr = fpr928)
            @test_throws PortfolioOptimisers.IsNothingError sigma_ucs(OrthogonalUncertaintySet(;
                                                                                               kappa = ResidualInflation(),
                                                                                               metric = IdentityMetric()),
                                                                      prbare)
        end

        @testset "A fit that left no degrees of freedom refuses, naming the asset" begin
            # Fewer observations than regressors leaves a count `<= 0` on the block, and no
            # chi-squared bound is defined there. A `NaN` count is refused the same way. The
            # message names the asset so the caller can find the short fit.
            for bad in (-1.0, 0.0, NaN)
                edof = fill(float(T928 - K928 - 1), N928)
                edof[3] = bad
                rrs = Regression(; M = B928, b = zeros(N928), esigma = d928, edof = edof)
                short = LowOrderPrior(; X = X928, mu = vec(mean(X928; dims = 1)),
                                      sigma = cov(X928), rr = rrs, fpr = fpr928)
                err = try
                    sigma_ucs(OrthogonalUncertaintySet(; kappa = ResidualInflation()),
                              short)
                    nothing
                catch e
                    e
                end
                @test isa(err, DomainError)
                @test occursin("degrees of freedom", sprint(showerror, err))
                @test occursin("i => 3", sprint(showerror, err))
            end
        end

        @testset "VarianceFraction puts the penalty at exactly f of the nominal variance" begin
            #=
            The rule's whole content is the unit it gives the caller, so the probe measures
            that unit rather than the number: the penalty the reference portfolio pays at
            the returned radius is `f` times the variance it pays at the nominal covariance.
            =#
            for f in (0.05, 0.1, 0.5)
                ue = OrthogonalUncertaintySet(; kappa = VarianceFraction(; f = f))
                s = sigma_ucs(ue, pr928)
                w0 = fill(1 / N928, N928)
                Cw = collect(s.C) .* w0
                penalty = sum(abs2, Cw - s.Q * (transpose(s.Q) * Cw))
                @test isapprox(s.kappa * penalty, f * dot(w0, pr928.sigma, w0);
                               rtol = 1e-12)
            end
            # A stated vector is the reference portfolio itself.
            wb = abs.(randn(StableRNG(5), N928))
            wb ./= sum(wb)
            s = sigma_ucs(OrthogonalUncertaintySet(;
                                                   kappa = VarianceFraction(; f = 0.2,
                                                                            w0 = wb)),
                          pr928)
            Cwb = collect(s.C) .* wb
            @test isapprox(s.kappa * sum(abs2, Cwb - s.Q * (transpose(s.Q) * Cwb)),
                           0.2 * dot(wb, pr928.sigma, wb); rtol = 1e-12)
            @test compact_reference_weights(nothing, 4, nothing, Float64) == fill(0.25, 4)
            @test_throws DimensionMismatch sigma_ucs(OrthogonalUncertaintySet(;
                                                                              kappa = VarianceFraction(;
                                                                                                       w0 = [0.5,
                                                                                                             0.5])),
                                                     pr928)
        end

        @testset "VarianceFraction runs an optimiser, and says so when it cannot" begin
            #=
            `w0` admits any non-finite-allocation optimiser, and each carries its own solver,
            so nothing is threaded into the fit. The returns data reaches the rule through
            the three-argument form the JuMP builders call, which used to discard it.
            =#
            rd928 = ReturnsResult(; X = X928, nx = ["A$(i)" for i in 1:N928])
            ue = OrthogonalUncertaintySet(;
                                          kappa = VarianceFraction(; f = 0.1,
                                                                   w0 = InverseVolatility()))
            s = sigma_ucs(ue, rd928, pr928)
            w0 = optimise(InverseVolatility(), rd928).w
            Cw = collect(s.C) .* w0
            @test isapprox(s.kappa * sum(abs2, Cw - s.Q * (transpose(s.Q) * Cw)),
                           0.1 * dot(w0, pr928.sigma, w0); rtol = 1e-10)
            # `EqualWeighted` reproduces the `nothing` default, which is the same portfolio.
            @test isapprox(sigma_ucs(OrthogonalUncertaintySet(;
                                                              kappa = VarianceFraction(;
                                                                                       w0 = EqualWeighted())),
                                     rd928, pr928).kappa,
                           sigma_ucs(OrthogonalUncertaintySet(; kappa = VarianceFraction()),
                                     pr928).kappa; rtol = 1e-10)
            # The two-argument form carries no returns data, so an optimiser has nothing to
            # run on and the refusal names the field rather than failing inside `optimise`.
            @test_throws PortfolioOptimisers.IsNothingError sigma_ucs(ue, pr928)
        end

        @testset "A span that covers the cross-section leaves an inert set, not a refusal" begin
            #=
            `r == N` makes the projector zero, so the penalty is zero on every portfolio and
            no radius changes the set. `ResidualInflation` reaches zero through the operator
            norm; `VarianceFraction` would divide by zero, so it carries the rank branch. The
            mean axis already returns a zero radius for the same span.
            =#
            Bfull = Matrix(1.0I, N928, N928)
            rrfull = Regression(; M = Bfull, b = zeros(N928), esigma = d928,
                                edof = fill(T928 - K928 - 1, N928))
            fprfull = LowOrderPrior(; X = randn(StableRNG(7), T928, N928), mu = zeros(N928),
                                    sigma = Matrix(0.01I, N928, N928))
            prfull = LowOrderPrior(; X = X928, mu = vec(mean(X928; dims = 1)),
                                   sigma = cov(X928), rr = rrfull, fpr = fprfull)
            _, _, Qf = orthogonal_factor_span(OrthogonalUncertaintySet(), prfull)
            @test size(Qf, 2) == N928
            @test iszero(sigma_ucs(OrthogonalUncertaintySet(; kappa = ResidualInflation()),
                                   prfull).kappa)
            @test iszero(sigma_ucs(OrthogonalUncertaintySet(; kappa = VarianceFraction()),
                                   prfull).kappa)
            @test iszero(mu_ucs(OrthogonalUncertaintySet(), prfull).kappa)
        end

        @testset "A reference portfolio inside the span sends the radius to infinity" begin
            #=
            `C * w0` in the column space of `Q` makes the penalty vanish at `w0` while other
            portfolios still pay it, so no finite radius states a fraction of it and the
            quotient diverges. In exact arithmetic it is not finite and
            `CompactCovarianceUncertaintySet`'s own range check refuses it; in floating point
            the projector leaves a rounding residue instead, so the number is finite and
            enormous. Either way the set is useless, and the probe pins the divergence rather
            than a refusal that only the exact case reaches.
            =#
            ue = OrthogonalUncertaintySet(; kappa = VarianceFraction(),
                                          metric = IdentityMetric())
            _, _, Q = orthogonal_factor_span(ue, pr928)
            generic = sigma_ucs(ue, pr928).kappa
            inside = sigma_ucs(OrthogonalUncertaintySet(;
                                                        kappa = VarianceFraction(;
                                                                                 w0 = Q[:,
                                                                                        1]),
                                                        metric = IdentityMetric()), pr928).kappa
            @test inside > 1e10 * generic
            # An exactly vanishing penalty is the case the set's constructor refuses, and it
            # refuses on the number rather than on the rule that produced it.
            @test_throws DomainError CompactCovarianceUncertaintySet(; kappa = Inf,
                                                                     C = ones(3),
                                                                     Q = zeros(3, 1))
        end

        @testset "The rules refuse a value outside their own range, at construction" begin
            @test_throws DomainError ResidualInflation(; q = 0.0)
            @test_throws DomainError ResidualInflation(; q = 1.0)
            @test_throws DomainError ResidualInflation(; dof = 0)
            @test_throws DomainError ResidualInflation(; dof = Inf)
            @test_throws DomainError VarianceFraction(; f = 0.0)
            @test_throws DomainError VarianceFraction(; f = -1.0)
            @test_throws PortfolioOptimisers.IsEmptyError VarianceFraction(; w0 = Float64[])
            # A bare call constructs, so a caller reads each rule's shape before choosing.
            @test ResidualInflation() isa PortfolioOptimisers.AbstractCompactRadiusAlgorithm
            @test VarianceFraction() isa PortfolioOptimisers.AbstractCompactRadiusAlgorithm
            # The bound admits a number and a rule of this family, and refuses another's.
            # The refusal is the keyword constructor's own type annotation, so it is a
            # `TypeError` raised where the caller wrote the field.
            @test_throws TypeError OrthogonalUncertaintySet(;
                                                            kappa = ScenarioCount(; n = 10))
            # A number still meets the estimator's own range check.
            @test_throws DomainError OrthogonalUncertaintySet(; kappa = -1.0)
            @test_throws DomainError OrthogonalUncertaintySet(; kappa = Inf)
        end

        @testset "The radius is searchable: `ucs.kappa` is a lens over numbers and rules" begin
            #=
            The ticket's other half. Walk-forward selection of the radius needed nothing
            built -- `kappa` is a plain field, so a search grid's `"key.path" => values` pair
            reaches it -- but nothing pinned that, and neither the docstring nor the example
            said so. This is the pin: the lens the grid builds resolves onto the field, and
            the grid may hold rules beside numbers.
            =#
            lens = parse_lens("kappa")
            for v in (0.0, 1.0, 100.0, ResidualInflation(), VarianceFraction(; f = 0.25))
                ue = Accessors.set(OrthogonalUncertaintySet(), lens, v)
                @test ue.kappa === v
                @test sigma_ucs(ue, pr928) isa CompactCovarianceUncertaintySet
            end
            # The path a caller actually writes reaches the field through the measure that
            # holds the set. `parse_lens` builds the same chain from the dotted string.
            nested = parse_lens("ucs.kappa")
            r = UncertaintySetVariance(; ucs = OrthogonalUncertaintySet())
            r2 = Accessors.set(r, nested, ResidualInflation(; q = 0.01))
            @test r2.ucs.kappa == ResidualInflation(; q = 0.01)
            @test isapprox(sigma_ucs(r2.ucs, pr928).kappa, rho928(0.01, T928 - K928 - 1);
                           rtol = 1e-12)
        end

        @testset "A view slices the set and carries the resolved number" begin
            # A set is a Result, so the radius it carries is a number whichever way it was
            # produced. `port_opt_view` has no estimator, no prior and no metric to re-run a
            # rule with, so the number crosses the slice unchanged.
            s = sigma_ucs(OrthogonalUncertaintySet(; kappa = ResidualInflation()), pr928)
            v = PortfolioOptimisers.port_opt_view(s, [1, 2, 3, 4])
            @test v.kappa === s.kappa
            @test length(v.C) == 4
        end

        @testset "ResidualInflation is the smallest radius that covers the inflation" begin
            #=
            With `v = P C w`, the penalty is `kappa |v|^2` and the inflation it covers is
            `|R^{1/2} D^{1/2} W^{1/2} v|^2`, with `R` the diagonal of the inflations. In weight
            space that is `kappa C P C >= Pi R D Pi'` with `Pi = C P C^{-1}`: the radius
            satisfies it, and a radius one part in a million smaller does not. The second
            block records a different count and divisor per asset, as a stepwise fit does.
            =#
            rrh946 = Regression(; M = B928, b = zeros(N928), esigma = d928,
                                edof = collect(range(200.0, 250.0; length = N928)),
                                ediv = collect(range(203.0, 259.0; length = N928)))
            prh946 = LowOrderPrior(; X = X928, mu = vec(mean(X928; dims = 1)),
                                   sigma = cov(X928), rr = rrh946, fpr = fpr928)
            rhoh946 = rrh946.ediv ./ quantile.(Chisq.(rrh946.edof), 0.05) .- 1
            for (pr946, rho946) in
                ((pr928, fill(rho928(0.05, T928 - K928 - 1), N928)), (prh946, rhoh946)),
                metric946 in (PortfolioOptimisers.InverseIdiosyncraticVarianceMetric(),
                              IdentityMetric())

                ue946 = OrthogonalUncertaintySet(; kappa = ResidualInflation(), q = 0.05,
                                                 metric = metric946)
                s946 = sigma_ucs(ue946, pr946)
                C946 = collect(s946.C)
                P946 = I - s946.Q * transpose(s946.Q)
                Pi946 = Diagonal(C946) * P946 * Diagonal(inv.(C946))
                M946(k946) = Symmetric(k946 * Diagonal(C946) * P946 * Diagonal(C946) -
                                       Pi946 * Diagonal(rho946 .* d928) * transpose(Pi946))
                scale946 = s946.kappa * opnorm(Diagonal(C946) * P946 * Diagonal(C946))
                @test eigmin(M946(s946.kappa)) >= -1e-10 * scale946
                @test eigmin(M946(s946.kappa * (1 - 1e-6))) < -1e-8 * scale946
            end
            # Under the default metric one inflation per asset is at most the largest.
            kh946 = sigma_ucs(OrthogonalUncertaintySet(; kappa = ResidualInflation()),
                              prh946).kappa
            @test minimum(rhoh946) <= kh946 <= maximum(rhoh946) * (1 + 1e-12)
        end

        @testset "ResidualInflation's level is 1 - q when the block records the divisor" begin
            #=
            Issue 1334. The bound `d <= (1 + rho) dhat` holds with probability `1 - q` when
            `m dhat / d` follows the chi-squared law at the recorded degrees of freedom, and
            `rho` reads the divisor `m` of the variance. `var` divides by `T - 1`, and an OLS
            fit over `K` factors and an intercept leaves `nu = T - K - 1`. A block that
            records both holds the level. A block that records no divisor makes the rule read
            `m = nu`, and the level falls to `1 - F_nu((T - 1) quantile(Chisq(nu), q) / nu)`,
            which is what every default fit gave before the fix. Both levels are measured
            over 400 fits of 50 Gaussian assets, 20000 draws, whose standard error is about
            0.002.
            =#
            rngl946 = StableRNG(946)
            Tl946, Nl946, Kl946, q946 = 30, 50, 3, 0.05
            nu946 = Tl946 - Kl946 - 1
            dtrue946 = 0.0004 .* (1 .+ rand(rngl946, Nl946))
            hits_div946 = 0
            hits_nodiv946 = 0
            for _ in 1:400
                F946 = randn(rngl946, Tl946, Kl946) * 0.01
                Bl946 = randn(rngl946, Nl946, Kl946)
                Xl946 = F946 * transpose(Bl946) .+
                        randn(rngl946, Tl946, Nl946) .* transpose(sqrt.(dtrue946))
                Z946 = hcat(ones(Tl946), F946)
                E946 = Xl946 - Z946 * (Z946 \ Xl946)
                dhat946 = vec(var(E946; dims = 1))
                fprl946 = LowOrderPrior(; X = F946, mu = vec(mean(F946; dims = 1)),
                                        sigma = cov(F946))
                for ediv946 in (fill(Tl946 - 1, Nl946), nothing)
                    rrl946 = Regression(; M = Bl946, b = zeros(Nl946), esigma = dhat946,
                                        edof = fill(nu946, Nl946), ediv = ediv946)
                    prl946 = LowOrderPrior(; X = Xl946, mu = vec(mean(Xl946; dims = 1)),
                                           sigma = cov(Xl946), rr = rrl946, fpr = fprl946)
                    k946 = sigma_ucs(OrthogonalUncertaintySet(; kappa = ResidualInflation(),
                                                              q = q946), prl946).kappa
                    h946 = count(dtrue946 .<= dhat946 .* (1 + k946))
                    if isnothing(ediv946)
                        hits_nodiv946 += h946
                    else
                        hits_div946 += h946
                    end
                end
            end
            @test isapprox(hits_div946 / (400 * Nl946), 1 - q946; atol = 0.01)
            level946 = 1 -
                       cdf(Chisq(nu946), (Tl946 - 1) * quantile(Chisq(nu946), q946) / nu946)
            @test isapprox(hits_nodiv946 / (400 * Nl946), level946; atol = 0.01)
            @test level946 < 1 - q946 - 0.03
        end
    end

    #=
    Issue 1334. `ResidualInflation` reads the sampling law of each idiosyncratic variance off
    the loadings block, so every fit that writes `esigma` records the degrees of freedom and
    the divisor beside it. These probes pin each source of the count: the variance
    estimators, the two regression estimators, the lift of a time-series prior, the
    cross-sectional prior, and the view, expansion and constructor that carry a block.
    =#
    @testset "Issue 1334: a fit records the sampling law of its idiosyncratic variances" begin
        PO = PortfolioOptimisers
        rng1334 = StableRNG(1334)
        T1334, N1334, K1334 = 60, 5, 3
        X1334 = randn(rng1334, T1334, N1334)

        @testset "variance_count states the count and the divisor of each estimator" begin
            c = PO.variance_count(SimpleVariance(), X1334)
            @test c.n == fill(T1334, N1334)
            @test c.m == fill(T1334 - 1, N1334)
            @test PO.variance_count(SimpleVariance(; corrected = false), X1334).m ==
                  fill(T1334, N1334)
            # The divisor is what the estimator divides the weighted sum of squares by, once
            # that sum is rescaled to the effective count: `m var = n SS_w / sum(w)`.
            for w in (aweights(rand(rng1334, T1334)), pweights(rand(rng1334, T1334)),
                      fweights(rand(rng1334, 1:4, T1334)))
                ve = SimpleVariance(; w = w)
                c = PO.variance_count(ve, X1334)
                mu = sum(w .* X1334; dims = 1) / sum(w)
                ss = vec(sum(w .* (X1334 .- mu) .^ 2; dims = 1)) / sum(w)
                @test c.m .* vec(var(ve, X1334; dims = 1)) ≈ c.n .* ss
                @test c.n ≈
                      fill(isa(w, FrequencyWeights) ? sum(w) : sum(w)^2 / sum(abs2, w),
                           N1334)
            end
            # A column reads its finite rows alone.
            Xg = copy(X1334)
            Xg[1:5, 2] .= NaN
            @test PO.variance_count(SimpleVariance(), Xg).n == [60, 55, 60, 60, 60]
            wg = aweights(rand(rng1334, T1334))
            @test PO.variance_count(SimpleVariance(; w = wg), Xg).n[2] ≈
                  sum(wg[6:end])^2 / sum(abs2, wg[6:end])
            # The exponential weights sum to one, so the count is also the divisor.
            for ve in (ExpWeightedVariance(), RegimeAdjustedExpWeightedVariance())
                lam = ve.decay
                a = lam .^ (0:(T1334 - 1))
                c = PO.variance_count(ve, X1334)
                @test c.n ≈ fill(sum(a)^2 / sum(abs2, a), N1334)
                @test c.m == c.n
            end
            c = PO.variance_count(WindowedVariance(; window = 20), X1334)
            @test c.n == fill(20, N1334)
            @test c.m == fill(19, N1334)
            # The root method states no count, which is not a count of its own.
            @test isnothing(invoke(PO.variance_count,
                                   Tuple{PO.AbstractVarianceEstimator, PO.MatNum},
                                   SimpleVariance(), X1334))
        end

        F1334 = randn(rng1334, T1334, K1334) * 0.01
        Y1334 = F1334 * randn(rng1334, K1334, N1334) .+ 0.01 .* randn(rng1334, T1334, N1334)
        rd1334 = ReturnsResult(; nx = string.(1:N1334), X = Y1334, nf = string.(1:K1334),
                               F = F1334)

        @testset "The regression estimators record the parameters each asset spent" begin
            rr = regression(StepwiseRegression(), Y1334, F1334)
            kept = vec(count(!iszero, rr.M; dims = 2))
            @test rr.edof == T1334 .- kept .- 1
            @test isnothing(rr.ediv)
            rrd = regression(DimensionReductionRegression(), Y1334, F1334)
            @test rrd.edof == fill(T1334 - size(rrd.L, 2) - 1, N1334)
        end

        @testset "The lift restates the count in the count of its variance estimator" begin
            pr = prior(FactorPrior(; rsd = true), rd1334)
            kept = vec(count(!iszero, pr.rr.M; dims = 2))
            @test pr.rr.edof == T1334 .- kept .- 1
            @test pr.rr.ediv == fill(T1334 - 1, N1334)
            # Under weights the spend comes off Kish's count, and the divisor follows.
            w = aweights(range(0.5, 1.5; length = T1334))
            prw = prior(FactorPrior(; rsd = true, ve = SimpleVariance(; w = w)), rd1334)
            keptw = vec(count(!iszero, prw.rr.M; dims = 2))
            n = sum(w)^2 / sum(abs2, w)
            @test prw.rr.edof ≈ n .- keptw .- 1
            @test prw.rr.ediv ≈ fill(n - 1, N1334)
            # With no residual block there is no variance to count, and the regression's
            # own count stays on the block.
            pr0 = prior(FactorPrior(; rsd = false), rd1334)
            @test isnothing(pr0.rr.esigma)
            @test isnothing(pr0.rr.ediv)
            @test pr0.rr.edof == T1334 .- vec(count(!iszero, pr0.rr.M; dims = 2)) .- 1
            # The rule reads one inflation per asset. Under the default metric the radius
            # is the norm of the projector scaled by the square roots of the inflations.
            ue = OrthogonalUncertaintySet(; kappa = ResidualInflation())
            s = sigma_ucs(ue, pr)
            rho = pr.rr.ediv ./ quantile.(Chisq.(pr.rr.edof), 0.05) .- 1
            P = I - s.Q * transpose(s.Q)
            @test isapprox(s.kappa, opnorm(sqrt.(rho) .* P)^2; rtol = 1e-10)
        end

        @testset "The counts combine by case" begin
            cnt = (; n = [4.0, 4.0], m = [3.0, 3.0])
            @test PO.residual_variance_counts(nothing, [1, 2], 5) ==
                  (; edof = nothing, ediv = nothing)
            @test PO.residual_variance_counts(cnt, nothing, 5) ==
                  (; edof = nothing, ediv = [3.0, 3.0])
            @test PO.residual_variance_counts(cnt, [3, 2], 5) ==
                  (; edof = [2.0, 1.0], ediv = [3.0, 3.0])
        end

        @testset "The default fit holds the level through the real prior" begin
            #=
            The issue measured `0.933` through `prior(FactorPrior(; rsd = true), rd)` at
            `T = 30`, `K = 3`. With the counts on the block the per-asset bound holds at
            about `1 - q`. The stepwise search spends more than the factors it keeps, and
            the count charges the kept factors alone, so the level sits a little below.
            =#
            rngc = StableRNG(2)
            Tc, Nc, Kc, qc = 30, 50, 8, 0.05
            dc = 0.0004 .* (1 .+ rand(rngc, Nc))
            h = 0
            for _ in 1:100
                Fc = randn(rngc, Tc, Kc) * 0.01
                Xc = Fc * transpose(randn(rngc, Nc, Kc)) .+
                     randn(rngc, Tc, Nc) .* transpose(sqrt.(dc))
                rdc = ReturnsResult(; nx = string.(1:Nc), X = Xc, nf = string.(1:Kc),
                                    F = Fc)
                prc = prior(FactorPrior(; rsd = true), rdc)
                rho = prc.rr.ediv ./ quantile.(Chisq.(prc.rr.edof), qc) .- 1
                h += count(dc .<= prc.rr.esigma .* (1 .+ rho))
            end
            @test isapprox(h / (100 * Nc), 1 - qc; atol = 0.015)
        end

        @testset "A cross-sectional fit charges each asset its share of the spend" begin
            rdp = synthetic_asset_panel(; n_assets = 40, n_observations = 200,
                                        n_industries = 3, rng = StableRNG(725_001)).rd
            pe = CrossSectionalFactorPrior(;
                                           factors = ["market" => ConstantExposure(),
                                                      "industry" => OneHotExposure(;
                                                                                   field = "industry",
                                                                                   family = "industry"),
                                                      "size" => CompositeExposure(;
                                                                                  descriptors = [LogMarketCap()],
                                                                                  family = "style")])
            prp = prior(pe, rdp)
            csr = prp.rr.csr
            cnt = PO.variance_count(pe.ve, csr.eps)
            p = size(csr.f, 2) + !isnothing(csr.b)
            phi = sum(csr.n .- p) / sum(csr.n)
            @test prp.rr.edof ≈ phi * cnt.n
            @test prp.rr.ediv == cnt.m
            @test 0 < phi < 1
            k = sigma_ucs(OrthogonalUncertaintySet(; kappa = ResidualInflation()), prp).kappa
            @test isfinite(k) && k > 0
        end

        @testset "A view, an expansion and the constructors carry the counts" begin
            rr = Regression(; M = ones(3, 2), esigma = [0.1, 0.2, 0.3], edof = [10, 11, 12],
                            ediv = [13, 14, 15])
            v = PO.port_opt_view(rr, [1, 3])
            @test v.edof == [10, 12]
            @test v.ediv == [13, 15]
            e = PO.expand_regression(rr, BitVector([true, false, true, true]))
            @test e.edof == [10, 0, 11, 12]
            @test e.ediv == [13, 0, 14, 15]
            @test_throws DimensionMismatch Regression(; M = ones(3, 2), edof = [1, 2])
            @test_throws PO.IsEmptyError Regression(; M = ones(3, 2), ediv = Int[])
            csfm = CrossSectionalFactorModel(; M = ones(3, 2), b = zeros(3),
                                             edof = [10.0, 11.0, 12.0], ediv = [1, 2, 3])
            cv = PO.port_opt_view(csfm, [2, 3])
            @test cv.edof == [11.0, 12.0]
            @test cv.ediv == [2, 3]
            @test_throws DimensionMismatch CrossSectionalFactorModel(; M = ones(3, 2),
                                                                     b = zeros(3),
                                                                     ediv = [1, 2])
        end
    end

    #=
    Every testset above hands the estimator a block built by hand over a universe that is
    fully investable. A `CrossSectionalFactorPrior` fitted on a point-in-time Asset Panel is
    not: it writes `NaN` on the loadings of every asset outside its Investable Mask (ADR
    0117), and the optimiser's builders hand this fit the prior reduced to that mask. The
    standalone fit that ADR 0111 offers meets the full prior, so it must refuse the `NaN`
    rows by name rather than hand them to LAPACK.
    =#
    @testset "A point-in-time prior fits on its Investable Mask and answers the full universe" begin
        PO = PortfolioOptimisers
        # This panel and factor set leave assets outside the Investable Mask at the latest
        # observation, which is the case this testset is about; the assertion below guards
        # the fixture.
        rdp = synthetic_asset_panel(; n_assets = 40, n_observations = 200, n_industries = 3,
                                    rng = StableRNG(725_001)).rd
        pe = CrossSectionalFactorPrior(;
                                       factors = ["market" => ConstantExposure(),
                                                  "industry" =>
                                                      OneHotExposure(; field = "industry",
                                                                     family = "industry"),
                                                  "size" => CompositeExposure(;
                                                                              descriptors = [LogMarketCap()],
                                                                              family = "style"),
                                                  "value" => CompositeExposure(;
                                                                               descriptors = [BookToPrice()],
                                                                               family = "style")])
        prp = prior(pe, rdp)
        msk = PO.investable_mask(prp)
        @test !isnothing(msk)
        @test count(msk) < length(msk)
        @test count(i -> !all(isfinite, view(prp.rr.L, i, :)), axes(prp.rr.L, 1)) > 0
        idx = findall(msk)
        N = length(msk)
        ue = OrthogonalUncertaintySet()
        # The standalone fit reduces to the mask, fits, and expands: both sets are over the
        # full universe, with a zero row on every asset outside the mask, and carry the
        # prior's own moments as their centre, `NaN` frame and all.
        mu_f, sg_f = ucs(ue, prp)
        @test size(mu_f.L, 1) == N
        @test size(sg_f.Q, 1) == N
        @test length(sg_f.C) == N
        @test all(iszero, view(mu_f.L, .!msk, :))
        @test all(iszero, view(sg_f.Q, .!msk, :))
        @test all(iszero, view(sg_f.C, .!msk))
        @test all(isfinite, mu_f.L)
        @test all(isfinite, sg_f.Q)
        @test mu_f.val === prp.mu
        @test sg_f.val === prp.sigma
        @test !all(isfinite, mu_f.val)
        @test mu_ucs(ue, prp).L == mu_f.L
        @test sigma_ucs(ue, prp).Q == sg_f.Q
        # The reduced prior is what an optimiser hands the fit, and the expanded set viewed
        # at the mask is that fit: `L` and `C` row for row, the basis as a projector, because
        # the view re-orthonormalises the slice through a pivoted QR that may permute or
        # flip its columns, and the radii untouched.
        prr = PO.port_opt_view(prp, idx)
        mu_r, sg_r = ucs(ue, prr)
        @test size(mu_r.L, 1) == count(msk)
        mu_v = PO.port_opt_view(mu_f, idx)
        sg_v = PO.port_opt_view(sg_f, idx)
        @test mu_v.L == mu_r.L
        @test mu_v.kappa == mu_r.kappa
        @test mu_v.val == mu_r.val
        @test sg_v.C == sg_r.C
        @test sg_v.kappa == sg_r.kappa
        @test sg_v.val == sg_r.val
        @test size(sg_v.Q, 2) == size(sg_r.Q, 2)
        @test isapprox(sg_v.Q * transpose(sg_v.Q), sg_r.Q * transpose(sg_r.Q); atol = 1e-12)
        # A non-finite loading inside the mask is a defect of the prior, and the span still
        # refuses it by name.
        # The block is rebuilt from a fresh fit and poisoned in place: `Accessors.@set`
        # materialises the `L => M` forward and the block's constructor refuses that.
        pr_bad = prior(pe, rdp)
        pr_bad.rr.M[idx[1], 1] = NaN
        err = try
            ucs(ue, pr_bad)
            nothing
        catch e
            e
        end
        @test isa(err, PO.IsNonFiniteError)
        @test occursin("1 of the $(count(msk)) assets", sprint(showerror, err))
        # The pre-built route: a set fitted standalone on the full universe is handed to an
        # optimiser on that universe, which views it at the mask, and reaches the weights the
        # estimator route reaches when it fits the same set on the reduced prior.
        slvp = Solver(; name = :clarabel777q, solver = Clarabel.Optimizer,
                      check_sol = (; allow_local = true, allow_almost = true),
                      settings = Dict("verbose" => false))
        opt_built = JuMPOptimiser(; pe = prp, slv = slvp, bgt = 1.0,
                                  wb = WeightBounds(; lb = 0.0, ub = 1.0),
                                  ret = ArithmeticReturn(; ucs = mu_f))
        res_built = optimise(MeanRisk(; r = UncertaintySetVariance(; ucs = sg_f),
                                      obj = MinimumRisk(), opt = opt_built), rdp)
        opt = JuMPOptimiser(; pe = prp, slv = slvp, bgt = 1.0,
                            wb = WeightBounds(; lb = 0.0, ub = 1.0),
                            ret = ArithmeticReturn(; ucs = ue))
        res = optimise(MeanRisk(; r = UncertaintySetVariance(; ucs = ue),
                                obj = MinimumRisk(), opt = opt), rdp)
        @test isa(res.retcode, PO.OptimisationSuccess)
        @test isa(res_built.retcode, PO.OptimisationSuccess)
        @test isapprox(sum(res.w), 1.0; rtol = 1e-6)
        @test all(i -> msk[i] || abs(res.w[i]) <= 1e-8, eachindex(res.w))
        @test all(i -> msk[i] || abs(res_built.w[i]) <= 1e-8, eachindex(res_built.w))
        @test isapprox(res_built.w, res.w; atol = 1e-6)
        @test length(res.model[:z_cucs1]) == size(sg_r.Q, 2)
    end
end
