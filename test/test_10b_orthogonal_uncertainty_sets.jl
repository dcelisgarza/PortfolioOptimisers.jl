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
end
