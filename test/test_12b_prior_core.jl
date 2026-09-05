include(joinpath(@__DIR__, "test12_setup.jl"))
using LinearAlgebra, FLoops, SparseArrays, StableRNGs

@testset "Empirical Prior" begin
    pes = [EmpiricalPrior(), EmpiricalPrior(; horizon = 252)]
    df = CSV.read(joinpath(@__DIR__, "./assets/EmpiricalPrior.csv.gz"), DataFrame)
    for (i, pe) in enumerate(pes)
        pr = prior(pe, rd)

        mut = reshape(df[1:20, i], size(pr.mu))
        sigmat = reshape(df[21:end, i], size(pr.sigma))

        rtol = 1e-6
        success = isapprox(pr.mu, mut; rtol = rtol)
        if !success
            println("Mu $i fails")
            find_tol(pr.mu, mut)
        end
        @test success

        rtol = 1e-6
        success = isapprox(pr.sigma, sigmat; rtol = rtol)
        if !success
            println("Sigma $i fails")
            find_tol(pr.sigma, sigmat)
        end
        @test success
    end
end

#=
The horizon conversion of `prior(::EmpiricalPrior{<:Any, <:Any, <:Number}, …)` runs in three
lines whose order is load-bearing: the covariance reads the `mu` that still holds
`\hat{\mu}_i + 1`, because the decrement has not run yet. The `# Algorithm` of that method
states it, and these are the numbers behind the statement. Sweep ticket #533.
=#
@testset "Empirical prior horizon conversion" begin
    Xh = rd.X[:, 1:4]
    h = 12.0
    pr = prior(EmpiricalPrior(; horizon = h), Xh)

    # The two closed forms, by hand, off the horizon-scaled log moments.
    Xl = log1p.(Xh)
    mu_log = vec(mean(SimpleExpectedReturns(), Xl))
    sigma_log = cov(PortfolioOptimisersCovariance(), Xl)
    mu_t = h .* mu_log
    sigma_t = h .* sigma_log
    mu_hand = exp.(mu_t .+ 0.5 .* diag(sigma_t)) .- 1
    sigma_hand = [(mu_hand[i] + 1) * (mu_hand[j] + 1) * (exp(sigma_t[i, j]) - 1)
                  for i in 1:4, j in 1:4]
    @test isapprox(pr.mu, mu_hand)
    @test isapprox(pr.sigma, sigma_hand)

    # The order is not free. Reading the decremented `mu` instead drops the covariance by
    # more than three orders of magnitude on this series.
    sigma_wrong = [mu_hand[i] * mu_hand[j] * (exp(sigma_t[i, j]) - 1)
                   for i in 1:4, j in 1:4]
    @test maximum(abs, sigma_wrong) < 1e-3 * maximum(abs, pr.sigma)

    # `X` is the arithmetic matrix the caller passed, not the log-returns.
    @test pr.X == Xh
    @test pr.X != Xl

    # `horizon = 1` is not the `Nothing` method. That one takes the arithmetic moments
    # directly, and this one still makes the log round trip.
    pr1 = prior(EmpiricalPrior(; horizon = 1), Xh)
    pr0 = prior(EmpiricalPrior(), Xh)
    @test isapprox(pr1.mu, exp.(mu_log .+ 0.5 .* diag(sigma_log)) .- 1)
    @test pr1.mu != pr0.mu
    @test pr1.sigma != pr0.sigma
    @test isapprox(pr0.mu, vec(mean(SimpleExpectedReturns(), Xh)))
    @test isapprox(pr0.sigma, cov(PortfolioOptimisersCovariance(), Xh))

    # The `horizon > 0` guard.
    @test_throws DomainError EmpiricalPrior(; horizon = 0)
    @test_throws DomainError EmpiricalPrior(; horizon = -3)
end

@testset "Factor Prior" begin
    df = CSV.read(joinpath(@__DIR__, "./assets/FactorPrior1.csv.gz"), DataFrame)
    pr = prior(FactorPrior(; rsd = false), rd)

    mut = reshape(df[1:20, 1], size(pr.mu))
    sigmat = reshape(df[21:420, 1], size(pr.sigma))
    cholt = reshape(df[421:end, 1], size(pr.chol))

    rtol = 1e-6
    success = isapprox(pr.mu, mut; rtol = rtol)
    if !success
        println("Mu 1 fails")
        find_tol(pr.mu, mut)
    end
    @test success

    rtol = 1e-6
    success = isapprox(pr.sigma, sigmat; rtol = rtol)
    if !success
        println("Sigma 1 fails")
        find_tol(pr.sigma, sigmat)
    end
    @test success

    rtol = 1e-6
    success = isapprox(pr.chol, cholt; rtol = rtol)
    if !success
        println("Chol 1 fails")
        find_tol(pr.chol, cholt)
    end
    @test success

    df = CSV.read(joinpath(@__DIR__, "./assets/FactorPrior2.csv.gz"), DataFrame)
    pr = prior(FactorPrior(; rsd = true), rd)

    mut = reshape(df[1:20, 1], size(pr.mu))
    sigmat = reshape(df[21:420, 1], size(pr.sigma))
    cholt = reshape(df[421:end, 1], size(pr.chol))

    rtol = 1e-6
    success = isapprox(pr.mu, mut; rtol = rtol)
    if !success
        println("Mu 2 fails")
        find_tol(pr.mu, mut)
    end
    @test success

    rtol = 1e-6
    success = isapprox(pr.sigma, sigmat; rtol = rtol)
    if !success
        println("Sigma 2 fails")
        find_tol(pr.sigma, sigmat)
    end
    @test success

    rtol = 1e-6
    success = isapprox(pr.chol, cholt; rtol = rtol)
    if !success
        println("Chol 2 fails")
        find_tol(pr.chol, cholt)
    end
    @test success
end

#=
The identities `factor_lift` states: `chol` factorises the covariance *before* matrix
processing, and the residual block enters `sigma` and `chol` once each. Sweep ticket #533.
=#
@testset "Factor lift identities" begin
    PO = PortfolioOptimisers
    re = StepwiseRegression()
    mp = MatrixProcessing()
    ve = SimpleVariance()
    rr, posterior_X = PO.factor_reconstruction(re, rd.X, rd.F)

    # The reconstruction is the loadings applied to the factors.
    @test isapprox(posterior_X, rd.F * transpose(rr.M) .+ transpose(rr.b))
    @test size(posterior_X) == size(rd.X)

    f_pr = prior(EmpiricalPrior(), rd.F)
    N, K = size(rr.M)
    systematic = rr.M * f_pr.sigma * transpose(rr.M)

    # `rsd = false`: `chol` carries the factor Cholesky factor through the loadings, and it
    # factorises the systematic block exactly.
    l0 = PO.factor_lift(mp, ve, false, rr, f_pr.mu, f_pr.sigma, rd.X, posterior_X)
    @test size(l0.chol) == (K, N)
    @test isapprox(l0.mu, rr.M * f_pr.mu .+ rr.b)
    @test isapprox(l0.sigma, systematic)
    @test isapprox(transpose(l0.chol) * l0.chol, systematic)
    @test isapprox(transpose(l0.chol), rr.M * cholesky(f_pr.sigma).L)

    # `rsd = true`: the residual block enters `sigma` once and `chol` once, and the two agree.
    l1 = PO.factor_lift(mp, ve, true, rr, f_pr.mu, f_pr.sigma, rd.X, posterior_X)
    err_sigma = diagm(vec(var(ve, rd.X - posterior_X; dims = 1)))
    @test size(l1.chol) == (K + N, N)
    @test isapprox(l1.sigma, systematic + err_sigma)
    @test isapprox(transpose(l1.chol) * l1.chol, l1.sigma)
    tail = transpose(l1.chol)[:, (K + 1):(K + N)]
    @test isapprox(tail, sqrt.(err_sigma))
    @test isapprox(tail * transpose(tail), err_sigma)

    # The systematic block alone is singular whenever `N > K`, and the residual block is what
    # makes the sum full rank. That is what the `posdef!` of the `rsd` branch re-conditions.
    @test N > K
    @test minimum(eigvals(Symmetric(l0.sigma))) < 1e-12
    @test minimum(eigvals(Symmetric(l1.sigma))) > 1e-12

    # Matrix processing moves `sigma` and leaves `chol` behind, so the stated identity holds
    # against the unprocessed covariance alone. The docstring of `factor_lift` says so.
    lt = PO.factor_lift(MatrixProcessing(; dt = Detone()), ve, false, rr, f_pr.mu,
                        f_pr.sigma, rd.X, posterior_X)
    @test !isapprox(lt.sigma, systematic)
    @test isapprox(transpose(lt.chol) * lt.chol, systematic)
    @test !isapprox(transpose(lt.chol) * lt.chol, lt.sigma)

    # Every field of the answer of `factor_residual_config` is the estimator's own.
    fp = FactorPrior(; ve = SimpleVariance(; corrected = false), rsd = false)
    cfg = PO.factor_residual_config(fp)
    @test (cfg.ve, cfg.pdm, cfg.rsd) === (fp.ve, fp.mp.pdm, fp.rsd)
    @test isnothing(PO.factor_residual_config(EmpiricalPrior()))

    # Issue #776. The lift returns the residual variances it measured, and they are the
    # diagonal it folded into `sigma`, not a second measurement of it.
    @test l1.esigma == diag(err_sigma)
    @test isapprox(l1.sigma - systematic, diagm(l1.esigma))
    # Under `rsd = false` no residual block was added, so there is nothing to report.
    @test isnothing(l0.esigma)
end

# Issue #776. A prior that lifts the factor moments writes the residual variances onto the
# loadings block, so a consumer reads them off `pr.rr` instead of recomputing them from the
# reconstruction error.
@testset "A time-series factor prior writes esigma onto its loadings block" begin
    PO = PortfolioOptimisers
    pe1 = FactorPrior(; rsd = true)
    pr1 = prior(pe1, rd)
    rr1, posterior_X1 = PO.factor_reconstruction(pe1.re, rd.X, rd.F)
    lift1 = PO.factor_lift(pe1.mp, pe1.ve, true, rr1, prior(pe1.pe, rd.F).mu,
                           prior(pe1.pe, rd.F).sigma, rd.X, posterior_X1)
    # The field holds the variances the lift measured, which are the diagonal it folded in.
    @test pr1.rr.esigma == lift1.esigma
    @test isapprox(pr1.rr.esigma, vec(var(pe1.ve, rd.X - posterior_X1; dims = 1)))
    # The reader answers off the block without a shape test at the call site.
    @test PO.idiosyncratic_variances(pr1.rr) == pr1.rr.esigma
    # Nothing else on the block moved, and `L` is still unset under a stepwise fit.
    @test pr1.rr.M == rr1.M
    @test pr1.rr.b == rr1.b
    @test isnothing(getfield(pr1.rr, :L))

    # Under `rsd = false` the lift adds no residual block, so the field is unset and the
    # reader refuses with a message that names the switch.
    pr0 = prior(FactorPrior(; rsd = false), rd)
    @test isnothing(pr0.rr.esigma)
    @test_throws PO.IsNothingError PO.idiosyncratic_variances(pr0.rr)

    # A dimension-reduction fit sets `L`, and the write leaves it set.
    pr2 = prior(FactorPrior(; re = DimensionReductionRegression(;), rsd = true), rd)
    @test !isnothing(getfield(pr2.rr, :L))
    @test !isnothing(pr2.rr.esigma)

    # `FactorBlackLittermanPrior` applies the same lift, so it writes the field too.
    fbl(rsd) = FactorBlackLittermanPrior(; pe = EmpiricalPrior(;), rsd = rsd, sets = xfsets,
                                         tau = 1 / size(rd.X, 1),
                                         views = LinearConstraintEstimator(;
                                                                           val = ["MTUM == 0.0001",
                                                                                  "QUAL - USMV == -0.0003"]))
    prb = prior(fbl(true), rd)
    @test !isnothing(prb.rr.esigma)
    @test length(prb.rr.esigma) == size(rd.X, 2)
    @test isnothing(prior(fbl(false), rd).rr.esigma)

    # A view of the prior slices the field with the assets, on the one axis it has.
    i = [1, 3, 5]
    @test PO.port_opt_view(pr1, i).rr.esigma == pr1.rr.esigma[i]
end

@testset "High Order Prior" begin
    pr = prior(HighOrderPriorEstimator(), rd)
    @test isapprox(pr.X, rd.X)
    @test isapprox(pr.mu, vec(mean(SimpleExpectedReturns(), rd.X)))
    @test isapprox(pr.sigma, cov(PortfolioOptimisersCovariance(), rd.X))
    @test isapprox(pr.kt, cokurtosis(Cokurtosis(; alg = FullMoment()), rd.X))
    @test all(isapprox.((pr.sk, pr.V), coskewness(Coskewness(; alg = FullMoment()), rd.X)))

    pe = HighOrderPriorEstimator(; kte = Cokurtosis(; alg = SemiMoment()),
                                 ske = Coskewness(; alg = SemiMoment()))
    pr = prior(pe, transpose(rd.X); dims = 2)
    @test isapprox(pr.X, rd.X)
    @test isapprox(pr.mu, vec(mean(SimpleExpectedReturns(), rd.X)))
    @test isapprox(pr.sigma, cov(PortfolioOptimisersCovariance(), rd.X))
    @test isapprox(pr.kt, cokurtosis(Cokurtosis(; alg = SemiMoment()), rd.X))
    @test all(isapprox.((pr.sk, pr.V), coskewness(Coskewness(; alg = SemiMoment()), rd.X)))

    pe1 = FactorPrior(; re = DimensionReductionRegression(;), rsd = true)
    pr1 = prior(pe1, rd)

    pe2 = HighOrderPriorEstimator(; pe = pe1)
    pr2 = prior(pe2, rd)
    @test isa(pe2.me, SimpleExpectedReturns)
    @test isa(pe2.ce, PortfolioOptimisersCovariance)

    @test pr1.X == pr2.X
    @test pr1.mu == pr2.mu
    @test pr1.sigma == pr2.sigma
    @test isapprox(pr2.kt, cokurtosis(Cokurtosis(; alg = FullMoment()), rd.X))
    @test all(isapprox.((pr2.sk, pr2.V),
                        coskewness(Coskewness(; alg = FullMoment()), rd.X)))

    # Error messages name the caller's domain quantity, not the bare field symbol.
    lopr = prior(EmpiricalPrior(), rd)
    kt_err = try
        HighOrderPrior(; pr = lopr, kt = zeros(0, 0), L2 = ones(1, 1), S2 = ones(1, 1))
    catch e
        e
    end
    @test kt_err isa PortfolioOptimisers.IsEmptyError
    @test occursin("cokurtosis", kt_err.msg)
    @test occursin("`kt`", kt_err.msg)

    # The factor low order prior is reachable as `fpr.pr` and as `pr.fpr`, and the two must
    # be the same object. Nesting the co-moments over a prior with no factor block at all.
    no_block_err = try
        HighOrderPrior(; pr = lopr, fpr = HighOrderPrior(; pr = lopr))
    catch e
        e
    end
    @test no_block_err isa PortfolioOptimisers.IsNothingError
    @test occursin("pr.fpr === nothing", no_block_err.msg)
    @test occursin("FactorPrior", no_block_err.msg)

    # Nesting a block whose inner prior is a *different* distribution from `pr.fpr`.
    mismatch_err = try
        HighOrderPrior(; pr = pr1, fpr = HighOrderPrior(; pr = lopr))
    catch e
        e
    end
    @test mismatch_err isa PortfolioOptimisers.ConflictingArgumentError
    @test occursin("`fpr.pr`", mismatch_err.msg)
    @test occursin("`pr.fpr`", mismatch_err.msg)

    # The consistent nesting is accepted, and a low order factor block with no factor
    # co-moments over it stays ordinary.
    @test HighOrderPrior(; pr = pr1, fpr = HighOrderPrior(; pr = pr1.fpr)).fpr.pr ===
          pr1.fpr
    @test isnothing(HighOrderPrior(; pr = pr1).fpr)
end

@testset "High Order Factor Prior" begin
    df = CSV.read(joinpath(@__DIR__, "./assets/HighOrderFactorPrior.csv.gz"), DataFrame)
    pr0 = prior(FactorPrior(), rd)
    pr = prior(HighOrderFactorPriorEstimator(), rd)
    @test isapprox(pr.X, pr0.X)
    @test isapprox(pr.mu, pr0.mu)
    @test isapprox(pr.sigma, pr0.sigma)
    @test isapprox(pr.chol, pr0.chol)
    @test isapprox(pr.f_mu, pr0.f_mu)
    @test isapprox(pr.f_sigma, pr0.f_sigma)
    N2 = length(pr.V)
    N3 = length(pr.sk)
    N4 = length(pr.kt)
    Nf2 = length(pr.f_V)
    Nf3 = length(pr.f_sk)
    Nf4 = length(pr.f_kt)
    @test isapprox(df[1:N4, 1], vec(pr.kt))
    @test isapprox(df[(N4 + 1):(N4 + N3), 1], vec(pr.sk))
    @test isapprox(df[(N4 + N3 + 1):(N4 + N3 + N2), 1], vec(pr.V))
    @test isapprox(df[(N4 + N3 + N2 + 1):(N4 + N3 + N2 + Nf4), 1], vec(pr.f_kt))
    @test isapprox(df[(N4 + N3 + N2 + Nf4 + 1):(N4 + N3 + N2 + Nf4 + Nf3), 1], vec(pr.f_sk))
    @test isapprox(df[(N4 + N3 + N2 + Nf4 + Nf3 + 1):(N4 + N3 + N2 + Nf4 + Nf3 + Nf2), 1],
                   vec(pr.f_V))

    # The flat `f_*` names are virtual reads of the nested block, and the nested block's own
    # prior is the low order carrier's factor block — the enforced invariant, on a real fit.
    @test pr.fpr isa HighOrderPrior
    @test pr.fpr.pr === pr.pr.fpr
    @test pr.f_kt === pr.fpr.kt
    @test pr.f_sk === pr.fpr.sk
    @test pr.f_V === pr.fpr.V
    @test pr.f_D2 === pr.fpr.D2
    @test pr.f_L2 === pr.fpr.L2
    @test pr.f_S2 === pr.fpr.S2
    @test pr.f_skmp === pr.fpr.skmp
    # `fpr` names the high order block, but reading through it still reaches the factor
    # low order moments, because the nested carrier forwards to its own `pr`.
    @test pr.fpr.mu === pr.f_mu
    @test pr.fpr.sigma === pr.f_sigma
    # An asset view forwards the factor block whole, so the invariant survives it.
    v = PortfolioOptimisers.port_opt_view(pr, [1, 3, 5])
    @test v.fpr === pr.fpr
    @test v.fpr.pr === v.pr.fpr
    # No factor block: every flat read is `nothing`, exactly as the fields were.
    hopr = prior(HighOrderPriorEstimator(), rd)
    @test isnothing(hopr.fpr)
    @test all(isnothing,
              (hopr.f_kt, hopr.f_sk, hopr.f_V, hopr.f_D2, hopr.f_L2, hopr.f_S2,
               hopr.f_skmp))
end

@testset "Vanilla and Bayesian Black Litterman" begin
    df = CSV.read(joinpath(@__DIR__, "./assets/BlackLitterman.csv.gz"), DataFrame)
    pes = [BlackLittermanPrior(; sets = sets, tau = 1 / size(rd.X, 1),
                               views = LinearConstraintEstimator(;
                                                                 val = ["AAPL == 0.00002",
                                                                        "BAC == CVX",
                                                                        "WMT == group2",
                                                                        "RRC-group1 == 0.0005"])),
           BayesianBlackLittermanPrior(; pe = FactorPrior(; pe = EmpiricalPrior(;)),
                                       sets = xfsets, tau = 1 / size(rd.X, 1),
                                       views = LinearConstraintEstimator(;
                                                                         val = ["MTUM == 0.0001",
                                                                                "QUAL - USMV == -0.0003"])),
           BlackLittermanPrior(; sets = sets, tau = 1 / size(rd.X, 1),
                               views_conf = [0.05, 0.2, 0.5, 0.9],
                               views = LinearConstraintEstimator(;
                                                                 val = ["AAPL == 0.00002",
                                                                        "BAC == CVX",
                                                                        "WMT == group2",
                                                                        "RRC-group1 == 0.0005"])),
           BlackLittermanPrior(; sets = sets, tau = 1 / size(rd.X, 1), views_conf = 0.05,
                               views = LinearConstraintEstimator(; val = "AAPL == 0.00002"))]
    for (i, pe) in enumerate(pes)
        pr = prior(pe, rd)
        success = isapprox(pr.mu, df[1:20, i]; rtol = 1e-6)
        if !success
            println("Mu $i fails")
            find_tol(pr.mu, df[1:20, i])
        end
        @test success

        success = isapprox(vec(pr.sigma), df[21:420, i]; rtol = 1e-6)
        if !success
            println("Sigma $i fails")
            find_tol(vec(pr.sigma), df[21:420, i])
        end
        @test success
    end

    pr = prior(BlackLittermanPrior(; sets = sets, tau = 1 / size(rd.X, 1),
                                   views = black_litterman_views(LinearConstraintEstimator(;
                                                                                           val = ["AAPL == 0.00002",
                                                                                                  "BAC == CVX",
                                                                                                  "WMT == group2",
                                                                                                  "RRC-group1 == 0.0005"]),
                                                                 sets)), rd)
    @test isapprox(df[!, 1], [pr.mu; vec(pr.sigma)], rtol = 1e-6)
end

@testset "Bayesian Black Litterman reads the declared factor axis" begin
    # The golden test above is the bit-identity proof: entry 2 of `pes` runs on `xfsets`,
    # whose factors live under `tfkey`, and still matches `BlackLitterman.csv.gz` to the same
    # tolerance it matched under the pre-migration shape. Only the *lookup* changed.
    f_views = LinearConstraintEstimator(; val = ["MTUM == 0.0001"])
    # The pre-migration shape — factor names under `xkey`, no factor axis at all. The naive
    # migration failure would be an *asset*-axis complaint about a user who never wrote an
    # asset name, so the message has to name the factor universe.
    msg = try
        prior(BayesianBlackLittermanPrior(; sets = fsets, views = f_views), rd)
    catch e
        sprint(showerror, e)
    end
    @test occursin("nf (the factor universe)", msg)
    @test occursin("required by BayesianBlackLittermanPrior", msg)
    @test occursin("it is not optional here", msg)
    @test !occursin("asset", msg)
    @test_throws KeyError prior(BayesianBlackLittermanPrior(; sets = fsets,
                                                            views = f_views), rd)
    # Declared, but not the axis `F` describes.
    shortf = UniverseSets(; dict = Dict("nx" => rd.nx, "nf" => rd.nf[1:2]))
    @test_throws DimensionMismatch prior(BayesianBlackLittermanPrior(; sets = shortf,
                                                                     views = f_views), rd)
    # …and says which two things disagree, rather than reporting an asset-axis length.
    msg = try
        prior(BayesianBlackLittermanPrior(; sets = shortf, views = f_views), rd)
    catch e
        sprint(showerror, e)
    end
    @test occursin("declared factor axis disagree", msg)
    # An *asset* name is now the thing that cannot resolve, and the message says so.
    msg = try
        prior(BayesianBlackLittermanPrior(; sets = xfsets,
                                          views = LinearConstraintEstimator(;
                                                                            val = ["$(rd.nx[1]) == 0.0001"])),
              rd; strict = true)
    catch e
        sprint(showerror, e)
    end
    @test occursin("not in factor universe", msg)
    @test occursin("$(length(rd.nf)) factors under key `nf`", msg)
    # The wrapped estimator's own `key` wins over the axis the estimator routes at, the same
    # precedence `rebase_linear_constraints` uses — the plumbing #230 added applies here
    # unchanged.
    msg = try
        prior(BayesianBlackLittermanPrior(; sets = xfsets,
                                          views = LinearConstraintEstimator(;
                                                                            val = ["MTUM == 0.0001",
                                                                                   "QUAL == 0.0001"],
                                                                            key = "nx")),
              rd; strict = true)
    catch e
        sprint(showerror, e)
    end
    @test occursin("not in asset universe", msg)
    @test occursin("$(length(rd.nx)) assets under key `nx`", msg)
    # A sets carrying both axes, under a view. The factor axis has no meaning for an asset
    # index, so it must come back untouched; the asset axis must slice. `sets` is `@vprop`
    # because the object can now carry both — the exemption is a property of the data.
    both = UniverseSets(;
                        dict = Dict("nx" => rd.nx, "nf" => rd.nf,
                                    "nx_sector" => repeat(["S"], length(rd.nx)),
                                    "nf_family" => string.("fam", eachindex(rd.nf))))
    pe = BayesianBlackLittermanPrior(; sets = both, views = f_views)
    i = [1, 3, 5]
    pev = PortfolioOptimisers.port_opt_view(pe, i)
    @test pev.sets.dict["nx"] == rd.nx[i]
    @test pev.sets.dict["nx_sector"] == repeat(["S"], length(i))
    @test pev.sets.dict["nf"] == rd.nf
    @test pev.sets.dict["nf_family"] == string.("fam", eachindex(rd.nf))
    # And the views still resolve after the slice: the axis they are written against did
    # not move.
    rdi = ReturnsResult(; nx = rd.nx[i], X = rd.X[:, i], nf = rd.nf, F = rd.F)
    @test isapprox(prior(pev, rdi).mu, prior(pe, rdi).mu)
    # The consistency identity of ADR 0046 is a property of the update, not of the lookup:
    # it must survive the migration exactly.
    pr = prior(BayesianBlackLittermanPrior(; sets = xfsets, views = f_views), rd)
    @test isapprox(pr.mu, pr.rr.M * pr.fpr.mu + pr.rr.b)
end

@testset "Factor Black Litterman" begin
    df = CSV.read(joinpath(@__DIR__, "./assets/FactorBlackLitterman1.csv.gz"), DataFrame)
    pe = FactorBlackLittermanPrior(; pe = EmpiricalPrior(;), rsd = false, sets = xfsets,
                                   tau = 1 / size(rd.X, 1),
                                   views = LinearConstraintEstimator(;
                                                                     val = ["MTUM == 0.0001",
                                                                            "QUAL - USMV == -0.0003"]))
    pr = prior(pe, rd)
    success = isapprox(pr.mu, df[1:20, 1]; rtol = 1e-6)
    if !success
        println("Mu $i fails")
        find_tol(pr.mu, df[1:20, i])
    end
    @test success

    success = isapprox(vec(pr.sigma), df[21:420, 1]; rtol = 1e-6)
    if !success
        println("Sigma $i fails")
        find_tol(vec(pr.sigma), df[21:420, i])
    end
    @test success

    success = isapprox(vec(pr.chol), df[421:end, 1]; rtol = 1e-6)
    if !success
        println("Chol $i fails")
        find_tol(vec(pr.chol), df[421:end, i])
    end
    @test success

    df = CSV.read(joinpath(@__DIR__, "./assets/FactorBlackLitterman2.csv.gz"), DataFrame)
    pe = FactorBlackLittermanPrior(; pe = EmpiricalPrior(;), sets = xfsets, l = 2,
                                   tau = 1 / size(rd.X, 1),
                                   views = LinearConstraintEstimator(;
                                                                     val = ["MTUM == 0.0001",
                                                                            "QUAL - USMV == -0.0003"]))
    pr = prior(pe, rd)
    success = isapprox(pr.mu, df[1:20, 1]; rtol = 1e-6)
    if !success
        println("Mu $i fails")
        find_tol(pr.mu, df[1:20, i])
    end
    @test success

    success = isapprox(vec(pr.sigma), df[21:420, 1]; rtol = 1e-6)
    if !success
        println("Sigma $i fails")
        find_tol(vec(pr.sigma), df[21:420, i])
    end
    @test success

    success = isapprox(vec(pr.chol), df[421:end, 1]; rtol = 1e-6)
    if !success
        println("Chol $i fails")
        find_tol(vec(pr.chol), df[421:end, i])
    end
    @test success
end

@testset "Factor Black Litterman reads the declared factor axis" begin
    # The golden tests above are the bit-identity proof: they run on `xfsets`, whose factors
    # live under `tfkey`, and still match `FactorBlackLitterman[12].csv.gz` to the same
    # tolerance they matched under the pre-migration shape. Only the *lookup* changed.
    f_views = LinearConstraintEstimator(; val = ["MTUM == 0.0001"])
    # The pre-migration shape — factor names under `xkey`, no factor axis at all — is the
    # error worth getting right, because the naive migration failure is an *asset*-axis
    # complaint about a user who never wrote an asset name.
    msg = try
        prior(FactorBlackLittermanPrior(; sets = fsets, views = f_views), rd)
    catch e
        sprint(showerror, e)
    end
    @test occursin("nf (the factor universe)", msg)
    @test occursin("required by FactorBlackLittermanPrior", msg)
    @test occursin("it is not optional here", msg)
    @test !occursin("asset", msg)
    @test_throws KeyError prior(FactorBlackLittermanPrior(; sets = fsets, views = f_views),
                                rd)
    # Declared, but not the axis `F` describes.
    shortf = UniverseSets(; dict = Dict("nx" => rd.nx, "nf" => rd.nf[1:2]))
    @test_throws DimensionMismatch prior(FactorBlackLittermanPrior(; sets = shortf,
                                                                   views = f_views), rd)
    # An *asset* name is now the thing that cannot resolve, and the message says so.
    msg = try
        prior(FactorBlackLittermanPrior(; sets = xfsets,
                                        views = LinearConstraintEstimator(;
                                                                          val = ["$(rd.nx[1]) == 0.0001"])),
              rd; strict = true)
    catch e
        sprint(showerror, e)
    end
    @test occursin("not in factor universe", msg)
    @test occursin("$(length(rd.nf)) factors under key `nf`", msg)
    # The wrapped estimator's own `key` wins over the axis the estimator routes at — the
    # precedence `rebase_linear_constraints` already uses. Before this migration the field
    # was dropped on the Black-Litterman path entirely, so pointing it at the asset axis
    # was a silent no-op; now it is honoured, and factor names cannot resolve there.
    msg = try
        prior(FactorBlackLittermanPrior(; sets = xfsets,
                                        views = LinearConstraintEstimator(;
                                                                          val = ["MTUM == 0.0001",
                                                                                 "QUAL == 0.0001"],
                                                                          key = "nx")), rd;
              strict = true)
    catch e
        sprint(showerror, e)
    end
    @test occursin("not in asset universe", msg)
    @test occursin("$(length(rd.nx)) assets under key `nx`", msg)
    # A sets carrying both axes, under a view. The factor axis has no meaning for an asset
    # index, so it must come back untouched; the asset axis must slice. `sets` is `@vprop`
    # because the object can now carry both — the exemption is a property of the data.
    both = UniverseSets(;
                        dict = Dict("nx" => rd.nx, "nf" => rd.nf,
                                    "nx_sector" => repeat(["S"], length(rd.nx)),
                                    "nf_family" => string.("fam", eachindex(rd.nf))))
    pe = FactorBlackLittermanPrior(; sets = both, views = f_views)
    i = [1, 3, 5]
    pev = PortfolioOptimisers.port_opt_view(pe, i)
    @test pev.sets.dict["nx"] == rd.nx[i]
    @test pev.sets.dict["nx_sector"] == repeat(["S"], length(i))
    @test pev.sets.dict["nf"] == rd.nf
    @test pev.sets.dict["nf_family"] == string.("fam", eachindex(rd.nf))
    # And the views still resolve after the slice: the axis they are written against did
    # not move.
    @test isapprox(prior(pev,
                         ReturnsResult(; nx = rd.nx[i], X = rd.X[:, i], nf = rd.nf,
                                       F = rd.F)).mu,
                   prior(pe,
                         ReturnsResult(; nx = rd.nx[i], X = rd.X[:, i], nf = rd.nf,
                                       F = rd.F)).mu)
end

@testset "Augmented Black Litterman" begin
    df = CSV.read(joinpath(@__DIR__, "./assets/AugmentedBlackLitterman.csv.gz"), DataFrame)
    pes = [AugmentedBlackLittermanPrior(; sets = afsets, tau = 1 / size(rd.X, 1),
                                        a_views = LinearConstraintEstimator(;
                                                                            val = Union{String,
                                                                                        Expr}[:(AAPL ==
                                                                                                0.00002),
                                                                                              :(BAC ==
                                                                                                CVX),
                                                                                              "WMT == group2",
                                                                                              "RRC-group1 == 0.0005"]),
                                        f_views = LinearConstraintEstimator(;
                                                                            val = [:(MTUM ==
                                                                                     0.0001),
                                                                                   :(QUAL -
                                                                                     USMV ==
                                                                                     -0.0003)])),
           AugmentedBlackLittermanPrior(; sets = afsets, tau = 1 / size(rd.X, 1), l = 2,
                                        a_views = LinearConstraintEstimator(;
                                                                            val = ["AAPL == 0.00002",
                                                                                   "BAC == CVX",
                                                                                   "WMT == group2",
                                                                                   "RRC-group1 == 0.0005"]),
                                        f_views = LinearConstraintEstimator(;
                                                                            val = ["MTUM == 0.0001",
                                                                                   "QUAL - USMV == -0.0003"]))]
    # The fixture is the published reference this estimator was ported from, and the
    # covariance still matches it exactly: ADR 0063's amendment moves the intercept and the
    # rate into the prior stack, and the covariance path never reads the prior mean.
    for (i, pe) in enumerate(pes)
        pr = prior(pe, rd)
        success = isapprox(vec(pr.sigma), df[21:420, i]; rtol = 1e-6)
        if !success
            println("Sigma $i fails")
            find_tol(vec(pr.sigma), df[21:420, i])
        end
        @test success
    end

    # `mu` departs from the reference, deliberately and by a stated amount (#570).
    #
    # Column 1, `l === nothing`: the reference adds the loadings constant to a historical
    # mean that already contains it, because least squares with an intercept makes the mean
    # of `X` equal `M * mu_f + b`. Ours is the reference less `rr.b`, entry by entry.
    pr1 = prior(pes[1], rd)
    @test isapprox(pr1.mu, df[1:20, 1] .- pr1.rr.b; atol = 1e-16)
    @test isapprox(maximum(abs, pr1.rr.b), 0.0016160796925727256; rtol = 1e-6)

    # Column 2, `l = 2`: the equilibrium premium carries no intercept, so only the *place*
    # differs. Ours blends the views against a prior that carries `b`, and the reference
    # adds `b` whole afterwards, so the departure is `(I - G*P)b - b` on the asset half.
    # That is not a plain offset, so it is pinned here as one measured number.
    pr2 = prior(pes[2], rd)
    @test isapprox(maximum(abs, pr2.mu .- df[1:20, 2]), 0.0006994080155258995; rtol = 1e-6)
    @test !isapprox(pr2.mu, df[1:20, 2] .- pr2.rr.b; rtol = 1e-3)
end

@testset "Augmented Black Litterman reads both declared axes" begin
    # The golden tests above are the bit-identity proof: they run on `afsets` — one dual-axis
    # sets — and still match `AugmentedBlackLitterman.csv.gz` to the tolerance they matched
    # when the asset views took `sets` and the factor views a separate factors-under-`xkey`
    # `fsets`. Only the *lookup* changed.
    a_views = LinearConstraintEstimator(; val = ["$(rd.nx[1]) == 0.0002"])
    f_views = LinearConstraintEstimator(; val = ["MTUM == 0.0001"])
    # Precomputed halves. A `BlackLittermanViews` result carries its own `P` and resolves no
    # names, so it is how each half is muted while the other keeps writing in names.
    a_blv = BlackLittermanViews(; P = reshape(Float64.(rd.nx .== rd.nx[1]), 1, :),
                                Q = [0.0002])
    f_blv = BlackLittermanViews(; P = reshape(Float64.(rd.nf .== "MTUM"), 1, :),
                                Q = [0.0001])
    # Each axis is required only by the half that resolves names against it, so a mandate
    # written on one axis alone is expressible with the same single sets.
    @test isapprox(prior(AugmentedBlackLittermanPrior(; sets = xfsets, a_views = a_views,
                                                      f_views = f_blv), rd).mu,
                   prior(AugmentedBlackLittermanPrior(; sets = xfsets, a_views = a_views,
                                                      f_views = f_views), rd).mu;
                   rtol = 1e-6)
    @test !isnothing(prior(AugmentedBlackLittermanPrior(; sets = xfsets, a_views = a_blv,
                                                        f_views = f_views), rd).mu)
    # Both halves precomputed: no universe is read at all, so no sets is needed. Under the two
    # unconditional length checks this shape threw on `nothing.dict` before reaching the model.
    @test isapprox(prior(AugmentedBlackLittermanPrior(; a_views = a_blv, f_views = f_blv),
                         rd).mu,
                   prior(AugmentedBlackLittermanPrior(; sets = xfsets, a_views = a_blv,
                                                      f_views = f_blv), rd).mu)
    # The pre-migration factor shape — factor names under `xkey`, no factor axis declared — is
    # the error worth getting right. Muting the asset half keeps the asset check out of the
    # way, so what is reported is the *factor* universe the user never declared.
    msg = try
        prior(AugmentedBlackLittermanPrior(; sets = fsets, a_views = a_blv,
                                           f_views = f_views), rd)
    catch e
        sprint(showerror, e)
    end
    @test occursin("nf (the factor universe)", msg)
    @test occursin("required by AugmentedBlackLittermanPrior", msg)
    @test occursin("`f_views` are written in factor names", msg)
    @test_throws KeyError prior(AugmentedBlackLittermanPrior(; sets = fsets,
                                                             a_views = a_blv,
                                                             f_views = f_views), rd)
    # Declared, but not the axis `F` describes.
    shortf = UniverseSets(; dict = Dict("nx" => rd.nx, "nf" => rd.nf[1:2]))
    @test_throws DimensionMismatch prior(AugmentedBlackLittermanPrior(; sets = shortf,
                                                                      a_views = a_views,
                                                                      f_views = f_views),
                                         rd)
    # And the asset axis is still checked, by the half that reads it.
    shortx = UniverseSets(; dict = Dict("nx" => rd.nx[1:2], "nf" => rd.nf))
    @test_throws DimensionMismatch prior(AugmentedBlackLittermanPrior(; sets = shortx,
                                                                      a_views = a_views,
                                                                      f_views = f_views),
                                         rd)
    # The latent bug, closed. One sets carrying both axes, under a view: the factor axis has no
    # meaning for an asset index and must come back untouched, the asset axis must slice. This
    # is the test the old shape could not have — `f_sets` was a *separate* object kept safe
    # only by the missing `@vprop`, so marking it would have sliced factor names by asset
    # indices and the mismatch would have surfaced as a length error inside `bl_preroll`.
    both = UniverseSets(;
                        dict = Dict("nx" => rd.nx, "nf" => rd.nf,
                                    "nx_sector" => repeat(["S"], length(rd.nx)),
                                    "nf_family" => string.("fam", eachindex(rd.nf))))
    pe = AugmentedBlackLittermanPrior(; sets = both, a_views = a_views, f_views = f_views)
    i = [1, 3, 5]
    pev = PortfolioOptimisers.port_opt_view(pe, i)
    @test pev.sets.dict["nx"] == rd.nx[i]
    @test pev.sets.dict["nx_sector"] == repeat(["S"], length(i))
    @test pev.sets.dict["nf"] == rd.nf
    @test pev.sets.dict["nf_family"] == string.("fam", eachindex(rd.nf))
    # The slice is load-bearing on one axis and inert on the other, and this is where the two
    # differ from the factor-only members: there the asset axis is unread, so a view changes
    # nothing. Here `a_views` reads it, so the *unsliced* estimator must reject the sliced
    # data, and the viewed one must be indistinguishable from the sets written by hand — the
    # asset entries cut to `i`, the factor entries whole.
    rdv = ReturnsResult(; nx = rd.nx[i], X = rd.X[:, i], nf = rd.nf, F = rd.F)
    @test_throws DimensionMismatch prior(pe, rdv)
    hand = UniverseSets(;
                        dict = Dict("nx" => rd.nx[i], "nf" => rd.nf,
                                    "nx_sector" => repeat(["S"], length(i)),
                                    "nf_family" => string.("fam", eachindex(rd.nf))))
    peh = AugmentedBlackLittermanPrior(; sets = hand, a_views = a_views, f_views = f_views)
    @test prior(pev, rdv).mu == prior(peh, rdv).mu
    # Had `sets` been sliced blindly the factor entries would have been cut to `i` too, and
    # the factor views would have resolved against a truncated axis. They do not: the factor
    # half of the posterior is the one the unsliced factor universe produces.
    @test prior(pev, rdv).fpr.mu == prior(peh, rdv).fpr.mu
    @test length(prior(pev, rdv).fpr.mu) == length(rd.nf)
end

@testset "LogEntropyPooling" begin
    pr0 = prior(EmpiricalPrior(), rd)
    opt = OptimEntropyPooling(; alg = LogEntropyPooling())
    jopt = JuMPEntropyPooling(; alg = LogEntropyPooling(), slv = slv)

    mu_views = LinearConstraintEstimator(; val = "AAPL == 0.002")
    pr = prior(EntropyPoolingPrior(; w = StatsBase.pweights(range(iT, iT; length = T)),
                                   sets = sets, mu_views = mu_views, opt = opt), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 1e-7)
    prmuw = prior(EntropyPoolingPrior(; sets = sets, opt = jopt, mu_views = mu_views), rd).w
    res = isapprox(pr.w, prmuw; rtol = 5e-5)
    if !res
        println("Weights fail")
        find_tol(pr.w, prmuw)
    end
    @test res

    pr = prior(EntropyPoolingPrior(;
                                   pe = FactorPrior(;
                                                    re = StepwiseRegression(; crit = :bic)),
                                   sets = sets, opt = opt, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 5e-4)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(;
                                             pe = FactorPrior(;
                                                              re = StepwiseRegression(;
                                                                                      crit = :bic)),
                                             sets = sets, opt = jopt, mu_views = mu_views),
                         rd).w, rtol = 5e-5)

    pr = prior(EntropyPoolingPrior(;
                                   pe = FactorPrior(;
                                                    re = DimensionReductionRegression(;
                                                                                      retgt = GeneralisedLinearModel())),
                                   sets = sets, opt = opt, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 5e-4)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(;
                                             pe = FactorPrior(;
                                                              re = DimensionReductionRegression(;
                                                                                                retgt = GeneralisedLinearModel())),
                                             sets = sets, opt = jopt, mu_views = mu_views),
                         rd).w, rtol = 5e-5)

    pr = prior(EntropyPoolingPrior(; w = StatsBase.pweights(range(iT, iT; length = T)),
                                   alg = H0_EntropyPooling(),
                                   pe = FactorPrior(;
                                                    re = StepwiseRegression(; crit = :bic)),
                                   sets = sets, opt = opt, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 5e-4)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; alg = H0_EntropyPooling(),
                                             pe = FactorPrior(;
                                                              re = StepwiseRegression(;
                                                                                      crit = :bic)),
                                             sets = sets, opt = jopt, mu_views = mu_views),
                         rd).w, rtol = 5e-5)

    pr = prior(EntropyPoolingPrior(; alg = H0_EntropyPooling(),
                                   pe = FactorPrior(;
                                                    re = DimensionReductionRegression(;
                                                                                      retgt = GeneralisedLinearModel())),
                                   sets = sets, opt = opt, mu_views = mu_views), rd)
    @test isapprox(pr.mu[1], 0.002, rtol = 5e-4)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; alg = H0_EntropyPooling(),
                                             pe = FactorPrior(;
                                                              re = DimensionReductionRegression(;
                                                                                                retgt = GeneralisedLinearModel())),
                                             sets = sets, opt = jopt, mu_views = mu_views),
                         rd).w, rtol = 5e-5)

    mu_views = LinearConstraintEstimator(; val = "AAPL >= 0.0025")
    pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views, opt = opt), rd)
    @test pr.mu[1] >= 0.0025
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             mu_views = mu_views), rd).w, rtol = 5e-5)

    mu_views = LinearConstraintEstimator(; val = "AAPL <= 0.001")
    pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views, opt = opt), rd)
    @test pr.mu[1] <= 0.001
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             mu_views = mu_views), rd).w, rtol = 5e-6)

    var_views = LinearConstraintEstimator(; val = "AAPL == 0.03264496113282452")
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = opt,
                                   var_views = ValueatRiskView(; views = var_views)), rd)
    var_tgt = 0.03264496113282452
    # The view is a constraint on the posterior tail mass, so that mass is what the solve meets.
    @test isapprox(sum(pr.w[i] for i in axes(rd.X, 1) if rd.X[i, 1] <= -var_tgt), 0.05,
                   rtol = 1e-6)
    # The reported value at risk is a sample order statistic, and the mass lands within `1e-8`
    # of `0.05` on either side, which reads one of the two observations that bracket the target.
    # See issues #573, #695 and #697.
    @test var_view_floor(rd.X[:, 1], var_tgt) <=
          ValueatRisk(; w = pr.w)(rd.X[:, 1]) <=
          var_view_ceiling(rd.X[:, 1], var_tgt)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = ValueatRiskView(;
                                                                         views = var_views)),
                         rd).w, rtol = 1e-6)

    var_views = LinearConstraintEstimator(; val = "AAPL >= 1.15*prior(AAPL)")
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = opt,
                                   var_views = ValueatRiskView(; views = var_views)), rd)
    var_tgt = 1.15 * ValueatRisk(;)(rd.X[:, 1])
    # The view is a constraint on the posterior tail mass, so that mass is what the solve meets.
    @test isapprox(sum(pr.w[i] for i in axes(rd.X, 1) if rd.X[i, 1] <= -var_tgt), 0.05,
                   rtol = 1e-6)
    # The reported value at risk is a sample order statistic, and the mass lands about `1e-8`
    # short of `0.05`, which reads one observation further down the tail. `var_view_floor` is
    # that observation. See issues #573 and #695.
    @test ValueatRisk(; w = pr.w)(rd.X[:, 1]) >= var_view_floor(rd.X[:, 1], var_tgt)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = ValueatRiskView(;
                                                                         views = var_views)),
                         rd).w, rtol = 1e-6)

    var_views = LinearConstraintEstimator(; val = "AAPL == 0.12865204867438676")
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = opt,
                                   var_views = ValueatRiskView(; views = var_views)), rd)
    var_tgt = 0.12865204867438676
    # The view is a constraint on the posterior tail mass, so that mass is what the solve meets.
    @test isapprox(sum(pr.w[i] for i in axes(rd.X, 1) if rd.X[i, 1] <= -var_tgt), 0.05,
                   rtol = 1e-6)
    # The target sits one observation from the worst realisation, so the reading is either it or
    # the second worst observation, depending on the sign of a residual near `1e-10`. See issues
    # #573, #695 and #697.
    @test var_view_floor(rd.X[:, 1], var_tgt) <=
          ValueatRisk(; w = pr.w)(rd.X[:, 1]) <=
          var_view_ceiling(rd.X[:, 1], var_tgt)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = ValueatRiskView(;
                                                                         views = var_views)),
                         rd).w, rtol = 5e-6)

    var_views = LinearConstraintEstimator(; val = ["AAPL == 0.028", "XOM >= 0.027"])
    pr = prior(EntropyPoolingPrior(; sets = sets, opt = opt,
                                   var_views = ValueatRiskView(; alpha = 0.07,
                                                               views = var_views)), rd)
    # The view is a constraint on the posterior tail mass, so that mass is what the solve meets.
    @test isapprox(sum(pr.w[i] for i in axes(rd.X, 1) if rd.X[i, 1] <= -0.028), 0.07,
                   rtol = 1e-6)
    # The reported value at risk is a sample order statistic, and the mass lands within `1e-10`
    # of `0.07` on either side, which reads one of the two observations that bracket the target.
    # A residual on the wrong side once put the reading outside the old `rtol = 7e-3` band. See
    # issues #573, #695 and #697.
    @test var_view_floor(rd.X[:, 1], 0.028) <=
          ValueatRisk(; alpha = 0.07, w = pr.w)(rd.X[:, 1]) <=
          var_view_ceiling(rd.X[:, 1], 0.028)
    @test ValueatRisk(; alpha = 0.07, w = pr.w)(rd.X[:, end]) >= 0.027
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             var_views = ValueatRiskView(; alpha = 0.07,
                                                                         views = var_views)),
                         rd).w, rtol = 1e-4)

    sigma_views = LinearConstraintEstimator(; val = "AAPL == 0.0007")
    pr = prior(EntropyPoolingPrior(; sets = sets, sigma_views = sigma_views, opt = opt), rd)
    r = LowOrderMoment(; w = pr.w, mu = pr.mu[1],
                       alg = SecondMoment(; ve = SimpleVariance(; w = pr.w)))
    @test isapprox(r([1], reshape(pr.X[:, 1], :, 1)), 0.0007, rtol = 1e-3)
    @test isapprox(pr.sigma[1, 1], r([1], reshape(pr.X[:, 1], :, 1)))
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, sigma_views = sigma_views,
                                             opt = jopt), rd).w, rtol = 1e-2)

    mu_views = LinearConstraintEstimator(; val = "AAPL == 1.7*prior(AAPL)")
    sigma_views = LinearConstraintEstimator(; val = "AAPL == 0.0008")
    pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                   sigma_views = sigma_views, opt = opt), rd)
    @test isapprox(pr.mu[1], pr0.mu[1] * 1.7, rtol = 5e-6)
    @test isapprox(pr.sigma[1, 1], 0.0008, rtol = 1e-3)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt, mu_views = mu_views,
                                             sigma_views = sigma_views), rd).w, rtol = 5e-4)

    sk_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*2")
    pr = prior(EntropyPoolingPrior(; sets = sets, sk_views = sk_views, opt = opt), rd)
    @test isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                         reshape(pr.X[:, 1],
                                                                                 :, 1)),
                   2 * Skewness()([1], reshape(pr0.X[:, 1], :, 1)), rtol = 2e-3)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             sk_views = sk_views), rd).w, rtol = 5e-3)

    mu_views = LinearConstraintEstimator(; val = "AAPL==1.5*prior(AAPL)")
    sigma_views = LinearConstraintEstimator(; val = "AAPL==1.3prior(AAPL)")
    sk_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*2")
    pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                   sigma_views = sigma_views, sk_views = sk_views,
                                   opt = opt), rd)
    @test isapprox(pr.mu[1], 1.5 * pr0.mu[1], rtol = 1e-4)
    @test isapprox(pr.sigma[1, 1], 1.3 * pr0.sigma[1, 1], rtol = 5e-3)
    @test isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                         reshape(pr.X[:, 1],
                                                                                 :, 1)),
                   2 * Skewness()([1], reshape(pr0.X[:, 1], :, 1)), rtol = 5e-3)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                             sigma_views = sigma_views,
                                             sk_views = sk_views), rd).w,
                   rtol = ifelse(Sys.islinux(), 1e-3, 1e-6))

    kt_views = LinearConstraintEstimator(; val = "AAPL == 7.5")
    pr = prior(EntropyPoolingPrior(; sets = sets, kt_views = kt_views, opt = opt), rd)
    @test isapprox(HighOrderMoment(; w = pr.w,
                                   alg = StandardisedHighOrderMoment(; alg = FourthMoment(),
                                                                     ve = SimpleVariance(;
                                                                                         w = pr.w)))([1],
                                                                                                     reshape(pr.X[:,
                                                                                                                  1],
                                                                                                             :,
                                                                                                             1)),
                   7.5, rtol = 5e-3)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             kt_views = kt_views), rd).w, rtol = 1e-2)

    mu_views = LinearConstraintEstimator(; val = "AAPL<=1.5*prior(AAPL)")
    sigma_views = LinearConstraintEstimator(; val = "AAPL==0.7prior(AAPL)")
    kt_views = LinearConstraintEstimator(; val = "AAPL >= prior(AAPL)*0.87")
    pr = prior(EntropyPoolingPrior(; sets = sets, mu_views = mu_views,
                                   sigma_views = sigma_views, kt_views = kt_views,
                                   opt = opt), rd)
    @test pr.mu[1] <= 1.5 * pr0.mu[1]
    @test isapprox(pr.sigma[1, 1], 0.7 * pr0.sigma[1, 1], rtol = 1e-3)
    @test isapprox(HighOrderMoment(; w = pr.w,
                                   alg = StandardisedHighOrderMoment(; alg = FourthMoment(),
                                                                     ve = SimpleVariance(;
                                                                                         w = pr.w)))([1],
                                                                                                     reshape(pr.X[:,
                                                                                                                  1],
                                                                                                             :,
                                                                                                             1)),
                   HighOrderMoment(;
                                   alg = StandardisedHighOrderMoment(;
                                                                     alg = FourthMoment()))([1],
                                                                                            reshape(pr.X[:,
                                                                                                         1],
                                                                                                    :,
                                                                                                    1)) *
                   0.87, rtol = 5.0e-6)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt, mu_views = mu_views,
                                             sigma_views = sigma_views,
                                             kt_views = kt_views), rd).w, rtol = 5e-3)

    cov_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == prior(AAPL, XOM)*1.1")
    pr = prior(EntropyPoolingPrior(; sets = sets, cov_views = cov_views, opt = opt), rd)
    @test isapprox(pr.sigma[1, end], pr0.sigma[1, end] * 1.1, rtol = 1e-3)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             cov_views = cov_views), rd).w, rtol = 5e-6)

    rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == 0.35")
    pr = prior(EntropyPoolingPrior(; sets = sets, rho_views = rho_views, opt = opt), rd)
    @test isapprox(StatsBase.cov2cor(pr.sigma)[1, end], 0.35, rtol = 5e-6)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             rho_views = rho_views), rd).w, rtol = 1e-2)

    rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == prior(AAPL,XOM)*0.94")
    pr = prior(EntropyPoolingPrior(; sets = sets, rho_views = rho_views, opt = opt), rd)
    @test isapprox(StatsBase.cov2cor(pr.sigma)[1, end],
                   StatsBase.cov2cor(pr0.sigma)[1, end] * 0.94, rtol = 5e-6)
    @test isapprox(pr.w,
                   prior(EntropyPoolingPrior(; sets = sets, opt = jopt,
                                             rho_views = rho_views), rd).w, rtol = 5e-2)

    pr = prior(HighOrderPriorEstimator(;
                                       pe = EntropyPoolingPrior(; alg = H2_EntropyPooling(),
                                                                sets = sets,
                                                                mu_views = LinearConstraintEstimator(;
                                                                                                     val = ["AAPL<=0.92*prior(AAPL)",
                                                                                                            "XOM >= 0.83*prior(XOM)"]),
                                                                sigma_views = LinearConstraintEstimator(;
                                                                                                        val = ["AAPL==1.2prior(AAPL)",
                                                                                                               "WMT==1.4prior(WMT)"]),
                                                                rho_views = LinearConstraintEstimator(;
                                                                                                      val = "(AAPL, XOM) == 0.35"),
                                                                opt = opt)), rd)
    @test pr.mu[1] <= 0.92 * pr0.mu[1] + sqrt(eps()) * length(pr0.mu) * 2
    @test pr.mu[end] >= 0.83 * pr0.mu[end] - sqrt(eps())
    @test isapprox(pr.sigma[1, 1], 1.2 * pr0.sigma[1, 1], rtol = 1e-2)
    @test isapprox(pr.sigma[19, 19], 1.4 * pr0.sigma[19, 19], rtol = 5e-3)
    @test isapprox(StatsBase.cov2cor(pr.sigma)[1, end], 0.35, rtol = 1e-3)

    cvar_views = LinearConstraintEstimator(; val = "AAPL == 0.07")
    pr = prior(MeucciEntropyPoolingPrior(; sets = sets, opt = opt,
                                         cvar_views = ConditionalValueatRiskView(;
                                                                                 views = cvar_views)),
               rd)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.07, rtol = 1e-6)
    @test isapprox(pr.w,
                   prior(MeucciEntropyPoolingPrior(; sets = sets, opt = jopt,
                                                   cvar_views = ConditionalValueatRiskView(;
                                                                                           views = cvar_views)),
                         rd).w, rtol = 5e-5)

    cvar_views = LinearConstraintEstimator(; val = "AAPL == prior(AAPL)*1.37")
    pr = prior(MeucciEntropyPoolingPrior(; sets = sets, opt = opt,
                                         cvar_views = ConditionalValueatRiskView(;
                                                                                 views = cvar_views)),
               rd)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]),
                   ConditionalValueatRisk(;)(rd.X[:, 1]) * 1.37, rtol = 1e-6)
    @test isapprox(pr.w,
                   prior(MeucciEntropyPoolingPrior(; sets = sets, opt = jopt,
                                                   cvar_views = ConditionalValueatRiskView(;
                                                                                           views = cvar_views)),
                         rd).w, rtol = 1e-5)

    cvar_views = LinearConstraintEstimator(; val = ["AAPL == 0.053", "XOM==0.045"])
    pr = prior(HighOrderPriorEstimator(;
                                       pe = MeucciEntropyPoolingPrior(; sets = sets,
                                                                      alg = H2_EntropyPooling(),
                                                                      opt = opt,
                                                                      cvar_views = ConditionalValueatRiskView(;
                                                                                                              views = cvar_views))),
               rd)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, 1]), 0.053, rtol = 5e-5)
    @test isapprox(ConditionalValueatRisk(; w = pr.w)(rd.X[:, end]), 0.045, rtol = 1e-4)
    @test isapprox(pr.w,
                   prior(HighOrderPriorEstimator(;
                                                 pe = MeucciEntropyPoolingPrior(;
                                                                                sets = sets,
                                                                                alg = H1_EntropyPooling(),
                                                                                cvar_views = ConditionalValueatRiskView(;
                                                                                                                        views = cvar_views))),
                         rd).w, rtol = 5e-3)

    mu_views = LinearConstraintEstimator(;
                                         val = ["AAPL<=0.75*prior(AAPL)",
                                                "XOM >= 0.4*prior(XOM)"])
    sigma_views = LinearConstraintEstimator(;
                                            val = ["AAPL==0.2prior(AAPL)",
                                                   "WMT==1.4prior(WMT)"])
    cov_views = LinearConstraintEstimator(; val = "(MSFT, PEP) <= prior(MSFT, PEP)*0.8")
    rho_views = LinearConstraintEstimator(; val = "(AAPL, XOM) == 0.35")
    kt_views = LinearConstraintEstimator(; val = "AAPL >= prior(AAPL)*0.3")
    sk_views = LinearConstraintEstimator(; val = "WMT == prior(WMT)*1.4")
    pr = prior(HighOrderPriorEstimator(;
                                       pe = EntropyPoolingPrior(; alg = H0_EntropyPooling(),
                                                                sets = sets,
                                                                mu_views = mu_views,
                                                                sigma_views = sigma_views,
                                                                cov_views = cov_views,
                                                                rho_views = rho_views,
                                                                kt_views = kt_views,
                                                                sk_views = sk_views,
                                                                opt = opt)), rd)
    @test isapprox(pr.mu[1], 0.75 * pr0.mu[1], rtol = 1e-5)
    @test pr.mu[end] >= 0.4 * pr0.mu[end]
    @test isapprox(pr.sigma[1, 1], 0.2 * pr0.sigma[1, 1], rtol = 1e-2)
    @test isapprox(pr.sigma[19, 19], 1.4 * pr0.sigma[19, 19], rtol = 5e-3)
    @test !isapprox(StatsBase.cov2cor(pr.sigma)[1, end], 0.35; rtol = 5e-4)
    @test pr0.sigma[13, 14] * 0.8 >= pr.sigma[13, 14]
    @test HighOrderMoment(; w = pr.w,
                          alg = StandardisedHighOrderMoment(; alg = FourthMoment(),
                                                            ve = SimpleVariance(; w = pr.w)))([1],
                                                                                              reshape(pr.X[:,
                                                                                                           1],
                                                                                                      :,
                                                                                                      1)) >=
          HighOrderMoment(; alg = StandardisedHighOrderMoment(; alg = FourthMoment()))([1],
                                                                                       reshape(pr.X[:,
                                                                                                    1],
                                                                                               :,
                                                                                               1)) *
          0.3
    @test !isapprox(Skewness(; w = pr.w, ve = SimpleVariance(; w = pr.w))([1],
                                                                          reshape(pr.X[:,
                                                                                       end - 1],
                                                                                  :, 1)),
                    1.4 * Skewness()([1], reshape(pr0.X[:, end - 1], :, 1)); rtol = 5e-3)
end

@testset "Opinion pooling" begin
    pr0 = prior(EmpiricalPrior(), rd)
    pr = prior(OpinionPoolingPrior(;
                                   pes = [EntropyPoolingPrior(; sets = sets,
                                                              mu_views = LinearConstraintEstimator(;
                                                                                                   val = "AAPL == prior(AAPL)*1.5")),
                                          EntropyPoolingPrior(; sets = sets,
                                                              mu_views = LinearConstraintEstimator(;
                                                                                                   val = "AAPL == prior(AAPL)*2"))],
                                   w = [0.4, 0.4]), rd)
    @test isapprox(pr.mu[1], 0.0022963115075039417, rtol = 1e-6)

    pr = prior(OpinionPoolingPrior(;
                                   pes = [EntropyPoolingPrior(; sets = sets,
                                                              mu_views = LinearConstraintEstimator(;
                                                                                                   val = "AAPL == prior(AAPL)*1.5")),
                                          EntropyPoolingPrior(; sets = sets,
                                                              mu_views = LinearConstraintEstimator(;
                                                                                                   val = "AAPL == prior(AAPL)*2"))],
                                   w = [0.5, 0.5], alg = LogarithmicOpinionPooling(),
                                   p = 5), rd)
    @test isapprox(pr.mu[1], 0.002511023272287914, rtol = 1e-6)

    # Regression: prior(OpinionPoolingPrior) must not mutate the stored weights `w`.
    # `w` sums to < 1 so the uniform-prior remainder branch is exercised; calling `prior`
    # twice on the same estimator used to grow `w` (via `push!` on an aliased vector),
    # throwing `length(w) == length(pes)` on the second call.
    opp = OpinionPoolingPrior(;
                              pes = [EntropyPoolingPrior(; sets = sets,
                                                         mu_views = LinearConstraintEstimator(;
                                                                                              val = "AAPL == prior(AAPL)*1.5")),
                                     EntropyPoolingPrior(; sets = sets,
                                                         mu_views = LinearConstraintEstimator(;
                                                                                              val = "AAPL == prior(AAPL)*2"))],
                              w = [0.3, 0.3])
    pr_a = prior(opp, rd)
    @test length(opp.w) == 2
    pr_b = prior(opp, rd)
    @test length(opp.w) == 2
    @test isapprox(pr_a.mu, pr_b.mu)

    # Uniform-weight branch (`w === nothing`). `range(inv(M), inv(M); length = M)` carries
    # `TwicePrecision` endpoints, so its entries sum to exactly `1.0` for every `M` measured
    # (2, 3, 5, 7, 11) and `rw` is `0.0`. The branch therefore does *not* take the remainder
    # path -- it takes the `else` arm, and `pw` gets `M` columns rather than `M + 1`.
    pr_unif = prior(OpinionPoolingPrior(;
                                        pes = [EntropyPoolingPrior(; sets = sets,
                                                                   mu_views = LinearConstraintEstimator(;
                                                                                                        val = "AAPL == prior(AAPL)*1.5")),
                                               EntropyPoolingPrior(; sets = sets,
                                                                   mu_views = LinearConstraintEstimator(;
                                                                                                        val = "AAPL == prior(AAPL)*2"))]),
                    rd)
    @test all(isfinite, pr_unif.mu)

    # Regression: `robust_probabilities` must not write into the vector it is handed.
    # It did `ow .*= exp.(-p * kldivs)`, and when the opinion weights already sum to one
    # `ow` *is* `pe.w`, the estimator's own field: one `prior` call left `[0.5, 0.5]` as
    # `[0.49996615, 0.49996620]`, so the stored weights no longer summed to one and the
    # second call pooled a uniform-prior remainder the caller never asked for.
    op_a = EntropyPoolingPrior(; sets = sets,
                               mu_views = LinearConstraintEstimator(;
                                                                    val = "AAPL == prior(AAPL)*1.5"))
    op_b = EntropyPoolingPrior(; sets = sets,
                               mu_views = LinearConstraintEstimator(;
                                                                    val = "AAPL == prior(AAPL)*2"))
    opp_p = OpinionPoolingPrior(; pes = [op_a, op_b], w = [0.5, 0.5], p = 5)
    w_before = copy(opp_p.w)
    pr_p1 = prior(opp_p, rd)
    @test opp_p.w == w_before
    pr_p2 = prior(opp_p, rd)
    @test opp_p.w == w_before
    @test pr_p1.mu == pr_p2.mu
    @test pr_p1.ow == pr_p2.ow

    # Same defect, other face: the uniform-weight branch builds `ow` as an immutable
    # `range` whose entries sum to exactly one, so `.*=` could not write to it at all and
    # a penalty with no `w` threw `CanonicalIndexError` before reaching the pooling.
    pr_unif_p = prior(OpinionPoolingPrior(; pes = [op_a, op_b], p = 5), rd)
    @test all(isfinite, pr_unif_p.mu)
    @test isapprox(sum(pr_unif_p.ow), 1)
    @test length(pr_unif_p.ow) == 2
end

@testset "Opinion pooling: the pooling rules against hand computations" begin
    P = [0.30 0.10 0.25
         0.10 0.40 0.15
         0.20 0.20 0.20
         0.25 0.20 0.10
         0.15 0.10 0.30]
    a = [0.5, 0.3, 0.2]

    lin = PortfolioOptimisers.compute_pooling(LinearOpinionPooling(), a, P)
    hand_lin = [sum(P[t, k] * a[k] for k in 1:3) for t in 1:5]
    @test isapprox(collect(lin), hand_lin)
    @test isapprox(sum(lin), 1)
    @test all(>=(0), lin)

    u = [sum(a[k] * log(P[t, k]) for k in 1:3) for t in 1:5]
    hand_log = exp.(u) / sum(exp.(u))
    lg = PortfolioOptimisers.compute_pooling(LogarithmicOpinionPooling(), a, P)
    @test isapprox(collect(lg), hand_log)
    @test isapprox(sum(lg), 1)
    @test all(>=(0), lg)
    # The two rules are genuinely different answers on the same input.
    @test !isapprox(collect(lin), collect(lg))

    # A single zero scenario weight zeroes that scenario under the logarithmic rule and
    # leaves it positive under the linear one.
    Pz = copy(P)
    Pz[3, 1] = 0.0
    Pz[5, 1] = 0.35
    lgz = PortfolioOptimisers.compute_pooling(LogarithmicOpinionPooling(), a, Pz)
    linz = PortfolioOptimisers.compute_pooling(LinearOpinionPooling(), a, Pz)
    @test iszero(lgz[3])
    @test !iszero(linz[3])
    @test isapprox(sum(lgz), 1)

    # A zero *opinion* weight against that zero scenario weight makes `0 * log(0)`, which is
    # `NaN`, and `StatsBase.pweights` refuses it. This is the case the closed form does not
    # cover; the docstring of `compute_pooling` states it under `# Validation`.
    @test_throws ArgumentError PortfolioOptimisers.compute_pooling(LogarithmicOpinionPooling(),
                                                                   [0.0, 0.5, 0.5], Pz)

    # The robustness penalty against a hand-computed `alpha_k * exp(-p * D_k)`.
    p = 2.0
    c = P * a
    D = [sum(P[t, k] * log(P[t, k] / c[t]) for t in 1:5) for k in 1:3]
    num = a .* exp.(-p * D)
    @test isapprox(PortfolioOptimisers.robust_probabilities(a, P, p), num / sum(num))
    @test isapprox(sum(PortfolioOptimisers.robust_probabilities(a, P, p)), 1)
    # The argument is never modified, and the no-penalty method returns it untouched.
    @test a == [0.5, 0.3, 0.2]
    @test PortfolioOptimisers.robust_probabilities(a, P, nothing) === a

    # A larger `p` concentrates the mass on the opinion of smallest divergence, and the
    # limit is that opinion alone.
    @test argmin(D) == 1
    masses = [PortfolioOptimisers.robust_probabilities(a, P, pp)[1]
              for pp in (1.0, 10.0, 100.0, 1000.0)]
    @test issorted(masses)
    @test isapprox(masses[end], 1)
end

@testset "OpinionPoolingPrior constructor guards" begin
    ep = EntropyPoolingPrior()
    @test_throws PortfolioOptimisers.IsEmptyError OpinionPoolingPrior(;
                                                                      pes = EntropyPoolingPrior[])
    # `p` is bounded below by zero strictly: there is no `p = 0`.
    @test_throws DomainError OpinionPoolingPrior(; pes = [ep], p = 0.0)
    @test_throws DomainError OpinionPoolingPrior(; pes = [ep], p = -1.0)
    @test_throws PortfolioOptimisers.IsEmptyError OpinionPoolingPrior(; pes = [ep],
                                                                      w = Float64[])
    @test_throws DimensionMismatch OpinionPoolingPrior(; pes = [ep, ep], w = [1.0])
    @test_throws DomainError OpinionPoolingPrior(; pes = [ep, ep], w = [-0.1, 0.5])
    @test_throws DomainError OpinionPoolingPrior(; pes = [ep, ep], w = [0.7, 0.7])
    # The sum guard is an inequality: weights summing to less than one are legal, and the
    # remainder becomes a uniform-prior opinion in `prior`.
    @test isa(OpinionPoolingPrior(; pes = [ep, ep], w = [0.3, 0.3]), OpinionPoolingPrior)
end

@testset "Factor block guard on wrapped priors" begin
    # `pe` is typed `AbstractLowOrderPriorEstimator_F_AF`, whose `_AF` half uses factor
    # returns only *optionally* — so the type constrains which returns an estimator
    # consumes, not whether the result it produces carries a regression. An estimator that
    # never computes one projects factor moments through loadings that do not exist, and
    # used to die on `rr === nothing` with a bare `FieldError`/`MethodError` from deep
    # inside the projection.
    no_rr = EntropyPoolingPrior(; pe = EmpiricalPrior())
    @test isnothing(prior(no_rr, rd).rr)

    hofpe_err = try
        prior(HighOrderFactorPriorEstimator(; pe = no_rr), rd)
    catch e
        e
    end
    @test hofpe_err isa PortfolioOptimisers.IsNothingError
    @test occursin("regression", hofpe_err.msg)
    @test occursin("`pe`", hofpe_err.msg)

    bbl_err = try
        prior(BayesianBlackLittermanPrior(; pe = no_rr, sets = xfsets,
                                          tau = 1 / size(rd.X, 1),
                                          views = LinearConstraintEstimator(;
                                                                            val = ["MTUM == 0.0001"])),
              rd)
    catch e
        e
    end
    @test bbl_err isa PortfolioOptimisers.IsNothingError
    @test occursin("regression", bbl_err.msg)

    # The *other* way to arrive here — a wrapper that computed loadings and discarded them
    # — no longer exists: every wrapping estimator forwards `rr` and the factor block
    # `fpr` (ADR 0046), so nesting order does not matter. `BlackLittermanPrior` over a
    # `FactorPrior` used to be the canonical case and now goes straight through.
    forwards_rr = BlackLittermanPrior(; pe = FactorPrior(), sets = sets,
                                      tau = 1 / size(rd.X, 1),
                                      views = LinearConstraintEstimator(;
                                                                        val = ["AAPL == 0.001"]))
    @test !isnothing(prior(forwards_rr, rd).rr)
    @test !isnothing(prior(HighOrderFactorPriorEstimator(; pe = forwards_rr), rd).rr)
    @test !isnothing(prior(BayesianBlackLittermanPrior(; pe = forwards_rr, sets = xfsets,
                                                       tau = 1 / size(rd.X, 1),
                                                       views = LinearConstraintEstimator(;
                                                                                         val = ["MTUM == 0.0001"])),
                           rd).rr)

    # A prior that does carry loadings still goes through untouched.
    @test !isnothing(prior(HighOrderFactorPriorEstimator(; pe = FactorPrior()), rd).rr)
end

@testset "The factor block is a nested prior result" begin
    pr = prior(FactorPrior(), rd)

    # `fpr` is the factor-axis prior itself, not a copy of three of its fields: its `X` is
    # the factor returns matrix, over the same observations as the asset block.
    @test pr.fpr isa LowOrderPrior
    @test size(pr.fpr.X) == size(rd.F)
    @test size(pr.fpr.X, 1) == size(pr.X, 1)
    @test length(pr.fpr.mu) == size(pr.rr.M, 2)

    # The flat `f_*` names are virtual reads of it, so code written against the pre-nesting
    # shape is unaffected — and the three that did not exist before come with them.
    @test pr.f_mu === pr.fpr.mu
    @test pr.f_sigma === pr.fpr.sigma
    @test pr.f_w === pr.fpr.w
    @test pr.f_ens === pr.fpr.ens
    @test pr.f_kld === pr.fpr.kld
    @test pr.f_ow === pr.fpr.ow
    for sym in (:f_mu, :f_sigma, :f_w, :f_ens, :f_kld, :f_ow)
        @test sym in propertynames(pr)
        @test !hasfield(LowOrderPrior, sym)
    end

    # With no factor block they read `nothing` rather than throwing — the behaviour the flat
    # fields had, and the reason these are `compute` with a lambda and not `alias`.
    bare = prior(EmpiricalPrior(), rd)
    @test isnothing(bare.fpr)
    @test all(isnothing,
              (bare.f_mu, bare.f_sigma, bare.f_w, bare.f_ens, bare.f_kld, bare.f_ow))

    # `rr` and `fpr` are one block: neither half is constructible alone.
    @test_throws ArgumentError LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma,
                                             rr = pr.rr)
    @test_throws ArgumentError LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma,
                                             fpr = pr.fpr)
    # The flag names in that message used to be inverted — a present `rr` reported
    # `isnothing(rr) = true`.
    block_err = try
        LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, rr = pr.rr)
    catch e
        e
    end
    @test occursin("isnothing(rr) = false", block_err.msg)
    @test occursin("isnothing(fpr) = true", block_err.msg)

    # A view slices assets, so the factor block is forwarded whole: factors are not assets.
    i = [1, 3, 5]
    prv = PortfolioOptimisers.port_opt_view(pr, i)
    @test prv.fpr === pr.fpr
    @test size(prv.rr.M, 1) == length(i)

    # Every wrapping estimator that reports a factor block reports it nested, and the flat
    # reads keep returning the same values they did as fields.
    for pe in (FactorPrior(),
               FactorBlackLittermanPrior(; sets = xfsets, tau = 1 / size(rd.X, 1),
                                         views = LinearConstraintEstimator(;
                                                                           val = ["MTUM == 0.0001"])),
               BayesianBlackLittermanPrior(; pe = FactorPrior(), sets = xfsets,
                                           tau = 1 / size(rd.X, 1),
                                           views = LinearConstraintEstimator(;
                                                                             val = ["MTUM == 0.0001"])),
               EntropyPoolingPrior(; pe = FactorPrior()),
               FeaturePrior(; pe = FactorPrior(), ze = RegressionFeatures()))
        prf = prior(pe, rd)
        @test prf.fpr isa LowOrderPrior
        @test prf.f_mu === prf.fpr.mu
        @test prf.f_sigma === prf.fpr.sigma
        @test size(prf.fpr.X, 1) == size(prf.X, 1)
    end

    # The one place `f_w`'s *value* changed: the pooling producers used to stamp their own
    # pooled weights onto the factor slot (`f_w = !isnothing(rr) ? w : nothing`), which was a
    # duplicate of `w` rather than anything the factor prior computed. The nested block now
    # reports what its own producer recorded — and `EmpiricalPrior` records no `w`, so behind
    # a plain `FactorPrior` that is `nothing`. Closing *that* gap belongs to the producer.
    pooled = prior(EntropyPoolingPrior(; pe = FactorPrior(), sets = sets,
                                       mu_views = LinearConstraintEstimator(;
                                                                            val = ["AAPL == 0.001"])),
                   rd)
    @test !isnothing(pooled.rr)
    @test pooled.w isa StatsBase.AbstractWeights
    @test isnothing(pooled.f_w)
    @test isnothing(pooled.fpr.w)
end

@testset "Returns source selector (x_src)" begin
    # `x_src` picks which of the two asset-returns carriers the clustering, phylogeny and
    # centrality bridges read: `:prior` takes `pr.X`, `:data` takes `rd.X`. It replaced the
    # `cle_pr::Bool` flag in ADR 0044, whose name and documented meaning were both wrong.
    pr = prior(EmpiricalPrior(), rd)
    rd_alt = ReturnsResult(; nx = reverse(rd.nx), X = reverse(rd.X; dims = 2))
    @test pr.X !== rd_alt.X

    # The picker is the single point where the decision is made; all eight bridge sites
    # delegate to it.
    @test PortfolioOptimisers.returns_matrix_picker(pr, nothing, :prior) === pr.X
    @test PortfolioOptimisers.returns_matrix_picker(pr, rd_alt, :prior) === pr.X
    @test PortfolioOptimisers.returns_matrix_picker(pr, rd_alt, :data) === rd_alt.X
    # Without a returns result there is nothing to select between, so `x_src` is inert.
    @test PortfolioOptimisers.returns_matrix_picker(pr, nothing, :data) === pr.X

    # A `Bool` could not be wrong; a `Symbol` can, so the typo throws where it was written.
    for src in (:Prior, :data_, :returns, :pr)
        @test_throws ArgumentError PortfolioOptimisers.returns_matrix_picker(pr, rd_alt,
                                                                             src)
    end

    # The kwarg is wired through the bridge under its new name.
    ne = NetworkEstimator(;)
    @test phylogeny_matrix(ne, pr; rd = rd_alt, x_src = :data).X ==
          phylogeny_matrix(ne, rd_alt.X).X
    @test phylogeny_matrix(ne, pr; rd = rd_alt, x_src = :prior).X ==
          phylogeny_matrix(ne, pr.X).X
    @test phylogeny_matrix(ne, pr; rd = rd_alt, x_src = :data).X !=
          phylogeny_matrix(ne, pr; rd = rd_alt, x_src = :prior).X
    @test_throws ArgumentError phylogeny_matrix(ne, pr; rd = rd_alt, x_src = :Prior)

    # Every optimiser carrying `x_src` validates it at construction.
    @test_throws ArgumentError HierarchicalOptimiser(; x_src = :typo)
    @test_throws ArgumentError JuMPOptimiser(; slv = slv, x_src = :typo)
    @test_throws ArgumentError NestedClustered(; opti = EqualWeighted(),
                                               opto = EqualWeighted(), x_src = :typo)
    # The `cle_pr` spelling is gone: a Bool no longer constructs.
    @test_throws Exception HierarchicalOptimiser(; x_src = true)
    @test HierarchicalOptimiser().x_src == :prior
    @test JuMPOptimiser(; slv = slv).x_src == :prior
    @test NestedClustered(; opti = EqualWeighted(), opto = EqualWeighted()).x_src == :prior
end

@testset "The prior carrier's `w` is documented as observation weights" begin
    # `field_dict[:w_prior]` is what `LowOrderPrior`'s `# Fields` entry for `w` interpolates,
    # and it read "Portfolio weights vector used in prior computation" — the wrong noun and
    # the wrong axis. `w::Option{<:ObsWeights}` is validated against `size(X, 1)` and is
    # consumed as observation weights: `@pprop w` selects it into risk measures, which
    # resolve it with `get_observation_weights` against a length-`T` return series.
    entry = PortfolioOptimisers.field_dict[:w_prior]
    @test occursin("Observation weights", entry)
    @test !occursin("Portfolio weights", entry)

    pr = prior(EntropyPoolingPrior(; sets = sets,
                                   mu_views = LinearConstraintEstimator(;
                                                                        val = "AAPL == 0.002")),
               rd)
    @test length(pr.w) == size(pr.X, 1)
    @test length(pr.w) != size(pr.X, 2)
    @test pr.w isa PortfolioOptimisers.ObsWeights
    # The axis the old wording named is the one the constructor refuses.
    @test_throws DimensionMismatch LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma,
                                                 w = StatsBase.pweights(fill(1 /
                                                                             size(pr.X, 2),
                                                                             size(pr.X, 2))))

    # The neighbouring entry that would have been reached for instead is gone: `:rr` read
    # "Returns result" while sitting in the prior-result block, where the field named `rr`
    # is the *regression*. It had no consumer, so nothing stated it — but the next docstring
    # to interpolate it would have. `:reg_rr` is the entry every prior docstring uses.
    @test !haskey(PortfolioOptimisers.arg_dict, :rr)
    @test occursin("Regression result", PortfolioOptimisers.field_dict[:reg_rr])

    # `kld` is `Option{<:Num_VecNum}`: scalar from entropy pooling, one entry per opinion
    # from opinion pooling. The entry described only the scalar.
    @test occursin("one entry per opinion", PortfolioOptimisers.field_dict[:kld])
    @test pr.kld isa Number
    pooled = prior(OpinionPoolingPrior(;
                                       pes = [EntropyPoolingPrior(; sets = sets,
                                                                  mu_views = LinearConstraintEstimator(;
                                                                                                       val = "AAPL == prior(AAPL)*1.5")),
                                              EntropyPoolingPrior(; sets = sets,
                                                                  mu_views = LinearConstraintEstimator(;
                                                                                                       val = "AAPL == prior(AAPL)*2"))],
                                       w = [0.5, 0.5]), rd)
    @test pooled.kld isa AbstractVector
    @test length(pooled.kld) == 2
    @test all(isfinite, pooled.kld)
end

@testset "Precomputed Black-Litterman views need no sets" begin
    # `assert_bl` deliberately permits `sets === nothing` when the views are a
    # `BlackLittermanViews` *result*: it carries its own `P` and `Q` and resolves no names.
    # Every member then read its universe unconditionally and died before reaching the model
    # — #232 closed this for `AugmentedBlackLittermanPrior`, the other three inherited it.
    fblv = BlackLittermanViews(; P = reshape(Float64.(rd.nf .== "MTUM"), 1, :),
                               Q = [0.0001])
    ablv = BlackLittermanViews(; P = reshape(Float64.(rd.nx .== rd.nx[1]), 1, :),
                               Q = [0.0002])

    @test isa(prior(FactorBlackLittermanPrior(; views = fblv), rd), LowOrderPrior)
    @test isa(prior(BayesianBlackLittermanPrior(; pe = FactorPrior(), views = fblv), rd),
              LowOrderPrior)
    @test isa(prior(BlackLittermanPrior(; views = ablv), rd), LowOrderPrior)
    @test isa(prior(AugmentedBlackLittermanPrior(; a_views = ablv, f_views = fblv), rd),
              LowOrderPrior)

    # The gate is on the views, not on the sets: supplying a sets alongside precomputed views
    # is still legal and still changes nothing, so the two agree bit for bit.
    @test prior(FactorBlackLittermanPrior(; views = fblv, sets = xfsets), rd).mu ==
          prior(FactorBlackLittermanPrior(; views = fblv), rd).mu
    @test prior(BlackLittermanPrior(; views = ablv, sets = sets), rd).mu ==
          prior(BlackLittermanPrior(; views = ablv), rd).mu

    # Named views still demand — and still check — the axis they resolve against.
    f_views = LinearConstraintEstimator(; val = "MTUM == 0.0001")
    @test_throws IsNothingError FactorBlackLittermanPrior(; views = f_views)
    @test_throws KeyError prior(FactorBlackLittermanPrior(; views = f_views, sets = sets),
                                rd)

    # Not crashing is not the same as being right. A precomputed `P` resolves no names, so
    # `bl_preroll` — where `P` first meets the distribution it updates — is the only place its
    # width is ever seen; before, a wrong one reached the linear algebra as a bare
    # `DimensionMismatch` about two matrices the caller never wrote down.
    wide = BlackLittermanViews(; P = ones(1, size(rd.F, 2) + 1), Q = [0.0001])
    @test_throws DimensionMismatch prior(FactorBlackLittermanPrior(; views = wide), rd)
    @test_throws DimensionMismatch prior(BlackLittermanPrior(; views = wide), rd)

    # Regression: a precomputed views object must feel the same `views_conf` bound the
    # equation-shaped routes enforce. `assert_bl_views_conf(_, ::BlackLittermanViews)`
    # checked only the length, so a confidence outside (0, 1) reached `calc_omega`, whose
    # `1/v - 1` scale is then negative — `views_conf = 2.5` gave a view variance of
    # -8.6e-7 and the estimator answered from it rather than raising.
    two = BlackLittermanViews(;
                              P = vcat(reshape(Float64.(rd.nx .== rd.nx[1]), 1, :),
                                       reshape(Float64.(rd.nx .== rd.nx[2]), 1, :)),
                              Q = [0.0002, 0.0003])
    @test_throws DomainError BlackLittermanPrior(; views = two, views_conf = [2.5, 0.5])
    @test_throws DomainError BlackLittermanPrior(; views = two, views_conf = [-1.0, 0.5])
    @test_throws DomainError BlackLittermanPrior(; views = two, views_conf = [1.0, 0.5])
    @test_throws DomainError BlackLittermanPrior(; views = two, views_conf = [0.0, 0.5])
    # A confidence inside the interval still constructs, and every view variance is positive.
    ok = BlackLittermanPrior(; views = two, views_conf = [0.4, 0.5])
    @test all(>(0),
              diag(PortfolioOptimisers.calc_omega(ok.views_conf, two.P,
                                                  prior(EmpiricalPrior(), rd).sigma)))
end

#=
The Black-Litterman core, swept under #535. Condition 2 of that ticket asks for the view
uncertainty matrix, the master equations and the risk-free rate to be checked with numbers
rather than read. Every assertion below is the closed form of `calc_omega` or of
`vanilla_posteriors` evaluated beside the code that implements it, on a 200 × 3 sample with
two views.
=#
@testset "Black-Litterman core: the view uncertainty matrix on all three branches" begin
    rng = StableRNG(987654321)
    Tn, N = 200, 3
    Xs = randn(rng, Tn, N) .* 0.01 .+ 0.0005
    Sigma = cov(Xs)
    P = [1.0 -1.0 0.0; 0.0 0.0 1.0]
    tau = inv(Tn)
    hand = Diagonal(P * (tau * Sigma) * transpose(P))

    # No confidence: Omega is the diagonal of P (tau Sigma) P'.
    @test diag(PortfolioOptimisers.calc_omega(nothing, P, tau * Sigma)) == diag(hand)
    # `bl_preroll` calls `calc_omega` on the unscaled covariance and scales the answer, so
    # the two orders must agree. `calc_omega` is homogeneous of degree one in `sigma`.
    @test diag(tau * PortfolioOptimisers.calc_omega(nothing, P, Sigma)) == diag(hand)

    # Scalar confidence: the 1/v - 1 rescale of Idzorek's method in Walters' closed form.
    for v in (0.1, 0.5, 0.9)
        @test isapprox(diag(PortfolioOptimisers.calc_omega(v, P, tau * Sigma)),
                       (inv(v) - one(v)) .* diag(hand))
    end
    # The direction is the thing a reader gets wrong: a high confidence shrinks Omega and a
    # low one widens it.
    lo = diag(PortfolioOptimisers.calc_omega(0.1, P, tau * Sigma))
    hi = diag(PortfolioOptimisers.calc_omega(0.9, P, tau * Sigma))
    @test all(lo .> hi)

    # Vector confidence: elementwise, and equal to the scalar branch for a constant vector.
    @test diag(PortfolioOptimisers.calc_omega(fill(0.6, 2), P, tau * Sigma)) ==
          diag(PortfolioOptimisers.calc_omega(0.6, P, tau * Sigma))
    vv = [0.25, 0.75]
    @test isapprox(diag(PortfolioOptimisers.calc_omega(vv, P, tau * Sigma)),
                   (inv.(vv) .- one(eltype(vv))) .* diag(hand))

    #=
    Both endpoints are refused, and the bound is strict on purpose. At `v = 1` the docstring
    of `calc_omega` says that `P tau Sigma P' + Omega` is singular for a rank-deficient `P`.
    Two identical views give exactly such a `P`, so build one and show the singularity.
    =#
    Pr = [1.0 -1.0 0.0; 1.0 -1.0 0.0]
    @test rank(Pr) == 1
    A = Pr * (tau * Sigma) * transpose(Pr)
    @test all(iszero, diag(PortfolioOptimisers.calc_omega(1.0, Pr, tau * Sigma)))
    Msing = A + PortfolioOptimisers.calc_omega(1.0, Pr, tau * Sigma)
    @test iszero(det(Msing))
    @test rank(Msing) == 1
    @test_throws SingularException Msing \ [1.0, 1.0]
    # Just inside the bound the same system is only ill-conditioned, never singular.
    for v in (1 - 1e-8, 0.99)
        Mv = A + PortfolioOptimisers.calc_omega(v, Pr, tau * Sigma)
        @test !iszero(det(Mv))
        @test isfinite(cond(Mv))
    end

    # Both endpoints, and just inside each, through the guard itself.
    blv = BlackLittermanViews(; P = P, Q = [0.002, 0.001])
    @test_throws DomainError PortfolioOptimisers.assert_bl_views_conf(fill(0.0, 2), blv)
    @test_throws DomainError PortfolioOptimisers.assert_bl_views_conf(fill(1.0, 2), blv)
    @test isnothing(PortfolioOptimisers.assert_bl_views_conf(fill(1e-12, 2), blv))
    @test isnothing(PortfolioOptimisers.assert_bl_views_conf(fill(1 - 1e-12, 2), blv))

    # The length guards. A vector states one confidence per view and is counted; a scalar is
    # one confidence for every view and is not.
    @test_throws DimensionMismatch PortfolioOptimisers.assert_bl_views_conf([0.5], blv)
    @test isnothing(PortfolioOptimisers.assert_bl_views_conf(0.5, blv))
    @test_throws DimensionMismatch PortfolioOptimisers.assert_bl_views_conf([0.5, 0.5],
                                                                            "A == 0.01")
    @test isnothing(PortfolioOptimisers.assert_bl_views_conf([0.5], "A == 0.01"))
    @test isnothing(PortfolioOptimisers.assert_bl_views_conf([0.5, 0.5],
                                                             ["A == 0.01", "B == 0.02"]))
end

#=
The master equations of `vanilla_posteriors`, against the two closed forms the library
publishes: the inverse-free pair on `vanilla_posteriors` itself, and the inverse-heavy pair
on `prior(::BlackLittermanPrior, …)`. Sweep ticket #535.
=#
@testset "Black-Litterman core: the master equations" begin
    rng = StableRNG(987654321)
    Tn, N = 200, 3
    Xs = randn(rng, Tn, N) .* 0.01 .+ 0.0005
    Sigma = cov(Xs)
    Pi = vec(mean(Xs; dims = 1))
    P = [1.0 -1.0 0.0; 0.0 0.0 1.0]
    Q = [0.002, 0.001]

    for t in (inv(Tn), 0.05, 0.5)
        om = PortfolioOptimisers.calc_omega(nothing, P, t * Sigma)
        pmu, psig = PortfolioOptimisers.vanilla_posteriors(t, Pi, Sigma, om, P, Q)
        ts = t * Sigma
        gain = ts * transpose(P) * inv(P * ts * transpose(P) + om)
        @test isapprox(pmu, Pi + gain * (Q - P * Pi))
        @test isapprox(psig, Sigma + ts - gain * P * ts)
        # The Woodbury form the `prior` docstring states. The two agree to 3.3e-19 on the
        # mean and 1.4e-20 on the covariance.
        minv = inv(inv(ts) + transpose(P) * inv(om) * P)
        @test isapprox(pmu, minv * (inv(ts) * Pi + transpose(P) * inv(om) * Q))
        @test isapprox(psig, Sigma + minv)
    end

    # A view that repeats the prior is a null update: the residual q - P Pi is zero, so the
    # posterior mean is the prior mean to machine precision. Measured at 0.0.
    tau = inv(Tn)
    om = PortfolioOptimisers.calc_omega(nothing, P, tau * Sigma)
    nullmu, _ = PortfolioOptimisers.vanilla_posteriors(tau, Pi, Sigma, om, P, P * Pi)
    @test isapprox(nullmu, Pi; atol = 1e-16)

    #=
    tau stands in both closed forms, but it cancels out of the posterior mean once Omega
    carries it too — `calc_omega` is homogeneous of degree one in the covariance it reads,
    and `bl_preroll` scales its answer by tau. It does not cancel out of the covariance.
    =#
    for vc in (nothing, 0.4, [0.25, 0.75])
        mus = map((inv(Tn), 0.05, 0.5)) do t
            omt = t * PortfolioOptimisers.calc_omega(vc, P, Sigma)
            return first(PortfolioOptimisers.vanilla_posteriors(t, Pi, Sigma, omt, P, Q))
        end
        @test isapprox(mus[1], mus[2]; atol = 1e-16)
        @test isapprox(mus[1], mus[3]; atol = 1e-16)
    end
    excess = map((0.001, 0.01, 0.1)) do t
        omt = t * PortfolioOptimisers.calc_omega(nothing, P, Sigma)
        return tr(last(PortfolioOptimisers.vanilla_posteriors(t, Pi, Sigma, omt, P, Q)) -
                  Sigma)
    end
    @test all(>(0), excess)
    @test isapprox(excess[2] / excess[1], 10; rtol = 1e-8)
    @test isapprox(excess[3] / excess[2], 10; rtol = 1e-8)
end

#=
The risk-free rate is added once, the default tau is read from the observation count, and
the two routes into `Lc_BLV` give the same rows. Sweep ticket #535.
=#
@testset "Black-Litterman core: the rate, tau, and the two view routes" begin
    lce = LinearConstraintEstimator(; val = ["$(rd.nx[1]) == 0.002"])
    m0 = prior(BlackLittermanPrior(; views = lce, sets = sets, rf = 0.0), rd).mu
    m3 = prior(BlackLittermanPrior(; views = lce, sets = sets, rf = 0.03), rd).mu
    # Not zero times, and not twice.
    @test isapprox(m3 - m0, fill(0.03, length(m0)); atol = 1e-16)
    # `apply_rf` shifts and does nothing else. Nothing in the library subtracts the rate:
    # ADR 0063's amendment puts a missing level on the prior mean instead of taking one off.
    @test isapprox(PortfolioOptimisers.apply_rf(0.03, m0), m0 .+ 0.03; atol = 1e-16)
    @test isapprox(PortfolioOptimisers.apply_rf(0.0, m0), m0; atol = 1e-16)

    # The default tau is 1/T, read from the observation count `bl_preroll` is handed, and a
    # user-supplied tau wins.
    psig = prior(EmpiricalPrior(), rd).sigma
    for Tn in (50, 120, 252)
        r = PortfolioOptimisers.bl_preroll(lce, sets, nothing, psig, nothing, Tn, Float64,
                                           false)
        @test r.tau == inv(Tn)
    end
    @test PortfolioOptimisers.bl_preroll(lce, sets, nothing, psig, 0.07, 252, Float64,
                                         false).tau == 0.07
    @test_throws DomainError BlackLittermanPrior(; views = lce, sets = sets, tau = 0.0)
    @test_throws DomainError BlackLittermanPrior(; views = lce, sets = sets, tau = -0.1)

    # `Lc_BLV` admits a `LinearConstraintEstimator` or a ready-made `BlackLittermanViews`,
    # and the two routes assemble the same rows and reach the same posterior.
    lce2 = LinearConstraintEstimator(;
                                     val = ["$(rd.nx[1]) + $(rd.nx[2]) == 0.05",
                                            "$(rd.nx[3]) == 0.02"])
    via_eqn = black_litterman_views(lce2, sets)
    ready = BlackLittermanViews(; P = collect(via_eqn.P), Q = copy(via_eqn.Q))
    via_res = black_litterman_views(ready, sets)
    @test collect(via_eqn.P) == collect(via_res.P)
    @test via_eqn.Q == via_res.Q
    @test prior(BlackLittermanPrior(; views = lce2, sets = sets), rd).mu ==
          prior(BlackLittermanPrior(; views = ready), rd).mu

    #=
    A scalar `views_conf` is one confidence for every view. The equation-shaped routes have
    always accepted it over any number of views, but the `BlackLittermanViews` method
    counted a scalar's `length` of 1 against the views and refused `views_conf = 0.4` over
    two precomputed views while accepting `[0.4, 0.4]`, which reaches `calc_omega` as the
    same number. Fixed under #535: only a vector is counted.
    =#
    @test isa(BlackLittermanPrior(; views = ready, views_conf = 0.4), BlackLittermanPrior)
    a = prior(BlackLittermanPrior(; views = ready, views_conf = 0.4), rd).mu
    b = prior(BlackLittermanPrior(; views = ready, views_conf = [0.4, 0.4]), rd).mu
    c = prior(BlackLittermanPrior(; views = lce2, sets = sets, views_conf = 0.4), rd).mu
    @test a == b
    @test a == c
    # A vector is still counted, on both routes.
    @test_throws DimensionMismatch BlackLittermanPrior(; views = ready,
                                                       views_conf = [0.4, 0.4, 0.4])
end

#=
The views the caller wrote: the constructor guards, the exclusion of a view that resolves
no name, and the two guards that stand between a view and the wrong axis. Sweep ticket #535.
=#
@testset "Black-Litterman core: the views the caller wrote" begin
    # The `BlackLittermanViews` guards.
    @test_throws IsEmptyError BlackLittermanViews(; P = zeros(0, 3), Q = Float64[])
    @test_throws IsEmptyError BlackLittermanViews(; P = ones(1, 3), Q = Float64[])
    @test_throws DimensionMismatch BlackLittermanViews(; P = ones(2, 3), Q = [0.01])
    @test_throws IsEmptyError BlackLittermanViews(; P = ones(2, 3), Q = [0.01, 0.02],
                                                  excl = Int[])
    @test_throws DimensionMismatch BlackLittermanViews(; P = ones(2, 3), Q = [0.01, 0.02],
                                                       excl = [1, 2, 3])
    @test isa(BlackLittermanViews(; P = ones(2, 3), Q = [0.01, 0.02], excl = [1, 2]),
              BlackLittermanViews)

    bsets = UniverseSets(; xkey = "nx", dict = Dict("nx" => ["A", "B", "C"]))
    #=
    A view that resolves no name is dropped, not refused. Its index joins `excl`, the
    remaining rows keep the order the caller wrote them in, and `remove_excl_views` drops
    the matching entry of a per-view confidence vector.
    =#
    blv = PortfolioOptimisers.get_black_litterman_views(parse_equation(["A == 0.01",
                                                                        "ZZZ == 0.02",
                                                                        "C == 0.03"]),
                                                        bsets)
    @test size(blv.P) == (2, 3)
    @test collect(blv.P) == [1.0 0.0 0.0; 0.0 0.0 1.0]
    @test blv.Q == [0.01, 0.03]
    @test blv.excl == [2]
    @test collect(PortfolioOptimisers.remove_excl_views([0.1, 0.2, 0.3], blv.excl)) ==
          [0.1, 0.3]
    # `strict = true` refuses the same input rather than dropping it.
    @test_throws ArgumentError PortfolioOptimisers.get_black_litterman_views(parse_equation(["ZZZ == 0.02"]),
                                                                             bsets;
                                                                             strict = true)

    #=
    When every view is dropped there is no row left, and `get_black_litterman_views`
    answers `nothing`. `bl_preroll` destructured that and the caller read `FieldError: type
    Nothing has no field P`, which names neither the views nor the universe they failed
    against. Fixed under #535.
    =#
    @test isnothing(PortfolioOptimisers.get_black_litterman_views(parse_equation(["YY == 0.01",
                                                                                  "ZZ == 0.02"]),
                                                                  bsets))
    Xb = randn(StableRNG(123456789), 60, 3) .* 0.01
    gone = BlackLittermanPrior(; views = LinearConstraintEstimator(; val = ["ZZ == 0.01"]),
                               sets = bsets)
    @test_throws IsNothingError prior(gone, Xb)

    # The three `remove_excl_views` methods.
    @test PortfolioOptimisers.remove_excl_views(0.4, [2]) == 0.4
    @test isnothing(PortfolioOptimisers.remove_excl_views(nothing, [2]))
    @test PortfolioOptimisers.remove_excl_views([0.1, 0.2, 0.3], nothing) == [0.1, 0.2, 0.3]
    @test collect(PortfolioOptimisers.remove_excl_views([0.1, 0.2, 0.3], [1, 3])) == [0.2]
    @test collect(PortfolioOptimisers.remove_excl_views([0.1, 0.2, 0.3, 0.4], [2])) ==
          [0.1, 0.3, 0.4]
    @test isempty(PortfolioOptimisers.remove_excl_views([0.1, 0.2], [1, 2]))

    # `bl_preroll`'s axis selector names a declared axis a view can land on, and nothing
    # else. A third axis is refused before anything is resolved.
    ready = BlackLittermanViews(; P = ones(1, 3), Q = [0.01])
    Sb = cov(Xb)
    for ax in (:xkey, :tfkey)
        @test isa(PortfolioOptimisers.bl_preroll(ready, nothing, nothing, Sb, nothing, 60,
                                                 Float64, false, ax), NamedTuple)
    end
    @test_throws DomainError PortfolioOptimisers.bl_preroll(ready, nothing, nothing, Sb,
                                                            nothing, 60, Float64, false,
                                                            :zkey)

    # An asset-axis view against a factor-axis covariance. The message names both widths.
    wide = BlackLittermanViews(; P = ones(1, 5), Q = [0.01])
    ew = try
        PortfolioOptimisers.bl_preroll(wide, nothing, nothing, Sb, nothing, 60, Float64,
                                       false)
    catch e
        e
    end
    @test isa(ew, DimensionMismatch)
    @test occursin("size(P, 2) => 5", ew.msg)
    @test occursin("size(prior_sigma, 1) => 3", ew.msg)

    # The universe and the returns matrix must agree, and the message names both counts.
    big = UniverseSets(; xkey = "nx", dict = Dict("nx" => ["A", "B", "C", "D"]))
    eb = try
        prior(BlackLittermanPrior(;
                                  views = LinearConstraintEstimator(; val = ["A == 0.01"]),
                                  sets = big), Xb)
    catch e
        e
    end
    @test isa(eb, DimensionMismatch)
    @test occursin("(4)", eb.msg)
    @test occursin("(3)", eb.msg)
end

@testset "The carrier records the original returns matrix" begin
    #=
    Three estimators overwrite `X` with the reconstruction `F * transpose(M) .+
    transpose(b)`, so on their carriers `X` is a posterior matrix rather than the returns
    the caller supplied. `o_X` holds the returns the caller supplied, and `original_X` is
    the read that is always a matrix. See ADR 0046's 2026-08-11 amendment.
    =#
    X, F = rd.X, rd.F
    a_view = LinearConstraintEstimator(; val = "$(rd.nx[1]) == 0.002")
    f_view = LinearConstraintEstimator(; val = "$(rd.nf[1]) == 0.001")

    # The three producers set it, and every other route leaves it `nothing`.
    fp = prior(FactorPrior(), X, F)
    fbl = prior(FactorBlackLittermanPrior(; sets = xfsets, views = f_view), X, F)
    abl = prior(AugmentedBlackLittermanPrior(; sets = afsets, a_views = a_view,
                                             f_views = f_view), X, F)
    for pr in (fp, fbl, abl)
        @test pr.o_X === X
        @test pr.original_X === X
        @test pr.X !== X
        @test size(pr.o_X) == size(pr.X)
    end
    bl = prior(BlackLittermanPrior(; sets = sets, views = a_view), X, F)
    for pr in (prior(EmpiricalPrior(), X), bl)
        @test isnothing(pr.o_X)
        @test pr.original_X === pr.X
    end

    # The nested factor block is fit on factors, so it has no original of its own.
    @test isnothing(fp.fpr.o_X)

    # `HighOrderPrior` declares neither name and reads both through its `forward(pr)`
    # block, so one declaration covers both carriers.
    @test !hasfield(HighOrderPrior, :o_X)
    hop = HighOrderPrior(; pr = fp)
    @test hop.o_X === X
    @test hop.original_X === X

    #=
    The two pooling estimators rebuild their carrier by explicit destructure rather than
    through `forward_prior`, so they are the sites that would drop the field silently.
    =#
    epe = EntropyPoolingPrior(; pe = FactorPrior(), sets = xfsets, mu_views = a_view)
    epp = prior(epe, X, F)
    @test epp.o_X === X
    @test epp.X !== X
    opp = prior(OpinionPoolingPrior(; pes = [epe, epe], pe2 = FactorPrior()), X, F)
    @test opp.o_X === X
    @test opp.X !== X

    # `forward_prior` needs no plumbing: it enumerates `fieldnames`, so the field travels
    # unchanged, and it gains no binding to `X`.
    @test PortfolioOptimisers.forward_prior(fp; mu = fp.mu .+ 0.001).o_X === X
    @test haskey(PortfolioOptimisers.prior_field_values(fp), :o_X)

    # A view slices it on the asset axis, exactly as it slices `X`.
    i = [1, 3, 5, 7]
    v = PortfolioOptimisers.port_opt_view(fp, i)
    @test v.o_X == X[:, i]
    @test v.original_X == X[:, i]
    @test size(v.o_X) == size(v.X)

    #=
    Three guards, all O(1): no matrix is ever compared by value.
    =#
    base = (mu = fp.mu, sigma = fp.sigma, rr = fp.rr, fpr = fp.fpr)
    # `o_X === X` says nothing, and would be a second encoding of "no reconstruction".
    @test_throws ArgumentError LowOrderPrior(; X = fp.X, o_X = fp.X, base...)
    #=
    The guard is an identity test and not an equality test, and the two calls read
    identically at a call site. A copy carrying the same values is admitted, because it is
    a matrix of its own that a later change to `X` cannot follow. What the guard rejects is
    the carrier with no original distinct from the one it asserts.
    =#
    twin = copy(fp.X)
    @test twin == fp.X
    @test twin !== fp.X
    @test isa(LowOrderPrior(; X = fp.X, o_X = twin, base...), LowOrderPrior)
    # The original describes the same observations and the same assets.
    @test_throws DimensionMismatch LowOrderPrior(; X = fp.X, o_X = X[1:(end - 1), :],
                                                 base...)
    #=
    `rr` is required whenever `o_X` is set. This is a present-tense constraint rather than
    a law of the domain: a future prior that transforms `X` without a regression is the
    case that relaxes it. ADR 0046's amendment carries the reasoning.
    =#
    @test_throws IsNothingError LowOrderPrior(; X = fp.X, o_X = X, mu = fp.mu,
                                              sigma = fp.sigma)
    # The legal shape passes all three.
    @test isa(LowOrderPrior(; X = fp.X, o_X = X, base...), LowOrderPrior)
end

@testset "The risk-free rate goes on the prior mean, and only where there is a premium" begin
    # ADR 0063 as amended, and issue #570. A Black-Litterman update blends the prior mean
    # against the view returns by forming `Q - P * mu`, so the prior mean must be on the
    # scale `Q` is written on, which is a total return. Nothing converts it away and back.
    #
    # Which members read `rf`, and what each therefore obeys:
    #
    #   * `BlackLittermanPrior` and `BayesianBlackLittermanPrior` have no equilibrium
    #     branch and so have nothing to convert. They add the rate to the posterior asset
    #     mean, last, and `mu(r) == mu(0) .+ r` exactly.
    #   * `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` read the field only
    #     where `l` is set, to convert the equilibrium risk premium. Where `l` is `nothing`
    #     the field does not reach the answer at all, and `mu(r) == mu(0)` exactly.
    rf = 0.02
    bl_v = LinearConstraintEstimator(;
                                     val = ["$(rd.nx[1]) == 0.003", "$(rd.nx[2]) == 0.002"])
    f_v = LinearConstraintEstimator(; val = ["$(rd.nf[1]) == 0.004"])

    # The two members that add the rate to the answer.
    shifted(r) = [BlackLittermanPrior(; sets = sets, views = bl_v, rf = r),
                  BayesianBlackLittermanPrior(; pe = FactorPrior(), sets = xfsets,
                                              views = f_v, rf = r)]
    for (pe0, pe1) in zip(shifted(0.0), shifted(rf))
        pr0 = prior(pe0, rd)
        pr1 = prior(pe1, rd)
        # The shift is exact, not approximate: the rate is added at one site, last.
        @test pr1.mu == pr0.mu .+ rf
        # It is added to the asset mean, so the factor block never carries it.
        # `BlackLittermanPrior` here wraps no factor prior and reports no block.
        if !isnothing(pr0.fpr)
            @test pr1.fpr.mu == pr0.fpr.mu
        end
    end

    # The two members with an equilibrium branch, on the branch where they have none. The
    # field is inert there: the wrapped prior mean is a total return already, and nothing
    # is added afterwards.
    inert(r) = [FactorBlackLittermanPrior(; sets = xfsets, views = f_v, rf = r),
                AugmentedBlackLittermanPrior(; sets = afsets, a_views = bl_v, f_views = f_v,
                                             rf = r)]
    for (pe0, pe1) in zip(inert(0.0), inert(rf))
        pr0 = prior(pe0, rd)
        pr1 = prior(pe1, rd)
        @test pr1.mu == pr0.mu
        @test pr1.fpr.mu == pr0.fpr.mu
    end

    # Isolation. The inner estimator applies its own rate; the outer neither undoes it nor
    # repeats it, and shifts only by its own.
    inner = BlackLittermanPrior(; sets = sets, views = bl_v, rf = 0.01)
    outer0 = prior(BlackLittermanPrior(; pe = inner, sets = sets, views = bl_v, rf = 0.0),
                   rd)
    outer1 = prior(BlackLittermanPrior(; pe = inner, sets = sets, views = bl_v, rf = rf),
                   rd)
    @test outer1.mu == outer0.mu .+ rf
    # The falsification witness: the inner rate does reach the result, so the assertion
    # above is about a value that is there to be undone.
    bare = BlackLittermanPrior(; sets = sets, views = bl_v, rf = 0.0)
    @test prior(BlackLittermanPrior(; pe = bare, sets = sets, views = bl_v), rd).mu !=
          outer0.mu
end

@testset "With `l` set the rate is blended, and a relative view makes the blend exact" begin
    # The equilibrium branch converts a risk premium, so the rate reaches the answer
    # *through* the update. Writing `G` for the update gain, the prior mean gains `r * 1`
    # and the posterior therefore gains `(I - G * P) * r * 1`.
    #
    # Write every view as a *difference* between two names and each row of `P` sums to
    # zero, so `P * 1 == 0`, `G * P * 1 == 0`, and the whole shift survives the blend. That
    # is the sharp case, and it is exact.
    rf = 0.02
    a_rel = LinearConstraintEstimator(; val = ["$(rd.nx[1]) - $(rd.nx[2]) == 0.001"])
    f_rel = LinearConstraintEstimator(; val = ["$(rd.nf[1]) - $(rd.nf[2]) == 0.001"])

    # The augmented member stacks assets over factors and blends one system, so both halves
    # gain the rate whole.
    ma(r) = AugmentedBlackLittermanPrior(; sets = afsets, a_views = a_rel, f_views = f_rel,
                                         rf = r, l = 2.0)
    pr0 = prior(ma(0.0), rd)
    pr1 = prior(ma(rf), rd)
    @test isapprox(pr1.mu, pr0.mu .+ rf)
    @test isapprox(pr1.fpr.mu, pr0.fpr.mu .+ rf)

    # The factor member updates the factors alone and then lifts, so the factor half gains
    # the rate whole and the assets receive it through the loadings: `mu(r) == mu(0) + r*s`
    # for the row sums `s` of `M`. The loadings do not sum to one, so this is a live
    # difference rather than a plain shift.
    mf(r) = FactorBlackLittermanPrior(; sets = xfsets, views = f_rel, rf = r, l = 2.0)
    fr0 = prior(mf(0.0), rd)
    fr1 = prior(mf(rf), rd)
    s = vec(sum(fr0.rr.M; dims = 2))
    @test isapprox(fr1.mu, fr0.mu .+ rf .* s)
    @test isapprox(fr1.fpr.mu, fr0.fpr.mu .+ rf)
    @test !isapprox(s, ones(length(s)))
    @test !isapprox(fr1.mu, fr0.mu .+ rf)
    # The carrier stays internally consistent across the axis crossing, and carries no `rf`
    # term of its own any more: the rate is inside `fpr.mu`.
    @test isapprox(fr1.mu, fr1.rr.M * fr1.fpr.mu + fr1.rr.b)
end

@testset "An absolute view feels the rate through the blend" begin
    # The complement of the testset above, and the reason `rf * (1 - s)` was the wrong
    # closed form for the arithmetic this replaced (#571): that expression is the no-view
    # limit `G = 0`, and no estimator reaches it, because one with no views cannot be
    # built. Where a view row does not sum to zero the gain absorbs part of the shift, so
    # the answer moves by neither `r` nor `r * s`.
    rf = 0.02
    bl_v = LinearConstraintEstimator(;
                                     val = ["$(rd.nx[1]) == 0.003", "$(rd.nx[2]) == 0.002"])
    f_v = LinearConstraintEstimator(; val = ["$(rd.nf[1]) == 0.004"])
    mkf(r) = FactorBlackLittermanPrior(; sets = xfsets, views = f_v, rf = r, l = 2.0)
    mka(r) = AugmentedBlackLittermanPrior(; sets = afsets, a_views = bl_v, f_views = f_v,
                                          rf = r, l = 2.0)
    for mk in (mkf, mka)
        pr0 = prior(mk(0.0), rd)
        pr1 = prior(mk(rf), rd)
        pr2 = prior(mk(2rf), rd)
        @test !isapprox(pr1.mu, pr0.mu .+ rf)
        @test !isapprox(pr1.mu, pr0.mu)
        # Linear in the rate, whatever the views: the update is affine in the prior mean.
        @test isapprox((pr1.mu .- pr0.mu) ./ rf, (pr2.mu .- pr0.mu) ./ (2rf))
    end
end

@testset "`apply_rf` and `equilibrium_mu` are the only sites" begin
    # A text census over the sources, so the convention is enforceable rather than
    # documented. It loads no package and reads no data.
    function code_lines(path)
        in_doc = false
        out = String[]
        for line in eachline(path)
            s = strip(line)
            if s == "\"\"\""
                in_doc = !in_doc
                continue
            end
            (in_doc || startswith(s, "#")) && continue
            push!(out, line)
        end
        return out
    end
    src = joinpath(@__DIR__, "..", "src")
    # ADR 0063 as amended: every member reads the field exactly once, through `apply_rf`,
    # and nothing subtracts it. The two with no equilibrium branch add it to the posterior
    # asset mean; the two with one convert the equilibrium mean before the update.
    for file in ["06_BlackLittermanPrior.jl", "07_BayesianBlackLittermanPrior.jl",
                 "08_FactorBlackLittermanPrior.jl", "09_AugmentedBlackLittermanPrior.jl"]
        lines = code_lines(joinpath(src, "13_Prior", file))
        reads = filter(l -> occursin("pe.rf", l), lines)
        @test length(reads) == 1
        # The one read goes through the verb; none is hand-written arithmetic.
        @test all(l -> occursin("apply_rf(pe.rf", l), reads)
        # No member writes the equilibrium mean itself.
        @test isempty(filter(l -> occursin("pe.l *", l), lines))
    end
    # The intercept is applied once too, and `AugmentedBlackLittermanPrior` is where it was
    # applied twice (#570). It goes into the prior stack now, so the only `rr.b` the body
    # reads is the one the equilibrium branch adds.
    aug = code_lines(joinpath(src, "13_Prior", "09_AugmentedBlackLittermanPrior.jl"))
    @test count(l -> occursin(r"\bb\b", l) && occursin("vcat(b,", l), aug) == 1
    # The kernel carries no rate of its own.
    kernel = code_lines(joinpath(src, "13_Prior", "06_BlackLittermanPrior.jl"))
    decl = only(filter(l -> occursin("function vanilla_posteriors(", l), kernel))
    @test !occursin("rf", decl)
    # The one verb is declared once, in the file that owns the family's kernel, and the
    # subtraction it used to pair with is gone from the library entirely.
    @test count(l -> occursin("function apply_rf(", l), kernel) == 1
    @test !isdefined(PortfolioOptimisers, :remove_rf)
    # `equilibrium_mu` owns `l * sigma * w` and its equal-weight fallback.
    owner = code_lines(joinpath(src, "08_Moments", "17_EquilibriumExpectedReturns.jl"))
    @test count(l -> occursin("function equilibrium_mu(", l), owner) == 2
end

@testset "A view of a high order carrier takes each co-moment branch" begin
    #=
    `port_opt_view(::HighOrderPrior, i)` is the one place in `01_Base_Prior.jl` where a
    wrong index gives a well-shaped wrong answer, so the index is checked against a
    hand-computed one rather than against the generator that produced it. `kt` is
    `N^2 x N^2` over ordered pairs of assets, laid out column-major, so the pair `(a, b)`
    sits at `a + (b - 1) * N`. For `N = 3` and the subset `[1, 3]` that is `[1, 3, 7, 9]`.
    =#
    X = [0.01 0.02 0.03; 0.04 0.05 0.06; 0.07 0.08 0.09; 0.10 0.11 0.12]
    mu = [0.02, 0.03, 0.04]
    sigma = [0.001 0.0 0.0; 0.0 0.001 0.0; 0.0 0.0 0.001]
    lo = LowOrderPrior(; X = X, mu = mu, sigma = sigma)
    i = [1, 3]
    idx = [a + (b - 1) * 3 for b in i for a in i]
    @test idx == [1, 3, 7, 9]
    @test collect(PortfolioOptimisers.fourth_moment_index_generator(3, i)) == idx

    kt = reshape(collect(1.0:81.0), 9, 9)
    D2 = PortfolioOptimisers.duplication_matrix(3)
    L2 = PortfolioOptimisers.elimination_matrix(3)
    S2 = PortfolioOptimisers.summation_matrix(3)
    sk = reshape(collect(1.0:27.0), 3, 9) / 100
    V = [0.5 0.1 0.2; 0.1 0.6 0.3; 0.2 0.3 0.7]
    skmp = Coskewness().mp

    # Every moment field present: the co-moments are cut by `idx`, and `V` is rebuilt.
    full = HighOrderPrior(; pr = lo, kt = kt, D2 = D2, L2 = L2, S2 = S2, sk = sk, V = V,
                          skmp = skmp)
    v = PortfolioOptimisers.port_opt_view(full, i)
    @test Matrix(v.kt) == kt[idx, idx]
    @test Matrix(v.sk) == sk[i, idx]
    # `V` is a spectral quantity of `sk`, so the submatrix of `V` is not the `V` of the
    # submatrix. It is rebuilt from the cut `sk` and the cut returns.
    @test Matrix(v.V) != V[i, i]
    @test Matrix(v.V) ==
          PortfolioOptimisers.negative_spectral_coskewness(sk[i, idx], X[:, i], skmp)
    # `D2`, `L2` and `S2` are rebuilt at the subproblem's asset count, never cut.
    @test size(v.D2) == (4, 3)
    @test size(v.L2) == size(v.S2) == (3, 4)
    @test v.skmp === skmp
    # The view is a carrier, so it passed every `@argcheck` of the constructor.
    @test isa(v, HighOrderPrior)

    # Issue #543: the constructor does not bind `skmp` to `sk`, so a hand-built carrier
    # can hold `sk` and `V` with `skmp === nothing`. `nothing` means "no processing"
    # everywhere else in the chain, so the slice rebuilds `V` with no processing step
    # instead of raising a `MethodError`.
    no_skmp = HighOrderPrior(; pr = lo, kt = kt, D2 = D2, L2 = L2, S2 = S2, sk = sk, V = V)
    @test isnothing(no_skmp.skmp)
    v_no_skmp = PortfolioOptimisers.port_opt_view(no_skmp, i)
    @test isa(v_no_skmp, HighOrderPrior)
    @test Matrix(v_no_skmp.sk) == sk[i, idx]
    @test Matrix(v_no_skmp.V) ==
          PortfolioOptimisers.negative_spectral_coskewness(sk[i, idx], X[:, i], nothing)
    @test isnothing(v_no_skmp.skmp)

    # `S2` without `D2`: the second branch takes `L2` and `S2` alone.
    no_D2 = HighOrderPrior(; pr = lo, kt = kt, L2 = L2, S2 = S2)
    v_no_D2 = PortfolioOptimisers.port_opt_view(no_D2, i)
    @test isnothing(v_no_D2.D2)
    @test size(v_no_D2.L2) == size(v_no_D2.S2) == (3, 4)
    # No `sk`, so nothing is rebuilt from it either.
    @test isnothing(v_no_D2.sk)
    @test isnothing(v_no_D2.V)

    # Neither: a carrier holding no co-moment at all still slices, and keeps every
    # `nothing` a `nothing`.
    bare = HighOrderPrior(; pr = lo)
    v_bare = PortfolioOptimisers.port_opt_view(bare, i)
    @test isnothing(v_bare.kt)
    @test isnothing(v_bare.D2)
    @test isnothing(v_bare.L2)
    @test isnothing(v_bare.S2)
    @test isnothing(v_bare.sk)
    @test isnothing(v_bare.V)
    @test v_bare.mu == mu[i]
end

@testset "A prior with nothing on the asset axis passes through a view unchanged" begin
    #=
    An estimator carries a recipe rather than data on an asset axis, so a subproblem refits
    it on its own universe. A vector arrives already resolved per subproblem, so the entry
    has been chosen by the time the view is taken. Both are returned by identity.
    =#
    i = [1, 3]
    pe = EmpiricalPrior()
    @test PortfolioOptimisers.port_opt_view(pe, i) === pe
    @test isnothing(PortfolioOptimisers.port_opt_view(nothing, i))
    # The second positional argument is the asset index, and neither method reads it, so a
    # value that indexes nothing is accepted too.
    @test PortfolioOptimisers.port_opt_view(pe, nothing) === pe
    # Further positional and keyword arguments are accepted and ignored.
    @test PortfolioOptimisers.port_opt_view(pe, i, :ignored; also = :ignored) === pe

    X = [0.01 0.02 0.03; 0.04 0.05 0.06; 0.07 0.08 0.09; 0.10 0.11 0.12]
    lo = LowOrderPrior(; X = X, mu = [0.02, 0.03, 0.04],
                       sigma = [0.001 0.0 0.0; 0.0 0.001 0.0; 0.0 0.0 0.001])
    prs = [lo, lo]
    @test PortfolioOptimisers.port_opt_view(prs, i) === prs
    pes = [pe, EmpiricalPrior(; horizon = 252)]
    @test PortfolioOptimisers.port_opt_view(pes, i) === pes
    @test PortfolioOptimisers.port_opt_view(pes, i, :ignored; also = :ignored) === pes
end

@testset "The property pool is the union of both carriers' fields, in a stable order" begin
    #=
    `prior_result_property_pool` is the candidate pool an `@pprop` field name is checked
    against, so a downstream reflection reads it positionally. The order is therefore part
    of the contract and is written out here rather than derived a second way.
    =#
    pool = PortfolioOptimisers.prior_result_property_pool()
    @test pool ==
          [:X, :o_X, :mu, :sigma, :chol, :w, :ens, :kld, :ow, :rr, :fpr, :pnl, :pr, :kt,
           :D2, :L2, :S2, :sk, :V, :skmp]
    @test allunique(pool)
    # `fpr` is a field of both carriers, so the concatenation is not already unique.
    @test :fpr in fieldnames(LowOrderPrior)
    @test :fpr in fieldnames(HighOrderPrior)
    @test length(pool) ==
          length(fieldnames(LowOrderPrior)) + length(fieldnames(HighOrderPrior)) - 1
    # A fresh call rebuilds the vector, and rebuilds it the same way.
    @test pool == PortfolioOptimisers.prior_result_property_pool()
    @test pool !== PortfolioOptimisers.prior_result_property_pool()
end

#=
The residual comoments of `src/13_Prior/14_HighOrderFactorPriorEstimator.jl`, checked against
hand-built matrices rather than read. Sweep ticket #534. Riskfolio-Lib carries the reference
implementation of both, and this library's branch chain matches it entry for entry; what differs
is that the reference builds the systematic covariance itself from the loadings, while this
library takes it as an argument and asks the caller to remove the residual block first.
=#
@testset "Residual coskewness and cokurtosis" begin
    PO = PortfolioOptimisers
    me = SimpleExpectedReturns()
    mean_ = PO.Statistics.mean

    # `coskewness_residuals` writes `E[e_i^3]` at `(i, (i - 1) * N + i)` and zero elsewhere.
    # The index arithmetic is the whole function, so it is checked against a hand-built
    # matrix at two asset counts.
    for N in (2, 3)
        rng = StableRNG(987)
        X = randn(rng, 5000, N) .* reshape([0.5, 1.5, 2.5][1:N], 1, N)
        sk = PO.coskewness_residuals(X, me)
        H = zeros(N, N^2)
        for i in 1:N
            H[i, (i - 1) * N + i] = mean_(X[:, i] .^ 3)
        end
        @test size(sk) == (N, N^2)
        @test Array(sk) ≈ H
        @test SparseArrays.nnz(sk) == N
    end

    #=
    Nothing is demeaned. The zero-mean assumption belongs to the factor model, so residuals
    with a mean of their own give the RAW third moment. For a unit-variance zero-mean `e`
    shifted by `c`, that is `c^3 + 3c`: `[0.5, -1.0]` predicts `[1.625, -4.0]` and the sample
    of 100000 draws lands on `[1.6072, -3.9911]`. The central third moment is zero, and the
    demeaned sample gives `[-0.0133, 0.0014]` instead.
    =#
    N = 2
    E = randn(StableRNG(7), 100_000, N)
    c = [0.5, -1.0]
    raw = Array(PO.coskewness_residuals(E .+ reshape(c, 1, N), me))
    central = Array(PO.coskewness_residuals(E, me))
    raw_entries = [raw[i, (i - 1) * N + i] for i in 1:N]
    central_entries = [central[i, (i - 1) * N + i] for i in 1:N]
    @test isapprox(raw_entries, c .^ 3 .+ 3 .* c; rtol = 5e-2)
    @test all(abs.(central_entries) .< 5e-2)
    @test !isapprox(raw_entries, central_entries; rtol = 1e-1)

    #=
    `cokurtosis_residuals` runs a `FLoops.@floop` over the `N^2` columns. Each column writes
    only its own entries, so the executor changes the visit order and nothing else: the
    threaded default and the sequential executor are BIT-identical, not merely close.
    =#
    N = 2
    Er = randn(StableRNG(11), 20_000, N) .* reshape([0.7, 1.3], 1, N)
    sig = [0.04 0.01; 0.01 0.09]
    kt_thr = PO.cokurtosis_residuals(sig, Er, me, FLoops.ThreadedEx())
    kt_seq = PO.cokurtosis_residuals(sig, Er, me, FLoops.SequentialEx())
    @test kt_thr == kt_seq
    @test PO.cokurtosis_residuals(sig, Er, me) == kt_seq
    @test kt_thr == transpose(kt_thr)
    @test size(kt_thr) == (N^2, N^2)

    #=
    With a zero systematic covariance the closed form collapses to the population cokurtosis
    of an independent residual set: `E[e_i^4]` on the four-equal pattern, `E[e_i^2] E[e_k^2]`
    on the two-pair pattern, and zero elsewhere. That makes the check exact rather than
    statistical.
    =#
    e2 = vec(mean_(Er .^ 2; dims = 1))
    e4 = vec(mean_(Er .^ 4; dims = 1))
    Pop = zeros(N^2, N^2)
    for i in 1:N, k in 1:N, j in 1:N, l in 1:N
        s = sort([i, k, j, l])
        Pop[(i - 1) * N + k, (j - 1) * N + l] = if s[1] == s[4]
            e4[s[1]]
        elseif s[1] == s[2] && s[3] == s[4]
            e2[s[1]] * e2[s[3]]
        else
            0.0
        end
    end
    @test PO.cokurtosis_residuals(zeros(N, N), Er, me) == Pop

    #=
    The docstring used to say that every pattern with a lone index is zero. It is not: only
    the pattern whose four indices are ALL DISTINCT vanishes. A pair with two singles gives
    `e2[a] * sigma[b, c]`, which is what the reference implementation gives too. Sweep ticket
    #534 moved the documentation, not the code.
    =#
    N4 = 4
    E4 = randn(StableRNG(99), 20_000, N4) .* reshape([0.5, 0.8, 1.2, 0.9], 1, N4)
    S4 = [1.0 0.5 0.3 0.2; 0.5 1.2 0.4 0.1; 0.3 0.4 0.9 0.6; 0.2 0.1 0.6 1.4]
    k4 = PO.cokurtosis_residuals(S4, E4, me)
    e2_4 = vec(mean_(E4 .^ 2; dims = 1))
    e4_4 = vec(mean_(E4 .^ 4; dims = 1))
    # All four distinct: (i, k, j, l) = (1, 2, 3, 4).
    @test k4[(1 - 1) * N4 + 2, (3 - 1) * N4 + 4] == 0
    # One pair and two singles: (1, 3, 1, 4) gives `e2[1] * S4[3, 4]`, not zero.
    @test k4[(1 - 1) * N4 + 3, (1 - 1) * N4 + 4] == e2_4[1] * S4[3, 4]
    @test !iszero(k4[(1 - 1) * N4 + 3, (1 - 1) * N4 + 4])
    # Three equal and one lone: (1, 1, 1, 2) gives `3 * e2[1] * S4[1, 2]`.
    @test k4[(1 - 1) * N4 + 1, (1 - 1) * N4 + 2] == 3 * e2_4[1] * S4[1, 2]
    # Two pairs: (1, 2, 1, 2) gives `e2[2] S[1, 1] + e2[1] S[2, 2] + e2[1] e2[2]`.
    @test k4[(1 - 1) * N4 + 2, (1 - 1) * N4 + 2] ==
          e2_4[2] * S4[1, 1] + e2_4[1] * S4[2, 2] + e2_4[1] * e2_4[2]
    # All four equal: `6 e2[1] S[1, 1] + e4[1]`.
    @test k4[1, 1] == 6 * e2_4[1] * S4[1, 1] + e4_4[1]
end

#=
The correction is a subtraction, and it must undo what the factor lift added. Sweep ticket #534.
=#
@testset "The residual block comes off before the cokurtosis correction" begin
    PO = PortfolioOptimisers
    var_ = PO.Statistics.var

    # The covariance the lift produced before it added the residual block: the projected
    # factor covariance, processed, and nothing else.
    function pre_block(pe, pr)
        S = pr.rr.M * pr.fpr.sigma * transpose(pr.rr.M)
        matrix_processing!(pe.mp, S, pr.X)
        return S
    end

    fp_t = FactorPrior(; rsd = true)
    fp_f = FactorPrior(; rsd = false)
    pr_t = prior(fp_t, rd)
    pr_f = prior(fp_f, rd)
    cfg_t = PO.factor_residual_config(fp_t)
    @test cfg_t.rsd
    @test !PO.factor_residual_config(fp_f).rsd

    # Subtracting the block recovers the systematic covariance the lift started from, to
    # 1.08e-19 against a `max|sigma|` of 6.06e-4.
    err = rd.X - pr_t.X
    recovered = pr_t.sigma - Diagonal(vec(var_(cfg_t.ve, err; dims = 1)))
    posdef!(cfg_t.pdm, recovered)
    @test isapprox(recovered, pre_block(fp_t, pr_t); atol = 1e-16)

    # With `rsd = false` no block was ever added, so the covariance is the pre-block one
    # exactly and the correction must leave it alone.
    @test pr_f.sigma == pre_block(fp_f, pr_f)

    #=
    `HighOrderFactorPriorEstimator` forwards the wrapped declaration and never answers for
    itself. Its OWN `rsd` governs the co-moment corrections, not the covariance, so flipping
    it does not change the answer.
    =#
    cfg = PO.factor_residual_config(fp_t)
    @test PO.factor_residual_config(HighOrderFactorPriorEstimator(; pe = fp_t)) === cfg
    @test PO.factor_residual_config(HighOrderFactorPriorEstimator(; pe = fp_t, rsd = false)) ===
          cfg
    @test !PO.factor_residual_config(HighOrderFactorPriorEstimator(; pe = fp_f)).rsd

    # A wrapped estimator that declares `nothing` forwards a `nothing`.
    abl = AugmentedBlackLittermanPrior(; sets = afsets,
                                       a_views = LinearConstraintEstimator(;
                                                                           val = "$(rd.nx[1]) == 0.01"),
                                       f_views = LinearConstraintEstimator(;
                                                                           val = "$(rd.nf[1]) == 0.02"))
    @test isnothing(PO.factor_residual_config(abl))
    @test isnothing(PO.factor_residual_config(HighOrderFactorPriorEstimator(; pe = abl)))

    #=
    A `nothing` declaration and one whose `rsd` is `false` both mean "no block was added", so
    the correction is built on `pr.sigma` untouched. The number the branch produces is the
    cokurtosis of the `rsd = false` route plus exactly that correction.
    =#
    h_off = HighOrderFactorPriorEstimator(; pe = fp_f)
    p_on = prior(h_off, rd)
    p_off = prior(HighOrderFactorPriorEstimator(; pe = fp_f, rsd = false), rd)
    kt_ref = PO.cokurtosis_residuals(pr_f.sigma, rd.X - pr_f.X, h_off.kte.me, h_off.ex)
    @test p_on.kt == p_off.kt + kt_ref
end

#=
`ShrunkFactorPrior` is the falsification witness for the guard in step 9 of
`prior(::HighOrderFactorPriorEstimator, …)`: a wrapper that declares a residual block and then
reports a covariance the block was never added to. Sweep ticket #534.
=#
struct ShrunkFactorPrior{T} <: PortfolioOptimisers.AbstractLowOrderPriorEstimator_AF
    pe::T
end
function PortfolioOptimisers.prior(pe::ShrunkFactorPrior, X::PortfolioOptimisers.MatNum,
                                   F::PortfolioOptimisers.Option{<:PortfolioOptimisers.MatNum} = nothing;
                                   dims::Int = 1, kwargs...)
    pr = PortfolioOptimisers.prior(pe.pe, X, F; dims = dims, kwargs...)
    return PortfolioOptimisers.forward_prior(pr; sigma = pr.sigma * 0.05, chol = nothing)
end
function PortfolioOptimisers.factor_residual_config(pe::ShrunkFactorPrior)
    return PortfolioOptimisers.factor_residual_config(pe.pe)
end

@testset "A residual variance above the prior variance keeps the prior variance" begin
    PO = PortfolioOptimisers
    var_ = PO.Statistics.var
    sfp = ShrunkFactorPrior(FactorPrior())
    pr_s = prior(sfp, rd)
    cfg = PO.factor_residual_config(sfp)
    @test cfg.rsd
    @test !isnothing(pr_s.rr)

    # Every one of the twenty assets trips the guard, because the wrapper shrank the whole
    # covariance to a twentieth while the residuals stayed where they were.
    err = rd.X - pr_s.X
    err_sigma = vec(var_(cfg.ve, err; dims = 1))
    @test count(err_sigma .> diag(pr_s.sigma)) == 20

    h = HighOrderFactorPriorEstimator(; pe = sfp)
    p_on = @test_logs (:warn,) match_mode = :any prior(h, rd)
    # The guard keeps `pr.sigma` whole, so the correction is the one built on it.
    p_off = prior(HighOrderFactorPriorEstimator(; pe = sfp, rsd = false), rd)
    kt_ref = PO.cokurtosis_residuals(pr_s.sigma, err, h.kte.me, h.ex)
    @test p_on.kt == p_off.kt + kt_ref
end

#=
The shapes `prior(::HighOrderPriorEstimator, …)` and `prior(::HighOrderFactorPriorEstimator, …)`
produce, on each of the three co-moment branches, and the constructor guards they must satisfy.
Sweep ticket #534.
=#
@testset "Every co-moment branch of both high order estimators" begin
    PO = PortfolioOptimisers
    N = size(rd.X, 2)
    N2 = N^2
    m = div(N * (N + 1), 2)

    # Both moments: all three structure matrices, at the shapes the nineteen constructor
    # guards of `HighOrderPrior` demand.
    both = prior(HighOrderPriorEstimator(), rd)
    @test size(both.kt) == (N2, N2)
    @test size(both.sk) == (N, N2)
    @test size(both.D2) == (N2, m)
    @test size(both.L2) == size(both.S2) == (m, N2)
    @test size(both.V) == (N, N)

    # Cokurtosis alone: `L2` and `S2` serve `kt`, so they are built and `D2` stays `nothing`.
    kt_only = prior(HighOrderPriorEstimator(; ske = nothing), rd)
    @test isnothing(kt_only.sk)
    @test isnothing(kt_only.V)
    @test isnothing(kt_only.D2)
    @test isnothing(kt_only.skmp)
    @test size(kt_only.kt) == (N2, N2)
    @test size(kt_only.L2) == size(kt_only.S2) == (m, N2)

    # Coskewness alone: no `kt`, so none of the three is built.
    sk_only = prior(HighOrderPriorEstimator(; kte = nothing), rd)
    @test isnothing(sk_only.kt)
    @test isnothing(sk_only.D2)
    @test isnothing(sk_only.L2)
    @test isnothing(sk_only.S2)
    @test size(sk_only.sk) == (N, N2)

    # Neither: a plain low order carrier wearing a high order jacket.
    neither = prior(HighOrderPriorEstimator(; kte = nothing, ske = nothing), rd)
    @test all(isnothing,
              (neither.kt, neither.sk, neither.V, neither.D2, neither.L2, neither.S2))

    # The same three branches at the factor dimension. The nested carrier validates its own
    # triple against the factor count, so both dimensions are built together.
    K = size(rd.F, 2)
    K2 = K^2
    mk = div(K * (K + 1), 2)
    fboth = prior(HighOrderFactorPriorEstimator(), rd)
    @test size(fboth.D2) == (N2, m)
    @test size(fboth.f_D2) == (K2, mk)
    @test size(fboth.f_L2) == size(fboth.f_S2) == (mk, K2)

    f_kt_only = prior(HighOrderFactorPriorEstimator(; ske = nothing), rd)
    @test isnothing(f_kt_only.sk)
    @test isnothing(f_kt_only.D2)
    @test isnothing(f_kt_only.f_D2)
    @test size(f_kt_only.L2) == size(f_kt_only.S2) == (m, N2)
    @test size(f_kt_only.f_L2) == size(f_kt_only.f_S2) == (mk, K2)

    # No cokurtosis: `kM` is built by the coskewness branch instead, and the projection still
    # lands on the asset axis.
    f_sk_only = prior(HighOrderFactorPriorEstimator(; kte = nothing), rd)
    @test isnothing(f_sk_only.kt)
    @test isnothing(f_sk_only.L2)
    @test size(f_sk_only.sk) == (N, N2)
    @test size(f_sk_only.f_sk) == (K, K2)

    # Neither: the nested factor carrier is dropped altogether.
    f_neither = prior(HighOrderFactorPriorEstimator(; kte = nothing, ske = nothing), rd)
    @test isnothing(f_neither.fpr)
    @test all(isnothing, (f_neither.kt, f_neither.sk, f_neither.V))
    @test isapprox(f_neither.sigma, prior(FactorPrior(), rd).sigma)
end

#=
`dup_elim_sum_view` is what `port_opt_view` reaches for a subset universe, and it rebuilds the
three matrices at the subproblem's asset count rather than cutting them. Sweep ticket #534.
=#
@testset "A subset universe rebuilds the structure matrices at its own size" begin
    PO = PortfolioOptimisers
    X4 = rd.X[:, 1:4]
    pr4 = prior(HighOrderPriorEstimator(), X4)
    i = [1, 3]
    v = PO.port_opt_view(pr4, i)

    # The full carrier holds the `n = 4` matrices and the view holds the `n = 2` ones.
    @test size(pr4.D2) == (16, 10)
    @test size(pr4.L2) == size(pr4.S2) == (10, 16)
    @test size(v.D2) == (4, 3)
    @test size(v.L2) == size(v.S2) == (3, 4)
    @test (v.D2, v.L2, v.S2) == PO.dup_elim_sum_matrices(2)

    # And they satisfy the identities on a hand-built symmetric two-asset matrix, which a cut
    # of the four-asset matrices could not.
    A = [2.0 5.0; 5.0 7.0]
    vechA = [A[a, b] for b in 1:2 for a in b:2]
    @test v.D2 * vechA == vec(A)
    @test v.L2 * vec(A) == vechA
    @test v.S2 == transpose(v.D2) * v.D2 * v.L2

    #=
    A carrier holding `D2` and no `kt` is the case the varargs fallback of
    `dup_elim_sum_view` answers. `D2` serves `sk`, which such a carrier does not hold, so the
    view drops all three rather than rebuilding a matrix with no consumer.
    =#
    lo4 = prior(EmpiricalPrior(), X4)
    d2_only = HighOrderPrior(; pr = lo4, D2 = PO.duplication_matrix(4))
    @test size(d2_only.D2) == (16, 10)
    v_d2 = PO.port_opt_view(d2_only, i)
    @test (v_d2.D2, v_d2.L2, v_d2.S2) == (nothing, nothing, nothing)
    @test isa(v_d2, HighOrderPrior)
end

#=
Sweep ticket #536. One fixture serves the four testsets below: a `250 x 5` sample over three
factors, built once as an exact factor model and once with a residual, and each of those with a
non-zero and with a zero intercept. Every number the three Black-Litterman variants' docstrings
quote was measured on it.
=#
const BL536_T, BL536_N, BL536_K = 250, 5, 3
let rng = StableRNG(987654321)
    global BL536_F = randn(rng, BL536_T, BL536_K) .* 0.02
    global BL536_M = [0.9 0.2 -0.1; 1.1 -0.3 0.4; 0.7 0.5 0.2; 1.3 0.1 -0.5; 0.5 0.8 0.3]
    global BL536_b = [0.001, -0.002, 0.0015, 0.0005, 0.0]
    global BL536_Xe = BL536_F * transpose(BL536_M) .+ transpose(BL536_b)
    global BL536_Xr = BL536_Xe .+ randn(rng, BL536_T, BL536_N) .* 0.005
end
const BL536_Xe0 = BL536_F * transpose(BL536_M)
const BL536_Xr0 = BL536_Xe0 .+ randn(StableRNG(4242), BL536_T, BL536_N) .* 0.005
const BL536_sets = UniverseSets(;
                                dict = Dict("nx" => ["A1", "A2", "A3", "A4", "A5"],
                                            "nf" => ["F1", "F2", "F3"]))
const BL536_av = LinearConstraintEstimator(; val = ["A1 == 0.004", "A3 - A4 == 0.001"])
const BL536_fv = LinearConstraintEstimator(; val = ["F1 == 0.003", "F2 == -0.001"])
const BL536_ablv = BlackLittermanViews(; P = [1.0 0 0 0 0; 0 0 1.0 -1.0 0],
                                       Q = [0.004, 0.001])
const BL536_fblv = BlackLittermanViews(; P = [1.0 0 0; 0 1.0 0], Q = [0.003, -0.001])
bl536_gap(pr) = maximum(abs.(pr.mu .- (pr.rr.M * pr.fpr.mu .+ pr.rr.b)))

#=
The identity `mu == rr.M * fpr.mu + rr.b` over all four Black-Litterman members, and the two
independent causes that open the augmented member's gap. Sweep ticket #536.
=#
@testset "The Black-Litterman family identity, and the two causes that break it" begin
    PO = PortfolioOptimisers
    F, N = BL536_F, BL536_N
    afs, av, fv = BL536_sets, BL536_av, BL536_fv

    # The table the `AugmentedBlackLittermanPrior` docstring states.
    prb = prior(BayesianBlackLittermanPrior(; sets = afs, views = fv), BL536_Xr, F)
    prf = prior(FactorBlackLittermanPrior(; sets = afs, views = fv), BL536_Xr, F)
    pra = prior(AugmentedBlackLittermanPrior(; sets = afs, a_views = av, f_views = fv),
                BL536_Xr, F)
    prl = prior(BlackLittermanPrior(; sets = afs, views = av), BL536_Xr, F)
    @test bl536_gap(prb) < 1e-16
    @test bl536_gap(prf) == 0
    @test 1e-5 < bl536_gap(pra) < 1e-3
    # The vanilla member computes no posterior factor distribution, so the right-hand side of
    # the identity cannot be formed at all.
    @test isnothing(prl.fpr)
    @test isnothing(prl.rr)

    # The intercept is no longer a cause, on either branch (#570). It goes into the prior
    # stack, where the update is affine in it, so on an exact factor model the identity
    # holds to machine precision whatever the intercept is.
    abl = AugmentedBlackLittermanPrior(; sets = afs, a_views = av, f_views = fv)
    pe_ = prior(abl, BL536_Xe, F)
    @test isapprox(maximum(abs.(pe_.rr.b)), 0.002; atol = 1e-14)
    @test bl536_gap(pe_) < 1e-13
    # Zeroing the intercept changes nothing, which is the point: it is no longer the cause.
    @test bl536_gap(prior(abl, BL536_Xe0, F)) < 1e-13
    # Cause one: idiosyncratic variance. It survives a fitted intercept of `2.4e-4`.
    @test 1e-4 < bl536_gap(prior(abl, BL536_Xr0, F)) < 1e-3

    # Setting `l` changes neither statement. This also covers the `!isempty(w)` guard of
    # the constructor, which no other test reaches.
    abl_l = AugmentedBlackLittermanPrior(; sets = afs, a_views = av, f_views = fv, l = 2.0,
                                         w = fill(1 / N, N))
    @test bl536_gap(prior(abl_l, BL536_Xe, F)) < 1e-13
    @test 1e-5 < bl536_gap(prior(abl_l, BL536_Xr, F)) < 1e-3
    @test_throws IsEmptyError AugmentedBlackLittermanPrior(; a_views = BL536_ablv,
                                                           f_views = BL536_fblv,
                                                           w = Float64[])

    # Cause two: a non-zero `rf` with `l` set. The equilibrium branch adds the rate to the
    # whole stack, so the asset half gains `rf` and the factor half gains `rf` as well. The
    # identity carries the factor half through the loadings, so the two agree only where an
    # asset's loadings sum to one, and the gap is `rf * (1 - s)` exactly. It is zero at the
    # default `rf = 0.0`, and it is the only way the rate reaches this identity.
    ablr = AugmentedBlackLittermanPrior(; sets = afs, a_views = av, f_views = fv, l = 2.0,
                                        rf = 0.03)
    prr = prior(ablr, BL536_Xe, F)
    s536 = vec(sum(prr.rr.M; dims = 2))
    @test isapprox(prr.mu .- (prr.rr.M * prr.fpr.mu .+ prr.rr.b), 0.03 .* (1 .- s536);
                   atol = 1e-13)
    @test !isapprox(s536, ones(N))

    # Muting both view sets with views that repeat the prior makes the update null, so the
    # posterior is the prior and the identity holds exactly. Before #570 the gap here was
    # `b`, which is what showed that the intercept cause was independent of the views.
    ap0 = prior(EmpiricalPrior(), BL536_Xr0)
    fp0 = prior(EmpiricalPrior(), F)
    Pa = [1.0 0 0 0 0; 0 0 1.0 -1.0 0]
    Pf = [1.0 0 0; 0 1.0 0]
    an = BlackLittermanViews(; P = Pa, Q = Pa * ap0.mu)
    fn = BlackLittermanViews(; P = Pf, Q = Pf * fp0.mu)
    pn = prior(AugmentedBlackLittermanPrior(; sets = afs, a_views = an, f_views = fn),
               BL536_Xr0, F)
    @test bl536_gap(pn) < 1e-17

    # Before the update: the two priors satisfy the identity when both means are the plain
    # sample mean, because least squares with an intercept zeroes the *unweighted* residual
    # mean. One shared non-uniform weighting is not enough.
    wodd = pweights(normalize(rand(StableRNG(7), BL536_T) .+ 0.5, 1))
    epw(w) = EmpiricalPrior(;
                            ce = PortfolioOptimisersCovariance(;
                                                               ce = Covariance(;
                                                                               me = SimpleExpectedReturns(;
                                                                                                          w = w),
                                                                               w = w)),
                            me = SimpleExpectedReturns(; w = w))
    function pgap(ape, fpe, X)
        ap = prior(ape, X)
        fp = prior(fpe, F)
        rr, _ = PO.factor_reconstruction(StepwiseRegression(), X, F)
        return maximum(abs.(ap.mu .- (rr.M * fp.mu .+ rr.b)))
    end
    @test pgap(EmpiricalPrior(), EmpiricalPrior(), BL536_Xr) < 1e-17
    @test pgap(epw(wodd), epw(wodd), BL536_Xr) > 1e-5
    @test pgap(EmpiricalPrior(), epw(wodd), BL536_Xr) > 1e-4
    # On an exact factor model there is no residual to weight, so it holds again.
    @test pgap(epw(wodd), epw(wodd), BL536_Xe) < 1e-17
end

#=
`BayesianBlackLittermanPrior`'s four closed forms against a hand computation, the width of its
view matrix, and the one site that adds the rate. Sweep ticket #536.
=#
@testset "Bayesian Black-Litterman, by hand" begin
    PO = PortfolioOptimisers
    F, N, K, T = BL536_F, BL536_N, BL536_K, BL536_T
    afs, fv = BL536_sets, BL536_fv

    pr_in = prior(FactorPrior(; pe = EmpiricalPrior(; me = EquilibriumExpectedReturns())),
                  BL536_Xr, F)
    f_mu, f_sigma = pr_in.fpr.mu, pr_in.fpr.sigma
    Sigma, Mm, bb = pr_in.sigma, pr_in.rr.M, pr_in.rr.b
    pk = PO.bl_preroll(fv, afs, nothing, f_sigma, nothing, T, Float64, false, :tfkey)
    P, Q, tau, Om = pk.P, pk.Q, pk.tau, pk.omega
    # `P` is over the factor axis, so it is `K` wide and not `N`, and `tau` is `1/T`.
    @test size(P) == (2, K)
    @test tau == 1 / T

    H = inv(f_sigma) + transpose(P) * (Om \ P)
    Pibar = H \ (f_sigma \ f_mu + transpose(P) * (Om \ Q))
    V = inv(H + transpose(Mm) * (Sigma \ Mm))
    Sig_bbl = inv(inv(Sigma) - (Sigma \ Mm) * V * transpose(Mm) * inv(Sigma))
    mu_bbl = Sig_bbl * (Sigma \ Mm) * V * H * Pibar .+ bb

    pr = prior(BayesianBlackLittermanPrior(; sets = afs, views = fv), BL536_Xr, F)
    @test isapprox(pr.fpr.mu, Pibar; atol = 1e-18)
    @test isapprox(pr.fpr.sigma, inv(H); atol = 1e-20)
    @test isapprox(pr.mu, mu_bbl; atol = 1e-16)
    @test isapprox(pr.sigma, Sig_bbl; atol = 1e-16)
    # `sigma_hat` is a precision, so the reported factor covariance is its inverse.
    @test isapprox(pr.fpr.sigma * H, Matrix(1.0I, K, K); atol = 1e-12)
    # Both blocks drop `chol`.
    @test isnothing(pr.chol)
    @test isnothing(pr.fpr.chol)

    # An asset-axis `P` is refused rather than answered with a wrong shape. The width is the
    # only thing that sees it, because a precomputed views object resolves no name.
    wide = BlackLittermanViews(; P = [1.0 0 0 0 0; 0 0 1.0 -1.0 0], Q = [0.004, 0.001])
    @test_throws DimensionMismatch prior(BayesianBlackLittermanPrior(; sets = afs,
                                                                     views = wide),
                                         BL536_Xr, F)
    # Views written in asset names resolve nothing against the factor universe.
    @test_throws PortfolioOptimisers.IsNothingError prior(BayesianBlackLittermanPrior(;
                                                                                      sets = afs,
                                                                                      views = LinearConstraintEstimator(;
                                                                                                                        val = ["A1 == 0.004"])),
                                                          BL536_Xr, F)

    # The rate is added once, on the assets, and the factor block never carries it.
    p0 = prior(BayesianBlackLittermanPrior(; sets = afs, views = fv, rf = 0.0), BL536_Xr, F)
    p3 = prior(BayesianBlackLittermanPrior(; sets = afs, views = fv, rf = 0.03), BL536_Xr,
               F)
    @test isapprox(p3.mu .- p0.mu, fill(0.03, N); atol = 1e-17)
    @test p3.fpr.mu == p0.fpr.mu
    # The carrier is internally consistent on both rates.
    @test isapprox(p3.mu, p3.rr.M * p3.fpr.mu .+ p3.rr.b .+ 0.03; atol = 1e-16)
end

#=
`FactorBlackLittermanPrior` is the ordinary Black-Litterman posterior over the factor axis,
lifted through the loadings. Sweep ticket #536.
=#
@testset "Factor Black-Litterman, by hand" begin
    PO = PortfolioOptimisers
    F, N, K, T = BL536_F, BL536_N, BL536_K, BL536_T
    afs, fv = BL536_sets, BL536_fv

    fp = prior(EmpiricalPrior(), F)
    rr, pX = PO.factor_reconstruction(StepwiseRegression(), BL536_Xr, F)
    pk = PO.bl_preroll(fv, afs, nothing, fp.sigma, nothing, T, Float64, false, :tfkey)
    # The factor moments are literally `vanilla_posteriors` run over the factor axis.
    fmu, fsig = PO.vanilla_posteriors(pk.tau, fp.mu, fp.sigma, pk.omega, pk.P, pk.Q)
    err = BL536_Xr .- pX
    Seps = Diagonal(vec(var(SimpleVariance(), err; dims = 1)))

    for (rsd, S) in ((true, Seps), (false, zeros(N, N)))
        pr = prior(FactorBlackLittermanPrior(; sets = afs, views = fv, rsd = rsd), BL536_Xr,
                   F)
        @test pr.fpr.mu == fmu
        @test pr.fpr.sigma == fsig
        @test pr.mu == rr.M * fmu .+ rr.b
        @test isapprox(pr.sigma, rr.M * fsig * transpose(rr.M) + S; atol = 1e-15)
        # `factor_lift`'s Cholesky identity survives the Black-Litterman moments, and the
        # residual block reaches the factor and the covariance together.
        @test isapprox(transpose(pr.chol) * pr.chol, pr.sigma; atol = 1e-15)
        # No `Z`, and the factor block drops `chol`.
        @test isnothing(panel_feature_matrix(pr.pnl)[2])
        @test isnothing(pr.fpr.chol)
    end
    # The residual block is what separates the two branches.
    pt = prior(FactorBlackLittermanPrior(; sets = afs, views = fv, rsd = true), BL536_Xr, F)
    pf = prior(FactorBlackLittermanPrior(; sets = afs, views = fv, rsd = false), BL536_Xr,
               F)
    @test all(diag(pt.sigma) .> diag(pf.sigma))

    # The rate reaches the answer through the factors, and only on the `l` branch: there the
    # equilibrium mean is a risk premium and `apply_rf` converts it before the update, so
    # the shift is the whole factor shift carried through the blend and the loadings. Where
    # `l` is `nothing` the factor prior's own mean is a total return already, nothing reads
    # `pe.rf`, and the answer does not move at all.
    G = pk.tau *
        fp.sigma *
        transpose(pk.P) *
        inv(pk.P * (pk.tau * fp.sigma) * transpose(pk.P) + pk.omega)
    shift = rr.M * ((I - G * pk.P) * ones(K))
    base = prior(FactorBlackLittermanPrior(; sets = afs, views = fv, rf = 0.0), BL536_Xr, F)
    base_l = prior(FactorBlackLittermanPrior(; sets = afs, views = fv, rf = 0.0, l = 2.0),
                   BL536_Xr, F)
    for rf in (0.03, 0.06)
        pr = prior(FactorBlackLittermanPrior(; sets = afs, views = fv, rf = rf), BL536_Xr,
                   F)
        @test pr.mu == base.mu
        @test pr.fpr.mu == base.fpr.mu
        prl = prior(FactorBlackLittermanPrior(; sets = afs, views = fv, rf = rf, l = 2.0),
                    BL536_Xr, F)
        @test isapprox((prl.mu .- base_l.mu) ./ rf, shift; atol = 1e-14)
    end
    # And it is not `rf * (1 - s)` for the row sums `s` of the loadings: that expression is
    # the no-view limit `G = 0` of the arithmetic this replaced, and no estimator reaches
    # it, because one with no views cannot be built (#571).
    @test !isapprox(shift, 1 .- vec(sum(rr.M; dims = 2)); atol = 1e-3)
end

#=
`AugmentedBlackLittermanPrior`'s stack, its two axes, its guards and what it forwards.
Sweep ticket #536.
=#
@testset "Augmented Black-Litterman, its stack and its two axes" begin
    PO = PortfolioOptimisers
    F, N, K, T = BL536_F, BL536_N, BL536_K, BL536_T
    afs, av, fv = BL536_sets, BL536_av, BL536_fv

    ap = prior(EmpiricalPrior(), BL536_Xr)
    fp = prior(EmpiricalPrior(), F)
    rr, _ = PO.factor_reconstruction(StepwiseRegression(), BL536_Xr, F)
    Sa, Sf = ap.sigma, fp.sigma
    aug = hcat(vcat(Sa, Sf * transpose(rr.M)), vcat(rr.M * Sf, Sf))
    # The off-diagonal blocks are built from the *factor* covariance, and they are the
    # cross-covariance the factor model implies.
    @test aug[1:N, (N + 1):(N + K)] == rr.M * Sf
    @test aug[(N + 1):(N + K), 1:N] == transpose(aug[1:N, (N + 1):(N + K)])
    Xc = BL536_Xr .- transpose(ap.mu)
    Fc = F .- transpose(fp.mu)
    @test isapprox((transpose(Xc) * Fc) ./ (T - 1), rr.M * Sf; atol = 1e-16)

    ak = PO.bl_preroll(av, afs, nothing, Sa, nothing, T, Float64, false)
    fk = PO.bl_preroll(fv, afs, nothing, Sf, nothing, T, Float64, false, :tfkey)
    augP = [ak.P zeros(size(ak.P, 1), K); zeros(size(fk.P, 1), N) fk.P]
    augQ = vcat(ak.Q, fk.Q)
    augOm = [ak.omega zeros(2, 2); zeros(2, 2) fk.omega]
    pmu, psig = PO.vanilla_posteriors(ak.tau, vcat(ap.mu, fp.mu), aug, augOm, augP, augQ)
    pr = prior(AugmentedBlackLittermanPrior(; sets = afs, a_views = av, f_views = fv),
               BL536_Xr, F)
    # The truncation reads the asset half from `1:N` and the factor half from `N+1:N+K`, and
    # adds nothing to either. The intercept is inside `ap.mu` already, because least squares
    # with an intercept makes the mean of `X` equal `M * fp.mu + b` (#570).
    @test pr.mu == pmu[1:N]
    @test isapprox(ap.mu, rr.M * fp.mu .+ rr.b; atol = 1e-17)
    @test pr.fpr.mu == pmu[(N + 1):(N + K)]
    @test pr.fpr.sigma == psig[(N + 1):(N + K), (N + 1):(N + K)]

    # Four mandates. Each axis is needed only by the views that resolve names against it, so a
    # precomputed pair needs no universe at all and answers the same as one given a universe.
    m_both = pr
    m_aonly = prior(AugmentedBlackLittermanPrior(; sets = afs, a_views = av,
                                                 f_views = BL536_fblv), BL536_Xr, F)
    m_fonly = prior(AugmentedBlackLittermanPrior(; sets = afs, a_views = BL536_ablv,
                                                 f_views = fv), BL536_Xr, F)
    m_nosets = prior(AugmentedBlackLittermanPrior(; a_views = BL536_ablv,
                                                  f_views = BL536_fblv), BL536_Xr, F)
    m_sets = prior(AugmentedBlackLittermanPrior(; sets = afs, a_views = BL536_ablv,
                                                f_views = BL536_fblv), BL536_Xr, F)
    @test all(isa.((m_both, m_aonly, m_fonly, m_nosets), LowOrderPrior))
    @test m_nosets.mu == m_sets.mu
    # The two routes into the same views agree exactly.
    @test m_aonly.mu == m_both.mu
    @test m_fonly.mu == m_both.mu

    # `port_opt_view` slices the asset entries and leaves the factor universe whole.
    pv = AugmentedBlackLittermanPrior(; sets = afs, a_views = av, f_views = fv,
                                      w = fill(1 / N, N))
    sl = PO.port_opt_view(pv, [1, 3, 5], N)
    @test sl.sets.dict["nx"] == ["A1", "A3", "A5"]
    @test sl.sets.dict["nf"] == ["F1", "F2", "F3"]
    @test length(sl.w) == 3

    # The guards this file holds of its own.
    short = UniverseSets(; dict = Dict("nx" => ["A1", "A2"], "nf" => ["F1", "F2", "F3"]))
    @test_throws DimensionMismatch prior(AugmentedBlackLittermanPrior(; sets = short,
                                                                      a_views = av,
                                                                      f_views = BL536_fblv),
                                         BL536_Xr, F)

    # `w`, `ens`, `kld` and `ow` are the asset prior's, and `fpr.w` the factor prior's.
    @test pr.w === ap.w
    @test pr.ens === ap.ens
    @test pr.kld === ap.kld
    @test pr.ow === ap.ow
    @test pr.fpr.w === fp.w
    @test isnothing(pr.chol)
    @test isnothing(pr.fpr.chol)

    # The rate goes on the whole equilibrium stack, so both halves move by the same gain
    # applied to a vector of ones. Where `l` is `nothing` the two wrapped means are total
    # returns already, nothing reads `pe.rf`, and neither half moves at all.
    Ga = ak.tau *
         aug *
         transpose(augP) *
         inv(augP * (ak.tau * aug) * transpose(augP) + augOm)
    shift = (I - Ga * augP) * ones(N + K)
    base_l = prior(AugmentedBlackLittermanPrior(; sets = afs, a_views = av, f_views = fv,
                                                l = 2.0), BL536_Xr, F)
    for rf in (0.03, 0.06)
        pf_ = prior(AugmentedBlackLittermanPrior(; sets = afs, a_views = av, f_views = fv,
                                                 rf = rf), BL536_Xr, F)
        @test pf_.mu == pr.mu
        @test pf_.fpr.mu == pr.fpr.mu
        pl_ = prior(AugmentedBlackLittermanPrior(; sets = afs, a_views = av, f_views = fv,
                                                 rf = rf, l = 2.0), BL536_Xr, F)
        @test isapprox((pl_.mu .- base_l.mu) ./ rf, shift[1:N]; atol = 1e-14)
        @test isapprox((pl_.fpr.mu .- base_l.fpr.mu) ./ rf, shift[(N + 1):(N + K)];
                       atol = 1e-14)
    end
    # This estimator adds no residual block, so it declares none.
    @test isnothing(PO.factor_residual_config(pv))
end
