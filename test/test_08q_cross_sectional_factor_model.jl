#=
Check `src/08_Moments/41_CrossSectionalFactorModel.jl` against the contract its docstrings
state, and against the reference implementation the map of issue #643 ports. Issue #706.

THREE FACTS SHAPE THE PROBES.

1. `csfm.L` NEVER READS `nothing`. The `swap(L, M)` rule answers `M` when the field is unset,
   so a probe of the unset case must read `getfield(csfm, :L)`. `An unset L reads back as M`
   pins both halves: the property answers `M`, and the field is still unset. `A view keeps an
   unset L unset` pins the same rule through `port_opt_view`, which is where a property read
   would silently materialise `L` as a copy of `M`.

2. THE ASSET AXIS IS NOT ONE AXIS. A loadings matrix holds one row per asset, a per-asset
   history one column per asset, an exposure history one column per asset within a slice, and
   a full idiosyncratic covariance one of each. `port_opt_view cuts every asset axis` measures
   all four against a hand-written slice, and it reorders the assets so that a probe cannot
   pass on a shape alone.

3. THE REFERENCE IMPLEMENTATION'S OWN SELECTION TESTS ARE THE ORACLE. They state which field
   moves under an asset selection and which passes through, that a full idiosyncratic
   covariance is cut on both axes while a diagonal one is cut once, and that a model may drop
   its exposure history, its idiosyncratic returns, its idiosyncratic variance history and its
   benchmark weights and stay a model. `A slim model keeps its loadings and drops its
   histories` is that last case.

`fcb` carries the family re-basis. Issue #651 settles what it is, so this file uses a matrix as
a stand-in and probes only the rule this result states: `L` and `fcb` are present together or
absent together.
=#

@testset "CrossSectionalFactorModel" begin
    # One fixture, written out rather than generated: 4 observations, 3 assets, 2 factors.
    # `Ms[end, :, :] == M`, as the exposure history of a fitted model does.
    M = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    b = [0.1, 0.2, 0.3]
    Ms = zeros(4, 3, 2)
    Ms[1, :, :] = [0.5 1.0; 1.5 2.0; 2.5 3.0]
    Ms[2, :, :] = [0.6 1.1; 1.6 2.1; 2.6 3.1]
    Ms[3, :, :] = [0.7 1.2; 1.7 2.2; 2.7 3.2]
    Ms[4, :, :] = M
    csr = CrossSectionalRegression(; f = [0.01 0.02; 0.03 0.04; 0.05 0.06; 0.07 0.08],
                                   eps = [0.11 0.12 0.13; 0.14 0.15 0.16; 0.17 0.18 0.19;
                                          0.21 0.22 0.23], n = [3, 3, 3, 3],
                                   b = [0.001, 0.002, 0.003, 0.004])
    vs = [0.31 0.32 0.33; 0.34 0.35 0.36; 0.37 0.38 0.39; 0.41 0.42 0.43]
    rw = [0.51 0.52 0.53; 0.54 0.55 0.56; 0.57 0.58 0.59; 0.61 0.62 0.63]
    bw = [0.71 0.72 0.73; 0.74 0.75 0.76; 0.77 0.78 0.79; 0.81 0.82 0.83]
    esigma_diag = [0.91, 0.92, 0.93]
    esigma_full = [0.91 0.01 0.02; 0.01 0.92 0.03; 0.02 0.03 0.93]
    fam = ["style", "sector"]
    # A family re-basis drops one factor of a constrained family, so `L` is narrower than `M`.
    L = reshape([1.0, 3.0, 5.0], 3, 1)
    fcb = reshape([1.0, -1.0], 2, 1)

    full_model(; kwargs...) = CrossSectionalFactorModel(; M = M, b = b, csr = csr, Ms = Ms,
                                                        vs = vs, esigma = esigma_diag,
                                                        rw = rw, bw = bw, fam = fam,
                                                        lag = 1, kwargs...)

    @testset "An unset L reads back as M" begin
        csfm = CrossSectionalFactorModel(; M = M, b = b)
        # The property answers `M`, so no consumer of the loadings needs a `Nothing` branch.
        @test csfm.L === csfm.M
        @test !isnothing(csfm.L)
        # The field is still unset, which is what the rule exists to express.
        @test isnothing(getfield(csfm, :L))
        @test isnothing(csfm.fcb)
        @test :L in propertynames(csfm)
        # Every optional field after `b` reads back as `nothing` when it was not given.
        @test all(isnothing,
                  (csfm.csr, csfm.Ms, csfm.vs, csfm.esigma, csfm.rw, csfm.bw, csfm.fam,
                   csfm.fcb, csfm.lag))
    end

    @testset "A re-basis makes L narrower than M" begin
        csfm = CrossSectionalFactorModel(; M = M, L = L, b = b, fcb = fcb)
        @test csfm.L === L
        @test getfield(csfm, :L) === L
        @test size(csfm.L, 2) < size(csfm.M, 2)
        @test csfm.fcb === fcb
    end

    @testset "The constructor guards every axis" begin
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalFactorModel(;
                                                                                M = Matrix{Float64}(undef,
                                                                                                    0,
                                                                                                    0),
                                                                                b = b)
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalFactorModel(; M = M,
                                                                                b = Float64[])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = [0.1, 0.2])
        # `L` and `fcb` are one decision, so either alone is refused.
        @test_throws ArgumentError CrossSectionalFactorModel(; M = M, L = L, b = b)
        @test_throws ArgumentError CrossSectionalFactorModel(; M = M, b = b, fcb = fcb)
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M,
                                                                 L = reshape([1.0, 3.0], 2,
                                                                             1), b = b,
                                                                 fcb = fcb)
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 fam = ["style"])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 Ms = zeros(4, 2, 2))
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 Ms = zeros(4, 3, 3))
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 csr = CrossSectionalRegression(;
                                                                                                f = csr.f,
                                                                                                eps = vs[:,
                                                                                                         1:2],
                                                                                                n = csr.n))
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 vs = vs[:, 1:2])
        # The three per-asset histories agree on the observation axis.
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b, rw = rw,
                                                                 bw = bw[1:3, :])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b, rw = rw,
                                                                 vs = vs[1:3, :])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b, bw = bw,
                                                                 vs = vs[1:3, :])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 esigma = [0.91, 0.92])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 esigma = esigma_full[1:2,
                                                                                      :])
        @test_throws DimensionMismatch CrossSectionalFactorModel(; M = M, b = b,
                                                                 esigma = esigma_full[1:2,
                                                                                      1:2])
        @test_throws DomainError CrossSectionalFactorModel(; M = M, b = b, lag = -1)
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalFactorModel(; M = M,
                                                                                b = b,
                                                                                vs = Matrix{Float64}(undef,
                                                                                                     0,
                                                                                                     3))
        @test_throws PortfolioOptimisers.IsEmptyError CrossSectionalFactorModel(; M = M,
                                                                                b = b,
                                                                                esigma = Float64[])
    end

    @testset "Both shapes of esigma" begin
        diag_model = full_model()
        full_esigma_model = full_model(; esigma = esigma_full)
        @test diag_model.esigma === esigma_diag
        @test full_esigma_model.esigma === esigma_full
        # A diagonal covariance is cut once, and a full one on both axes. The reference
        # implementation's own selection tests state the pair.
        i = [3, 1]
        @test PortfolioOptimisers.port_opt_view(diag_model, i).esigma == esigma_diag[i]
        @test PortfolioOptimisers.port_opt_view(full_esigma_model, i).esigma ==
              esigma_full[i, i]
        @test size(PortfolioOptimisers.port_opt_view(full_esigma_model, i).esigma) == (2, 2)
        @test isnothing(PortfolioOptimisers.idiosyncratic_covariance_view(nothing, i))
    end

    @testset "port_opt_view cuts every asset axis" begin
        csfm = full_model(; L = L, fcb = fcb)
        # The selection reorders the assets, so a probe cannot pass on a shape alone.
        i = [3, 1]
        v = PortfolioOptimisers.port_opt_view(csfm, i)
        @test v.M == M[i, :]
        @test v.L == L[i, :]
        @test v.b == b[i]
        @test v.Ms == Ms[:, i, :]
        @test v.vs == vs[:, i]
        @test v.esigma == esigma_diag[i]
        @test v.rw == rw[:, i]
        @test v.bw == bw[:, i]
        @test v.csr.eps == csr.eps[:, i]
        # The factor axis and the observation axis follow no asset selection.
        @test v.csr.f === csr.f
        @test v.csr.n === csr.n
        @test v.csr.b === csr.b
        @test v.fam === fam
        @test v.fcb === fcb
        @test v.lag === 1
    end

    @testset "A view keeps an unset L unset" begin
        csfm = full_model()
        v = PortfolioOptimisers.port_opt_view(csfm, [1, 3])
        # A property read of `L` inside the view would have materialised a copy of `M`.
        @test isnothing(getfield(v, :L))
        @test v.L === v.M
        @test isnothing(v.fcb)
    end

    @testset "A slim model keeps its loadings and drops its histories" begin
        # The fields the reference implementation's own slim mode drops.
        csfm = CrossSectionalFactorModel(; M = M, b = b, esigma = esigma_full, rw = rw,
                                         L = L, fcb = fcb, lag = 1)
        @test all(isnothing, (csfm.csr, csfm.Ms, csfm.vs, csfm.bw))
        @test csfm.M === M
        @test csfm.esigma === esigma_full
        @test csfm.rw === rw
        v = PortfolioOptimisers.port_opt_view(csfm, [2, 3])
        @test all(isnothing, (v.csr, v.Ms, v.vs, v.bw))
        @test v.M == M[[2, 3], :]
        @test v.esigma == esigma_full[[2, 3], [2, 3]]
        @test v.rw == rw[:, [2, 3]]
    end

    @testset "The realised identity carries the intercept of the fit, not b" begin
        # `# Mathematical definition` states two equations, and the first one is measurable.
        # A fit of real numbers reproduces `x_t = b_t 1 + M_t f_t + eps_t`, where `b_t` is
        # the intercept of observation `t`. The per-asset `b` of the model is a term of the
        # expected return, so it never closes this identity. The docstring stated `b` in
        # that place until issue #715 measured the residue.
        rng = StableRNG(987654321)
        To, No, Ko = 6, 5, 2
        Z = randn(rng, To, No, Ko)
        Xr = randn(rng, To, No) ./ 20
        Wr = fill(1.0, To, No)
        fit = cross_sectional_regression(CrossSectionalLinearRegression(; intercept = true),
                                         Z, Xr, Wr)
        csfm = CrossSectionalFactorModel(; M = Z[end, :, :], b = randn(rng, No) ./ 100,
                                         csr = fit, Ms = Z)
        # The loadings are the last slice of the exposure history.
        @test csfm.M == csfm.Ms[end, :, :]
        for t in 1:To
            r = Xr[t, :] - csfm.Ms[t, :, :] * fit.f[t, :] - fit.eps[t, :]
            # The residue of the identity is one number repeated across the assets.
            @test all(x -> isapprox(x, fit.b[t]; atol = 1e-12), r)
            # It is the intercept of the fit, and not the factor-orthogonal expected return.
            @test !isapprox(r, csfm.b)
        end
    end

    @testset "The model is a loadings result the verbs already reach" begin
        csfm = full_model()
        @test isa(csfm, PortfolioOptimisers.AbstractLoadingsRegressionResult)
        @test isa(csfm, PortfolioOptimisers.RegE_Reg)
        # `regression` passes a result through, as it does for a `Regression`.
        @test regression(csfm) === csfm
        @test regression(csfm, csfm.M, csfm.M) === csfm
    end

    # Issue #776. One reader answers the idiosyncratic variances off either loadings block,
    # whatever shape that block stores them in, and it refuses rather than inventing a
    # weighting when the block carries none.
    @testset "idiosyncratic_variances reads either shape off either block" begin
        PO = PortfolioOptimisers
        # A vector of variances comes back unchanged, off both blocks.
        @test PO.idiosyncratic_variances(full_model()) == esigma_diag
        @test PO.idiosyncratic_variances(PO.Regression(; M = M, esigma = esigma_diag)) ==
              esigma_diag
        # A full covariance comes back as its diagonal, off both blocks.
        @test PO.idiosyncratic_variances(full_model(; esigma = esigma_full)) ==
              LinearAlgebra.diag(esigma_full)
        @test PO.idiosyncratic_variances(PO.Regression(; M = M, esigma = esigma_full)) ==
              LinearAlgebra.diag(esigma_full)
        # The diagonal of the full fixture is the diagonal fixture, so the two shapes agree.
        @test LinearAlgebra.diag(esigma_full) == esigma_diag
    end

    @testset "idiosyncratic_variances refuses a block that carries none, and names its filler" begin
        PO = PortfolioOptimisers
        # A `Regression` is filled by the prior that lifts the factor moments, so the message
        # names the switch that makes that prior add a residual block.
        reg_err = try
            PO.idiosyncratic_variances(PO.Regression(; M = M))
            nothing
        catch e
            e
        end
        @test isa(reg_err, PO.IsNothingError)
        @test occursin("rsd", reg_err.msg)
        @test occursin("Regression", reg_err.msg)
        # A cross-sectional model is filled by its own fit, so the message names the field.
        cs_err = try
            PO.idiosyncratic_variances(CrossSectionalFactorModel(; M = M, b = b))
            nothing
        catch e
            e
        end
        @test isa(cs_err, PO.IsNothingError)
        @test occursin("esigma", cs_err.msg)
        @test occursin("CrossSectionalFactorModel", cs_err.msg)
        # The two messages are not one message: each names its own block.
        @test !occursin("rsd", cs_err.msg)
    end
end
