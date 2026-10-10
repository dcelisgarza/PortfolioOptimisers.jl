#=
Parity of the Cross-Sectional Factor Prior away from its defaults (#1385, map #1375): one keyword
at a time, and the combinations a caller uses. Every `Parity_*` file this test reads is an output of
the oracle, fitted on the exchange `parity_write(dir, grid_fixture(fx).rd; filled = true)` writes.

Every case sets the factor prior and the idiosyncratic variance explicitly, with the regime clip
(0.7, 1.6) and no variance floor, so a change of the defaults of the prior (#1383) moves no stored
case. The base model is a market factor and the two passthrough styles with `minra = 5`, and each
case changes what its name says.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))

# The cases that change the structure of the fit store every output. A Return Forecast changes
# the mean alone, so its cases store `mu`, the factor mean and the forecast, and the test checks
# that every other output equals the fit without the forecast, bit for bit.
const GRID_STRUCT = ["Base", "Lag2", "Lag3", "ScoredBp1", "ScoredBp05", "ScoredBp0",
                     "Target", "Blend", "Neutralised", "FamOne", "FamStated", "FamTwo",
                     "NeutFam", "Currency", "CurrencyLx", "Macro", "CurrencyBase"]
const GRID_FORECAST = ["FcFixed" => "Base", "FcFixedSharpe" => "Base", "FcEW" => "Base",
                       "FcTarget" => "Base", "FcTargetIntercept" => "Base",
                       "FcTargetRaw" => "Base", "FcCustom" => "Base",
                       "FcFamily" => "FamOne"]
const GRID_FAMILY = ["FamOne", "FamStated", "FamTwo", "NeutFam", "FcFamily"]

# Compare one fit with the stored outputs of its case. The large panel stores its factor returns
# and `sigma` for nine cases alone, to keep the stored files small; the other large cases were
# measured to the same tolerances.
function grid_check(pr, fix::AbstractString, c::AbstractString, rest)
    load(o) = parity_load("CrossSectionalFactorPrior", "Grid$(fix)$(c)", o)
    has(o) = isfile(parity_asset("CrossSectionalFactorPrior", "Grid$(fix)$(c)", o))
    nm = "$(fix) $(c)"
    # Measured maxrel 5.5e-13 at most, on the large constrained families.
    @test parity_compare(pr.mu[rest], vec(load("Mu"))[rest]; name = "$(nm) mu").ok
    @test parity_compare(pr.fpr.mu, vec(load("FactorMu")); name = "$(nm) factor mu").ok
    if has("Alpha")
        # Measured maxrel 2.3e-13 at most.
        @test parity_compare(pr.rr.rf.mu, vec(load("Alpha")); name = "$(nm) forecast").ok
    end
    if !(has("FactorCov"))
        return nothing
    end
    fam = c in GRID_FAMILY
    # A family case has near-zero off-diagonal factor covariances, which compare against the
    # largest entry, as `sigma` does. Measured maxrel 7.0e-14 cell by cell elsewhere, and
    # maxscaled 6.2e-13 on the families, where the cell-by-cell maxrel is 3.4e-12.
    @test parity_compare(pr.fpr.sigma, load("FactorCov"); scale = fam ? :array : :cell,
                         name = "$(nm) factor cov").ok
    # Measured maxrel 4.8e-14 at most, on the neutralised loadings.
    @test parity_compare(pr.rr.M, load("Loadings"); name = "$(nm) loadings").ok
    # The one member of "Utilities" has a residual of round-off size under a constrained
    # industry family, so its variance is a round-off zero (1e-34) on both sides, and a family
    # case compares against the largest entry. Measured maxrel 3.1e-15 cell by cell elsewhere,
    # and maxscaled 4.4e-15 on the families.
    if has("IdioVariances")
        @test parity_compare(pr.rr.vs, load("IdioVariances"); scale = fam ? :array : :cell,
                             name = "$(nm) vs").ok
    else
        @test parity_compare(pr.rr.vs[end, :], vec(load("IdioVarLatest"));
                             scale = fam ? :array : :cell, name = "$(nm) vs").ok
    end
    # Measured maxrel 2.2e-16.
    if has("BenchmarkWeights")
        @test parity_compare(pr.rr.bw, load("BenchmarkWeights"); name = "$(nm) bw").ok
    elseif has("BenchmarkWeightsLatest")
        @test parity_compare(pr.rr.bw[end, :], vec(load("BenchmarkWeightsLatest"));
                             name = "$(nm) bw").ok
    end
    if !(has("Sigma"))
        return nothing
    end
    # A covariance compares against its largest entry, because its small off-diagonal
    # entries come from a cancellation (#1376). Measured maxscaled 1.2e-14, and maxrel
    # 6.2e-12 cell by cell. It was 6.9e-13 when the lift repaired the systematic block `L F L'`, which the oracle does not repair (#1576).
    @test parity_compare(pr.sigma[rest, rest], load("Sigma")[rest, rest]; scale = :array,
                         name = "$(nm) sigma").ok
    # A factor return is a difference of weighted sums, and one near zero carries the
    # cancellation, so the factor returns compare against their largest entry too. Under a
    # constrained industry family, "Utilities" is empty after observation 60 of the small
    # panel, so its factor return is not identified there, and both sides give a round-off
    # zero. Measured maxscaled 7.7e-15, and maxrel 3.7e-12 cell by cell outside that column.
    @test parity_compare(pr.fpr.X, load("FactorReturns"); scale = :array,
                         name = "$(nm) factor returns").ok
    return nothing
end

@testset "Parity: the prior away from its defaults (#1385)" begin
    @testset "$(fix)" for (fix, fx) in (("Small", parity_small_panel()),
                                        ("Large", parity_large_panel()))
        rd = grid_fixture(fx)
        rdu = grid_fixture(fx; base_usd = true)
        N = size(rd.X, 2)
        # Asset 4 of the small panel is in its warm-up at the latest observation. Both sides
        # state its mean and its covariances, which the model determines, and neither states
        # its variance, so every asset compares, with its `NaN` pattern (#1384).
        rest = 1:N
        fits = Dict{String, Any}()
        @testset "$(c)" for c in GRID_STRUCT
            rdc = c == "CurrencyBase" ? rdu : rd
            pe = grid_prior(c, rdc)
            pr = prior(pe, rdc)
            fits[c] = pr
            grid_check(pr, fix, c, rest)
            # The factor axis the prior declares before the fit is the axis it fits.
            ax = cross_sectional_factor_axis(pe, rdc)
            @test ax.nf == pr.rr.nf
            @test ax.fam == pr.rr.fam
            sets = cross_sectional_factor_sets(pe, rdc)
            @test sets.dict[PortfolioOptimisers.factor_axis_key(sets, pr.rr)] == pr.rr.nf
            @test all(f -> sort(sets.dict[f]) == sort(pr.rr.nf[pr.rr.fam .== f]),
                      unique(pr.rr.fam))
        end
        @testset "$(c)" for (c, b) in GRID_FORECAST
            pr = prior(grid_prior(c, rd), rd)
            grid_check(pr, fix, c, rest)
            # The forecast enters the mean alone.
            p0 = fits[b]
            @test isequal(pr.rr.M, p0.rr.M) && isequal(pr.fpr.sigma, p0.fpr.sigma)
            @test isequal(pr.fpr.X, p0.fpr.X) && isequal(pr.rr.vs, p0.rr.vs)
            @test isequal(pr.sigma, p0.sigma)
        end
        @testset "The smallest eligible asset count" begin
            # The fewest eligible assets at an observation of the small panel is 10, so
            # `minra = 10` does not bind, and neither does the default `max(2K, 30)` on the
            # large panel. The fit equals `Base` there.
            pr = prior(grid_prior("Minra10", rd), rd)
            @test isequal(pr.mu, fits["Base"].mu) && isequal(pr.sigma, fits["Base"].sigma)
            if fix == "Large"
                pr = prior(grid_prior("MinraNone", rd), rd)
                @test isequal(pr.mu, fits["Base"].mu) &&
                      isequal(pr.fpr.X, fits["Base"].fpr.X)
            else
                # Where it binds, both sides refuse the fit and count the same observations:
                # 79 under the default 30, and 19 under 11, the fewest being 10 at the first.
                for (c, n) in (("MinraNone", 79), ("Minra11", 19))
                    e = try
                        prior(grid_prior(c, rd), rd)
                        nothing
                    catch err
                        err
                    end
                    @test e isa ArgumentError
                    @test occursin("$(n) observation(s) carry fewer", e.msg)
                    @test occursin("the fewest being 10 at observation 1", e.msg)
                end
            end
        end
        @testset "The name of the benchmark-weight field" begin
            # `bw` names the Panel Field the prior writes, so another name changes no value.
            # The oracle has no such keyword.
            sc(f) = CompositeExposure(; descriptors = [Passthrough(; field = f)],
                                      family = "style", bw = "bench_w")
            cfg = grid_config("ScoredBp05", rd)
            pr = prior(CrossSectionalFactorPrior(; lambda = 1, cfg..., bw = "bench_w",
                                                 factors = ["market" => ConstantExposure(),
                                                            "style1" => sc("style1"),
                                                            "style2" => sc("style2")]), rd)
            p0 = fits["ScoredBp05"]
            @test isequal(pr.mu, p0.mu) && isequal(pr.sigma, p0.sigma)
            @test isequal(pr.rr.bw, p0.rr.bw) && isequal(pr.rr.M, p0.rr.M)
            # A member that reads another field is refused.
            @test_throws ArgumentError prior(CrossSectionalFactorPrior(; lambda = 1, cfg...,
                                                                       bw = "bench_w"), rd)
        end
        @testset "The Factor Family Basis of $(c)" for c in
                                                       ("FamOne", "FamStated", "FamTwo")
            PO = PortfolioOptimisers
            pr = fits[c]
            fcb = pr.rr.fcb
            load(o) = parity_load("CrossSectionalFactorPrior", "Grid$(fix)$(c)", o)
            has(o) = isfile(parity_asset("CrossSectionalFactorPrior", "Grid$(fix)$(c)", o))
            # The factor return of row `t` was fitted on the exposures of row `t - 1`, so it is
            # written in the basis of that row, and the round trip through the reduced axis
            # is exact on those rows. Read against the basis of its own row, it is not: the
            # dropped member moves by 0.22 of the largest factor return on the small panel.
            T = size(pr.fpr.X, 1)
            fs = PO.factor_basis_slice(fcb, 1:(T - 1))
            g = PO.reduce_factor_returns(fs, pr.fpr.X[2:T, :])
            @test PO.expand_factor_returns(fs, g) == pr.fpr.X[2:T, :]
            has("BasisRatios") || continue
            # The same dropped member, automatic or stated, and the same ratios. Measured
            # maxrel 5.5e-16.
            @test PO.dropped_factor_indices(fcb) == Int.(vec(load("BasisDropped"))) .+ 1
            @test parity_compare(fcb.ratios, load("BasisRatios"); name = "$(c) ratios").ok
            has("BasisReducedLoadings") || continue
            # Every transform on the fitted outputs. The expansions read the ratios of the
            # second and the third observation, so a stale row would show. Measured maxrel
            # 6.2e-13 at most, on the covariance of two families.
            mu = PO.reduce_factor_mu(fcb, pr.fpr.mu)
            S = PO.reduce_factor_covariance(fcb, pr.fpr.sigma)
            @test parity_compare(PO.reduce_loadings(fcb, pr.rr.M),
                                 load("BasisReducedLoadings")).ok
            @test parity_compare(mu, vec(load("BasisReducedMu"))).ok
            @test parity_compare(PO.expand_factor_mu(fcb, mu, 2),
                                 vec(load("BasisExpandedMu1"))).ok
            # Measured maxrel cell by cell: 2.1e-13 on one CI host and 1.4e-12 on the other,
            # over 80 runs.
            @test parity_compare(S, load("BasisReducedCov"); rtol = 3e-12,
                                 name = "BasisReducedCov").ok
            # An entry of the expanded covariance near zero cancels: measured cell maxrel
            # 1.6e-11 and maxscaled 6.9e-14, so the check is `:array`.
            @test parity_compare(PO.expand_factor_covariance(fcb, S, 3),
                                 load("BasisExpandedCov2"); scale = :array,
                                 name = "BasisExpandedCov2").ok
            @test parity_compare(PO.project_factor_coordinates(fcb,
                                                               vec(load("BasisProjectInput"))),
                                 vec(load("BasisProject"))).ok
        end
    end
end

@testset "The default Unseen Member rule differs from the oracle's where the oracle's depends on the dropped member (#1606)" begin
    # The oracle keeps an Unseen Member in the zero-sum condition, so the row of its observation
    # is rank-deficient, and its pseudo-inverse answer depends on the dropped member. The default
    # rule gives the member a return of zero. Each grid case states the oracle's rule, and this
    # testset compares the default with it on the automatic and the stated drop of the industry
    # family. The only asset in "Utilities" of the small panel delists, which gives one such row.
    maxabs(a, b) = maximum(abs, a .- b)
    @testset "$(fix)" for (fix, fx) in (("Small", parity_small_panel()),
                                        ("Large", parity_large_panel()))
        rd = grid_fixture(fx)
        fit(name; kw...) = prior(CrossSectionalFactorPrior(; lambda = 1,
                                                           grid_config(name, rd)..., kw...),
                                 rd).rr
        za = fit("FamOne"; unseen = ZeroUnseenMember())
        zb = fit("FamStated"; unseen = ZeroUnseenMember())
        sa = fit("FamOne")
        sb = fit("FamStated")
        # The default does not depend on the dropped member at any row.
        @test maxabs(za.fr, zb.fr) < 1e-14
        # It differs from the oracle's rule only at the rows where the oracle's rule depends on
        # the dropped member.
        moved = findall(t -> maxabs(za.fr[t, :], sa.fr[t, :]) > 1e-13, axes(za.fr, 1))
        split = findall(t -> maxabs(sa.fr[t, :], sb.fr[t, :]) > 1e-13, axes(sa.fr, 1))
        @test moved == split
        if fix == "Small"
            @test !isempty(moved)
        end
        u = findfirst(==("industry=Utilities"), za.nf)
        @test all(iszero, za.fr[moved, u])
        rest = setdiff(axes(za.fr, 1), moved)
        @test maxabs(za.fr[rest, :], sa.fr[rest, :]) < 1e-13
    end
end
