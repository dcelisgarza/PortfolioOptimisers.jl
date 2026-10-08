#=
The Carry Rule of the Cross-Sectional Factor Prior (#1602, decided on #1590, ADR 0193).

`carry = FoldOrRefit()`, the default, folds each part that has a fold and fits every other part
again at each step, so the carry fold equals the batch fit at a cost that grows with the stream.
`carry = FoldOnly()` makes the constructor refuse a part whose step cost grows, and the error
names each such part. Every test reads the configuration alone, so the refusal comes before any
data. A configuration that `FoldOnly()` accepts folds as under `FoldOrRefit()`.
=#
include(joinpath(@__DIR__, "parity_harness.jl"))
include(joinpath(@__DIR__, "parity_grid.jl"))
# `Accessors.@set` expands when the test set is lowered, before a `using` inside it runs.
using Accessors: Accessors

# A user Descriptor that states no `lookback`. It reads its own row, but the prior cannot know it.
struct CarryRuleUserDescriptor <: PortfolioOptimisers.AbstractDescriptorEstimator
    field::String
end
function PortfolioOptimisers.descriptor(de::CarryRuleUserDescriptor, rd::ReturnsResult)
    return descriptor(Passthrough(; field = de.field), rd)
end

@testset "The Carry Rule of the Cross-Sectional Factor Prior (#1602)" begin
    po = PortfolioOptimisers
    rd = grid_fixture(parity_large_panel())
    edges = (0, 90, 170, 250)
    rows(r, i) = po.port_opt_view(r, i, :)
    # Equal cell by cell, a `NaN` equal to a `NaN`.
    same(a, b) = size(a) == size(b) && all(((x, y),) -> isequal(x, y) || x == y, zip(a, b))
    function agrees(x, b)
        return same(x.mu, b.mu) &&
               same(x.sigma, b.sigma) &&
               same(x.X, b.X) &&
               same(x.rr.csr.f, b.rr.csr.f) &&
               same(x.rr.vs, b.rr.vs) &&
               same(x.rr.Ms, b.rr.Ms)
    end
    # The prior after each step of a stream, and its call with no data at once.
    function stream(pe, e = edges)
        out = []
        for k in 1:(length(e) - 1)
            pe = partial_fit!(pe, rows(rd, (e[k] + 1):e[k + 1]))
            push!(out, (; pe = pe, pr = prior(pe)))
        end
        return out
    end
    batch(pe, k) = prior(pe, rows(rd, 1:edges[k + 1]))
    function message(f)
        return try
            f()
            ""
        catch err
            sprint(showerror, err)
        end
    end
    base = grid_config("Base", rd)
    wv = WindowedVariance(; ve = ExpWeightedVariance(; decay = 2.0^(-1 / 20), min_obs = 5),
                          window = 60)
    user = vcat(base.factors,
                ["user" =>
                     CompositeExposure(; descriptors = [CarryRuleUserDescriptor("style1")],
                                       outlier = nothing, scoring = nothing,
                                       family = "user")])

    @testset "The default is FoldOrRefit, and it accepts every part" begin
        @test CrossSectionalFactorPrior(; base...).carry === FoldOrRefit()
        @test isempty(po.carry_growing_parts(CrossSectionalFactorPrior(; base...)))
        pe = CrossSectionalFactorPrior(; base..., factors = user, ve = wv, th = 0.2,
                                       pe = EntropyPoolingPrior(; pe = GRID_PE))
        @test length(po.carry_growing_parts(pe)) == 4
        @test isnothing(po.assert_carry_rule(FoldOrRefit(), pe))
    end

    @testset "FoldOnly refuses each part that grows in the constructor, and names it" begin
        cases = [(; factors = vcat(base.factors, ["beta" => CompositeExposure(; descriptors = [EWMarketBeta()])])) => "the factor \"beta\" (CompositeExposure) has no finite look-back",
                 (; factors = user) => "the factor \"user\" (CompositeExposure) has no finite look-back",
                 grid_config("FcTarget", rd) => "the Return Forecast `rfe` (TargetReturnForecast) has no fold of its rows",
                 (; ofit = UnadjustedForecast(), rfe = ExpWeightedReturnForecast(; scores = DescriptorScores(; descriptors = [EWMomentum()]), half_life = 10.0)) => "the Return Forecast `rfe` (ExpWeightedReturnForecast) has no finite look-back",
                 (; ve = wv) => "`ve` (WindowedVariance) does not fold",
                 (; pe = EntropyPoolingPrior(; pe = GRID_PE)) => "the factor prior `pe` (EntropyPoolingPrior) does not fold",
                 (; pe = EmpiricalPrior(; ce = Covariance(; alg = SemiMoment()))) => "the factor prior `pe` (EmpiricalPrior) does not fold",
                 (; th = 0.2) => "`th = 0.2` estimates the idiosyncratic correlation again"]
        for (cfg, part) in cases
            m = message(() -> CrossSectionalFactorPrior(; base..., cfg...,
                                                        carry = FoldOnly()))
            @test startswith(m, "ArgumentError: carry = FoldOnly() refuses")
            @test occursin(part, m)
            @test occursin("Use carry = FoldOrRefit()", m)
            # The same configuration constructs under the default rule.
            @test isa(CrossSectionalFactorPrior(; base..., cfg...),
                      CrossSectionalFactorPrior)
        end
        # One error names every part that grows.
        m = message(() -> CrossSectionalFactorPrior(; base..., factors = user, ve = wv,
                                                    th = 0.2,
                                                    pe = EntropyPoolingPrior(;
                                                                             pe = GRID_PE),
                                                    carry = FoldOnly()))
        @test count("\n  - ", m) == 4
        for part in ("\"user\"", "`ve`", "`pe`", "`th = 0.2`")
            @test occursin(part, m)
        end
        # The predicate of the factor prior reads the members of an `EmpiricalPrior`.
        @test po.cross_sectional_factor_prior_folds(GRID_PE)
        @test po.cross_sectional_factor_prior_folds(EmpiricalPrior())
        @test !po.cross_sectional_factor_prior_folds(EmpiricalPrior(;
                                                                    ce = Covariance(;
                                                                                    alg = SemiMoment())))
        @test !po.cross_sectional_factor_prior_folds(EntropyPoolingPrior(; pe = GRID_PE))
    end

    @testset "A bounded window and a ve that folds pass FoldOnly, and the carry equals the batch fit" begin
        pe = CrossSectionalFactorPrior(; lambda = 1, base..., carry = FoldOnly())
        @test po.cross_sectional_carry_rows(pe) == 2
        s = stream(pe)
        @test all(k -> agrees(s[k].pr, batch(pe, k)), 1:3)
        # Every rebuild of the prior keeps the rule, and runs no check that the prior fails.
        @test all(x -> x.pe.carry === FoldOnly(), s)
        last_pe = s[end].pe
        @test (Accessors.@set last_pe.cache = nothing).carry === FoldOnly()
        @test po.copy_states(last_pe).carry === FoldOnly()
        # `partial_fit` rebuilds the prior around a copy of each state at each step.
        p = pe
        for k in 1:3
            p = partial_fit(p, rows(rd, (edges[k] + 1):edges[k + 1]))
        end
        @test p.carry === FoldOnly()
        @test agrees(prior(p), batch(pe, 3))
        # The refit under `Online` seeds a sample buffer in `cache`, and ignores the rule.
        o = po.update_online_estimator(Online(pe))
        o = partial_fit!(o, rows(rd, 1:250))
        @test agrees(prior(o), batch(pe, 3))
    end

    @testset "FoldOrRefit fits a user Descriptor with no lookback and a ve that does not fold again at each step" begin
        pe = CrossSectionalFactorPrior(; lambda = 1, base..., factors = user, ve = wv)
        @test isnothing(po.cross_sectional_carry_rows(pe))
        @test !po.supports_partial_fit(pe.ve)
        s = stream(pe)
        @test all(k -> agrees(s[k].pr, batch(pe, k)), 1:3)
    end
end
