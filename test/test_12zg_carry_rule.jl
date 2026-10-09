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

# A user factor prior that folds: it wraps a prior, and states `carry_folds` beside the fold and
# the read with no data (#1595).
struct CarryRuleUserPrior{P} <: PortfolioOptimisers.AbstractLowOrderPriorEstimator_A
    pe::P
end
function PortfolioOptimisers.prior(pe::CarryRuleUserPrior, X::PortfolioOptimisers.MatNum,
                                   args...; kwargs...)
    return prior(pe.pe, X; kwargs...)
end
function PortfolioOptimisers.prior(pe::CarryRuleUserPrior; kwargs...)
    return prior(pe.pe; kwargs...)
end
function PortfolioOptimisers.partial_fit!(pe::CarryRuleUserPrior,
                                          f::PortfolioOptimisers.MatNum)
    return CarryRuleUserPrior(partial_fit!(pe.pe, f))
end
function PortfolioOptimisers.carry_folds(pe::CarryRuleUserPrior)
    return PortfolioOptimisers.carry_folds(pe.pe)
end
# The same prior with no `carry_folds`, so the carry refits it.
struct NoCarryUserPrior{P} <: PortfolioOptimisers.AbstractLowOrderPriorEstimator_A
    pe::P
end
function PortfolioOptimisers.prior(pe::NoCarryUserPrior, X::PortfolioOptimisers.MatNum,
                                   args...; kwargs...)
    return prior(pe.pe, X; kwargs...)
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
    # A covariance estimator with no fold, which the idiosyncratic correlation reads under
    # `th > 0`.
    semi = Covariance(; alg = SemiMoment())
    user = vcat(base.factors,
                ["user" =>
                     CompositeExposure(; descriptors = [CarryRuleUserDescriptor("style1")],
                                       outlier = nothing, scoring = nothing,
                                       family = "user")])

    @testset "The default is FoldOrRefit, and it accepts every part" begin
        @test CrossSectionalFactorPrior(; base...).carry === FoldOrRefit()
        @test isempty(po.carry_growing_parts(CrossSectionalFactorPrior(; base...)))
        pe = CrossSectionalFactorPrior(; base..., factors = user, ve = wv, th = 0.2,
                                       ce = semi, pe = EntropyPoolingPrior(; pe = GRID_PE))
        @test length(po.carry_growing_parts(pe)) == 4
        @test isnothing(po.assert_carry_rule(FoldOrRefit(), pe))
    end

    @testset "FoldOnly refuses each part that grows in the constructor, and names it" begin
        # `EWMarketBeta` folds from a state since #1608, so it no longer makes the carry grow.
        beta = vcat(base.factors,
                    ["beta" => CompositeExposure(; descriptors = [EWMarketBeta()])])
        @test isa(CrossSectionalFactorPrior(; base..., factors = beta, carry = FoldOnly()),
                  CrossSectionalFactorPrior)
        cases = [(; factors = user) => "the factor \"user\" (CompositeExposure) has no finite look-back",
                 grid_config("FcTarget", rd) => "the Return Forecast `rfe` (TargetReturnForecast) has no fold of its rows",
                 # A user Descriptor states no look-back and carries no state. `EWMomentum`
                 # folds from a state since #1586, so it no longer makes the carry grow.
                 (; ofit = UnadjustedForecast(), rfe = ExpWeightedReturnForecast(; scores = DescriptorScores(; descriptors = [CarryRuleUserDescriptor("style1")]), half_life = 10.0)) => "the Return Forecast `rfe` (ExpWeightedReturnForecast) has no finite look-back",
                 (; ve = wv) => "`ve` (WindowedVariance) does not fold",
                 (; pe = EntropyPoolingPrior(; pe = GRID_PE)) => "the factor prior `pe` (EntropyPoolingPrior) does not fold",
                 (; pe = EmpiricalPrior(; ce = Covariance(; alg = SemiMoment()))) => "the factor prior `pe` (EmpiricalPrior) does not fold",
                 (; th = 0.2, ce = semi) => "`th = 0.2` estimates the idiosyncratic correlation again over every row, because `ce` (Covariance) does not fold"]
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
                                                    th = 0.2, ce = semi,
                                                    pe = EntropyPoolingPrior(;
                                                                             pe = GRID_PE),
                                                    carry = FoldOnly()))
        @test count("\n  - ", m) == 4
        for part in ("\"user\"", "`ve`", "`pe`", "`th = 0.2`")
            @test occursin(part, m)
        end
        # The verb of the factor prior reads the members of an `EmpiricalPrior` (#1595).
        @test Base.ispublic(po, :carry_folds)
        @test po.carry_folds(GRID_PE)
        @test po.carry_folds(EmpiricalPrior())
        @test !po.carry_folds(EmpiricalPrior(; ce = Covariance(; alg = SemiMoment())))
        @test !po.carry_folds(EntropyPoolingPrior(; pe = GRID_PE))
        # A sample buffer refits over every row, so it does not fold on the carry.
        @test !po.carry_folds(po.update_online_estimator(Online(EmpiricalPrior())))
        # The verb is not `supports_partial_fit`, which keeps its answer for the prior.
        @test !po.supports_partial_fit(EmpiricalPrior())
    end

    @testset "A user factor prior that states carry_folds folds on the carry and passes FoldOnly (#1595)" begin
        up = CarryRuleUserPrior(GRID_PE)
        @test po.carry_folds(up)
        pe = CrossSectionalFactorPrior(; lambda = 1, base..., pe = up, carry = FoldOnly())
        @test isempty(po.carry_growing_parts(pe))
        s = stream(pe)
        @test all(k -> agrees(s[k].pr, batch(pe, k)), 1:3)
        # The carry folded the prior, so the state of the carry holds the state of the prior.
        @test all(x -> isa(x.pe.cache.pe.pe.cache, po.PriorCarryState), s)
        # The same prior without the verb is refused, and its carry refits it.
        np = NoCarryUserPrior(GRID_PE)
        @test !po.carry_folds(np)
        m = message(() -> CrossSectionalFactorPrior(; base..., pe = np, carry = FoldOnly()))
        @test occursin("the factor prior `pe` (NoCarryUserPrior) does not fold", m)
        @test occursin("It needs a method of `carry_folds`", m)
        pn = CrossSectionalFactorPrior(; lambda = 1, base..., pe = np)
        sn = stream(pn)
        @test all(k -> agrees(sn[k].pr, batch(pn, k)), 1:3)
        @test all(x -> isnothing(x.pe.cache.pe.pe.cache), sn)
    end

    @testset "FoldOnly refuses a BatchChoice with an automatic member (#1605)" begin
        # A move folds the factor prior again over every carried factor return, so its cost
        # grows with the stream. The message names both ways out.
        for name in ("FamOne", "FamTwo", "NeutFam")
            m = message(() -> CrossSectionalFactorPrior(; grid_config(name, rd)...,
                                                        carry = FoldOnly()))
            @test startswith(m, "ArgumentError: carry = FoldOnly() refuses")
            @test occursin("`choice = BatchChoice()` chooses the dropped member", m)
            @test occursin("Set `choice = PinnedChoice()`", m)
            @test occursin("\"industry\" => \"industry=<member>\"", m)
            @test count("\n  - ", m) == 1
            @test isa(CrossSectionalFactorPrior(; grid_config(name, rd)...,
                                                choice = PinnedChoice(),
                                                carry = FoldOnly()),
                      CrossSectionalFactorPrior)
        end
        @test occursin("[\"industry\", \"region\"]",
                       message(() -> CrossSectionalFactorPrior(;
                                                               grid_config("FamTwo", rd)...,
                                                               carry = FoldOnly())))
        # A named member never moves, and a prior with no family has no choice to make.
        @test isa(CrossSectionalFactorPrior(; grid_config("FamStated", rd)...,
                                            carry = FoldOnly()), CrossSectionalFactorPrior)
        @test isempty(po.carry_choice_parts(BatchChoice(), nothing))
        @test isempty(po.carry_choice_parts(BatchChoice(),
                                            ["industry" => "industry=Banks"]))
        @test isempty(po.carry_choice_parts(PinnedChoice(), ["industry" => nothing]))
        @test length(po.carry_choice_parts(BatchChoice(),
                                           ["industry" => "industry=Banks",
                                            "region" => nothing])) == 1
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
        # The default `ce` folds the idiosyncratic correlation, so a threshold passes (#1594).
        pt = CrossSectionalFactorPrior(; lambda = 1, base..., th = 0.2, carry = FoldOnly())
        @test isempty(po.carry_growing_parts(pt))
        st = stream(pt)
        @test all(k -> agrees(st[k].pr, batch(pt, k)), 1:3)
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
        # An `EmpiricalPrior` whose `ce` does not fold is refitted over every factor return.
        # Its moments differ from the batch fit by round-off alone, about 1e-19, as those of a
        # folded `EmpiricalPrior()` do. The factor returns are equal to the bit.
        pf = CrossSectionalFactorPrior(; lambda = 1, base...,
                                       pe = EmpiricalPrior(;
                                                           ce = Covariance(;
                                                                           alg = SemiMoment())))
        sf = stream(pf)
        near(a, b) = size(a) == size(b) &&
                     all(((x, y),) -> isequal(x, y) || isapprox(x, y; atol = 1e-15),
                         zip(a, b))
        @test all(1:3) do k
            x, b = sf[k].pr, batch(pf, k)
            return near(x.mu, b.mu) &&
                   near(x.sigma, b.sigma) &&
                   same(x.X, b.X) &&
                   same(x.rr.csr.f, b.rr.csr.f) &&
                   same(x.rr.vs, b.rr.vs)
        end
        @test all(x -> isnothing(x.pe.cache.pe.cache), sf)
    end
end
