using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Clarabel, Logging

# A universe that names the columns of the returns data in another order, or with a name the
# data does not have, binds each view, bound and group to the wrong column (#2016). Every
# entry that holds both the returns data and a tree with a `UniverseSets` refuses it.

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false))
X = randn(StableRNG(12), 300, 3) ./ 100
nx = ["A", "C", "B"]
rd = ReturnsResult(; nx = nx, X = X)
# A gap in one asset sends every head through the masked branch of its reduction.
Xg = copy(X)
Xg[1:5, 2] .= NaN
rdg = ReturnsResult(; nx = nx, X = Xg)
F = randn(StableRNG(7), 300, 2) ./ 100
rdf = ReturnsResult(; nx = nx, X = X, nf = ["F1", "F2"], F = F)
mv = LinearConstraintEstimator(; val = ["B == 0.004"])
fv = LinearConstraintEstimator(; val = ["F1 == 0.002"])
usets(names) = UniverseSets(; dict = Dict("nx" => names))
good = usets(nx)
order = usets(["A", "B", "C"])
alien = usets(["A", "B", "Z"])
short = usets(["A", "B"])
quietly(f) = with_logger(f, NullLogger())

@testset "A universe that does not name the columns of rd is refused" begin
    @testset "The message names the order, the names or the length" begin
        ep(s) = EntropyPoolingPrior(; sets = s, mu_views = mv)
        e = try
            prior(ep(order), rd)
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("the order differs, first at position 2: `B` vs `C`", e.msg)
        e = try
            prior(ep(alien), rd)
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("the names differ", e.msg)
        @test_throws DimensionMismatch prior(ep(short), rd)
        # The universe in the order of the data binds the view to the column that `rd` names.
        @test isapprox(prior(ep(good), rd).mu[3], 0.004; atol = 1e-6)
        # A universe checked on its own.
        @test_throws ArgumentError PortfolioOptimisers.assert_universe_axis_order(order, rd)
        @test isnothing(PortfolioOptimisers.assert_universe_axis_order(good, rd))
        # A list that is the vector of the data itself passes with no comparison.
        @test isnothing(PortfolioOptimisers.assert_universe_axis_order(usets(rd.nx), rd))
    end
    @testset "The walk reaches each prior that holds a universe" begin
        @test_throws ArgumentError prior(BlackLittermanPrior(; sets = order, views = mv),
                                         rd)
        # A vector of priors.
        op(s) = OpinionPoolingPrior(;
                                    pes = [EntropyPoolingPrior(; sets = s, mu_views = mv)])
        @test_throws ArgumentError prior(op(order), rd)
        @test prior(op(good), rd) isa PortfolioOptimisers.AbstractPriorResult
        # One universe that two estimators share.
        shared = InverseVolatility(; pe = EntropyPoolingPrior(; sets = good, mu_views = mv),
                                   sets = good)
        @test optimise(shared, rd) isa PortfolioOptimisers.OptimisationResult
    end
    @testset "A child fit on the factor returns names the factors of rd" begin
        fep(names) = EntropyPoolingPrior(; sets = usets(names), mu_views = fv)
        e = try
            quietly(() -> prior(FactorPrior(; pe = fep(["F2", "F1"])), rdf))
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("the factor universe declared under key `nx`", e.msg)
        @test occursin("to `rd.nf`", e.msg)
        @test quietly(() -> prior(FactorPrior(; pe = fep(["F1", "F2"])), rdf)) isa
              PortfolioOptimisers.AbstractPriorResult
        # The factor key of a universe names the factors of rd.
        fbl(names) = FactorBlackLittermanPrior(;
                                               sets = UniverseSets(;
                                                                   dict = Dict("nx" => nx,
                                                                               "nf" =>
                                                                                   names)),
                                               views = fv)
        @test_throws ArgumentError quietly(() -> prior(fbl(["F2", "F1"]), rdf))
        @test quietly(() -> prior(fbl(["F1", "F2"]), rdf)) isa
              PortfolioOptimisers.AbstractPriorResult
        abl(names) = AugmentedBlackLittermanPrior(;
                                                  sets = UniverseSets(;
                                                                      dict = Dict("nx" =>
                                                                                      nx,
                                                                                  "nf" =>
                                                                                      ["F1",
                                                                                       "F2"])),
                                                  a_views = LinearConstraintEstimator(;
                                                                                      val = ["A == 0.001"]),
                                                  f_views = LinearConstraintEstimator(;
                                                                                      val = ["F1 == 0.001"]),
                                                  f_pe = fep(names))
        @test_throws ArgumentError quietly(() -> prior(abl(["F2", "F1"]), rdf))
        @test quietly(() -> prior(abl(["F1", "F2"]), rdf)) isa
              PortfolioOptimisers.AbstractPriorResult
        # A FactorRiskContribution solves for the factor weights, so its universe names the
        # factors of rd under the asset key.
        frc(names) = FactorRiskContribution(; opt = JuMPOptimiser(; slv = slv),
                                            sets = usets(names))
        @test_throws ArgumentError quietly(() -> optimise(frc(["F2", "F1"]), rdf))
        @test quietly(() -> optimise(frc(["F1", "F2"]), rdf)) isa
              PortfolioOptimisers.OptimisationResult
        # A universe that names the factors under the asset key of an asset consumer.
        e = try
            prior(EntropyPoolingPrior(; sets = usets(rdf.nf), mu_views = mv), rdf)
        catch err
            err
        end
        @test e isa DimensionMismatch
        @test occursin("Declare the factors under the factor key `nf`", e.msg)
    end
    @testset "Each optimisation head refuses it, on full data and with a gap" begin
        heads(s) = [EqualWeighted(; sets = s), InverseVolatility(; sets = s),
                    MeanRisk(; opt = JuMPOptimiser(; slv = slv, sets = s)),
                    RiskBudgeting(; opt = JuMPOptimiser(; slv = slv),
                                  rba = AssetRiskBudgeting(; sets = s)),
                    HierarchicalRiskParity(; opt = HierarchicalOptimiser(; sets = s)),
                    Stacking(; sets = s, opti = [EqualWeighted(), InverseVolatility()],
                             opto = EqualWeighted()),
                    BudgetedHindsightPath(; L = 1.0, slv = slv, sets = s)]
        for r in (rd, rdg), (hb, hg) in zip(heads(order), heads(good))
            @test_throws ArgumentError quietly(() -> optimise(hb, r))
            @test quietly(() -> optimise(hg, r)) isa PortfolioOptimisers.OptimisationResult
        end
        # The walk stops at a nested head, which checks its own tree when it runs.
        @test optimise(EqualWeighted(; sets = good, fb = InverseVolatility(; sets = order)),
                       rd) isa PortfolioOptimisers.OptimisationResult
    end
    @testset "An online head refuses it" begin
        ops(s) = OnlinePortfolioSelection(; alg = ConstantRebalancedPortfolio(),
                                          set = BoundedAllocationSet(; sets = s))
        @test_throws ArgumentError optimise(ops(order), rd)
        @test optimise(ops(good), rd) isa PortfolioOptimisers.OptimisationResult
    end
    @testset "The axis lists" begin
        po = PortfolioOptimisers
        # A struct with `@vprop` fields lists them; any other struct lists its estimator fields.
        @test po.asset_axis_fields(EqualWeighted(; sets = good)) ==
              (:wb, :fees, :sets, :fb, :cache)
        @test :sets in po.asset_axis_fields(JuMPOptimiser(; slv = slv, sets = good))
        # The factor prior of a factor model is not on the asset axis.
        @test !(:pe in po.asset_axis_fields(FactorPrior()))
        @test po.factor_axis_fields(FactorPrior()) == (:pe,)
        @test po.factor_axis_fields(EqualWeighted()) == ()
    end
end
