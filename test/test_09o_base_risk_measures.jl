# The sweep of `src/16_RiskMeasures/01_Base_RiskMeasures.jl` (#596): the four scalarisers
# against their sums, maxima and log-sum-exps computed by hand, the two relaxations against
# the value they bound, the three bound conversions, a deferred co-moment against a hand fit,
# the centre a deferred tensor is taken about, the four family traits over every concrete
# measure, and the refusals and pass-throughs no other file reaches.
using Clarabel, InteractiveUtils, JuMP, LinearAlgebra, Statistics, StableRNGs

const PO = PortfolioOptimisers

const _X596 = randn(StableRNG(596), 120, 5) .* 0.02 .+ 0.001
const _pr596 = prior(EmpiricalPrior(), _X596)
const _slv596 = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                       settings = Dict("verbose" => false, "tol_gap_abs" => 1e-10,
                                       "tol_gap_rel" => 1e-10, "tol_feas" => 1e-10),
                       check_sol = (; allow_local = true, allow_almost = true))

# Three measures of degree one, each with its own `scale`, so a scalariser that reads the
# wrong weight or the wrong element gives a different number.
function _measures596()
    return [StandardDeviation(; settings = RiskMeasureSettings(; scale = 2.0)),
            ConditionalValueatRisk(; settings = RiskMeasureSettings(; scale = 1.0)),
            MaximumDrawdown(; settings = RiskMeasureSettings(; scale = 0.5))]
end

# The scaled value of each measure at `w`, computed one measure at a time.
function _scaled596(rs, w)
    return [expected_risk(r, w, _pr596) * r.settings.scale for r in rs]
end

# The value the model holds in `risk`, and the value of the scalarised measures at the
# optimal weights.
function _model_and_value596(rs, sca, obj)
    opt = JuMPOptimiser(; pe = _pr596, slv = _slv596, sca = sca)
    sol = optimise(MeanRisk(; r = rs, obj = obj, opt = opt))
    rf = [factory(r, _pr596, _slv596) for r in rs]
    return JuMP.value(sol.model[:risk]), expected_risk(rf, sol.w, _pr596; sca = sca),
           _scaled596(rf, sol.w)
end

@testset "Base risk measures" begin
    @testset "The four scalarisers combine the scaled values" begin
        rs = [factory(r, _pr596, _slv596) for r in _measures596()]
        w = fill(0.2, 5)
        v = _scaled596(rs, w)
        @test expected_risk(rs, w, _pr596; sca = SumScalariser()) ≈ sum(v)
        @test expected_risk(rs, w, _pr596; sca = MaxScalariser()) ≈ maximum(v)
        @test expected_risk(rs, w, _pr596; sca = MinScalariser()) ≈ minimum(v)
        for g in (0.5, 50.0)
            lse = expected_risk(rs, w, _pr596; sca = LogSumExpScalariser(; gamma = g))
            @test lse ≈ log(sum(exp.(g .* v))) / g
            # The bound of the `## What gamma does` block.
            @test maximum(v) <= lse <= maximum(v) + log(3) / g
        end
        # The upper end of the bound is reached when every scaled value is equal.
        lse_eq = PO.scalarise(identity, LogSumExpScalariser(; gamma = 2.0), [0.3, 0.3, 0.3])
        @test lse_eq ≈ 0.3 + log(3) / 2
        # A small `gamma` gives the mean once the divergent term is taken away, never the sum.
        xs = [0.01, 0.03, 0.05]
        small = PO.scalarise(identity, LogSumExpScalariser(; gamma = 1e-5), xs)
        @test small - log(3) / 1e-5 ≈ mean(xs) atol = 1e-8
        @test !isapprox(small, sum(xs); rtol = 0.1)
        # A `by` key picks one whole value, and a tie keeps the earlier element.
        vals = [(1.0, :a), (3.0, :b), (3.0, :c), (1.0, :d)]
        @test PO.scalarise(identity, MaxScalariser(), vals; by = first) == (3.0, :b)
        @test PO.scalarise(identity, MinScalariser(), vals; by = first) == (1.0, :a)
        # Without a key the two pick slot by slot.
        tups = [(1.0, [4.0, 0.0]), (2.0, [3.0, 5.0])]
        @test PO.scalarise(identity, MaxScalariser(), tups) == (2.0, [4.0, 5.0])
        @test PO.scalarise(identity, MinScalariser(), tups) == (1.0, [3.0, 0.0])
        @test PO.scalarise(identity, SumScalariser(), tups) == (3.0, [7.0, 5.0])
        lse_t = PO.scalarise(identity, LogSumExpScalariser(), tups)
        @test lse_t[1] ≈ log(exp(1) + exp(2))
        @test lse_t[2] ≈ [log(exp(4) + exp(3)), log(exp(0) + exp(5))]
    end

    @testset "The model holds the aggregation under a minimum-risk objective" begin
        for sca in (SumScalariser(), MaxScalariser(), LogSumExpScalariser(; gamma = 50.0))
            m, v, _ = _model_and_value596(_measures596(), sca, MinimumRisk())
            @test m ≈ v rtol = 1e-6
        end
        # The smoothed maximum stands above the maximum by no more than log N / gamma.
        m, _, s = _model_and_value596(_measures596(), LogSumExpScalariser(; gamma = 50.0),
                                      MinimumRisk())
        @test maximum(s) < m <= maximum(s) + log(3) / 50
    end

    @testset "Under a maximum-return objective the model reports an upper bound" begin
        # The epigraph variables of the three measures stand above their values, so even the
        # sum reports more than the aggregation.
        for sca in (SumScalariser(), MaxScalariser(), LogSumExpScalariser(; gamma = 50.0))
            m, v, _ = _model_and_value596(_measures596(), sca, MaximumReturn())
            @test m > v * (1 + 1e-3)
        end
        # Two measures with exact quadratic expressions: the sum adds no gap of its own.
        rq = [Variance(; alg = PO.QuadRiskExpr()),
              Variance(; alg = PO.QuadRiskExpr(),
                       settings = RiskMeasureSettings(; scale = 2.0))]
        m, v, _ = _model_and_value596(rq, SumScalariser(), MaximumReturn())
        @test m ≈ v rtol = 1e-8
    end

    @testset "The three bounds convert one number three ways" begin
        ub = 0.04
        @test PO.variance_risk_bounds_val(SquareRootBound(), ub) ≈ 0.2
        @test PO.variance_risk_bounds_val(LinearBound(), ub) == 0.04
        @test PO.variance_risk_bounds_val(SquaredBound(), ub) ≈ 0.0016
        @test PO.variance_risk_bounds_val(SquareRootBound(), [ub, 0.09]) ≈ [0.2, 0.3]
        @test PO.variance_risk_bounds_val(SquaredBound(), [ub, 0.09]) ≈ [0.0016, 0.0081]
        @test isnothing(PO.variance_risk_bounds_val(SquaredBound(), nothing))
        f = PO.variance_risk_bounds_val(SquaredBound(), Frontier(; N = 7))
        @test (f.N, f.factor, f.bound) == (7, 1, SquaredBound())
    end

    @testset "A deferred co-moment equals a hand fit about the sample mean" begin
        X = _X596
        T = size(X, 1)
        mu = vec(mean(X; dims = 1))
        Xc = X .- transpose(mu)
        kt = sum(kron(Xc[t, :], Xc[t, :]) * transpose(kron(Xc[t, :], Xc[t, :]))
                 for t in 1:T) / T
        r = PO.resolve_deferred_quantities(Kurtosis(; kt = Cokurtosis()), _pr596)
        @test r.kt ≈ kt
        # The functor reads the matrix, and the sample cokurtosis about the mean of each
        # asset gives the fourth moment of the portfolio returns about their mean.
        w = fill(0.2, 5)
        @test r(w)^2 ≈ mean((X * w .- dot(w, mu)) .^ 4)
        # A measure that carries the quantity is returned as it is.
        stated = Kurtosis(; kt = kt)
        @test PO.resolve_deferred_quantities(stated, _pr596) === stated
    end

    @testset "The four traits answer for every concrete measure" begin
        function concrete(T)
            acc = Any[]
            for S in InteractiveUtils.subtypes(T)
                isabstracttype(S) ? append!(acc, concrete(S)) : push!(acc, S)
            end
            return acc
        end
        Ts = concrete(PO.AbstractBaseRiskMeasure)
        @test !isempty(Ts)
        for f in (PO.bigger_is_better, PO.needs_previous_weights), T in Ts
            @test Bool <: reduce(typejoin, Base.return_types(f, (T,)); init = Union{})
        end
        # Every measure that a call with no arguments builds answers all three predicates.
        # Four composites route through their own `expected_risk` methods and declare no
        # input kind, so the fallback of `risk_input_kind` refuses them. `ExpectedReturn`
        # declares one (#1917).
        explicit = (:ExpectedReturnRiskRatio, :MeanReturnRiskRatio,
                    :NonOptimisationRiskRatio, :RiskRatio)
        built = 0
        for T in Ts
            r = try
                Base.typename(T).wrapper()
            catch
                continue
            end
            built += 1
            @test PO.bigger_is_better(r) isa Bool
            @test PO.needs_previous_weights(r) isa Bool
            @test PO.supports_precomputed_returns(r) isa Bool
            if nameof(T) in explicit
                @test_throws ArgumentError PO.risk_input_kind(r)
            else
                @test PO.risk_input_kind(r) isa PO.RiskInputKind
            end
        end
        @test built >= 50
    end

    @testset "A vector of measures answers the predicate by all" begin
        @test PO.supports_precomputed_returns([ConditionalValueatRisk(), MaximumDrawdown()])
        @test !PO.supports_precomputed_returns([ConditionalValueatRisk(), Variance()])
        @test PO.needs_previous_weights([ConditionalValueatRisk(),
                                         TurnoverRiskMeasure(; w = fill(0.2, 5))])
        @test !PO.needs_previous_weights([ConditionalValueatRisk(), Variance()])
    end

    @testset "The fallbacks refuse a measure that declares nothing" begin
        @test_throws ArgumentError PO.risk_input_kind(RiskRatio())
        # `VarianceSkewKurtosis` carries a weights-only variance and has no form on a bare
        # return series.
        @test_throws ArgumentError VarianceSkewKurtosis()(_X596[:, 1])
        # A `WeightsReturnsFeesInput` measure that declares no predicate meets the tripwire.
        # The type lives in a module no name binds, so `subtypes` never reaches it and the
        # census of `test_09_risk_measures.jl` cannot see it.
        m = Module(:Tripwire596)
        Core.eval(m, :(struct Trip <: $(PO.RiskMeasure) end))
        Core.eval(m, :($(PO).risk_input_kind(::Trip) = $(PO.WeightsReturnsFeesInput())))
        @test_throws ArgumentError PO.supports_precomputed_returns(Core.eval(m, :(Trip())))
        @test !any(T -> nameof(T) === :Trip, InteractiveUtils.subtypes(PO.RiskMeasure))
    end

    @testset "A moment target is weight-independent only for nothing or a scalar" begin
        @test PO.weight_independent_target(nothing)
        @test PO.weight_independent_target(0.01)
        @test PO.weight_independent_target(PO.MeanCentering())
        @test !PO.weight_independent_target([0.01, 0.02])
        @test PO.supports_precomputed_returns(PO.NetReturnsInput(), nothing)
        @test !PO.supports_precomputed_returns(PO.WeightsInput(), nothing)
    end

    @testset "The selectors pass a stated value through and view an array" begin
        v = [1.0, 2.0, 3.0]
        @test PO.risk_measure_nothing_scalar_array_view(v, nothing, 2:3) == [2.0, 3.0]
        @test PO.risk_measure_nothing_scalar_array_view(nothing, v, [1, 3]) == [1.0, 3.0]
        @test PO.risk_measure_nothing_scalar_array_view(0.5, v, 1:2) == 0.5
        ucs = BoxUncertaintySet(; lb = [0.0, 0.0], ub = [1.0, 1.0])
        ucs2 = BoxUncertaintySet(; lb = [0.0, 0.0], ub = [2.0, 2.0])
        @test PO.sel(ucs, ucs2) === ucs
        @test PO.sel(nothing, ucs2) === ucs2
        # `_ctx` finds the one solver in the variadic tail, and answers `nothing` without one.
        @test PO._ctx(1, :a, _slv596) === _slv596
        @test isnothing(PO._ctx(1, :a))
    end
end
