# Tests for the custom JuMP objective/constraint extension points.
#
# `cobj`/`ccnt` on a `JuMPOptimiser` are the user-facing escape hatch: subtype
# `CustomJuMPObjective`/`CustomJuMPConstraint` and implement one builder method. Nothing else
# in the suite exercises them, so this file pins the contract:
#   - a single estimator reaches the model and moves the solution,
#   - a VECTOR of them applies every element (both vector paths were broken: a `ccnt` vector
#     silently hit the no-op `args...` fallback, a `cobj` vector raised a MethodError),
#   - the `get_k` idiom is load-bearing under a ratio objective,
#   - the hook still fires through `NearOptimalCentering`, whose objective builder calls it
#     from its own site and must pass arguments in the main-path order,
#   - a subtype with no method defined stays a documented no-op.
#
# The custom types must be defined at module top level (a `struct` cannot be declared inside
# the `@testset` block).
using Test, PortfolioOptimisers, CSV, TimeSeries, Clarabel, JuMP, LinearAlgebra, Statistics,
      Logging
const PO = PortfolioOptimisers

# A soft reward on a per-asset score. Every objective exercised here is a MINIMISATION
# (`MinimumRisk`, and NOC's centring objective), so the reward is subtracted.
struct ScoreTilt{T1, T2} <: PO.CustomJuMPObjective
    score::T1
    lambda::T2
end
function PO.add_custom_objective_term!(model::JuMP.Model, obj, pret, cobj::ScoreTilt,
                                       obj_expr, opt, pr, args...)
    w = PO.get_w(model)
    JuMP.add_to_expression!(obj_expr, -cobj.lambda * (cobj.score' * w))
    return nothing
end

# A hard floor on the same score exposure, following the model idiom: scale by the constraint
# scale, and lift the constant bound to `k`.
struct ScoreFloor{T1, T2} <: PO.CustomJuMPConstraint
    score::T1
    floor::T2
end
function PO.add_custom_constraint!(model::JuMP.Model, ccnt::ScoreFloor, opt, attrs)
    w = PO.get_w(model)
    k = PO.get_k(model)
    sc = PO.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (ccnt.score' * w - ccnt.floor * k) >= 0)
    return nothing
end

# The floor's mirror image, used to prove a `ccnt` vector applies *both* elements.
struct ScoreCap{T1, T2} <: PO.CustomJuMPConstraint
    score::T1
    cap::T2
end
function PO.add_custom_constraint!(model::JuMP.Model, ccnt::ScoreCap, opt, attrs)
    w = PO.get_w(model)
    k = PO.get_k(model)
    sc = PO.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (ccnt.cap * k - ccnt.score' * w) >= 0)
    return nothing
end

# The same floor with the `* k` omitted — correct only where `k == 1`.
struct ScoreFloorNoK{T1, T2} <: PO.CustomJuMPConstraint
    score::T1
    floor::T2
end
function PO.add_custom_constraint!(model::JuMP.Model, ccnt::ScoreFloorNoK, opt, attrs)
    w = PO.get_w(model)
    sc = PO.get_constraint_scale(model)
    JuMP.@constraint(model, sc * (ccnt.score' * w - ccnt.floor) >= 0)
    return nothing
end

# Records the arguments the objective hook is handed, without touching `obj_expr`. Used to
# pin the calling convention on every path that invokes the hook.
struct SpyObjective{T} <: PO.CustomJuMPObjective
    log::T
end
function PO.add_custom_objective_term!(model::JuMP.Model, obj, pret, cobj::SpyObjective,
                                       obj_expr, opt, pr, args...)
    push!(cobj.log, (; obj, pret, obj_expr, opt))
    return nothing
end

# Subtypes that never define a builder method: they must hit the no-op fallback.
struct InertConstraint <: PO.CustomJuMPConstraint end
struct InertObjective <: PO.CustomJuMPObjective end

@testset "custom JuMP objective/constraint hooks" begin
    rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "./assets/SP500.csv.gz"));
                                     timestamp = :Date)[(end - 252):end])
    pr = prior(EmpiricalPrior(), rd)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 check_sol = (; allow_local = true, allow_almost = true),
                 settings = "verbose" => false)
    rf = 4.2 / 100 / 252

    # Standardised trailing-63-day momentum: a continuous per-asset preference.
    score = let m = vec(sum(rd.X[(end - 62):end, :]; dims = 1))
        (m .- sum(m) / length(m)) ./ std(m)
    end
    exposure(res) = dot(score, res.w)

    minrisk(; kwargs...) = optimise(MeanRisk(; obj = MinimumRisk(),
                                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                 kwargs...)))
    ratio(; kwargs...) = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                               kwargs...)))

    base = minrisk()
    base_exp = exposure(base)

    @testset "objective hook: a single term reaches the objective" begin
        tilted = minrisk(; cobj = ScoreTilt(score, 1e-4))
        # The tilt rewards score exposure, so it must rise above the untilted baseline.
        @test exposure(tilted) > base_exp + 1e-3
    end

    @testset "objective hook: a vector composes additively" begin
        # Regression: a `cobj` vector used to raise a MethodError, because every
        # `set_portfolio_objective_function!` typed `cobj::Option{<:CustomJuMPObjective}`.
        # Two λ terms accumulate into the same expression, so they must equal one 2λ term.
        two = minrisk(; cobj = [ScoreTilt(score, 1e-4), ScoreTilt(score, 1e-4)])
        one = minrisk(; cobj = ScoreTilt(score, 2e-4))
        @test isapprox(exposure(two), exposure(one); rtol = 1e-5)
        @test exposure(two) > base_exp + 1e-3
    end

    @testset "constraint hook: a single constraint binds at its bound" begin
        f = base_exp + 0.5                     # above the free optimum, so it must bind
        res = minrisk(; ccnt = ScoreFloor(score, f))
        @test isapprox(exposure(res), f; atol = 1e-4)
    end

    @testset "constraint hook: a vector applies every element" begin
        # Regression: a `ccnt` vector used to hit the no-op `args...` fallback and be
        # SILENTLY DROPPED — the solve returned the unconstrained weights.
        lo = base_exp + 0.3
        hi = lo + 0.2
        res = minrisk(; ccnt = [ScoreFloor(score, lo), ScoreCap(score, hi)])
        e = exposure(res)
        @test e >= lo - 1e-4                   # the floor element was applied
        @test e <= hi + 1e-4                   # ... and so was the cap element
        @test e > base_exp + 1e-3              # the pair genuinely moved the solution
    end

    @testset "constraint hook: the `k` idiom is load-bearing under a ratio objective" begin
        # `MaximumRatio` solves in a rescaled space (`w_real == w / k`), so a constant bound
        # must be multiplied by `k`. Under `MinimumRisk` `k == 1` and both spellings agree —
        # which is exactly why omitting it hides.
        f_min = base_exp + 0.3
        @test isapprox(exposure(minrisk(; ccnt = ScoreFloor(score, f_min))),
                       exposure(minrisk(; ccnt = ScoreFloorNoK(score, f_min))); rtol = 1e-5)

        # Under the ratio objective only the k-scaled floor honours the requested level.
        ratio_exp = exposure(ratio())
        f = ratio_exp + 0.02                   # just above the free optimum: binds, stays feasible
        right = ratio(; ccnt = ScoreFloor(score, f))
        wrong = ratio(; ccnt = ScoreFloorNoK(score, f))
        @test isapprox(exposure(right), f; atol = 1e-4)
        @test !isapprox(exposure(wrong), f; atol = 1e-3)
    end

    @testset "objective hook is called with the documented argument order" begin
        # The contract is `(model, obj, pret, cobj, obj_expr, opt, pr, args...)`. A caller that
        # omits `obj`/`pret` shifts `cobj` out of position 4, so the user's method silently
        # misses dispatch and the custom term vanishes without a word. `NearOptimalCentering`
        # invokes the hook from its own objective builder and used to do exactly that, so
        # assert the convention holds on both paths. Only the *constrained* NOC algorithm
        # routes through that call.
        #
        # This is deliberately solver-free: the spy records the call during model assembly, so
        # it pins the calling convention even where the subsequent solve is ill-conditioned.
        function spy_log(build)
            log = Any[]
            with_logger(NullLogger()) do
                return build(SpyObjective(log))
            end
            return log
        end

        main = spy_log(spy -> minrisk(; cobj = spy))
        @test length(main) == 1
        @test main[1].obj isa PO.ObjectiveFunction
        @test main[1].pret isa PO.JuMPReturnsEstimator
        @test main[1].obj_expr isa JuMP.AbstractJuMPScalar

        noc = spy_log(spy -> optimise(NearOptimalCentering(; r = StandardDeviation(),
                                                           obj = MinimumRisk(),
                                                           alg = ConstrainedNearOptimalCentering(),
                                                           opt = JuMPOptimiser(; pe = pr,
                                                                               slv = slv,
                                                                               cobj = spy))))
        # NOC also solves `MeanRisk` sub-problems for its reference points, and those reach the
        # hook through the main path — so a non-empty log proves nothing. The centring
        # objective's own call is the one under test, and it is the only invocation that passes
        # the `JuMPOptimiser` (rather than an optimiser estimator) in the `opt` slot.
        @test any(e -> isa(e.opt, JuMPOptimiser), noc)   # regression: this call used to miss dispatch
        @test all(e -> e.obj isa PO.ObjectiveFunction, noc)
        @test all(e -> e.pret isa PO.JuMPReturnsEstimator, noc)
        @test all(e -> e.obj_expr isa JuMP.AbstractJuMPScalar, noc)
    end

    @testset "fallback: a subtype with no builder method is a no-op" begin
        # The documented fallback: subtyping without implementing the method changes nothing.
        res = minrisk(; ccnt = InertConstraint(), cobj = InertObjective())
        @test isapprox(res.w, base.w; atol = 1e-6)
    end
end
