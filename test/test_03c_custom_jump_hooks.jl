# Tests for the custom JuMP objective/constraint extension points.
#
# `cobj`/`ccnt` on a `JuMPOptimiser` are the user-facing escape hatch: subtype
# `CustomJuMPObjective`/`CustomJuMPConstraint` and implement one builder method. Nothing else
# in the suite exercises them, so this file pins the contract:
#   - a single estimator reaches the model and moves the solution,
#   - a VECTOR of them applies every element (both vector paths were broken: a `ccnt` vector
#     silently hit the no-op `args...` fallback, a `cobj` vector raised a MethodError),
#   - the `get_k` idiom is load-bearing under a ratio objective,
#   - the hook still fires through `NearOptimalCentering`, which calls it from its own
#     objective builder rather than through `set_portfolio_objective_function!`,
#   - a subtype with no method defined RAISES rather than silently contributing nothing,
#   - the same term is sense-correct under a minimisation and a maximisation with no
#     sign dispatch by the implementer, and a QUADRATIC term works against an AFFINE
#     objective — both consequences of routing custom terms through the objective penalty
#     (ADR 0036).
#
# The custom types must be defined at module top level (a `struct` cannot be declared inside
# the `@testset` block).
using Test, PortfolioOptimisers, CSV, TimeSeries, Clarabel, JuMP, LinearAlgebra, Statistics,
      Logging
const PO = PortfolioOptimisers

# A soft reward on a per-asset score. Contributed as a NEGATIVE penalty: the library applies
# the factor matching each objective's optimisation sense, so this one definition rewards
# score exposure under a minimisation and a maximisation alike.
struct ScoreTilt{T1, T2} <: PO.CustomJuMPObjective
    score::T1
    lambda::T2
end
function PO.add_custom_objective_term!(model::JuMP.Model, obj, cobj::ScoreTilt, optimiser,
                                       attrs)
    w = PO.get_w(model)
    PO.add_to_objective_penalty!(model, -cobj.lambda * (cobj.score' * w))
    return nothing
end

# A quadratic concentration penalty. Before ADR 0036 this was impossible against an affine
# objective: the hook mutated `obj_expr` in place, and `add_to_expression!(::AffExpr,
# ::QuadExpr)` is a MethodError. The penalty accumulator promotes instead.
struct SpreadPenalty{T} <: PO.CustomJuMPObjective
    lambda::T
end
function PO.add_custom_objective_term!(model::JuMP.Model, obj, cobj::SpreadPenalty,
                                       optimiser, attrs)
    w = PO.get_w(model)
    PO.add_to_objective_penalty!(model, cobj.lambda * JuMP.@expression(model, dot(w, w)))
    return nothing
end

# A hard floor on the same score exposure, following the model idiom: scale by the constraint
# scale, and lift the constant bound to `k`.
struct ScoreFloor{T1, T2} <: PO.CustomJuMPConstraint
    score::T1
    floor::T2
end
function PO.add_custom_constraint!(model::JuMP.Model, ccnt::ScoreFloor, optimiser, attrs)
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
function PO.add_custom_constraint!(model::JuMP.Model, ccnt::ScoreCap, optimiser, attrs)
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
function PO.add_custom_constraint!(model::JuMP.Model, ccnt::ScoreFloorNoK, optimiser, attrs)
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
function PO.add_custom_objective_term!(model::JuMP.Model, obj, cobj::SpyObjective,
                                       optimiser, attrs)
    push!(cobj.log, (; obj, optimiser, attrs))
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

    @testset "objective hook: the same term is sense-correct without sign dispatch" begin
        # ADR 0036: a term contributed to the objective penalty is oriented by the library,
        # so ONE definition rewards score exposure under a minimisation and a maximisation
        # alike. Before this, the implementer had to dispatch a sign on the objective type
        # and get it right, and `MaximumRatio` had no correct fixed sign at all.
        util(; kwargs...) = optimise(MeanRisk(; obj = MaximumUtility(),
                                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                  kwargs...)))
        @test exposure(util(; cobj = ScoreTilt(score, 5e-3))) > exposure(util()) + 1e-3
        # ... and the same term still rewards under the minimisation.
        @test exposure(minrisk(; cobj = ScoreTilt(score, 1e-4))) > base_exp + 1e-3
    end

    @testset "objective hook: a quadratic term works against an affine objective" begin
        # Regression: `MaximumReturn` builds an AFFINE `obj_expr`, so a quadratic term used to
        # be a MethodError from `add_to_expression!(::AffExpr, ::QuadExpr)` — a custom term
        # whose validity depended on the risk measure and objective. The penalty accumulator
        # promotes, so it now works everywhere.
        maxret(; kwargs...) = optimise(MeanRisk(; obj = MaximumReturn(),
                                                opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                                    kwargs...)))
        plain = maxret()
        spread = maxret(; cobj = SpreadPenalty(1e2))
        # An unconstrained max-return solve piles into the highest-mean asset; the quadratic
        # penalty prices concentration, so the book must spread out.
        @test dot(spread.w, spread.w) < dot(plain.w, plain.w) - 1e-4
    end

    @testset "objective hook is called with the documented argument order" begin
        # The contract is `(model, obj, cobj, optimiser, attrs)`. A caller that shifts `cobj`
        # out of position 3 means the user's method misses dispatch — which now RAISES via the
        # typed fallback rather than silently dropping the term. `NearOptimalCentering` invokes
        # the hook from its own objective builder rather than through
        # `set_portfolio_objective_function!`, so assert the convention holds on both paths.
        # Only the *constrained* NOC algorithm routes through that call.
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
        @test main[1].optimiser isa MeanRisk
        @test main[1].attrs isa ProcessedJuMPOptimiserAttributes

        noc = spy_log(spy -> optimise(NearOptimalCentering(; r = StandardDeviation(),
                                                           obj = MinimumRisk(),
                                                           alg = ConstrainedNearOptimalCentering(),
                                                           opt = JuMPOptimiser(; pe = pr,
                                                                               slv = slv,
                                                                               cobj = spy))))
        # NOC also solves `MeanRisk` sub-problems for its reference points, and those reach the
        # hook through the main path — so a non-empty log proves nothing. The centring
        # objective's own call is the one under test, and it is the only invocation carrying
        # the `NearOptimalCentering` estimator in the `optimiser` slot. (Before ADR 0036 this
        # call was the odd one out for the wrong reason: it passed a `JuMPOptimiser` there,
        # a different branch of the type tree from every other caller.)
        @test any(e -> isa(e.optimiser, NearOptimalCentering), noc)
        @test all(e -> e.obj isa PO.ObjectiveFunction, noc)
        @test all(e -> e.optimiser isa PO.OptimisationEstimator, noc)
        @test all(e -> e.attrs isa ProcessedJuMPOptimiserAttributes, noc)
    end

    @testset "fallback: a subtype with no builder method raises" begin
        # ADR 0036: the untyped `args...` catch-all used to absorb these silently, which is how
        # a `ccnt` vector once vanished without a word — and is how every stale hook would have
        # vanished when this signature changed. A subtype that defines no method is a user
        # error, and now says so.
        @test_throws ArgumentError minrisk(; cobj = InertObjective())
        @test_throws ArgumentError minrisk(; ccnt = InertConstraint())
        # `nothing` — no custom term configured — remains the one legitimate no-op.
        @test isapprox(minrisk(; cobj = nothing, ccnt = nothing).w, base.w; atol = 1e-6)
    end
end
