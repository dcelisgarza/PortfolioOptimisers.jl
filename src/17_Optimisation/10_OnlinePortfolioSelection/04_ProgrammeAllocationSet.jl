"""
$(DocStringExtensions.TYPEDEF)

The Allocation Set of the full constraint vocabulary: weight bounds, universe sets, linear constraints in the asset basis, a turnover ceiling, a variance or standard-deviation ceiling, a tracking error and the MIP kinds, with a solver required by its field bound, because a projection onto it is a programme.

The projection is a bare JuMP model in the Weight Finaliser's idiom — `w`, `k = 1`, `Σw = 1`, the constraint scale and the objective scale — assembled by the shared builders through [`set_allocation_set_constraints!`](@ref) and given the objective of the rule's Projection Geometry: ``\\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{q} \\rVert^2`` for [`EuclideanProjection`](@ref), ``\\sum_i w_i \\log (w_i / q_i)`` for [`EntropicProjection`](@ref), an exponential-cone programme, and ``(\\boldsymbol{w} - \\boldsymbol{q})^\\intercal A (\\boldsymbol{w} - \\boldsymbol{q})`` for [`GramProjection`](@ref). The budget is one and there is no budget field: cash is an asset with price relative one, which the rule allocates like any other.

**The per-constraint mechanism.**

  - `wb`, `sets`: [`set_weight_constraints!`](@ref) with the budget one.
  - `lcs`: [`set_linear_weight_constraints!`](@ref) on [`linear_constraints`](@ref) over `sets`.
  - `tn`: [`_set_turnover_constraints!`](@ref), the library's per-asset ceiling ``\\lvert w_i - \\hat{w}_i \\rvert \\leq \\mathrm{tn}_i``, with the reference set to the Price-Adjusted Allocation ``\\hat{\\boldsymbol{w}}_t = \\boldsymbol{w}_t \\odot \\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` of the step — the book the fund trades from, never the distance between two targets. The ceiling is a number, or one per asset.
  - `r`, `pe`: the set's own second-order cone ``[u; G \\boldsymbol{w}] \\in \\mathcal{K}_{\\mathrm{SOC}}`` with ``G^\\intercal G = \\boldsymbol{\\Sigma}`` and ``u`` the ceiling — ``\\sqrt{\\mathrm{ub}}`` for a [`Variance`](@ref), ``\\mathrm{ub}`` for a [`StandardDeviation`](@ref) — the covariance read off `r` when it holds a matrix and otherwise off `pe` fitted on the head's rows at every step.
  - `te`: [`set_tracking_error_constraints!`](@ref) over the head's rows, against a [`WeightsTracking`](@ref) benchmark.
  - `card`, `gcard`, `lt`, `st`, `ss`: [`set_mip_constraints!`](@ref), the same builders [`JuMPOptimiser`](@ref) uses, under a MIP-capable `slv`.

A `pe` or `te` on the set reads every row the head has folded — [`rows_needed`](@ref) answers `nothing` — unless `r` holds a matrix and `te` is absent, so a step costs one covariance fit over the whole prefix on top of its programme; a caller who wants a window states it on `pe`. A covariance of one observation does not exist, so while the head holds fewer than two rows the fit is not attempted and the step is a Held Step, recorded as such ([`allocation_set_ready`](@ref)); a caller who wants the ceiling from row one gives `r` its matrix. A MIP projection is not unique, so the identity between the Causal Pass and the Recursion Read-out is a claim about the code path — the same solves in the same order on the same rows — and holds exactly with a deterministic solver.

The set refuses, by having no field for them: the SDP kinds, exposure constraints in a factor Constraint Space, every other risk measure, fees and a budget. A negative lower bound is admitted under the Euclidean and Gram geometries and refused under the entropic one, at the head's construction when the bound is a value and at the projection when it is resolved from an estimator. The wealth factor of a leveraged allocation can reach zero on an extreme day, where the log wealth and the next gradient are undefined; that is documented here, not guarded.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ProgrammeAllocationSet(;
        slv::Slv_VecSlv,
        wb::Option{<:WbE_Wb} = WeightBounds(),
        sets::Option{<:UniverseSets} = nothing,
        lcs::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
        tn::Option{<:Num_VecNum} = nothing,
        r::Option{<:Union{<:Variance, <:StandardDeviation}} = nothing,
        pe::AbstractPriorEstimator = EmpiricalPrior(),
        te::Option{<:TrackingError{<:WeightsTracking}} = nothing,
        card::Option{<:Integer} = nothing,
        gcard::Option{<:LcE_Lc} = nothing,
        lt::Option{<:BtE_Bt} = nothing,
        st::Option{<:BtE_Bt} = nothing,
        ss::Option{<:Number} = nothing,
        sc::Number = 1,
        so::Number = 1
    ) -> ProgrammeAllocationSet

Keywords correspond to the struct's fields. `slv` has no default: a set that needs a solver cannot be built without one.

## Validation

  - If any of `wb`, `lcs`, `gcard`, `lt` or `st` is an estimator: `!isnothing(sets)`. An `IsNothingError` is thrown otherwise.
  - If `tn` is given: `all(>= 0, tn)`. A `DomainError` is thrown otherwise.
  - If `r` is given: `r.settings.ub` is a finite non-negative number, the ceiling. An `ArgumentError` is thrown otherwise.
  - If `card` is given: `card >= 1`. A `DomainError` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, `wb`, `sets`, `lcs`, `r`, `te`, `gcard`, `lt` and `st` are viewed recursively, a vector `tn` is sliced, and the rest is carried unchanged. A precomputed [`LinearConstraint`](@ref) is the identity under a view, as it is everywhere.

# Examples

```jldoctest
julia> ProgrammeAllocationSet(; slv = Solver(; solver = nothing), tn = 0.1)
ProgrammeAllocationSet
     wb ┼ WeightBounds
        │   lb ┼ Float64: 0.0
        │   ub ┴ Float64: 1.0
   sets ┼ nothing
    lcs ┼ nothing
     tn ┼ Float64: 0.1
      r ┼ nothing
     pe ┼ EmpiricalPrior
        │           ce ┼ PortfolioOptimisersCovariance
        │              │   ce ┼ Covariance
        │              │      │    me ┼ SimpleExpectedReturns
        │              │      │       │   w ┴ nothing
        │              │      │    ce ┼ GeneralCovariance
        │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │              │      │       │    w ┴ nothing
        │              │      │   alg ┼ FullMoment()
        │              │      │     w ┴ nothing
        │              │   mp ┼ MatrixProcessing
        │              │      │     pdm ┼ Posdef
        │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
        │              │      │      dn ┼ nothing
        │              │      │      dt ┼ nothing
        │              │      │     alg ┼ nothing
        │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
        │           me ┼ SimpleExpectedReturns
        │              │   w ┴ nothing
        │      horizon ┼ nothing
        │   fill_limit ┴ nothing
     te ┼ nothing
   card ┼ nothing
  gcard ┼ nothing
     lt ┼ nothing
     st ┼ nothing
     ss ┼ nothing
    slv ┼ Solver
        │          name ┼ String: \"\"
        │        solver ┼ nothing
        │      settings ┼ nothing
        │     check_sol ┼ @NamedTuple{}: NamedTuple()
        │   add_bridges ┴ Bool: true
     sc ┼ Int64: 1
     so ┴ Int64: 1
```

# Related

  - [`AbstractAllocationSet`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`project`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`HeldStep`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
"""
@concrete struct ProgrammeAllocationSet <: AbstractAllocationSet
    """
    $(field_dict[:wb])
    """
    wb
    """
    $(field_dict[:sets])
    """
    sets
    """
    Linear constraints in the asset basis, estimators resolved over `sets`, or `nothing`.
    """
    lcs
    """
    The per-asset turnover ceiling against the step's Price-Adjusted Allocation, a number broadcast over the assets or one per asset, or `nothing`.
    """
    tn
    """
    The variance or standard-deviation ceiling, a [`Variance`](@ref) or a [`StandardDeviation`](@ref) whose `settings.ub` is the ceiling, or `nothing`.
    """
    r
    """
    The prior estimator the covariance of `r` and the rows of `te` are read from, fitted on the head's rows at every step; unread when `r` holds a matrix and `te` is absent.
    """
    pe
    """
    The tracking error over the head's rows against a weights benchmark, or `nothing`.
    """
    te
    """
    $(field_dict[:card])
    """
    card
    """
    Grouped cardinality constraint, an estimator resolved over `sets`, or `nothing`.
    """
    gcard
    """
    $(field_dict[:lt])
    """
    lt
    """
    $(field_dict[:st])
    """
    st
    """
    $(field_dict[:ss])
    """
    ss
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    function ProgrammeAllocationSet(wb::Option{<:WbE_Wb}, sets::Option{<:UniverseSets},
                                    lcs::Option{<:LcE_Lc_VecLcE_Lc},
                                    tn::Option{<:Num_VecNum},
                                    r::Option{<:Union{<:Variance, <:StandardDeviation}},
                                    pe::AbstractPriorEstimator,
                                    te::Option{<:TrackingError{<:WeightsTracking}},
                                    card::Option{<:Integer}, gcard::Option{<:LcE_Lc},
                                    lt::Option{<:BtE_Bt}, st::Option{<:BtE_Bt},
                                    ss::Option{<:Number}, slv::Slv_VecSlv, sc::Number,
                                    so::Number)
        if any(name_keyed, (wb, lcs, gcard, lt, st))
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when wb, lcs, gcard, lt or st is an estimator"))
        end
        if !isnothing(tn)
            assert_nonneg(tn, :tn)
        end
        assert_risk_ceiling(r)
        if !isnothing(card)
            @argcheck(card >= 1, DomainError(card, "card must be at least one"))
        end
        return new{typeof(wb), typeof(sets), typeof(lcs), typeof(tn), typeof(r), typeof(pe),
                   typeof(te), typeof(card), typeof(gcard), typeof(lt), typeof(st),
                   typeof(ss), typeof(slv), typeof(sc), typeof(so)}(wb, sets, lcs, tn, r,
                                                                    pe, te, card, gcard, lt,
                                                                    st, ss, slv, sc, so)
    end
end
function ProgrammeAllocationSet(; slv::Slv_VecSlv, wb::Option{<:WbE_Wb} = WeightBounds(),
                                sets::Option{<:UniverseSets} = nothing,
                                lcs::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
                                tn::Option{<:Num_VecNum} = nothing,
                                r::Option{<:Union{<:Variance, <:StandardDeviation}} = nothing,
                                pe::AbstractPriorEstimator = EmpiricalPrior(),
                                te::Option{<:TrackingError{<:WeightsTracking}} = nothing,
                                card::Option{<:Integer} = nothing,
                                gcard::Option{<:LcE_Lc} = nothing,
                                lt::Option{<:BtE_Bt} = nothing,
                                st::Option{<:BtE_Bt} = nothing,
                                ss::Option{<:Number} = nothing, sc::Number = 1,
                                so::Number = 1)::ProgrammeAllocationSet
    return ProgrammeAllocationSet(wb, sets, lcs, tn, r, pe, te, card, gcard, lt, st, ss,
                                  slv, sc, so)
end
function port_opt_view(set::ProgrammeAllocationSet, i, args...)
    return ProgrammeAllocationSet(; slv = set.slv, wb = port_opt_view(set.wb, i, args...),
                                  sets = port_opt_view(set.sets, i, args...),
                                  lcs = port_opt_view(set.lcs, i, args...),
                                  tn = nothing_scalar_array_view(set.tn, i),
                                  r = port_opt_view(set.r, i, args...), pe = set.pe,
                                  te = port_opt_view(set.te, i, args...), card = set.card,
                                  gcard = port_opt_view(set.gcard, i, args...),
                                  lt = port_opt_view(set.lt, i, args...),
                                  st = port_opt_view(set.st, i, args...), ss = set.ss,
                                  sc = set.sc, so = set.so)
end
"""
    name_keyed(x)
    name_keyed(x::AbstractVector)

Whether a constraint slot holds an estimator that resolves its names over `sets`, so a [`ProgrammeAllocationSet`](@ref) must carry them: a [`WeightBoundsEstimator`](@ref), a [`LinearConstraintEstimator`](@ref), a [`ThresholdEstimator`](@ref), or a vector holding one.

# Related

  - [`ProgrammeAllocationSet`](@ref)
"""
function name_keyed(::Any)
    return false
end
function name_keyed(::Union{<:WeightBoundsEstimator, <:LinearConstraintEstimator,
                            <:ThresholdEstimator})
    return true
end
function name_keyed(x::AbstractVector)
    return any(name_keyed, x)
end
"""
    assert_risk_ceiling(r::Nothing)
    assert_risk_ceiling(r::Union{<:Variance, <:StandardDeviation})

Refuses a risk measure on a [`ProgrammeAllocationSet`](@ref) whose `settings.ub` is not the ceiling: a finite non-negative number.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`allocation_risk_ceiling`](@ref)
"""
function assert_risk_ceiling(::Nothing)::Nothing
    return nothing
end
function assert_risk_ceiling(r::Union{<:Variance, <:StandardDeviation})::Nothing
    ub = r.settings.ub
    @argcheck(isa(ub, Number) && isfinite(ub) && ub >= zero(ub),
              ArgumentError("the ceiling of a `$(nameof(typeof(r)))` on a ProgrammeAllocationSet is its `settings.ub`, a finite non-negative number; got `$ub`"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves every name-keyed constraint of a [`ProgrammeAllocationSet`](@ref) to a value over `N` assets: the weight bounds, the linear constraints, the grouped cardinality and the thresholds. The ceilings, the prior estimator, the tracking error and the solver are carried unchanged.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`weight_bounds_constraints`](@ref)
  - [`linear_constraints`](@ref)
  - [`threshold_constraints`](@ref)
"""
function resolve_allocation_set(set::ProgrammeAllocationSet, N::Integer, strict::Bool,
                                datatype::DataType)
    wb = weight_bounds_constraints(set.wb, set.sets; N = N, strict = strict,
                                   datatype = datatype)
    lcs = linear_constraints(set.lcs, set.sets; datatype = datatype, strict = strict)
    gcard = linear_constraints(set.gcard, set.sets; datatype = Int, strict = strict)
    lt = threshold_constraints(set.lt, set.sets; datatype = datatype, strict = strict)
    st = threshold_constraints(set.st, set.sets; datatype = datatype, strict = strict)
    return ProgrammeAllocationSet(; slv = set.slv, wb = wb, sets = set.sets, lcs = lcs,
                                  tn = set.tn, r = set.r, pe = set.pe, te = set.te,
                                  card = set.card, gcard = gcard, lt = lt, st = st,
                                  ss = set.ss, sc = set.sc, so = set.so)
end
"""
    rows_needed(set::ProgrammeAllocationSet)

The rows a programme set reads at a step: `nothing`, every row folded, when it carries a tracking error or a ceiling whose covariance is fitted; `0` otherwise.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`rows_needed`](@ref)
"""
function rows_needed(set::ProgrammeAllocationSet)
    fitted = !isnothing(set.r) && !isa(set.r.sigma, AbstractMatrix)
    return (fitted || !isnothing(set.te)) ? nothing : 0
end
"""
    assert_geometry_admits_set(proj::AbstractProjectionGeometry, set::AbstractAllocationSet)

Refuses, where the head's `alg` and `set` meet, a negative lower bound under [`EntropicProjection`](@ref), whose `log w` is undefined below zero.

The refusal is at construction when the bound is a value; a bound resolved from an estimator is refused at the projection. Any other geometry admits a negative bound, so a long-short reversion costs nothing.

# Validation

  - Under [`EntropicProjection`](@ref) with a [`WeightBounds`](@ref) whose `lb` is a number or a vector: `all(>= 0, lb)`. A `DomainError` is thrown otherwise.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`project`](@ref)
"""
function assert_geometry_admits_set(::AbstractProjectionGeometry,
                                    ::AbstractAllocationSet)::Nothing
    return nothing
end
function assert_geometry_admits_set(::EntropicProjection,
                                    set::AbstractAllocationSet)::Nothing
    wb = set.wb
    if isa(wb, WeightBounds) && !isnothing(wb.lb)
        @argcheck(all(x -> x >= zero(x), wb.lb),
                  DomainError(wb.lb,
                              "the entropic projection admits no negative lower bound: `log w` is undefined below zero. Use a Euclidean rule, or a non-negative bound."))
    end
    return nothing
end
"""
    set_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet, w::AbstractVector, X)
    set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X)

Adds the constraints of a resolved Allocation Set to a model that already carries `w`, `k`, the scales and the observation count: the budget one and the weight bounds on both kinds, and on the programme set every admissible kind through the shared builders, the turnover ceiling measured from `w`, the set's own variance cone and the tracking error over `X`.

The bare projection model of [`project`](@ref) is one caller; a JuMP head that takes the set as its programme's feasible region is another.

# Arguments

  - $(arg_dict[:model])
  - `set`: The Allocation Set, resolved.
  - `w`: The Price-Adjusted Allocation the step trades from, the reference of the turnover ceiling.
  - `X`: The rows of returns the head holds through the period, `observations × assets`, or `nothing`.

# Validation

  - A covariance ceiling whose covariance is fitted, or a tracking error, with `X === nothing`. An `ArgumentError` is thrown: the rows reach a projection inside an Online Update alone.

# Returns

  - `nothing`.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`set_weight_constraints!`](@ref)
  - [`set_linear_weight_constraints!`](@ref)
  - [`_set_turnover_constraints!`](@ref)
  - [`set_tracking_error_constraints!`](@ref)
  - [`set_mip_constraints!`](@ref)
"""
function set_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet,
                                         ::AbstractVector, ::Any)::Nothing
    set_allocation_set_bounds!(model, set.wb)
    return nothing
end
function set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                         w::AbstractVector, X)::Nothing
    set_allocation_set_bounds!(model, set.wb)
    set_linear_weight_constraints!(model, set.lcs, :lcs_ineq_, :lcs_eq_)
    set_mip_constraints!(model, set.wb, set.card, set.gcard, nothing, set.lt, set.st,
                         nothing, set.ss)
    if !isnothing(set.tn)
        _set_turnover_constraints!(model, Turnover(; w = w, val = set.tn))
    end
    pr = allocation_set_prior(set, X)
    set_allocation_risk_cone!(model, set.r, pr)
    if !isnothing(set.te)
        set_tracking_error_constraints!(model, 1, pr, set.te)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds an Allocation Set's weight bounds and its budget of one to the model: the bounds through [`set_weight_constraints!`](@ref) with no budget group, so a negative lower bound builds the long-short decomposition without pinning either side's budget, and then `Σw = k` through [`set_budget_constraints!`](@ref).

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub` over the resolved bounds, through [`assert_feasible_bounds`](@ref).

# Related

  - [`set_allocation_set_constraints!`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
"""
function set_allocation_set_bounds!(model::JuMP.Model, wb::WeightBounds)::Nothing
    assert_feasible_bounds(wb)
    set_weight_constraints!(model, wb, nothing, nothing)
    set_budget_constraints!(model, 1, get_w(model))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The prior result a programme set's covariance cone and tracking error read: `pe` fitted on the head's rows, or `nothing` when the set reads none.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
"""
function allocation_set_prior(set::ProgrammeAllocationSet, X)
    if isnothing(rows_needed(set))
        @argcheck(!isnothing(X),
                  ArgumentError("a ProgrammeAllocationSet with a fitted covariance ceiling or a tracking error reads the head's rows, which reach a projection inside an Online Update alone; `project` was called with none."))
        assert_factor_returns(set.pe, nothing)
        return prior(set.pe, X, nothing, nothing)
    end
    return nothing
end
"""
    set_allocation_risk_cone!(model::JuMP.Model, r::Nothing, pr)
    set_allocation_risk_cone!(model::JuMP.Model, r::Variance, pr)
    set_allocation_risk_cone!(model::JuMP.Model, r::StandardDeviation, pr)

Adds a programme set's own second-order cone, ``[u k; G \\boldsymbol{w}] \\in \\mathcal{K}_{\\mathrm{SOC}}`` with ``G`` the Cholesky factor of the covariance and ``u`` the ceiling: the square root of a [`Variance`](@ref)'s `settings.ub`, and a [`StandardDeviation`](@ref)'s as it is.

The covariance is `r`'s own matrix when it holds one, and the prior result's otherwise. The factor is taken without a definiteness check: a covariance the prior's own processing left indefinite reaches the solver, whose failure is the step's Held Step.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`HeldStep`](@ref)
"""
function set_allocation_risk_cone!(::JuMP.Model, ::Nothing, ::Any)::Nothing
    return nothing
end
function set_allocation_risk_cone!(model::JuMP.Model,
                                   r::Union{<:Variance, <:StandardDeviation}, pr)::Nothing
    sigma = isa(r.sigma, AbstractMatrix) ? r.sigma : factory(r, pr).sigma
    G = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(sigma); check = false).U
    u = allocation_risk_ceiling(r)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, set_risk_soc,
                     [sc * u * k; sc * (G * w)] in JuMP.SecondOrderCone())
    return nothing
end
"""
    allocation_risk_ceiling(r::Variance)
    allocation_risk_ceiling(r::StandardDeviation)

The bound on the cone variable of [`set_allocation_risk_cone!`](@ref): the square root of a [`Variance`](@ref)'s `settings.ub`, and a [`StandardDeviation`](@ref)'s as it is.

# Related

  - [`set_allocation_risk_cone!`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
"""
function allocation_risk_ceiling(r::Variance)
    return sqrt(r.settings.ub)
end
function allocation_risk_ceiling(r::StandardDeviation)
    return r.settings.ub
end
"""
    set_projection_objective!(model::JuMP.Model, proj::EuclideanProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::EntropicProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::GramProjection, q::AbstractVector)

Sets the projection programme's objective in the geometry's divergence from the raw step `q`: the Euclidean distance through a second-order cone, the relative entropy ``\\sum_i w_i \\log (w_i / q_i)`` through a relative-entropy cone over the positive entries of `q` with the zero entries pinned at zero, and the Gram norm ``\\lVert G (\\boldsymbol{w} - \\boldsymbol{q}) \\rVert`` with ``G^\\intercal G = A`` through a second-order cone.

The Euclidean and Gram arms minimise the norm rather than its square, which has the same minimiser and keeps the programme conic on every solver.

# Validation

  - Under [`EntropicProjection`](@ref): `all(>= 0, q)` and `sum(q) > 0`. A `DomainError` is thrown otherwise.
  - Under [`GramProjection`](@ref): `proj.A` is not `nothing`. An `ArgumentError` is thrown otherwise: the rule binds the matrix.

# Related

  - [`project`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
"""
function set_projection_objective!(model::JuMP.Model, ::EuclideanProjection,
                                   q::AbstractVector)::Nothing
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t_proj)
    JuMP.@constraint(model, proj_soc, [sc * t_proj; sc * (w - q)] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t_proj)
    return nothing
end
function set_projection_objective!(model::JuMP.Model, ::EntropicProjection,
                                   q::AbstractVector)::Nothing
    @argcheck(all(x -> x >= zero(x), q),
              DomainError(q,
                          "the entropic projection is defined on non-negative raw steps alone: `log w` is undefined below zero"))
    @argcheck(sum(q) > zero(eltype(q)),
              DomainError(q,
                          "the entropic projection needs a positive entry to normalise: a raw step of zeros has no projection onto the simplex"))
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    pos = findall(x -> x > zero(x), q)
    zer = findall(iszero, q)
    JuMP.@variable(model, t_proj)
    # `w_i log(w_i / q_i)` with `q_i = 0` is finite at `w_i = 0` alone: a zero entry stays
    # zero, as it does under the closed form.
    if !isempty(zer)
        JuMP.@constraint(model, proj_zero, sc * w[zer] .== 0)
    end
    JuMP.@constraint(model, proj_rec,
                     [sc * t_proj; sc * q[pos]; sc * w[pos]] in
                     JuMP.MOI.RelativeEntropyCone(1 + 2 * length(pos)))
    JuMP.@objective(model, Min, so * t_proj)
    return nothing
end
function set_projection_objective!(model::JuMP.Model, proj::GramProjection,
                                   q::AbstractVector)::Nothing
    @argcheck(!isnothing(proj.A),
              ArgumentError("a GramProjection projects in the norm of a Gram matrix the rule binds at each step, and this one has none bound: it is the rule's geometry, not a caller's."))
    G = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(proj.A)).U
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t_proj)
    JuMP.@constraint(model, proj_soc,
                     [sc * t_proj; sc * (G * (w - q))] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t_proj)
    return nothing
end
"""
    projection_solver(proj::AbstractProjectionGeometry, set::ProgrammeAllocationSet)
    projection_solver(proj::GramProjection, set::AbstractAllocationSet)
    projection_solver(proj::GramProjection, set::ProgrammeAllocationSet)
    projection_solver(proj::GramProjection, set::BoundedAllocationSet)
    projection_solver(proj::AbstractProjectionGeometry, set::BoundedAllocationSet)

The solver a projection programme runs on: the geometry's own when it carries one, the set's otherwise. A scalar-root geometry on the bounded set has no programme to run and is refused by name; [`project`](@ref) never reaches it.

# Related

  - [`GramProjection`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
  - [`project`](@ref)
"""
function projection_solver(::AbstractProjectionGeometry, set::ProgrammeAllocationSet)
    return set.slv
end
function projection_solver(proj::GramProjection, ::AbstractAllocationSet)
    return proj.slv
end
function projection_solver(proj::GramProjection, ::ProgrammeAllocationSet)
    return proj.slv
end
function projection_solver(proj::GramProjection, ::BoundedAllocationSet)
    return proj.slv
end
function projection_solver(proj::AbstractProjectionGeometry, ::BoundedAllocationSet)
    return throw(ArgumentError("a `$(nameof(typeof(proj)))` onto a BoundedAllocationSet is a scalar root, not a programme: no solver is read for it."))
end
"""
    allocation_set_ready(set::AbstractAllocationSet, X)
    allocation_set_ready(set::ProgrammeAllocationSet, X)

Whether a set's constraints can be formed on the rows `X` the step holds: `nothing` when they can, and the reason for a Held Step when they cannot — a programme set whose prior is fitted, while the head holds fewer than two rows, because a covariance of one observation does not exist.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`projection_programme`](@ref)
  - [`HeldStep`](@ref)
"""
function allocation_set_ready(::AbstractAllocationSet, ::Any)
    return nothing
end
function allocation_set_ready(set::ProgrammeAllocationSet, X)
    if isnothing(rows_needed(set)) && !isnothing(X) && size(X, 1) < 2
        return "the set's prior estimator fits on the head's rows and a covariance of one observation does not exist, so the step trades nothing until the head holds two"
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The projection programme: a bare model with `w`, `k = 1`, the set's scales and the observation count of the step's rows, the set's constraints from [`set_allocation_set_constraints!`](@ref) with `w` as the turnover reference, and the geometry's objective from [`set_projection_objective!`](@ref), solved on [`projection_solver`](@ref).

A solved programme answers its weights. A failed one, or one whose constraints cannot be formed on the step's rows ([`allocation_set_ready`](@ref)), is the Held Step: the record goes to the current [`ProjectionStep`](@ref) through [`record_held_step!`](@ref), and the answer is a copy of `w`, the book the fund already holds.

# Arguments

  - `proj`: The geometry.
  - `set`: The set, resolved.
  - `q`: The raw step.
  - `w`: The Price-Adjusted Allocation the step trades from.

# Returns

  - `w'::Vector`: The projected allocation, or `w` copied on a Held Step.

# Related

  - [`project`](@ref)
  - [`HeldStep`](@ref)
  - [`optimise_JuMP_model!`](@ref)
"""
function projection_programme(proj::AbstractProjectionGeometry, set::AbstractAllocationSet,
                              q::AbstractVector, w::AbstractVector)
    X = projection_step_rows()
    reason = allocation_set_ready(set, X)
    if !isnothing(reason)
        record_held_step!(reason, nothing)
        return copy(w)
    end
    model = JuMP.Model()
    set_model_scales!(model, projection_scale(set, :sc), projection_scale(set, :so))
    set_model_observations!(model, isnothing(X) ? 0 : size(X, 1))
    JuMP.@expression(model, k, 1)
    JuMP.@variable(model, wv[1:length(q)])
    model[:w] = wv
    set_allocation_set_constraints!(model, set, w, X)
    set_projection_objective!(model, proj, q)
    res = optimise_JuMP_model!(model, projection_solver(proj, set))
    if res.success
        return JuMP.value.(wv)
    end
    record_held_step!("the projection onto the `$(nameof(typeof(set)))` in the `$(nameof(typeof(proj)))` geometry did not solve, and the step trades nothing",
                      res.trials)
    return copy(w)
end
"""
    projection_scale(set::ProgrammeAllocationSet, f::Symbol)
    projection_scale(set::BoundedAllocationSet, f::Symbol)

The constraint scale `:sc` or the objective scale `:so` a projection programme registers on its model: the programme set's own, and one on the bounded set, which carries none.

# Related

  - [`projection_programme`](@ref)
  - [`set_model_scales!`](@ref)
"""
function projection_scale(set::ProgrammeAllocationSet, f::Symbol)
    return getfield(set, f)
end
function projection_scale(::BoundedAllocationSet, ::Symbol)
    return 1
end
"""
    project(proj::GramProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::EuclideanProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::EntropicProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::GramProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)

The programme arms of the Constrained Update: every pair but the two scalar roots on the bounded set is the bare-model programme of [`projection_programme`](@ref).

# Validation

  - Under [`EntropicProjection`](@ref): `all(>= 0, lb)` over the resolved bounds. A `DomainError` is thrown otherwise.

# Related

  - [`project`](@ref)
  - [`projection_programme`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
  - [`GramProjection`](@ref)
"""
function project(proj::GramProjection, set::BoundedAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    return projection_programme(proj, set, q, w)
end
function project(proj::EuclideanProjection, set::ProgrammeAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    return projection_programme(proj, set, q, w)
end
function project(proj::EntropicProjection, set::ProgrammeAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    @argcheck(all(x -> x >= zero(x), set.wb.lb),
              DomainError(set.wb.lb,
                          "the entropic projection admits no negative lower bound: `log w` is undefined below zero"))
    return projection_programme(proj, set, q, w)
end
function project(proj::GramProjection, set::ProgrammeAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    return projection_programme(proj, set, q, w)
end
"""
    blend_projection(proj::EuclideanProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    blend_projection(proj::EuclideanProjection, set::AbstractAllocationSet, q::AbstractVector, w::AbstractVector)

The second projection of an [`ExpertMixture`](@ref)'s blend onto the head's set: skipped by dispatch on a [`BoundedAllocationSet`](@ref), where a blend of bounded allocations is bounded and the projection would be the identity, and [`project`](@ref) on every other set, where it is the repair a turnover ceiling or a MIP kind needs.

# Related

  - [`ExpertMixture`](@ref)
  - [`project`](@ref)
"""
function blend_projection(::EuclideanProjection, ::BoundedAllocationSet, q::AbstractVector,
                          ::AbstractVector)
    return q
end
function blend_projection(proj::EuclideanProjection, set::AbstractAllocationSet,
                          q::AbstractVector, w::AbstractVector)
    return project(proj, set, q, w)
end
export ProgrammeAllocationSet
public set_allocation_set_constraints!
