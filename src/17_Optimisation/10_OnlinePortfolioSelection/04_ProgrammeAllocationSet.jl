"""
$(DocStringExtensions.TYPEDEF)

The Allocation Set of the full constraint vocabulary: weight bounds, universe sets, linear constraints in the asset basis, a turnover ceiling, a risk ceiling under any [`RiskMeasure`](@ref), a tracking error and the MIP kinds, with a solver required by its field bound, because a projection onto it is a programme.

The projection is a bare JuMP model in the Weight Finaliser's idiom — `w`, `k = 1`, `Σw = 1`, the constraint scale and the objective scale — assembled by the shared builders through [`set_allocation_set_constraints!`](@ref) and given the objective of the rule's Projection Geometry: ``\\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{q} \\rVert^2`` for [`EuclideanProjection`](@ref), ``\\sum_i w_i \\log (w_i / q_i)`` for [`EntropicProjection`](@ref), an exponential-cone programme, and ``(\\boldsymbol{w} - \\boldsymbol{q})^\\intercal A (\\boldsymbol{w} - \\boldsymbol{q})`` for [`GramProjection`](@ref). The budget is one and there is no budget field: cash is an asset with price relative one, which the rule allocates like any other.

**The per-constraint mechanism.**

  - `wb`, `sets`: [`set_weight_constraints!`](@ref) with the budget one.
  - `lcs`: [`set_linear_weight_constraints!`](@ref) on [`linear_constraints`](@ref) over `sets`.
  - `tn`: [`_set_turnover_constraints!`](@ref), the library's per-asset ceiling ``\\lvert w_i - \\hat{w}_i \\rvert \\leq \\mathrm{tn}_i``, with the reference set to the Price-Adjusted Allocation ``\\hat{\\boldsymbol{w}}_t = \\boldsymbol{w}_t \\odot \\boldsymbol{x}_t / \\langle \\boldsymbol{w}_t, \\boldsymbol{x}_t \\rangle`` of the step — the book the fund trades from, never the distance between two targets. The ceiling is a number, or one per asset.
  - `r`, `pe`: the risk measure's own JuMP builder, [`set_risk_constraints!`](@ref), with the set as the [`RiskConstraintOwner`](@ref) and `settings.ub` as the ceiling — a conditional value at risk, a drawdown, a variance, any measure the JuMP optimisers bound — on the prior result of `pe` fitted on the head's rows at every step, the measure materialised against that prior through [`factory`](@ref) first, so a moment it carries itself is the one it is built on. A [`Variance`](@ref) or [`StandardDeviation`](@ref) that holds its matrix reads no rows and no prior: it is the set's own second-order cone ``[u; G \\boldsymbol{w}] \\in \\mathcal{K}_{\\mathrm{SOC}}`` with ``G^\\intercal G = \\boldsymbol{\\Sigma}`` and ``u`` the ceiling, ``\\sqrt{\\mathrm{ub}}`` for the variance and ``\\mathrm{ub}`` for the standard deviation, the same cone the shared builder writes ([`set_allocation_risk_ceiling!`](@ref)).
  - `te`: [`set_tracking_error_constraints!`](@ref) over the head's rows, against a [`WeightsTracking`](@ref) benchmark.
  - `card`, `gcard`, `lt`, `st`, `ss`: [`set_mip_constraints!`](@ref), the same builders [`JuMPOptimiser`](@ref) uses, under a MIP-capable `slv`.

A ceiling or a `te` on the set reads every row the head has folded — [`rows_needed`](@ref) answers `nothing` — unless `r` is a variance or standard deviation holding its matrix and `te` is absent, so a step costs one prior fit over the whole prefix on top of its programme; a caller who wants a window states it on `pe`. A covariance of one observation does not exist, so while the head holds fewer than two rows the fit is not attempted and the step is a Held Step, recorded as such ([`allocation_set_ready`](@ref)); a caller who wants a covariance ceiling from row one gives `r` its matrix. A MIP projection is not unique, so the identity between the Causal Pass and the Recursion Read-out is a claim about the code path — the same solves in the same order on the same rows — and holds exactly with a deterministic solver.

The set refuses, by having no field for them: the SDP kinds, exposure constraints in a factor Constraint Space, fees and a budget; a [`Variance`](@ref) with risk-contribution rows on `rc` is an SDP kind and is refused at construction, as is a frontier or a per-asset `ub` ([`assert_risk_ceiling`](@ref)). A measure's `rke` and `scale` are not read: the projection's objective is the geometry's divergence, and the ceiling is a constraint alone. A negative lower bound is admitted under the Euclidean and Gram geometries and refused under the entropic one, at the head's construction when the bound is a value and at the projection when it is resolved from an estimator. The wealth factor of a leveraged allocation can reach zero on an extreme day, where the log wealth and the next gradient are undefined; that is documented here, not guarded.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ProgrammeAllocationSet(;
        slv::Slv_VecSlv,
        wb::Option{<:WbE_Wb} = WeightBounds(),
        sets::Option{<:UniverseSets} = nothing,
        lcs::Option{<:LcE_Lc_VecLcE_Lc} = nothing,
        tn::Option{<:Num_VecNum} = nothing,
        r::Option{<:RiskMeasure} = nothing,
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
@concrete struct ProgrammeAllocationSet <: AbstractProgrammeAllocationSet
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
    The risk ceiling, a [`RiskMeasure`](@ref) whose `settings.ub` is the ceiling, or `nothing`.
    """
    r
    """
    The prior estimator `r` is built on and the rows of `te` are read from, fitted on the head's rows at every step; unread when `r` is a variance or standard deviation holding its matrix and `te` is absent.
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
                                    tn::Option{<:Num_VecNum}, r::Option{<:RiskMeasure},
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
                                r::Option{<:RiskMeasure} = nothing,
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
                                  r = port_opt_view(set.r, i, nothing), pe = set.pe,
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

The rows a programme set reads at a step: `nothing`, every row folded, when it carries a tracking error or a ceiling that reads the head's rows ([`risk_reads_rows`](@ref)); `0` otherwise.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`rows_needed`](@ref)
  - [`risk_reads_rows`](@ref)
"""
function rows_needed(set::ProgrammeAllocationSet)
    return (risk_reads_rows(set.r) || !isnothing(set.te)) ? nothing : 0
end
"""
$(DocStringExtensions.TYPEDEF)

The Tsallis Projection Geometry: the raw step is projected onto the Allocation Set in the Bregman divergence of the power potential ``\\Psi_\\alpha(\\boldsymbol{w}) = \\sum_i w_i^\\alpha / (\\alpha (\\alpha - 1))``, ``\\alpha \\in (0, 1)``.

# Mathematical definition

The mirror image of an allocation is ``\\nabla \\Psi_\\alpha(\\boldsymbol{w})_i = w_i^{\\alpha - 1} / (\\alpha - 1)``, so a first-order step of length ``\\eta`` on a gradient ``\\boldsymbol{g}`` and its projection onto the default set are one scalar root,

```math
\\begin{align}
w_{t+1, i} &= \\left( w_{t, i}^{\\alpha - 1} + (1 - \\alpha) (\\eta g_i + \\lambda) \\right)^{1 / (\\alpha - 1)}\\,,
\\end{align}
```

with ``\\lambda`` the budget multiplier, in which the budget is monotone. On a [`BoundedAllocationSet`](@ref) the bounds are clips of the same root, as the entropic arm's are; on a [`ProgrammeAllocationSet`](@ref) the projection is the programme ``\\min_{\\boldsymbol{w}} \\Psi_\\alpha(\\boldsymbol{w}) - \\langle \\nabla \\Psi_\\alpha(\\boldsymbol{q}), \\boldsymbol{w} \\rangle`` through a power cone on the set's solver. The limit ``\\alpha \\to 1`` is the relative entropy of [`EntropicProjection`](@ref) and ``\\alpha \\to 0`` the log barrier of [`LogBarrierProjection`](@ref); the shipped range is the open interval between them. It is the geometry of the Tsallis-entropy mirror-descent and follow-the-regularised-leader steps of Abernethy, Lee and Tewari (2015) and Zimmert and Seldin (2021), whose regret on the simplex is ``O(\\sqrt{T N / (\\alpha (1 - \\alpha))})`` at the tuned rate. Like the entropic map, it cannot zero a positive entry, a zero entry stays zero, and a negative lower bound is refused, because the potential is undefined below zero.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TsallisProjection(; alpha::Real = 0.5) -> TsallisProjection

Keywords correspond to the struct's fields. The default is the ``\\alpha = 1/2`` of Zimmert and Seldin (2021).

## Validation

  - `0 < alpha < 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> TsallisProjection()
TsallisProjection
  alpha ┴ Float64: 0.5
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`LogBarrierProjection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`MirrorDescent`](@ref)
  - [`project`](@ref)

# References

  - $(ref_dict[:abernethy2015])
  - $(ref_dict[:zimmertseldin2021])
"""
struct TsallisProjection{T1 <: Real} <: AbstractProjectionGeometry
    """
    The power of the potential, in `(0, 1)`.
    """
    alpha::T1
    function TsallisProjection(alpha::Real)
        @argcheck(zero(alpha) < alpha < one(alpha),
                  DomainError(alpha, "alpha must be in (0, 1)"))
        return new{typeof(alpha)}(alpha)
    end
end
function TsallisProjection(; alpha::Real = 0.5)::TsallisProjection
    return TsallisProjection(alpha)
end
"""
$(DocStringExtensions.TYPEDEF)

The log-barrier Projection Geometry: the raw step is projected onto the Allocation Set in the Bregman divergence of the Burg entropy ``\\Psi(\\boldsymbol{w}) = -\\sum_i \\log w_i``, which is the Itakura–Saito divergence ``\\sum_i \\left( w_i / q_i - 1 - \\log (w_i / q_i) \\right)``.

# Mathematical definition

The mirror image is ``\\nabla \\Psi(\\boldsymbol{w})_i = -1 / w_i``, so a first-order step of length ``\\eta`` on a gradient ``\\boldsymbol{g}`` and its projection onto the default set are one scalar root,

```math
\\begin{align}
w_{t+1, i} &= \\frac{1}{1 / w_{t, i} + \\eta g_i + \\lambda}\\,,
\\end{align}
```

with ``\\lambda`` the budget multiplier, in which the budget is monotone. On a [`BoundedAllocationSet`](@ref) the bounds are clips of the same root; on a [`ProgrammeAllocationSet`](@ref) the projection is the programme ``\\min_{\\boldsymbol{w}} -\\sum_i \\log w_i + \\sum_i w_i / q_i`` through an exponential cone on the set's solver. It is the geometry Orseau, Lattimore and Legg (2017, §7) name as the mirror-descent alternative to their Soft-Bayes step: the one first-order geometry whose portfolio regret, ``O(\\sqrt{N T \\log (T / N)})``, needs no lower bound on the price relatives, because a weight that has shrunk towards zero moves by its own scale. It cannot zero a positive entry, a zero entry stays zero, and a negative lower bound is refused, because the logarithm is undefined below zero.

# Examples

```jldoctest
julia> LogBarrierProjection()
LogBarrierProjection()
```

# Related

  - [`AbstractProjectionGeometry`](@ref)
  - [`TsallisProjection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`MirrorDescent`](@ref)
  - [`project`](@ref)

# References

  - $(ref_dict[:orseau2017])
"""
struct LogBarrierProjection <: AbstractProjectionGeometry end
"""
    mirror_step(proj::EuclideanProjection, u::AbstractVector, s::AbstractVector)
    mirror_step(proj::EntropicProjection, u::AbstractVector, s::AbstractVector)
    mirror_step(proj::TsallisProjection, u::AbstractVector, s::AbstractVector)
    mirror_step(proj::LogBarrierProjection, u::AbstractVector, s::AbstractVector)

The unconstrained mirror step from the iterate `u` along the scaled gradient `s = η g`, in the geometry's potential: the raw step ``\\nabla \\Psi^*(\\nabla \\Psi(\\boldsymbol{u}) - \\boldsymbol{s})`` that [`project`](@ref) then puts onto the Allocation Set.

The Euclidean arm is `u - s`, the entropic `u ⊙ exp(-s)`, the Tsallis ``(u_i^{\\alpha - 1} + (1 - \\alpha) s_i)^{1 / (\\alpha - 1)}`` and the log-barrier ``1 / (1 / u_i + s_i)``. The last two are defined while every base is positive: under the log barrier that is ``\\eta \\hat{w}_{t, i} < 1`` for every asset, with ``\\hat{\\boldsymbol{w}}_t`` the Price-Adjusted Allocation, so it always holds at a rate below one and fails only where one asset carries more than ``1 / \\eta`` of the period's wealth; the Tsallis condition is ``(1 - \\alpha) \\eta \\hat{w}_{t, i} w_{t, i}^{-\\alpha} < 1``. A base at or below zero is a step to an unbounded allocation, and it is refused, not clipped.

# Validation

  - Under [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref): every base is positive. A `DomainError` naming the rate is thrown otherwise.

# Related

  - [`MirrorDescent`](@ref)
  - [`project`](@ref)
"""
function mirror_step(::EuclideanProjection, u::AbstractVector, s::AbstractVector)
    return u .- s
end
function mirror_step(::EntropicProjection, u::AbstractVector, s::AbstractVector)
    return u .* exp.(-s)
end
function mirror_step(proj::TsallisProjection, u::AbstractVector, s::AbstractVector)
    a = proj.alpha
    base = u .^ (a - 1) .+ (1 - a) .* s
    assert_mirror_base(base, s)
    return base .^ inv(a - 1)
end
function mirror_step(::LogBarrierProjection, u::AbstractVector, s::AbstractVector)
    base = inv.(u) .+ s
    assert_mirror_base(base, s)
    return inv.(base)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a barrier mirror step one of whose bases is not positive, naming the scaled gradient that produced it.

# Related

  - [`mirror_step`](@ref)
"""
function assert_mirror_base(base::AbstractVector, s::AbstractVector)::Nothing
    @argcheck(all(x -> x > zero(x), base),
              DomainError(s,
                          "the mirror step leaves the geometry's domain: a base of the barrier potential is not positive, so the unconstrained step is unbounded in some asset. Lower the learning rate; under the log barrier the step exists while `eta * w_i * x_i / <w, x> < 1` for every asset."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The scalar root of a barrier geometry's projection onto a bounded set: the budget multiplier `μ` at which `Σ_i clip(φ(b_i + μ), lb_i, ub_i)` is one, with `b` the mirror image of the raw step and `φ` the inverse mirror map, read as the upper bound where the base is not positive.

`φ` is decreasing on positive bases and unbounded as the base falls to zero, so the clipped budget is non-increasing and continuous in `μ`: every base at or below zero puts its asset at its cap, every base at infinity — a zero raw entry — at its floor. The bracket is `[-max_i b_i, hi]`, where the lower end caps every finite entry and `hi` is widened by doubling until the budget is at most one, and [`bounded_root`](@ref) bisects it. A set whose floors already sum to one is the point `lb`.

# Arguments

  - `b`: The mirror image of the raw step, `Inf` at a zero entry.
  - `phi`: The inverse mirror map of a positive base.
  - `wb`: The resolved bounds.

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub`. An `ArgumentError` is thrown otherwise.
  - At least one raw entry is positive, and the caps of the positive entries together with the floors of the zero ones reach the budget. A `DomainError` is thrown otherwise: the zeros stay zero under a barrier, and the rest cannot fill the budget.

# Returns

  - `w'::Vector`: The projected allocation.

# Related

  - [`project`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
  - [`bounded_root`](@ref)
"""
function barrier_projection(b::AbstractVector, phi, wb::WeightBounds)
    assert_feasible_bounds(wb)
    lb, ub = wb.lb, wb.ub
    if sum(lb) >= one(eltype(lb))
        return collect(lb)
    end
    finite = filter(isfinite, b)
    @argcheck(!isempty(finite),
              DomainError(b,
                          "a barrier projection needs a positive raw entry: a raw step of zeros has no projection, because a zero stays zero under the potential"))
    clipped = (m, l, u) -> m > zero(m) ? clamp(phi(m), l, u) : u
    f = mu -> sum(clipped.(b .+ mu, lb, ub))
    lo = -maximum(finite)
    @argcheck(f(lo) >= one(lo),
              DomainError(b,
                          "the zeros of the raw step stay zero under a barrier potential, and the caps of the remaining assets do not reach the budget"))
    mu = bounded_root(f, lo, barrier_upper_bracket(f, lo))
    return clipped.(b .+ mu, lb, ub)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The upper end of a barrier root's bracket: a multiplier above `lo` and above zero at which the clipped budget is at most one, found by doubling.

# Related

  - [`barrier_projection`](@ref)
"""
function barrier_upper_bracket(f, lo)
    hi = max(lo, zero(lo)) + one(lo)
    while f(hi) > one(hi)
        hi *= 2
    end
    return hi
end
"""
    project(proj::TsallisProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::LogBarrierProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)

The barrier arms of the Constrained Update on the bounded set: the scalar root of [`barrier_projection`](@ref) in the geometry's mirror image, on every bound, the simplex included, because neither potential has a closed form there.

# Validation

  - `all(>= 0, q)` and every entry finite. A `DomainError` is thrown otherwise.
  - `all(>= 0, lb)` over the resolved bounds. A `DomainError` is thrown otherwise.

# Related

  - [`project`](@ref)
  - [`barrier_projection`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
"""
function project(proj::TsallisProjection, set::BoundedAllocationSet, q::AbstractVector,
                 ::AbstractVector)
    assert_barrier_raw_step(q, set.wb)
    a = proj.alpha
    return barrier_projection(q .^ (a - 1), m -> m^inv(a - 1), set.wb)
end
function project(::LogBarrierProjection, set::BoundedAllocationSet, q::AbstractVector,
                 ::AbstractVector)
    assert_barrier_raw_step(q, set.wb)
    return barrier_projection(inv.(q), inv, set.wb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses, for a barrier geometry, a raw step with a negative or non-finite entry and a bound with a negative floor: the potential is undefined below zero.

# Related

  - [`project`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
"""
function assert_barrier_raw_step(q::AbstractVector, wb::WeightBounds)::Nothing
    @argcheck(all(x -> isfinite(x) && x >= zero(x), q),
              DomainError(q,
                          "a barrier projection is defined on finite non-negative raw steps alone: the potential is undefined below zero"))
    @argcheck(all(x -> x >= zero(x), wb.lb),
              DomainError(wb.lb,
                          "a barrier projection admits no negative lower bound: the potential is undefined below zero"))
    return nothing
end
"""
    assert_geometry_admits_set(proj::AbstractProjectionGeometry, set::AbstractAllocationSet)

Refuses, where the head's `alg` and `set` meet, a negative lower bound under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref), whose potentials are undefined below zero.

The refusal is at construction when the bound is a value; a bound resolved from an estimator is refused at the projection. Any other geometry admits a negative bound, so a long-short reversion costs nothing.

# Validation

  - Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) or [`LogBarrierProjection`](@ref) with a [`WeightBounds`](@ref) whose `lb` is a number or a vector: `all(>= 0, lb)`. A `DomainError` is thrown otherwise.

# Related

  - [`OnlinePortfolioSelection`](@ref)
  - [`EntropicProjection`](@ref)
  - [`project`](@ref)
"""
function assert_geometry_admits_set(::AbstractProjectionGeometry,
                                    ::AbstractAllocationSet)::Nothing
    return nothing
end
function assert_geometry_admits_set(proj::Union{<:EntropicProjection, <:TsallisProjection,
                                                <:LogBarrierProjection},
                                    set::AbstractAllocationSet)::Nothing
    wb = set.wb
    if isa(wb, WeightBounds) && !isnothing(wb.lb)
        @argcheck(all(x -> x >= zero(x), wb.lb),
                  DomainError(wb.lb,
                              "a `$(nameof(typeof(proj)))` admits no negative lower bound: its potential is undefined below zero. Use a Euclidean rule, or a non-negative bound."))
    end
    return nothing
end
"""
    set_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet, w::AbstractVector, X)
    set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X)

Adds the constraints of a resolved Allocation Set to a model that already carries `w`, `k`, the scales and the observation count: the budget one and the weight bounds on both kinds, and on the programme set every admissible kind through the shared builders, the turnover ceiling measured from `w`, the risk ceiling and the tracking error over `X`.

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
  - [`set_allocation_risk_ceiling!`](@ref)
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
    set_allocation_risk_ceiling!(model, set.r, set, pr)
    if !isnothing(set.te)
        set_tracking_error_constraints!(model, 1, pr, set.te)
    end
    return nothing
end
"""
    add_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet, w::AbstractVector, X)
    add_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X)

Adds the constraints of a resolved Allocation Set to a JuMP head's model mid-assembly, the arm of the Allocation Set Constraint a [`FollowTheLeader`](@ref) rule appends to its held optimiser.

The head's own builders have already registered the model's named entries — its weight bounds, its budget, its turnover and tracking-error terms under their indices — so this arm adds the set's rows without a name where the bare projection model names them: the bounds and the budget of one as anonymous constraints, the linear constraints under the `:aset_` prefix, the turnover ceiling and the tracking error at the first index the model has not used, the risk ceiling under the `:aset_` namespace with the head's `w` registered there, so its entries never meet the head's own measures'. Every row is the same inequality [`set_allocation_set_constraints!`](@ref) writes, so the leader's feasible region is the set intersected with whatever the head carries itself. The tracking error is written over the head's rows `X` and their count, not the selection's, which the model's own observation count and net-return expression describe.

A MIP kind of the set — cardinality, a threshold, a group cardinality — registers the model's indicator variables, which one model holds once, so a MIP kind is stated in one home: on the set, or on the held optimiser.

# Arguments

  - $(arg_dict[:model])
  - `set`: The Allocation Set, resolved.
  - `w`: The Price-Adjusted Allocation the step trades from, the reference of the turnover ceiling.
  - `X`: The rows of returns the head holds through the period, `observations × assets`, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_allocation_set_constraints!`](@ref)
  - [`AllocationSetConstraint`](@ref)
  - [`FollowTheLeader`](@ref)
"""
function add_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet,
                                         ::AbstractVector, ::Any)::Nothing
    add_allocation_set_bounds!(model, set.wb)
    return nothing
end
function add_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                         w::AbstractVector, X)::Nothing
    add_allocation_set_bounds!(model, set.wb)
    set_linear_weight_constraints!(model, set.lcs, :aset_lcs_ineq_, :aset_lcs_eq_)
    set_mip_constraints!(model, set.wb, set.card, set.gcard, nothing, set.lt, set.st,
                         nothing, set.ss)
    if !isnothing(set.tn)
        _set_turnover_constraints!(model, Turnover(; w = w, val = set.tn),
                                   free_state_index(model, :t_tn_))
    end
    pr = allocation_set_prior(set, X)
    set_allocation_risk_ceiling!(model, set.r, set, pr; prefix = :aset_)
    if !isnothing(set.te)
        add_allocation_tracking_error!(model, set.te, pr.X)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds an Allocation Set's weight bounds and its budget of one to a model that already names its own: the same three inequalities [`set_allocation_set_bounds!`](@ref) writes, as anonymous constraints.

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub` over the resolved bounds, through [`assert_feasible_bounds`](@ref).

# Related

  - [`add_allocation_set_constraints!`](@ref)
  - [`set_allocation_set_bounds!`](@ref)
"""
function add_allocation_set_bounds!(model::JuMP.Model, wb::WeightBounds)::Nothing
    assert_feasible_bounds(wb)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    if w_finite_flag(wb.lb)
        JuMP.@constraint(model, sc * (w ⊖ k * wb.lb) >= 0)
    end
    if w_finite_flag(wb.ub)
        JuMP.@constraint(model, sc * (w ⊖ k * wb.ub) <= 0)
    end
    JuMP.@constraint(model, sc * (sum(w) - k) == 0)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The first index `i` at which the Model State entry `name` is not registered, so a builder keyed by an index can add its term beside a head's own.

# Related

  - [`add_allocation_set_constraints!`](@ref)
  - [`state_set!`](@ref)
"""
function free_state_index(model::JuMP.Model, name::Symbol)
    i = 1
    while haskey(model, state_key(Symbol(""), name, i))
        i += 1
    end
    return i
end
"""
    add_allocation_tracking_error!(model::JuMP.Model, te::TrackingError, X::AbstractMatrix)

Adds an Allocation Set's tracking error to a JuMP head's model as anonymous constraints over the head's rows `X`: the deviation `X w − k b` from the benchmark series, its norm in the error's own `alg` — the `L1` norm as a norm-one cone, the `L2` and squared-`L2` norms as a second-order cone, the `p`-norm as `T` power cones, the `∞`-norm as a norm-infinity cone — and the ceiling `err` scaled by the row count as the head's own builders scale it.

# Related

  - [`add_allocation_set_constraints!`](@ref)
  - [`set_tracking_error_constraints!`](@ref)
"""
function add_allocation_tracking_error!(model::JuMP.Model, te::TrackingError,
                                        X::AbstractMatrix)::Nothing
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    T = size(X, 1)
    wb = tracking_benchmark(te.tr, X)
    dev = JuMP.@expression(model, X * w - wb * k)
    t = JuMP.@variable(model)
    f = add_allocation_tracking_cone!(model, te.alg, dev, t, T, te.err, sc)
    JuMP.@constraint(model, sc * (t - f * k) <= 0)
    return nothing
end
"""
    add_allocation_tracking_cone!(model::JuMP.Model, alg, dev, t, T::Integer, err::Number, sc)

The cone of one tracking-error norm over the deviation `dev` and its bound variable `t`, answering the ceiling's scale factor for `T` rows.

# Related

  - [`add_allocation_tracking_error!`](@ref)
"""
function add_allocation_tracking_cone!(model::JuMP.Model, ::L1Norm, dev, t, T::Integer,
                                       err::Number, sc)
    JuMP.@constraint(model, [sc * t; sc * dev] in JuMP.MOI.NormOneCone(1 + T))
    return err * T
end
function add_allocation_tracking_cone!(model::JuMP.Model,
                                       alg::Union{<:L2Norm, <:SquaredL2Norm}, dev, t,
                                       T::Integer, err::Number, sc)
    JuMP.@constraint(model, [sc * t; sc * dev] in JuMP.SecondOrderCone())
    return tracking_error_soc_factor(alg, err, T)
end
function add_allocation_tracking_cone!(model::JuMP.Model, alg::LpNorm, dev, t, T::Integer,
                                       err::Number, sc)
    @argcheck(alg.p > 1,
              DomainError(alg.p,
                          "`LpNorm.p` is $(alg.p), and the tracking error is the `p`-norm of the deviation, which the model states with a power cone of exponent `1 / p`, so `1 < p` must hold. State a value greater than `1`."))
    p_inv = inv(alg.p)
    r = JuMP.@variable(model, [1:T])
    for i in 1:T
        JuMP.@constraint(model,
                         [sc * r[i], sc * t, sc * dev[i]] in JuMP.MOI.PowerCone(p_inv))
    end
    JuMP.@constraint(model, sc * (sum(r) - t) == 0)
    scale = T - alg.ddof
    return err * (alg.p == 3 ? cbrt(scale) : scale^p_inv)
end
function add_allocation_tracking_cone!(model::JuMP.Model, alg::LInfNorm, dev, t, T::Integer,
                                       err::Number, sc)
    JuMP.@constraint(model, [sc * t; sc * dev] in JuMP.MOI.NormInfinityCone(1 + T))
    return err * (T - alg.ddof)
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
    set_projection_objective!(model::JuMP.Model, proj::EuclideanProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::EntropicProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::GramProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::TsallisProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::LogBarrierProjection, q::AbstractVector)

Sets the projection programme's objective in the geometry's divergence from the raw step `q`: the Euclidean distance through a second-order cone, the relative entropy ``\\sum_i w_i \\log (w_i / q_i)`` through a relative-entropy cone over the positive entries of `q` with the zero entries pinned at zero, the Gram norm ``\\lVert G (\\boldsymbol{w} - \\boldsymbol{q}) \\rVert`` with ``G^\\intercal G = A`` through a second-order cone, the Tsallis divergence ``-\\sum_i w_i^\\alpha / \\alpha + \\sum_i q_i^{\\alpha - 1} w_i`` through one power cone per positive entry, and the Itakura–Saito divergence ``-\\sum_i \\log w_i + \\sum_i w_i / q_i`` through one exponential cone per positive entry; under both barriers a zero entry of `q` is pinned at zero, as under the entropic arm.

The Euclidean and Gram arms minimise the norm rather than its square, which has the same minimiser and keeps the programme conic on every solver. The two barrier arms drop the terms constant in `w`, so their objective value is the divergence up to a constant, with the same minimiser.

# Validation

  - Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref): `all(>= 0, q)` and `sum(q) > 0`. A `DomainError` is thrown otherwise.
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
function set_projection_objective!(model::JuMP.Model, proj::TsallisProjection,
                                   q::AbstractVector)::Nothing
    pos, w, sc, so = barrier_objective_entries(model, q)
    a = proj.alpha
    JuMP.@variable(model, t_proj[1:length(pos)])
    # `t_i ≤ w_i^α` is the power cone `w_i^α · 1^(1 − α) ≥ |t_i|`.
    for (i, j) in enumerate(pos)
        JuMP.@constraint(model, [sc * w[j], sc, sc * t_proj[i]] in JuMP.MOI.PowerCone(a))
    end
    obj = LinearAlgebra.dot(so .* q[pos] .^ (a - 1), w[pos]) - (so / a) * sum(t_proj)
    JuMP.set_objective(model, JuMP.MIN_SENSE, obj)
    return nothing
end
function set_projection_objective!(model::JuMP.Model, ::LogBarrierProjection,
                                   q::AbstractVector)::Nothing
    pos, w, sc, so = barrier_objective_entries(model, q)
    JuMP.@variable(model, t_proj[1:length(pos)])
    # `t_i ≤ log w_i` is the exponential cone `1 · exp(t_i) ≤ w_i`.
    for (i, j) in enumerate(pos)
        JuMP.@constraint(model,
                         [sc * t_proj[i], sc, sc * w[j]] in JuMP.MOI.ExponentialCone())
    end
    obj = LinearAlgebra.dot(so .* inv.(q[pos]), w[pos]) - so * sum(t_proj)
    JuMP.set_objective(model, JuMP.MIN_SENSE, obj)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The entries a barrier objective is written over: refuses a raw step with a negative entry or none positive, pins every zero entry of `q` at zero, and answers the positive indices with the model's `w` and its two scales.

# Related

  - [`set_projection_objective!`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
"""
function barrier_objective_entries(model::JuMP.Model, q::AbstractVector)
    @argcheck(all(x -> x >= zero(x), q),
              DomainError(q,
                          "a barrier projection is defined on non-negative raw steps alone: the potential is undefined below zero"))
    @argcheck(sum(q) > zero(eltype(q)),
              DomainError(q,
                          "a barrier projection needs a positive entry: a raw step of zeros has no projection, because a zero stays zero under the potential"))
    w = get_w(model)
    sc = get_constraint_scale(model)
    for i in findall(iszero, q)
        JuMP.@constraint(model, sc * w[i] == 0)
    end
    return findall(x -> x > zero(x), q), w, sc, get_objective_scale(model)
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

Whether a set's constraints can be formed on the rows `X` the step holds: `nothing` when they can, and the reason for a Held Step when they cannot — a programme set that fits its prior on the rows, while the head holds fewer than two, because a covariance of one observation does not exist.

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
    project(proj::TsallisProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::LogBarrierProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)

The programme arms of the Constrained Update: every pair but the scalar roots on the bounded set is the bare-model programme of [`projection_programme`](@ref).

# Validation

  - Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref): `all(>= 0, lb)` over the resolved bounds. A `DomainError` is thrown otherwise. Under the same three geometries the programme's answer passes through [`clip_at_zero`](@ref), so a leg the solver closed to within its tolerance below zero is zero to the next step.

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
    return clip_at_zero(projection_programme(proj, set, q, w))
end
function project(proj::GramProjection, set::ProgrammeAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    return projection_programme(proj, set, q, w)
end
function project(proj::Union{<:TsallisProjection, <:LogBarrierProjection},
                 set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)
    @argcheck(all(x -> x >= zero(x), set.wb.lb),
              DomainError(set.wb.lb,
                          "a barrier projection admits no negative lower bound: the potential is undefined below zero"))
    return clip_at_zero(projection_programme(proj, set, q, w))
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
export ProgrammeAllocationSet, TsallisProjection, LogBarrierProjection
public set_allocation_set_constraints!, mirror_step
