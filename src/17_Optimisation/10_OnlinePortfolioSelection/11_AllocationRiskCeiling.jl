"""
    assert_risk_ceiling(r::Nothing)
    assert_risk_ceiling(r::RiskMeasure)
    assert_risk_ceiling(r::Variance)

Refuses a risk measure on a [`ProgrammeAllocationSet`](@ref) whose `settings.ub` is not the ceiling — a finite non-negative number, so neither a frontier nor a per-asset vector — and a [`Variance`](@ref) with risk-contribution rows on `rc`, which need the semidefinite formulation the set does not admit.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`set_allocation_risk_ceiling!`](@ref)
"""
function assert_risk_ceiling(::Nothing)::Nothing
    return nothing
end
function assert_risk_ceiling(r::RiskMeasure)::Nothing
    ub = r.settings.ub
    @argcheck(isa(ub, Number) && isfinite(ub) && ub >= zero(ub),
              ArgumentError("the ceiling of a `$(nameof(typeof(r)))` on a ProgrammeAllocationSet is its `settings.ub`, a finite non-negative number; got `$ub`"))
    return nothing
end
function assert_risk_ceiling(r::Variance)::Nothing
    @argcheck(isnothing(r.rc),
              ArgumentError("a Variance on a ProgrammeAllocationSet admits no risk-contribution rows on `rc`: they need the semidefinite variance, which the set does not build"))
    invoke(assert_risk_ceiling, Tuple{RiskMeasure}, r)
    return nothing
end
"""
    risk_reads_rows(r::Nothing)
    risk_reads_rows(r::Union{<:Variance, <:StandardDeviation})
    risk_reads_rows(r::RiskMeasure)

Whether a programme set's ceiling reads the head's rows: a variance or standard deviation holding its matrix reads none, and every other measure is built on the prior result fitted on them.

# Related

  - [`rows_needed`](@ref)
  - [`set_allocation_risk_ceiling!`](@ref)
"""
function risk_reads_rows(::Nothing)
    return false
end
function risk_reads_rows(r::Union{<:Variance, <:StandardDeviation})
    return !isa(r.sigma, AbstractMatrix)
end
function risk_reads_rows(::RiskMeasure)
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The prior result a programme set's risk ceiling and tracking error read: `pe` fitted on the head's rows, or `nothing` when the set reads none.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
"""
function allocation_set_prior(set::ProgrammeAllocationSet, X)
    if isnothing(rows_needed(set))
        @argcheck(!isnothing(X),
                  ArgumentError("a ProgrammeAllocationSet with a risk ceiling built on its prior or a tracking error reads the head's rows, which reach a projection inside an Online Update alone; `project` was called with none."))
        assert_factor_returns(set.pe, nothing)
        return prior(set.pe, X, nothing, nothing)
    end
    return nothing
end
"""
    set_allocation_risk_ceiling!(model::JuMP.Model, r::Nothing, set::ProgrammeAllocationSet, pr; prefix::Symbol = Symbol(""))
    set_allocation_risk_ceiling!(model::JuMP.Model, r::Union{<:Variance, <:StandardDeviation}, set::ProgrammeAllocationSet, pr::Nothing; prefix::Symbol = Symbol(""))
    set_allocation_risk_ceiling!(model::JuMP.Model, r::RiskMeasure, set::ProgrammeAllocationSet, pr::AbstractPriorResult; prefix::Symbol = Symbol(""))

Adds a programme set's risk ceiling to the model.

On a prior result, the ceiling is the measure's own JuMP builder, [`set_risk_constraints!`](@ref), with the set as the [`RiskConstraintOwner`](@ref): the measure is materialised against the prior through [`factory`](@ref) first, so a moment it carries itself — its own covariance, its own cokurtosis — is the one it is built on and a cache the head filled from another prior is never read, and its `rke` is cleared through [`no_risk_expr_risk_measure`](@ref), so the expression joins no objective. The bound is `settings.ub` through [`set_risk_upper_bound!`](@ref). Under a `prefix` the builder's entries are namespaced, and the model's `w` is registered under it when it is not yet, which is how the head's arm keeps the ceiling's entries apart from the head's own measures'.

Without a prior result — a [`Variance`](@ref) or [`StandardDeviation`](@ref) holding its matrix on a set that reads no rows — there is no prior to build on, and the ceiling is the set's own second-order cone ``[u k; G \\boldsymbol{w}] \\in \\mathcal{K}_{\\mathrm{SOC}}`` with ``G`` the factor of the matrix and ``u`` the ceiling of [`allocation_risk_ceiling`](@ref): the same cone the shared builder writes. The factor is taken without a definiteness check: a matrix the caller left indefinite reaches the solver, whose failure is the step's Held Step.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`allocation_risk_ceiling`](@ref)
  - [`HeldStep`](@ref)
"""
function set_allocation_risk_ceiling!(::JuMP.Model, ::Nothing, ::ProgrammeAllocationSet,
                                      ::Any; prefix::Symbol = Symbol(""))::Nothing
    return nothing
end
function set_allocation_risk_ceiling!(model::JuMP.Model,
                                      r::Union{<:Variance, <:StandardDeviation},
                                      ::ProgrammeAllocationSet, ::Nothing;
                                      prefix::Symbol = Symbol(""))::Nothing
    G = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(r.sigma); check = false).U
    u = allocation_risk_ceiling(r)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    state_set!(model, prefix, :set_risk_soc,
               JuMP.@constraint(model,
                                [sc * u * k; sc * (G * w)] in JuMP.SecondOrderCone()))
    return nothing
end
function set_allocation_risk_ceiling!(model::JuMP.Model, r::RiskMeasure,
                                      set::ProgrammeAllocationSet, pr::AbstractPriorResult;
                                      prefix::Symbol = Symbol(""))::Nothing
    if !state_has(model, prefix, :w)
        state_set!(model, prefix, :w, get_w(model))
    end
    ceiling = no_risk_expr_risk_measure(factory(r, pr, set.slv))
    set_risk_constraints!(model, ceiling, set, pr, nothing, nothing; prefix = prefix)
    return nothing
end
"""
    risk_constraint_solver(set::ProgrammeAllocationSet)

The solver a Deferred Quantity of the set's ceiling is resolved against: the set's own `slv`.

# Related

  - [`RiskConstraintOwner`](@ref)
  - [`set_allocation_risk_ceiling!`](@ref)
"""
function risk_constraint_solver(set::ProgrammeAllocationSet)
    return set.slv
end
"""
    risk_contribution_constraints(r::Variance, set::ProgrammeAllocationSet, pr::AbstractPriorResult)

The risk-contribution rows of a [`Variance`](@ref) ceiling on a programme set: `r.rc` as it is, which the set's constructor holds at `nothing` ([`assert_risk_ceiling`](@ref)), so the shared builder takes the second-order-cone formulation.

# Related

  - [`RiskConstraintOwner`](@ref)
  - [`assert_risk_ceiling`](@ref)
"""
function risk_contribution_constraints(r::Variance, ::ProgrammeAllocationSet,
                                       ::AbstractPriorResult)
    return r.rc
end
"""
    allocation_risk_ceiling(r::Variance)
    allocation_risk_ceiling(r::StandardDeviation)

The bound on the cone variable of [`set_allocation_risk_ceiling!`](@ref)'s own cone: the square root of a [`Variance`](@ref)'s `settings.ub`, and a [`StandardDeviation`](@ref)'s as it is.

# Related

  - [`set_allocation_risk_ceiling!`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
"""
function allocation_risk_ceiling(r::Variance)
    return sqrt(r.settings.ub)
end
function allocation_risk_ceiling(r::StandardDeviation)
    return r.settings.ub
end
"""
    clip_at_zero(w::AbstractVector)

The answer of a projection programme in a geometry whose domain is the non-negative orthant, with every entry below zero raised to it. A solver answers a bound it holds to its own tolerance, so a leg the programme closed comes back a few ulps below zero, and the next step's raw step, which the geometry forms multiplicatively from the allocation it holds, would carry the sign into `log w`. The clip is the size of the solver's tolerance and never renormalises: the sum stays what the solver answered.

# Arguments

  - `w`: The programme's answer.

# Returns

  - `w'::Vector`: `w` with no entry below zero.

# Related

  - [`project`](@ref)
  - [`projection_programme`](@ref)
"""
function clip_at_zero(w::AbstractVector)
    return map(x -> max(x, zero(x)), w)
end
