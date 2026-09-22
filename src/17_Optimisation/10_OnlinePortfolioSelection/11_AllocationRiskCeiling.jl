"""
    assert_risk_ceiling(r::Nothing)
    assert_risk_ceiling(r::RiskMeasure)
    assert_risk_ceiling(rs::VecRM)

Refuses a risk measure on a [`ProgrammeAllocationSet`](@ref) whose `settings.ub` is not the ceiling — a finite non-negative number, so neither a frontier nor a per-asset vector — and an empty vector of them.

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
function assert_risk_ceiling(rs::VecRM)::Nothing
    @argcheck(!isempty(rs), IsEmptyError("r cannot be empty"))
    for r in rs
        assert_risk_ceiling(r)
    end
    return nothing
end
"""
    risk_reads_rows(r::Nothing)
    risk_reads_rows(r::Variance)
    risk_reads_rows(r::StandardDeviation)
    risk_reads_rows(r::RiskMeasure)
    risk_reads_rows(rs::VecRM)

Whether a programme set's ceilings read the head's rows: a variance or standard deviation holding its matrix reads none, unless the variance carries risk-contribution rows on `rc`, which the shared semidefinite builder writes on the prior; every other measure is built on the prior result fitted on them; a vector reads them when any member does.

# Related

  - [`rows_needed`](@ref)
  - [`set_allocation_risk_ceiling!`](@ref)
"""
function risk_reads_rows(::Nothing)
    return false
end
function risk_reads_rows(r::Variance)
    return !isa(r.sigma, AbstractMatrix) || !isnothing(r.rc)
end
function risk_reads_rows(r::StandardDeviation)
    return !isa(r.sigma, AbstractMatrix)
end
function risk_reads_rows(::RiskMeasure)
    return true
end
function risk_reads_rows(rs::VecRM)
    return any(risk_reads_rows, rs)
end
"""
    allocation_set_prior(set::ProgrammeAllocationSet, X::Option{<:ReturnsResult})
    allocation_set_prior(set::BoundedAllocationSet, X::Option{<:ReturnsResult})

The prior result a programme set's row-reading slots are built on: `pe` fitted on the head's rows carrier as a batch prior is fitted on any carrier — the rows verbatim and the Asset Panel beside them, so the result answers `NaN` at an asset outside the prior's Coverage Universe — or `nothing` when the set reads none, and always `nothing` on a bounded set, which fits nothing.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`projection_programme`](@ref)
"""
function allocation_set_prior(set::ProgrammeAllocationSet, X::Option{<:ReturnsResult})
    if isnothing(rows_needed(set))
        @argcheck(!isnothing(X),
                  ArgumentError("a ProgrammeAllocationSet with a slot that reads the head's rows — a ceiling built on its prior, a tracking error, a centrality or phylogeny estimator, an exposure row, a return floor or a Calibration Rule — reads them inside an Online Update alone; `project` was called with none."))
        return prior(set.pe, X)
    end
    return nothing
end
function allocation_set_prior(::BoundedAllocationSet, ::Option{<:ReturnsResult})
    return nothing
end
"""
    prior_investable_mask(pr::Nothing)
    prior_investable_mask(pr::AbstractPriorResult)

The Investable Mask of a programme set's prior result, or `nothing` when the set fitted none.

# Related

  - [`investable_mask`](@ref)
  - [`projection_programme`](@ref)
"""
function prior_investable_mask(::Nothing)
    return nothing
end
function prior_investable_mask(pr::AbstractPriorResult)
    return investable_mask(pr)
end
"""
    assert_set_prior_priced(pr::Nothing, X)
    assert_set_prior_priced(pr::AbstractPriorResult, X::Nothing)
    assert_set_prior_priced(pr::AbstractPriorResult, X::ReturnsResult)

Refuses a programme set prior that cannot price an asset the model trades, by name.

The projection programme reduces the model to the prior's Investable Mask before it builds, so its prior passes. A leader's model trades the Investable Mask of the leader's own prior, and the set's prior is fitted on the same rows: under a plain `pe` on both the two masks agree, and the check passes; a set whose `pe` is plain beside a leader whose prior is mask-aware can leave a young asset unpriced that the leader trades, and its ceiling would carry `NaN` there, so it is refused here rather than at the solver.

# Related

  - [`assemble_allocation_set!`](@ref)
  - [`add_allocation_set_constraints!`](@ref)
  - [`investable_mask`](@ref)
"""
function assert_set_prior_priced(::Nothing, ::Any)::Nothing
    return nothing
end
function assert_set_prior_priced(::AbstractPriorResult, ::Nothing)::Nothing
    return nothing
end
function assert_set_prior_priced(pr::AbstractPriorResult, X::ReturnsResult)::Nothing
    imsk = investable_mask(pr)
    if isnothing(imsk)
        return nothing
    end
    return throw(ArgumentError("the Allocation Set's prior cannot price $(non_investable_names(X.nx, imsk)), which the programme trades: an asset outside the set prior's Coverage Universe over the rows it was fitted on has no covariance, no mean and no loadings for a ceiling, a floor or a tracking error to read. Give the set a `pe` whose universe covers the programme's — a mask-aware one under a `CoveragePolicy`, or the held estimator's own — or shorten the window the set reads."))
end
"""
    set_allocation_risk_ceiling!(model::JuMP.Model, r::Nothing, set::ProgrammeAllocationSet, pr, pl; prefix::Symbol = Symbol(""))
    set_allocation_risk_ceiling!(model::JuMP.Model, r::RiskMeasure, set::ProgrammeAllocationSet, pr, pl; prefix::Symbol = Symbol(""))
    set_allocation_risk_ceiling!(model::JuMP.Model, rs::VecRM, set::ProgrammeAllocationSet, pr::Nothing, pl; prefix::Symbol = Symbol(""))
    set_allocation_risk_ceiling!(model::JuMP.Model, rs::VecRM, set::ProgrammeAllocationSet, pr::AbstractPriorResult, pl; prefix::Symbol = Symbol(""))

Adds a programme set's risk ceilings to the model, one measure or a vector, each at its index.

On a prior result, every ceiling is the measure's own JuMP builder, [`set_risk_constraints!`](@ref), with the set as the [`RiskConstraintOwner`](@ref) and the resolved phylogeny `pl` in hand, so a [`Variance`](@ref) under a semidefinite phylogeny takes the semidefinite formulation as it does in a head: the measure is materialised against the prior through [`factory`](@ref) first, so a moment it carries itself — its own covariance, its own cokurtosis — is the one it is built on and a cache the head filled from another prior is never read, its `rke` is cleared through [`no_risk_expr_risk_measure`](@ref), so the expression joins no objective, and its `scale` is dropped, because a ceiling is not a combination. The bound is `settings.ub` through [`set_risk_upper_bound!`](@ref). Under a `prefix` the builder's entries are namespaced, and the head's `w` is registered under it by [`assemble_allocation_set!`](@ref) before any builder runs, which is how the head's arm keeps the ceiling's entries apart from the head's own measures'.

Without a prior result — every ceiling a [`Variance`](@ref) or [`StandardDeviation`](@ref) holding its matrix on a set that reads no rows — there is no prior to build on, and each ceiling is the set's own cone ([`set_matrix_risk_ceiling!`](@ref)).

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_matrix_risk_ceiling!`](@ref)
  - [`HeldStep`](@ref)
"""
function set_allocation_risk_ceiling!(::JuMP.Model, ::Nothing, ::ProgrammeAllocationSet,
                                      ::Any, ::Any; prefix::Symbol = Symbol(""))::Nothing
    return nothing
end
function set_allocation_risk_ceiling!(model::JuMP.Model, r::RiskMeasure,
                                      set::ProgrammeAllocationSet, pr, pl;
                                      prefix::Symbol = Symbol(""))::Nothing
    set_allocation_risk_ceiling!(model, [r], set, pr, pl; prefix = prefix)
    return nothing
end
function set_allocation_risk_ceiling!(model::JuMP.Model, rs::VecRM,
                                      ::ProgrammeAllocationSet, ::Nothing, pl;
                                      prefix::Symbol = Symbol(""))::Nothing
    for (i, r) in enumerate(rs)
        set_matrix_risk_ceiling!(model, i, r, pl; prefix = prefix)
    end
    return nothing
end
function set_allocation_risk_ceiling!(model::JuMP.Model, rs::VecRM,
                                      set::ProgrammeAllocationSet, pr::AbstractPriorResult,
                                      pl; prefix::Symbol = Symbol(""))::Nothing
    ceilings = [unit_scale_risk_measure(no_risk_expr_risk_measure(factory(r, pr, set.slv)))
                for r in rs]
    set_risk_constraints!(model, ceilings, set, pr, pl, nothing; prefix = prefix)
    return nothing
end
"""
    set_matrix_risk_ceiling!(model::JuMP.Model, i::Integer, r::Variance, pl; prefix::Symbol = Symbol(""))
    set_matrix_risk_ceiling!(model::JuMP.Model, i::Integer, r::StandardDeviation, pl; prefix::Symbol = Symbol(""))

The set's own cone for a ceiling that holds its matrix and reads no prior, at index `i`: the second-order cone ``[u k; G \\boldsymbol{w}] \\in \\mathcal{K}_{\\mathrm{SOC}}`` with ``G`` the factor of the matrix and ``u`` the ceiling of [`allocation_risk_ceiling`](@ref), the same cone the shared builder writes; or, for a [`Variance`](@ref) when [`sdp_variance_flag!`](@ref) selects the semidefinite formulation — a semidefinite phylogeny in `pl` — the row ``\\mathrm{tr}(\\boldsymbol{\\Sigma} W) \\leq \\mathrm{ub}\\, k`` on the model's one lifted `W`, which the phylogeny's rows share. The factor is taken without a definiteness check: a matrix the caller left indefinite reaches the solver, whose failure is the step's Held Step.

# Related

  - [`set_allocation_risk_ceiling!`](@ref)
  - [`allocation_risk_ceiling`](@ref)
  - [`set_sdp_constraints!`](@ref)
  - [`sdp_variance_flag!`](@ref)
"""
function set_matrix_risk_ceiling!(model::JuMP.Model, i::Integer, r::Variance, pl;
                                  prefix::Symbol = Symbol(""))::Nothing
    if isa(sdp_variance_flag!(model, false, pl; prefix = prefix), LinearBound)
        W = set_sdp_constraints!(model)
        k = get_k(model)
        sc = get_constraint_scale(model)
        ub = r.settings.ub
        state_set!(model, prefix, :set_risk_sdp_, i,
                   JuMP.@constraint(model,
                                    sc * (LinearAlgebra.tr(r.sigma * W) - ub * k) <= 0))
        return nothing
    end
    set_matrix_risk_soc!(model, i, r; prefix = prefix)
    return nothing
end
function set_matrix_risk_ceiling!(model::JuMP.Model, i::Integer, r::StandardDeviation,
                                  ::Any; prefix::Symbol = Symbol(""))::Nothing
    set_matrix_risk_soc!(model, i, r; prefix = prefix)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The second-order cone of [`set_matrix_risk_ceiling!`](@ref) at index `i`.

# Related

  - [`set_matrix_risk_ceiling!`](@ref)
"""
function set_matrix_risk_soc!(model::JuMP.Model, i::Integer,
                              r::Union{<:Variance, <:StandardDeviation};
                              prefix::Symbol = Symbol(""))::Nothing
    G = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(r.sigma); check = false).U
    u = allocation_risk_ceiling(r)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    state_set!(model, prefix, :set_risk_soc_, i,
               JuMP.@constraint(model,
                                [sc * u * k; sc * (G * w)] in JuMP.SecondOrderCone()))
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

The risk-contribution rows of a [`Variance`](@ref) ceiling on a programme set: `r.rc` resolved over the set's `sets` under the head's `strict`, as the optimiser resolves its own, so the shared builder takes the semidefinite formulation when there are rows.

# Related

  - [`RiskConstraintOwner`](@ref)
  - [`linear_constraints`](@ref)
"""
function risk_contribution_constraints(r::Variance, set::ProgrammeAllocationSet,
                                       pr::AbstractPriorResult)
    return linear_constraints(r.rc, set.sets; datatype = eltype(pr.X),
                              strict = projection_step_strict())
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
