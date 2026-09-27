"""
    assert_risk_ceiling(r::Nothing)
    assert_risk_ceiling(r::RiskMeasure)
    assert_risk_ceiling(rs::VecRM)

Refuses a risk measure that cannot be the ceiling of a [`ProgrammeAllocationSet`](@ref).

The ceiling of a measure is its `settings.ub`, and a one-step projection can bound the measure by one number only. The function refuses a measure with no `ub`, with a `Frontier` or with a per-asset vector, and it refuses an empty vector of measures. A set with no measure passes.

# Arguments

  - `r`: The set's risk measure, `nothing`, or a vector of measures.

# Validation

  - `settings.ub` of a measure is not a finite non-negative number. An `ArgumentError` is thrown.
  - `rs` is empty. An `IsEmptyError` is thrown.

# Returns

  - `nothing`.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`set_allocation_risk_ceiling!`](@ref)
  - [`set_risk_upper_bound!`](@ref): refuses a frontier or a vector for a set that reaches the builder without this check.
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

Tells whether a programme set's ceiling reads the head's rows.

A [`Variance`](@ref) or a [`StandardDeviation`](@ref) that holds its covariance matrix reads no rows. A variance with risk-contribution rows in `rc` is the exception: the shared semidefinite builder resolves those rows against a prior, so it reads them. Every other measure is built on a prior fitted on the rows, so it reads them too. A vector reads the rows when one of its members does.

# Arguments

  - `r`: The set's risk measure, `nothing`, or a vector of measures.

# Returns

  - `flag::Bool`: `true` when the ceiling reads the head's rows.

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

Fits the prior result that a programme set's row-reading slots are built on.

The set fits its `pe` on the head's rows carrier the same way a batch head fits a prior on any carrier. The fit reads the rows as they are and the Asset Panel beside them, so the result answers `NaN` at an asset outside the prior's Coverage Universe. A set that reads no rows fits nothing, and a bounded set never fits anything.

# Algorithm

 1. Ask [`rows_needed`](@ref) whether the set reads the head's rows. It answers `nothing` when a slot reads them.
 2. If no slot reads them, return `nothing`.
 3. Check that `X` is not `nothing`.
 4. Fit `set.pe` on `X` with [`prior`](@ref), and return the result.

# Arguments

  - `set`: The resolved Allocation Set.
  - `X`: The head's rows carrier, a [`ReturnsResult`](@ref), or `nothing` outside an Online Update.

# Validation

  - A slot reads the rows and `X` is `nothing`. An `ArgumentError` is thrown.

# Returns

  - `pr::Option{<:AbstractPriorResult}`: The fitted prior result, or `nothing`.

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

Reads the Investable Mask of a programme set's prior result.

# Arguments

  - `pr`: The set's prior result, or `nothing` when the set fitted none.

# Returns

  - `imsk::Option{<:BitVector}`: The mask, or `nothing` when the set fitted no prior or the prior prices every asset.

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

Refuses a programme set prior that cannot price an asset the model trades, and names the asset.

The projection programme reduces the model to the prior's Investable Mask before it builds, so its prior always passes. A leader's model trades the Investable Mask of the leader's own prior, and the set fits its prior on the same rows. When both priors are plain, the two masks agree and the check passes. When the set's `pe` is plain and the leader's prior is mask-aware, a young asset that the leader trades can have no price in the set's prior. Its ceiling would then carry `NaN`, so the check refuses it here, before the model reaches the solver.

# Algorithm

 1. Read the Investable Mask of `pr`, `imsk`.
 2. If `imsk` is `nothing`, the prior prices every asset: return.
 3. Otherwise, throw an error that names the assets of `X.nx` outside `imsk`.

# Arguments

  - `pr`: The set's prior result, or `nothing` when the set fitted none.
  - `X`: The head's rows carrier, or `nothing`.

# Validation

  - `pr` carries an Investable Mask. An `ArgumentError` is thrown, and it names the assets the prior cannot price.

# Returns

  - `nothing`.

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

Adds a programme set's risk ceilings to the model, one row block for each measure, at the measure's index.

Each ceiling goes through the measure's own JuMP builder, [`set_risk_constraints!`](@ref), with the set as the [`RiskConstraintOwner`](@ref). The builder bounds each measure at its `settings.ub` through [`set_risk_upper_bound!`](@ref), and its own `# JuMP formulation` names the rows. So the set writes the rows that a JuMP optimiser writes for the same bound. A [`Variance`](@ref) takes the semidefinite formulation under the conditions that select it in a head, [`sdp_variance_flag!`](@ref): risk-contribution rows, an earlier variance with such rows on the same weights, or a semidefinite phylogeny in `pl`. Otherwise it takes the second-order cone. The variance marks `variance_flag`, so a semidefinite phylogeny on the same weights adds no `p · tr(W)` penalty, as in a head.

The source of the covariance matrix does not change the formulation. With a prior result, each measure is resolved against it first. Without one, every ceiling is a [`Variance`](@ref) or a [`StandardDeviation`](@ref) that holds its matrix, on a set that reads no rows, and the builder reads that matrix with `pr = nothing`.

Under a `prefix`, the builders put their entries in that namespace. [`assemble_allocation_set!`](@ref) registers the head's `w` under the prefix before any builder runs. This keeps the ceiling's entries apart from the entries of the head's own measures. The lifted matrix ``\\mathbf{W}`` and the marks belong to the weights, [`weights_prefix`](@ref), so in a leader's model the ceiling, the set's phylogeny and the leader's measures read one ``\\mathbf{W}``.

# Algorithm

For one measure:

 1. Wrap `r` in a vector and call the vector method.

For a vector and a prior result:

 1. Resolve each measure against `pr` with [`factory`](@ref) and `set.slv`. A moment that the measure holds, such as its own covariance, stays. A cache that the head filled from another prior is not read.
 2. Clear the measure's `rke` with [`no_risk_expr_risk_measure`](@ref), so its expression joins no objective. The measure's `scale` weights its expression only in the objective, so a ceiling ignores it.
 3. Call [`set_risk_constraints!`](@ref) on the vector of ceilings, with the set as owner, `pr`, `pl` and `prefix`.

For a vector and no prior result:

 1. For each measure at index `i`, clear its `rke`, as above.
 2. Call the index-aware [`set_risk_constraints!`](@ref) at `i`, with the set as owner, `pr = nothing`, `pl` and `prefix`.

# Arguments

  - $(arg_dict[:model])
  - `r`: The set's risk measure, `nothing`, or a vector of measures.
  - `set`: The resolved programme Allocation Set.
  - `pr`: The set's prior result, or `nothing` when the set reads no rows.
  - `pl`: The set's resolved phylogeny constraints, or `nothing`.
  - `prefix`: The Model State namespace of the entries, `Symbol("")` on the bare projection model.

# Returns

  - `nothing`.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`sdp_variance_flag!`](@ref)
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
                                      set::ProgrammeAllocationSet, ::Nothing, pl;
                                      prefix::Symbol = Symbol(""))::Nothing
    for (i, r) in enumerate(rs)
        set_risk_constraints!(model, i, no_risk_expr_risk_measure(r), set, nothing, pl,
                              nothing, nothing; prefix = prefix)
    end
    return nothing
end
function set_allocation_risk_ceiling!(model::JuMP.Model, rs::VecRM,
                                      set::ProgrammeAllocationSet, pr::AbstractPriorResult,
                                      pl; prefix::Symbol = Symbol(""))::Nothing
    ceilings = [no_risk_expr_risk_measure(factory(r, pr, set.slv)) for r in rs]
    set_risk_constraints!(model, ceilings, set, pr, pl, nothing; prefix = prefix)
    return nothing
end
"""
    risk_constraint_solver(set::ProgrammeAllocationSet)

Returns the set's own `slv`, the solver that resolves a Deferred Quantity of the set's ceiling.

# Arguments

  - `set`: The programme Allocation Set.

# Returns

  - `slv`: The set's solver or vector of solvers.

# Related

  - [`RiskConstraintOwner`](@ref)
  - [`set_allocation_risk_ceiling!`](@ref)
"""
function risk_constraint_solver(set::ProgrammeAllocationSet)
    return set.slv
end
"""
    risk_contribution_constraints(r::Variance, set::ProgrammeAllocationSet, pr::AbstractPriorResult)
    risk_contribution_constraints(r::Variance, set::ProgrammeAllocationSet, pr::Nothing)

Resolves the risk-contribution rows of a [`Variance`](@ref) ceiling on a programme set.

The set resolves `r.rc` over its own `sets`, under the `strict` of the head that runs the step, the same way an optimiser resolves its own rows. When there are rows, the shared builder takes the semidefinite formulation.

Without a prior result the set reads no rows. [`risk_reads_rows`](@ref) sends a variance with rows to the prior, so `r.rc` is `nothing` here, and the method returns it.

# Arguments

  - `r`: The variance ceiling.
  - `set`: The programme Allocation Set.
  - `pr`: The set's prior result, or `nothing` when the set reads no rows.

# Returns

  - `rc::Option{<:LinearConstraint}`: The resolved rows, or `nothing` when `r.rc` is `nothing`.

# Related

  - [`RiskConstraintOwner`](@ref)
  - [`linear_constraints`](@ref)
  - [`risk_reads_rows`](@ref)
"""
function risk_contribution_constraints(r::Variance, set::ProgrammeAllocationSet,
                                       pr::AbstractPriorResult)
    return linear_constraints(r.rc, set.sets; datatype = eltype(pr.X),
                              strict = projection_step_strict())
end
function risk_contribution_constraints(r::Variance, ::ProgrammeAllocationSet, ::Nothing)
    return r.rc
end
"""
    clip_at_zero(w::AbstractVector)

Raises every negative entry of a projection programme's answer to zero.

A geometry whose domain is the non-negative orthant needs this. The solver meets a zero lower bound only to within its tolerance, so a leg that the programme closes can come back slightly below zero. The next step forms its raw step multiplicatively from the allocation it holds, and would then take `log w` of a negative number. The clip does not renormalise, so the sum rises by the clipped mass, which is at the size of the solver's tolerance.

# Mathematical definition

```math
\\begin{align}
w'_i &= \\max(w_i, 0)\\,.
\\end{align}
```

Where:

  - ``w_i``: Entry ``i`` of the programme's answer.
  - ``w'_i``: Entry ``i`` of the clipped answer.

# Arguments

  - `w`: The programme's answer.

# Returns

  - `w'::Vector`: `w` with no entry below zero, in the element type of `w`.

# Related

  - [`project`](@ref)
  - [`projection_programme`](@ref)
"""
function clip_at_zero(w::AbstractVector)
    return map(x -> max(x, zero(x)), w)
end
