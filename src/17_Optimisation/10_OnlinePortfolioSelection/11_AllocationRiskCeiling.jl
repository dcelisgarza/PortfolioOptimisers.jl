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

With a prior result, each ceiling goes through the measure's own JuMP builder, [`set_risk_constraints!`](@ref). The set is the [`RiskConstraintOwner`](@ref), and the builder also reads the resolved phylogeny `pl`, so a [`Variance`](@ref) under a semidefinite phylogeny takes the semidefinite formulation, as it does in a head. The builder bounds each measure at its `settings.ub` through [`set_risk_upper_bound!`](@ref), and its own `# JuMP formulation` names the rows.

Without a prior result, every ceiling is a [`Variance`](@ref) or a [`StandardDeviation`](@ref) that holds its matrix, on a set that reads no rows. There is no prior to build on, so each ceiling is the set's own cone, [`set_matrix_risk_ceiling!`](@ref).

Under a `prefix`, the builders put their entries in that namespace. [`assemble_allocation_set!`](@ref) registers the head's `w` under the prefix before any builder runs. This keeps the ceiling's entries apart from the entries of the head's own measures.

# Algorithm

For one measure:

 1. Wrap `r` in a vector and call the vector method.

For a vector and a prior result:

 1. Resolve each measure against `pr` with [`factory`](@ref) and `set.slv`. A moment that the measure holds, such as its own covariance, stays. A cache that the head filled from another prior is not read.
 2. Clear the measure's `rke` with [`no_risk_expr_risk_measure`](@ref), so its expression joins no objective.
 3. Set the measure's `scale` to one with [`unit_scale_risk_measure`](@ref), because a ceiling is not a combination of measures.
 4. Call [`set_risk_constraints!`](@ref) on the vector of ceilings, with the set as owner, `pr`, `pl` and `prefix`.

For a vector and no prior result:

 1. For each measure at index `i`, call [`set_matrix_risk_ceiling!`](@ref).

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

Adds the set's own row for a ceiling that holds its matrix and reads no prior, at index `i`.

A [`StandardDeviation`](@ref) always takes the second-order cone of [`set_matrix_risk_soc!`](@ref). So does a [`Variance`](@ref), unless [`sdp_variance_flag!`](@ref) selects the semidefinite formulation because `pl` holds a semidefinite phylogeny. The variance then takes one linear row on the model's lifted matrix ``\\mathbf{W}``. The phylogeny's rows use the same ``\\mathbf{W}``, because one lifted matrix belongs to one weight vector.

# Mathematical definition

The semidefinite ceiling on the lifted matrix:

```math
\\begin{align}
\\mathrm{tr}(\\mathbf{\\Sigma} \\mathbf{W}) &\\leq \\bar{r} k\\,, \\\\
\\begin{bmatrix} \\mathbf{W} & \\boldsymbol{w} \\\\ \\boldsymbol{w}^\\intercal & k \\end{bmatrix} &\\succeq 0\\,.
\\end{align}
```

Where:

  - ``\\mathbf{W}``: Lifted symmetric matrix of the semidefinite formulation, ``N \\times N``.
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:rbar_ceil])
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])

For ``k > 0`` the second line gives ``\\mathbf{W} \\succeq \\boldsymbol{w}\\boldsymbol{w}^\\intercal / k``, so every allocation that the first line admits satisfies ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} \\leq \\bar{r} k^2``.

# Algorithm

 1. Ask [`sdp_variance_flag!`](@ref) for the formulation of a variance. A standard deviation skips this step.
 2. Under the semidefinite formulation, get the lifted matrix `W` from [`set_sdp_constraints!`](@ref), which builds it and its cone when the model has none.
 3. Register the row `set_risk_sdp_` at index `i`, and return.
 4. Otherwise, call [`set_matrix_risk_soc!`](@ref).

# JuMP formulation

## Variables

  - `W`: ``\\mathbf{W}``, the lifted matrix, read from the model or created with its cone.
  - `k`: ``k``, read from the model.

## Constraints

  - `set_risk_sdp_i`, under `prefix`: ``s_c \\left(\\mathrm{tr}(\\mathbf{\\Sigma} \\mathbf{W}) - \\bar{r} k\\right) \\leq 0``.

Where:

  - ``\\mathbf{W}``: Lifted symmetric matrix of the semidefinite formulation, ``N \\times N``.
  - ``\\mathbf{A}``: Adjacency matrix of the semidefinite phylogeny.
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:rbar_ceil])
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

The row bounds ``\\mathrm{tr}(\\mathbf{\\Sigma} \\mathbf{W}) / k``, which lies above the variance ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} / k^2``. So the row is conservative. The variance of the answer never exceeds the ceiling, and it can stay below it. The bound is tight when ``\\mathbf{W} = \\boldsymbol{w}\\boldsymbol{w}^\\intercal / k``. The phylogeny's rows ``\\mathbf{A} \\odot \\mathbf{W} = \\mathbf{0}`` can forbid that matrix, and then the ceiling binds below ``\\bar{r}``.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r`: The ceiling, a measure that holds its covariance matrix.
  - $(arg_dict[:pl_opt])
  - `prefix`: The Model State namespace of the row.

# Returns

  - `nothing`.

# Related

  - [`set_allocation_risk_ceiling!`](@ref)
  - [`set_matrix_risk_soc!`](@ref)
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

Adds the second-order cone of [`set_matrix_risk_ceiling!`](@ref) at index `i`.

The factor is the measure's `chol` when it states one, and otherwise [`covariance_factor`](@ref) of its matrix, as in the shared builder. So a singular matrix, such as the covariance of a book with a cash leg, gives an exact factor, and an indefinite matrix throws its `PosDefException`. The shared builder writes the same bound as two rows through an auxiliary variable. This function writes it as one row.

# Mathematical definition

```math
\\begin{align}
\\left\\lVert \\mathbf{G} \\boldsymbol{w} \\right\\rVert_2 &\\leq u k\\,, \\quad \\mathbf{G}^\\intercal \\mathbf{G} = \\mathbf{\\Sigma}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{G}``: Upper factor of ``\\mathbf{\\Sigma}``.
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:u_cone])
  - $(math_dict[:rbar_ceil])
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])

For ``k \\geq 0`` the row is ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} \\leq u^2 k^2``. So the row bounds a variance by ``\\bar{r} k^2``, and a standard deviation by ``\\bar{r} k``.

# Algorithm

 1. Take the factor `G`: `r.chol` when it is set, and `covariance_factor(r.sigma)` otherwise.
 2. Read the norm bound `u` from [`allocation_risk_ceiling`](@ref).
 3. Read `w`, `k` and the constraint scale from the model.
 4. Register the row `set_risk_soc_` at index `i`.

# JuMP formulation

## Variables

  - `w`: ``\\boldsymbol{w}``, read from the model.
  - `k`: ``k``, read from the model.

## Constraints

  - `set_risk_soc_i`, under `prefix`: ``\\left(s_c u k,\\; s_c \\mathbf{G} \\boldsymbol{w}\\right) \\in \\mathcal{K}_{\\mathrm{SOC}}``.

Where:

  - ``\\mathbf{G}``: Upper factor of ``\\mathbf{\\Sigma}``.
  - ``\\mathcal{K}_{\\mathrm{SOC}}``: Second-order cone.
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:u_cone])
  - $(math_dict[:rbar_ceil])
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - $(math_dict[:sc_scale])

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r`: The ceiling, a measure that holds its covariance matrix.
  - `prefix`: The Model State namespace of the row.

# Validation

  - `r.sigma` is not positive semidefinite and `r.chol` is `nothing`. A `PosDefException` is thrown.

# Returns

  - `nothing`.

# Related

  - [`set_matrix_risk_ceiling!`](@ref)
  - [`covariance_factor`](@ref)
  - [`chol_sigma_selector`](@ref)
"""
function set_matrix_risk_soc!(model::JuMP.Model, i::Integer,
                              r::Union{<:Variance, <:StandardDeviation};
                              prefix::Symbol = Symbol(""))::Nothing
    G = isnothing(r.chol) ? covariance_factor(r.sigma) : r.chol
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

Resolves the risk-contribution rows of a [`Variance`](@ref) ceiling on a programme set.

The set resolves `r.rc` over its own `sets`, under the `strict` of the head that runs the step, the same way an optimiser resolves its own rows. When there are rows, the shared builder takes the semidefinite formulation.

# Arguments

  - `r`: The variance ceiling.
  - `set`: The programme Allocation Set.
  - $(arg_dict[:pr])

# Returns

  - `rc::Option{<:LinearConstraint}`: The resolved rows, or `nothing` when `r.rc` is `nothing`.

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

Returns the norm bound of the set's own second-order cone for a ceiling that holds its matrix.

# Mathematical definition

```math
\\begin{align}
u &= \\sqrt{\\bar{r}} \\quad \\text{for a variance,} \\\\
u &= \\bar{r} \\quad \\text{for a standard deviation.}
\\end{align}
```

Where:

  - $(math_dict[:u_cone])
  - $(math_dict[:rbar_ceil])

# Arguments

  - `r`: The ceiling.

# Returns

  - `u::Number`: The norm bound, in the type that `sqrt` gives for a variance and in the type of `settings.ub` for a standard deviation.

# Related

  - [`set_matrix_risk_soc!`](@ref)
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
