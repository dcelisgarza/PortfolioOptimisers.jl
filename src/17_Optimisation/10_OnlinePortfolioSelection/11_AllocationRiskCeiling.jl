"""
    assert_risk_ceiling(r::Nothing)
    assert_risk_ceiling(r::RiskMeasure)
    assert_risk_ceiling(rs::VecRM)

Refuses a risk measure that cannot be the ceiling of a [`ProgrammeAllocationSet`](@ref).

The ceiling of a measure is its `settings.ub`, and a one-step projection can bound the measure by one number only. The function refuses a measure with no `ub`, with a `Frontier` or with a per-asset vector, and it refuses an empty vector of measures. A set with no measure passes.

The ceiling reads every row that the head holds, and those rows change at every step. A [`ReturnsTracking`](@ref) series is fixed at construction, so it cannot follow them, and the function refuses a measure that holds one. A [`WeightsTracking`](@ref) benchmark reads the rows themselves and passes.

# Arguments

  - `r`: The set's risk measure, `nothing`, or a vector of measures.

# Validation

  - `settings.ub` of a measure is not a finite non-negative number. An `ArgumentError` is thrown.
  - No field of a measure holds a [`ReturnsTracking`](@ref). An `ArgumentError` that names the field is thrown otherwise.
  - `rs` is empty. An `IsEmptyError` is thrown.

# Returns

  - `nothing`.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`set_allocation_risk_ceiling!`](@ref)
  - [`set_risk_upper_bound!`](@ref): refuses a frontier or a vector for a set that reaches the builder without this check.
  - [`returns_tracking_path`](@ref)
"""
function assert_risk_ceiling(::Nothing)::Nothing
    return nothing
end
function assert_risk_ceiling(r::RiskMeasure)::Nothing
    ub = r.settings.ub
    @argcheck(isa(ub, Number) && isfinite(ub) && ub >= zero(ub),
              ArgumentError("the ceiling of a `$(nameof(typeof(r)))` on a ProgrammeAllocationSet is its `settings.ub`, a finite non-negative number; got `$ub`"))
    path = returns_tracking_path(r)
    @argcheck(isnothing(path),
              ArgumentError("the ceiling of a `$(nameof(typeof(r)))` on a ProgrammeAllocationSet holds a `ReturnsTracking` at `$(path)`, and the ceiling reads every row the head holds, which change at every step. A fixed series cannot follow them. Track the benchmark that the rows carry with `DataBenchmarkTracking`, or a benchmark portfolio with `WeightsTracking`, which reads the rows themselves, or bound the tracking error with the set's `tr`, which cuts its series to the rows."))
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
    assert_return_floor(ret::Nothing)
    assert_return_floor(ret::JuMPReturnsEstimator)
    assert_return_floor(rets::VecJRE)

Refuses a return term whose floor a [`ProgrammeAllocationSet`](@ref) cannot hold.

The floor of a term is its `settings.lb`. A [`Frontier`](@ref) or a vector of floors asks for one allocation at each floor, and a one-step projection returns one allocation. The model would hold the floors in `:ret_frontier`, and no solve would read them. So the function refuses both forms, as [`assert_risk_ceiling`](@ref) refuses them for a ceiling. A floor that is `nothing` or a number passes.

# Arguments

  - `ret`: The return floor of the set, `nothing`, or a vector of return terms.

# Validation

  - `settings.lb` of a term is `nothing` or a number. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`assert_risk_ceiling`](@ref)
  - [`set_return_bounds!`](@ref)
  - [`assert_frontier_sweep_owner`](@ref): the same refusal for an optimiser that solves its model once.
"""
function assert_return_floor(::Nothing)::Nothing
    return nothing
end
function assert_return_floor(ret::JuMPReturnsEstimator)::Nothing
    lb = ret.settings.lb
    @argcheck(!isa(lb, Front_NumVec),
              ArgumentError("the floor of a `$(nameof(typeof(ret)))` on a ProgrammeAllocationSet is its `settings.lb`, `nothing` or a number; got `$lb`. A frontier or a vector of floors asks for many allocations, and a one-step projection returns one."))
    return nothing
end
function assert_return_floor(rets::VecJRE)::Nothing
    for ret in rets
        assert_return_floor(ret)
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

The set fits its `pe` on the rows that the head holds, the same way a batch head fits a prior on any returns data. The prior reads its own source, whatever the source of the head's step is. When it states `x_src = ExcessSource()` the fit reads `X - B`, which [`benchmark_relative_returns`](@ref) gives. The fit reads the Asset Panel beside the rows, so the result answers `NaN` at an asset outside the prior's Coverage Universe. A set that reads no rows fits nothing, and a bounded set never fits anything.

# Algorithm

 1. Ask [`rows_needed`](@ref) whether the set reads the head's rows. It answers `nothing` when a slot reads them.
 2. If no slot reads them, return `nothing`.
 3. Check that `X` is not `nothing`.
 4. Fit `set.pe` on `X` with [`prior`](@ref), and return the result.

# Arguments

  - `set`: The resolved Allocation Set.
  - `X`: The rows that the head holds, a [`ReturnsResult`](@ref), or `nothing` outside an Online Update.

# Validation

  - A slot reads the rows and `X` is `nothing`. An `ArgumentError` is thrown.
  - The rules of [`benchmark_relative_returns`](@ref), under the source of `set.pe`.

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
                  ArgumentError("a ProgrammeAllocationSet with a slot that reads the head's rows — weight bounds that read the returns, a ceiling built on its prior, a tracking error, a centrality or phylogeny estimator, an exposure row, a return floor or a Calibration Rule — reads them inside an Online Update alone; `project` was called with none."))
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
  - `X`: The rows that the head holds, or `nothing`.

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
    fit_allocation_set_block(b::Tuple{<:ProgrammeAllocationSet, <:ReturnsResult})
    fit_allocation_set_block(b::Tuple{<:ProgrammeAllocationSet, <:ReturnsResult, <:Option{<:AbstractPriorResult}})

Fits the prior of an entry of the keyword `blocks` of [`resolve_jump_optimiser`](@ref), and gives the entry with its prior.

An entry is the block of a [`ProgrammeAllocationSet`](@ref) beside the block of a JuMP head. It is the set after stage 1, [`resolve_allocation_set`](@ref), and after `factory(set, w)`, with the rows `X` that the online head holds. The set reads its risk ceilings `r`, its return floor `ret` and the slots of its block against its own prior on `X`, so the entry carries the rows and no Result of a step lives in an estimator. An entry with a third value carries the prior result of the set already. The entry of a fit again after a refusal has one, and so has an entry for a sub-problem, from [`allocation_set_block_step`](@ref). The function fits nothing for such an entry.

# Arguments

  - `b`: The entry, `(set, X)` or `(set, X, pr)`.

# Returns

  - `(set, X, pr)::Tuple`: The entry with the prior result of the set, [`allocation_set_prior`](@ref) of the set on `X`. `pr` is `nothing` when the set reads no rows.

# Related

  - [`resolve_jump_optimiser`](@ref)
  - [`allocation_set_prior`](@ref)
  - [`allocation_set_entry_mask`](@ref)
  - [`resolve_allocation_set_block`](@ref)
"""
function fit_allocation_set_block(b::Tuple{<:ProgrammeAllocationSet, <:ReturnsResult})
    set, X = b
    return (set, X, allocation_set_prior(set, X))
end
function fit_allocation_set_block(b::Tuple{<:ProgrammeAllocationSet, <:ReturnsResult,
                                           <:Option{<:AbstractPriorResult}})
    return b
end
"""
    allocation_set_priced_mask(imsk::Option{BitVector}, pmsk::Nothing, X::ReturnsResult, strict::Bool)
    allocation_set_priced_mask(imsk::Option{BitVector}, pmsk::BitVector, X::ReturnsResult, strict::Bool)

Intersects the Investable Mask of a JuMP head with the assets that the prior of a block of a set prices.

The prior of a set reads the rows of the online head, and the prior of the head reads its own rows. A young asset that the head keeps can then have no price in the prior of the set, when the `pe` of the set is plain and the prior of the head is mask-aware. A ceiling, a floor or a tracking error of the set would read `NaN` at that asset. So the asset departs from every block, as an asset that a distance reader refuses departs, see [`distance_reader_refusal_mask`](@ref). The function warns once, and it names the assets and the remedies. Under `strict` it refuses them. An asset that the head already left out raises no warning.

**Breaking change.** Before the entry took a list of blocks, the leader re-solve of [`FollowTheLeader`](@ref) refused such a set prior with the `ArgumentError` of [`assert_set_prior_priced`](@ref).

# Algorithm

 1. When `pmsk` is `nothing`, the prior of the set prices every asset. Return `imsk`.
 2. Find the assets that `imsk` keeps and `pmsk` leaves out. A `nothing` mask keeps every asset. When there is none, return the intersection of the two masks.
 3. Refuse the assets under `strict`. Otherwise warn, and name them from `X.nx`, or by their indices when `X` has no names.
 4. Return the intersection of the two masks.

# Arguments

  - `imsk`: The Investable Mask of the head and of the blocks before this one, or `nothing` when every asset is investable.
  - `pmsk`: The Investable Mask of the prior of the set, [`investable_mask`](@ref).
  - `X`: The rows of the set, read for the asset names.
  - `strict`: The `strict` flag of the online head that holds the set, [`projection_step_strict`](@ref).

# Validation

  - Under `strict = true`, the prior of the set prices every asset that `imsk` keeps. Raises an [`IsNonFiniteError`](@ref).
  - At least one asset stays. Raises an [`IsEmptyError`](@ref).

# Returns

  - `imsk::Option{BitVector}`: The intersected mask.

# Related

  - [`allocation_set_entry_mask`](@ref)
  - [`distance_reader_refusal_mask`](@ref)
  - [`investable_mask`](@ref)
"""
function allocation_set_priced_mask(imsk::Option{BitVector}, ::Nothing, ::ReturnsResult,
                                    ::Bool)
    return imsk
end
function allocation_set_priced_mask(imsk::Option{BitVector}, pmsk::BitVector,
                                    X::ReturnsResult, strict::Bool)
    held = something(imsk, trues(length(pmsk)))
    keep = held .& pmsk
    gap = held .& .!pmsk
    if !any(gap)
        return keep
    end
    assets = something(X.nx, eachindex(gap))[gap]
    msg = "the prior of the Allocation Set cannot price $(count(gap)) asset(s) that the programme keeps: $(assets). An asset outside the Coverage Universe of the set prior over the rows that it reads has no mean, no covariance and no loadings for a ceiling, a floor or a tracking error to read"
    remedy = "To keep them, give the set a `pe` whose universe covers the programme's, a mask-aware one under a `CoveragePolicy` or the prior of the held estimator, or shorten the window that the set reads."
    @argcheck(!strict,
              IsNonFiniteError("$(msg), and `strict = true` refuses them. $(remedy)"))
    @warn("$(msg), so the fit leaves these assets out of the Investable Mask and gives each a zero weight. $(remedy) Pass `strict = true` to refuse them.",
          maxlog = log_limit(:warn), _id = @log_id())
    @argcheck(any(keep),
              IsEmptyError("the prior of the Allocation Set prices no asset that the programme keeps, so the fit has no asset left. $(remedy)"))
    return keep
end
"""
    allocation_set_entry_mask(imsk::Option{BitVector}, b::Tuple{<:ProgrammeAllocationSet, <:ReturnsResult, Nothing})
    allocation_set_entry_mask(imsk::Option{BitVector}, b::Tuple{<:ProgrammeAllocationSet, <:ReturnsResult, <:AbstractPriorResult})

Intersects the Investable Mask of a JuMP head with the mask of a block of a set, at the entry of the head.

[`resolve_jump_optimiser`](@ref) folds the function over the entries of its keyword `blocks`, after it composes the mask of its own block with [`distance_reader_entry_mask`](@ref). The mask of the block of a set follows the same rules as the mask of the head, on the prior and the rows of the set. So an asset that one block cannot read departs from every block. A set that reads no rows fits no prior and has no distance reader, so its block keeps the mask.

# Algorithm

 1. When the prior `pr` of the entry is `nothing`, return `imsk`.
 2. Read the `strict` flag of the online head with [`projection_step_strict`](@ref).
 3. Intersect `imsk` with the assets that `pr` prices, with [`allocation_set_priced_mask`](@ref).
 4. Compose the result with the assets that a [`FeatureDistance`](@ref) of the readers `ple` and `cte` of the set can read, [`feature_readable_mask`](@ref), on `X`.
 5. Compose the result with the assets that a plain graph reader of the set can read, [`graph_reader_mask`](@ref), on `pr` and `X`, and return it.

# Arguments

  - `imsk`: The Investable Mask of the head and of the blocks before this one, or `nothing` when every asset is investable.
  - `b`: The fitted entry `(set, X, pr)`, from [`fit_allocation_set_block`](@ref).

# Validation

  - The rules of [`allocation_set_priced_mask`](@ref), [`feature_readable_mask`](@ref) and [`graph_reader_mask`](@ref), under the `strict` flag of the online head.

# Returns

  - `imsk::Option{BitVector}`: The intersected mask.

# Related

  - [`resolve_jump_optimiser`](@ref)
  - [`distance_reader_entry_mask`](@ref)
  - [`allocation_set_priced_mask`](@ref)
  - [`resolve_allocation_set_block`](@ref)
"""
function allocation_set_entry_mask(imsk::Option{BitVector},
                                   ::Tuple{<:ProgrammeAllocationSet, <:ReturnsResult,
                                           Nothing})
    return imsk
end
function allocation_set_entry_mask(imsk::Option{BitVector},
                                   b::Tuple{<:ProgrammeAllocationSet, <:ReturnsResult,
                                            <:AbstractPriorResult})
    set, X, pr = b
    strict = projection_step_strict()
    readers = (set.ple, set.cte)
    pmsk = allocation_set_priced_mask(imsk, investable_mask(pr), X, strict)
    return graph_reader_mask(readers, feature_readable_mask(readers, pmsk, X), pr, X,
                             strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves the block of a set at the Investable Mask of a JuMP head, to its [`ResolvedConstraintBlock`](@ref), or to the refusal of one of its distance readers.

[`resolve_jump_optimiser`](@ref) calls it for each entry of its keyword `blocks`, after the mask holds the masks of every block. The block resolves against the prior of the set on the rows of the set, and not against the prior of the head. A distance reader of the set can still refuse an asset that the prior of the set keeps. The function then returns the [`DistanceReaderRefusalError`](@ref), and the one refusal branch of the entry fits again without the asset.

# Algorithm

 1. Reduce the set, its rows `X` and its prior `pr` to `imsk` with [`programme_investable_reduction`](@ref). The reduction declares the Non-Investable Axis on the `sets` of the set.
 2. Resolve stage 2 of the reduced block with [`resolve_constraint_block`](@ref), through [`distance_reader_call`](@ref). The block reads the solver of the set, and the `strict` flag of the online head, [`projection_step_strict`](@ref).
 3. Return the block Result, or the refusal.

# Arguments

  - `b`: The fitted entry `(set, X, pr)`, from [`fit_allocation_set_block`](@ref).
  - `imsk`: The Investable Mask of the entry, or `nothing` when every asset is investable.

# Validation

  - The rules of [`resolve_constraint_block`](@ref).

# Returns

  - `cbr::Union{ResolvedConstraintBlock, DistanceReaderRefusalError}`: The block Result, or the refusal of a distance reader.

# Related

  - [`resolve_jump_optimiser`](@ref)
  - [`resolve_constraint_block`](@ref)
  - [`distance_reader_call`](@ref)
  - [`allocation_set_entry_mask`](@ref)
"""
function resolve_allocation_set_block(b::Tuple{<:ProgrammeAllocationSet, <:ReturnsResult,
                                               <:Option{<:AbstractPriorResult}},
                                      imsk::Option{BitVector})
    set, X, pr = b
    setr, Xr, prr = programme_investable_reduction(imsk, set, X, pr)
    return distance_reader_call(resolve_constraint_block, setr.cbe, prr, Xr; slv = set.slv,
                                strict = projection_step_strict())
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gives an entry of the keyword `blocks` for a sub-problem on the reduced universe, from the block Result that the entry resolved to.

[`constraint_block_steps`](@ref) calls it for each block after the first of a bundle. A sub-problem of [`NearOptimalCentering`](@ref) solves on the reduced prior and rows of the head. So its entry holds the set and the rows viewed at the Investable Mask, the reduced prior of the block, and the resolved slots of the block. The entry then fits no prior again, its mask keeps every asset, and its readers hold their results, so no reader runs again. The risk ceilings `r` and the return floor `ret` of the set are viewed at the mask with the rest of the set.

# Algorithm

 1. View the set and its rows `X` at `imsk` with [`programme_investable_reduction`](@ref), with no prior. The block Result already holds the reduced prior.
 2. Put the resolved slots of `cbr` into the block of the viewed set with [`constraint_block_from_resolved`](@ref).
 3. Return the set, the viewed rows and the prior of `cbr`.

# Arguments

  - `b`: The entry that the bundle resolved, `(set, X)` or `(set, X, pr)`.
  - `cbr`: The block Result of the entry.
  - `imsk`: The Investable Mask of the bundle, or `nothing` when every asset is investable.

# Returns

  - `(set, X, pr)::Tuple`: The entry for the sub-problem.

# Related

  - [`constraint_block_steps`](@ref)
  - [`fit_allocation_set_block`](@ref)
  - [`jump_optimiser_from_attributes`](@ref)
"""
function allocation_set_block_step(b::Tuple{<:ProgrammeAllocationSet, <:ReturnsResult,
                                            Vararg}, cbr::ResolvedConstraintBlock,
                                   imsk::Option{BitVector})
    setr, Xr, _ = programme_investable_reduction(imsk, b[1], b[2], nothing)
    return (rebuild_estimator(setr,
                              (; cbe = constraint_block_from_resolved(setr.cbe, cbr))), Xr,
            cbr.pr)
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
    clip_at_zero(w::AbstractVector, lb = zero(eltype(w)))

Sets every entry of a projection programme's answer at or below the tolerance of its element type to its floor `lb`, which is zero unless the set states a positive floor.

A geometry whose domain is the non-negative orthant needs this. The solver meets a zero lower bound only to within its tolerance, so a leg that the programme closes comes back slightly above or slightly below zero. The next step forms its raw step multiplicatively from the allocation it holds. It would take `log w` of a negative number, and an exact zero of the raw step keeps the weight at zero. So the clip reads every entry within the tolerance as zero, whatever the sign of the rounding, and a weight that the set forces to zero is zero after the clip. The tolerance is [`weights_bound_tolerance`](@ref): ``\\sqrt{\\epsilon}`` for a floating-point type, which is above the default feasibility tolerance of Clarabel, and zero for a `Rational`. The clip does not renormalise, so the sum moves by the clipped mass, which is at most ``N \\tau``. An entry whose floor is positive goes to that floor and not to zero, so the clip keeps a floor at or below ``\\tau`` that the set states.

# Mathematical definition

```math
\\begin{align}
w'_i &= \\begin{cases} l_i & w_i \\leq \\tau\\,, \\\\ w_i & w_i > \\tau\\,. \\end{cases}
\\end{align}
```

Where:

  - ``w_i``: Entry ``i`` of the programme's answer.
  - ``w'_i``: Entry ``i`` of the clipped answer.
  - ``l_i``: Entry ``i`` of the floor `lb`, non-negative.
  - $(math_dict[:tau_wb_tol])
  - $(math_dict[:N])

# Arguments

  - `w`: The programme's answer.
  - `lb`: The floor of the set, a number or a vector, non-negative. A caller's `project` checks the sign.

# Returns

  - `w'::Vector`: `w` with every entry at or below ``\\tau`` set to its floor, in the element type of `w` when `lb` has that type.

# Related

  - [`project`](@ref)
  - [`projection_programme`](@ref)
  - [`weights_bound_tolerance`](@ref)
"""
function clip_at_zero(w::AbstractVector, lb = zero(eltype(w)))
    tol = weights_bound_tolerance(eltype(w))
    return ifelse.(w .<= tol, zero.(w) .+ lb, w)
end
