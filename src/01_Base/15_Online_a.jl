"""
$(DocStringExtensions.TYPEDEF)

Carries the observations an estimator keeps when its estimate has no exact incremental fold.

This is the buffer of the [`partial_fit!`](@ref) interface. It is a state like any other. It lives in the `cache` field of the estimator, it copies, it slices by asset, and [`obs_weights_view`](@ref) drops it. It holds the observations verbatim, `NaN` included. So the call with no data over the buffer gives the answer of a batch fit over the same rows, and the two have the same Coverage Universe by construction.

The buffer also holds the per-observation masks verbatim. A [`CoveragePolicy`](@ref) reads two facts from an active mask that the rows alone do not carry. A cell that is finite but inactive is excluded, and an asset that is active at one observation and inactive at the next is a delisting, not a holiday. A buffer that kept the rows and dropped the mask would give a different answer from the batch fit, with no warning. `A` is the active mask, and `M` is the estimation mask that the two regime-adjusted families read. Each is a backing matrix of the shape of `X`, or `nothing`. So a wrapped estimator folded under a policy gives the batch fit over the same window, as it does without a policy.

The buffer holds the factor observations on the same terms. A prior whose batch verb is `prior(pe, X, F)` regresses the asset returns on the factor returns, so its refit reads two matrices whose `t`-th rows are contemporaneous. `F` is the second matrix, a backing matrix of `capacity × factors`, or `nothing`. The same `off` and `n` index `F` and `X`, so the rows stay contemporaneous by construction. [`needs_factor_returns`](@ref) answers whether a fit reads `F`. That is a fact of the estimator tree, and the buffer records what the fold gives it. There is one buffer type, and a cap drops the factor rows with the other rows.

The buffer holds the Panel Fields of a time-varying [`AssetPanel`](@ref) on the same terms, for a prior whose estimator tree reads them. [`reads_panel_fields`](@ref) answers whether a tree reads them. Such a prior computes its factors from the Panel Fields at each row, so for it a Panel Field is sample and not fold context. `P` is the vector of the Panel Fields over the valid region alone. The same `n` and the same cap index `P`, and `off` does not. An append joins the rows with `vcat` of two Asset Panels, so it copies the rows that the buffer holds and costs `O(n)`. A refit reads every row at each step, so this cost is smaller than the cost of its call with no data. A panel is a Result and nothing writes into its arrays, so the buffer holds the Panel Fields by reference, and a copy of the buffer shares them. A time-varying panel carries both masks, so a buffer that records `P` records `A` and `M` too.

The buffer holds the Exogenous Series on the same terms, for a prior whose estimator tree reads it. [`reads_exogenous_series`](@ref) answers whether a tree reads it. An observed factor reads its return from the series by name, so for such a prior the series is sample and not fold context. `E` is a backing matrix of `capacity × series`, and `ne` holds the name of each column. The buffer keeps every column, not only the columns that the tree names, so the Fold Context keeps no copy. The same `off` and `n` index `E` and `X`. So `E` means the Exogenous Series on the whole online step, as it does on a [`ReturnsResult`](@ref).

The buffer holds the implied volatilities on the same terms as the masks, for every estimator. A row of `iv` belongs to the asset observation of the same row, so `iv` is a backing matrix of the shape of `X`, and [`ImpliedVolatility`](@ref) reads it beside the returns. Its regression reads windows that end at the last observation, so a new row moves every window, and no statistic of the regression folds. The call with no data refits it over the rows of the buffer. `ivpa` is one adjustment for each asset or one for all, so it has no observation axis, and the first append pins it.

The first append fixes whether the buffer records a mask, whether it records factor rows, whether it records the Exogenous Series, whether it records implied volatilities and whether it records Panel Fields, as it fixes the width and the element type. It also fixes the names of the series and the adjustment `ivpa`. A later fold that disagrees is refused with an error that names the disagreement, in both directions. A buffer that holds no observations records nothing about them, so the next fold seeds it again from its block.

The buffer marks the observations of a Shared Series gap in `G`, and keeps them. A Shared Series is a vector benchmark of a group that reads the Excess Frame, or the factor returns of a prior that reads them, and a batch fit drops an observation with a gap in one of them, see [`shared_series_view`](@ref). An online step cannot drop the row, because the buffer of a prior and the Fold Context of its optimiser hold one row each for every observation, and the call with no data rebuilds the returns data from both. So the step keeps the row and marks it, and every fit through the buffer reads the rows of [`shared_series_buffer`](@ref), which drops the marked rows. A cap counts every row, as the training window of a batch walk-forward counts the observations before the drop. A fold that marks no gap leaves `G` at `nothing`, so a buffer that never meets a gap records nothing.

An estimator whose `cache` holds this buffer is a refit. It has no recursion of its own, and its call with no data runs the batch verb over the rows of the buffer. [`Online`](@ref) seeds the buffer, and a wrapped estimator takes this route also when its statistic has an exact update. A fold-and-carry prior keeps its observations in one of these buffers too, inside a [`PriorCarryState`](@ref). That prior folds its moments exactly. It keeps the rows because [`LowOrderPrior`](@ref) carries `X` for the scenario risk measures, and its buffer has no cap. `max_history` is the window of a refit. A capped buffer holds the last `max_history` observations, and the call with no data gives the batch fit over them.

The backing matrix has spare capacity, so an append costs amortised `O(1)`. `off` is the number of rows before the valid region, and `n` is the length of the region. [`reserve_sample_buffer`](@ref) moves the region to the front of the backing matrix only when an append would go past its end. [`sample_buffer`](@ref) reads the valid region.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SampleBufferState(;
        n::Integer = 0,
        off::Integer = 0,
        X::MatNum = Matrix{Float64}(undef, 0, 0),
        A::Option{<:AbstractMatrix{<:Bool}} = nothing,
        M::Option{<:AbstractMatrix{<:Bool}} = nothing,
        F::Option{<:MatNum} = nothing,
        ne::Option{<:VecStr} = nothing,
        E::Option{<:MatNum} = nothing,
        iv::Option{<:MatNum} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing,
        P::Option{<:AbstractVector} = nothing,
        G::Option{<:AbstractVector{Bool}} = nothing,
        max_history::Option{<:Integer} = nothing
    ) -> SampleBufferState

Keywords correspond to the struct's fields. The default is the empty seed that [`Online`](@ref) builds. It carries the cap and no observations. The first append replaces every backing matrix, so the width, the element type, the masks, the factor rows, the Exogenous Series, the implied volatilities and the Panel Fields come from the observations of that append.

## Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `off >= 0`. A `DomainError` is thrown otherwise.
  - `off + n <= size(X, 1)`. A `DimensionMismatch` is thrown otherwise.
  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.
  - `n <= max_history` when `max_history` is not `nothing`. A `DimensionMismatch` is thrown otherwise.
  - `A`, `M` and `iv`, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.
  - `F`, when it is not `nothing`, has as many rows as `X`. A `DimensionMismatch` is thrown otherwise.
  - The rules of [`assert_buffer_series_shape`](@ref) hold for `ne` and `E`.
  - `P`, when it is not `nothing`, holds time-varying Panel Fields over `n` observations and the assets of `X`. The rules of [`assert_buffer_panel_shape`](@ref) apply.
  - `G`, when it is not `nothing`, has one entry for each row of `X`. A `DimensionMismatch` is thrown otherwise.

## View parameters

When [`port_opt_view`](@ref) is called on this type, its fields are subset to the selected assets:

  - `X`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `A`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `M`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `F`: Copied unchanged, because the selection indexes assets and this backing describes factors.
  - `ne`, `E`: Copied unchanged, because the selection indexes assets and the Exogenous Series belongs to no asset.
  - `iv`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `ivpa`: Sliced to the selected indices when it is a vector, and kept when it is one number or `nothing`.
  - `P`: Each Panel Field viewed over the selected assets with [`panel_fields_view`](@ref).
  - `G`: Copied unchanged, because the selection indexes assets and a Shared Series gap belongs to the observation.

A buffer that holds no observations views to the empty seed with the same cap.

# Examples

```jldoctest
julia> PortfolioOptimisers.SampleBufferState(; max_history = 2)
PortfolioOptimisers.SampleBufferState
            n ┼ Int64: 0
          off ┼ Int64: 0
            X ┼ 0×0 Matrix{Float64}
            A ┼ nothing
            M ┼ nothing
            F ┼ nothing
           ne ┼ nothing
            E ┼ nothing
           iv ┼ nothing
         ivpa ┼ nothing
            P ┼ nothing
            G ┼ nothing
  max_history ┴ Int64: 2
```

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`Online`](@ref)
  - [`PriorCarryState`](@ref)
  - [`sample_buffer`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`factor_buffer`](@ref)
  - [`fold_buffer`](@ref)
  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`needs_factor_returns`](@ref)
  - [`reads_panel_fields`](@ref)
  - [`reads_exogenous_series`](@ref)
  - [`sample_buffer_panel`](@ref)
  - [`exogenous_buffer_kwargs`](@ref)
  - [`implied_volatility_buffer_kwargs`](@ref)
  - [`shared_series_buffer`](@ref)
"""
@concrete struct SampleBufferState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    Number of rows of `X` before the valid region. It rises when a capped buffer drops its oldest observation, and it returns to zero when the valid region moves to the front of `X`.
    """
    off
    """
    Backing matrix of the buffer, `capacity × assets`. Rows `off + 1` to `off + n` are the observations, in the order of the folds. The other rows are spare capacity, and their values have no meaning. The first append fixes the width and the element type.
    """
    X
    """
    $(field_dict[:pf_buffer_A])
    """
    A
    """
    $(field_dict[:pf_buffer_M])
    """
    M
    """
    Backing matrix of the factor observations, `capacity × factors`, or `nothing` when the fold carries none. The same `off` and `n` index it and `X`. The first append that carries it fixes its width and its element type.
    """
    F
    """
    Names of the columns of `E`, or `nothing` when the fold carries no Exogenous Series. The first append that carries the series fixes them.
    """
    ne
    """
    Backing matrix of the Exogenous Series, `capacity × series`, or `nothing` when the fold carries none. The same `off` and `n` index it and `X`. It holds every column of the series that the fold receives.
    """
    E
    """
    Backing matrix of the implied volatilities, `capacity × assets`, or `nothing` when the fold carries none. The same `off` and `n` index it and `X`.
    """
    iv
    """
    The implied volatility risk premium adjustment, one number or one entry per asset, or `nothing`. The first append pins it.
    """
    ivpa
    """
    The Panel Fields of the observations, or `nothing` when the fold carries none. Each is a time-varying Panel Field over the valid region alone, so `n` and the cap index it and `off` does not. The first append that carries them fixes their names and their kinds.
    """
    P
    """
    Backing vector of the Shared Series gaps, `capacity` long, or `nothing` when no fold marked a gap. Entry `off + t` is `true` when observation `t` has a gap in a Shared Series that the fit reads, so every fit through the buffer drops it, see [`shared_series_buffer`](@ref). The same `off` and `n` index it and `X`. The field has a declared type and is not a type parameter, so the type of the buffer, and of every estimator that holds it, does not depend on whether a step met a gap. A step with no gap then compiles to the code it had before the marks existed.
    """
    G::Union{Nothing, Vector{Bool}}
    """
    $(field_dict[:pf_max_history])
    """
    max_history
end
function SampleBufferState(; n::Integer = 0, off::Integer = 0,
                           X::MatNum = Matrix{Float64}(undef, 0, 0),
                           A::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           M::Option{<:AbstractMatrix{<:Bool}} = nothing,
                           F::Option{<:MatNum} = nothing, ne::Option{<:VecStr} = nothing,
                           E::Option{<:MatNum} = nothing, iv::Option{<:MatNum} = nothing,
                           ivpa::Option{<:Num_VecNum} = nothing,
                           P::Option{<:AbstractVector} = nothing,
                           G::Option{<:AbstractVector{Bool}} = nothing,
                           max_history::Option{<:Integer} = nothing)::SampleBufferState
    assert_sample_buffer_state(n, off, X, max_history)
    assert_buffer_mask_shape(X, A, M, iv)
    assert_buffer_factor_shape(X, F)
    assert_buffer_series_shape(X, ne, E)
    assert_buffer_panel_shape(X, n, P)
    assert_buffer_gap_shape(X, G)
    return SampleBufferState(n, off, X, A, M, F, ne, E, iv, ivpa, P, G, max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a sample buffer whose valid region does not sit inside its backing matrix.

# Algorithm

 1. Refuse a negative `n` and a negative `off`.
 2. Refuse a valid region that runs past the last row of `X`.
 3. Return when `max_history` is `nothing`.
 4. Refuse a non-positive `max_history`, and refuse a valid region longer than it.

# Arguments

  - $(arg_dict[:pf_n])
  - `off`: Number of rows of `X` before the valid region.
  - `X`: Backing matrix of the buffer.
  - $(arg_dict[:pf_max_history])

# Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `off >= 0`. A `DomainError` is thrown otherwise.
  - `off + n <= size(X, 1)`. A `DimensionMismatch` is thrown otherwise.
  - `max_history > 0` when it is not `nothing`. A `DomainError` is thrown otherwise.
  - `n <= max_history` when `max_history` is not `nothing`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
"""
function assert_sample_buffer_state(n::Integer, off::Integer, X::MatNum,
                                    max_history::Option{<:Integer})
    assert_nonneg(n, :n)
    assert_nonneg(off, :off)
    @argcheck(off + n <= size(X, 1),
              DimensionMismatch("the valid region must sit inside the backing matrix, but `off` is $off, `n` is $n and `X` has $(size(X, 1)) rows."))
    if !isnothing(max_history)
        @argcheck(max_history > zero(max_history),
                  DomainError(max_history, "max_history must be positive"))
        @argcheck(n <= max_history,
                  DimensionMismatch("a capped buffer cannot hold more observations than its cap, but `n` is $n and `max_history` is $max_history."))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a per-observation mask, or a matrix of implied volatilities, whose shape is not the shape of the observations it belongs to.

A mask that a buffer holds matches its observations cell for cell, and a mask that a fold receives matches its block cell for cell. So one rule covers both. A mask is absent, or it has the shape of the matrix beside it. The implied volatilities follow the same rule, because each cell belongs to the asset observation of the same cell. The batch verbs run the same check on their own arguments.

# Algorithm

 1. Skip a mask that is `nothing`. With no policy, both masks are `nothing`.
 2. Refuse a mask whose size is not `size(X)`, and name the mask. Apply the same rule to `iv`.

# Arguments

  - `X`: The observations the masks belong to.
  - `A`: The active mask, or `nothing`.
  - `M`: The estimation mask, or `nothing`.
  - `iv`: The implied volatilities, or `nothing`.

# Validation

  - `A`, `M` and `iv`, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function assert_buffer_mask_shape(X::AbstractMatrix, A::Option{<:AbstractMatrix{<:Bool}},
                                  M::Option{<:AbstractMatrix{<:Bool}},
                                  iv::Option{<:AbstractMatrix} = nothing)
    if !isnothing(iv)
        @argcheck(size(iv) == size(X),
                  DimensionMismatch("size(X) ($(size(X))) must match size(iv) ($(size(iv)))"))
    end
    if !isnothing(A)
        @argcheck(size(A) == size(X),
                  DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(A)))"))
    end
    if !isnothing(M)
        @argcheck(size(M) == size(X),
                  DimensionMismatch("size(X) ($(size(X))) must match size(estimation_mask) ($(size(M)))"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a factor block whose rows are not the rows of the observations it belongs to.

A factor observation is contemporaneous with the asset observation of the same row. So a factor block that a fold receives has one row for each row of the block beside it, and the factor backing of a buffer has one row for each row of `X`. The check does not compare the two widths, because an asset count and a factor count are unrelated.

# Algorithm

 1. Skip a factor block that is `nothing`. With no factors, the block is `nothing`.
 2. Refuse a factor block whose row count is not `size(X, 1)`.

# Arguments

  - `X`: The observations the factor rows belong to.
  - `F`: The factor rows, or `nothing`.

# Validation

  - `F`, when it is not `nothing`, has as many rows as `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`needs_factor_returns`](@ref)
"""
function assert_buffer_factor_shape(X::AbstractMatrix, F::Option{<:AbstractMatrix})
    if !isnothing(F)
        @argcheck(size(F, 1) == size(X, 1),
                  DimensionMismatch("size(X, 1) ($(size(X, 1))) must match size(F, 1) ($(size(F, 1))): a factor observation is contemporaneous with the asset observation of the same row."))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses marks of the Shared Series gaps that are not one mark for each row of the observations.

A mark belongs to the observation of the same row, so the marks that a fold receives have one entry for each row of its block, and the backing vector of a buffer has one entry for each row of `X`.

# Algorithm

 1. Skip marks that are `nothing`.
 2. Refuse marks whose length is not `size(X, 1)`.

# Arguments

  - `X`: The observations that the marks belong to.
  - `G`: The marks, `true` at an observation with a Shared Series gap, or `nothing`.

# Validation

  - `G`, when it is not `nothing`, has one entry for each row of `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`shared_series_buffer`](@ref)
"""
function assert_buffer_gap_shape(X::AbstractMatrix, G::Option{<:AbstractVector{Bool}})
    if !isnothing(G)
        @argcheck(length(G) == size(X, 1),
                  DimensionMismatch("size(X, 1) ($(size(X, 1))) must match length(G) ($(length(G))): a Shared Series gap marks the observation of the same row."))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an Exogenous Series whose rows are not the rows of the observations it belongs to, or whose names do not name its columns.

A row of the series is contemporaneous with the asset observation of the same row, as a factor row is. The names travel with the series, so the two are both absent or both present, and there is one name for each column.

# Algorithm

 1. Refuse names without a series, and a series without names.
 2. Skip a series that is `nothing`.
 3. Refuse a series whose row count is not `size(X, 1)`, and names whose count is not the column count of the series.

# Arguments

  - `X`: The observations the series belongs to.
  - `ne`: The names of the series, or `nothing`.
  - `E`: The series, `observations × series`, or `nothing`.

# Validation

  - `ne` and `E` are both `nothing`, or neither is. An `ArgumentError` is thrown otherwise.
  - `E`, when it is not `nothing`, has as many rows as `X`, and `length(ne) == size(E, 2)`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`reads_exogenous_series`](@ref)
"""
function assert_buffer_series_shape(X::AbstractMatrix, ne::Option{<:VecStr},
                                    E::Option{<:AbstractMatrix})
    @argcheck(isnothing(ne) == isnothing(E),
              ArgumentError("the names of the Exogenous Series travel with the series, so `ne` and `E` are both `nothing` or neither is, but `ne` is $(isnothing(ne) ? "" : "not ")`nothing` and `E` is $(isnothing(E) ? "" : "not ")`nothing`."))
    if !isnothing(E)
        @argcheck(size(E, 1) == size(X, 1),
                  DimensionMismatch("size(X, 1) ($(size(X, 1))) must match size(E, 1) ($(size(E, 1))): a row of the Exogenous Series is contemporaneous with the asset observation of the same row."))
        @argcheck(length(ne) == size(E, 2),
                  DimensionMismatch("`ne` names each column of the Exogenous Series, but it holds $(length(ne)) name(s) and `E` has $(size(E, 2)) column(s)."))
    end
    return nothing
end
"""
    buffer_rows_view(M::Nothing, rows) -> Nothing
    buffer_rows_view(M::AbstractMatrix, rows) -> SubArray
    buffer_rows_view(G::AbstractVector, rows) -> SubArray

Reads a row range out of a per-observation backing, and passes a backing that is not there through.

The buffer slices a mask or the factor rows on the observation axis only through this function, so the `nothing` case has one method. [`sample_buffer_kwargs`](@ref), [`factor_buffer`](@ref) and [`merge_states`](@ref) read the valid region of a stored backing with it, and [`partial_fit!`](@ref) truncates an incoming block with it.

# Arguments

  - `M`: The backing to slice, or `nothing`. A vector is the backing of the marks of the Shared Series gaps, one entry per row.
  - `rows`: The rows to keep.

# Returns

  - `M`: The rows of the backing, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`factor_buffer`](@ref)
  - [`partial_fit!`](@ref)
"""
function buffer_rows_view(::Nothing, args...)
    return nothing
end
function buffer_rows_view(M::AbstractMatrix, rows)
    return view(M, rows, :)
end
function buffer_rows_view(G::AbstractVector, rows)
    return view(G, rows)
end
"""
    sample_buffer(state::SampleBufferState)
    sample_buffer(est)

Reads the observations a sample buffer holds, `observations × assets`.

This function reads the rows out of a [`SampleBufferState`](@ref). It is a view of the valid region of the backing matrix, in the order of the folds, so a refit reads it with no copy. The estimator form reads the state from the `cache` field, and it refuses an estimator that carries no buffer.

# Arguments

  - `state`: The buffer to read.
  - `est`: Estimator whose `cache` field carries the buffer.

# Validation

  - `est.cache` holds a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `X::SubArray`: The observations the buffer holds, `observations × assets`.

# Related

  - [`SampleBufferState`](@ref)
  - [`assert_sample_buffer`](@ref)
  - [`Online`](@ref)
"""
function sample_buffer(state::SampleBufferState)
    return view(state.X, (state.off + 1):(state.off + state.n), :)
end
function sample_buffer(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    return sample_buffer(assert_sample_buffer(est))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the per-observation masks a sample buffer holds, as the keywords a batch verb takes.

[`sample_buffer`](@ref) gives the rows, and this function gives the masks of those rows. The call with no data runs the batch verb on the rows and splats these keywords into the call, so the wrapped answer is the unwrapped answer over the same window. A buffer that records no mask gives an empty set of keywords, and the arm then makes the same call as with no policy. The branch reads fields of concrete type, so the compiler resolves it.

# Algorithm

 1. Slice each stored mask to the valid region with [`buffer_rows_view`](@ref).
 2. Name the masks that are there, and omit the ones that are not.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `kwargs::NamedTuple`: The `active_mask` and `estimation_mask` keywords the buffer holds, of which either or both may be absent.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer`](@ref)
  - [`buffer_rows_view`](@ref)
  - [`CoveragePolicy`](@ref)
"""
function sample_buffer_kwargs(state::SampleBufferState)
    rows = (state.off + 1):(state.off + state.n)
    A = buffer_rows_view(state.A, rows)
    M = buffer_rows_view(state.M, rows)
    return if isnothing(A)
        isnothing(M) ? (;) : (; estimation_mask = M)
    elseif isnothing(M)
        (; active_mask = A)
    else
        (; active_mask = A, estimation_mask = M)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the factor observations a sample buffer holds, `observations × factors`, or `nothing` when it holds none.

This is the third function that reads the buffer out, with [`sample_buffer`](@ref) and [`sample_buffer_kwargs`](@ref). A refit whose batch verb reads a factor matrix takes it from here, as the second positional argument of that verb. Its `t`-th row is contemporaneous with the `t`-th row of [`sample_buffer`](@ref), because the same `off` and `n` index the two backings. A buffer that records no factor rows gives `nothing`, which is the value that the batch verb of a prior with no factors receives. The branch reads a field of concrete type, so the compiler resolves it.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `F`: The factor observations the buffer holds, as a view of the valid region, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`buffer_rows_view`](@ref)
  - [`needs_factor_returns`](@ref)
"""
function factor_buffer(state::SampleBufferState)
    return buffer_rows_view(state.F, (state.off + 1):(state.off + state.n))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the Exogenous Series a sample buffer holds, as the keywords a batch verb takes.

This is the fourth function that reads the buffer out. The call with no data of a refit splats these keywords into the batch verb, as `prior(pe, rd)` passes `rd.ne` and `rd.E`. The Fold Context reads them too, when the prior owns the series. Each row of `E` is contemporaneous with the row of [`sample_buffer`](@ref) of the same index, because the same `off` and `n` index the two backings. A buffer that records no series gives an empty set of keywords. The branch reads a field of concrete type, so the compiler resolves it.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `kwargs::NamedTuple`: `ne` and `E`, the names and a view of the valid region of the series, or an empty `NamedTuple`.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer`](@ref)
  - [`factor_buffer`](@ref)
  - [`reads_exogenous_series`](@ref)
"""
function exogenous_buffer_kwargs(state::SampleBufferState)
    return if isnothing(state.E)
        (;)
    else
        (; ne = state.ne,
         E = buffer_rows_view(state.E, (state.off + 1):(state.off + state.n)))
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the implied volatilities a sample buffer holds, and the pinned adjustment, as the keywords a batch verb takes.

This is the fifth function that reads the buffer out. The call with no data of a refit splats these keywords into the batch verb, as `prior(pe, rd)` passes `rd.iv` and `rd.ivpa`, and the Fold Context reads them back into the `ReturnsResult` that it rebuilds. Each row of `iv` belongs to the row of [`sample_buffer`](@ref) of the same index, because the same `off` and `n` index the two backings. A buffer that records neither gives an empty set of keywords.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `kwargs::NamedTuple`: `iv`, a view of the valid region of the implied volatilities or `nothing`, and `ivpa`, or an empty `NamedTuple`.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer`](@ref)
  - [`exogenous_buffer_kwargs`](@ref)
  - [`ImpliedVolatility`](@ref)
"""
function implied_volatility_buffer_kwargs(state::SampleBufferState)
    return if isnothing(state.iv) && isnothing(state.ivpa)
        (;)
    else
        (; iv = buffer_rows_view(state.iv, (state.off + 1):(state.off + state.n)),
         ivpa = state.ivpa)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the marks of the Shared Series gaps of the valid region of a [`SampleBufferState`](@ref).

A buffer with no marks has no gap, so it gives `false` for each of its observations. That answer allocates a vector, so a reader on the path of every step tests `isnothing(state.G)` first, as [`buffer_fit_count`](@ref), [`buffer_fits_none`](@ref) and [`shared_series_buffer`](@ref) do. [`merge_buffer_gaps`](@ref) reads it only when one of the two buffers has marks, and the merge copies every row anyway.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `marks::AbstractVector{Bool}`: One mark for each observation of the buffer, `true` at an observation with a Shared Series gap.

# Related

  - [`SampleBufferState`](@ref)
  - [`shared_series_buffer`](@ref)
  - [`merge_buffer_gaps`](@ref)
"""
function buffer_gap_marks(state::SampleBufferState)
    rows = (state.off + 1):(state.off + state.n)
    return isnothing(state.G) ? fill(false, state.n) : view(state.G, rows)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Count the observations of a [`SampleBufferState`](@ref) that a fit keeps, the rows that [`shared_series_buffer`](@ref) returns.

A reader that holds a history of the fitted observations beside the buffer, as the carry of a [`CrossSectionalFactorPrior`](@ref) does, indexes that history by this count and not by `n`, because the history never holds an observation with a Shared Series gap.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `n::Int`: The number of observations of the buffer with no Shared Series gap.

# Related

  - [`shared_series_buffer`](@ref)
  - [`buffer_gap_marks`](@ref)
"""
function buffer_fit_count(state::SampleBufferState)
    return isnothing(state.G) ? state.n : state.n - count(buffer_gap_marks(state))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the observations of a [`SampleBufferState`](@ref) that a fit keeps, or `nothing` when it keeps every observation.

A fit through the buffer drops each observation with a Shared Series gap, see [`shared_series_buffer`](@ref), so it counts the position of a row among the kept observations. Kept row `k` is observation `rows[k]` of the buffer. A reader that records the position of a fitted row maps it through `rows`, as [`shared_series_row_key`](@ref) does. A buffer with no gap answers `nothing` and allocates nothing.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `rows::Option{Vector{Int}}`: The observations of the buffer with no Shared Series gap, or `nothing` when no observation has a gap.

# Related

  - [`buffer_fit_count`](@ref)
  - [`buffer_gap_marks`](@ref)
  - [`shared_series_row_key`](@ref)
"""
function buffer_kept_rows(state::SampleBufferState)::Option{Vector{Int}}
    if isnothing(state.G)
        return nothing
    end
    marks = buffer_gap_marks(state)
    return any(marks) ? findall(!, marks) : nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Answer `true` when every observation of a [`SampleBufferState`](@ref) has a Shared Series gap, so a fit through the buffer keeps no observation.

A reader that fits on the buffer after each step, as the Choice Rule of a refit does, asks this first, because [`shared_series_buffer`](@ref) refuses such a buffer and the first steps of a run can hold only gaps. A buffer with no marks answers `false` at once and allocates nothing, so the step of a buffer that never meets a gap costs what it cost before the marks existed. A buffer that holds no observation and no mark answers `false` too, as it did before.

# Arguments

  - `state`: The buffer to read.

# Returns

  - `ans::Bool`: Whether the buffer has marks and keeps no observation.

# Related

  - [`buffer_fit_count`](@ref)
  - [`shared_series_buffer`](@ref)
"""
function buffer_fits_none(state::SampleBufferState)::Bool
    return !isnothing(state.G) && iszero(buffer_fit_count(state))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the buffer that a fit reads, with the observations of a Shared Series gap dropped.

This is the buffer form of [`shared_series_view`](@ref). An online step keeps an observation with a Shared Series gap in the buffer and marks it in `G`, because the buffer of a prior and the Fold Context of its optimiser hold one row each for every observation, see [`SampleBufferState`](@ref). A fit drops the observation, as the batch fit over the same rows drops it at its entry. So every reader that fits on the rows of the buffer of a prior reads this buffer, and the call with no data gives the batch fit over the same observations. A reader that rebuilds the returns data, [`returns_result`](@ref), reads the buffer itself, because the batch fit that it hands the data to drops the same observations again.

The rows are copied. The masks, the factor rows, the Exogenous Series, the implied volatilities and the Panel Fields keep the same rows. The result has no marks, so the function is idempotent.

# Algorithm

 1. Return `state` when it has no marks, which allocates nothing, or when no observation of its valid region is marked.
 2. Find the observations that are not marked, and refuse a buffer that has none.
 3. Copy the rows of the observations into a buffer of their own, with the same cap.

# Arguments

  - `state`: The buffer to read.

# Validation

  - At least one observation of the buffer has no Shared Series gap. An [`IsEmptyError`](@ref) is thrown otherwise.

# Returns

  - `state::SampleBufferState`: `state` itself when it has no gap, and otherwise a buffer of its observations with no gap.

# Related

  - [`SampleBufferState`](@ref)
  - [`shared_series_view`](@ref)
  - [`buffer_gap_marks`](@ref)
"""
function shared_series_buffer(state::SampleBufferState)::SampleBufferState
    if isnothing(state.G)
        return state
    end
    marks = buffer_gap_marks(state)
    if !any(marks)
        return state
    end
    rows = findall(!, marks)
    @argcheck(!isempty(rows),
              IsEmptyError("every observation that the online step folded has a gap in a Shared Series, the vector benchmark under `x_src = ExcessSource()` or the factor returns, so the fit keeps no observation. Fold an observation whose Shared Series is finite."))
    r = state.off .+ rows
    return SampleBufferState(; n = length(rows), X = state.X[r, :],
                             A = trim_merged_array(state.A, r),
                             M = trim_merged_array(state.M, r),
                             F = trim_merged_array(state.F, r), ne = state.ne,
                             E = trim_merged_array(state.E, r),
                             iv = trim_merged_array(state.iv, r), ivpa = state.ivpa,
                             P = panel_fields_view(state.P, rows, :),
                             max_history = state.max_history)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the sample buffer an estimator carries, and refuses an estimator that carries none.

The buffering arm of [`partial_fit!`](@ref) reads its state through this function. So an estimator that was never wrapped in [`Online`](@ref) gets a message that names the wrapper, not a `MethodError`. An estimator with no `cache` field gets the same message. An unresolved [`Online`](@ref) has a method of its own, because it does not lack a buffer. It lacks the warm-up that seeds one.

# Arguments

  - `est`: Estimator whose `cache` field carries the buffer.

# Validation

  - `est` has a `cache` field holding a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer the estimator carries.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`partial_fit_cache`](@ref)
"""
function assert_sample_buffer(est::Union{<:AbstractEstimator,
                                         <:StatsBase.CovarianceEstimator})
    cache = hasfield(typeof(est), :cache) ? getfield(est, :cache) : nothing
    @argcheck(isa(cache, SampleBufferState),
              ArgumentError("`$(typeof(est))` carries no sample buffer, so it has no incremental fold of its own to fall back on. Wrap it in `Online` and resolve the wrapper with `update_online_estimator` before folding, which seeds the buffer its refit reads."))
    return cache::SampleBufferState
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the [`SampleBufferState`](@ref) a carry folds into, seeding an empty one when the estimator carries none.

The seed has its own function, so the fold is one line and the branch on the `cache` field has one method. A family that carries its observations calls [`fold_buffer`](@ref), which is this seed followed by the fold. The seed has no cap, because a cap is configuration, and [`Online`](@ref) carries it.

# Arguments

  - `cache`: The buffer the estimator carries, or `nothing`.

# Returns

  - `state::SampleBufferState`: The buffer `cache` holds, or an empty uncapped buffer.

# Related

  - [`SampleBufferState`](@ref)
  - [`fold_buffer`](@ref)
  - [`Online`](@ref)
"""
function sample_buffer_seed(cache::Option{<:SampleBufferState})
    return isnothing(cache) ? SampleBufferState() : cache
end
"""
    fold_buffer(cache::Option{<:SampleBufferState}, x::VecNum, f::Option{<:VecNum} = nothing; kwargs...)
    fold_buffer(cache::Option{<:SampleBufferState}, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)

Folds observations into a sample buffer, seeding an empty one when the estimator carries none.

A carry calls this function inside its own [`partial_fit!`](@ref) method. The carry folds its estimate exactly through its members, and it gives its observations to this function. The function is [`sample_buffer_seed`](@ref) followed by [`partial_fit!`](@ref). The carry gives the masks and the factor rows of the fold with the observations, and the buffer records them beside their rows.

# Arguments

  - `cache`: The buffer the estimator carries, or `nothing`.
  - `x`: One observation, whose entries are the assets.
  - `f`: The contemporaneous factor observation, whose entries are the factors, or `nothing`.
  - $(arg_dict[:X])
  - `F`: Factor observations to fold, oriented as `X` is, or `nothing`.
  - $(arg_dict[:dims])
  - `kwargs...`: The `active_mask`, the `estimation_mask`, the Exogenous Series `ne` and `E`, and the implied volatilities `iv` and `ivpa` of the fold, forwarded to [`partial_fit!`](@ref).

# Returns

  - `state::SampleBufferState`: The buffer after the last observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`sample_buffer_seed`](@ref)
  - [`partial_fit!`](@ref)
"""
function fold_buffer(cache::Option{<:SampleBufferState}, x::VecNum,
                     f::Option{<:VecNum} = nothing; kwargs...)
    return partial_fit!(sample_buffer_seed(cache), x, f; kwargs...)
end
function fold_buffer(cache::Option{<:SampleBufferState}, X::MatNum,
                     F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    return partial_fit!(sample_buffer_seed(cache), X, F; dims = dims, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds every observation of a block into a [`SampleBufferState`](@ref).

This is the block arm of the [`partial_fit!`](@ref) interface, and all of the fold of the buffer. The single-observation arm reshapes its argument and calls this arm. A cap truncates the incoming rows before the copy, so the fold never copies the whole of a block longer than the cap. [`reserve_sample_buffer`](@ref) grows the backing matrices, and its docstring states why an append costs amortised `O(1)`.

The masks, the factor rows, the Exogenous Series, the implied volatilities and the Panel Fields stay with their rows through each step: the orientation, the truncation, the growth and the copy. The first append fixes whether the buffer records a mask, whether it records factor rows, whether it records the Exogenous Series, whether it records implied volatilities and whether it records Panel Fields, as it fixes the width and the element type. It also fixes the names of the series and the adjustment `ivpa`. [`assert_buffer_presence_agreement`](@ref) refuses a later fold that disagrees. A buffer that knows the activity of some of its rows and not of others answers neither question. A buffer with factor rows for some observations and not for others describes no regression. A buffer that holds no observations has nothing to disagree with, so the fold seeds it again from the block.

The Panel Fields are observations first whatever `dims` is, as the Panel Fields of an [`AssetPanel`](@ref) always are. They keep the rows of the buffer that the cap keeps, and [`panel_fields_append`](@ref) joins them to the rows of the block.

The marks `gaps` of the Shared Series gaps are one entry for each observation whatever `dims` is. They travel with their rows as the masks do, but they have no presence rule: a buffer with no marks has no gap, so [`append_buffer_gaps`](@ref) writes the marks of a later fold onto `false` marks for the rows before it, and a fold with no marks gives `false` to a buffer that has them.

# Algorithm

 1. Orient `X`, the masks, `iv`, `F` and `E` to `observations × assets`, `observations × factors` and `observations × series`. Transpose them when `dims == 2`.
 2. Refuse a mask or an `iv` that does not have the shape of the block, a factor block whose row count is not the row count of the block, a series that [`assert_buffer_series_shape`](@ref) refuses, and marks that [`assert_buffer_gap_shape`](@ref) refuses.
 3. Return the state unchanged when the block is empty.
 4. Refuse Panel Fields that do not make a time-varying Asset Panel with the masks of the block, over the observations and the assets of the block.
 5. When a cap is set, truncate the incoming rows, their masks, their implied volatilities, their factor rows, their series, their Panel Fields and their marks to the last `max_history` of them.
 6. When the buffer holds no observations, seed it from the block with [`seed_sample_buffer`](@ref), seed `iv` with [`seed_buffer_array`](@ref), seed `G` with [`seed_buffer_gaps`](@ref), pin `ivpa`, and return it. The seed fixes the width, the element type, the masks, the implied volatilities with `ivpa`, the factor rows, the series with its names, and the Panel Fields.
 7. Refuse a block whose width is not the fixed width. Refuse masks, implied volatilities, factor rows, a series or Panel Fields that are absent where the buffer records them or present where it does not, factor rows of a different width, a series under other names with [`assert_buffer_series_names`](@ref), and an `ivpa` that is not the pinned one with [`assert_pinned_value`](@ref).
 8. Make room for the block with [`reserve_sample_buffer`](@ref), which drops the rows that the cap pushes out and grows the backing matrices.
 9. Copy the block, its masks, its implied volatilities, its factor rows and its series after the valid region, and write its marks with [`append_buffer_gaps`](@ref). Join the Panel Fields that the buffer keeps to the Panel Fields of the block with [`panel_fields_append`](@ref), and rebind `P` and `n` with `Accessors.@reset`.

# Arguments

  - `state`: The buffer to fold into.
  - $(arg_dict[:X])
  - `F`: Factor observations to fold, oriented as `X` is, or `nothing`.
  - $(arg_dict[:dims])
  - $(arg_dict[:pf_active_mask])
  - $(arg_dict[:pf_estimation_mask])
  - `ne`: The names of the Exogenous Series of the block, or `nothing`.
  - `E`: The Exogenous Series of the block, oriented as `X` is, or `nothing`.
  - `iv`: The implied volatilities of the block, oriented as `X` is, or `nothing`.
  - `ivpa`: The implied volatility risk premium adjustment, or `nothing`.
  - `panel_fields`: The time-varying Panel Fields of the block, observations first, or `nothing`.
  - `gaps`: One mark for each observation of the block, `true` at an observation with a Shared Series gap, or `nothing` when the block has none.

# Validation

  - $(val_dict[:dims])
  - `gaps`, when it is not `nothing`, has one entry for each observation of the block. A `DimensionMismatch` is thrown otherwise.
  - `X` has the width the first append fixed. A `DimensionMismatch` is thrown otherwise.
  - `active_mask`, `estimation_mask` and `iv`, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.
  - `ivpa` equals the adjustment that the first append pinned. An `ArgumentError` is thrown otherwise.
  - `F`, when it is not `nothing`, has as many rows as `X`, and the width that the first append with factor rows fixed. A `DimensionMismatch` is thrown otherwise.
  - `ne` and `E` obey the rules of [`assert_buffer_series_shape`](@ref), and `ne` equals the names that the first append with a series fixed. An `ArgumentError` is thrown otherwise.
  - `panel_fields`, when it is not `nothing`, obeys the rules of [`assert_buffer_panel_block`](@ref), and it holds the Panel Fields that the first append with Panel Fields fixed, as `vcat` of two Asset Panels states them.
  - A buffer that holds observations receives the masks that it records, and it receives implied volatilities, factor rows, a series and Panel Fields if and only if it records them. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer after the last row.

# Related

  - [`SampleBufferState`](@ref)
  - [`reserve_sample_buffer`](@ref)
  - [`seed_sample_buffer`](@ref)
  - [`assert_buffer_presence_agreement`](@ref)
  - [`partial_fit!`](@ref)
  - [`sample_buffer`](@ref)
  - [`factor_buffer`](@ref)
"""
function partial_fit!(state::SampleBufferState, X::MatNum, F::Option{<:MatNum} = nothing;
                      dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                      estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                      ne::Option{<:VecStr} = nothing, E::Option{<:MatNum} = nothing,
                      iv::Option{<:MatNum} = nothing, ivpa::Option{<:Num_VecNum} = nothing,
                      panel_fields::Option{<:AbstractVector} = nothing,
                      gaps::Option{<:AbstractVector{Bool}} = nothing)
    Xo, Ao, Mo, Fo, Eo, Vo = dims_oriented(dims, X, active_mask, estimation_mask, F, E, iv)
    assert_buffer_mask_shape(Xo, Ao, Mo, Vo)
    assert_buffer_factor_shape(Xo, Fo)
    assert_buffer_series_shape(Xo, ne, Eo)
    assert_buffer_gap_shape(Xo, gaps)
    Go = gaps
    t = size(Xo, 1)
    if iszero(t)
        return state
    end
    assert_buffer_panel_block(Xo, Ao, Mo, panel_fields)
    Po = panel_fields
    w = state.max_history
    if !isnothing(w) && t > w
        rows = (t - w + 1):t
        Xo = view(Xo, rows, :)
        Ao = buffer_rows_view(Ao, rows)
        Mo = buffer_rows_view(Mo, rows)
        Fo = buffer_rows_view(Fo, rows)
        Eo = buffer_rows_view(Eo, rows)
        Vo = buffer_rows_view(Vo, rows)
        Po = panel_fields_view(Po, rows, :)
        Go = buffer_rows_view(Go, rows)
        t = w
    end
    if iszero(state.n)
        state = seed_sample_buffer(state, Xo, Ao, Mo, Fo, ne, Eo, Po)
        state = Accessors.@reset state.iv = seed_buffer_array(Vo)
        state = Accessors.@reset state.G = seed_buffer_gaps(Go)
        return Accessors.@reset state.ivpa = ivpa
    end
    N = size(Xo, 2)
    B = state.X
    @argcheck(size(B, 2) == N,
              DimensionMismatch("the width of a sample buffer is fixed by its first append, but the buffer describes $(size(B, 2)) assets and the block has $N columns."))
    assert_buffer_presence_agreement(state.A, Ao, "active_mask")
    assert_buffer_presence_agreement(state.M, Mo, "estimation_mask")
    assert_buffer_presence_agreement(state.F, Fo, "factor returns")
    assert_buffer_presence_agreement(state.E, Eo, "Exogenous Series")
    assert_buffer_presence_agreement(state.iv, Vo, "implied volatilities")
    assert_buffer_presence_agreement(state.P, Po, "Panel Fields")
    assert_buffer_factor_width(state.F, Fo)
    assert_buffer_series_names(state.ne, ne)
    assert_pinned_value(state.ivpa, ivpa, :ivpa)
    state = reserve_sample_buffer(state, t)
    rows = (state.off + state.n + 1):(state.off + state.n + t)
    copyto!(view(state.X, rows, :), Xo)
    copy_buffer_rows!(state.A, rows, Ao)
    copy_buffer_rows!(state.M, rows, Mo)
    copy_buffer_rows!(state.F, rows, Fo)
    copy_buffer_rows!(state.E, rows, Eo)
    copy_buffer_rows!(state.iv, rows, Vo)
    state = append_buffer_gaps(state, rows, Go)
    state = Accessors.@reset state.P = panel_fields_append(state.P, Po, state.n + t)
    return Accessors.@reset state.n = state.n + t
end
"""
    assert_buffer_factor_width(B::Nothing, Fo) -> Nothing
    assert_buffer_factor_width(B::AbstractMatrix, Fo::AbstractMatrix) -> Nothing

Refuses a factor block whose width is not the width the first factor append fixed.

This is the width check of `X`, for the factor backing, which can be absent. A buffer that records no factor rows has no width to compare with. [`assert_buffer_presence_agreement`](@ref) runs first and refuses a block that gives factor rows to such a buffer, so the `nothing` method runs only when `Fo` is absent too.

# Arguments

  - `B`: The factor backing the buffer holds, or `nothing`.
  - `Fo`: The oriented factor block, or `nothing`.

# Validation

  - `size(B, 2) == size(Fo, 2)` when the buffer records factor rows. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`assert_buffer_presence_agreement`](@ref)
"""
function assert_buffer_factor_width(::Nothing, ::Any)
    return nothing
end
function assert_buffer_factor_width(B::AbstractMatrix, Fo::AbstractMatrix)
    @argcheck(size(B, 2) == size(Fo, 2),
              DimensionMismatch("the width of a sample buffer's factor rows is fixed by the first append that carries them, but the buffer describes $(size(B, 2)) factors and the block has $(size(Fo, 2)) columns."))
    return nothing
end
"""
    assert_buffer_series_names(ne::Nothing, neo) -> Nothing
    assert_buffer_series_names(ne::VecStr, neo::VecStr) -> Nothing

Refuses an Exogenous Series whose names are not the names the first append fixed.

A consumer reads a column of the series by its name, so a buffer whose columns change their names between two rows gives a column with two meanings. The first append that carries the series fixes the names, as [`ReturnsBufferState`](@ref) pins them. [`assert_buffer_presence_agreement`](@ref) runs first, so the `nothing` method runs only when the block carries no series either.

# Arguments

  - `ne`: The names the buffer records, or `nothing`.
  - `neo`: The names of the incoming block, or `nothing`.

# Validation

  - `neo == ne` when the buffer records a series. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`merge_states`](@ref)
"""
function assert_buffer_series_names(::Nothing, ::Any)
    return nothing
end
function assert_buffer_series_names(ne::VecStr, neo::VecStr)
    @argcheck(ne == neo,
              ArgumentError("the first append fixes the names of the Exogenous Series of a sample buffer, and a consumer reads a column by its name, so a later block must carry the same names. The buffer records $(ne) and the block carries $(neo). Fold the whole run with one set of names, or seed a fresh buffer."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a fold whose masks, factor rows, Exogenous Series or Panel Fields disagree with what the buffer records.

This is the mixture rule of the buffer. It refuses the fold, because neither way to resolve the mixture gives a correct answer. A buffer that dropped an incoming mask would fit a held gap that the policy excludes, and it would read each delisting as a holiday. The mask exists to prevent that wrong answer. A buffer that made up a mask for its earlier rows would state an activity that it never saw. The same rule holds for the factor rows. A buffer with factor rows for some observations and not for others describes no regression, and the buffer cannot make up the missing rows. A prior that reads the Panel Fields computes its factors from them at every row, so a buffer that holds them for some rows and not for others gives it no complete panel. The same holds for the Exogenous Series that an observed factor reads at every fitted row. So the first append fixes what a buffer records, and a caller that wants a change starts a new buffer.

# Arguments

  - `M`: The backing the buffer records, or `nothing`.
  - `Mo`: The same backing of the incoming block, or `nothing`.
  - `name`: Name of the backing, for the message.

# Validation

  - `M` and `Mo` are both `nothing`, or neither is. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`seed_sample_buffer`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`needs_factor_returns`](@ref)
"""
function assert_buffer_presence_agreement(M::Option{<:AbstractArray},
                                          Mo::Option{<:AbstractArray}, name::AbstractString)
    @argcheck(isnothing(M) == isnothing(Mo),
              ArgumentError("a sample buffer records `$name` for every observation it holds or for none of them, and this fold $(isnothing(Mo) ? "gives none where the buffer records one" : "gives one where the buffer records none"). A buffer that records it for some of its rows and not for others answers neither the question it asks nor the question it does not. Fold the whole run with `$name` or fold it without, or seed a fresh buffer."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fills a buffer that holds no observations from a block, which fixes the width, the element type, the masks, the factor rows, the Exogenous Series and the Panel Fields.

[`partial_fit!`](@ref) calls it for the first append, and for each append to a buffer that holds no observations. Each backing matrix is an exact fit for the block, so a buffer that is folded once and read once allocates no spare row. Each array comes from `similar` on the argument that it holds, so the function chooses no element type. The Panel Fields are not copied, because nothing writes into the arrays of a panel.

# Algorithm

 1. Replace `X`, `A`, `M`, `F` and `E` with copies of the block, its masks, its factor rows and its series, made by [`seed_buffer_array`](@ref).
 2. Replace `ne` with the names of the series, and `P` with the Panel Fields of the block.
 3. Set `off` to zero and `n` to the row count of the block.

# Arguments

  - `state`: The buffer to fill. It holds no observations.
  - `Xo`: The oriented block.
  - `Ao`: The oriented active mask of the block, or `nothing`.
  - `Mo`: The oriented estimation mask of the block, or `nothing`.
  - `Fo`: The oriented factor rows of the block, or `nothing`.
  - `ne`: The names of the series of the block, or `nothing`.
  - `Eo`: The oriented Exogenous Series of the block, or `nothing`.
  - `Po`: The Panel Fields of the block, or `nothing`.

# Returns

  - `state::SampleBufferState`: The buffer holding the block, its masks, its factor rows, its series and its Panel Fields.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`seed_buffer_array`](@ref)
"""
function seed_sample_buffer(state::SampleBufferState, Xo::MatNum,
                            Ao::Option{<:AbstractMatrix{<:Bool}},
                            Mo::Option{<:AbstractMatrix{<:Bool}}, Fo::Option{<:MatNum},
                            ne::Option{<:VecStr}, Eo::Option{<:MatNum},
                            Po::Option{<:AbstractVector})
    state = Accessors.@reset state.X = seed_buffer_array(Xo)
    state = Accessors.@reset state.A = seed_buffer_array(Ao)
    state = Accessors.@reset state.M = seed_buffer_array(Mo)
    state = Accessors.@reset state.F = seed_buffer_array(Fo)
    state = Accessors.@reset state.ne = ne
    state = Accessors.@reset state.E = seed_buffer_array(Eo)
    state = Accessors.@reset state.P = Po
    state = Accessors.@reset state.off = 0
    return Accessors.@reset state.n = size(Xo, 1)
end
"""
    seed_buffer_array(Mo::Nothing) -> Nothing
    seed_buffer_array(Mo::AbstractMatrix) -> AbstractMatrix

Copies an incoming block into a backing matrix of its own, and passes a backing that is not there through.

The block of a fold is an array of the caller, and the buffer lives after the call. A later fold writes into the backing matrix, so the buffer takes a copy, as [`Base.copy`](@ref) and [`port_opt_view`](@ref) do. Without the copy, the fold would change the array of the caller. `similar` reads the element type from the block, so an observation, a mask and a factor row each keep their own element type.

# Arguments

  - `Mo`: The oriented block, mask or factor rows, or `nothing`.

# Returns

  - `M`: A backing matrix holding the block, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`seed_sample_buffer`](@ref)
"""
function seed_buffer_array(::Nothing)
    return nothing
end
function seed_buffer_array(Mo::AbstractMatrix)
    return copyto!(similar(Mo, size(Mo)...), Mo)
end
"""
    copy_buffer_rows!(M::Nothing, rows, Mo) -> Nothing
    copy_buffer_rows!(M::AbstractMatrix, rows, Mo::AbstractMatrix) -> AbstractMatrix

Copies an incoming backing into the rows of the buffer's backing matrix that were just made room for.

This is the optional part of the append. The two masks and the factor rows each take one line of the fold, and the `nothing` case has one method. [`assert_buffer_presence_agreement`](@ref) runs first, so the two arguments are both `nothing` or both matrices.

# Arguments

  - `M`: The buffer's backing matrix for this mask or the factor rows, or `nothing`.
  - `rows`: The rows of `M` to write.
  - `Mo`: The oriented mask or factor rows of the block, or `nothing`.

# Returns

  - `M`: The rows written, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
  - [`assert_buffer_presence_agreement`](@ref)
"""
function copy_buffer_rows!(::Nothing, args...)
    return nothing
end
function copy_buffer_rows!(M::AbstractMatrix, rows, Mo::AbstractMatrix)
    return copyto!(view(M, rows, :), Mo)
end
"""
    seed_buffer_gaps(Go::Nothing) -> Nothing
    seed_buffer_gaps(Go::AbstractVector{Bool}) -> Option{Vector{Bool}}

Seeds the marks of the Shared Series gaps of a [`SampleBufferState`](@ref) from the first block that it holds.

A block with no gap seeds no marks, so a buffer that never meets a gap keeps `G` at `nothing`. A block with a gap seeds a copy of its marks, because a later fold writes into the backing vector.

# Arguments

  - `Go`: The marks of the block, `true` at an observation with a Shared Series gap, or `nothing`.

# Returns

  - `G::Option{Vector{Bool}}`: The backing vector of the marks, or `nothing` when the block has no gap.

# Related

  - [`SampleBufferState`](@ref)
  - [`append_buffer_gaps`](@ref)
"""
function seed_buffer_gaps(::Nothing)
    return nothing
end
function seed_buffer_gaps(Go::AbstractVector{Bool})
    return any(Go) ? collect(Go) : nothing
end
"""
    append_buffer_gaps(state::SampleBufferState, rows, Go::Nothing) -> SampleBufferState
    append_buffer_gaps(state::SampleBufferState, rows, Go::AbstractVector{Bool}) -> SampleBufferState

Writes the marks of the Shared Series gaps of a block into the rows of a [`SampleBufferState`](@ref) that were just made room for.

The marks have no presence rule, because a buffer with no marks has no gap. A buffer with no marks and a block with no gap keep no marks. The first block with a gap gives the buffer a backing vector of `false` marks for the rows before it, and a block with no marks writes `false` into a buffer that has them.

# Arguments

  - `state`: The buffer, after [`reserve_sample_buffer`](@ref) made room for the block.
  - `rows`: The rows of the backing to write.
  - `Go`: The marks of the block, or `nothing`.

# Returns

  - `state::SampleBufferState`: The buffer with the marks of the block written.

# Related

  - [`SampleBufferState`](@ref)
  - [`seed_buffer_gaps`](@ref)
  - [`partial_fit!`](@ref)
"""
function append_buffer_gaps(state::SampleBufferState, rows, ::Nothing)
    if !isnothing(state.G)
        fill!(view(state.G, rows), false)
    end
    return state
end
function append_buffer_gaps(state::SampleBufferState, rows, Go::AbstractVector{Bool})
    if isnothing(state.G) && !any(Go)
        return state
    end
    G = isnothing(state.G) ? fill(false, size(state.X, 1)) : state.G
    copyto!(view(G, rows), Go)
    return Accessors.@reset state.G = G
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes room for `t` further observations in a [`SampleBufferState`](@ref), and returns the buffer that has it.

This is the growth part of the fold of the buffer. It writes no observation. It drops the rows that the cap pushes out, moves the valid region to the front of the backing matrices, and allocates larger matrices when the front does not have enough room.

These steps make an append cost amortised `O(1)`. A drop moves `off` and copies no row, so a capped buffer pays nothing for the observation that it drops. The masks, the factor rows and the series of that observation drop with it, because the same `off` and `n` index them. A compaction in place happens only when the valid region and the block fill at most half of the backing matrix. So it copies at most half of the matrix, and at least half of the matrix is free after the append. A growth at least doubles the capacity until the clamp at twice the cap stops it, and it copies fewer rows than the old capacity. So all the growths of a buffer copy fewer rows than its last capacity. A block longer than twice the capacity grows the matrix to fit the valid region and the block, and the next growth doubles it. A capped buffer never holds more than twice its cap, so a cap of `w` costs at most `2w` rows.

# Algorithm

 1. Drop the oldest observations the cap pushes out, by moving `off` forward and `n` back.
 2. Return when the valid region and the block already fit inside the backing matrix.
 3. Take a capacity of twice the current one, or of the room needed if that is larger, when the valid region and the block would fill more than half of the current one. Clamp it to twice the cap when a cap is set.
 4. Compact the observations, each mask, the factor rows, the series and the marks of the Shared Series gaps that the buffer records to that capacity with [`compact_buffer_array`](@ref), which allocates only when the capacity changes, and set `off` to zero.

# Arguments

  - `state`: The buffer to make room in.
  - `t`: Number of observations about to be appended.

# Returns

  - `state::SampleBufferState`: The buffer whose backing matrices have room for `t` rows after their valid region.

# Related

  - [`SampleBufferState`](@ref)
  - [`compact_buffer_array`](@ref)
  - [`partial_fit!`](@ref)
"""
function reserve_sample_buffer(state::SampleBufferState, t::Integer)
    n = state.n
    off = state.off
    w = state.max_history
    if !isnothing(w) && n + t > w
        drop = n + t - w
        off += drop
        n -= drop
        state = Accessors.@reset state.n = n
        state = Accessors.@reset state.off = off
    end
    B = state.X
    cap = size(B, 1)
    if off + n + t <= cap
        return state
    end
    newcap = cap
    if 2 * (n + t) > cap
        newcap = max(n + t, 2 * cap)
        if !isnothing(w)
            newcap = min(newcap, 2 * w)
        end
    end
    state = Accessors.@reset state.X = compact_buffer_array(B, off, n, newcap)
    state = Accessors.@reset state.A = compact_buffer_array(state.A, off, n, newcap)
    state = Accessors.@reset state.M = compact_buffer_array(state.M, off, n, newcap)
    state = Accessors.@reset state.F = compact_buffer_array(state.F, off, n, newcap)
    state = Accessors.@reset state.E = compact_buffer_array(state.E, off, n, newcap)
    state = Accessors.@reset state.iv = compact_buffer_array(state.iv, off, n, newcap)
    state = Accessors.@reset state.G = compact_buffer_array(state.G, off, n, newcap)
    return Accessors.@reset state.off = 0
end
"""
    compact_buffer_array(B::Nothing, off, n, newcap) -> Nothing
    compact_buffer_array(B::AbstractMatrix, off::Integer, n::Integer, newcap::Integer) -> AbstractMatrix
    compact_buffer_array(G::AbstractVector, off::Integer, n::Integer, newcap::Integer) -> AbstractVector

Moves the valid region of one backing matrix to the front, allocating a matrix of `newcap` rows when the capacity changes. The backing vector of the marks of the Shared Series gaps has one entry per row, and compacts in the same way.

This is the array part of [`reserve_sample_buffer`](@ref). The buffer carries five backing matrices with one index, and this function compacts each of them. A copy in place with an increasing row index is safe, because the destination row `i` is never after the source row `off + i`. So the loop reads each source row before a write can reach it.

# Algorithm

 1. Take `D` as `B` when `B` has `newcap` rows, and as a new matrix of `newcap` rows otherwise.
 2. Copy row `off + i` of `B` to row `i` of `D`, for `i` from 1 to `n`, one column at a time.

# Arguments

  - `B`: The backing matrix to compact, or `nothing`.
  - `off`: Number of rows before the valid region.
  - `n`: Length of the valid region.
  - `newcap`: Number of rows the compacted matrix has.

# Returns

  - `D`: The compacted matrix, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`reserve_sample_buffer`](@ref)
"""
function compact_buffer_array(::Nothing, args...)
    return nothing
end
function compact_buffer_array(B::AbstractMatrix, off::Integer, n::Integer, newcap::Integer)
    D = size(B, 1) == newcap ? B : similar(B, newcap, size(B, 2))
    for j in axes(B, 2), i in 1:n
        D[i, j] = B[off + i, j]
    end
    return D
end
function compact_buffer_array(G::AbstractVector, off::Integer, n::Integer, newcap::Integer)
    D = length(G) == newcap ? G : similar(G, newcap)
    for i in 1:n
        D[i] = G[off + i]
    end
    return D
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds one observation into a [`SampleBufferState`](@ref).

This is the single-observation arm of the [`partial_fit!`](@ref) interface. It reshapes the observation into a one-row block with no copy, and it folds the block through the block arm. It reshapes the masks, the factor observation, the observation of the Exogenous Series and the implied volatilities in the same way, so one observation with a policy, with factors, with a series or with implied volatilities folds through the code of the block.

# Arguments

  - `state`: The buffer to fold into.
  - `x`: One observation, whose entries are the assets.
  - `f`: The contemporaneous factor observation, whose entries are the factors, or `nothing`.
  - `active_mask`: The active mask of the observation, one entry per asset, or `nothing`.
  - `estimation_mask`: The estimation mask of the observation, one entry per asset, or `nothing`.
  - `ne`: The names of the Exogenous Series, or `nothing`.
  - `E`: The observation of the Exogenous Series, one entry per series, or `nothing`.
  - `iv`: The implied volatilities of the observation, one entry per asset, or `nothing`.
  - `ivpa`: The implied volatility risk premium adjustment, or `nothing`.

# Validation

  - `x` has the width the first append fixed. A `DimensionMismatch` is thrown otherwise.
  - `active_mask` and `estimation_mask`, when they are not `nothing`, have one entry per asset. A `DimensionMismatch` is thrown otherwise.
  - `f`, when it is not `nothing`, has the width that the first append with factor rows fixed. A `DimensionMismatch` is thrown otherwise.
  - A buffer that holds observations receives `f` if and only if it records factor rows. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer after the observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(state::SampleBufferState, x::VecNum, f::Option{<:VecNum} = nothing;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      ne::Option{<:VecStr} = nothing, E::Option{<:VecNum} = nothing,
                      iv::Option{<:VecNum} = nothing, ivpa::Option{<:Num_VecNum} = nothing)
    return partial_fit!(state, observation_row(x), observation_row(f);
                        active_mask = observation_row(active_mask),
                        estimation_mask = observation_row(estimation_mask), ne = ne,
                        E = observation_row(E), iv = observation_row(iv), ivpa = ivpa)
end
"""
    observation_row(m::Nothing) -> Nothing
    observation_row(m::AbstractVector) -> AbstractMatrix

Reshapes one observation, its mask or its factor observation into a one-row block, and passes one that is not there through.

This is the vector part of the single-observation arm of [`partial_fit!`](@ref). `reshape` makes no copy, so one observation with a policy or with factors allocates no more than one observation without them.

# Arguments

  - `m`: One observation, one mask or one factor observation, or `nothing`.

# Returns

  - `M`: The vector as a one-row block, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit!`](@ref)
"""
function observation_row(::Nothing)
    return nothing
end
function observation_row(m::AbstractVector)
    return reshape(m, 1, length(m))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`SampleBufferState`](@ref) fitted on disjoint blocks into the buffer of the concatenated block.

The merge is a concatenation, and it is exact. A buffer holds its observations verbatim, so the buffer of two blocks holds the rows of the first block followed by the rows of the second. When a cap is set, the result keeps the last `max_history` rows of the concatenation. The masks, the factor rows, the Exogenous Series and the Panel Fields concatenate with their rows. Two buffers that disagree about what they record, or that name their series differently, are refused, for the reason that [`assert_buffer_presence_agreement`](@ref) states for a fold. A buffer that holds no observations records nothing, so the merge returns a copy of the other buffer.

The method does not call [`assert_mergeable_states`](@ref). That check compares every array field on every axis. The backing matrix of a buffer has the observation axis as well as the asset axis, so the check refuses two buffers over the same assets with different numbers of observations. This method checks the cap, the width, the masks and the factor rows itself.

# Algorithm

 1. Refuse two buffers of different caps.
 2. Return a copy of the other buffer when one buffer holds no observations.
 3. Refuse two buffers of different widths.
 4. Refuse two buffers that do not record the same masks, that disagree about factor rows, a series or Panel Fields, whose factor rows have different widths, or whose series have different names.
 5. Concatenate the valid region of the first with the valid region of the second, and each mask, the factor rows and the series with their own. Concatenate the marks of the Shared Series gaps with [`merge_buffer_gaps`](@ref).
 6. Keep the last `max_history` rows when a cap is set.
 7. Join the Panel Fields of the two buffers over the rows that the result keeps, with [`panel_fields_append`](@ref).

# Arguments

  - `a`: The buffer of the first block of observations.
  - `b`: The buffer of the second block of observations.

# Validation

  - `a` and `b` carry the same cap. An `ArgumentError` is thrown otherwise.
  - `a` and `b` describe the same number of assets, when both hold observations. A `DimensionMismatch` is thrown otherwise.
  - `a` and `b` record the same masks, both record factor rows or neither does, both record a series under the same names or neither does, and both record Panel Fields or neither does, when both hold observations. An `ArgumentError` is thrown otherwise.
  - `a` and `b` describe the same number of factors, when they record factor rows. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `state::SampleBufferState`: The buffer the two blocks give when they are folded as one block.

# Related

  - [`SampleBufferState`](@ref)
  - [`merge_states`](@ref)
  - [`assert_buffer_presence_agreement`](@ref)
  - [`sample_buffer`](@ref)
"""
function merge_states(a::SampleBufferState, b::SampleBufferState)
    w = a.max_history
    @argcheck(w == b.max_history,
              ArgumentError("two sample buffers of different caps cannot be merged, but `a` has a `max_history` of $(w) and `b` has $(b.max_history)."))
    if iszero(a.n)
        return copy(b)
    elseif iszero(b.n)
        return copy(a)
    end
    Xa = sample_buffer(a)
    Xb = sample_buffer(b)
    @argcheck(size(Xa, 2) == size(Xb, 2),
              DimensionMismatch("two sample buffers over different numbers of assets cannot be merged, but `a` describes $(size(Xa, 2)) assets and `b` describes $(size(Xb, 2))."))
    assert_buffer_presence_agreement(a.A, b.A, "active_mask")
    assert_buffer_presence_agreement(a.M, b.M, "estimation_mask")
    assert_buffer_presence_agreement(a.F, b.F, "factor returns")
    assert_buffer_presence_agreement(a.E, b.E, "Exogenous Series")
    assert_buffer_presence_agreement(a.iv, b.iv, "implied volatilities")
    assert_buffer_presence_agreement(a.P, b.P, "Panel Fields")
    assert_buffer_factor_width(a.F, b.F)
    assert_buffer_series_names(a.ne, b.ne)
    assert_pinned_value(a.ivpa, b.ivpa, :ivpa)
    ra = (a.off + 1):(a.off + a.n)
    rb = (b.off + 1):(b.off + b.n)
    X = vcat(Xa, Xb)
    A = merge_buffer_array(buffer_rows_view(a.A, ra), buffer_rows_view(b.A, rb))
    M = merge_buffer_array(buffer_rows_view(a.M, ra), buffer_rows_view(b.M, rb))
    F = merge_buffer_array(buffer_rows_view(a.F, ra), buffer_rows_view(b.F, rb))
    E = merge_buffer_array(buffer_rows_view(a.E, ra), buffer_rows_view(b.E, rb))
    V = merge_buffer_array(buffer_rows_view(a.iv, ra), buffer_rows_view(b.iv, rb))
    G = merge_buffer_gaps(a, b)
    t = size(X, 1)
    if !isnothing(w) && t > w
        rows = (t - w + 1):t
        X = X[rows, :]
        A = trim_merged_array(A, rows)
        M = trim_merged_array(M, rows)
        F = trim_merged_array(F, rows)
        E = trim_merged_array(E, rows)
        V = trim_merged_array(V, rows)
        G = trim_merged_array(G, rows)
        t = w
    end
    return SampleBufferState(t, 0, X, A, M, F, a.ne, E, V, a.ivpa,
                             panel_fields_append(a.P, b.P, t), G, w)
end
"""
    merge_buffer_array(Ma::Nothing, Mb::Nothing) -> Nothing
    merge_buffer_array(Ma::AbstractMatrix, Mb::AbstractMatrix) -> AbstractMatrix

Concatenates the same mask, or the factor rows, of two buffers, and passes a backing neither records through.

This is the optional part of [`merge_states`](@ref). [`assert_buffer_presence_agreement`](@ref) runs first, so the two arguments are both `nothing` or both matrices.

# Arguments

  - `Ma`: The backing of the first buffer, or `nothing`.
  - `Mb`: The backing of the second buffer, or `nothing`.

# Returns

  - `M`: The two backings stacked, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_buffer_array(::Nothing, ::Any)
    return nothing
end
function merge_buffer_array(Ma::AbstractMatrix, Mb::AbstractMatrix)
    return vcat(Ma, Mb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Concatenates the marks of the Shared Series gaps of two buffers over their valid regions.

The marks have no presence rule, because a buffer with no marks has no gap. So two buffers with no marks give no marks, and a buffer with no marks gives `false` for each of its observations beside a buffer that has them.

# Arguments

  - `a`: The buffer of the first block of observations.
  - `b`: The buffer of the second block of observations.

# Returns

  - `G::Option{Vector{Bool}}`: The marks of the rows of `a` followed by the marks of the rows of `b`, or `nothing` when neither buffer has marks.

# Related

  - [`merge_states`](@ref)
  - [`SampleBufferState`](@ref)
"""
function merge_buffer_gaps(a::SampleBufferState, b::SampleBufferState)
    if isnothing(a.G) && isnothing(b.G)
        return nothing
    end
    return vcat(buffer_gap_marks(a), buffer_gap_marks(b))
end
"""
    trim_merged_array(M::Nothing, rows) -> Nothing
    trim_merged_array(M::AbstractMatrix, rows) -> AbstractMatrix
    trim_merged_array(G::AbstractVector, rows) -> AbstractVector

Keeps the rows a cap admits of a merged mask or of the merged factor rows, by index copy, and passes a backing that is not there through.

This is the cap part of [`merge_states`](@ref). It copies the rows and does not take a view. A view would keep the whole concatenation in memory, with the rows that the cap drops, and the merged buffer would carry a backing of a different type from the backing that a fold makes.

# Arguments

  - `M`: The merged backing, or `nothing`. A vector is the backing of the marks of the Shared Series gaps.
  - `rows`: The rows the cap admits.

# Returns

  - `M`: The admitted rows, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`merge_states`](@ref)
"""
function trim_merged_array(::Nothing, ::Any)
    return nothing
end
function trim_merged_array(M::AbstractMatrix, rows)
    return M[rows, :]
end
function trim_merged_array(G::AbstractVector, rows)
    return G[rows]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`SampleBufferState`](@ref), so the copy shares no array with the original.

This is the `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. It copies each backing matrix whole, spare capacity included. So the copy appends on the same terms as the original, and the same `off` and `n` index the masks, the factor rows, the series and the observations. The copy shares the Panel Fields and the names of the series with the original. Nothing writes into the arrays of a panel or into the names, and an append makes new arrays.

# Arguments

  - `x`: The buffer to copy.

# Returns

  - `state::SampleBufferState`: A fresh buffer, equal to `x`, whose backing matrices are fresh matrices and whose Panel Fields are the Panel Fields of `x`.

# Related

  - [`SampleBufferState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::SampleBufferState)
    return SampleBufferState(x.n, x.off, copy(x.X), copy_buffer_array(x.A),
                             copy_buffer_array(x.M), copy_buffer_array(x.F), x.ne,
                             copy_buffer_array(x.E), copy_buffer_array(x.iv),
                             copy_column(x.ivpa), x.P, copy_buffer_array(x.G),
                             x.max_history)
end
"""
    copy_buffer_array(M::Nothing) -> Nothing
    copy_buffer_array(M::AbstractArray) -> AbstractArray

Copies one backing matrix of a buffer, and passes a matrix that is not there through.

This is the `nothing` case of `copy` for the two masks, the factor rows, the series and the backing vector of the marks of the Shared Series gaps. [`Base.copy`](@ref) and [`port_opt_view`](@ref) use it to copy each optional field in one line.

# Arguments

  - `M`: The backing matrix to copy, or `nothing`.

# Returns

  - `M`: A fresh matrix equal to it, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`Base.copy`](@ref)
"""
function copy_buffer_array(::Nothing)
    return nothing
end
function copy_buffer_array(M::AbstractArray)
    return copy(M)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`SampleBufferState`](@ref) to the selected assets.

A buffer holds its observations verbatim, so the slice of the buffer is the buffer of the sliced universe, column for column. The observation axis does not change. The masks and the implied volatilities are per cell, so they slice on the asset axis by the same indices, and a vector `ivpa` slices with them. The factor rows and the Exogenous Series are copied and not sliced, because the selection indexes assets, a factor and a series belong to no asset, and a fit over a subset of the universe reads the same factors and the same series. The Panel Fields are viewed over the selected assets and not copied, because nothing writes into the arrays of a panel. A tensor Panel Field keeps its whole label axis, because the buffer holds no asset names to find the square case with. The slice copies by index and does not take a `view`. A later [`partial_fit!`](@ref) on the viewed estimator writes into its backing matrices, and with a view it would change the buffer of the original estimator. A buffer that holds no observations has no width to slice, so its view is the empty seed with the same cap.

# Algorithm

 1. Return the empty seed with the same cap when the buffer holds no observations.
 2. Copy the columns `i` of `X`, of each mask and of `iv`, view a vector `ivpa` over the entries `i`, copy `F`, `E` and the marks `G` whole, and view each Panel Field over the columns `i`.

# Arguments

  - `x`: The buffer to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `state::SampleBufferState`: The buffer of the same observations over the selected assets.

# Related

  - [`SampleBufferState`](@ref)
  - [`port_opt_view`](@ref)
  - [`partial_fit!`](@ref)
"""
function port_opt_view(x::SampleBufferState, i, args...)
    if iszero(x.n)
        return SampleBufferState(; max_history = x.max_history)
    end
    return SampleBufferState(x.n, x.off, x.X[:, i], slice_buffer_mask(x.A, i),
                             slice_buffer_mask(x.M, i), copy_buffer_array(x.F), x.ne,
                             copy_buffer_array(x.E), slice_buffer_mask(x.iv, i),
                             nothing_scalar_array_view(x.ivpa, i),
                             panel_fields_view(x.P, :, i), copy_buffer_array(x.G),
                             x.max_history)
end
"""
    slice_buffer_mask(M::Nothing, i) -> Nothing
    slice_buffer_mask(M::AbstractMatrix, i) -> AbstractMatrix

Slices one backing matrix of a buffer that has an asset axis to the selected assets, and passes a matrix that is not there through.

This is the per-asset part of [`port_opt_view`](@ref) for this state, so the slice is one line for each field. It slices the two masks and the implied volatilities. It copies by index, for the reason that the method states.

# Arguments

  - `M`: The backing mask or implied volatilities to slice, or `nothing`.
  - `i`: Index or indices of the assets to keep.

# Returns

  - `M`: The matrix over the selected assets, or `nothing`.

# Related

  - [`SampleBufferState`](@ref)
  - [`port_opt_view`](@ref)
"""
function slice_buffer_mask(::Nothing, ::Any)
    return nothing
end
function slice_buffer_mask(M::AbstractMatrix, i)
    return M[:, i]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds observations into the sample buffer an estimator carries.

This is the buffering arm of [`partial_fit!`](@ref). Each estimator that carries a [`SampleBufferState`](@ref) reaches this method. A family that folds exactly has methods of its own, and each of them narrows the `cache` type parameter of its estimator to the state of that fold. So a buffer never reaches them, and it reaches this method. The type of the state selects the route, and no method refuses the step.

The method covers both arms of the interface. The families that refuse the step also declare one method over both arms, and two narrower methods here would be ambiguous against each of them. So the body selects the arm by the type of `X`, and the compiler resolves that branch at each call site.

A buffer carries the per-observation masks beside the observations. So a [`CoveragePolicy`](@ref) mask goes through the wrapper as it goes through the accumulator of an estimator, and the call with no data gives it back to the batch verb. A wrapped estimator folded under a policy gives the answer of a batch fit over the same window under the same policy. The implied volatilities and `ivpa` go through the wrapper in the same way, so a wrapped estimator that reads them, such as an [`ImpliedVolatility`](@ref) inside a [`PortfolioOptimisersCovariance`](@ref), gives the answer of a batch fit over the same rows.

# Algorithm

 1. Read the buffer out of the `cache` field with [`assert_sample_buffer`](@ref), which refuses an estimator that was never wrapped in [`Online`](@ref).
 2. Fold a matrix, its masks and its implied volatilities through the block arm of [`partial_fit!`](@ref), and a vector, its masks and its implied volatilities through the single-observation arm.
 3. Rebind `est.cache` with `Accessors.@reset`, and return the estimator.

# Arguments

  - `est`: Estimator whose buffer is folded forward.
  - `X`: Observations to fold. A matrix holds one observation per row when `dims == 1`, and one per column when `dims == 2`. A vector is a single observation across the assets, and `dims` is ignored.
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the block, of the shape of `X`, or of one entry per asset when `X` is one observation, or `nothing`.
  - `estimation_mask`: The estimation mask, on the same terms as `active_mask`.
  - `iv`: The implied volatilities, on the same terms as `active_mask`.
  - `ivpa`: The implied volatility risk premium adjustment, or `nothing`.

# Validation

  - `est` carries a [`SampleBufferState`](@ref). An `ArgumentError` is thrown otherwise.
  - The masks and `iv`, when they are not `nothing`, have the shape of `X`. A `DimensionMismatch` is thrown otherwise.
  - A buffer that holds observations receives the masks that it records, and it receives `iv` if and only if it records implied volatilities. An `ArgumentError` is thrown otherwise.
  - `ivpa` equals the adjustment that the first append pinned. An `ArgumentError` is thrown otherwise.
  - $(val_dict[:dims])

# Returns

  - `est`: The estimator, with its `cache` field rebound to the buffer after the last observation.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer_kwargs`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator},
                      X::VecNum_MatNum; dims::Int = 1, active_mask = nothing,
                      estimation_mask = nothing, iv = nothing, ivpa = nothing)
    state = assert_sample_buffer(est)
    state = if isa(X, MatNum)
        partial_fit!(state, X; dims = dims, active_mask = active_mask,
                     estimation_mask = estimation_mask, iv = iv, ivpa = ivpa)
    else
        partial_fit!(state, X; active_mask = active_mask, estimation_mask = estimation_mask,
                     iv = iv, ivpa = ivpa)
    end
    return Accessors.@set est.cache = state
end
