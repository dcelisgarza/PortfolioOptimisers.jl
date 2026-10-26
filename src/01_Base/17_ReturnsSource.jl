"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the source of the returns that a reader of the returns data fits on.

A reader of the returns is an estimator that receives the returns data `rd` and picks the matrix that it fits on: a prior, a clustering, phylogeny or centrality estimator, a factor regression, and the step of [`OnlinePortfolioSelection`](@ref). Each reader holds its source in its `x_src` field, so no optimiser gives a source to it. The field is `nothing` when the caller states no source, and otherwise an instance of a concrete subtype. Dispatch on the value picks the matrix, see [`benchmark_relative_returns`](@ref).

| Value                  | A prior or a graph reader reads                                  | A regression reader takes                       |
|:---------------------- |:---------------------------------------------------------------- |:----------------------------------------------- |
| `nothing`              | `rd.X`. A graph reader reads `pr.X` when the call gives no `rd`. | the loadings of the prior, else a fit on `rd.X` |
| [`DataSource`](@ref)   | `rd.X`                                                           | a fit on `rd.X`                                 |
| [`ExcessSource`](@ref) | `rd.X - rd.B`, the Excess Frame                                  | a fit on `rd.X - rd.B`                          |
| [`PriorSource`](@ref)  | `pr.X`, in the frame of the prior                                | the loadings of the prior only                  |

Each family of readers admits its own set of values, and its keyword constructor refuses a different value with [`assert_returns_source`](@ref). A prior and the step of [`OnlinePortfolioSelection`](@ref) read no prior, so they refuse `PriorSource()`.

All concrete types that name a source of the returns should subtype `AbstractReturnsSource`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractReturnsSource` and implement the following methods. A reader refuses a source that its admitted set does not name, see `assert_returns_source`, so a new source reaches only a reader whose keyword constructor admits it. A new source reads no benchmark gap: only [`ExcessSource`](@ref) drops the gap rows of a vector benchmark, see `shared_series_view`.

## benchmark_relative_returns

  - `benchmark_relative_returns(rd::ReturnsResult, x_src::AbstractReturnsSource) -> Option{<:MatNum}`: The asset returns that a reader at `x_src` fits on.

### Arguments

  - `rd`: The returns data that the reader receives.
  - `x_src`: The concrete subtype instance.

### Returns

  - `X::Option{<:MatNum}`: A matrix with the rows and the columns of `rd.X`, or `nothing` when `rd` holds no returns. The function does not modify `rd`.

## source_frames

  - `source_frames(x_src::AbstractReturnsSource, path::AbstractString, raw::Bool) -> ReturnsFrames`: The frames that an outermost reader at `x_src` adds to the group of its optimiser.

### Arguments

  - `x_src`: The concrete subtype instance.
  - $(arg_dict[:frame_path])
  - `raw`: Whether the reader reads the raw returns data at `x_src = nothing`.

### Returns

  - `fr::ReturnsFrames`: `PortfolioOptimisers.ReturnsFrames(true, String[])` when the source reads the Excess Frame, and `PortfolioOptimisers.ReturnsFrames()` otherwise.

### Examples

```jldoctest
julia> struct HalfSource <: PortfolioOptimisers.AbstractReturnsSource end

julia> PortfolioOptimisers.benchmark_relative_returns(rd::ReturnsResult, ::HalfSource) = rd.X / 2;

julia> function PortfolioOptimisers.source_frames(::HalfSource, ::AbstractString, ::Bool)
           return PortfolioOptimisers.ReturnsFrames()
       end;

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; 0.3 0.4]);

julia> benchmark_relative_returns(rd, HalfSource())
2×2 Matrix{Float64}:
 0.05  0.1
 0.15  0.2
```

# Related

  - [`DataSource`](@ref)
  - [`ExcessSource`](@ref)
  - [`PriorSource`](@ref)
  - [`benchmark_relative_returns`](@ref)
  - [`source_frames`](@ref)
  - [`assert_returns_source`](@ref)
"""
abstract type AbstractReturnsSource <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Makes a reader of the returns fit on the asset returns `rd.X`, as the caller gave them.

On a prior and on a graph reader, `DataSource()` reads the same matrix as `x_src = nothing`, but the caller states the choice. An optimiser that holds a reader of the Excess Frame warns about a reader at `nothing` that reads the raw returns, and it does not warn about a reader at `DataSource()`, see [`warn_raw_readers`](@ref). On a regression reader, `DataSource()` always fits the regression on `rd.X`, where `nothing` first takes the loadings of the prior.

# Related

  - [`AbstractReturnsSource`](@ref)
  - [`ExcessSource`](@ref)
  - [`PriorSource`](@ref)
  - [`benchmark_relative_returns`](@ref)
"""
struct DataSource <: AbstractReturnsSource end
"""
$(DocStringExtensions.TYPEDEF)

Makes a reader of the returns fit on the returns in excess of the benchmark, `rd.X - rd.B`, the Excess Frame.

The returns data must carry a benchmark `B`, because no excess return exists without one. A vector benchmark gives one series for every asset, and a matrix benchmark gives one series for each asset. An optimiser drops each observation at which a vector benchmark has a gap when a reader of its group states `ExcessSource()`, so that every reader of the optimiser pairs the same observations, see [`shared_series_view`](@ref).

# Related

  - [`AbstractReturnsSource`](@ref)
  - [`DataSource`](@ref)
  - [`PriorSource`](@ref)
  - [`benchmark_relative_returns`](@ref)
"""
struct ExcessSource <: AbstractReturnsSource end
"""
$(DocStringExtensions.TYPEDEF)

Makes a reader of the returns read the Prior Result in place of the returns data.

A graph reader reads the returns of the prior, `pr.X`, in the frame that the prior fitted in. Under a factor prior `pr.X` is a reconstruction of the rank of the factors, and a scenario window cuts its rows. A regression reader takes the loadings of the prior, `pr.rr`, and refuses a Prior Result that holds none. A prior and the step of [`OnlinePortfolioSelection`](@ref) read no prior, so they refuse `PriorSource()`.

# Related

  - [`AbstractReturnsSource`](@ref)
  - [`DataSource`](@ref)
  - [`ExcessSource`](@ref)
"""
struct PriorSource <: AbstractReturnsSource end
"""
    const DATA_RETURNS_SOURCES = (nothing, DataSource(), ExcessSource())

The values of `x_src` that a reader with no prior to read admits: a prior estimator, the step of [`OnlinePortfolioSelection`](@ref), and the `x_src` keyword of a covariance forecast evaluation.

# Related

  - [`ALL_RETURNS_SOURCES`](@ref)
  - [`assert_returns_source`](@ref)
"""
const DATA_RETURNS_SOURCES = (nothing, DataSource(), ExcessSource())
"""
    const ALL_RETURNS_SOURCES = (nothing, DataSource(), ExcessSource(), PriorSource())

The values of `x_src` that a reader of the prior admits: a clustering, phylogeny or centrality estimator, and a factor regression.

# Related

  - [`DATA_RETURNS_SOURCES`](@ref)
  - [`assert_returns_source`](@ref)
"""
const ALL_RETURNS_SOURCES = (nothing, DataSource(), ExcessSource(), PriorSource())
"""
    source_label(x_src::Nothing) -> String
    source_label(x_src::AbstractReturnsSource) -> String

Return the text that an error message or a warning writes for a value of `x_src`: `"nothing"`, or the name of the type followed by `()`.

# Related

  - [`assert_returns_source`](@ref)
  - [`assert_inner_reader`](@ref)
"""
function source_label(::Nothing)
    return "nothing"
end
function source_label(x_src::AbstractReturnsSource)
    return string(nameof(typeof(x_src)), "()")
end
"""
    assert_returns_source(x_src::Option{<:AbstractReturnsSource}, admitted::Tuple,
                          reader::Sym_Str, sym::Sym_Str = :x_src) -> Nothing

Assert that a reader of the returns admits its source.

Every reader holds the same field type, so the field type alone admits every value. The keyword constructor of each reader calls this function with the set of values that its family admits, [`DATA_RETURNS_SOURCES`](@ref) or [`ALL_RETURNS_SOURCES`](@ref). The configuration alone decides the refusal, so the constructor refuses before a fit starts.

# Arguments

  - $(arg_dict[:x_src_read])
  - `admitted`: The values of `x_src` that the reader admits.
  - `reader`: The name of the reader, for the error message.
  - $(arg_dict[:sym_msg])

# Validation

  - `x_src in admitted`. An `ArgumentError` that names the field, the reader, the value and the admitted values is thrown otherwise.

# Returns

  - `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.assert_returns_source(ExcessSource(),
                                                 PortfolioOptimisers.DATA_RETURNS_SOURCES,
                                                 :EmpiricalPrior) === nothing
true
```

# Related

  - [`AbstractReturnsSource`](@ref)
  - [`DATA_RETURNS_SOURCES`](@ref)
  - [`ALL_RETURNS_SOURCES`](@ref)
  - [`assert_inner_reader`](@ref)
"""
function assert_returns_source(x_src::Option{<:AbstractReturnsSource}, admitted::Tuple,
                               reader::Sym_Str, sym::Sym_Str = :x_src)::Nothing
    @argcheck(x_src in admitted,
              ArgumentError("`$sym = $(source_label(x_src))` is not a source that `$reader` reads. Its `$sym` admits $(join(map(source_label, admitted), ", ", " or "))."))
    return nothing
end
"""
    returns_source(x) -> Option{<:AbstractReturnsSource}

Return the source that the reader `x` states, its `x_src` field.

The default method answers `nothing`, because a value that is not a reader of the returns states no source. Each family of readers adds a method that returns its field. [`assert_inner_reader`](@ref) reads the source through this function, so it accepts any value in a nesting field.

# Arguments

  - `x`: A reader of the returns, or any other value.

# Returns

  - `x_src::Option{<:AbstractReturnsSource}`: The source that `x` states, or `nothing`.

# Related

  - [`assert_inner_reader`](@ref)
  - [`AbstractReturnsSource`](@ref)
"""
function returns_source(::Any)
    return nothing
end
"""
    assert_inner_reader(x, field::Sym_Str, holder::Sym_Str) -> Nothing

Assert that a reader in a nesting field states no source of its own.

An inner reader receives a matrix from the estimator that holds it, and not the returns data. Examples are the `pe` and the `re` of a prior, and the network estimator of a centrality estimator. The holder reads the returns data and picks the matrix, so only the holder states the source. An inner reader keeps `x_src = nothing`, which means that the caller stated no source, so the refusal has no ambiguous case. The keyword constructor of the holder calls this function on each nesting field.

# Arguments

  - `x`: The value in the nesting field. A value that is not a reader of the returns passes, see [`returns_source`](@ref).
  - `field`: The name of the nesting field, for the error message.
  - `holder`: The name of the estimator that holds `x`, for the error message.

# Validation

  - `returns_source(x) === nothing`. An `ArgumentError` that names the field and tells the caller to state the source on the holder is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`returns_source`](@ref)
  - [`assert_returns_source`](@ref)
"""
function assert_inner_reader(x, field::Sym_Str, holder::Sym_Str)::Nothing
    x_src = returns_source(x)
    @argcheck(isnothing(x_src),
              ArgumentError("the `$field` of `$holder` reads the returns matrix that `$holder` picks, so it states no source of its own, and its `x_src` must be `nothing`. It is `$(source_label(x_src))`. State the source on `$holder`, and keep the `x_src` of its `$field` at `nothing`."))
    return nothing
end
"""
    ReturnsFrames(excess::Bool, raw::Vector{String})
    ReturnsFrames()

The frames that the readers of one group read, as the frame walk finds them.

A group is an optimiser and every reader in its slots, the readers inside its risk measures, constraints, uncertainty sets and losses included. The walk stops at an inner optimiser, a fallback and a leader, because each one is an optimiser with its own group. The walk answers two questions about a group: does a reader state [`ExcessSource`](@ref), and which outermost readers state no source and read the raw returns data. [`warn_raw_readers`](@ref) reads both answers, and [`group_source`](@ref) reads the first. The constructor with no arguments gives the frames of a value that holds no reader.

# Related

  - [`reader_frames`](@ref)
  - [`group_frames`](@ref)
  - [`merge_frames`](@ref)
"""
struct ReturnsFrames
    """
    Whether a reader of the group states `x_src = ExcessSource()`.
    """
    excess::Bool
    """
    The paths of the outermost readers of the group that state no source and read the raw returns data.
    """
    raw::Vector{String}
end
function ReturnsFrames()
    return ReturnsFrames(false, String[])
end
"""
    merge_frames(fs::ReturnsFrames...) -> ReturnsFrames

Join the frames of the parts of one group.

# Algorithm

 1. The group states the Excess Frame when one part states it.
 2. The raw readers of the group are the raw readers of each part, in the order of the parts.

# Arguments

  - `fs`: The frames of the parts.

# Returns

  - `fr::ReturnsFrames`: The frames of the group.

# Related

  - [`ReturnsFrames`](@ref)
  - [`reader_frames`](@ref)
"""
function merge_frames(fs::ReturnsFrames...)
    excess = false
    raw = String[]
    for f in fs
        excess |= f.excess
        append!(raw, f.raw)
    end
    return ReturnsFrames(excess, raw)
end
"""
    join_path(path::AbstractString, name) -> String

Return the path of the field `name` of the value at `path`, for the frame walk. The path of a field of the head itself is the name of the field.

# Examples

```jldoctest
julia> PortfolioOptimisers.join_path(\"\", :pe)
"pe"

julia> PortfolioOptimisers.join_path(\"opt\", :pe)
"opt.pe"
```

# Related

  - [`reader_frames`](@ref)
"""
function join_path(path::AbstractString, name)
    return isempty(path) ? string(name) : string(path, ".", name)
end
"""
    source_frames(x_src::ExcessSource, path::AbstractString, raw::Bool) -> ReturnsFrames
    source_frames(x_src::Nothing, path::AbstractString, raw::Bool) -> ReturnsFrames
    source_frames(x_src::Union{DataSource, PriorSource}, path::AbstractString,
                  raw::Bool) -> ReturnsFrames

Return the frames of one outermost reader from the source that it states.

The method of [`reader_frames`](@ref) for a reader calls this function with its own `x_src`. A reader at `PriorSource()` reads the frame of the prior, so it never counts by itself.

# Algorithm

The method that Julia selects is the algorithm.

 1. `ExcessSource()`: the reader states the Excess Frame.
 2. `nothing`: the reader is a raw reader at `path` when `raw` is `true`.
 3. `DataSource()` or `PriorSource()`: the reader adds nothing to the group.

# Arguments

  - $(arg_dict[:x_src_read])
  - $(arg_dict[:frame_path])
  - `raw`: Whether the reader reads the raw returns data at `x_src = nothing`. It is `true` for a prior and a graph reader, and `refit` for a regression reader, see [`refits_regression`](@ref).

# Returns

  - `fr::ReturnsFrames`: The frames of the reader.

# Related

  - [`reader_frames`](@ref)
  - [`ReturnsFrames`](@ref)
"""
function source_frames(::ExcessSource, ::AbstractString, ::Bool)
    return ReturnsFrames(true, String[])
end
function source_frames(::Nothing, path::AbstractString, raw::Bool)
    return ReturnsFrames(false, raw ? String[path] : String[])
end
function source_frames(::Union{DataSource, PriorSource}, ::AbstractString, ::Bool)
    return ReturnsFrames()
end
"""
    reader_frames(x, path::AbstractString, refit::Bool) -> ReturnsFrames
    reader_frames(v::AbstractVector, path::AbstractString, refit::Bool) -> ReturnsFrames

Return the frames that `x` adds to the group of the head that holds it.

This is the verb of the frame walk. The default method answers that `x` adds nothing, which is correct for a value that holds no reader, and for an inner optimiser, a fallback and a leader, where the walk stops. A vector adds the frames of each element, at the path of its position. Two kinds of types add a method:

  - A reader of the returns returns `source_frames(x.x_src, path, raw)`, see [`source_frames`](@ref). It does not walk its own nesting fields, because an inner reader states no source, see [`assert_inner_reader`](@ref).
  - A value that holds readers, such as a risk measure, a constraint, an uncertainty set or a loss, joins the frames of each field that can hold a reader with [`merge_frames`](@ref), at the path that [`join_path`](@ref) gives.

# Arguments

  - $(arg_dict[:frame_x])
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames that `x` adds.

# Related

  - [`group_frames`](@ref)
  - [`source_frames`](@ref)
  - [`merge_frames`](@ref)
  - [`ReturnsFrames`](@ref)
"""
function reader_frames(::Any, ::AbstractString, ::Bool)
    return ReturnsFrames()
end
function reader_frames(v::AbstractVector, path::AbstractString, refit::Bool)
    excess = false
    raw = String[]
    for (i, x) in enumerate(v)
        f = reader_frames(x, string(path, "[", i, "]"), refit)::ReturnsFrames
        excess |= f.excess
        append!(raw, f.raw)
    end
    return ReturnsFrames(excess, raw)
end
"""
    refits_regression(pe) -> Bool

Answer `true` when a regression reader at `x_src = nothing`, in an optimiser whose prior is `pe`, fits its regression on the raw returns data.

A regression reader at `nothing` takes the loadings of the Prior Result when it holds them, and then it reads the frame of the prior. It fits on `rd.X` only when the Prior Result holds no loadings. So it counts as a raw reader only when the type of `pe` shows that the prior gives no loadings. The default method answers `false`: a precomputed Prior Result, and any value whose type does not show it, do not count. Each prior estimator that gives no loadings adds a method that answers `true`.

# Arguments

  - `pe`: The prior of the optimiser: a prior estimator, a Prior Result, or `nothing`.

# Returns

  - `refit::Bool`: Whether a regression reader at `nothing` fits on the raw returns data.

# Related

  - [`reader_frames`](@ref)
  - [`group_frames`](@ref)
"""
function refits_regression(::Any)
    return false
end
"""
    group_frames(head) -> ReturnsFrames

Return the frames of the group of `head`.

An optimiser, a `Pipeline` and an [`OnlinePortfolioSelection`](@ref) add a method. The method finds `refit` from the prior of the head with [`refits_regression`](@ref), calls [`reader_frames`](@ref) on each slot that can hold a reader, and joins the answers with [`merge_frames`](@ref). The default method answers that the head holds no reader.

# Arguments

  - $(arg_dict[:frame_head])

# Returns

  - `fr::ReturnsFrames`: The frames of the group of `head`.

# Related

  - [`reader_frames`](@ref)
  - [`group_source`](@ref)
  - [`warn_raw_readers`](@ref)
"""
function group_frames(::Any)
    return ReturnsFrames()
end
"""
    group_source(fr::ReturnsFrames) -> Option{<:ExcessSource}
    group_source(head) -> Option{<:ExcessSource}

Return the source that the gap rule of a group reads: `ExcessSource()` when a reader of the group states it, and `nothing` otherwise.

A head drops each observation at which a vector benchmark has a gap when a reader of its group states [`ExcessSource`](@ref). A raw reader of the group loses the observations too, so every reader of the head pairs the same observations. The head passes the answer to [`shared_series_view`](@ref) in batch, and to [`shared_series_gaps`](@ref) in an online step. A [`Pipeline`](@ref) passes the answer for its group to each step of the group, see [`pipeline_gap_sources`](@ref). The method on a head reads [`group_frames`](@ref).

# Arguments

  - `fr`: The frames of the group.
  - $(arg_dict[:frame_head])

# Returns

  - `x_src::Option{<:ExcessSource}`: `ExcessSource()` or `nothing`.

# Related

  - [`group_frames`](@ref)
  - [`shared_series_view`](@ref)
"""
function group_source(fr::ReturnsFrames)
    return fr.excess ? ExcessSource() : nothing
end
function group_source(head)
    return group_source(group_frames(head))
end
"""
    gap_source(gsrc::ExcessSource, x_src) -> ExcessSource
    gap_source(gsrc::Nothing, x_src) -> typeof(x_src)

Return the source that decides the gap drop of one reader in a group: the source of the group when it is [`ExcessSource`](@ref), and the source of the reader otherwise.

A group drops a gap of a vector benchmark when any reader of it states `ExcessSource()`, see [`group_source`](@ref). A group source of `nothing` drops nothing for the group, and the reader still drops by its own source. So the reader drops a gap when the group or the reader reads the Excess Frame. The answer decides only which observations the reader reads, never its frame.

# Algorithm

The method that Julia selects is the algorithm.

 1. `gsrc` is `ExcessSource()`: return `gsrc`.
 2. `gsrc` is `nothing`: return `x_src`.

# Arguments

  - `gsrc`: The source of the group, from [`group_source`](@ref).
  - $(arg_dict[:x_src_read])

# Returns

  - `src::Option{<:AbstractReturnsSource}`: The source of the gap drop of the reader.

# Examples

```jldoctest
julia> PortfolioOptimisers.gap_source(ExcessSource(), DataSource())
ExcessSource()

julia> PortfolioOptimisers.gap_source(nothing, DataSource())
DataSource()
```

# Related

  - [`group_source`](@ref)
  - [`shared_series_gaps`](@ref)
  - [`fold_pipeline_owner`](@ref)
"""
function gap_source(gsrc::ExcessSource, ::Option{<:AbstractReturnsSource})
    return gsrc
end
function gap_source(::Nothing, x_src::Option{<:AbstractReturnsSource})
    return x_src
end
"""
    warn_raw_readers(head, fr::ReturnsFrames = group_frames(head)) -> Nothing

Warn when the group of `head` holds a reader of the Excess Frame and a reader that states no source and reads the raw returns data.

The readers of one head can read different frames. A reader that states `DataSource()` beside a reader that states `ExcessSource()` is a mix that the caller chose, and it is silent. A reader at `nothing` that reads the raw returns beside a reader of the Excess Frame is a reader that the caller can have forgotten, so the head warns and then builds as configured. The configuration alone decides the warning, so the constructor of the head calls this function. A `Pipeline` passes the frames of its steps joined to the frames of its optimiser.

Each different mismatch logs one time. The message names the type of the head and the path of each raw reader, and it is the identity of the log record, so a head that a cross-validation builds again logs one time, and a different mismatch still logs.

# Arguments

  - $(arg_dict[:frame_head])
  - `fr`: The frames of the group of `head`.

# Returns

  - `nothing`.

# Related

  - [`group_frames`](@ref)
  - [`DataSource`](@ref)
  - [`ExcessSource`](@ref)
"""
function warn_raw_readers(head, fr::ReturnsFrames = group_frames(head))::Nothing
    if fr.excess && !isempty(fr.raw)
        msg = "`$(nameof(typeof(head)))` holds a reader that states `x_src = ExcessSource()`, and these readers state no source and read the raw returns data: $(join(map(p -> "`$p`", fr.raw), ", ")). State `x_src = DataSource()` on each one to keep the raw returns, or `x_src = ExcessSource()` to read the returns in excess of the benchmark."
        @warn(msg, maxlog = log_limit(:warn, 1), _id = hash(msg))
    end
    return nothing
end

export DataSource, ExcessSource, PriorSource
public AbstractReturnsSource, source_frames, returns_source
