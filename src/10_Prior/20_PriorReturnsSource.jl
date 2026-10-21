"""
    reader_frames(pe::AbstractPriorEstimator, path::AbstractString, refit::Bool)
        -> ReturnsFrames

[`AbstractPriorEstimator`](@ref) method of [`reader_frames`](@ref): the frames that a prior adds to the group of its optimiser.

A prior is an outermost reader of the returns data, and at `x_src = nothing` it reads the raw returns `rd.X`. So the method passes the source of the prior, [`returns_source`](@ref), to [`source_frames`](@ref) with `raw = true`. It does not walk the nesting fields of the prior, because an inner reader states no source, see [`assert_inner_reader`](@ref). A prior that states no field answers `nothing`, so it counts as a raw reader.

# Arguments

  - `pe`: The prior estimator.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit]) A prior does not read it.

# Returns

  - `fr::ReturnsFrames`: The frames of the prior.

# Related

  - [`reader_frames`](@ref)
  - [`source_frames`](@ref)
  - [`returns_source`](@ref)
"""
function reader_frames(pe::AbstractPriorEstimator, path::AbstractString, ::Bool)
    return source_frames(returns_source(pe), path, true)
end
"""
    reader_frames(pr::AbstractPriorResult, path::AbstractString, refit::Bool)
        -> ReturnsFrames

[`AbstractPriorResult`](@ref) method of [`reader_frames`](@ref): the frames that a Prior Result in the place of a prior adds to the group of its optimiser.

A Prior Result records the source that its prior stated, see [`prior_returns_source`](@ref). So a Result that the call with no data on an online fold, a prior step of a [`Pipeline`](@ref) or the caller puts in the place of the prior keeps the frame of that prior. The method passes the recorded source to [`source_frames`](@ref) with `raw = false`. A Result at `nothing` then adds nothing: it was fitted before the head was built, so the head cannot warn that its source was forgotten.

# Arguments

  - $(arg_dict[:pr])
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit]) A Prior Result does not read it.

# Returns

  - `fr::ReturnsFrames`: The frames of the Prior Result.

# Related

  - [`reader_frames`](@ref)
  - [`source_frames`](@ref)
  - [`prior_returns_source`](@ref)
"""
function reader_frames(pr::AbstractPriorResult, path::AbstractString, ::Bool)
    return source_frames(prior_returns_source(pr), path, false)
end
"""
    picked_returns_data(pr::Pr_RR, rd::Option{<:ReturnsResult}) -> Option{<:ReturnsResult}

Return the returns data that a graph verb received, or `nothing` when it received none.

A graph verb over a [`Pr_RR`](@ref) receives a prior result or returns data in `pr`, and returns data in its `rd` keyword. The returns data are `rd` when it holds asset returns. When the call gives no such `rd` and `pr` is itself returns data, `pr` is the returns data. Otherwise the call holds no returns data.

# Arguments

  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])

# Returns

  - `data::Option{<:ReturnsResult}`: The returns data of the call, or `nothing`.

# Related

  - [`returns_matrix_picker`](@ref)
"""
function picked_returns_data(pr::Pr_RR, rd::Option{<:ReturnsResult})
    if !isnothing(rd) && !isnothing(rd.X)
        return rd
    end
    if isa(pr, ReturnsResult) && !isnothing(pr.X)
        return pr
    end
    return nothing
end
"""
    warn_source_fallback(x_src::AbstractReturnsSource) -> Nothing

Warn that a graph reader states a source that the call cannot give, and that the reader reads the `X` of `pr` in its place.

The warning is logged once for each source.

# Arguments

  - $(arg_dict[:x_src_read])

# Returns

  - `nothing`.

# Related

  - [`returns_matrix_picker`](@ref)
"""
function warn_source_fallback(x_src::Union{<:DataSource, <:ExcessSource})::Nothing
    msg = "a graph reader states `x_src = $(source_label(x_src))`, which reads the returns data, and the call gives no returns data, so the reader reads the `X` of the prior result. Pass the returns data as `rd` to read them, or state `x_src = nothing` to read the prior result with no warning."
    @warn(msg, maxlog = log_limit(:warn, 1), _id = hash(msg))
    return nothing
end
function warn_source_fallback(::PriorSource)::Nothing
    msg = "a graph reader states `x_src = PriorSource()`, which reads the prior result, and the call gives returns data and no prior result, so the reader reads the `X` of the returns data. Pass a prior result to read it, or state `x_src = nothing` to read the returns data with no warning."
    @warn(msg, maxlog = log_limit(:warn, 1), _id = hash(msg))
    return nothing
end
"""
    returns_matrix_picker(pr::Pr_RR, rd::Option{<:ReturnsResult},
                          x_src::Option{<:AbstractReturnsSource}) -> MatNum

Pick the returns matrix that a clustering, phylogeny or centrality estimator reads, from the source that it states.

Two results can give asset returns: the prior result and the returns data. Each graph reader states its source in its `x_src` field, and [`returns_source`](@ref) reads it. The default `nothing` reads the returns data, because under a factor prior `pr.X` is a reconstruction of the rank of the factors, and a scenario window cuts its rows. [`picked_returns_data`](@ref) finds the returns data of the call.

A stated source that the call cannot give warns once, and the reader reads `pr.X` in its place, see [`warn_source_fallback`](@ref). `nothing` falls back with no warning.

# Algorithm

The method that Julia selects is the algorithm. `data` is the returns data of the call.

 1. `nothing`: the asset returns of `data`, or `pr.X` when the call holds no returns data.
 2. `DataSource()` or `ExcessSource()`: the returns of `data` that [`benchmark_relative_returns`](@ref) reads under `x_src`, `rd.X` or `rd.X - rd.B`. When the call holds no returns data, warn and return `pr.X`.
 3. `PriorSource()`: `pr.X` when `pr` is a prior result, which holds it in the frame that the prior fitted in. When `pr` is returns data, warn and return `pr.X`.

# Arguments

  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - $(arg_dict[:x_src_read])

# Validation

  - The rules of [`benchmark_relative_returns`](@ref).

# Returns

  - `X::MatNum`: The asset returns matrix of the source.

# Related

  - [`returns_source`](@ref)
  - [`picked_returns_data`](@ref)
  - [`warn_source_fallback`](@ref)
  - [`benchmark_relative_returns`](@ref)
  - [`clusterise`](@ref)
  - [`phylogeny_matrix`](@ref)
  - [`centrality_vector`](@ref)
"""
function returns_matrix_picker(pr::Pr_RR, rd::Option{<:ReturnsResult}, x_src::Nothing)
    data = picked_returns_data(pr, rd)
    return isnothing(data) ? pr.X : benchmark_relative_returns(data, x_src)
end
function returns_matrix_picker(pr::Pr_RR, rd::Option{<:ReturnsResult},
                               x_src::Union{<:DataSource, <:ExcessSource})
    data = picked_returns_data(pr, rd)
    if isnothing(data)
        warn_source_fallback(x_src)
        return pr.X
    end
    return benchmark_relative_returns(data, x_src)
end
function returns_matrix_picker(pr::AbstractPriorResult, ::Option{<:ReturnsResult},
                               ::PriorSource)
    return pr.X
end
function returns_matrix_picker(pr::ReturnsResult, ::Option{<:ReturnsResult},
                               x_src::PriorSource)
    warn_source_fallback(x_src)
    return pr.X
end
"""
    plain_graph_reader(x) -> Bool

Answer whether `x` reads the returns matrix with a plain moment estimator, which refuses a gap in the matrix.

A graph reader, a clustering, network, phylogeny or centrality estimator, computes the correlation of the returns matrix with its `ce`. [`gap_fill_value`](@ref) on `ce` states what the estimator does with a gap. A finite answer marks a plain estimator, which refuses a gapped sample. `NaN` marks a mask-aware estimator, which reads the observed cells. A [`FeatureDistance`](@ref) reads the features of an Asset Panel and not the returns matrix, so a reader that holds one is not plain. A precomputed result reads nothing.

# Algorithm

The method that Julia selects is the algorithm.

 1. A `Tuple` or a vector: `true` when one of its entries is plain.
 2. A [`ClustersEstimator`](@ref) or a [`NetworkEstimator`](@ref): `true` when `gap_fill_value(x.ce)` is finite, and `false` when `x.de` is a [`FeatureDistance`](@ref).
 3. An estimator that holds a graph reader: forward to the reader that it holds.
 4. Every other value, a precomputed result and `nothing` among them: `false`.

# Arguments

  - `x`: A graph reader, a `Tuple` or a vector of graph readers, or any other value.

# Returns

  - `flag::Bool`: `true` when `x` reads the returns matrix with a plain moment estimator.

# Related

  - [`assert_graph_readable`](@ref)
  - [`graph_reader_mask`](@ref)
  - [`gap_fill_value`](@ref)
"""
function plain_graph_reader(::Any)
    return false
end
function plain_graph_reader(xs::Union{<:Tuple, <:AbstractVector})
    return any(plain_graph_reader, xs)
end
function plain_graph_reader(x::Union{<:ClustersEstimator, <:NetworkEstimator})
    return isfinite(gap_fill_value(x.ce))
end
function plain_graph_reader(::Union{<:ClustersEstimator{<:Any, <:FeatureDistance},
                                    <:NetworkEstimator{<:Any, <:FeatureDistance}})
    return false
end
function plain_graph_reader(x::NetworkClustersEstimator)
    return plain_graph_reader(x.nte)
end
function plain_graph_reader(x::Union{<:CentralityEstimator,
                                     <:SemiDefinitePhylogenyEstimator,
                                     <:IntegerPhylogenyEstimator})
    return plain_graph_reader(x.pl)
end
function plain_graph_reader(x::CentralityConstraint)
    return plain_graph_reader(x.A)
end
"""
    graph_gap_remedy

The two remedies for a gap that a plain graph reader meets. The warning of a fit and the refusal of a direct call both append this text, so the two give the same remedies.

A graph reader that states no source reads the returns data, which keep the `NaN` of each gap. A mask-aware prior fills the gap with zero in the returns of the prior only, see [`scenario_fill`](@ref). So the reader can read the observed cells with a mask-aware `ce`, or it can read the returns of the prior.

# Related

  - [`assert_graph_readable`](@ref)
  - [`graph_reader_mask`](@ref)
  - [`plain_graph_reader`](@ref)
"""
const graph_gap_remedy = "To read the asset, set a `CoveragePolicy` on the `ce` of the graph reader, the same one as on the prior, so that it reads the observed cells, for example `ClustersEstimator(; ce = PortfolioOptimisersCovariance(; ce = Covariance(; cvg = CoveragePolicy())))`. Or state the prior as the source of the reader, `x_src = PriorSource()`, so that it reads the returns of the prior, where a mask-aware prior fills each gap of an asset that it kept."
"""
    assert_graph_readable(x, X::MatNum) -> nothing

Refuse a gapped returns matrix in a direct call of a graph verb whose reader is plain.

Inside a fit, [`graph_reader_mask`](@ref) leaves an asset that a plain graph reader cannot read out of the Investable Mask at the entry of the head. A direct call of a graph verb over a prior result has no entry of a fit, so it refuses the gap here. The error names the two remedies of [`graph_gap_remedy`](@ref). A mask-aware reader reads the gap, so this function lets it pass.

# Arguments

  - `x`: The graph reader of the verb.
  - `X`: The returns matrix that [`returns_matrix_picker`](@ref) picked, `observations × assets`.

# Validation

  - `all(isfinite, X)` when [`plain_graph_reader`](@ref) answers `true` for `x`. Raises an [`IsNonFiniteError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`plain_graph_reader`](@ref)
  - [`graph_gap_remedy`](@ref)
  - [`graph_reader_mask`](@ref)
  - [`returns_matrix_picker`](@ref)
"""
function assert_graph_readable(x, X::MatNum)::Nothing
    if plain_graph_reader(x)
        @argcheck(all(isfinite, X),
                  IsNonFiniteError("a graph reader with a plain moment estimator has no correct answer for a gapped sample, so it refuses one: the returns matrix that it reads holds $(count(!isfinite, X)) non-finite entries, the first at $(findfirst(!isfinite, X)). $(graph_gap_remedy)"))
    end
    return nothing
end
"""
    graph_reader_gaps(x, pr::AbstractPriorResult, rd::ReturnsResult) -> Option{BitVector}
    graph_reader_gaps(xs::Union{<:Tuple, <:AbstractVector}, pr::AbstractPriorResult,
                      rd::ReturnsResult) -> Option{BitVector}

Find the assets that hold a non-finite cell in the rows that the plain graph readers of a fit read.

Each graph reader states its own source, so the readers of one fit can read different matrices. This function picks the matrix of each plain reader from the source that the reader states, and joins the gaps of the readers.

# Algorithm

The method that Julia selects is the algorithm.

 1. A `Tuple` or a vector: the union of the gaps of its entries, or `nothing` when no entry has a gap to report.
 2. Any other value: `nothing` when [`plain_graph_reader`](@ref) answers `false`. Otherwise pick the matrix `X` that the reader reads with [`returns_matrix_picker`](@ref) under its own source, [`returns_source`](@ref), and mark each column of `X` that holds a non-finite cell.

# Arguments

  - `x`: A graph reader, or a `Tuple` or a vector of graph readers.
  - `pr`: The prior result of the fit, on the full asset universe.
  - $(arg_dict[:rd])

# Validation

  - The rules of [`returns_matrix_picker`](@ref).

# Returns

  - `gap::Option{BitVector}`: One mark for each asset, `true` at an asset with a non-finite cell, or `nothing` when no plain reader reads the returns.

# Related

  - [`graph_reader_mask`](@ref)
  - [`plain_graph_reader`](@ref)
  - [`returns_matrix_picker`](@ref)
"""
function graph_reader_gaps(x, pr::AbstractPriorResult, rd::ReturnsResult)
    if !plain_graph_reader(x)
        return nothing
    end
    return .!vec(all(isfinite, returns_matrix_picker(pr, rd, returns_source(x)); dims = 1))
end
function graph_reader_gaps(xs::Union{<:Tuple, <:AbstractVector}, pr::AbstractPriorResult,
                           rd::ReturnsResult)
    gap = nothing
    for x in xs
        g = graph_reader_gaps(x, pr, rd)
        if isnothing(g)
            continue
        end
        gap = isnothing(gap) ? g : gap .| g
    end
    return gap
end
"""
    graph_reader_mask(x, imsk::Option{<:BitVector}, pr::AbstractPriorResult,
                      rd::ReturnsResult, strict::Bool) -> Option{BitVector}

Compose the Investable Mask with the assets that the plain graph readers of a fit can read, at the entry of the fit.

A graph reader that reads the returns data meets each gap that a mask-aware prior filled in its own returns only, see [`scenario_fill`](@ref). The `ce` of the reader decides what the gap is. A mask-aware `ce` reads the observed cells, so this function keeps the asset. A plain `ce` has no correct answer for a gapped sample. So this function drops each kept asset that holds a non-finite cell in the rows that the readers read, as a plain estimator over an Asset Panel leaves the asset out of its Coverage Universe. The asset then departs as a non-investable asset: [`investable_reduction`](@ref) announces it, the fit gives it a zero weight, and it is listed on the Non-Investable Axis.

The fit warns once, and the warning names the assets and the two remedies of [`graph_gap_remedy`](@ref). Under `strict = true` the fit refuses the gap instead. A reader that states the prior as its source, `x_src = PriorSource()`, reads the returns of the prior, where a kept column is finite, so this function keeps every asset for it. Each reader states its own source, so [`graph_reader_gaps`](@ref) finds the gaps reader by reader.

Every fit that calls [`feature_readable_mask`](@ref) calls this function on the mask that it gives. A direct call of a graph verb has no entry of a fit, so it refuses the gap with [`assert_graph_readable`](@ref).

# Algorithm

 1. Find the assets that hold a non-finite cell in the rows that the plain readers of `x` read, with [`graph_reader_gaps`](@ref). Return `imsk` when no plain reader reads the returns.
 2. Keep the assets that `imsk` keeps. Return `imsk` when none of them holds a gap.
 3. Refuse under `strict`. Otherwise warn, and return `imsk` with those assets dropped.

# Arguments

  - `x`: A graph reader, or a `Tuple` or a vector of the graph readers of the fit.
  - `imsk`: The Investable Mask of the prior, or `nothing` when every asset is investable.
  - `pr`: The prior result of the fit, on the full asset universe.
  - $(arg_dict[:rd])
  - `strict`: The `strict` field of the optimiser. `true` refuses a gap in place of the warning.

# Validation

  - Under `strict = true`, no kept asset holds a non-finite cell in `X`. Raises an [`IsNonFiniteError`](@ref).
  - At least one asset stays. Raises an [`IsEmptyError`](@ref).

# Returns

  - `imsk::Option{BitVector}`: The composed mask, or `nothing` when every asset stays.

# Related

  - [`graph_reader_gaps`](@ref)
  - [`plain_graph_reader`](@ref)
  - [`graph_gap_remedy`](@ref)
  - [`feature_readable_mask`](@ref)
  - [`investable_reduction`](@ref)
"""
function graph_reader_mask(x, imsk::Option{<:BitVector}, pr::AbstractPriorResult,
                           rd::ReturnsResult, strict::Bool)
    gap = graph_reader_gaps(x, pr, rd)
    if isnothing(gap)
        return imsk
    end
    if !isnothing(imsk)
        gap .&= imsk
    end
    if !any(gap)
        return imsk
    end
    assets = isnothing(rd.nx) ? findall(gap) : rd.nx[gap]
    msg = "a graph reader of this optimiser reads the returns data with a plain moment estimator, and the rows that it reads hold a gap for $(count(gap)) asset(s) that the prior kept: $(assets). A plain moment estimator has no correct answer for a gapped sample"
    @argcheck(!strict,
              IsNonFiniteError("$(msg), and `strict = true` refuses the gap. $(graph_gap_remedy)"))
    @warn("$(msg), so the fit leaves these assets out of the Investable Mask and gives each a zero weight. $(graph_gap_remedy) Pass `strict = true` to refuse the gap.",
          maxlog = log_limit(:warn), _id = @log_id())
    keep = isnothing(imsk) ? .!gap : imsk .& .!gap
    @argcheck(any(keep),
              IsEmptyError("every asset that the prior kept holds a gap in the rows that a plain graph reader of this optimiser reads, so the fit has no asset left. $(graph_gap_remedy)"))
    return keep
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Pick the rows of the implied volatility matrix `iv` that pair the rows of the returns matrix `X` that [`returns_matrix_picker`](@ref) picked.

An [`ImpliedVolatility`](@ref) estimator pairs row `t` of `iv` with row `t` of `X`. The returns data carries `iv`, one row per observation of `rd.X`. A prior result can carry fewer rows in `X` than the returns data: a Scenario Cap keeps the last rows, and a [`CrossSectionalFactorPrior`](@ref) also drops the rows of the Descriptor warm-up and of the exposure lag. Each of them drops leading rows. A prior also drops the observations of a Shared Series gap, which are not leading rows, and it records them in `pr.ss_gaps`. So when `iv` is the implied volatility of the returns data that the prior read, the rows of `pr.X` are the last observations that the marks keep. An optimiser drops the same observations from the returns data before its readers read it, so inside an optimiser `iv` has fewer rows than the marks, and the last rows of `iv` pair the rows of `pr.X`. [`prior_data_rows`](@ref) tells the two cases apart. [`scenario_rows`](@ref) reads the probabilities `w` of a prior result on the last rows of the matrix that the prior fitted.

# Algorithm

 1. Return `iv` unchanged when it is `nothing`, or when it has no more rows than `X`. A `iv` with fewer rows than `X` pairs no row, and the estimator that reads it refuses it.
 2. Otherwise return a view of the rows of `iv` that [`prior_data_rows`](@ref) pairs with the `size(X, 1)` rows of `X`.

# Arguments

  - `pr`: The prior result or the returns data that the graph verb received. Only a prior result records the observations of a Shared Series gap.
  - `X`: The returns matrix that [`returns_matrix_picker`](@ref) picked, `observations × assets`.
  - `iv`: The implied volatility matrix of the returns data, `observations × assets`, or `nothing`.

# Returns

  - `iv::Option{<:MatNum}`: The rows of `iv` that pair the rows of `X`, or `nothing`.

# Related

  - [`returns_matrix_picker`](@ref)
  - [`prior_data_rows`](@ref)
  - [`scenario_rows`](@ref)
  - [`ImpliedVolatility`](@ref)
  - [`clusterise`](@ref)
"""
function implied_volatility_picker(pr::Pr_RR, X::MatNum, iv::Option{<:MatNum})
    if isnothing(iv) || size(iv, 1) <= size(X, 1)
        return iv
    end
    return view(iv, prior_data_rows(pr, size(iv, 1), size(X, 1)), :)
end
"""
    graph_verb_keywords(; ivpa = nothing, branchorder = nothing, strict::Bool = false,
                        dims::Integer = 1) -> Nothing

Refuse a keyword that a graph verb over a [`Pr_RR`](@ref) does not forward.

A graph verb forwards its keywords to the asset-returns method, and the moment estimators of that method ignore a keyword that they do not know. So a keyword that the library removed would run with no error. Each graph verb over a `Pr_RR` calls this function with its own keywords, so an unknown keyword raises the `MethodError` of the language. The four keywords are the ones that the callers of the library forward: `ivpa` to an implied volatility estimator, `branchorder` to a hierarchical clustering, `strict` to a centrality constraint, and `dims`, which the call overrides with `dims = 1`.

# Returns

  - `nothing`.

# Related

  - [`clusterise`](@ref)
  - [`centrality_constraints`](@ref)
"""
function graph_verb_keywords(; ivpa = nothing, branchorder = nothing, strict::Bool = false,
                             dims::Integer = 1)
    return nothing
end
"""
    clusterise(cle::AbstractClustersEstimator, pr::AbstractPriorResult; kwargs...)

Clusterise asset or factor returns from a prior result using a clustering estimator.

`clusterise` applies the specified clustering estimator to the asset returns matrix contained in the prior result object, producing a clustering result for use in phylogeny analysis, constraint generation, or portfolio construction.

# Algorithm

 1. Pick the asset returns matrix `X` from the source that the reader states, [`returns_source`](@ref), with [`returns_matrix_picker`](@ref). Refuse a gap in `X` that a plain graph reader cannot read, with [`assert_graph_readable`](@ref).
 2. Call the asset-returns method of [`clusterise`](@ref) with `X`, passing both results on as `pr` and `rd` and `dims = 1` last, and return the clustering result it produces.

# Arguments

  - `cle`: Clustering estimator.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` under the source of the reader, and passed on to the estimator tree.
  - $(arg_dict[:iv_x_src])
  - `kwargs...`: Additional keyword arguments passed to the clustering estimator. An unknown keyword raises a `MethodError`, see [`graph_verb_keywords`](@ref).

# Returns

  - `clr::AbstractClusteringResult`: Result object containing clustering information.

# Related

  - [`ClustersEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`clusterise`](@ref)
"""
function clusterise(cle::AbstractClustersEstimator, pr::Pr_RR;
                    rd::Option{<:ReturnsResult} = nothing, iv::Option{<:MatNum} = nothing,
                    kwargs...)
    graph_verb_keywords(; kwargs...)
    X = returns_matrix_picker(pr, rd, returns_source(cle))
    assert_graph_readable(cle, X)
    return clusterise(cle, X; pr = pr, rd = rd, iv = implied_volatility_picker(pr, X, iv),
                      kwargs..., dims = 1)
end
"""
    phylogeny_matrix(pl::NwE_ClE_Cl, pr::AbstractPriorResult;
                     kwargs...)

Compute the phylogeny matrix from asset returns in a prior result using a network or clustering estimator.

`phylogeny_matrix` applies the specified network or clustering estimator to the asset returns matrix contained in the prior result object, producing a phylogeny matrix for use in constraint generation, centrality analysis, or portfolio construction.

# Algorithm

 1. Pick the asset returns matrix `X` from the source that the reader states, [`returns_source`](@ref), with [`returns_matrix_picker`](@ref). Refuse a gap in `X` that a plain graph reader cannot read, with [`assert_graph_readable`](@ref).
 2. Call the asset-returns method of [`phylogeny_matrix`](@ref) with `X`, passing both results on as `pr` and `rd` and `dims = 1` last, and return the phylogeny result it produces.

# Arguments

  - `pl`: Network estimator, clusters estimator, or clustering result.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` under the source of the reader, and passed on to the estimator tree.
  - $(arg_dict[:iv_x_src])
  - `kwargs...`: Additional keyword arguments passed to the estimator. An unknown keyword raises a `MethodError`, see [`graph_verb_keywords`](@ref).

# Returns

  - `plr::PhylogenyResult`: Result object containing the phylogeny matrix.

# Related

  - [`NetworkEstimator`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_matrix(pl::NwE_ClE_Cl, pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing,
                          iv::Option{<:MatNum} = nothing, kwargs...)
    graph_verb_keywords(; kwargs...)
    X = returns_matrix_picker(pr, rd, returns_source(pl))
    assert_graph_readable(pl, X)
    return phylogeny_matrix(pl, X; pr = pr, rd = rd,
                            iv = implied_volatility_picker(pr, X, iv), kwargs..., dims = 1)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute phylogeny constraints from asset returns in a prior result using a phylogeny constraint estimator.

`phylogeny_constraints` delegates to the asset-returns variant with the `X` that `plc` states as its source, `pr.X` or the returns of `rd`.

# Algorithm

 1. Pick the asset returns matrix `X` from the source that the reader states, [`returns_source`](@ref), with [`returns_matrix_picker`](@ref). Refuse a gap in `X` that a plain graph reader cannot read, with [`assert_graph_readable`](@ref).
 2. Call the asset-returns method of [`phylogeny_constraints`](@ref) with `X`, passing both results on as `pr` and `rd` and `dims = 1` last, and return the constraint result it produces.

# Arguments

  - `plc`: Phylogeny constraint estimator.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` under the source of the reader, and passed on to the estimator tree.
  - $(arg_dict[:iv_x_src])
  - `kwargs...`: Additional keyword arguments passed to the estimator. An unknown keyword raises a `MethodError`, see [`graph_verb_keywords`](@ref).

# Returns

  - Phylogeny constraint result.

# Related

  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
function phylogeny_constraints(plc::AbstractPhylogenyConstraintEstimator, pr::Pr_RR;
                               rd::Option{<:ReturnsResult} = nothing,
                               iv::Option{<:MatNum} = nothing, kwargs...)
    graph_verb_keywords(; kwargs...)
    X = returns_matrix_picker(pr, rd, returns_source(plc))
    assert_graph_readable(plc, X)
    return phylogeny_constraints(plc, X; pr = pr, rd = rd,
                                 iv = implied_volatility_picker(pr, X, iv), kwargs...,
                                 dims = 1)
end
"""
    centrality_vector(cte::CentralityEstimator, pr::AbstractPriorResult; kwargs...)

Compute the centrality vector for a centrality estimator and prior result.

`centrality_vector` applies the centrality algorithm in the estimator to the network constructed from the asset returns in the prior result, returning centrality scores for each asset.

# Algorithm

 1. Pick the asset returns matrix `X` from the source that the reader states, [`returns_source`](@ref), with [`returns_matrix_picker`](@ref). Refuse a gap in `X` that a plain graph reader cannot read, with [`assert_graph_readable`](@ref).
 2. Call the asset-returns method of [`centrality_vector`](@ref) with `X`, passing both results on as `pr` and `rd` and `dims = 1` last, and return the centrality result it produces.

# Arguments

  - $(arg_dict[:cte])
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` under the source of the reader, and passed on to the estimator tree.
  - $(arg_dict[:iv_x_src])
  - `kwargs...`: Additional keyword arguments passed to the asset-returns method. An unknown keyword raises a `MethodError`, see [`graph_verb_keywords`](@ref).

# Returns

  - `plr::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(cte::CentralityEstimator, pr::Pr_RR;
                           rd::Option{<:ReturnsResult} = nothing,
                           iv::Option{<:MatNum} = nothing, kwargs...)
    graph_verb_keywords(; kwargs...)
    X = returns_matrix_picker(pr, rd, returns_source(cte))
    assert_graph_readable(cte, X)
    return centrality_vector(cte, X; pr = pr, rd = rd,
                             iv = implied_volatility_picker(pr, X, iv), kwargs..., dims = 1)
end
"""
    centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm,
                      pr::AbstractPriorResult; kwargs...)

Compute the centrality vector for a network or clustering estimator and centrality algorithm.

`centrality_vector` constructs the phylogeny matrix from the asset returns in the prior result, builds a graph, and computes node centrality scores using the specified centrality algorithm.

# Algorithm

 1. Pick the asset returns matrix `X` from the source that the reader states, [`returns_source`](@ref), with [`returns_matrix_picker`](@ref). Refuse a gap in `X` that a plain graph reader cannot read, with [`assert_graph_readable`](@ref).
 2. Call the asset-returns method of [`centrality_vector`](@ref) with `X`, passing both results on as `pr` and `rd` and `dims = 1` last, and return the centrality result it produces.

# Arguments

  - `pl`: Network estimator, clusters estimator, or clustering result.
  - $(arg_dict[:cta])
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` under the source of the reader, and passed on to the estimator tree.
  - $(arg_dict[:iv_x_src])
  - `kwargs...`: Additional keyword arguments passed to the asset-returns method. An unknown keyword raises a `MethodError`, see [`graph_verb_keywords`](@ref).

# Returns

  - `plr::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm, pr::Pr_RR;
                           rd::Option{<:ReturnsResult} = nothing,
                           iv::Option{<:MatNum} = nothing, kwargs...)
    graph_verb_keywords(; kwargs...)
    X = returns_matrix_picker(pr, rd, returns_source(pl))
    assert_graph_readable(pl, X)
    return centrality_vector(pl, ct, X; pr = pr, rd = rd,
                             iv = implied_volatility_picker(pr, X, iv), kwargs..., dims = 1)
end
"""
    average_centrality(pl::NwE_Pl_ClE_Cl,
                       ct::AbstractCentralityAlgorithm, w::VecNum,
                       pr::AbstractPriorResult; kwargs...)

Compute the weighted average centrality for a network or phylogeny result.

`average_centrality` computes the centrality vector using the specified network or phylogeny estimator and centrality algorithm, then returns the weighted average using the provided portfolio weights.

# Algorithm

 1. Compute the centrality result with the [`Pr_RR`](@ref) method of [`centrality_vector`](@ref), forwarding `rd` unchanged. That method picks the matrix from the source that `pl` states, so this method never reads a result itself.
 2. Return the dot product of that result's `X`, the centrality vector, with the weights `w`.

# Arguments

  - `pl`: Network estimator or phylogeny result.
  - $(arg_dict[:cta])
  - `w`: Portfolio weights vector.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` under the source of the reader, and passed on to the estimator tree.
  - `kwargs...`: Additional keyword arguments passed to the asset-returns method. An unknown keyword raises a `MethodError`, see [`graph_verb_keywords`](@ref).

# Returns

  - `ac::Number`: Weighted average centrality.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`Pr_RR`](@ref)
  - [`centrality_vector`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(pl::NwE_Pl_ClE_Cl, ct::AbstractCentralityAlgorithm, w::VecNum,
                            pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing, kwargs...)
    return LinearAlgebra.dot(centrality_vector(pl, ct, pr; rd = rd, kwargs...).X, w)
end
"""
    average_centrality(cte::CentralityEstimator, w::VecNum, pr::AbstractPriorResult;
                       kwargs...)

Compute the weighted average centrality for a centrality estimator.

`average_centrality` applies the centrality algorithm in the estimator to the network constructed from the asset returns in the prior result, then returns the weighted average using the provided portfolio weights.

# Algorithm

 1. Pick the asset returns matrix `X` from the source that the reader states, [`returns_source`](@ref), with [`returns_matrix_picker`](@ref). Refuse a gap in `X` that a plain graph reader cannot read, with [`assert_graph_readable`](@ref).
 2. Call the asset-returns method of [`average_centrality`](@ref) with `X`, passing both results on as `pr` and `rd` and `dims = 1` last, and return the weighted average it produces.

The estimator method picks between the results itself, where the network-and-algorithm method above delegates that to [`centrality_vector`](@ref). The two reach the same selection: `cte` carries `pl` and `ct` in its own fields, so the asset-returns method it calls is the one the other method's step 1 would have reached.

# Arguments

  - $(arg_dict[:cte])
  - `w`: Portfolio weights vector.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` under the source of the reader, and passed on to the estimator tree.
  - $(arg_dict[:iv_x_src])
  - `kwargs...`: Additional keyword arguments passed to the asset-returns method. An unknown keyword raises a `MethodError`, see [`graph_verb_keywords`](@ref).

# Returns

  - `ac::Number`: Weighted average centrality.

# Related

  - [`CentralityEstimator`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`centrality_vector`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(cte::CentralityEstimator, w::VecNum, pr::Pr_RR;
                            rd::Option{<:ReturnsResult} = nothing,
                            iv::Option{<:MatNum} = nothing, kwargs...)
    graph_verb_keywords(; kwargs...)
    X = returns_matrix_picker(pr, rd, returns_source(cte))
    assert_graph_readable(cte, X)
    return average_centrality(cte, w, X; pr = pr, rd = rd,
                              iv = implied_volatility_picker(pr, X, iv), kwargs...,
                              dims = 1)
end
"""
    asset_phylogeny(pl::NwE_ClE_Cl,
                    w::VecNum, pr::AbstractPriorResult; kwargs...)

Compute the asset phylogeny score for a portfolio allocation using a phylogeny estimator or clustering result and a prior result.

This function computes the phylogeny matrix from the asset returns in the prior result using the specified phylogeny estimator or clustering result, then evaluates the asset phylogeny score for the given portfolio weights. The asset phylogeny score quantifies the degree of phylogenetic (network or cluster-based) structure present in the portfolio allocation.

# Algorithm

 1. Pick the asset returns matrix `X` from the source that the reader states, [`returns_source`](@ref), with [`returns_matrix_picker`](@ref). Refuse a gap in `X` that a plain graph reader cannot read, with [`assert_graph_readable`](@ref).
 2. Call the asset-returns method of [`asset_phylogeny`](@ref) with `X`, passing both results on as `pr` and `rd` and `dims = 1` last, and return the score it produces.

# Arguments

  - `pl`: Phylogeny estimator or clustering result used to compute the phylogeny matrix.
  - `w`: Portfolio weights vector.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` under the source of the reader, and passed on to the estimator tree.
  - $(arg_dict[:iv_x_src])
  - `kwargs...`: Additional keyword arguments passed to the phylogeny matrix computation. An unknown keyword raises a `MethodError`, see [`graph_verb_keywords`](@ref).

# Returns

  - `score::Number`: Asset phylogeny score.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`asset_phylogeny`](@ref): The asset-returns methods this one delegates to. They build the phylogeny matrix, add up the gross weight of the related pairs, and divide by the gross weight of every pair. That is where the score's closed form and its numbered steps are stated.
"""
function asset_phylogeny(pl::NwE_ClE_Cl, w::VecNum, pr::Pr_RR;
                         rd::Option{<:ReturnsResult} = nothing,
                         iv::Option{<:MatNum} = nothing, kwargs...)
    graph_verb_keywords(; kwargs...)
    X = returns_matrix_picker(pr, rd, returns_source(pl))
    assert_graph_readable(pl, X)
    return asset_phylogeny(pl, w, X; pr = pr, rd = rd,
                           iv = implied_volatility_picker(pr, X, iv), kwargs..., dims = 1)
end
"""
    centrality_constraints(ccs::CentralityConstraint, pr::Pr_RR;
                           rd::Option{<:ReturnsResult} = nothing,
                           iv::Option{<:MatNum} = nothing, kwargs...)
    centrality_constraints(ccs::VecCC, pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing,
                           iv::Option{<:MatNum} = nothing, kwargs...)

Compute centrality constraints from asset returns in a prior result using a centrality constraint estimator.

`centrality_constraints` delegates to the asset-returns variant with the `X` that each constraint states as its source, `pr.X` or the returns of `rd`.

Each constraint of a vector states its own source, so two constraints of one vector can read different matrices. A vector whose constraints state one source reads one matrix. A vector whose constraints state different sources computes the rows of each constraint from its own matrix, and joins them in the order of the vector with [`merge_linear_constraints`](@ref), which gives the rows of the one call.

# Algorithm

 1. A vector whose constraints state different sources: compute the result of each constraint with the method for one constraint, and join the results that are not `nothing` with [`merge_linear_constraints`](@ref). Return `nothing` when every result is `nothing`.
 2. Otherwise pick the asset returns matrix `X` from the source that the constraints state, [`returns_source`](@ref), with [`returns_matrix_picker`](@ref). Refuse a gap in `X` that a plain graph reader cannot read, with [`assert_graph_readable`](@ref).
 3. Call the asset-returns method of [`centrality_constraints`](@ref) with `X`, passing both results on as `pr` and `rd` and `dims = 1` last, and return the constraint result it produces.

# Arguments

  - `ccs`: Centrality constraint estimator or vector thereof.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for `X` under the source of the reader, and passed on to the estimator tree.
  - $(arg_dict[:iv_x_src])
  - `kwargs...`: Additional keyword arguments passed to the estimator. `strict` is read by the asset-returns variant, which reports a dropped zero centrality vector through it. An unknown keyword raises a `MethodError`, see [`graph_verb_keywords`](@ref).

# Returns

  - Centrality constraint result.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`returns_matrix_picker`](@ref)
  - [`merge_linear_constraints`](@ref)
  - [`centrality_constraints`](@ref)
"""
function centrality_constraints(ccs::VecCC, pr::Pr_RR;
                                rd::Option{<:ReturnsResult} = nothing,
                                iv::Option{<:MatNum} = nothing, kwargs...)
    if isempty(ccs)
        return centrality_constraints_from(ccs, pr, rd, iv, nothing; kwargs...)
    end
    if allequal(returns_source, ccs)
        return centrality_constraints_from(ccs, pr, rd, iv, returns_source(first(ccs));
                                           kwargs...)
    end
    lcs = [centrality_constraints(cc, pr; rd = rd, iv = iv, kwargs...) for cc in ccs]
    kept = filter(!isnothing, lcs)
    return isempty(kept) ? nothing : merge_linear_constraints(kept)
end
function centrality_constraints(ccs::CentralityConstraint, pr::Pr_RR;
                                rd::Option{<:ReturnsResult} = nothing,
                                iv::Option{<:MatNum} = nothing, kwargs...)
    return centrality_constraints_from(ccs, pr, rd, iv, returns_source(ccs); kwargs...)
end
"""
    centrality_constraints_from(ccs::CC_VecCC, pr::Pr_RR, rd::Option{<:ReturnsResult},
                                iv::Option{<:MatNum},
                                x_src::Option{<:AbstractReturnsSource}; kwargs...)

Compute the centrality constraints of `ccs` from the returns matrix of the source `x_src`, the steps 2 and 3 of [`centrality_constraints`](@ref) over a [`Pr_RR`](@ref).

# Arguments

  - `ccs`: Centrality constraint estimator or vector thereof, whose constraints state the source `x_src`.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - $(arg_dict[:iv_x_src])
  - $(arg_dict[:x_src_read])
  - `kwargs...`: Additional keyword arguments passed to the asset-returns variant. An unknown keyword raises a `MethodError`, see [`graph_verb_keywords`](@ref).

# Validation

  - The rules of [`returns_matrix_picker`](@ref) and [`assert_graph_readable`](@ref).

# Returns

  - Centrality constraint result.

# Related

  - [`centrality_constraints`](@ref)
  - [`returns_matrix_picker`](@ref)
"""
function centrality_constraints_from(ccs::CC_VecCC, pr::Pr_RR, rd::Option{<:ReturnsResult},
                                     iv::Option{<:MatNum},
                                     x_src::Option{<:AbstractReturnsSource}; kwargs...)
    graph_verb_keywords(; kwargs...)
    X = returns_matrix_picker(pr, rd, x_src)
    assert_graph_readable(ccs, X)
    return centrality_constraints(ccs, X; pr = pr, rd = rd,
                                  iv = implied_volatility_picker(pr, X, iv), kwargs...,
                                  dims = 1)
end
"""
    returns_source(pe::Union{<:EmpiricalPrior, <:FactorPrior, <:HighOrderPriorEstimator,
                             <:HighOrderFactorPriorEstimator, <:BlackLittermanPrior,
                             <:BayesianBlackLittermanPrior, <:FactorBlackLittermanPrior,
                             <:AugmentedBlackLittermanPrior, <:EntropyPoolingPrior,
                             <:MeucciEntropyPoolingPrior, <:OpinionPoolingPrior,
                             <:CrossSectionalFactorPrior, <:LowDimensionEnsemblePrior})
        -> Option{<:AbstractReturnsSource}

The prior estimators of the library method of [`returns_source`](@ref): the source that the prior states in its `x_src` field.

The [`ReturnsResult`](@ref) method of [`prior`](@ref), [`partial_fit!`](@ref) and the frame walk of an optimiser read the source of a prior through this method. A prior that a caller defines states no source unless it adds a method, see [`AbstractPriorEstimator`](@ref).

# Arguments

  - `pe`: A prior estimator of the library.

# Returns

  - `x_src::Option{<:AbstractReturnsSource}`: The `x_src` field of `pe`.

# Related

  - [`returns_source`](@ref)
  - [`prior`](@ref)
  - [`reader_frames`](@ref)
"""
function returns_source(pe::Union{<:EmpiricalPrior, <:FactorPrior,
                                  <:HighOrderPriorEstimator,
                                  <:HighOrderFactorPriorEstimator, <:BlackLittermanPrior,
                                  <:BayesianBlackLittermanPrior,
                                  <:FactorBlackLittermanPrior,
                                  <:AugmentedBlackLittermanPrior, <:EntropyPoolingPrior,
                                  <:MeucciEntropyPoolingPrior, <:OpinionPoolingPrior,
                                  <:CrossSectionalFactorPrior, <:LowDimensionEnsemblePrior})
    return pe.x_src
end
"""
    refits_regression(pe::Union{<:EmpiricalPrior, <:LowDimensionEnsemblePrior}) -> true
    refits_regression(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior,
                                <:BayesianBlackLittermanPrior, <:EntropyPoolingPrior,
                                <:MeucciEntropyPoolingPrior}) -> Bool
    refits_regression(pe::OpinionPoolingPrior) -> Bool

The prior estimators of the library methods of [`refits_regression`](@ref): whether the prior gives no loadings, so that a regression reader at `x_src = nothing` fits its own regression on the raw returns data.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`EmpiricalPrior`](@ref) and [`LowDimensionEnsemblePrior`](@ref) fit no regression, so the answer is `true`.
 2. A prior that forwards the regression of the prior that it embeds, [`HighOrderPriorEstimator`](@ref), [`BlackLittermanPrior`](@ref), [`BayesianBlackLittermanPrior`](@ref), [`EntropyPoolingPrior`](@ref) and [`MeucciEntropyPoolingPrior`](@ref), answers for its `pe`.
 3. [`OpinionPoolingPrior`](@ref) returns the regression of the prior that it refits on the pooled weights, so it answers for its `pe2`.

Every other prior of the library gives loadings, and the fallback answers `false`.

# Arguments

  - `pe`: A prior estimator of the library.

# Returns

  - `refit::Bool`: Whether a regression reader at `nothing` fits on the raw returns data.

# Related

  - [`refits_regression`](@ref)
  - [`reader_frames`](@ref)
"""
function refits_regression(::Union{<:EmpiricalPrior, <:LowDimensionEnsemblePrior})
    return true
end
function refits_regression(pe::Union{<:HighOrderPriorEstimator, <:BlackLittermanPrior,
                                     <:BayesianBlackLittermanPrior, <:EntropyPoolingPrior,
                                     <:MeucciEntropyPoolingPrior})
    return refits_regression(pe.pe)
end
function refits_regression(pe::OpinionPoolingPrior)
    return refits_regression(pe.pe2)
end
"""
    stamp_returns_source(pr::LowOrderPrior, pe) -> LowOrderPrior
    stamp_returns_source(pr::HighOrderPrior, pe) -> HighOrderPrior
    stamp_returns_source(pr::AbstractPriorResult, pe) -> AbstractPriorResult

Record in a Prior Result the source that its prior estimator states.

[`prior`](@ref) calls this function when it fits on a [`ReturnsResult`](@ref), and when it reads out the state of an online fold. A consumer that receives the Result in place of the estimator then reads the frame of `X` from the Result, see [`outer_returns_source`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. [`LowOrderPrior`](@ref): read the source of `pe` with [`returns_source`](@ref). Return `pr` when its `x_src` is already that source, and otherwise rebuild it with [`reconstruct_prior`](@ref) and the new `x_src`.
 2. [`HighOrderPrior`](@ref): stamp the `LowOrderPrior` that it wraps, and rebuild the outer Result around it. The outer Result answers `x_src` from the inner one.
 3. A Prior Result of another type has no field for the source, so it is returned unchanged, and a consumer reads it as raw returns.

# Arguments

  - $(arg_dict[:pr])
  - $(arg_dict[:pe])

# Returns

  - `pr::AbstractPriorResult`: The Prior Result, with the source of `pe` recorded when its type holds the field.

# Related

  - [`prior_returns_source`](@ref)
  - [`returns_source`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function stamp_returns_source(pr::LowOrderPrior, pe)
    x_src = returns_source(pe)
    return pr.x_src === x_src ? pr : reconstruct_prior(pr, (; x_src = x_src))
end
function stamp_returns_source(pr::HighOrderPrior, pe)
    inner = stamp_returns_source(pr.pr, pe)
    return inner === pr.pr ? pr : reconstruct_prior(pr, (; pr = inner))
end
function stamp_returns_source(pr::AbstractPriorResult, ::Any)
    return pr
end
"""
    prior_returns_source(pr::Union{<:LowOrderPrior, <:HighOrderPrior})
        -> Option{<:AbstractReturnsSource}
    prior_returns_source(pr::AbstractPriorResult) -> Nothing

Return the source that a Prior Result records, see [`stamp_returns_source`](@ref).

A Prior Result of a type that the library does not ship records no source, so the answer is `nothing`, and a consumer reads its `X` as raw returns.

# Arguments

  - $(arg_dict[:pr])

# Returns

  - `x_src::Option{<:AbstractReturnsSource}`: The `x_src` of the [`LowOrderPrior`](@ref), or `nothing`.

# Related

  - [`stamp_returns_source`](@ref)
  - [`outer_returns_source`](@ref)
"""
function prior_returns_source(pr::Union{<:LowOrderPrior, <:HighOrderPrior})
    return pr.x_src
end
function prior_returns_source(::AbstractPriorResult)
    return nothing
end
