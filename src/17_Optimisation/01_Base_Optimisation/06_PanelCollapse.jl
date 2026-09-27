"""
$(DocStringExtensions.TYPEDSIGNATURES)

Obtains the fees to use for net return calculations from an optimisation result.

An explicitly provided `fees` wins. Otherwise the fees are read from the `fees` property of `res`, and a result exposing no such property gives `nothing`.

# Arguments

  - `res`: Optimisation result, potentially containing a `fees` property.
  - `fees`: Optional fees to use, which take precedence over `res.fees` if provided.

# Returns

  - `Option{<:Fees}`: The fees to use for net return calculations, or `nothing` if not found.

A result's fee sits on the universe the fit **solved**, so a consumer that pairs it with `res.w`, which is on the caller's, reads both through [`result_investable_view`](@ref) rather than through this verb alone.

# Related

  - [`calc_net_returns`](@ref)
  - [`result_investable_view`](@ref)
  - [`OptimisationResult`](@ref)
  - [`Fees`](@ref)
"""
function extract_fees(res::OptimisationResult, fees::Option{<:Fees} = nothing)
    if isnothing(fees) && hasproperty(res, :fees)
        fees = res.fees
    end
    return fees
end
"""
    calc_net_returns(res::OptimisationResult, X::MatNum, fees = nothing, wd = nothing, obs = nothing)
    calc_net_returns(res::OptimisationResult, pr::Pr_RR, fees = nothing, wd = nothing, obs = nothing)

Compute net returns for a [`OptimisationResult`](@ref).

`fees` takes precedence over `res.fees` if both are provided. Delegates to [`calc_net_returns(w, X, fees, wd, obs)`](@ref).

When `pr::Pr_RR` is passed, the carrier is paired whole and its `X` is read after.

The weights, the matrix and the fee meet on the investable universe of `res`, through [`result_investable_view`](@ref): `res.w` is on the caller's universe and `res.fees` on the one the fit solved, so the weights and a caller's `X` are viewed at the result's Investable Mask, and a caller's `fees` takes the door a fee takes at the fit.

`wd` is the Weight Drift the window is read under. `nothing` reads the window at the constant weights `res.w`, which is the library's original behaviour. A [`SelfFinancingDrift`](@ref) reads it as the wealth ratio of the drifted holdings, and `obs` then names the observations of the message a non-positive wealth raises.

# Related

  - [`calc_net_returns`](@ref)
  - [`result_investable_view`](@ref)
  - [`OptimisationResult`](@ref)
  - [`Pr_RR`](@ref)
  - [`AbstractWeightDrift`](@ref)
  - [`SelfFinancingDrift`](@ref)
"""
function calc_net_returns(res::OptimisationResult, X::MatNum,
                          fees::Option{<:Fees} = nothing,
                          wd::Option{<:AbstractWeightDrift} = nothing, obs = nothing)
    _, w, X, fees = result_investable_view(res, X, fees)
    return calc_net_returns(w, X, fees, wd, obs)
end
function calc_net_returns(res::OptimisationResult, pr::Pr_RR,
                          fees::Option{<:Fees} = nothing,
                          wd::Option{<:AbstractWeightDrift} = nothing, obs = nothing)
    # The carrier is paired whole, so the result's own prior handed back is known by
    # identity; its matrix alone would be viewed a second time.
    _, w, pr, fees = result_investable_view(res, pr, fees)
    return calc_net_returns(w, pr.X, fees, wd, obs)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Normalises inner weights into the convex weights that collapse real assets onto a meta-optimiser's synthetic assets.

Quantities carried alongside the returns matrix are either *extensive* (returns, benchmark returns) and collapse as a plain weighted sum `w'x`, or *intensive* (rates such as `rd.iv` and `rd.ivpa`) and collapse as a weighted *average*. A plain weighted sum scales an intensive quantity by the gross exposure `sⱼ = Σᵢ|wᵢⱼ|`, so a shorting or leveraged portfolio (`sⱼ ≠ 1`) inflates a rate that should not depend on gross exposure at all.

Normalising the weights once here makes every subsequent product a convex combination, so callers collapsing an intensive quantity need only pass their weights through this function.

# Arguments

  - `w`: Inner weights. A vector collapses onto a single synthetic asset; a matrix (assets × synthetic assets) collapses each column independently.

# Returns

  - `w`: `abs.(w)`, with each column scaled to sum to one. A column summing to zero — a degenerate synthetic asset — is left as-is rather than divided, preserving the zero row it already produced.

# Related

  - [`prepare_outer_rd`](@ref)
  - [`reconstruct_rd`](@ref)
"""
function synthetic_asset_weights(w::VecNum)
    w = abs.(w)
    s = sum(w)
    return iszero(s) ? w : w / s
end
function synthetic_asset_weights(w::MatNum)
    w = abs.(w)
    s = sum(w; dims = 1)
    return w ./ map(x -> iszero(x) ? one(x) : x, s)
end
"""
    collapse_panel_numeric(A::AbstractVector, W::MatNum) -> Vector
    collapse_panel_numeric(A::AbstractMatrix, W::MatNum) -> Matrix

Collapse one numeric Panel Field array onto the synthetic assets, as a convex combination.

`W` is the normalised weight matrix [`synthetic_asset_weights`](@ref) returns, `assets × synthetic assets`. A static array is one value per asset and a time-varying one is `observations × assets`, so the asset axis is the only one contracted in both.

# Algorithm

The method that Julia selects is the algorithm. A vector contracts as `transpose(W) * A`, and a matrix as `A * W`.

# Arguments

  - `A`: The values, `assets` or `observations × assets`.
  - `W`: Normalised inner weights, assets × synthetic assets.

# Returns

  - The collapsed values, `synthetic assets` or `observations × synthetic assets`.

# Related

  - [`collapse_asset_panel`](@ref)
  - [`synthetic_asset_weights`](@ref)
  - [`NumericPanelField`](@ref)
"""
function collapse_panel_numeric(A::AbstractVector, W::MatNum)
    return transpose(W) * A
end
function collapse_panel_numeric(A::AbstractMatrix, W::MatNum)
    return A * W
end
"""
    collapse_panel_tensor(A::AbstractMatrix, W::MatNum, sq::Bool) -> Matrix
    collapse_panel_tensor(A::AbstractArray{<:Any, 3}, W::MatNum, sq::Bool) -> Array

Collapse one tensor Panel Field array onto the synthetic assets, as a convex combination.

A tensor array is `assets × labels` when static and `observations × assets × labels` when time-varying. The asset axis is always contracted. When `sq` is `true` the label axis **is** the asset axis ([`features_are_assets`](@ref)), so it is contracted too and the result is square again on the synthetic universe.

# Algorithm

The method that Julia selects is the algorithm. A matrix contracts as `transpose(W) * A`, and again as `* W` when `sq`. A three-dimensional array does the same one observation at a time, into a preallocated result.

# Arguments

  - `A`: The values.
  - `W`: Normalised inner weights, assets × synthetic assets.
  - `sq`: Whether the label axis is the asset axis.

# Returns

  - The collapsed values, with the asset axis, and the label axis under `sq`, replaced by the synthetic assets.

# Related

  - [`collapse_asset_panel`](@ref)
  - [`synthetic_asset_weights`](@ref)
  - [`features_are_assets`](@ref)
  - [`TensorPanelField`](@ref)
"""
function collapse_panel_tensor(A::AbstractMatrix, W::MatNum, sq::Bool)
    C = transpose(W) * A
    return sq ? C * W : C
end
function collapse_panel_tensor(A::AbstractArray{<:Any, 3}, W::MatNum, sq::Bool)
    k = size(W, 2)
    nl = sq ? k : size(A, 3)
    C = Array{promote_type(eltype(A), eltype(W))}(undef, size(A, 1), k, nl)
    @inbounds for t in axes(A, 1)
        Ct = transpose(W) * view(A, t, :, :)
        C[t, :, :] = sq ? Ct * W : Ct
    end
    return C
end
"""
    collapse_panel_mask(m::Nothing, W::MatNum) -> nothing
    collapse_panel_mask(m::AbstractVector{Bool}, W::MatNum) -> BitVector
    collapse_panel_mask(m::AbstractMatrix{Bool}, W::MatNum) -> BitMatrix

Collapse one mask onto the synthetic assets, as the support of its convex combination.

A synthetic asset is observed, active or in estimation at an observation when **any** member carrying weight is. So one kernel serves the values and the masks, the mask stays `Bool` by type, and the subset invariant between the estimation mask and the active mask survives with no second check.

# Algorithm

The method that Julia selects is the algorithm. `nothing` stays `nothing`; otherwise the mask collapses through [`collapse_panel_numeric`](@ref) and the result is compared against zero.

# Arguments

  - `m`: The mask, or `nothing`.
  - `W`: Normalised inner weights, assets × synthetic assets.

# Returns

  - The collapsed mask, or `nothing`.

# Related

  - [`collapse_asset_panel`](@ref)
  - [`synthetic_asset_weights`](@ref)
  - [`AssetPanel`](@ref)
"""
function collapse_panel_mask(::Nothing, ::MatNum)
    return nothing
end
function collapse_panel_mask(m::AbstractArray{Bool}, W::MatNum)
    return collapse_panel_numeric(m, W) .> 0
end
"""
    collapse_panel_field(f::NumericPanelField, W, nx, syn) -> NumericPanelField
    collapse_panel_field(f::CategoricalPanelField, W, nx, syn) -> TensorPanelField
    collapse_panel_field(f::TensorPanelField, W, nx, syn) -> TensorPanelField

Collapse one Panel Field onto the synthetic assets a meta-optimiser builds.

The collapse acts **one field at a time** and returns a field, so the collapsed panel is a panel like any other and a selector written for the inner problem resolves on it unchanged.

  - A numeric field stays numeric.
  - A tensor field stays a tensor field of the same name and labels, one label at a time. When its labels are the asset names the contraction is two-sided, its labels are renamed after the synthetic assets and its groups are dropped, so the square case holds one level up.
  - A categorical field becomes a **tensor field of membership fractions**: the same name, the axis `"level"`, the levels as labels, and the convex combination of its one-hot block as values. A one-hot column of the Feature Matrix is a `0`/`1` feature, so its convex combination is the share of the synthetic asset's weight in that level, and the collapsed panel's Feature Matrix equals the collapse of the original panel's. A convex combination of integer codes would mean nothing, and a majority level would lose the fractions and need a tie rule.

# Algorithm

The method that Julia selects is the algorithm. Each kind contracts its own values and its own observed mask, through [`collapse_panel_numeric`](@ref), [`collapse_panel_tensor`](@ref) and [`collapse_panel_mask`](@ref).

# Arguments

  - `f`: The Panel Field.
  - `W`: Normalised inner weights, assets × synthetic assets.
  - `nx`: The carrier's asset names, or `nothing`. Read for the square case alone.
  - `syn`: The synthetic asset names.

# Returns

  - The collapsed Panel Field.

# Related

  - [`collapse_asset_panel`](@ref)
  - [`features_are_assets`](@ref)
  - [`panel_onehot`](@ref)
  - [`AbstractPanelField`](@ref)
"""
function collapse_panel_field(f::NumericPanelField, W::MatNum, ::Any, ::Any)
    return NumericPanelField(; name = f.name, vals = collapse_panel_numeric(f.vals, W),
                             omsk = collapse_panel_mask(f.omsk, W))
end
function collapse_panel_field(f::CategoricalPanelField, W::MatNum, ::Any, ::Any)
    return TensorPanelField(; name = f.name, axis = "level", labels = f.levels,
                            vals = collapse_panel_tensor(panel_onehot(f), W, false),
                            omsk = collapse_categorical_mask(f.omsk, W, length(f.levels)))
end
function collapse_panel_field(f::TensorPanelField, W::MatNum, nx::Option{<:VecStr},
                              syn::VecStr)
    sq = features_are_assets(f, nx)
    return TensorPanelField(; name = f.name, axis = f.axis, labels = sq ? syn : f.labels,
                            groups = sq ? nothing : f.groups,
                            vals = collapse_panel_tensor(f.vals, W, sq),
                            omsk = if isnothing(f.omsk)
                                nothing
                            else
                                collapse_panel_tensor(f.omsk, W, sq) .> 0
                            end)
end
"""
    collapse_categorical_mask(m::Nothing, W::MatNum, nl::Integer) -> nothing
    collapse_categorical_mask(m::AbstractVector{Bool}, W::MatNum, nl::Integer) -> BitMatrix
    collapse_categorical_mask(m::AbstractMatrix{Bool}, W::MatNum, nl::Integer) -> BitArray

Collapse a categorical Panel Field's observed mask onto the tensor field its collapse returns.

The collapsed field carries one label per level, so its mask needs the level axis the categorical mask does not have. The asset mask is collapsed once and then repeated across the levels: a cell was observed or not for the whole label, never per level.

# Algorithm

The method that Julia selects is the algorithm. `nothing` stays `nothing`; otherwise the asset mask collapses through [`collapse_panel_mask`](@ref) and is repeated over `nl` levels.

# Arguments

  - `m`: The categorical field's observed mask, or `nothing`.
  - `W`: Normalised inner weights, assets × synthetic assets.
  - `nl`: Number of levels.

# Returns

  - The collapsed mask over the level axis, or `nothing`.

# Related

  - [`collapse_panel_field`](@ref)
  - [`collapse_panel_mask`](@ref)
  - [`CategoricalPanelField`](@ref)
"""
function collapse_categorical_mask(::Nothing, ::MatNum, ::Integer)
    return nothing
end
function collapse_categorical_mask(m::AbstractVector{Bool}, W::MatNum, nl::Integer)
    return repeat(collapse_panel_mask(m, W), 1, nl)
end
function collapse_categorical_mask(m::AbstractMatrix{Bool}, W::MatNum, nl::Integer)
    return repeat(collapse_panel_mask(m, W), 1, 1, nl)
end
"""
    collapse_asset_panel(pnl::Nothing, wi::MatNum, nx) -> nothing
    collapse_asset_panel(pnl::AssetPanel, wi::MatNum, nx::Option{<:VecStr}) -> AssetPanel

Aggregate an [`AssetPanel`](@ref) onto the synthetic assets a meta-optimiser builds for its outer problem.

A meta-optimiser's outer problem allocates across *synthetic* assets — [`NestedClustered`](@ref)'s clusters, [`Stacking`](@ref)'s inner portfolios — each of which is a weighted combination of the real ones. Every quantity the outer [`ReturnsResult`](@ref) carries has to be re-expressed on that universe, and the panel is no exception: without this collapse the outer optimiser has no panel at all, so a [`FeatureDistance`](@ref) there throws rather than clustering the synthetic universe.

Features are treated as **intensive**, exactly as `iv` and `ivpa` are: the collapse is a convex combination, obtained by pushing the inner weights through [`synthetic_asset_weights`](@ref) first. An un-normalised weighted sum would scale each synthetic asset's feature vector by its gross exposure `sⱼ = Σᵢ|wᵢⱼ|`, inflating it under leverage or shorting. Under the default [`AngularDist`](@ref) the normalisation is a mathematical no-op for a rectangular field — scaling one row of the result leaves every cosine unchanged — but it is *not* one in the square case, where the two-sided product rescales the label axis as well, and it is what keeps the collapse bounded for any `sⱼ > 0`. An extensive feature (a market capitalisation, a headcount) wanting a weighted *sum* is not supported: the divisor depends on the inner solve, so a caller cannot pre-scale their way to one.

## Degenerate synthetic assets

A synthetic asset whose weights are entirely zero has `sⱼ = 0`; [`synthetic_asset_weights`](@ref) leaves the column alone rather than dividing, so the collapse gives that asset a **zero feature vector** instead of throwing. It then lands on the zero-feature-vector convention the distance kernel already implements, matching the zero returns column, `iv` and `ivpa` the same degenerate weights already produce.

# Algorithm

 1. Return `nothing` when the carrier holds no panel.
 2. Normalise the inner weights with [`synthetic_asset_weights`](@ref).
 3. Collapse every Panel Field with [`collapse_panel_field`](@ref).
 4. Collapse both universe masks with [`collapse_panel_mask`](@ref), which keeps them `nothing` for a static panel.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `wi`: Inner weights, assets × synthetic assets.
  - `nx`: The carrier's asset names, or `nothing`. Read for the square case alone.

# Returns

  - `pnl::Option{AssetPanel}`: The Asset Panel on the synthetic universe, or `nothing`.

# Related

  - [`synthetic_asset_weights`](@ref)
  - [`collapse_panel_field`](@ref)
  - [`features_are_assets`](@ref)
  - [`prepare_outer_rd`](@ref)
  - [`FeatureDistance`](@ref)
"""
function collapse_asset_panel(::Nothing, ::MatNum, ::Any)
    return nothing
end
function collapse_asset_panel(pnl::AssetPanel, wi::MatNum, nx::Option{<:VecStr})
    W = synthetic_asset_weights(wi)
    syn = ["_$(k)" for k in 1:size(W, 2)]
    #! A panel with no Panel Field is the ingestion layer's shape. An untyped comprehension
    #! over an empty vector answers a `Vector{Any}`, which the panel's constructor refuses,
    #! so the comprehension is typed: it answers the same vector empty or full.
    pf = AbstractPanelField[collapse_panel_field(f, W, nx, syn) for f in pnl.pf]
    return AssetPanel(; pf = pf, amsk = collapse_panel_mask(pnl.amsk, W),
                      emsk = collapse_panel_mask(pnl.emsk, W))
end
