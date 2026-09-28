"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the fees for a net return calculation from an optimisation result.

A `fees` that the caller gives has priority. Else the function reads the property `fees` of `res`, and a result with no such property gives `nothing`.

The fee of a result is on the universe that the fit solved, and `res.w` is on the universe of the caller. So a consumer that reads the fee with `res.w` reads both through [`result_investable_view`](@ref), and not through this function alone.

# Arguments

  - `res`: The optimisation result, which can have a property `fees`.
  - `fees`: Fees that the caller gives, or `nothing`.

# Returns

  - `fees::Option{<:Fees}`: The fees for the net return calculation, or `nothing` when none is found.

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

Computes the net returns of the weights of an [`OptimisationResult`](@ref).

The weights, the returns and the fee meet on the investable universe of `res`, through [`result_investable_view`](@ref). `res.w` is on the universe of the caller, and `res.fees` is on the universe that the fit solved. So the function views the weights and a matrix `X` of the caller at the Investable Mask of the result. A `fees` of the caller goes through [`investable_fees_view`](@ref), as a fee does at the fit. A `fees` of the caller has priority over `res.fees`.

The method that takes `pr` gives the whole of `pr` to the view, and reads its `X` after the view.

The argument `wd` is the Weight Drift of the window. With `nothing`, the window is read at the constant weights `res.w`. A [`SelfFinancingDrift`](@ref) reads it as the wealth ratio of the drifted holdings, and `obs` names the observations in the message of a wealth that is not positive.

# Algorithm

 1. View the weights, the returns data and the fee on the investable universe of `res` with [`result_investable_view`](@ref).
 2. Return [`calc_net_returns(w, X, fees, wd, obs)`](@ref) of the viewed values.

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
    # `pr` is paired whole, so the result's own prior handed back is known by
    # identity; its matrix alone would be viewed a second time.
    _, w, pr, fees = result_investable_view(res, pr, fees)
    return calc_net_returns(w, pr.X, fees, wd, obs)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Normalises inner weights into the convex weights that collapse the real assets onto the sub-portfolios of a meta-optimiser.

A quantity beside the returns is extensive or intensive. An extensive quantity, such as a return or a benchmark return, collapses as the weighted sum ``\\mathbf{W}^\\intercal \\boldsymbol{x}``. An intensive quantity, such as the rates `rd.iv` and `rd.ivpa`, collapses as a weighted mean. A weighted sum multiplies an intensive quantity by the gross exposure ``s_k``, so a portfolio with short positions or leverage makes the rate larger. These weights make each product a convex combination, so a caller that collapses an intensive quantity passes its weights through this function.

# Mathematical definition

```math
\\begin{align}
s_{k} &= \\sum_{i=1}^{N} \\lvert W_{ik} \\rvert\\,,\\\\
\\tilde{W}_{ik} &= \\begin{cases} \\lvert W_{ik} \\rvert / s_{k} & s_{k} > 0\\,,\\\\ 0 & s_{k} = 0\\,. \\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:W_inner])
  - $(math_dict[:W_tilde_syn])
  - ``s_{k}``: Gross exposure of sub-portfolio ``k``.
  - $(math_dict[:N])

Each column of ``\\tilde{\\mathbf{W}}`` sums to one, except a column of zeros, which stays a column of zeros.

# Arguments

  - `w`: The inner weights. A vector collapses onto one sub-portfolio. A matrix, `assets × sub-portfolios`, collapses each column separately.

# Returns

  - `w`: The normalised weights, with the shape of the input.

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

Collapses the values of one numeric Panel Field onto the sub-portfolios, as a convex combination.

`W` is the normalised weight matrix that [`synthetic_asset_weights`](@ref) returns, `assets × sub-portfolios`. A static field holds one value for each asset, and a field that changes over time holds `observations × assets`. So the asset axis is the one axis that both forms contract. [`collapse_asset_panel`](@ref) states the mathematics.

# Algorithm

The method that Julia selects is the algorithm. A vector contracts as `transpose(W) * A`, and a matrix as `A * W`.

# Arguments

  - `A`: The values, `assets` or `observations × assets`.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.

# Returns

  - The collapsed values, `sub-portfolios` or `observations × sub-portfolios`.

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

Collapses the values of one tensor Panel Field onto the sub-portfolios, as a convex combination.

A tensor array is `assets × labels` when it is static, and `observations × assets × labels` when it changes over time. The function always contracts the asset axis. When `sq` is `true`, the label axis is the asset axis, see [`features_are_assets`](@ref). Then the function contracts the label axis too, and the answer is square on the sub-portfolios. [`collapse_asset_panel`](@ref) states the mathematics.

# Algorithm

The method that Julia selects is the algorithm.

 1. A matrix contracts as `transpose(W) * A`. When `sq` is `true`, the answer is multiplied by `W` on the right.
 2. A three-dimensional array does the same for each observation, into a result that the method makes before the loop.

# Arguments

  - `A`: The values.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.
  - `sq`: `true` when the label axis is the asset axis.

# Returns

  - The collapsed values. The sub-portfolios replace the asset axis, and also the label axis when `sq` is `true`.

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

Collapses one mask onto the sub-portfolios, as the support of its convex combination.

A sub-portfolio is observed, active or in estimation at an observation when one member with weight is. So the values and the masks use one kernel, and the mask stays `Bool` by its type. The estimation mask is a subset of the active mask, and the collapse keeps that relation with no second check. [`collapse_asset_panel`](@ref) states the mathematics.

# Algorithm

The method that Julia selects is the algorithm. `nothing` stays `nothing`. Else the mask collapses through [`collapse_panel_numeric`](@ref), and an entry of the answer is `true` where the collapsed value is larger than zero.

# Arguments

  - `m`: The mask, or `nothing`.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.

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

Collapses one Panel Field onto the sub-portfolios of a meta-optimiser.

The collapse acts on one field at a time and returns a field. So the collapsed panel is an ordinary panel, and a selector that the caller wrote for the inner problem resolves on it with no change.

  - A numeric field stays numeric.
  - A tensor field stays a tensor field with the same name and labels. When its labels are the asset names, the contraction acts on both axes. Then the sub-portfolios name the labels, and the groups are dropped, so the field is square on the sub-portfolios too.
  - A categorical field becomes a tensor field of membership fractions. It has the same name, the axis `"level"`, the levels as labels, and the convex combination of its one-hot block as values. A one-hot column of the Feature Matrix holds `0` and `1`. So its convex combination is the fraction of the weight of the sub-portfolio in that level, and the Feature Matrix of the collapsed panel is the collapse of the Feature Matrix of the panel. A convex combination of integer codes has no meaning, and a majority level loses the fractions and needs a rule for ties.

# Algorithm

The method that Julia selects is the algorithm. Each kind contracts its values and its observed mask, through [`collapse_panel_numeric`](@ref), [`collapse_panel_tensor`](@ref), [`collapse_panel_mask`](@ref) and [`collapse_categorical_mask`](@ref).

# Arguments

  - `f`: The Panel Field.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.
  - `nx`: The asset names of the returns data, or `nothing`. Only a tensor field reads it, to find the square case.
  - `syn`: The names of the sub-portfolios.

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

Collapses the observed mask of a categorical Panel Field onto the tensor field that its collapse returns.

The collapsed field has one label for each level, so its mask needs a level axis, which the categorical mask does not have. A cell is observed or not for the whole label, never for one level. So the function collapses the asset mask once, and repeats it for each level.

# Algorithm

The method that Julia selects is the algorithm. `nothing` stays `nothing`. Else the asset mask collapses through [`collapse_panel_mask`](@ref), and the answer repeats it `nl` times on a new last axis.

# Arguments

  - `m`: The observed mask of the categorical field, `assets` or `observations × assets`, or `nothing`.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.
  - `nl`: The number of levels.

# Returns

  - The collapsed mask, `sub-portfolios × levels` or `observations × sub-portfolios × levels`, or `nothing`.

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

Collapses an [`AssetPanel`](@ref) onto the sub-portfolios that a meta-optimiser builds for its outer problem.

The outer problem of a meta-optimiser allocates over sub-portfolios, which are the clusters of [`NestedClustered`](@ref) and the inner portfolios of [`Stacking`](@ref). Each of them is a weighted combination of the real assets. The outer [`ReturnsResult`](@ref) must state each quantity on that universe, and the panel too. With no collapse, the outer optimiser has no panel, and a [`FeatureDistance`](@ref) in it throws and does not cluster.

A feature is intensive, as `iv` and `ivpa` are. So the collapse is a convex combination, with the weights of [`synthetic_asset_weights`](@ref). A weighted sum with no normalisation multiplies the feature vector of a sub-portfolio by its gross exposure, which makes it larger under leverage or short positions. Under the default [`AngularDist`](@ref), the normalisation does not change a rectangular field, because a scale of one row changes no cosine. In the square case it does change the field, because the product on both sides scales the label axis too. The normalisation also keeps the collapse bounded for every gross exposure larger than zero.

The collapse does not support an extensive feature, such as a market capitalisation or a headcount, that needs a weighted sum. The divisor comes from the inner solve, so a caller cannot scale the feature before the solve.

A sub-portfolio whose weights are all zero has a gross exposure of zero. [`synthetic_asset_weights`](@ref) keeps its column of zeros and does not divide, so the collapse gives that sub-portfolio a feature vector of zeros and does not throw. The distance kernel already handles a feature vector of zeros, and the same weights give a returns column of zeros, and zero `iv` and `ivpa`.

# Mathematical definition

At each observation of a field that changes over time:

```math
\\begin{align}
\\boldsymbol{a}^{o} &= \\tilde{\\mathbf{W}}^\\intercal \\boldsymbol{a}\\,,\\\\
\\mathbf{C}^{o} &= \\tilde{\\mathbf{W}}^\\intercal \\mathbf{C}\\,,\\\\
\\mathbf{S}^{o} &= \\tilde{\\mathbf{W}}^\\intercal \\mathbf{S} \\tilde{\\mathbf{W}}\\,,\\\\
\\mathbf{F}^{o} &= \\tilde{\\mathbf{W}}^\\intercal \\mathbf{H}\\,,\\\\
m^{o}_{k} &= \\mathbf{1}\\left[\\left(\\tilde{\\mathbf{W}}^\\intercal \\boldsymbol{m}\\right)_{k} > 0\\right]\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{a}``, ``\\boldsymbol{a}^{o}``: Values of a numeric field, one entry for each asset and one for each sub-portfolio.
  - ``\\mathbf{C}``, ``\\mathbf{C}^{o}``: Values of a tensor field whose labels are not the asset names, `assets × labels` and `sub-portfolios × labels`.
  - ``\\mathbf{S}``, ``\\mathbf{S}^{o}``: Values of a tensor field whose labels are the asset names, `assets × assets` and `sub-portfolios × sub-portfolios`.
  - ``\\mathbf{H}``: One-hot matrix of a categorical field, `assets × levels`.
  - ``\\mathbf{F}^{o}``: Membership fractions, `sub-portfolios × levels`.
  - ``\\boldsymbol{m}``, ``m^{o}_{k}``: A mask, one entry for each asset, and its entry for sub-portfolio ``k``.
  - ``\\mathbf{1}[\\cdot]``: Indicator, ``1`` when the condition holds and ``0`` otherwise.
  - $(math_dict[:W_tilde_syn])

Each column of ``\\tilde{\\mathbf{W}}`` sums to one or is a column of zeros. So each collapsed value is a convex combination of the values of the members, or zero. A row of ``\\mathbf{F}^{o}`` sums to one when the sub-portfolio has weight and every member has a level.

# Algorithm

 1. Return `nothing` when `pnl` is `nothing`.
 2. Normalise the inner weights with [`synthetic_asset_weights`](@ref), giving `W`.
 3. Name the sub-portfolios `"_1"`, `"_2"`, …, giving `syn`.
 4. Collapse each Panel Field with [`collapse_panel_field`](@ref).
 5. Collapse the active mask and the estimation mask with [`collapse_panel_mask`](@ref). They stay `nothing` for a static panel.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `wi`: The inner weights, `assets × sub-portfolios`.
  - `nx`: The asset names of the returns data, or `nothing`. Only a tensor field reads it, to find the square case.

# Returns

  - `pnl::Option{AssetPanel}`: The Asset Panel on the sub-portfolios, or `nothing`.

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
