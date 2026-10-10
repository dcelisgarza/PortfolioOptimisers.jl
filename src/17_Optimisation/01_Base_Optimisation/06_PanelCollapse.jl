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
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that state how the panel collapse of a meta-optimiser weighs a member at an observation where the member is outside the universe.

A sub-portfolio of a meta-optimiser holds each member with one weight for the whole history, and a time-varying Asset Panel lists and delists the members. At an observation where a member is inactive, the panel stores a finite value with no meaning. So the collapse reads only the members that are active there, and the rule states what the weight of an inactive member becomes. The rule is the `pcol` field of [`NestedClustered`](@ref) and of [`Stacking`](@ref), and [`collapse_asset_panel`](@ref) states the mathematics. A static panel has no inactive member, so every rule gives the collapse of all the members.

# Interfaces

To add a rule, subtype `AbstractPanelCollapseAlgorithm` and implement the method below. The collapse sets the value of each inactive member to zero before it contracts, so a rule states only the divisor of each collapsed value.

## `active_weight_divisor`

  - `active_weight_divisor(alg::MyRule, m::AbstractMatrix{Bool}, W::MatNum) -> Option{<:MatNum}`: The divisor of each sub-portfolio at each observation, or `nothing` for no division.

### Arguments

  - `alg`: The concrete subtype instance.
  - `m`: The mask of the cells that the collapse reads, `observations × members`: a member is read where it is active and its cell holds data. A member is an asset, and a pair of assets for a tensor field whose labels are the asset names.
  - `W`: The normalised inner weights, `members × sub-portfolios`, or `kron(W, W)` for the pairs.

### Returns

  - The divisors, `observations × sub-portfolios`, or `nothing`. A divisor must not be zero.

# Related

  - [`RenormaliseActive`](@ref)
  - [`InactiveAsCash`](@ref)
  - [`collapse_asset_panel`](@ref)
  - [`active_weight_divisor`](@ref)
"""
abstract type AbstractPanelCollapseAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Collapses an Asset Panel over the members that are active at each observation, and divides by their weight. It is the default rule of the panel collapse of [`NestedClustered`](@ref) and of [`Stacking`](@ref).

The collapse already divides out the gross exposure of a sub-portfolio: its leverage, its short positions and the cash that a budget leaves. The weight of an inactive member is one more part of the portfolio outside the universe, and this rule divides it out in the same way. So each collapsed value stays a convex combination of the values of the active members. A sub-portfolio whose weighted members are all active divides by one, so its collapse does not change.

# Related

  - [`AbstractPanelCollapseAlgorithm`](@ref)
  - [`InactiveAsCash`](@ref)
  - [`collapse_asset_panel`](@ref)
"""
struct RenormaliseActive <: AbstractPanelCollapseAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Collapses an Asset Panel over the members that are active at each observation, and reads the weight of an inactive member as cash with a zero feature.

The outer returns of a meta-optimiser are read off the returns of the prior, which hold a zero at an inactive cell of an admitted asset. So the weight of that asset sits in cash. For an additive feature, such as a loading, this rule gives the feature of the outer returns column. For a characteristic, such as a ratio or the fraction of an industry, a zero is not a value of cash, and the collapsed value moves toward zero as the active weight falls.

# Related

  - [`AbstractPanelCollapseAlgorithm`](@ref)
  - [`RenormaliseActive`](@ref)
  - [`collapse_asset_panel`](@ref)
"""
struct InactiveAsCash <: AbstractPanelCollapseAlgorithm end
"""
    active_weight_divisor(alg::AbstractPanelCollapseAlgorithm, m::Nothing, W::MatNum) -> nothing
    active_weight_divisor(alg::InactiveAsCash, m::AbstractMatrix{Bool}, W::MatNum) -> nothing
    active_weight_divisor(alg::RenormaliseActive, m::AbstractMatrix{Bool}, W::MatNum) -> Matrix

Returns the divisor of each collapsed value of a time-varying Asset Panel under a rule of the panel collapse, or `nothing` for no division.

[`RenormaliseActive`](@ref) divides by the weight of the active members. The divisor is one where a sub-portfolio has no active member, because its active values are all zero, and where it has no inactive member with weight, because its weights already sum to one. So the second case keeps the collapse of all the members bit for bit, where a division by a sum that rounds to one would change the last bit. [`InactiveAsCash`](@ref) divides by nothing. A static panel, `m = nothing`, has no inactive member, and no rule divides.

# Mathematical definition

```math
\\begin{align}
d_{tk} &= \\begin{cases} \\sum_{i=1}^{N} \\tilde{W}_{ik} m_{ti} & \\sum_{i=1}^{N} \\tilde{W}_{ik} m_{ti} > 0 \\text{ and } \\sum_{i=1}^{N} \\tilde{W}_{ik} (1 - m_{ti}) > 0\\,,\\\\ 1 & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``d_{tk}``: Divisor of sub-portfolio ``k`` at observation ``t``.
  - $(math_dict[:m_active_panel])
  - $(math_dict[:W_tilde_syn])
  - $(math_dict[:N])

# Arguments

  - `alg`: The rule of the panel collapse.
  - `m`: The active mask of the panel, `observations × assets`, or `nothing` for a static panel.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.

# Returns

  - The divisors, `observations × sub-portfolios`, or `nothing`.

# Related

  - [`AbstractPanelCollapseAlgorithm`](@ref)
  - [`collapse_asset_panel`](@ref)
"""
function active_weight_divisor(::AbstractPanelCollapseAlgorithm, ::Nothing, ::MatNum)
    return nothing
end
function active_weight_divisor(::InactiveAsCash, ::AbstractMatrix{Bool}, ::MatNum)
    return nothing
end
function active_weight_divisor(::RenormaliseActive, m::AbstractMatrix{Bool}, W::MatNum)
    a = m * W
    u = (.!m) * W
    return map((x, y) -> iszero(x) || iszero(y) ? one(x) : x, a, u)
end
"""
    active_divide!(C::AbstractArray, d::Nothing, sq::Bool = false) -> C
    active_divide!(C::AbstractArray, d::MatNum, sq::Bool = false) -> C

Divides the collapsed values of a time-varying Panel Field by the divisors of [`active_weight_divisor`](@ref), in place.

The divisor of row `t` and sub-portfolio `k` divides every label of that cell. When `sq` is `true`, the label axis is the sub-portfolio axis too, so the cell of labels `k` and `l` also divides by the divisor of `l`.

# Algorithm

The method that Julia selects is the algorithm. `nothing` leaves `C` as it is. Else `C` divides by `d` along its first two axes, and, when `sq` is `true`, along its first and third axes.

# Arguments

  - `C`: The collapsed values, `observations × sub-portfolios`, or `observations × sub-portfolios × labels`.
  - `d`: The divisors, `observations × sub-portfolios`, or `nothing`.
  - `sq`: `true` when the label axis is the sub-portfolio axis.

# Returns

  - `C`, divided.

# Related

  - [`active_weight_divisor`](@ref)
  - [`collapse_panel_numeric`](@ref)
  - [`collapse_panel_tensor`](@ref)
"""
function active_divide!(C::AbstractArray, ::Nothing, ::Bool = false)
    return C
end
function active_divide!(C::AbstractArray, d::MatNum, sq::Bool = false)
    C ./= d
    if sq
        C ./= reshape(d, size(d, 1), 1, size(d, 2))
    end
    return C
end
"""
    collapse_panel_numeric(A::AbstractVector, W::MatNum) -> Vector
    collapse_panel_numeric(A::AbstractMatrix, W::MatNum) -> Matrix
    collapse_panel_numeric(A::AbstractArray, W::MatNum, m::Nothing, d) -> Array
    collapse_panel_numeric(A::AbstractMatrix, W::MatNum, m::AbstractMatrix{Bool}, d) -> Matrix

Collapses the values of one numeric Panel Field onto the sub-portfolios.

`W` is the normalised weight matrix that [`synthetic_asset_weights`](@ref) returns, `assets × sub-portfolios`. A static field holds one value for each asset, and a field that changes over time holds `observations × assets`. So the asset axis is the one axis that both forms contract. The two methods that take `m` read the active mask: a static panel passes `nothing`, contracts every member and has no divisor to read, and a time-varying panel contracts the active members of each row alone, then divides by `d`. [`collapse_asset_panel`](@ref) states the mathematics.

# Algorithm

The method that Julia selects is the algorithm.

 1. A vector contracts as `transpose(W) * A`, and a matrix as `A * W`.
 2. With a mask, the value of each inactive cell becomes zero, the matrix contracts, and the answer divides by `d` through [`active_divide!`](@ref).

# Arguments

  - `A`: The values, `assets` or `observations × assets`.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.
  - `m`: The active mask of the panel, `observations × assets`, or `nothing` for a static panel.
  - `d`: The divisors of [`active_weight_divisor`](@ref), or `nothing`.

# Returns

  - The collapsed values, `sub-portfolios` or `observations × sub-portfolios`.

# Related

  - [`collapse_asset_panel`](@ref)
  - [`synthetic_asset_weights`](@ref)
  - [`active_weight_divisor`](@ref)
  - [`NumericPanelField`](@ref)
"""
function collapse_panel_numeric(A::AbstractVector, W::MatNum)
    return transpose(W) * A
end
function collapse_panel_numeric(A::AbstractMatrix, W::MatNum)
    return A * W
end
function collapse_panel_numeric(A::AbstractArray, W::MatNum, ::Nothing, ::Any)
    return collapse_panel_numeric(A, W)
end
function collapse_panel_numeric(A::AbstractMatrix, W::MatNum, m::AbstractMatrix{Bool},
                                d::Option{<:MatNum})
    return active_divide!(collapse_panel_numeric(ifelse.(m, A, zero(eltype(A))), W), d)
end
"""
    collapse_panel_tensor(A::AbstractMatrix, W::MatNum, sq::Bool) -> Matrix
    collapse_panel_tensor(A::AbstractArray{<:Any, 3}, W::MatNum, sq::Bool) -> Array
    collapse_panel_tensor(A::AbstractArray, W::MatNum, sq::Bool, m::Nothing, d) -> Array
    collapse_panel_tensor(A::AbstractArray{<:Any, 3}, W::MatNum, sq::Bool, m::AbstractMatrix{Bool}, d) -> Array

Collapses the values of one tensor Panel Field onto the sub-portfolios.

A tensor array is `assets × labels` when it is static, and `observations × assets × labels` when it changes over time. The function always contracts the asset axis. When `sq` is `true`, the label axis is the asset axis, see [`features_are_assets`](@ref). Then the function contracts the label axis too, and the answer is square on the sub-portfolios. The two methods that take `m` read the active mask, as [`collapse_panel_numeric`](@ref) does, and a square field restricts both axes to the active members. [`collapse_asset_panel`](@ref) states the mathematics.

# Algorithm

The method that Julia selects is the algorithm.

 1. A matrix contracts as `transpose(W) * A`. When `sq` is `true`, the answer is multiplied by `W` on the right.
 2. A three-dimensional array does the same for each observation, into a result that the method makes before the loop.
 3. With a mask, the value of each inactive cell becomes zero, on the asset axis and, when `sq` is `true`, on the label axis. The array contracts by step 2, and the answer divides by `d` through [`active_divide!`](@ref).

# Arguments

  - `A`: The values.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.
  - `sq`: `true` when the label axis is the asset axis.
  - `m`: The active mask of the panel, `observations × assets`, or `nothing` for a static panel.
  - `d`: The divisors of [`active_weight_divisor`](@ref), or `nothing`.

# Returns

  - The collapsed values. The sub-portfolios replace the asset axis, and also the label axis when `sq` is `true`.

# Related

  - [`collapse_asset_panel`](@ref)
  - [`synthetic_asset_weights`](@ref)
  - [`active_weight_divisor`](@ref)
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
function collapse_panel_tensor(A::AbstractArray, W::MatNum, sq::Bool, ::Nothing, ::Any)
    return collapse_panel_tensor(A, W, sq)
end
function collapse_panel_tensor(A::AbstractArray{<:Any, 3}, W::MatNum, sq::Bool,
                               m::AbstractMatrix{Bool}, d::Option{<:MatNum})
    #! A square field holds the members on its label axis too, so an inactive member leaves
    #! both axes.
    ma = sq ? m .& reshape(m, size(m, 1), 1, size(m, 2)) : m
    return active_divide!(collapse_panel_tensor(ifelse.(ma, A, zero(eltype(A))), W, sq), d,
                          sq)
end
"""
    collapse_panel_mask(m::Nothing, W::MatNum) -> nothing
    collapse_panel_mask(m::AbstractArray{Bool}, W::MatNum) -> BitArray
    collapse_panel_mask(o::Nothing, W::MatNum, m) -> nothing
    collapse_panel_mask(o::AbstractArray{Bool}, W::MatNum, m::Nothing) -> BitArray
    collapse_panel_mask(o::AbstractMatrix{Bool}, W::MatNum, m::AbstractMatrix{Bool}) -> BitMatrix

Collapses one mask onto the sub-portfolios, as the support of its combination.

A sub-portfolio is active or in estimation at an observation when one member with weight is. So the values and the masks use one kernel, and the mask stays `Bool` by its type. The estimation mask is a subset of the active mask, and the collapse keeps that relation with no second check. The two methods with two arguments collapse the universe masks. The methods with three arguments collapse a mask `o` of the cells of a Panel Field, its observed mask or its data mask: a sub-portfolio is observed, or holds data, where one member with weight is active and is observed, or holds data. So a member that is inactive there does not count, as its value does not. [`collapse_asset_panel`](@ref) states the mathematics.

# Algorithm

The method that Julia selects is the algorithm. `nothing` stays `nothing`. Else the mask collapses through [`collapse_panel_numeric`](@ref), with the active mask `m` when it is given, and an entry of the answer is `true` where the collapsed value is larger than zero.

# Arguments

  - `m`: A universe mask, or `nothing`, in the methods with two arguments. The active mask of the panel, or `nothing` for a static panel, in the methods with three arguments.
  - `o`: The observed mask or the data mask of a Panel Field, or `nothing`.
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
function collapse_panel_mask(::Nothing, ::MatNum, ::Any)
    return nothing
end
function collapse_panel_mask(o::AbstractArray{Bool}, W::MatNum, ::Nothing)
    return collapse_panel_mask(o, W)
end
function collapse_panel_mask(o::AbstractMatrix{Bool}, W::MatNum, m::AbstractMatrix{Bool})
    return collapse_panel_numeric(o, W, m, nothing) .> 0
end
"""
    collapse_panel_field(f::NumericPanelField, W, nx, syn, m, alg) -> NumericPanelField
    collapse_panel_field(f::CategoricalPanelField, W, nx, syn, m, alg) -> TensorPanelField
    collapse_panel_field(f::TensorPanelField, W, nx, syn, m, alg) -> TensorPanelField

Collapses one Panel Field onto the sub-portfolios of a meta-optimiser.

The collapse acts on one field at a time and returns a field. So the collapsed panel is an ordinary panel, and a selector that the caller wrote for the inner problem resolves on it with no change.

  - A numeric field stays numeric.
  - A tensor field stays a tensor field with the same name and labels. When its labels are the asset names, the contraction acts on both axes. Then the sub-portfolios name the labels, and the groups are dropped, so the field is square on the sub-portfolios too.
  - A categorical field becomes a tensor field of membership fractions. It has the same name, the axis `"level"`, the levels as labels, and the combination of its one-hot block as values. A one-hot column of the Feature Matrix holds `0` and `1`. So its combination is the fraction of the weight of the sub-portfolio in that level, and the Feature Matrix of the collapsed panel is the collapse of the Feature Matrix of the panel. A combination of integer codes has no meaning, and a majority level loses the fractions and needs a rule for ties.

A field that a time-varying panel lifts from a static input, see [`RepeatedLeading`](@ref), holds the same value at every row, and its collapse reads the active members of each row. So the collapsed field changes over time where the active members change.

The collapse reads a member only where it is active and its cell holds data: a value that the raw input carried, or that a fill policy wrote. A cell that holds the placeholder of a blank that no fill reached holds no data, so the rule `alg` divides by the weight of the active members that hold data. The collapsed value is then the same for every placeholder. A collapsed cell is observed where an active member with weight is observed, and it holds a placeholder where no active member with weight holds data.

# Algorithm

The method that Julia selects is the algorithm. Each kind contracts its values over the active cells that hold data, through [`panel_data_mask`](@ref), [`read_cells`](@ref), [`collapse_read_numeric`](@ref), [`collapse_read_onehot`](@ref) and [`collapse_read_tensor`](@ref). It contracts its observed mask and its data mask through [`collapse_panel_mask`](@ref), [`collapse_panel_tensor`](@ref) and [`collapse_categorical_mask`](@ref). Its placeholder mask is the negation of the collapsed data mask, through [`collapse_placeholder_mask`](@ref).

# Arguments

  - `f`: The Panel Field.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.
  - `nx`: The asset names of the returns data, or `nothing`. Only a tensor field reads it, to find the square case.
  - `syn`: The names of the sub-portfolios.
  - `m`: The active mask of the panel, or `nothing` for a static panel.
  - `alg`: The rule of the panel collapse.

# Returns

  - The collapsed Panel Field.

# Related

  - [`collapse_asset_panel`](@ref)
  - [`features_are_assets`](@ref)
  - [`panel_onehot`](@ref)
  - [`AbstractPanelField`](@ref)
"""
function collapse_panel_field(f::NumericPanelField, W::MatNum, ::Any, ::Any,
                              m::Option{<:AbstractMatrix{Bool}},
                              alg::AbstractPanelCollapseAlgorithm)
    d = panel_data_mask(f)
    return NumericPanelField(; name = f.name,
                             vals = collapse_read_numeric(f.vals, W, read_cells(m, d), alg),
                             omsk = collapse_panel_mask(f.omsk, W, m),
                             pmsk = collapse_placeholder_mask(collapse_panel_mask(d, W, m)))
end
function collapse_panel_field(f::CategoricalPanelField, W::MatNum, ::Any, ::Any,
                              m::Option{<:AbstractMatrix{Bool}},
                              alg::AbstractPanelCollapseAlgorithm)
    d = panel_data_mask(f)
    nl = length(f.levels)
    return TensorPanelField(; name = f.name, axis = "level", labels = f.levels,
                            vals = collapse_read_onehot(panel_onehot(f), W,
                                                        read_cells(m, d), alg),
                            omsk = collapse_categorical_mask(f.omsk, W, nl, m),
                            pmsk = collapse_placeholder_mask(collapse_categorical_mask(d, W,
                                                                                       nl,
                                                                                       m)))
end
function collapse_panel_field(f::TensorPanelField, W::MatNum, nx::Option{<:VecStr},
                              syn::VecStr, m::Option{<:AbstractMatrix{Bool}},
                              alg::AbstractPanelCollapseAlgorithm)
    sq = features_are_assets(f, nx)
    d = panel_data_mask(f)
    return TensorPanelField(; name = f.name, axis = f.axis, labels = sq ? syn : f.labels,
                            groups = sq ? nothing : f.groups,
                            vals = collapse_read_tensor(f.vals, W, sq, m, d, alg),
                            omsk = collapse_tensor_mask(f.omsk, W, sq, m),
                            pmsk = collapse_placeholder_mask(collapse_tensor_mask(d, W, sq,
                                                                                  m)))
end
"""
    collapse_tensor_mask(o::Nothing, W::MatNum, sq::Bool, m) -> nothing
    collapse_tensor_mask(o::AbstractArray{Bool}, W::MatNum, sq::Bool, m) -> BitArray

Collapses the observed mask or the data mask `o` of a tensor Panel Field, as the support of its combination through [`collapse_panel_tensor`](@ref). `nothing` stays `nothing`.

# Related

  - [`collapse_panel_field`](@ref)
  - [`collapse_panel_mask`](@ref)
"""
function collapse_tensor_mask(::Nothing, ::MatNum, ::Bool, ::Any)
    return nothing
end
function collapse_tensor_mask(o::AbstractArray{Bool}, W::MatNum, sq::Bool,
                              m::Option{<:AbstractMatrix{Bool}})
    return collapse_panel_tensor(o, W, sq, m, nothing) .> 0
end
"""
    collapse_placeholder_mask(d::Nothing) -> nothing
    collapse_placeholder_mask(d::AbstractArray{Bool}) -> BitArray

Give the placeholder mask of a collapsed Panel Field from its collapsed data mask `d`: a collapsed cell holds a placeholder where no active member with weight holds data. `nothing` stays `nothing`, because a Panel Field that holds no placeholder collapses to one that holds none.

# Related

  - [`collapse_panel_field`](@ref)
  - [`panel_data_mask`](@ref)
"""
function collapse_placeholder_mask(::Nothing)
    return nothing
end
function collapse_placeholder_mask(d::AbstractArray{Bool})
    return .!d
end
"""
    read_cells(m, o)

Give the mask of the cells that the panel collapse reads for a numeric or a categorical Panel Field: a member is read where it is active and its cell holds data.

A cell that holds the placeholder of a blank that no fill reached holds no data, so the collapse reads it as it reads an inactive member. `nothing` reads every cell. A static panel has no active mask, so its mask is the data mask, and `nothing` when every cell holds data. Then the collapse keeps the static contraction bit for bit.

# Arguments

  - `m`: The active mask of the panel, `observations × assets`, or `nothing` for a static panel.
  - `o`: The data mask of the Panel Field, see [`panel_data_mask`](@ref), or `nothing`.

# Returns

  - The mask, `observations × assets` or `assets`, or `nothing`.

# Related

  - [`collapse_panel_field`](@ref)
  - [`collapse_read_numeric`](@ref)
"""
function read_cells(m::Option{<:AbstractMatrix{Bool}}, ::Nothing)
    return m
end
function read_cells(::Nothing, o::AbstractVector{Bool})
    return all(o) ? nothing : o
end
function read_cells(m::AbstractMatrix{Bool}, o::AbstractMatrix{Bool})
    return m .& o
end
"""
    collapse_read_numeric(A::AbstractArray, W::MatNum, r, alg::AbstractPanelCollapseAlgorithm) -> Array

Collapse the values of a numeric Panel Field over the cells that the mask `r` of [`read_cells`](@ref) marks.

The rule `alg` divides by the weight of the read members, through [`active_weight_divisor`](@ref) on `r`. Where every cell holds data, `r` is the active mask, and the collapse is the one that reads the active members alone. A static mask is a window of one row: the values and the mask take an observation axis of length one, collapse as a time-varying field does, and drop it after.

# Arguments

  - `A`: The values, `assets` or `observations × assets`.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.
  - `r`: The mask of the read cells, or `nothing`.
  - `alg`: The rule of the panel collapse.

# Returns

  - The collapsed values.

# Related

  - [`collapse_panel_field`](@ref)
  - [`collapse_panel_numeric`](@ref)
  - [`collapse_read_onehot`](@ref)
"""
function collapse_read_numeric(A::AbstractArray, W::MatNum, ::Nothing,
                               ::AbstractPanelCollapseAlgorithm)
    return collapse_panel_numeric(A, W)
end
function collapse_read_numeric(A::AbstractMatrix, W::MatNum, r::AbstractMatrix{Bool},
                               alg::AbstractPanelCollapseAlgorithm)
    return collapse_panel_numeric(A, W, r, active_weight_divisor(alg, r, W))
end
function collapse_read_numeric(A::AbstractVector, W::MatNum, r::AbstractVector{Bool},
                               alg::AbstractPanelCollapseAlgorithm)
    return vec(collapse_read_numeric(reshape(A, 1, :), W, reshape(r, 1, :), alg))
end
"""
    collapse_read_onehot(H::AbstractArray, W::MatNum, r, alg::AbstractPanelCollapseAlgorithm) -> Array

Collapse the one-hot block `H` of a categorical Panel Field, `assets × levels` or `observations × assets × levels`, over the cells that the mask `r` of [`read_cells`](@ref) marks. A cell holds data or not for every level at once, so one mask serves each level. It is [`collapse_read_numeric`](@ref) with a trailing level axis, through [`collapse_panel_tensor`](@ref).

# Related

  - [`collapse_panel_field`](@ref)
  - [`collapse_read_numeric`](@ref)
  - [`collapse_panel_tensor`](@ref)
"""
function collapse_read_onehot(H::AbstractMatrix, W::MatNum, ::Nothing,
                              ::AbstractPanelCollapseAlgorithm)
    return collapse_panel_tensor(H, W, false)
end
function collapse_read_onehot(H::AbstractArray{<:Any, 3}, W::MatNum,
                              r::AbstractMatrix{Bool}, alg::AbstractPanelCollapseAlgorithm)
    return collapse_panel_tensor(H, W, false, r, active_weight_divisor(alg, r, W))
end
function collapse_read_onehot(H::AbstractMatrix, W::MatNum, r::AbstractVector{Bool},
                              alg::AbstractPanelCollapseAlgorithm)
    C = collapse_read_onehot(reshape(H, 1, size(H)...), W, reshape(r, 1, :), alg)
    return dropdims(C; dims = 1)
end
"""
    collapse_read_tensor(A::AbstractArray, W::MatNum, sq::Bool, m, o, alg::AbstractPanelCollapseAlgorithm) -> Array

Collapse the values of a tensor Panel Field over the cells that are active and hold data.

The data mask `o` of a tensor Panel Field has a label axis, so a member can hold data at one label and not at another. Where each active member holds data at every label, the cells to read are the active members, and [`collapse_panel_tensor`](@ref) collapses the field as it does with no data mask. Otherwise each collapsed value is a weighted mean over its own read cells.

  - A rectangular field collapses each label as a numeric field, through [`collapse_read_numeric`](@ref), on the active members that hold data at that label.
  - A square field reads a pair of members, the asset and the label, with the weight ``\\tilde{W}_{ik} \\tilde{W}_{jl}``. A pair is read where both are active and the cell holds data. The pairs collapse as the members of a numeric field, with the weights `kron(W, W)`, so the rule divides by the weight of the read pairs. This keeps a symmetric field symmetric, and a separable mask gives the collapse of the active members.

A static field takes an observation axis of length one, and drops it after.

# Arguments

  - `A`: The values, `assets × labels` or `observations × assets × labels`.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.
  - `sq`: `true` when the label axis is the asset axis.
  - `m`: The active mask of the panel, or `nothing` for a static panel.
  - `o`: The data mask of the Panel Field, with the shape of `A`, or `nothing`. See [`panel_data_mask`](@ref).
  - `alg`: The rule of the panel collapse.

# Returns

  - The collapsed values.

# Related

  - [`collapse_panel_field`](@ref)
  - [`collapse_panel_tensor`](@ref)
  - [`active_weight_divisor`](@ref)
"""
function collapse_read_tensor(A::AbstractArray, W::MatNum, sq::Bool,
                              m::Option{<:AbstractMatrix{Bool}}, ::Nothing,
                              alg::AbstractPanelCollapseAlgorithm)
    return collapse_panel_tensor(A, W, sq, m, active_weight_divisor(alg, m, W))
end
function collapse_read_tensor(A::AbstractMatrix, W::MatNum, sq::Bool, ::Nothing,
                              o::AbstractMatrix{Bool}, alg::AbstractPanelCollapseAlgorithm)
    if all(o)
        return collapse_panel_tensor(A, W, sq)
    end
    C = collapse_read_tensor(reshape(A, 1, size(A)...), W, sq, trues(1, size(A, 1)),
                             reshape(o, 1, size(o)...), alg)
    return dropdims(C; dims = 1)
end
function collapse_read_tensor(A::AbstractArray{<:Any, 3}, W::MatNum, sq::Bool,
                              m::AbstractMatrix{Bool}, o::AbstractArray{Bool, 3},
                              alg::AbstractPanelCollapseAlgorithm)
    ma = sq ? m .& reshape(m, size(m, 1), 1, size(m, 2)) : m
    #! A cell of `o` sits at the leading indices of `ma`, which holds no label axis on a
    #! rectangular field.
    if all(k -> o[k] || !ma[CartesianIndex(Tuple(k)[1:ndims(ma)])], CartesianIndices(o))
        return collapse_panel_tensor(A, W, sq, m, active_weight_divisor(alg, m, W))
    end
    T, N, L = size(A)
    if sq
        r = reshape(ma .& o, T, N * L)
        W2 = kron(W, W)
        k = size(W, 2)
        C = collapse_panel_numeric(reshape(A, T, N * L), W2, r,
                                   active_weight_divisor(alg, r, W2))
        return reshape(C, T, k, k)
    end
    return stack((collapse_read_numeric(view(A, :, :, l), W, m .& view(o, :, :, l), alg)
                  for l in 1:L))
end
"""
    collapse_categorical_mask(o::Nothing, W::MatNum, nl::Integer, m) -> nothing
    collapse_categorical_mask(o::AbstractVector{Bool}, W::MatNum, nl::Integer, m::Nothing) -> BitMatrix
    collapse_categorical_mask(o::AbstractMatrix{Bool}, W::MatNum, nl::Integer, m::AbstractMatrix{Bool}) -> BitArray

Collapses the observed mask or the data mask of a categorical Panel Field onto the tensor field that its collapse returns.

The collapsed field has one label for each level, so its mask needs a level axis, which the categorical mask does not have. A cell is observed, or holds data, for the whole label, never for one level. So the function collapses the asset mask once, and repeats it for each level.

# Algorithm

The method that Julia selects is the algorithm. `nothing` stays `nothing`. Else the asset mask collapses through [`collapse_panel_mask`](@ref), with the active mask `m`, and the answer repeats it `nl` times on a new last axis.

# Arguments

  - `o`: The observed mask or the data mask of the categorical field, `assets` or `observations × assets`, or `nothing`.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.
  - `nl`: The number of levels.
  - `m`: The active mask of the panel, or `nothing` for a static panel.

# Returns

  - The collapsed mask, `sub-portfolios × levels` or `observations × sub-portfolios × levels`, or `nothing`.

# Related

  - [`collapse_panel_field`](@ref)
  - [`collapse_panel_mask`](@ref)
  - [`CategoricalPanelField`](@ref)
"""
function collapse_categorical_mask(::Nothing, ::MatNum, ::Integer, ::Any)
    return nothing
end
function collapse_categorical_mask(o::AbstractVector{Bool}, W::MatNum, nl::Integer,
                                   ::Nothing)
    return repeat(collapse_panel_mask(o, W), 1, nl)
end
function collapse_categorical_mask(o::AbstractMatrix{Bool}, W::MatNum, nl::Integer,
                                   m::AbstractMatrix{Bool})
    return repeat(collapse_panel_mask(o, W, m), 1, 1, nl)
end
"""
    collapse_asset_panel(pnl::Nothing, wi::MatNum, nx, alg) -> nothing
    collapse_asset_panel(pnl::AssetPanel, wi::MatNum, nx::Option{<:VecStr},
                         alg::AbstractPanelCollapseAlgorithm) -> AssetPanel

Collapses an [`AssetPanel`](@ref) onto the sub-portfolios that a meta-optimiser builds for its outer problem.

The outer problem of a meta-optimiser allocates over sub-portfolios, which are the clusters of [`NestedClustered`](@ref) and the inner portfolios of [`Stacking`](@ref). Each of them is a weighted combination of the real assets. The outer [`ReturnsResult`](@ref) must state each quantity on that universe, and the panel too. With no collapse, the outer optimiser has no panel, and a [`FeatureDistance`](@ref) in it throws and does not cluster.

A feature is intensive, as `iv` and `ivpa` are. So the collapse starts from the weights of [`synthetic_asset_weights`](@ref), which divide out the gross exposure ``s_k`` of each sub-portfolio. A weighted sum with no normalisation multiplies the feature vector of a sub-portfolio by its gross exposure, which makes it larger under leverage or short positions. Under the default [`AngularDist`](@ref), the normalisation does not change a rectangular field, because a scale of one row changes no cosine. In the square case it does change the field, because the product on both sides scales the label axis too. The normalisation also keeps the collapse bounded for every gross exposure larger than zero.

At an observation where a member is outside the universe, the panel stores a finite value that has no meaning, so the collapse reads only the active members. A cell that a fill did not reach holds a placeholder too, and its placeholder mask is `true`. So for each Panel Field the collapse reads only the members that are active and hold data. A cell that a fill policy wrote holds data, and the collapse reads it. The rule `alg` states what the weight of an inactive member becomes. The default, [`RenormaliseActive`](@ref), divides by the weight of the active members, as the normalisation divides out the gross exposure, and each value stays a convex combination. [`InactiveAsCash`](@ref) reads the missing weight as cash with a zero feature. It matches the outer returns for an additive feature, because those returns read the weight of an inactive asset as cash.

The collapse does not support an extensive feature, such as a market capitalisation or a headcount, that needs a weighted sum. The divisor comes from the inner solve, so a caller cannot scale the feature before the solve.

A sub-portfolio whose weights are all zero has a gross exposure of zero. [`synthetic_asset_weights`](@ref) keeps its column of zeros and does not divide, so the collapse gives that sub-portfolio a feature vector of zeros and does not throw. The distance kernel already handles a feature vector of zeros, and the same weights give a returns column of zeros, and zero `iv` and `ivpa`. A sub-portfolio with no active member at an observation gets the same zeros there, and it is inactive there.

# Mathematical definition

At each observation ``t`` of a panel that changes over time:

```math
\\begin{align}
D_{t,ik} &= \\frac{\\tilde{W}_{ik} r_{ti}}{d_{tk}}\\,, \\quad r_{ti} = m_{ti} u_{ti}\\,,\\\\
\\boldsymbol{a}^{o}_{t} &= \\mathbf{D}_{t}^\\intercal \\boldsymbol{a}_{t}\\,,\\\\
\\mathbf{C}^{o}_{t} &= \\mathbf{D}_{t}^\\intercal \\mathbf{C}_{t}\\,,\\\\
\\mathbf{S}^{o}_{t} &= \\mathbf{D}_{t}^\\intercal \\mathbf{S}_{t} \\mathbf{D}_{t}\\,,\\\\
\\mathbf{F}^{o}_{t} &= \\mathbf{D}_{t}^\\intercal \\mathbf{H}_{t}\\,,\\\\
m^{o}_{tk} &= \\mathbf{1}\\left[\\left(\\tilde{\\mathbf{W}}^\\intercal \\boldsymbol{m}_{t}\\right)_{k} > 0\\right]\\,,\\\\
o^{o}_{tk} &= \\mathbf{1}\\left[\\left(\\tilde{\\mathbf{W}}^\\intercal (\\boldsymbol{m}_{t} \\odot \\boldsymbol{o}_{t})\\right)_{k} > 0\\right]\\,,\\\\
p^{o}_{tk} &= 1 - \\mathbf{1}\\left[\\left(\\tilde{\\mathbf{W}}^\\intercal (\\boldsymbol{m}_{t} \\odot \\boldsymbol{u}_{t})\\right)_{k} > 0\\right]\\,.
\\end{align}
```

The divisor and the restricted weights belong to one Panel Field, because each field has its own data mask. Under [`RenormaliseActive`](@ref), ``d_{tk} = \\sum_{i} \\tilde{W}_{ik} r_{ti}``, so a numeric value is

```math
\\begin{align}
a^{o}_{tk} &= \\frac{\\sum_{i=1}^{N} \\tilde{W}_{ik} r_{ti} a_{ti}}{\\sum_{i=1}^{N} \\tilde{W}_{ik} r_{ti}}\\,.
\\end{align}
```

Under [`InactiveAsCash`](@ref), ``d_{tk} = 1``, so ``a^{o}_{tk} = \\sum_{i} \\tilde{W}_{ik} r_{ti} a_{ti}``. A static panel has no active mask, so ``m_{ti} = 1`` on its one row, and ``r_{ti} = u_{ti}``. A tensor field reads the data mask of each label, and a square field reads a pair of members, see [`collapse_read_tensor`](@ref). Where every active cell holds data, ``r_{ti} = m_{ti}``.

Where:

  - ``\\mathbf{D}_{t}``: Restricted weights of observation ``t``, `assets × sub-portfolios`.
  - ``d_{tk}``: Divisor of sub-portfolio ``k`` at observation ``t``, see [`active_weight_divisor`](@ref). It is ``1`` where the sum is zero, and where every member with weight is active.
  - ``\\boldsymbol{a}_{t}``, ``\\boldsymbol{a}^{o}_{t}``: Values of a numeric field at observation ``t``, one entry for each asset and one for each sub-portfolio.
  - ``\\mathbf{C}_{t}``, ``\\mathbf{C}^{o}_{t}``: Values of a tensor field whose labels are not the asset names, `assets × labels` and `sub-portfolios × labels`.
  - ``\\mathbf{S}_{t}``, ``\\mathbf{S}^{o}_{t}``: Values of a tensor field whose labels are the asset names, `assets × assets` and `sub-portfolios × sub-portfolios`.
  - ``\\mathbf{H}_{t}``: One-hot matrix of a categorical field, `assets × levels`.
  - ``\\mathbf{F}^{o}_{t}``: Membership fractions, `sub-portfolios × levels`.
  - $(math_dict[:m_active_panel]) ``\\boldsymbol{m}_{t}`` is its row ``t``.
  - ``m^{o}_{tk}``: Active mask of the collapsed panel. The estimation mask collapses in the same way.
  - ``\\boldsymbol{o}_{t}``, ``o^{o}_{tk}``: Observed mask of a Panel Field at observation ``t``, and its collapse. ``o_{ti}`` is ``1`` for a Panel Field that cannot blank.
  - ``\\boldsymbol{u}_{t}``: Data mask of a Panel Field at observation ``t``: ``u_{ti}`` is ``1`` where the cell of member ``i`` holds a value that the raw input carried or that a fill policy wrote, and ``0`` where it holds a placeholder. See [`panel_data_mask`](@ref).
  - ``p^{o}_{tk}``: Placeholder mask of the collapsed Panel Field, ``1`` where no active member with weight holds data.
  - ``r_{ti}``: Mask of the cells that the collapse reads: ``1`` where member ``i`` is active and holds data at observation ``t``.
  - ``\\mathbf{1}[\\cdot]``: Indicator, ``1`` when the condition holds and ``0`` otherwise.
  - ``\\odot``: Element-wise product.
  - $(math_dict[:W_tilde_syn])
  - $(math_dict[:N])

Under [`RenormaliseActive`](@ref), each column of ``\\mathbf{D}_{t}`` sums to one or is a column of zeros. So each collapsed value is a convex combination of the values of the active members that hold data, or zero. A row of ``\\mathbf{F}^{o}_{t}`` sums to one when the sub-portfolio has an active member with weight and every member has a level.

# Algorithm

 1. Return `nothing` when `pnl` is `nothing`.
 2. Normalise the inner weights with [`synthetic_asset_weights`](@ref), giving `W`.
 3. Name the sub-portfolios `"_1"`, `"_2"`, …, giving `syn`.
 4. Collapse each Panel Field with [`collapse_panel_field`](@ref), on the active mask of the panel and the rule `alg`. Each field computes its divisors on its own active cells that hold data.
 5. Collapse the active mask and the estimation mask with [`collapse_panel_mask`](@ref). They stay `nothing` for a static panel.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `wi`: The inner weights, `assets × sub-portfolios`.
  - `nx`: The asset names of the returns data, or `nothing`. Only a tensor field reads it, to find the square case.
  - `alg`: The rule of the panel collapse, the `pcol` field of the meta-optimiser.

# Returns

  - `pnl::Option{AssetPanel}`: The Asset Panel on the sub-portfolios, or `nothing`.

# Related

  - [`AbstractPanelCollapseAlgorithm`](@ref)
  - [`synthetic_asset_weights`](@ref)
  - [`active_weight_divisor`](@ref)
  - [`collapse_panel_field`](@ref)
  - [`features_are_assets`](@ref)
  - [`prepare_outer_rd`](@ref)
  - [`FeatureDistance`](@ref)
"""
function collapse_asset_panel(::Nothing, ::MatNum, ::Any, ::AbstractPanelCollapseAlgorithm)
    return nothing
end
function collapse_asset_panel(pnl::AssetPanel, wi::MatNum, nx::Option{<:VecStr},
                              alg::AbstractPanelCollapseAlgorithm)
    W = synthetic_asset_weights(wi)
    syn = ["_$(k)" for k in 1:size(W, 2)]
    m = pnl.amsk
    #! A panel with no Panel Field is the ingestion layer's shape. An untyped comprehension
    #! over an empty vector answers a `Vector{Any}`, which the panel's constructor refuses,
    #! so the comprehension is typed: it answers the same vector empty or full.
    pf = AbstractPanelField[collapse_panel_field(f, W, nx, syn, m, alg) for f in pnl.pf]
    return AssetPanel(; pf = pf, amsk = collapse_panel_mask(pnl.amsk, W),
                      emsk = collapse_panel_mask(pnl.emsk, W))
end
"""
    collapse_rate(a::Nothing, W::MatNum, m, alg) -> nothing
    collapse_rate(a::Number, W::MatNum, m, alg) -> Number
    collapse_rate(a::MatNum, W::MatNum, m::Option{<:AbstractMatrix{Bool}}, alg) -> Matrix
    collapse_rate(a::VecNum, W::MatNum, m::Nothing, alg) -> Vector
    collapse_rate(a::VecNum, W::MatNum, m::AbstractMatrix{Bool}, alg) -> Vector

Collapses one implied volatility quantity of the returns data onto the sub-portfolios, with the rule of the panel collapse.

The implied volatilities `iv` and the implied volatility risk premium adjustment `ivpa` are rates, so they collapse as the values of a numeric Panel Field do, see [`collapse_asset_panel`](@ref). `iv` holds `observations × assets` and collapses row by row against the active mask of the panel. A vector `ivpa` holds one value for each asset, and it collapses over the members that are active at the last row of the mask, the row at which the outer fit stands. With no panel, or a static one, `m` is `nothing`, and every member counts.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing` and a scalar stay as they are.
 2. A matrix collapses through [`collapse_panel_numeric`](@ref), with the divisors of [`active_weight_divisor`](@ref).
 3. A vector collapses as `transpose(W) * a` with no mask. With a mask, the value of each member that is inactive at the last row of `m` becomes zero, the vector collapses in the same way, and the answer divides by the divisors of that row.

# Arguments

  - `a`: `iv`, `ivpa`, or `nothing`.
  - `W`: The normalised inner weights, `assets × sub-portfolios`.
  - `m`: The active mask of the panel, `observations × assets`, or `nothing`.
  - `alg`: The rule of the panel collapse.

# Returns

  - The collapsed quantity.

# Related

  - [`collapse_asset_panel`](@ref)
  - [`prepare_outer_rd`](@ref)
  - [`rebuild_fold_rates`](@ref)
"""
function collapse_rate(::Nothing, ::MatNum, ::Any, ::AbstractPanelCollapseAlgorithm)
    return nothing
end
function collapse_rate(a::Number, ::MatNum, ::Any, ::AbstractPanelCollapseAlgorithm)
    return a
end
function collapse_rate(a::MatNum, W::MatNum, m::Option{<:AbstractMatrix{Bool}},
                       alg::AbstractPanelCollapseAlgorithm)
    return collapse_panel_numeric(a, W, m, active_weight_divisor(alg, m, W))
end
function collapse_rate(a::VecNum, W::MatNum, ::Nothing, ::AbstractPanelCollapseAlgorithm)
    return collapse_panel_numeric(a, W)
end
function collapse_rate(a::VecNum, W::MatNum, m::AbstractMatrix{Bool},
                       alg::AbstractPanelCollapseAlgorithm)
    t = lastindex(m, 1)
    #! The product of a vector keeps the summation of the static method, so a row with no
    #! inactive member gives its answer bit for bit.
    c = collapse_panel_numeric(ifelse.(view(m, t, :), a, zero(eltype(a))), W)
    d = active_weight_divisor(alg, view(m, t:t, :), W)
    return isnothing(d) ? c : c ./ vec(d)
end

export RenormaliseActive, InactiveAsCash
public AbstractPanelCollapseAlgorithm, active_weight_divisor
