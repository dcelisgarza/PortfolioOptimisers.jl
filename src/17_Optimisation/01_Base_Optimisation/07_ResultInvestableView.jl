"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the prior result for a risk calculation from an optimisation result.

A `pr` that the caller gives has priority. Else the function tests `hasproperty(res, :pr)`, which property forwarding answers for a nested result. For example, a JuMP result holds its prior at `res.jr.pa.pr`, and `res.pr` resolves to it.

The prior of a result is on the universe that the fit solved, and `res.w` is on the universe of the caller. So a consumer that reads the prior with `res.w` reads both through [`result_investable_view`](@ref), and not through this function alone.

# Arguments

  - `res`: The optimisation result, which holds a prior result as its property `pr`, directly or by property forwarding.
  - `pr`: A prior result that the caller gives, or `nothing`.

# Validation

  - `pr` is not `nothing`, or `res` has the property `pr`, else an `ArgumentError` that names the type of `res` is thrown.

# Returns

  - `pr::Pr_RR`: The prior result for the risk calculation.

# Related

  - [`expected_risk`](@ref)
  - [`extract_fees`](@ref)
  - [`result_investable_view`](@ref)
  - [`OptimisationResult`](@ref)
"""
function extract_pr(res::OptimisationResult, pr::Option{<:Pr_RR} = nothing)
    return if !isnothing(pr)
        pr
    elseif hasproperty(res, :pr)
        res.pr
    else
        throw(ArgumentError("`$(nameof(typeof(res)))` exposes no `.pr` property, directly or through property forwarding; provide `pr` explicitly"))
    end
end
"""
    result_investable_returns(imsk, res::OptimisationResult, pr::Nothing)
    result_investable_returns(imsk::Nothing, res::OptimisationResult, pr::MatNum_Pr)
    result_investable_returns(imsk::BitVector, res::OptimisationResult, pr::Pr_RR)
    result_investable_returns(imsk::BitVector, res::OptimisationResult, X::MatNum)

Returns the returns data that a consumer of a result reads: the prior of the result, or the returns data of the caller viewed at the Investable Mask of the result.

This is the half of [`result_investable_view`](@ref) for the returns data. The prior of the result is on the universe that the fit solved, so the function reads it through [`extract_pr`](@ref) and keeps it as it is. The same holds when the caller gives that prior back, as in `expected_risk(r, res, res.pr)`. It is the only returns data that is already on the investable universe. The function finds it by identity and not by width, because reduced data and data on the full universe can have the same width by chance.

Every other returns data of the caller is on the universe of `res.w`. Under a `BitVector` mask, the function views it at `findall(imsk)`. A prior result or a [`ReturnsResult`](@ref) goes through its own [`port_opt_view`](@ref), which is the rule that [`fold_factor_returns`](@ref) applies to the `rd` of a caller on a fold. A matrix is viewed by its columns. Under a mask `nothing`, the fit left out no asset, so the returns data of the caller does not change.

# Algorithm

The method that Julia selects is the algorithm.

 1. When `pr` is `nothing`, return the prior of `res` with [`extract_pr`](@ref).
 2. When `imsk` is `nothing`, return `pr` unchanged.
 3. When `pr` is the prior of `res`, by identity, return it unchanged.
 4. Else return the view of `pr` at `imsk`.

# Arguments

  - `imsk`: The Investable Mask of `res`, or `nothing`.
  - `res::OptimisationResult`: The optimisation result.
  - `pr`: Returns data of the caller on the universe of `res.w`, or `nothing`.

# Returns

  - The returns data on the investable universe of `res`.

# Related

  - [`result_investable_view`](@ref)
  - [`result_investable_fees`](@ref)
  - [`extract_pr`](@ref)
  - [`port_opt_view`](@ref)
  - [`fold_factor_returns`](@ref)
"""
function result_investable_returns(::Any, res::OptimisationResult, ::Nothing)
    return extract_pr(res, nothing)
end
function result_investable_returns(::Nothing, ::OptimisationResult, pr::MatNum_Pr)
    return pr
end
function result_investable_returns(imsk::BitVector, res::OptimisationResult, pr::Pr_RR)
    # The result's own prior handed back is already on the investable universe. It is
    # known by identity: a width can coincide, an object cannot.
    return if hasproperty(res, :pr) && pr === res.pr
        pr
    else
        port_opt_view(pr, findall(imsk))
    end
end
function result_investable_returns(imsk::BitVector, ::OptimisationResult, X::MatNum)
    return view(X, :, imsk)
end
"""
    result_investable_fees(imsk, res::OptimisationResult, fees::Nothing)
    result_investable_fees(imsk::Nothing, res::OptimisationResult, fees::Fees)
    result_investable_fees(imsk::BitVector, res::OptimisationResult, fees::Fees)

Returns the fee that a consumer of a result charges: the fee of the result, or the fee of the caller viewed at the Investable Mask of the result.

This is the half of [`result_investable_view`](@ref) for the fee, and it is the rule that [`fold_fees`](@ref) applies on a fold. At the fit, [`investable_fees_view`](@ref) reduced the fee of the result. It sliced the five per-asset fields to the mask and the two liquidation charges, `lq` and `flq`, to the complement, and it marked the fee with the mask. So the function reads that fee through [`extract_fees`](@ref) and keeps it as it is.

A fee of the caller is on the universe of the caller, so it goes through the same function as a fee at the fit, [`investable_fees_view`](@ref). That function slices the per-asset fields to the mask and `lq` and `flq` to the complement. Under a mask `nothing`, the fit left out no asset, so it removes `lq` and `flq`. The function needs the width of the full universe to find the complement. A result holds no matrix of that width, so the width is the length of the mask.

# Algorithm

The method that Julia selects is the algorithm.

 1. When `fees` is `nothing`, return the fee of `res` with [`extract_fees`](@ref).
 2. When `imsk` is `nothing`, return `investable_fees_view(fees, nothing, nothing)`.
 3. Else return `investable_fees_view(fees, imsk, length(imsk))`.

# Arguments

  - `imsk`: The Investable Mask of `res`, or `nothing`.
  - `res::OptimisationResult`: The optimisation result.
  - `fees`: A [`Fees`](@ref) of the caller on the universe of `res.w`, or `nothing`.

# Returns

  - `fees::Option{<:Fees}`: The fee on the two axes that the mask of the result gives, or `nothing`.

# Related

  - [`result_investable_view`](@ref)
  - [`result_investable_returns`](@ref)
  - [`extract_fees`](@ref)
  - [`investable_fees_view`](@ref)
  - [`fold_fees`](@ref)
"""
function result_investable_fees(::Any, res::OptimisationResult, ::Nothing)
    return extract_fees(res, nothing)
end
function result_investable_fees(::Nothing, ::OptimisationResult, fees::Fees)
    return investable_fees_view(fees, nothing, nothing)
end
function result_investable_fees(imsk::BitVector, ::OptimisationResult, fees::Fees)
    return investable_fees_view(fees, imsk, length(imsk))
end
"""
    result_investable_view(res::OptimisationResult, pr = nothing, fees = nothing, nx = nothing)
    result_investable_view(imsk::Nothing, res::OptimisationResult, pr, fees, nx)
    result_investable_view(imsk::BitVector, res::OptimisationResult, pr, fees, nx)

Puts the weights, the returns data, the fee and the asset names that a consumer of a result reads on the investable universe of the result.

A result holds three values on two universes. An optimisation reduces its universe to the Investable Mask at its entry, and expands the solved weights back. So `res.w` is on the full universe of the caller, and `res.pr` is the prior of the universe that the fit solved. The entry also reduced `res.fees`, with its five per-asset fields on the mask and its two liquidation charges, `lq` and `flq`, on the complement.

A consumer that reads two of the three values separately puts a full vector with a reduced one. A per-asset fee indexed at a mask of full length throws a `BoundsError`. A reduced returns matrix with a full weight vector throws a `DimensionMismatch`. A reduced `mu` under the full names of the caller draws each bar after the gap under the name of the asset before it. So this function is the one place that puts the values together. Every method of [`expected_risk`](@ref), [`calc_net_returns`](@ref), [`factor_attribution`](@ref) and [`performance_summary`](@ref) that takes a result reads the values through it, and so does the plotting extension.

The answers are on the investable universe of `res`, where its prior and its fee already are. The weights are viewed at the mask through [`investable_weights_view`](@ref). [`result_investable_returns`](@ref) gives the returns data, and [`result_investable_fees`](@ref) gives the fee. Each keeps the value of the result as it is, and views a value of the caller at the mask. The asset names are on the asset axis, so they take the mask directly, and a figure labels each bar with its own asset.

A result whose mask is `nothing` left out no asset, so the weights and the names do not change. A consumer that forms a per-asset answer on the investable universe expands it back through [`expand_investable_weights`](@ref), with the mask that this function returns first.

# Algorithm

 1. Read the Investable Mask of `res` with [`result_investable_mask`](@ref).
 2. For a mask `nothing`, return `nothing`, `res.w`, the returns data from [`result_investable_returns`](@ref), the fee from [`result_investable_fees`](@ref), and `nx` unchanged.
 3. For a `BitVector` mask, return the mask, the view of `res.w` at it, the returns data and the fee from the same two functions, and the view of `nx` at the mask.

# Arguments

  - `res::OptimisationResult`: The optimisation result.
  - `pr`: Returns data of the caller on the universe of `res.w`: a prior result, a [`ReturnsResult`](@ref) or a returns matrix. With `nothing`, the function reads the prior of the result.
  - `fees`: A [`Fees`](@ref) of the caller on the universe of `res.w`. With `nothing`, the function reads the fee of the result.
  - `nx`: Asset names on the universe of `res.w`, or `nothing`.
  - `imsk`: The Investable Mask of `res`, or `nothing`.

# Returns

  - `(imsk, w, pr, fees, nx)`: The Investable Mask, and the weights, the returns data, the fee and the names on the investable universe of `res`.

# Related

  - [`result_investable_mask`](@ref)
  - [`result_investable_returns`](@ref)
  - [`result_investable_fees`](@ref)
  - [`investable_weights_view`](@ref)
  - [`investable_fees_view`](@ref)
  - [`expand_investable_weights`](@ref)
  - [`extract_pr`](@ref)
  - [`extract_fees`](@ref)
  - [`fold_fees`](@ref)
"""
function result_investable_view(res::OptimisationResult, pr::Option{<:MatNum_Pr} = nothing,
                                fees::Option{<:Fees} = nothing,
                                nx::Option{<:AbstractVector} = nothing)
    return result_investable_view(result_investable_mask(res), res, pr, fees, nx)
end
function result_investable_view(::Nothing, res::OptimisationResult, pr::Option{<:MatNum_Pr},
                                fees::Option{<:Fees}, nx::Option{<:AbstractVector})
    return nothing, res.w, result_investable_returns(nothing, res, pr),
           result_investable_fees(nothing, res, fees), nx
end
function result_investable_view(imsk::BitVector, res::OptimisationResult,
                                pr::Option{<:MatNum_Pr}, fees::Option{<:Fees},
                                nx::Option{<:AbstractVector})
    return imsk, investable_weights_view(imsk, res.w),
           result_investable_returns(imsk, res, pr),
           result_investable_fees(imsk, res, fees), nothing_scalar_array_view(nx, imsk)
end
"""
    expected_risk(r::BaseRM_VecBaseRM, res::OptimisationResult, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::BaseRM_VecBaseRM, res::OptimisationResult, pr = nothing, fees = nothing; kwargs...)

Computes the expected risk of the weights of an [`OptimisationResult`](@ref).

The function reads `w` from `res` and calls the [`expected_risk`](@ref) that takes weights. A `fees` of the caller has priority over `res.fees`.

The weights, the returns data and the fee meet on the investable universe of `res`, through [`result_investable_view`](@ref). The prior of the result is on the universe that the fit solved, and the entry of the fit reduced its fee. But `res.w` is expanded back to the universe of the caller. So the weights are viewed at the Investable Mask of the result, [`result_investable_mask`](@ref), and a result with no mask views nothing. A `pr`, an `X` and a `fees` of the caller are on the universe of `res.w`, and are viewed at the same mask.

The argument `r` is one measure or a vector of measures. The keyword `sca` scalarises a vector, and its default is [`SumScalariser`](@ref). The function does not read the measure from `res`. So a result that holds its own `r` and `sca` gives the value that it optimised only when the caller passes them back, as `expected_risk(res.r, res; sca = res.sca)`.

The method that takes a prior gives the whole of `pr` to the kernel, so a measure of the caller resolves against it. A slot that the measure does not state takes the field of the prior, and a Deferred Quantity is fitted, as in `expected_risk(r, w, pr)`. Pass a matrix instead to state every slot yourself.

# Algorithm

 1. View the weights, the returns data and the fee on the investable universe of `res` with [`result_investable_view`](@ref).
 2. Return `expected_risk(r, w, X, fees; kwargs...)` or `expected_risk(r, w, pr, fees; kwargs...)` of the viewed values.

# Related

  - [`expected_risk`](@ref)
  - [`OptimisationResult`](@ref)
  - [`BaseRM_VecBaseRM`](@ref)
  - [`resolve_risk_inputs`](@ref)
  - [`result_investable_mask`](@ref)
  - [`result_investable_view`](@ref)
"""
function expected_risk(r::BaseRM_VecBaseRM, res::OptimisationResult, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    _, w, X, fees = result_investable_view(res, X, fees)
    return expected_risk(r, w, X, fees; kwargs...)
end
function expected_risk(r::BaseRM_VecBaseRM, res::OptimisationResult,
                       pr::Option{<:Pr_RR} = nothing, fees::Option{<:Fees} = nothing;
                       kwargs...)
    # `pr` is forwarded whole, never unwrapped to `pr.X`. `expected_risk`'s own `Pr_RR`
    # route resolves the measure through `resolve_risk_inputs`, which has an arm for each
    # type of `pr`: a prior result resolves the measure, a `ReturnsResult` only unwraps `X`.
    # Unwrapping here dropped the prior fallback, so `expected_risk(Variance(), res)` — the
    # call this docstring asks callers to make — hit the kernel with an unstated `sigma`.
    # The result's own prior and fee are on the investable universe and its weights are
    # expanded back to the caller's, so the three meet at the mask; a caller's `pr` and
    # `fees` are on the weights' own universe, and are viewed at the same mask.
    _, w, pr, fees = result_investable_view(res, pr, fees)
    return expected_risk(r, w, pr, fees; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns `false`, the fallback for `nothing` and for every estimator, algorithm and result that has no method of its own.

A type whose value reads the weights of the previous fold adds a method that returns `true`, for example a turnover, a tracking error or a fee.

# Related

  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(::Option{<:Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                                 <: AbstractResult}})
    return false
end
