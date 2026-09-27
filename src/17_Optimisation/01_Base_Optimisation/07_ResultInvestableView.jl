"""
$(DocStringExtensions.TYPEDSIGNATURES)

Extracts the prior result for risk calculation from an optimisation result.

An explicitly provided `pr` wins. Otherwise the one test is `hasproperty(res, :pr)`, which property forwarding answers for a nested result: a JuMP leaf reaches its prior at `res.jr.pa.pr`, and `res.pr` resolves to it. A result that exposes no `pr` property throws.

# Arguments

  - `res`: Optimisation result, which carries a prior result as its `pr` property or reaches one by property forwarding.
  - `pr`: Optional prior result to use for risk calculation, which takes precedence over the one found in `res`.

# Returns

  - `pr::Pr_RR`: The prior result to use for risk calculation. Throws an `ArgumentError` when none is found.

A result's prior is on the universe the fit **solved**, so a consumer that pairs it with `res.w`, which is on the caller's, reads both through [`result_investable_view`](@ref) rather than through this verb alone.

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
    result_investable_carrier(imsk, res::OptimisationResult, pr::Nothing)
    result_investable_carrier(imsk::Nothing, res::OptimisationResult, pr::MatNum_Pr)
    result_investable_carrier(imsk::BitVector, res::OptimisationResult, pr::Pr_RR)
    result_investable_carrier(imsk::BitVector, res::OptimisationResult, X::MatNum)

Resolve the returns carrier a result-taking consumer reads: the result's own prior, or a caller's viewed at the result's Investable Mask.

The carrier half of [`result_investable_view`](@ref). The result's own prior is on the universe the fit solved, so it is read through [`extract_pr`](@ref) and taken as it is, and so is that prior when a caller hands it back, `expected_risk(r, res, res.pr)`: it is the one carrier already on the investable universe, and it is recognised by identity, never by width, because a reduced and a full-universe carrier share a width only by coincidence. Every other caller's carrier is on the caller's universe, the one `res.w` spans, so under a `BitVector` mask it is viewed at `findall(imsk)`: a prior result or a [`ReturnsResult`](@ref) through the [`port_opt_view`](@ref) its owner writes, which is the rule [`fold_factor_returns`](@ref) applies to a caller's `rd` on a fold, and a bare matrix by its columns. Under a `nothing` mask nothing left, and a caller's carrier is returned unchanged.

# Arguments

  - `imsk`: The Investable Mask of `res`, or `nothing`.
  - `res::OptimisationResult`: Fitted optimisation result.
  - `pr`: A caller's carrier on the universe of `res.w`, or `nothing`.

# Returns

  - The carrier on the investable universe of `res`.

# Related

  - [`result_investable_view`](@ref)
  - [`result_investable_fees`](@ref)
  - [`extract_pr`](@ref)
  - [`port_opt_view`](@ref)
  - [`fold_factor_returns`](@ref)
"""
function result_investable_carrier(::Any, res::OptimisationResult, ::Nothing)
    return extract_pr(res, nothing)
end
function result_investable_carrier(::Nothing, ::OptimisationResult, pr::MatNum_Pr)
    return pr
end
function result_investable_carrier(imsk::BitVector, res::OptimisationResult, pr::Pr_RR)
    # The result's own prior handed back is already on the investable universe. It is
    # known by identity: a width can coincide, an object cannot.
    return if hasproperty(res, :pr) && pr === res.pr
        pr
    else
        port_opt_view(pr, findall(imsk))
    end
end
function result_investable_carrier(imsk::BitVector, ::OptimisationResult, X::MatNum)
    return view(X, :, imsk)
end
"""
    result_investable_fees(imsk, res::OptimisationResult, fees::Nothing)
    result_investable_fees(imsk::Nothing, res::OptimisationResult, fees::Fees)
    result_investable_fees(imsk::BitVector, res::OptimisationResult, fees::Fees)

Resolve the fee a result-taking consumer charges: the result's own, or a caller's viewed at the result's Investable Mask.

The fee half of [`result_investable_view`](@ref), and the rule [`fold_fees`](@ref) applies on a fold. The result's own fee was reduced at the fit's door, its five per-asset fields to the mask and its two liquidation carriers to the complement, and is marked with that mask, so it is read through [`extract_fees`](@ref) and taken as it is. A caller's fee is stated on the caller's universe, so it takes the same door a fee takes at the fit, [`investable_fees_view`](@ref), which slices the per-asset fields to the mask and the carriers to its complement, and strips the carriers under a `nothing` mask, because nothing left. The door needs the width of the unreduced universe to derive the complement, and a result holds no full-width matrix, so the width is the mask's own length.

# Arguments

  - `imsk`: The Investable Mask of `res`, or `nothing`.
  - `res::OptimisationResult`: Fitted optimisation result.
  - `fees`: A caller's [`Fees`](@ref) on the universe of `res.w`, or `nothing`.

# Returns

  - `fees::Option{<:Fees}`: The fee on the two axes the result's mask leaves, or `nothing`.

# Related

  - [`result_investable_view`](@ref)
  - [`result_investable_carrier`](@ref)
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

Pair the weights, the returns carrier, the fee and the axis names a result-taking consumer reads, on the investable universe of the result.

A result carries three things on two universes. An optimisation reduces to its Investable Mask at the entry and expands the solved weights back, so `res.w` is on the **full** universe the caller stated, while `res.pr` is the prior of the universe the fit **solved**, and `res.fees` was reduced at the same door, its five per-asset fields to the mask and its two liquidation carriers to the complement. A consumer that reads two of the three separately pairs a full vector with a reduced one: a per-asset fee indexed at a full-length mask raises a `BoundsError`, a reduced returns matrix against a full weight vector a `DimensionMismatch`, and a reduced `mu` under the caller's full names draws every bar after the gap under the name of the asset before it. This verb is the one place the pairing is made, so every result-taking arity of [`expected_risk`](@ref), [`calc_net_returns`](@ref), [`factor_attribution`](@ref), [`performance_summary`](@ref) and the plotting extension reads all three through it.

The answers are on the **investable** universe of `res`, which is the universe its own prior and fee are already on. The weights are viewed at the mask through [`investable_weights_view`](@ref). The carrier is resolved by [`result_investable_carrier`](@ref) and the fee by [`result_investable_fees`](@ref): the result's own are taken as they are, and a caller's, stated on the universe `res.w` spans, is viewed at the mask, the carrier through the [`port_opt_view`](@ref) its owner writes and the fee through the door a fee takes at the fit. The axis names ride the asset axis, so they take the mask directly, and a drawn figure labels each bar with its own asset.

A result whose mask is `nothing` reduced on nothing, so the weights and the names are returned unchanged. A per-asset answer a consumer forms on the investable universe expands back through [`expand_investable_weights`](@ref) with the mask this verb returns first, as the value-level doors do.

# Algorithm

 1. Read the Investable Mask of `res` with [`result_investable_mask`](@ref).
 2. On a `nothing` mask, return `nothing`, `res.w`, the carrier through [`result_investable_carrier`](@ref), the fee through [`result_investable_fees`](@ref), and `nx` unchanged.
 3. On a `BitVector` mask, return it, the view of `res.w` at it, the carrier and the fee through the same two verbs, and the view of `nx` at it.

# Arguments

  - `res::OptimisationResult`: Fitted optimisation result.
  - `pr`: A caller's returns carrier on the universe of `res.w`, a prior result, a [`ReturnsResult`](@ref) or a returns matrix, or `nothing` to read the result's own prior.
  - `fees`: A caller's [`Fees`](@ref) on the universe of `res.w`, or `nothing` to read the result's own.
  - `nx`: Asset names on the universe of `res.w`, or `nothing`.
  - `imsk`: The Investable Mask of `res`, or `nothing`.

# Returns

  - `(imsk, w, pr, fees, nx)`: The Investable Mask, and the weights, the carrier, the fee and the names on the investable universe of `res`.

# Related

  - [`result_investable_mask`](@ref)
  - [`result_investable_carrier`](@ref)
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
    return nothing, res.w, result_investable_carrier(nothing, res, pr),
           result_investable_fees(nothing, res, fees), nx
end
function result_investable_view(imsk::BitVector, res::OptimisationResult,
                                pr::Option{<:MatNum_Pr}, fees::Option{<:Fees},
                                nx::Option{<:AbstractVector})
    return imsk, investable_weights_view(imsk, res.w),
           result_investable_carrier(imsk, res, pr),
           result_investable_fees(imsk, res, fees), nothing_scalar_array_view(nx, imsk)
end
"""
    expected_risk(r::BaseRM_VecBaseRM, res::OptimisationResult, X::MatNum, fees = nothing; kwargs...)
    expected_risk(r::BaseRM_VecBaseRM, res::OptimisationResult, pr = nothing, fees = nothing; kwargs...)

Compute the expected risk for an [`OptimisationResult`](@ref).

Extracts `w` from `res` and delegates to the weight-based [`expected_risk`](@ref). `fees` takes precedence over `res.fees` if both are provided.

The weights, the carrier and the fee meet on the investable universe of `res`, through [`result_investable_view`](@ref). The result's own prior is on the universe the fit solved on, and its fee was reduced at the same door, while `res.w` is expanded back to the caller's universe, so the weights are viewed at the result's Investable Mask ([`result_investable_mask`](@ref)) before they meet them; a result with no mask views nothing. A caller's own `pr`, a caller's `X` and a caller's `fees` are on the universe of `res.w`, and are viewed at the same mask.

`r` is one measure or a vector of them; a vector is scalarised by `sca`, defaulting to [`SumScalariser`](@ref). The measure is **not** read from `res`, so a result that carries its own `r` and `sca` reports the figure it optimised only when the caller passes them back, as `expected_risk(res.r, res; sca = res.sca)`.

The prior-taking method forwards the carrier whole, so a caller's **own** measure resolves against it: an unstated slot takes the prior's field and a **Deferred Quantity** is fitted, exactly as in `expected_risk(r, w, pr)`. Pass a matrix instead to opt out and supply every slot yourself.

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
    # The carrier is forwarded whole, never unwrapped to `pr.X`. `expected_risk`'s own
    # `Pr_RR` route resolves the measure through `resolve_risk_inputs`, which has an arm for
    # each carrier: a prior result resolves the measure, a `ReturnsResult` only unwraps `X`.
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

Return `false`.

`nothing` never requires previous portfolio weights.

# Related

  - [`needs_previous_weights`](@ref)
"""
function needs_previous_weights(::Option{<:Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                                 <: AbstractResult}})
    return false
end
