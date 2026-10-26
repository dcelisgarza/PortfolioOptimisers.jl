"""
$(DocStringExtensions.TYPEDEF)

Calibrates the radius of an ``\\ell_1`` uncertainty set to a target number of active assets.

The radius ``\\epsilon`` of an ``\\ell_1`` uncertainty set has no natural scale. It is a sum of differences of the characteristic. On daily returns it is of order ``10^{-3}``, and on annualised returns it is about ``250`` times larger. A caller can state a number of assets to hold more easily than a radius. This algorithm finds the radius that activates that number of assets, from the inverse of the closed forms of [quintile](@cite). The inverse is exact only for the problem of those closed forms, which has the budget and sign constraints and no other constraint.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ActiveAssetsUncertaintyAlgorithm(;
        active::Union{Integer, AbstractFloat} = 0.2
    ) -> ActiveAssetsUncertaintyAlgorithm

Keywords correspond to the struct's fields.

## Validation

  - If `active` is an `Integer`: `active >= 1`.
  - If `active` is an `AbstractFloat`: `0 < active < 1`.

The type of `active` selects the rule. The constructor accepts `active = 1`, a count of one asset. It throws a `DomainError` on `active = 1.0`, a fraction equal to the whole universe.

# Examples

```jldoctest
julia> ActiveAssetsUncertaintyAlgorithm()
ActiveAssetsUncertaintyAlgorithm
  active ┴ Float64: 0.2
```

# Related

  - [`AbstractUncertaintyEpsAlgorithm`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`SignedL1UncertaintySetAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
  - [`l1_resolve_eps`](@ref): the function that reads this algorithm and returns the radius.
  - [`JuMPOptimiser`](@ref): its `card` field sets a hard limit on the number of assets held. A weight bound, a cardinality constraint or a linear constraint can make the number of active assets differ from the target of this algorithm. This algorithm does not check the number.

# References

  - $(ref_dict[:quintile])
"""
@concrete struct ActiveAssetsUncertaintyAlgorithm <: AbstractUncertaintyEpsAlgorithm
    """
    $(field_dict[:active_ucs])
    """
    active
    function ActiveAssetsUncertaintyAlgorithm(active::Union{Integer, AbstractFloat})
        if isa(active, Integer)
            @argcheck(active >= one(active),
                      DomainError(active, "active must be >= 1 when given as a count"))
        else
            @argcheck(zero(active) < active < one(active),
                      DomainError(active,
                                  "active must be in (0, 1) when given as a fraction"))
        end
        return new{typeof(active)}(active)
    end
end
function ActiveAssetsUncertaintyAlgorithm(;
                                          active::Union{Integer, AbstractFloat} = 0.2)::ActiveAssetsUncertaintyAlgorithm
    return ActiveAssetsUncertaintyAlgorithm(active)
end
"""
$(DocStringExtensions.TYPEDEF)

``\\ell_1`` uncertainty set on the characteristic vector, a cross-polytope around the estimate.

The set is ``\\mathcal{S} = \\{\\hat{\\boldsymbol{\\mu}} + \\boldsymbol{\\sigma} \\odot \\boldsymbol{u} : \\lVert \\boldsymbol{u} \\rVert_1 \\leq \\epsilon\\}``. One error budget covers every asset and both signs. [`CharacteristicUncertaintySet`](@ref) fits the set, and [`ArithmeticReturn`](@ref) reads it. The set bounds a characteristic vector, such as the mean, and it has no covariance analogue.

When every ``\\sigma_i > 0``, the set is ``\\{\\hat{\\boldsymbol{\\mu}} + \\boldsymbol{e} : \\lVert \\boldsymbol{e} \\oslash \\boldsymbol{\\sigma} \\rVert_1 \\leq \\epsilon\\}``, the form of [quintile](@cite). The form above also admits ``\\sigma_i = 0``, which fixes the entry ``i`` at its estimate. It is the [`NormBallUncertaintySet`](@ref) with the map ``\\mathrm{diag}(\\boldsymbol{\\sigma})``, the order ``p = 1`` and the radius ``\\epsilon``. A zero scale is the zero row of that map, and [`expand_investable_ucs`](@ref) writes it outside the Investable Mask.

# Mathematical definition

The worst case of a linear characteristic over the set is the estimate less a scaled infinity norm. Lemmas 1 and 8 of [quintile](@cite) state it:

```math
\\underset{\\boldsymbol{\\mu} \\in \\mathcal{S}}{\\min}\\, \\boldsymbol{\\mu}^{\\intercal} \\boldsymbol{w} = \\hat{\\boldsymbol{\\mu}}^{\\intercal} \\boldsymbol{w} - \\epsilon \\lVert \\boldsymbol{\\sigma} \\odot \\boldsymbol{w} \\rVert_{\\infty}\\,.
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated characteristic vector.
  - ``\\epsilon``: Radius of the set.
  - ``\\boldsymbol{\\sigma}``: Scaling of each asset, the `sd` field. It is ``\\boldsymbol{1}`` when `sd` is `nothing`.
  - ``\\boldsymbol{u}``: Scaled estimation error.
  - ``\\odot``, ``\\oslash``: Element-wise product and division.

The right-hand side is concave in ``\\boldsymbol{w}``. With an epigraph variable for the infinity norm, the worst case is a set of linear constraints, and the problem needs no conic solver. The equality holds for every ``\\boldsymbol{\\sigma} \\geq \\boldsymbol{0}``, because the minimum of ``\\boldsymbol{u}^{\\intercal} (\\boldsymbol{\\sigma} \\odot \\boldsymbol{w})`` over the ``\\ell_1`` ball of radius ``\\epsilon`` is ``-\\epsilon \\lVert \\boldsymbol{\\sigma} \\odot \\boldsymbol{w} \\rVert_{\\infty}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L1UncertaintySet(;
        eps::Number,
        sd::Option{<:VecNum} = nothing,
        mu::Option{<:VecNum} = nothing
    ) -> L1UncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(eps)` and `eps >= 0`.
  - If `sd` is not `nothing`: `!isempty(sd)`, `all(isfinite, sd)` and `all(sd .>= 0)`. A zero scale fixes its entry at the estimate.
  - If `mu` is not `nothing`: `!isempty(mu)` and `!any(isinf, mu)`. A `NaN` marks an asset outside the Investable Mask, as in the centre `val` of the other sets.
  - If `sd` and `mu` are not `nothing`: `length(mu) == length(sd)`.

# Examples

```jldoctest
julia> L1UncertaintySet(; eps = 0.1)
L1UncertaintySet
  eps ┼ Float64: 0.1
   sd ┼ nothing
   mu ┴ nothing
```

# Related

  - [`SignedL1UncertaintySet`](@ref)
  - [`CharacteristicUncertaintySet`](@ref): the estimator that fits this set. It fits a set on the mean only, and [`sigma_ucs`](@ref) throws an `ArgumentError` on it.
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`AbstractUncertaintySetResult`](@ref)
  - [`mu_ucs`](@ref): a set that it fits holds, in `mu`, the characteristic vector that the Radius Calibration read. The consumer then bounds that vector and no other.
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:quintile])
"""
@concrete struct L1UncertaintySet <: AbstractUncertaintySetResult
    """
    $(field_dict[:eps_ucs])
    """
    eps
    """
    $(field_dict[:sd_ucs])
    """
    sd
    """
    $(field_dict[:mu_l1_ucs])
    """
    mu
    function L1UncertaintySet(eps::Number, sd::Option{<:VecNum}, mu::Option{<:VecNum})
        @argcheck(isfinite(eps) && eps >= zero(eps),
                  DomainError(eps, "eps must be finite and >= 0"))
        if isa(sd, VecNum)
            @argcheck(!isempty(sd), IsEmptyError("sd cannot be empty"))
            @argcheck(all(isfinite, sd),
                      IsNonFiniteError("all entries of sd must be finite"))
            @argcheck(all(x -> x >= zero(x), sd),
                      DomainError(sd, "all entries of sd must be >= 0"))
        end
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(!any(isinf, mu),
                      IsNonFiniteError("mu must not have an infinite entry. A NaN marks an asset outside the Investable Mask and is accepted."))
            if isa(sd, VecNum)
                @argcheck(length(mu) == length(sd),
                          DimensionMismatch("mu ($(length(mu))) must match sd ($(length(sd)))"))
            end
        end
        return new{typeof(eps), typeof(sd), typeof(mu)}(eps, sd, mu)
    end
end
function L1UncertaintySet(eps::Number, sd::Option{<:VecNum})::L1UncertaintySet
    return L1UncertaintySet(eps, sd, nothing)
end
function L1UncertaintySet(; eps::Number, sd::Option{<:VecNum} = nothing,
                          mu::Option{<:VecNum} = nothing)::L1UncertaintySet
    return L1UncertaintySet(eps, sd, mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of an [`L1UncertaintySet`](@ref) restricted to the asset indices `i`.

The radius is one scalar budget for the universe, and the view keeps it unchanged. The view takes the entries of `sd` and `mu` at `i`, in the order of `i`. The restricted set then indexes the assets in the order that the optimiser gives.

# Algorithm

 1. Take `nothing_scalar_array_view(risk_ucs.sd, i)`, the scaling of the selected assets. A `nothing` stays `nothing`.
 2. Take `nothing_scalar_array_view(risk_ucs.mu, i)`, the characteristic vector of the same assets. A `nothing` stays `nothing`.
 3. Build an [`L1UncertaintySet`](@ref) from `risk_ucs.eps` and the two views. The step does not rescale the radius, because the radius is the budget for the total error over the assets that the set covers.

# Arguments

  - `risk_ucs`: ``\\ell_1`` uncertainty set.
  - `i`: Index of the cluster or of the assets.
  - `args...`: More positional arguments, which the method ignores.

# Returns

  - `risk_ucs::L1UncertaintySet`: The set restricted to `i`.

# Examples

```jldoctest
julia> ucs = L1UncertaintySet(; eps = 0.5, sd = [0.1, 0.2, 0.3], mu = [0.05, 0.03, 0.01]);

julia> PortfolioOptimisers.port_opt_view(ucs, [1, 3])
L1UncertaintySet
  eps ┼ Float64: 0.5
   sd ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}: [0.1, 0.3]
   mu ┴ SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}: [0.05, 0.01]
```

# Related

  - [`L1UncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::L1UncertaintySet, i, args...)::L1UncertaintySet
    return L1UncertaintySet(; eps = risk_ucs.eps,
                            sd = nothing_scalar_array_view(risk_ucs.sd, i),
                            mu = nothing_scalar_array_view(risk_ucs.mu, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Writes an [`L1UncertaintySet`](@ref) fitted on the Investable Mask back onto the full asset universe.

The expansion is the inverse of [`port_opt_view`](@ref). The view slices `sd` and `mu` by the asset index, and keeps the radius. So the view removes all of the frame outside the mask, and it gives back the fitted set. The radius is one budget for the total error over the assets that the set covers, so it passes through.

The scale `sd` is geometry. The set is the norm ball with the map ``\\mathrm{diag}(\\boldsymbol{\\sigma})``, so the frame is zero and not `NaN`, as the zero row of the map of a [`NormBallUncertaintySet`](@ref). A zero scale fixes the entry at its estimate, so the set moves nothing on an asset that the prior did not estimate. The worst case ``\\epsilon \\lVert \\boldsymbol{\\sigma} \\odot \\boldsymbol{w} \\rVert_{\\infty}`` then reads only the assets inside the mask. A `sd` of `nothing` holds no entry for an asset, so it passes through, and the view gives back `nothing`. The centre `mu` is the full `pr.mu`, with its `NaN` entries, as the centre `val` of the other sets.

# Algorithm

 1. If `set.sd` is `nothing`, keep `nothing`. Else, make a zero frame of `length(imsk)` entries, and write `set.sd` at the entries that the mask keeps.
 2. Build an [`L1UncertaintySet`](@ref) from `set.eps` and the frame, with `pr.mu` as `mu`.

# Arguments

  - `set`: ``\\ell_1`` uncertainty set, fitted on the reduced prior.
  - `imsk`: The Investable Mask of the full prior.
  - $(arg_dict[:pr])

# Returns

  - `set::L1UncertaintySet`: The set on the full asset universe.

# Related

  - [`L1UncertaintySet`](@ref)
  - [`investable_ucs_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function expand_investable_ucs(set::L1UncertaintySet, imsk::BitVector,
                               pr::AbstractPriorResult)::L1UncertaintySet
    sd = if isnothing(set.sd)
        nothing
    else
        frame = zeros(eltype(set.sd), length(imsk))
        frame[imsk] = set.sd
        frame
    end
    return L1UncertaintySet(; eps = set.eps, sd = sd, mu = pr.mu)
end
"""
$(DocStringExtensions.TYPEDEF)

Signed ``\\ell_1`` uncertainty set on the characteristic vector, with a separate error budget per sign.

The set is ``\\mathcal{A}_2 = \\{\\hat{\\boldsymbol{\\mu}} + \\boldsymbol{\\sigma} \\odot \\boldsymbol{u} : \\boldsymbol{1}^{\\intercal} [\\boldsymbol{u}]_{+} \\leq \\epsilon_{+},\\, -\\boldsymbol{1}^{\\intercal} [\\boldsymbol{u}]_{-} \\leq \\epsilon_{-}\\}``. When every ``\\sigma_i > 0``, ``\\boldsymbol{u} = \\boldsymbol{e} \\oslash \\boldsymbol{\\sigma}``, and the set is the set of [quintile](@cite). A zero scale fixes its entry at the estimate, as in [`L1UncertaintySet`](@ref), and the worst case below holds for every ``\\boldsymbol{\\sigma} \\geq \\boldsymbol{0}``. It bounds a characteristic vector, such as the mean, and it has no covariance analogue. [quintile](@cite) uses the set to split the long-short problem into two independent problems, its Equations 27 and 28. Its Remark 12 joins the two solutions only when the two legs have disjoint supports. The worst case below keeps the long-short problem as one problem, so it needs no condition on the supports.

# Mathematical definition

```math
\\underset{\\boldsymbol{\\mu} \\in \\mathcal{A}_2}{\\min}\\, \\boldsymbol{\\mu}^{\\intercal} \\boldsymbol{w} = \\hat{\\boldsymbol{\\mu}}^{\\intercal} \\boldsymbol{w} - \\epsilon_{+} \\left[\\underset{i}{\\max}\\, (-\\sigma_i w_i)\\right]_{+} - \\epsilon_{-} \\left[\\underset{i}{\\max}\\, (\\sigma_i w_i)\\right]_{+}\\,.
```

Where:

  - ``\\epsilon_{+}``, ``\\epsilon_{-}``: Radii of the positive-error and the negative-error sides, the `ep` and `en` fields.
  - ``[\\cdot]_{+}``, ``[\\cdot]_{-}``: Element-wise positive and negative parts.

The worst case is concave. With one epigraph variable for each sign, it is a set of linear constraints.

With ``\\epsilon_{+} = \\epsilon_{-} = \\epsilon``, this set is not ``\\mathcal{S}``. Let ``t_{+}`` and ``t_{-}`` be the two positive parts above. The joint set has one budget for both signs, and its penalty is ``\\epsilon \\max(t_{+}, t_{-})``. This set has one budget for each sign, and its penalty is ``\\epsilon_{+} t_{+} + \\epsilon_{-} t_{-}``. The two worst cases agree only when all the entries of ``\\boldsymbol{w}`` have the same sign.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SignedL1UncertaintySet(;
        ep::Number,
        en::Number,
        sd::Option{<:VecNum} = nothing,
        mu::Option{<:VecNum} = nothing
    ) -> SignedL1UncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(ep)` and `ep >= 0`.
  - `isfinite(en)` and `en >= 0`.
  - If `sd` is not `nothing`: `!isempty(sd)`, `all(isfinite, sd)` and `all(sd .>= 0)`. A zero scale fixes its entry at the estimate.
  - If `mu` is not `nothing`: `!isempty(mu)` and `!any(isinf, mu)`. A `NaN` marks an asset outside the Investable Mask, as in the centre `val` of the other sets.
  - If `sd` and `mu` are not `nothing`: `length(mu) == length(sd)`.

# Examples

```jldoctest
julia> SignedL1UncertaintySet(; ep = 0.1, en = 0.2)
SignedL1UncertaintySet
  ep ┼ Float64: 0.1
  en ┼ Float64: 0.2
  sd ┼ nothing
  mu ┴ nothing
```

# Related

  - [`L1UncertaintySet`](@ref): the joint set. Under a long-only budget all the weights have the same sign, and the two worst cases agree. The joint set is then the simpler choice.
  - [`SignedL1UncertaintySetAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref): the estimator that fits this set. It fits a set on the mean only, and [`sigma_ucs`](@ref) throws an `ArgumentError` on it.
  - [`mu_ucs`](@ref): a set that it fits holds, in `mu`, the characteristic vector that the Radius Calibration of the two radii read. The consumer then bounds that vector and no other.
  - [`AbstractUncertaintySetResult`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:quintile])
"""
@concrete struct SignedL1UncertaintySet <: AbstractUncertaintySetResult
    """
    $(field_dict[:ep_ucs])
    """
    ep
    """
    $(field_dict[:en_ucs])
    """
    en
    """
    $(field_dict[:sd_ucs])
    """
    sd
    """
    $(field_dict[:mu_l1_ucs])
    """
    mu
    function SignedL1UncertaintySet(ep::Number, en::Number, sd::Option{<:VecNum},
                                    mu::Option{<:VecNum})
        @argcheck(isfinite(ep) && ep >= zero(ep),
                  DomainError(ep, "ep must be finite and >= 0"))
        @argcheck(isfinite(en) && en >= zero(en),
                  DomainError(en, "en must be finite and >= 0"))
        if isa(sd, VecNum)
            @argcheck(!isempty(sd), IsEmptyError("sd cannot be empty"))
            @argcheck(all(isfinite, sd),
                      IsNonFiniteError("all entries of sd must be finite"))
            @argcheck(all(x -> x >= zero(x), sd),
                      DomainError(sd, "all entries of sd must be >= 0"))
        end
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(!any(isinf, mu),
                      IsNonFiniteError("mu must not have an infinite entry. A NaN marks an asset outside the Investable Mask and is accepted."))
            if isa(sd, VecNum)
                @argcheck(length(mu) == length(sd),
                          DimensionMismatch("mu ($(length(mu))) must match sd ($(length(sd)))"))
            end
        end
        return new{typeof(ep), typeof(en), typeof(sd), typeof(mu)}(ep, en, sd, mu)
    end
end
function SignedL1UncertaintySet(ep::Number, en::Number,
                                sd::Option{<:VecNum})::SignedL1UncertaintySet
    return SignedL1UncertaintySet(ep, en, sd, nothing)
end
function SignedL1UncertaintySet(; ep::Number, en::Number, sd::Option{<:VecNum} = nothing,
                                mu::Option{<:VecNum} = nothing)::SignedL1UncertaintySet
    return SignedL1UncertaintySet(ep, en, sd, mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`SignedL1UncertaintySet`](@ref) restricted to the asset indices `i`.

The two radii are scalar budgets for the universe, and the view keeps them unchanged. The view takes the entries of `sd` and `mu` at `i`, in the order of `i`. The restricted set then indexes the assets in the order that the optimiser gives.

# Algorithm

 1. Take `nothing_scalar_array_view(risk_ucs.sd, i)`, the scaling of the selected assets. A `nothing` stays `nothing`.
 2. Take `nothing_scalar_array_view(risk_ucs.mu, i)`, the characteristic vector of the same assets. A `nothing` stays `nothing`.
 3. Build a [`SignedL1UncertaintySet`](@ref) from `risk_ucs.ep`, `risk_ucs.en` and the two views. The step rescales neither radius, because each radius is the budget for the total error of its sign over the assets that the set covers.

# Arguments

  - `risk_ucs`: Signed ``\\ell_1`` uncertainty set.
  - `i`: Index of the cluster or of the assets.
  - `args...`: More positional arguments, which the method ignores.

# Returns

  - `risk_ucs::SignedL1UncertaintySet`: The set restricted to `i`.

# Examples

```jldoctest
julia> ucs = SignedL1UncertaintySet(; ep = 0.4, en = 0.6, sd = [0.1, 0.2, 0.3],
                                    mu = [0.05, 0.03, 0.01]);

julia> PortfolioOptimisers.port_opt_view(ucs, [1, 3])
SignedL1UncertaintySet
  ep ┼ Float64: 0.4
  en ┼ Float64: 0.6
  sd ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}: [0.1, 0.3]
  mu ┴ SubArray{Float64, 1, Vector{Float64}, Tuple{Vector{Int64}}, false}: [0.05, 0.01]
```

# Related

  - [`SignedL1UncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::SignedL1UncertaintySet, i, args...)::SignedL1UncertaintySet
    return SignedL1UncertaintySet(; ep = risk_ucs.ep, en = risk_ucs.en,
                                  sd = nothing_scalar_array_view(risk_ucs.sd, i),
                                  mu = nothing_scalar_array_view(risk_ucs.mu, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Writes a [`SignedL1UncertaintySet`](@ref) fitted on the Investable Mask back onto the full asset universe.

The expansion is the inverse of [`port_opt_view`](@ref), and it writes the same frame as the [`L1UncertaintySet`](@ref) method. The two radii are budgets for the total error of each sign over the assets that the set covers, so they pass through. The scale `sd` is geometry, so its frame is zero. A zero scale fixes the entry at its estimate, and the two positive parts of the worst case then read only the assets inside the mask. A `sd` of `nothing` passes through. The centre `mu` is the full `pr.mu`, with its `NaN` entries.

# Algorithm

 1. If `set.sd` is `nothing`, keep `nothing`. Else, make a zero frame of `length(imsk)` entries, and write `set.sd` at the entries that the mask keeps.
 2. Build a [`SignedL1UncertaintySet`](@ref) from `set.ep`, `set.en` and the frame, with `pr.mu` as `mu`.

# Arguments

  - `set`: Signed ``\\ell_1`` uncertainty set, fitted on the reduced prior.
  - `imsk`: The Investable Mask of the full prior.
  - $(arg_dict[:pr])

# Returns

  - `set::SignedL1UncertaintySet`: The set on the full asset universe.

# Related

  - [`SignedL1UncertaintySet`](@ref)
  - [`investable_ucs_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function expand_investable_ucs(set::SignedL1UncertaintySet, imsk::BitVector,
                               pr::AbstractPriorResult)::SignedL1UncertaintySet
    sd = if isnothing(set.sd)
        nothing
    else
        frame = zeros(eltype(set.sd), length(imsk))
        frame[imsk] = set.sd
        frame
    end
    return SignedL1UncertaintySet(; ep = set.ep, en = set.en, sd = sd, mu = pr.mu)
end
"""
$(DocStringExtensions.TYPEDEF)

Shape algorithm that selects the joint ``\\ell_1`` uncertainty set.

`scaled` selects one of two sets of [quintile](@cite). `false` gives ``\\mathcal{S}``, Equation 5 of [quintile](@cite), and `true` gives ``\\mathcal{A}_1``, its Equation 18. The default is `false`, because ``\\mathcal{S}`` is the base set of [quintile](@cite). Its closed forms give the ``1/N`` portfolio and the quintile portfolio. The default of [`SignedL1UncertaintySetAlgorithm`](@ref) is `true`, because [quintile](@cite) defines the signed set only in the scaled form.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    L1UncertaintySetAlgorithm(;
        method::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
        scaled::Bool = false,
        paired::Bool = false
    ) -> L1UncertaintySetAlgorithm

Keywords correspond to the struct's fields.

## Validation

  - If `method` is a `Number`: `isfinite(method)` and `method >= 0`.

# Examples

```jldoctest
julia> L1UncertaintySetAlgorithm()
L1UncertaintySetAlgorithm
  method ┼ ActiveAssetsUncertaintyAlgorithm
         │   active ┴ Float64: 0.2
  scaled ┼ Bool: false
  paired ┴ Bool: false
```

# Related

  - [`L1UncertaintySet`](@ref)
  - [`SignedL1UncertaintySetAlgorithm`](@ref)
  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`l1_resolve_eps`](@ref): the function that reads `method` and `paired` and returns the radius.

# References

  - $(ref_dict[:quintile])
"""
@concrete struct L1UncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm
    """
    $(field_dict[:method_l1_ucs])
    """
    method
    """
    $(field_dict[:scaled_ucs])
    """
    scaled
    """
    `paired`: Selects the closed form that calibrates the radius when `method` is an [`AbstractUncertaintyEpsAlgorithm`](@ref). The number of assets that a radius activates depends on the signs that the problem lets the weights take. An uncertainty set cannot read the problem, so the caller must state it. `false` selects the long-only ladder, for a problem with `w >= 0` and `bgt = 1`, as in Corollaries 4 and 11 of [quintile](@cite). `true` selects the paired ladder, for a dollar-neutral problem with `bgt = 0` and `sbgt = 1/2`. In that problem the assets become active in pairs of one long asset and one short asset, as in Corollary 7. The wrong value gives a wrong radius, but the optimisation stays correct. `paired` has no effect when `method` is a number.
    """
    paired
    function L1UncertaintySetAlgorithm(method::Num_UcSEps, scaled::Bool, paired::Bool)
        if isa(method, Number)
            @argcheck(isfinite(method) && method >= zero(method),
                      DomainError(method, "method must be finite and >= 0"))
        end
        return new{typeof(method), typeof(scaled), typeof(paired)}(method, scaled, paired)
    end
end
function L1UncertaintySetAlgorithm(;
                                   method::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
                                   scaled::Bool = false,
                                   paired::Bool = false)::L1UncertaintySetAlgorithm
    return L1UncertaintySetAlgorithm(method, scaled, paired)
end
"""
$(DocStringExtensions.TYPEDEF)

Shape algorithm that selects the signed ``\\ell_1`` uncertainty set.

The construction of [quintile](@cite) sets the two ladders, so this algorithm has no `paired` field. ``\\epsilon_{-}`` is the radius for the long leg, and its calibration reads the top of the ranking. ``\\epsilon_{+}`` is the radius for the short leg, and its calibration reads the bottom of the ranking. Corollary 13 of [quintile](@cite) states this. The default of `scaled` is `true`, because [quintile](@cite) defines ``\\mathcal{A}_2`` and states Corollary 13 only in the form scaled by volatility. The default of [`L1UncertaintySetAlgorithm`](@ref) is `false`, because its base set has no scaling.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SignedL1UncertaintySetAlgorithm(;
        mp::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
        mm::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
        scaled::Bool = true
    ) -> SignedL1UncertaintySetAlgorithm

Keywords correspond to the struct's fields.

## Validation

  - If `mp` is a `Number`: `isfinite(mp)` and `mp >= 0`.
  - If `mm` is a `Number`: `isfinite(mm)` and `mm >= 0`.

# Examples

```jldoctest
julia> SignedL1UncertaintySetAlgorithm()
SignedL1UncertaintySetAlgorithm
      mp ┼ ActiveAssetsUncertaintyAlgorithm
         │   active ┴ Float64: 0.2
      mm ┼ ActiveAssetsUncertaintyAlgorithm
         │   active ┴ Float64: 0.2
  scaled ┴ Bool: true
```

# Related

  - [`SignedL1UncertaintySet`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
  - [`AbstractUncertaintySetAlgorithm`](@ref)
  - [`l1_resolve_eps`](@ref): the function that reads `mp` and `mm` and returns each radius.

# References

  - $(ref_dict[:quintile])
"""
@concrete struct SignedL1UncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm
    """
    $(field_dict[:mp_ucs])
    """
    mp
    """
    $(field_dict[:mm_ucs])
    """
    mm
    """
    $(field_dict[:scaled_ucs])
    """
    scaled
    function SignedL1UncertaintySetAlgorithm(mp::Num_UcSEps, mm::Num_UcSEps, scaled::Bool)
        if isa(mp, Number)
            @argcheck(isfinite(mp) && mp >= zero(mp),
                      DomainError(mp, "mp must be finite and >= 0"))
        end
        if isa(mm, Number)
            @argcheck(isfinite(mm) && mm >= zero(mm),
                      DomainError(mm, "mm must be finite and >= 0"))
        end
        return new{typeof(mp), typeof(mm), typeof(scaled)}(mp, mm, scaled)
    end
end
function SignedL1UncertaintySetAlgorithm(;
                                         mp::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
                                         mm::Num_UcSEps = ActiveAssetsUncertaintyAlgorithm(),
                                         scaled::Bool = true)::SignedL1UncertaintySetAlgorithm
    return SignedL1UncertaintySetAlgorithm(mp, mm, scaled)
end
"""
$(DocStringExtensions.TYPEDEF)

Fits an ``\\ell_1`` uncertainty set with a calibrated radius on the characteristic vector, for the mean only.

`pe` gives the characteristic vector ``\\hat{\\boldsymbol{\\mu}}``. When the shape algorithm is `scaled`, it also gives the scaling of each asset, ``\\hat{\\boldsymbol{\\sigma}} = \\sqrt{\\mathrm{diag}(\\hat{\\mathbf{\\Sigma}})}``. The sets are the robust sets of [quintile](@cite) for the best characteristic. The maximum of the worst-case characteristic over one of these sets is a robust optimisation problem. Its exact solutions are the ``1/N`` portfolio, the quintile portfolio and the inverse-volatility portfolio, and the radius sets the number of assets held. For this reason the library has no separate quintile optimiser. Use the set in [`MeanRisk`](@ref):

```julia
MeanRisk(; r = NoRisk(), obj = MaximumReturn(),
         opt = JuMPOptimiser(; slv = slv, bgt = 1.0,
                             ret = ArithmeticReturn(; ucs = CharacteristicUncertaintySet())))
```

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CharacteristicUncertaintySet(;
        pe::Option{<:AbstractLowOrderPriorEstimator} = EmpiricalPrior(),
        alg::AbstractUncertaintySetAlgorithm = L1UncertaintySetAlgorithm()
    ) -> CharacteristicUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `alg` must be an [`L1UncertaintySetAlgorithm`](@ref) or a [`SignedL1UncertaintySetAlgorithm`](@ref).
  - [`ucs`](@ref) and [`sigma_ucs`](@ref) always throw an `ArgumentError` on this estimator. The ``\\ell_1`` ball bounds a characteristic vector, and [quintile](@cite) defines no covariance analogue. So neither function has a covariance set to return.

# Examples

The estimator calibrates the radius. Three assets with the sample means ``0.125``, ``0.0625`` and ``0.03125`` give the Activation Ladder ``[0, 0.0625, 0.125]``. The default `active` of `0.2` gives a target of one active asset, so the radius is the midpoint of the first and second rungs:

```jldoctest
julia> X = [0.1875 0.09375 0.0625
            0.0625 0.03125 0.0];

julia> mu_ucs(CharacteristicUncertaintySet(), X)
L1UncertaintySet
  eps ┼ Float64: 0.03125
   sd ┼ nothing
   mu ┴ Vector{Float64}: [0.125, 0.0625, 0.03125]
```

# Related

  - [`L1UncertaintySet`](@ref)
  - [`SignedL1UncertaintySet`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`SignedL1UncertaintySetAlgorithm`](@ref)
  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`mu_ucs`](@ref): the only verb of the ucs Triple that returns a set for this estimator.
  - [`StandardDeviationExpectedReturns`](@ref): the characteristic does not have to be an expected return. A prior that uses this estimator ranks the assets by volatility, with the highest volatility first. Section VI-C of [quintile](@cite) puts the lowest volatility first, for the Low Volatility factor of its Table III. That ranking needs the negative of the volatility as the characteristic.
  - [`AbstractUncertaintySetEstimator`](@ref)

# References

  - $(ref_dict[:quintile])
"""
@concrete struct CharacteristicUncertaintySet <: AbstractUncertaintySetEstimator
    """
    $(field_dict[:pe_ucs])
    """
    pe
    """
    `alg`: Shape algorithm, an [`L1UncertaintySetAlgorithm`](@ref) or a [`SignedL1UncertaintySetAlgorithm`](@ref).
    """
    alg
    function CharacteristicUncertaintySet(pe::Option{<:AbstractLowOrderPriorEstimator},
                                          alg::AbstractUncertaintySetAlgorithm)
        @argcheck(isa(alg,
                      Union{L1UncertaintySetAlgorithm, SignedL1UncertaintySetAlgorithm}),
                  ArgumentError("alg must be an L1UncertaintySetAlgorithm or a SignedL1UncertaintySetAlgorithm, got $(typeof(alg))"))
        return new{typeof(pe), typeof(alg)}(pe, alg)
    end
end
function CharacteristicUncertaintySet(;
                                      pe::Option{<:AbstractLowOrderPriorEstimator} = EmpiricalPrior(),
                                      alg::AbstractUncertaintySetAlgorithm = L1UncertaintySetAlgorithm())::CharacteristicUncertaintySet
    return CharacteristicUncertaintySet(pe, alg)
end
"""
    l1_activation_ladder(mu::VecNum, sd::Option{<:VecNum})

Return the vector `g` whose `k`-th entry is the radius at which the `k`-th asset becomes active in the long-only problem.

`mu` must be sorted in non-increasing order. If `sd` is a vector, it must have the same permutation.

# Mathematical definition

```math
\\begin{align}
L_k &= \\sum_{i=1}^{k} \\frac{\\hat{\\mu}_i - \\hat{\\mu}_k}{\\sigma_i}\\,, \\quad k = 1,\\, \\ldots,\\, N\\,.
\\end{align}
```

Where:

  - ``L_k``: Activation threshold of the ``k``-th asset of the ranking.
  - $(math_dict[:mu_hat_i_rank])
  - $(math_dict[:sigma_i_ucs])
  - $(math_dict[:N])

``L_k`` is the threshold of Lemma 2 of [quintile](@cite) when ``\\sigma_i = 1``, and of its Lemma 9 when the ladder is scaled. Both lemmas state that the number of active assets is the largest ``k`` with ``L_k < \\epsilon``. So a radius in the open interval ``(L_q,\\, L_{q+1})`` activates exactly ``q`` assets. The active weights are equal under Lemma 2, and they are inverse-volatility weights under Lemma 9. Corollaries 4 and 11 of [quintile](@cite) apply this result at the quintile.

The sequence has two properties. Every summand of ``L_1`` is zero, so ``L_1 = 0``, and a radius above zero always activates at least one asset. The difference of two adjacent rungs is ``L_{k+1} - L_k = \\sum_{i=1}^{k} (\\hat{\\mu}_k - \\hat{\\mu}_{k+1}) / \\sigma_i \\geq 0``, because the ranking is non-increasing. The sum stops at ``k`` because the summand ``i = k+1`` of ``L_{k+1}`` is zero. So the sequence is non-decreasing, and two adjacent rungs bracket each count.

Two adjacent rungs are equal exactly when ``\\hat{\\mu}_k = \\hat{\\mu}_{k+1}``. [quintile](@cite) admits such a tie. Its Assumption 1 excludes only a radius equal to a rung, so that the number of active assets is unique.

# Algorithm

 1. Read `N = length(mu)`, the number of rungs to build.
 2. For each `k` in `1:N`, sum ``(\\hat{\\mu}_i - \\hat{\\mu}_k) / \\sigma_i`` over `i` in `1:k`. Use ``\\sigma_i = 1`` when `sd` is `nothing`. The sum is the `k`-th rung.

# Arguments

  - `mu`: Characteristic vector, sorted in non-increasing order.
  - `sd`: Scaling of each asset in the same permutation, or `nothing` for a ladder with no scaling.

# Returns

  - `g::Vector{<:Number}`: The Activation Ladder, with one rung for each asset.

# Examples

```jldoctest
julia> PortfolioOptimisers.l1_activation_ladder([0.125, 0.0625, 0.03125], nothing)
3-element Vector{Float64}:
 0.0
 0.0625
 0.125

julia> PortfolioOptimisers.l1_activation_ladder([0.125, 0.0625, 0.03125], [0.5, 0.25, 0.125])
3-element Vector{Float64}:
 0.0
 0.125
 0.3125
```

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`l1_eps_from_ladder`](@ref)
  - [`l1_resolve_eps`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)

# References

  - $(ref_dict[:quintile])
"""
function l1_activation_ladder(mu::VecNum, sd::Option{<:VecNum})
    N = length(mu)
    return [sum(i -> (mu[i] - mu[k]) / (isnothing(sd) ? one(eltype(mu)) : sd[i]), 1:k)
            for k in 1:N]
end
"""
    l1_active_count(active::Union{Integer, AbstractFloat}, N::Int)

Convert an `active` target, a count or a fraction of the universe, to a number of assets in `1:N`.

The type of `active` selects the rule. The function clamps a target outside `1:N`, and it does not throw an error. A count above `N` becomes `N`, and a fraction that rounds to zero becomes `1`.

# Algorithm

 1. If `active` is an `Integer`, use it as the count `q`.
 2. Else, take `q = round(Int, active * N)`, the fraction of the universe rounded to the nearest asset.
 3. Return `clamp(q, 1, N)`, a count that the ranking can give.

# Arguments

  - `active`: Target number of active assets. An `Integer` is a count, and an `AbstractFloat` is a fraction of the universe.
  - `N`: Number of assets in the universe.

# Returns

  - `q::Int`: The target count, in `1:N`.

# Examples

```jldoctest
julia> PortfolioOptimisers.l1_active_count(3, 10)
3

julia> PortfolioOptimisers.l1_active_count(0.2, 10)
2

julia> PortfolioOptimisers.l1_active_count(20, 10)
10
```

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`l1_resolve_eps`](@ref)

# References

  - $(ref_dict[:quintile])
"""
function l1_active_count(active::Union{Integer, AbstractFloat}, N::Int)
    q = isa(active, Integer) ? Int(active) : round(Int, active * N)
    return clamp(q, 1, N)
end
"""
    l1_eps_from_ladder(method, ladder::VecNum, q::Int)

Find the radius of a [`Num_UcSEps`](@ref) on an Activation Ladder.

The method that takes a `Number` returns the radius that the caller gave. It has no procedure, so the `# Algorithm` and `# Validation` sections are not about it. The method that takes an [`ActiveAssetsUncertaintyAlgorithm`](@ref) returns the midpoint of the open interval ``(L_q,\\, L_{q+1})``. The closed forms of [quintile](@cite) need a radius in that interval for exactly `q` active entries.

Every radius above the last rung activates every entry. So the top of the ladder has no interval with a midpoint, and the method adds half of the last increment to the last rung. A ladder of one rung has no increment, and the method returns `one(eltype(ladder))`. That radius has no effect. A ladder of one rung belongs to a universe of one asset, where the budget sets the single weight. Every radius then gives the same portfolio.

# Algorithm

 1. Read `L = length(ladder)`, the number of rungs.
 2. Clamp `q` to `1:L`, so that the target is a rung of the ladder.
 3. If `q < L`, read `lo, hi = ladder[q], ladder[q+1]`, the two rungs on each side of the target. Return their midpoint, a radius inside the open interval.
 4. Else, `q == L`. If `L == 1`, return `one(eltype(ladder))`, because the ladder has no increment. Else, return `ladder[L] + (ladder[L] - ladder[L-1]) / 2`, half of the last increment above the last rung.

# Arguments

  - `method`: A radius as a `Number`, or an [`ActiveAssetsUncertaintyAlgorithm`](@ref).
  - `ladder`: Activation Ladder, as [`l1_activation_ladder`](@ref) returns it.
  - `q`: Target number of active entries.
  - `args...`: More positional arguments, which the `Number` method ignores.

# Validation

  - `length(ladder) >= 1`. Else, the method throws an `IsEmptyError`.
  - `ladder[q+1] > ladder[q]` when `q < L`. Else, the method throws a `DomainError`. The two rungs are equal when the `q`-th and the `(q+1)`-th entries of the ranking are equal. The interval that the closed form needs is then empty, and no radius gives the target count. [quintile](@cite) admits ties, and its Assumption 1 excludes only a radius equal to a rung. The midpoint needs a strictly increasing ladder, so the message names two fixes. The caller can break the tie, or give a radius.

# Returns

  - `eps::Number`: The radius.

# Examples

```jldoctest
julia> PortfolioOptimisers.l1_eps_from_ladder(0.5)
0.5

julia> alg = ActiveAssetsUncertaintyAlgorithm(; active = 2);

julia> PortfolioOptimisers.l1_eps_from_ladder(alg, [0.0, 0.0625, 0.125], 2)
0.09375

julia> PortfolioOptimisers.l1_eps_from_ladder(alg, [0.0, 0.0625, 0.125], 3)
0.15625

julia> PortfolioOptimisers.l1_eps_from_ladder(alg, [0.0], 1)
1.0
```

# Related

  - [`ActiveAssetsUncertaintyAlgorithm`](@ref)
  - [`l1_activation_ladder`](@ref)
  - [`l1_resolve_eps`](@ref)

# References

  - $(ref_dict[:quintile])
"""
function l1_eps_from_ladder(method::Number, args...)
    return method
end
function l1_eps_from_ladder(method::ActiveAssetsUncertaintyAlgorithm, ladder::VecNum,
                            q::Int)
    L = length(ladder)
    @argcheck(L >= 1, IsEmptyError("activation ladder cannot be empty"))
    q = clamp(q, 1, L)
    if q < L
        lo, hi = ladder[q], ladder[q+1]
        @argcheck(hi > lo,
                  DomainError(method.active,
                              "cannot calibrate a radius for $(q) active entries: the characteristic has ties across the cut, so the interval ($(lo), $(hi)) required by the closed form is empty. Break the tie, or pass an explicit radius."))
        return (lo + hi) / 2
    end
    # Top of the ladder: any radius above ladder[end] activates everything.
    return L == 1 ? one(eltype(ladder)) : ladder[L] + (ladder[L] - ladder[L-1]) / 2
end
"""
    l1_resolve_eps(method, mus::VecNum, sds::Option{<:VecNum}, paired::Bool)

Find the radius of a [`Num_UcSEps`](@ref) on the characteristic vector.

`mus` must be sorted in non-increasing order, and `sds` must have the same permutation. The method that takes a `Number` returns the radius that the caller gave. It has no procedure, so the `# Algorithm` and `# Validation` sections are not about it. The method that takes an [`ActiveAssetsUncertaintyAlgorithm`](@ref) calibrates the radius on the long-only ladder when `paired` is `false`, and on the paired ladder when `paired` is `true`.

# Mathematical definition

The paired ladder is the sum of the ladder of the ranking and the ladder of the reversed, negated ranking. With ``\\sigma_i = 1`` it is the threshold of Lemma 6 of [quintile](@cite). The scaled form adds the two scaled ladders of Corollary 13 of [quintile](@cite), which states them for two radii on ``\\mathcal{A}_2``. [quintile](@cite) states no lemma for one radius on the scaled set. So the scaled form is an extension of this library, and no lemma proves the number of active assets that it gives:

```math
\\begin{align}
F_m &= \\sum_{i=1}^{m} \\frac{\\hat{\\mu}_i - \\hat{\\mu}_m}{\\sigma_i}
     + \\sum_{j=1}^{m} \\frac{\\hat{\\mu}_{N-m+1} - \\hat{\\mu}_{N-j+1}}{\\sigma_{N-j+1}}\\,,
     \\quad m = 1,\\, \\ldots,\\, \\left\\lfloor N/2 \\right\\rfloor\\,.
\\end{align}
```

Where:

  - ``F_m``: Activation threshold of the ``m``-th pair of one long asset and one short asset.
  - $(math_dict[:mu_hat_i_rank])
  - $(math_dict[:sigma_i_ucs])
  - $(math_dict[:N])

A radius in ``(F_q,\\, F_{q+1})`` makes the top ``q`` assets long and the bottom ``q`` assets short, so it activates ``2q`` assets. Corollary 7 states this with ``\\sigma_i = 1``. Lemma 5, also with ``\\sigma_i = 1``, gives the reason that the ladder stops at ``\\lfloor N/2 \\rfloor``. An optimal dollar-neutral solution satisfies ``w_i = -w_{N+1-i}``. With an odd ``N``, the middle asset is its own pair, so its weight satisfies ``w = -w`` and the asset is never active. The largest count that the paired branch can reach is ``2 \\lfloor N/2 \\rfloor``, which is ``N-1`` when ``N`` is odd.

# Algorithm

 1. Check `mus` and `sds`, as `# Validation` states.
 2. Read `N = length(mus)`, the size of the universe.
 3. If `paired` is `false`, build the long-only ladder with [`l1_activation_ladder`](@ref), and convert `method.active` to a count with [`l1_active_count`](@ref). Return the radius that [`l1_eps_from_ladder`](@ref) finds from the ladder and the count.
 4. Else, read `half = N ÷ 2`, the number of pairs that the universe has.
 5. Add the ladder of `mus` to the ladder of the reversed, negated ranking. Keep the first `half` entries as `ladder`, the paired ladder.
 6. Convert the target to a number of pairs. An `Integer` `active` is a count of assets, and it gives `max(active ÷ 2, 1)` pairs. An `AbstractFloat` `active` is a fraction of the universe, and it gives `clamp(round(Int, active * N / 2), 1, half)` pairs. A count and the equal fraction give the same number of pairs when the count is even. For an odd count, the integer division rounds down and the fraction rounds to the nearest integer. So the two can differ by one pair. For example, with `N = 10`, `active = 7` gives three pairs, and `active = 0.7` gives four.
 7. Return the radius that [`l1_eps_from_ladder`](@ref) finds from `ladder` and the number of pairs.

# Arguments

  - `method`: A radius as a `Number`, or an [`ActiveAssetsUncertaintyAlgorithm`](@ref).
  - `mus`: Characteristic vector, sorted in non-increasing order.
  - `sds`: Scaling of each asset in the same permutation, or `nothing`.
  - `paired`: If `true`, the calibration reads the paired ladder. If `false`, it reads the long-only ladder.

# Validation

  - `all(isfinite, mus)`. Else, the method throws an `IsNonFiniteError`. A `NaN` entry makes every rung that reads it `NaN`, and the tie check of [`l1_eps_from_ladder`](@ref) then names the wrong cause. A prior fitted on a point-in-time Asset Panel has `NaN` outside its Investable Mask, and [`mu_ucs`](@ref) reduces it to the mask before the calibration.
  - If `sds` is not `nothing`: every entry is finite and `> 0`. Else, the method throws a `DomainError`. The ladder divides by each entry. A set can hold a zero scale, but the closed form of the radius cannot read one.
  - `N ÷ 2 >= 1` when `paired` is `true`. Else, the method throws an `ArgumentError`. One asset cannot make a pair of a long asset and a short asset, so it has no paired ladder.

# Returns

  - `eps::Number`: The radius.

# Examples

```jldoctest
julia> PortfolioOptimisers.l1_resolve_eps(0.25)
0.25

julia> alg = ActiveAssetsUncertaintyAlgorithm(; active = 2);

julia> mus = [0.125, 0.0625, 0.03125, 0.0];

julia> PortfolioOptimisers.l1_resolve_eps(alg, mus, nothing, false)
0.09375

julia> PortfolioOptimisers.l1_resolve_eps(alg, mus, nothing, true)
0.046875
```

# Related

  - [`l1_activation_ladder`](@ref)
  - [`l1_active_count`](@ref)
  - [`l1_eps_from_ladder`](@ref)
  - [`L1UncertaintySetAlgorithm`](@ref)
  - [`SignedL1UncertaintySetAlgorithm`](@ref)

# References

  - $(ref_dict[:quintile])
"""
function l1_resolve_eps(method::Number, args...)
    return method
end
function l1_resolve_eps(method::ActiveAssetsUncertaintyAlgorithm, mus::VecNum,
                        sds::Option{<:VecNum}, paired::Bool)
    @argcheck(all(isfinite, mus),
              IsNonFiniteError("cannot calibrate a radius on a characteristic with a non-finite entry: a NaN makes every rung of the activation ladder that reads it NaN. Reduce the prior to its Investable Mask, or pass an explicit radius."))
    # `nothing` is the unit scale, so it has no entry to refuse.
    @argcheck(all(x -> isfinite(x) && x > zero(x), something(sds, ())),
              DomainError(sds,
                          "cannot calibrate a radius on a scale with an entry that is not finite and > 0: the activation ladder divides by each entry. Pass an explicit radius, or set scaled = false."))
    N = length(mus)
    if !paired
        return l1_eps_from_ladder(method, l1_activation_ladder(mus, sds),
                                  l1_active_count(method.active, N))
    end
    half = N ÷ 2
    @argcheck(half >= 1,
              ArgumentError("a paired radius calibration needs at least 2 assets, got $N"))
    ladder = (l1_activation_ladder(mus, sds)+l1_activation_ladder(reverse(-mus),
                                                                  if isnothing(sds)
                                                                      nothing
                                                                  else
                                                                      reverse(sds)
                                                                  end))[1:half]
    # `active` counts assets, but each paired ladder entry activates a long/short pair.
    q = if isa(method.active, Integer)
        max(Int(method.active) ÷ 2, 1)
    else
        clamp(round(Int, method.active * N / 2), 1, half)
    end
    return l1_eps_from_ladder(method, ladder, q)
end
"""
    mu_ucs(ue::CharacteristicUncertaintySet, X::MatNum,
           F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
    mu_ucs(ue::CharacteristicUncertaintySet{Nothing}, pr::AbstractPriorResult; rd = nothing, kwargs...)

Fits an ``\\ell_1`` uncertainty set on the characteristic vector, from returns data or from a Prior Result.

The two signatures are the two routes of a set with a `pe` field, and the two routes end in the same steps. The returns-data method fits `ue.pe` once through [`ucs_prior`](@ref), which throws an `ArgumentError` when `ue.pe` is `nothing`. The method then gives the Prior Result to the prior-result method of the same set with `pe = nothing`. That method selects the shape of the set by dispatch on `alg`.

The prior-result method has a definition only for a set whose `pe` is `nothing`. Inside an optimiser, `pr` is the prior that the optimiser solves on. So the Radius Calibration reads the same `mu` as the objective.

The calibration reads the ranking, but the set keeps the order of the universe. `mu` and `sd` have one entry for each asset, in the order of the prior. The optimiser indexes the assets in that order, and [`port_opt_view`](@ref) takes its view in that order. Only the ladders read the sorted vector. A ladder is a list of radii, and it holds no index of an asset. The set holds ``\\hat{\\boldsymbol{\\mu}}`` in its `mu` field, so the consumer bounds the characteristic vector that the calibration read.

# Algorithm

The returns-data method does step 1, and then it calls the prior-result method. The two prior-result methods do steps 2 to 5 and step 7 in the same way, and they do step 6 differently.

 1. Fit the prior `pr` with [`ucs_prior`](@ref) on `ue.pe`, `X` and `F`. Call the prior-result method of the set with `pe = nothing`.
 2. Reduce the Prior Result `pr` to the Investable Mask with [`investable_ucs_reduction`](@ref) to get `prr`. Inside an optimiser, `pr` is already reduced, and the step changes nothing. Outside an optimiser, on a prior fitted on a point-in-time Asset Panel, the step makes the reduction that an optimiser makes. The ranking then reads no `NaN`.
 3. Read `alg = ue.alg`, the shape algorithm.
 4. If `alg.scaled` is `true`, take `sd = sqrt.(diag(prr.sigma))`, the scaling of each asset. Else, take `sd = nothing`.
 5. Take `idx = sortperm(prr.mu; rev = true)`, the ranking that sorts the characteristic in non-increasing order. Apply it to `prr.mu` and to `sd`.
 6. Find the radii with [`l1_resolve_eps`](@ref) on the sorted vectors. Build the set from the radii, with `mu = prr.mu` and `sd` in the order of the universe.
 7. Write the set back onto the full universe with [`expand_investable_ucs`](@ref), and return it. A set fitted outside an optimiser then covers the same assets as the prior. A view of the set at the mask gives the reduced fit again.

For an [`L1UncertaintySetAlgorithm`](@ref), step 6 finds the radius `eps` from `alg.method` and `alg.paired`, and builds an [`L1UncertaintySet`](@ref).

For a [`SignedL1UncertaintySetAlgorithm`](@ref), step 6 finds two radii on two ladders, and builds a [`SignedL1UncertaintySet`](@ref). `en` comes from `alg.mm` on the sorted ranking. `ep` comes from `alg.mp` on the reversed, negated ranking, which is the bottom of the same ranking. Corollary 13 of [quintile](@cite) states this construction. Both calls pass `paired = false`, because the construction sets both ladders. The consumer multiplies `en` by ``[\\max_i (\\sigma_i w_i)]_{+}``, and `ep` by ``[\\max_i (-\\sigma_i w_i)]_{+}``. So `en` is the radius for the long leg and `ep` is the radius for the short leg, and each leg matches the end of the ranking that its calibration read.

# Arguments

  - `ue`: Characteristic uncertainty set estimator.
  - `X`: Data matrix, such as the returns.
  - `F`: Factor matrix, or `nothing`. The prior estimator reads it.
  - $(arg_dict[:dims])
  - `pr`: Prior Result for the calibration. The method reads `pr.mu` at the Investable Mask, and `pr.sigma` there when `alg.scaled` is `true`.
  - `rd`: Returns Result that the three-argument form gives with the prior. The method reduces it with the prior, and does not read it.
  - `kwargs...`: More keyword arguments, which the returns-data method gives to the prior estimator. The prior-result method ignores them.

# Returns

  - `mu_ucs::Union{L1UncertaintySet, SignedL1UncertaintySet}`: The uncertainty set, on the full asset universe of `pr`.

# Examples

```jldoctest
julia> X = [0.1875 0.09375 0.0625
            0.0625 0.03125 0.0];

julia> alg = SignedL1UncertaintySetAlgorithm(; scaled = false);

julia> mu_ucs(CharacteristicUncertaintySet(; alg = alg), X)
SignedL1UncertaintySet
  ep ┼ Float64: 0.015625
  en ┼ Float64: 0.03125
  sd ┼ nothing
  mu ┴ Vector{Float64}: [0.125, 0.0625, 0.03125]
```

# Related

  - [`CharacteristicUncertaintySet`](@ref)
  - [`L1UncertaintySet`](@ref)
  - [`SignedL1UncertaintySet`](@ref)
  - [`l1_resolve_eps`](@ref)
  - [`ucs_prior`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:quintile])
"""
function mu_ucs(ue::CharacteristicUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing;
                dims::Int = 1, kwargs...)
    pr = ucs_prior(ue.pe, X, F; dims = dims, kwargs...)
    return mu_ucs(rebuild_estimator(ue, (; pe = nothing)), pr)
end
function mu_ucs(ue::CharacteristicUncertaintySet{Nothing, <:L1UncertaintySetAlgorithm},
                pr::AbstractPriorResult; rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    alg = ue.alg
    sd = alg.scaled ? sqrt.(LinearAlgebra.diag(prr.sigma)) : nothing
    idx = sortperm(prr.mu; rev = true)
    eps = l1_resolve_eps(alg.method, prr.mu[idx], alg.scaled ? sd[idx] : nothing,
                         alg.paired)
    return expand_investable_ucs(L1UncertaintySet(; eps = eps, sd = sd, mu = prr.mu), imsk,
                                 pr)
end
function mu_ucs(ue::CharacteristicUncertaintySet{Nothing,
                                                 <:SignedL1UncertaintySetAlgorithm},
                pr::AbstractPriorResult; rd = nothing, kwargs...)
    imsk, prr, _ = investable_ucs_reduction(pr, rd)
    alg = ue.alg
    sd = alg.scaled ? sqrt.(LinearAlgebra.diag(prr.sigma)) : nothing
    idx = sortperm(prr.mu; rev = true)
    mus = prr.mu[idx]
    sds = isnothing(sd) ? nothing : sd[idx]
    # en governs the long leg, calibrated against the top of the ranking; ep governs the
    # short leg, calibrated against the bottom — i.e. the top of the reversed, negated
    # ranking, which is the same ladder read from the other end (Corollary 13).
    en = l1_resolve_eps(alg.mm, mus, sds, false)
    ep = l1_resolve_eps(alg.mp, reverse(-mus), alg.scaled ? reverse(sds) : nothing, false)
    return expand_investable_ucs(SignedL1UncertaintySet(; ep = ep, en = en, sd = sd,
                                                        mu = prr.mu), imsk, pr)
end
"""
    ucs(ue::CharacteristicUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; kwargs...)
    sigma_ucs(ue::CharacteristicUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; kwargs...)
    ucs(ue::CharacteristicUncertaintySet{Nothing}, pr::AbstractPriorResult; kwargs...)
    sigma_ucs(ue::CharacteristicUncertaintySet{Nothing}, pr::AbstractPriorResult; kwargs...)

Always throw an `ArgumentError`. [`CharacteristicUncertaintySet`](@ref) fits a set on the mean only.

The four methods have no procedure, so they have no `# Algorithm` section. The returns-data pair has the same `(X, F)` signature as the other families, not a catch-all signature. So the [`ReturnsResult`](@ref) methods of the base reach these methods with no ambiguity. The prior-result pair has the prior-result signature of the other families. So a set with no prior of its own reaches the same error through the three-argument route.

# Arguments

  - `ue`: Characteristic uncertainty set estimator.
  - `X`: Data matrix, such as the returns. The methods ignore it.
  - `F`: Factor matrix, or `nothing`. The methods ignore it.
  - `pr`: Prior Result, which the methods ignore.
  - `kwargs...`: More keyword arguments, which the methods ignore.

# Validation

  - All four methods always throw an `ArgumentError`. The ``\\ell_1`` set bounds a characteristic vector, such as the mean, and [quintile](@cite) defines no covariance analogue. So the methods have no covariance set to return. Each message names [`NormalUncertaintySet`](@ref), [`DeltaUncertaintySet`](@ref) and [`ARCHUncertaintySet`](@ref) for a covariance set. The message of `ucs` also names [`mu_ucs`](@ref) for the mean set.

# Returns

  - No method returns. Each method throws.

# Related

  - [`CharacteristicUncertaintySet`](@ref)
  - [`mu_ucs`](@ref)
  - [`NormalUncertaintySet`](@ref)
  - [`DeltaUncertaintySet`](@ref)
  - [`ARCHUncertaintySet`](@ref)

# References

  - $(ref_dict[:quintile])
"""
function ucs(::CharacteristicUncertaintySet, ::MatNum, ::Option{<:MatNum} = nothing;
             kwargs...)
    return throw(ArgumentError("CharacteristicUncertaintySet is mean-only: it bounds a characteristic vector and has no covariance analogue, so it cannot produce the (mu, sigma) pair `ucs` returns. Use `mu_ucs` for the mean set, and NormalUncertaintySet, DeltaUncertaintySet or ARCHUncertaintySet for a covariance set."))
end
function sigma_ucs(::CharacteristicUncertaintySet, ::MatNum, ::Option{<:MatNum} = nothing;
                   kwargs...)
    return throw(ArgumentError("CharacteristicUncertaintySet is mean-only: the l1 set bounds a characteristic vector, and no covariance analogue is defined for it. Use NormalUncertaintySet, DeltaUncertaintySet or ARCHUncertaintySet for a covariance uncertainty set."))
end
function ucs(::CharacteristicUncertaintySet{Nothing}, ::AbstractPriorResult; kwargs...)
    return throw(ArgumentError("CharacteristicUncertaintySet is mean-only: it bounds a characteristic vector and has no covariance analogue, so it cannot produce the (mu, sigma) pair `ucs` returns. Use `mu_ucs` for the mean set, and NormalUncertaintySet, DeltaUncertaintySet or ARCHUncertaintySet for a covariance set."))
end
function sigma_ucs(::CharacteristicUncertaintySet{Nothing}, ::AbstractPriorResult;
                   kwargs...)
    return throw(ArgumentError("CharacteristicUncertaintySet is mean-only: the l1 set bounds a characteristic vector, and no covariance analogue is defined for it. Use NormalUncertaintySet, DeltaUncertaintySet or ARCHUncertaintySet for a covariance uncertainty set."))
end
# A CharacteristicUncertaintySet with no prior of its own reads the prior result it is handed
# (see [`reads_prior_result`](@ref)).
function reads_prior_result(::CharacteristicUncertaintySet{Nothing})::Bool
    return true
end

export ActiveAssetsUncertaintyAlgorithm, L1UncertaintySet, SignedL1UncertaintySet,
       L1UncertaintySetAlgorithm, SignedL1UncertaintySetAlgorithm,
       CharacteristicUncertaintySet
public l1_resolve_eps
