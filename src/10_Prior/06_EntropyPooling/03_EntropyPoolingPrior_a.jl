"""
    ep_evar(x::VecNum, w::VecNum, alpha::Number; args::Tuple = (),
            kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)

Compute the sample entropic value at risk of a loss series and the dual variable that attains it.

`ep_evar` minimises the scalar convex objective of the sample EVaR formula with [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). It is used by the entropy pooling view machinery, which needs both the value (to compare a view against its prior) and the minimiser (to centre the grid of [`GridEntropicValueatRiskView`](@ref)).

# Mathematical definition

```math
\\mathrm{EVaR}_{\\alpha}(X) = \\min_{z > 0} \\; z \\ln\\left(\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/z)}{\\alpha}\\right)\\,.
```

# Algorithm

 1. Normalise the observation probabilities in the logarithmic domain, giving `lw`.
 2. Bracket the dual variable. The upper end `hi` is `(maximum(x) - dot(w, x)) / log(inv(alpha))`, replaced by `eps` of its own type where that is not positive, and the lower end is `hi * zlo_frac`.
 3. Minimise the objective over the bracket with [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). Each evaluation goes through `LogExpFunctions.logsumexp`, so a small `z` does not overflow.
 4. Return the minimum as `evar`, and the minimiser as `z`.

# Arguments

  - `x`: Loss series (`-returns`).
  - `w`: Observation probabilities. Normalised to sum to one.
  - `alpha`: Significance level.
  - $(arg_dict[:optargs]) Left empty it takes `Optim.Brent()`, which is what `Optim.optimize` selects for a bracketed scalar minimisation.
  - $(arg_dict[:optkwargs])
  - `zlo_frac`: Lower end of the bracket of the dual variable, as a fraction of the upper end. `nothing` takes `sqrt(eps(T))` for the element type `T`, which the caller cannot state because the type follows from the data. The upper end is `(maximum(x) - dot(w, x)) / log(inv(alpha))`, above which the objective already exceeds `maximum(x)`, which bounds the EVaR from above. That is a proof, so the upper end is not a knob and only the lower one is.

# Validation

  - `0 < zlo_frac < 1`.
  - The search converges. It is a bracketed scalar minimisation of a convex function, so it fails only under `args` or `kwargs` that stop it early.

# Returns

  - `res::@NamedTuple{evar::Number, z::Number}`: The value and the dual variable that attains it.

## The incremental fit

This prior has no exact incremental recursion, so it takes the online step by **refitting from a sample buffer**: [`Online`](@ref) seeds `cache`, [`partial_fit!`](@ref) appends each observation to it verbatim, and the one-argument [`prior`](@ref) runs this estimator's own batch verb over the rows the buffer kept. The answer is therefore exactly a batch fit over those rows, and a `max_history` on the wrapper windows the whole fit.

`cache` travels the three propagation channels as every partial-fit state does: [`factory`](@ref) carries it unchanged, [`port_opt_view`](@ref) slices it to the selected assets, and [`obs_weights_view`](@ref) drops it, because no slice of a state exists on the observation axis. It is not rendered, because a running buffer is not the configuration a reader looks the type up for.

# Related

  - [`GridEntropicValueatRiskView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`EntropicValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
function ep_evar(x::VecNum, w::VecNum, alpha::Number; args::Tuple = (),
                 kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)
    lw = log.(w)
    lw .-= LogExpFunctions.logsumexp(lw)
    ila = -log(alpha)
    f = function (z)
        return z * (LogExpFunctions.logsumexp(lw .+ x ./ z) + ila)
    end
    hi = (maximum(x) - LinearAlgebra.dot(exp.(lw), x)) / ila
    ehi = eps(typeof(hi))
    hi = ifelse(hi > zero(hi), hi, ehi)
    # The default lower end is the one the element type states, which a caller that holds no
    # data cannot, so `nothing` resolves here rather than in the view that carries it.
    zlo_frac = isnothing(zlo_frac) ? sqrt(ehi) : zlo_frac
    @argcheck(zero(zlo_frac) < zlo_frac < one(zlo_frac),
              DomainError(zlo_frac, "zlo_frac must be in (0, 1)"))
    lo = hi * zlo_frac
    res = Optim.optimize(f, lo, hi, args...; kwargs...)
    @argcheck(Optim.converged(res),
              ErrorException("The search for the sample EVaR did not converge. Relax the `args` and `kwargs` of the view group, or leave them empty to take the defaults."))
    return (; evar = Optim.minimum(res), z = Optim.minimizer(res))
end
"""
    ep_evar_grid_row(x::VecNum, ebar::Number, z::Number)

Build one scaled row of the grid formulation of an entropic value-at-risk view.

`ep_evar_grid_row` returns the coefficients of `exp((x - ebar) / z)` divided by their largest entry, together with the reciprocal of that entry, which the right hand side must be multiplied by. Scaling the row keeps the coefficients in `(0, 1]` however small `z` is, so the row does not overflow before it is built. The bound `alpha * isc` then falls far below one at a small `z`, so an upper-bound row reaches the model divided by that bound too, by [`add_ep_tail_view!`](@ref). A lower-bound row keeps the norm scale of [`ep_add_grid_tail_view!`](@ref).

# Algorithm

 1. Exponentiate the shifted loss series, giving `c`, the unscaled coefficients `exp((x - ebar) / z)`.
 2. Read the largest entry of `c` into `sc`.
 3. Return `c` divided by `sc`, and `isc`, the reciprocal of `sc`.

# Arguments

  - `x`: Loss series (`-returns`).
  - `ebar`: Target entropic value-at-risk.
  - `z`: Grid point of the entropic value-at-risk dual variable.

# Returns

  - `c::VecNum`: Scaled coefficients.
  - `isc::Number`: Scaling factor to apply to the right hand side.

# Related

  - [`GridEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskViewConstraint`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`ep_add_grid_tail_view!`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_evar_grid_row(x::VecNum, ebar::Number, z::Number)
    c = exp.((x .- ebar) ./ z)
    sc = maximum(c)
    return c ./ sc, inv(sc)
end
"""
    ep_rlvar_tail(u::Number, z::Number, kappa::Number)

Evaluate the smallest tail penalty the pair of power cones of one observation allows.

The primal programme of the relativistic value at risk carries two power cones and two non-negative variables per observation. Their sum is minimised out in closed form, which is what turns a point of the primal programme into a row that is linear in the posterior probabilities.

# Mathematical definition

```math
\\begin{align}
\\varphi_{\\kappa}(u, z) &= \\dfrac{\\kappa}{1+\\kappa} \\left(\\dfrac{2\\kappa}{(1+\\kappa) z}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma - u}{2}\\right)^{\\frac{1+\\kappa}{\\kappa}} + \\kappa (1-\\kappa)^{\\frac{1-\\kappa}{\\kappa}} \\left(\\dfrac{z}{2\\kappa}\\right)^{\\frac{1}{\\kappa}} \\left(\\dfrac{\\sigma + u}{2}\\right)^{-\\frac{1-\\kappa}{\\kappa}}\\,,\\\\
\\sigma &= \\sqrt{u^{2} + \\dfrac{(1 - \\kappa^{2}) z^{2}}{\\kappa^{2}}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rlvar_phi])
  - $(math_dict[:rlvar_u])
  - $(math_dict[:rlvar_z])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:rlvar_sigma])

# Arguments

  - `u`: Shifted return of the observation, `t - x` for a loss series `x`.
  - `z`: Dual variable of the primal programme.
  - `kappa`: Deformation parameter, in `(0, 1)`.

# Returns

  - `phi::Number`: Smallest sum of the two tail variables of the observation.

# Related

  - [`ep_rlvar`](@ref)
  - [`ep_rlvar_grid_row`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
function ep_rlvar_tail(u::Number, z::Number, kappa::Number)
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik = inv(kappa)
    sigma = sqrt(u^2 + opk * omk * (z * ik)^2)
    psi = kappa / opk * (2 * kappa / (opk * z))^ik * ((sigma - u) / 2)^(opk * ik)
    theta = kappa * omk^(omk * ik) * (z / (2 * kappa))^ik * ((sigma + u) / 2)^(-omk * ik)
    return psi + theta
end
"""
    ep_rlvar_shift(x::VecNum, w::VecNum, kappa::Number, lnk::Number, z::Number;
                   args::Tuple = (), kwargs::NamedTuple = (;),
                   bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Minimise the primal objective of the relativistic value at risk over its shift variable, at a fixed dual variable.

# Mathematical definition

```math
\\begin{align}
\\underset{t}{\\min} &\\; t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t - x_{j},\\, z)\\,.
\\end{align}
```

Where:

  - $(math_dict[:rlvar_t])
  - $(math_dict[:rlvar_z])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:rlvar_phi])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:kappa_rm])

# Arguments

  - `x`: Loss series (`-returns`).
  - `w`: Observation probabilities, summing to one.
  - `kappa`: Deformation parameter, in `(0, 1)`.
  - `lnk`: Kaniadakis logarithm of `inv(alpha * T)`, from [`kappa_log`](@ref).
  - `z`: Dual variable of the primal programme.
  - $(arg_dict[:optargs]) Left empty it takes `Optim.Brent()`, which is what `Optim.optimize` selects for a bracketed scalar minimisation.
  - $(arg_dict[:optkwargs])
  - `bracket`: Spans of the searches of a relativistic value-at-risk view, or `nothing` to take the one [`RelativisticValueatRiskViewBracket`](@ref) states. This function reads `tspan` alone. It is a margin, not a proof, so widen it where the minimising shift lands on an end of the bracket. [`ep_rlvar`](@ref) reads the other two fields.

# Validation

  - The search converges. It is a bracketed scalar minimisation of a convex function, so it fails only under `args` or `kwargs` that stop it early. A minimiser that lands on an end of the bracket does not fail it: `Optim` reports that end as converged.

# Returns

  - `res::@NamedTuple{risk::Number, t::Number}`: The value at the minimising shift, and that shift.

# Algorithm

 1. Bracket the shift by the loss range widened by `tspan` of its spans on each side. The minimising shift sits near the largest loss, so the bracket holds it with a wide margin.
 2. Minimise the objective over the bracket with [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl).

# Related

  - [`ep_rlvar`](@ref)
  - [`ep_rlvar_tail`](@ref)
  - [`RelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
function ep_rlvar_shift(x::VecNum, w::VecNum, kappa::Number, lnk::Number, z::Number;
                        args::Tuple = (), kwargs::NamedTuple = (;),
                        bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    # `nothing` resolves to the default bracket rather than to bare numbers, so the spans
    # are written once, in the constructor that validates them.
    tspan = something(bracket, RelativisticValueatRiskViewBracket()).tspan
    T = length(x)
    xmin, xmax = extrema(x)
    span = xmax - xmin
    span = ifelse(span > zero(span), span, max(abs(xmax), one(xmax)))
    f = function (t)
        acc = zero(t * one(eltype(w)))
        for j in eachindex(x, w)
            acc += w[j] * ep_rlvar_tail(t - x[j], z, kappa)
        end
        return t + z * lnk + T * acc
    end
    res = Optim.optimize(f, xmin - tspan * span, xmax + tspan * span, args...; kwargs...)
    @argcheck(Optim.converged(res),
              ErrorException("The search for the shift of the sample RLVaR did not converge. Relax the `args` and `kwargs` of the view group, or leave them empty to take the defaults."))
    return (; risk = Optim.minimum(res), t = Optim.minimizer(res))
end
"""
    ep_rlvar(x::VecNum, w::VecNum, alpha::Number, kappa::Number; args::Tuple = (),
             kwargs::NamedTuple = (;), bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Compute the sample relativistic value at risk of a loss series and the primal point that attains it.

`ep_rlvar` minimises the two-variable primal objective of the sample RLVaR, whose per-observation power cones [`ep_rlvar_tail`](@ref) has already minimised out. It is used by the entropy pooling view machinery, which needs both the value (to compare a view against its prior) and the minimiser (to centre the grid of [`GridRelativisticValueatRiskView`](@ref)).

# Mathematical definition

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(X) &= \\underset{t,\\, z > 0}{\\min} \\; t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right) + T \\sum_{j=1}^{T} w_{j} \\varphi_{\\kappa}(t - x_{j},\\, z)\\,.
\\end{align}
```

Where:

  - $(math_dict[:rlvar_stat])
  - $(math_dict[:rlvar_t])
  - $(math_dict[:rlvar_z])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:rlvar_phi])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:kappa_rm])

# Arguments

  - `x`: Loss series (`-returns`).
  - `w`: Observation probabilities. Normalised to sum to one.
  - `alpha`: Significance level.
  - `kappa`: Deformation parameter, in `(0, 1)`.
  - $(arg_dict[:optargs]) It reaches both searches. Left empty it takes `Optim.Brent()`, which is what `Optim.optimize` selects for a bracketed scalar minimisation.
  - $(arg_dict[:optkwargs]) They reach both searches.
  - `bracket`: Spans of the searches, or `nothing` to take the one [`RelativisticValueatRiskViewBracket`](@ref) states. This function reads `log_zlo` and `log_zhi`, the ends of the bracket of the logarithm of the dual variable, as offsets from the logarithm of the loss range. They are a margin, not a proof, so widen one where the minimising dual variable lands on an end of the bracket. [`ep_rlvar_shift`](@ref) reads `tspan`.

# Validation

  - Both searches converge. Each is a bracketed scalar minimisation of a convex function, so one fails only under `args` or `kwargs` that stop it early. A minimiser that lands on an end of a bracket does not fail it: `Optim` reports that end as converged.

# Returns

  - `res::@NamedTuple{rlvar::Number, t::Number, z::Number}`: The value and the primal pair that attains it.

# Algorithm

 1. Minimise over the logarithm of the dual variable with [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl), over a bracket running from `exp(log_zlo)` to `exp(log_zhi)` times the loss range, which is about `2e-9` to about `2e4` under the default bracket. The objective is convex in the pair, so the partial minimum over the shift is convex in the dual variable, and the logarithm is increasing, so the outer minimisation sees a unimodal function.
 2. Minimise over the shift at each candidate dual variable with [`ep_rlvar_shift`](@ref).
 3. Re-run the inner minimisation at the minimising dual variable, so the shift returned is the one that attains the value.

# Related

  - [`ep_rlvar_tail`](@ref)
  - [`ep_rlvar_shift`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`RelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
function ep_rlvar(x::VecNum, w::VecNum, alpha::Number, kappa::Number; args::Tuple = (),
                  kwargs::NamedTuple = (;),
                  bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    # `nothing` resolves to the default bracket rather than to bare numbers, so the spans
    # are written once, in the constructor that validates them.
    bkt = something(bracket, RelativisticValueatRiskViewBracket())
    log_zlo, log_zhi = bkt.log_zlo, bkt.log_zhi
    T = length(x)
    wi = w ./ sum(w)
    lnk = kappa_log(inv(alpha * T), kappa)
    xmin, xmax = extrema(x)
    span = xmax - xmin
    span = ifelse(span > zero(span), span, max(abs(xmax), one(xmax)))
    lspan = log(span)
    res = Optim.optimize(u -> ep_rlvar_shift(x, wi, kappa, lnk, exp(u); args = args,
                                             kwargs = kwargs, bracket = bracket).risk,
                         lspan + log_zlo, lspan + log_zhi, args...; kwargs...)
    @argcheck(Optim.converged(res),
              ErrorException("The search for the dual variable of the sample RLVaR did not converge. Relax the `args` and `kwargs` of the view group, or leave them empty to take the defaults."))
    z = exp(Optim.minimizer(res))
    shift = ep_rlvar_shift(x, wi, kappa, lnk, z; args = args, kwargs = kwargs,
                           bracket = bracket)
    return (; rlvar = shift.risk, t = shift.t, z = z)
end
"""
    ep_rlvar_grid_row(x::VecNum, vbar::Number, t::Number, z::Number, alpha::Number,
                      kappa::Number)

Build one scaled row of the grid formulation of a relativistic value-at-risk view.

`ep_rlvar_grid_row` returns the coefficients `T * phi(t - x, z)` divided by their largest entry, together with the target of the row divided by that same entry. Scaling the row keeps the coefficients in `(0, 1]` however small `z` is, so the row does not overflow before it is built. The target then falls far below one at a small `z`, so an upper-bound row reaches the model divided by that target too, by [`add_ep_tail_view!`](@ref). A lower-bound row keeps the norm scale of [`ep_add_grid_tail_view!`](@ref).

# Arguments

  - `x`: Loss series (`-returns`).
  - `vbar`: Target relativistic value at risk.
  - `t`: Shift variable of the grid point.
  - `z`: Dual variable of the grid point.
  - `alpha`: Significance level.
  - `kappa`: Deformation parameter, in `(0, 1)`.

# Returns

  - `c::VecNum`: Scaled coefficients.
  - `b::Number`: Scaled target the row is compared against.

# Related

  - [`GridRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskViewConstraint`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`ep_add_grid_tail_view!`](@ref)
  - [`ep_tail_views!`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
function ep_rlvar_grid_row(x::VecNum, vbar::Number, t::Number, z::Number, alpha::Number,
                           kappa::Number)
    T = length(x)
    lnk = kappa_log(inv(alpha * T), kappa)
    c = T .* ep_rlvar_tail.(t .- x, z, kappa)
    sc = maximum(c)
    return c ./ sc, (vbar - t - z * lnk) / sc
end
"""
    ep_row_tilt(w::VecNum, c::VecNum, b::Number; iters::Integer = 200)

Tilt a probability vector so that one linear row holds with equality, at the smallest relative entropy.

The row of a grid point is linear in the posterior probabilities, so the posterior that makes it tight and stays closest to the prior is an exponential tilt of the prior along the row's coefficients. It is the entropy pooling answer to that single row, and it needs no solver.

# Mathematical definition

```math
\\begin{align}
q_{j}(\\theta) &= \\dfrac{w_{j} e^{-\\theta c_{j}}}{\\sum_{i=1}^{T} w_{i} e^{-\\theta c_{i}}}\\,,\\\\
\\sum_{j=1}^{T} q_{j}(\\theta) c_{j} &= b\\,.
\\end{align}
```

The row's value under the tilt falls strictly as ``\\theta`` rises, from ``\\max_{j} c_{j}`` to ``\\min_{j} c_{j}`` over the observations with ``w_{j} > 0``, so the tilt exists exactly when ``b`` sits strictly inside that range. A tilt cannot put mass on an observation of zero prior probability, so the coefficient of such an observation bounds nothing.

# Arguments

  - `w`: Prior probabilities, summing to one.
  - `c`: Coefficients of the row.
  - `b`: Value the row is to take.
  - `iters::Integer = 200`: Largest number of bisection steps. The bisection stops on its own when the midpoint stops moving, which for `Float64` happens near step 64, so this binds only a type of higher precision.

# Validation

  - `iters >= 1`.

# Returns

  - `q::Option{VecNum}`: The tilted probabilities, or `nothing` when `b` sits outside the range of `c` over the support of `w` and no tilt attains it, or when the tilted row is not finite.

# Algorithm

 1. Return `nothing` when `b` sits outside the open range of `c` over the observations with `w > 0`.
 2. Bracket the root by doubling the tilt away from zero until the row's value crosses `b`.
 3. Bisect the bracket to the resolution of the floating-point type, or for `iters` steps, whichever comes first.
 4. Return `nothing` when the tilted weights sum to zero or to a value that is not a number, and the normalised weights otherwise.

# Related

  - [`ep_rlvar_anchor`](@ref)
  - [`ep_rlvar_grid_row`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
function ep_row_tilt(w::VecNum, c::VecNum, b::Number; iters::Integer = 200)
    @argcheck(iters >= one(iters), DomainError(iters, "iters must be >= 1"))
    # The range is read over the support of `w`: a tilt leaves a zero prior probability at
    # zero, so a target past the support's extreme sends the doubling below to underflow
    # every `exp`, and the row to `0/0`. Issue #1260.
    lo, hi = extrema(ci for (ci, wi) in zip(c, w) if wi > zero(wi))
    if !(lo < b < hi)
        return nothing
    end
    q = Vector{promote_type(eltype(w), eltype(c), typeof(b))}(undef, length(w))
    # Each call leaves the unnormalised tilt in `q`, so the last call is the answer.
    row = function (th)
        q .= (-th) .* c
        q .= w .* exp.(q .- maximum(q))
        return LinearAlgebra.dot(q, c) / sum(q)
    end
    # The row's value falls as the tilt rises, so one side of zero holds the root. Doubling
    # reaches it, because the value tends to an end of the range of `c` and `b` is inside.
    sgn = ifelse(row(zero(b)) >= b, one(b), -one(b))
    thb = sgn
    while (row(thb) - b) * sgn > zero(b)
        thb *= 2
    end
    tha = zero(b)
    for _ in 1:iters
        thm = (tha + thb) / 2
        if (thm == tha || thm == thb)
            break
        end
        ((row(thm) - b) * sgn > zero(b)) ? (tha = thm) : (thb = thm)
    end
    row(thb)
    # A sum that is zero or not a number means the tilt underflowed.
    return sum(q) > zero(eltype(q)) ? q ./ sum(q) : nothing
end
"""
    ep_evar_anchor(x::VecNum, w::VecNum, alpha::Number, rhs::Number, z::Number;
                   iters::Integer = 50, tol::Number = 1e-10, tilt_iters::Integer = 200,
                   args::Tuple = (), kwargs::NamedTuple = (;),
                   zlo_frac::Option{<:Number} = nothing)

Find the dual variable of the entropic value at risk that a posterior meeting an upper-bound view attains.

A grid point states the view as one row, and a posterior that makes the row tight reaches the target only where that point is the point the posterior itself attains. `ep_evar_anchor` solves for the dual variable that satisfies both conditions at once, which is the point the grid of [`GridEntropicValueatRiskView`](@ref) is centred on. It calls no solver.

# Arguments

  - `x`: Loss series (`-returns`).
  - `w`: Prior probabilities, summing to one.
  - `alpha`: Significance level.
  - `rhs`: Target entropic value at risk.
  - `z`: Dual variable the iteration starts from.
  - `iters::Integer = 50`: Largest number of steps the iteration takes.
  - `tol::Number = 1e-10`: Relative distance from the target at which the iteration stops.
  - `tilt_iters::Integer = 200`: Largest number of bisection steps the tilt of one row takes (see [`ep_row_tilt`](@ref)).
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `zlo_frac`: Lower end of the bracket of the dual variable, as a fraction of the upper end, forwarded to [`ep_evar`](@ref).

# Returns

  - `res::Option{@NamedTuple{z::Number, w::VecNum}}`: The dual variable and the posterior that attains it, or `nothing` when the iteration does not reach the target.

# Algorithm

 1. Build the row of the current dual variable with [`ep_evar_grid_row`](@ref). Return `nothing` when it is not finite.
 2. Tilt the prior so the row is tight with [`ep_row_tilt`](@ref). Return `nothing` when no probability vector makes it tight.
 3. Recompute the dual variable as the minimiser at the tilted probabilities with [`ep_evar`](@ref).
 4. Stop when the entropic value at risk of the tilted probabilities is within `tol` of the target, and return the dual variable and those probabilities.
 5. Return `nothing` after `iters` steps without that.

# Related

  - [`ep_row_tilt`](@ref)
  - [`ep_evar`](@ref)
  - [`ep_evar_grid`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
function ep_evar_anchor(x::VecNum, w::VecNum, alpha::Number, rhs::Number, z::Number;
                        iters::Integer = 50, tol::Number = 1e-10, tilt_iters::Integer = 200,
                        args::Tuple = (), kwargs::NamedTuple = (;),
                        zlo_frac::Option{<:Number} = nothing)
    for _ in 1:iters
        c, isc = ep_evar_grid_row(x, rhs, z)
        b = alpha * isc
        if !(all(isfinite, c) && isfinite(b))
            return nothing
        end
        q = ep_row_tilt(w, c, b; iters = tilt_iters)
        if isnothing(q)
            return nothing
        end
        res = ep_evar(x, q, alpha; args = args, kwargs = kwargs, zlo_frac = zlo_frac)
        if abs(res.evar - rhs) <= tol * abs(rhs)
            return (; z = res.z, w = q)
        end
        z = res.z
    end
    return nothing
end
"""
    ep_evar_grid(x::VecNum, w::VecNum, alpha::Number, op::Symbol, rhs::Number,
                 zstar::Number, pct::Number, K::Integer; iters::Integer = 50,
                 tol::Number = 1e-10, tilt_iters::Integer = 200, args::Tuple = (),
                 kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)

Build the grid of dual variables an entropic value-at-risk view is written on.

A view that carries an upper-bound half is centred on the dual variable [`ep_evar_anchor`](@ref) finds. A lower-bound view, and a view whose anchor does not converge, is centred on the prior's dual variable instead. The grid of [`GridRelativisticValueatRiskView`](@ref) also translates its shift, and an EVaR grid needs no counterpart of that: the shift of the primal programme of EVaR is closed form in the target and the dual variable, and the row `exp((x - rhs) / z)` of [`ep_evar_grid_row`](@ref) already carries it.

# Arguments

  - `x`: Loss series (`-returns`).
  - `w`: Prior probabilities, summing to one.
  - `alpha`: Significance level.
  - `op`: Comparison operator of the view.
  - `rhs`: Target entropic value at risk.
  - `zstar`: Dual variable that attains the prior EVaR of the asset.
  - `pct`: Half-width of the grid, as a fraction of the dual variable it is centred on.
  - `K`: Number of grid points.
  - `iters::Integer = 50`: Largest number of steps the anchor takes.
  - `tol::Number = 1e-10`: Relative distance from the target at which the anchor stops.
  - `tilt_iters::Integer = 200`: Largest number of bisection steps the tilt of one row takes (see [`ep_row_tilt`](@ref)).
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `zlo_frac`: Lower end of the bracket of the dual variable, as a fraction of the upper end, forwarded to [`ep_evar`](@ref).

# Returns

  - `z::VecNum`: Dual variable of each grid point.

# Algorithm

 1. Take the prior's dual variable as the centre.
 2. Where the view carries an upper-bound half, replace it with the dual variable of [`ep_evar_anchor`](@ref). Keep the prior's where the anchor does not converge.
 3. Span the dual variable from `zc * (1 - pct)` to `zc * (1 + pct)` in `K` points. `K` is odd, so the centre is a point of the grid, and a grid of one point is the centre alone.

# Related

  - [`ep_evar_anchor`](@ref)
  - [`ep_evar_grid_row`](@ref)
  - [`ep_add_evar_view!`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
function ep_evar_grid(x::VecNum, w::VecNum, alpha::Number, op::Symbol, rhs::Number,
                      zstar::Number, pct::Number, K::Integer; iters::Integer = 50,
                      tol::Number = 1e-10, tilt_iters::Integer = 200, args::Tuple = (),
                      kwargs::NamedTuple = (;), zlo_frac::Option{<:Number} = nothing)
    # EVaR is translation-equivariant and its dual variable is translation-invariant, so
    # the row of a grid point already carries the whole of the translation to the target.
    # The upper-bound half needs more: it reaches the target only where the grid holds the
    # dual variable the posterior itself attains, which is not the prior's. The anchor puts
    # the centre of the grid on it.
    zc = zstar
    anc = if op == :geq
        nothing
    else
        ep_evar_anchor(x, w, alpha, rhs, zstar; iters = iters, tol = tol,
                       tilt_iters = tilt_iters, args = args, kwargs = kwargs,
                       zlo_frac = zlo_frac)
    end
    if !isnothing(anc)
        zc = anc.z
    end
    # `K` is odd, so the centre is a grid point, and a grid of one point is the centre
    # alone. `range` refuses a single point between two ends that differ, so that case is
    # written out rather than left to raise from `Base`.
    return if isone(K)
        [zc]
    else
        collect(range(zc * (one(pct) - pct), zc * (one(pct) + pct); length = K))
    end
end
"""
    ep_rlvar_anchor(x::VecNum, w::VecNum, alpha::Number, kappa::Number, rhs::Number,
                    t::Number, z::Number; iters::Integer = 50, tol::Number = 1e-10,
                    tilt_iters::Integer = 200, args::Tuple = (),
                    kwargs::NamedTuple = (;),
                    bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Find the primal point of the relativistic value at risk that a posterior meeting an upper-bound view attains.

A grid point states the view as one row, and a posterior that makes the row tight reaches the target only where that point is the point the posterior itself attains. `ep_rlvar_anchor` solves for the pair that satisfies both conditions at once, which is the pair the grid of [`GridRelativisticValueatRiskView`](@ref) is centred on. It calls no solver.

# Arguments

  - `x`: Loss series (`-returns`).
  - `w`: Prior probabilities, summing to one.
  - `alpha`: Significance level.
  - `kappa`: Deformation parameter, in `(0, 1)`.
  - `rhs`: Target relativistic value at risk.
  - `t`: Shift variable the iteration starts from.
  - `z`: Dual variable the iteration starts from.
  - `iters::Integer = 50`: Largest number of steps the iteration takes.
  - `tol::Number = 1e-10`: Relative distance from the target at which the iteration stops.
  - `tilt_iters::Integer = 200`: Largest number of bisection steps the tilt of one row takes (see [`ep_row_tilt`](@ref)).
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `bracket`: Spans of the searches, forwarded to [`ep_rlvar`](@ref) and [`ep_rlvar_shift`](@ref).

# Returns

  - `res::Option{@NamedTuple{t::Number, z::Number, w::VecNum}}`: The pair and the posterior that attains it, or `nothing` when the iteration does not reach the target.

# Algorithm

 1. Build the row of the current pair with [`ep_rlvar_grid_row`](@ref). Return `nothing` when it is not finite.
 2. Tilt the prior so the row is tight with [`ep_row_tilt`](@ref). Return `nothing` when no probability vector makes it tight.
 3. Recompute the pair as the minimiser at the tilted probabilities with [`ep_rlvar`](@ref).
 4. Stop when the relativistic value at risk of the tilted probabilities is within `tol` of the target, and return the pair and those probabilities.
 5. Return `nothing` after `iters` steps without that.

# Related

  - [`ep_row_tilt`](@ref)
  - [`ep_rlvar`](@ref)
  - [`ep_rlvar_grid`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
function ep_rlvar_anchor(x::VecNum, w::VecNum, alpha::Number, kappa::Number, rhs::Number,
                         t::Number, z::Number; iters::Integer = 50, tol::Number = 1e-10,
                         tilt_iters::Integer = 200, args::Tuple = (),
                         kwargs::NamedTuple = (;),
                         bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    for _ in 1:iters
        c, b = ep_rlvar_grid_row(x, rhs, t, z, alpha, kappa)
        if !(all(isfinite, c) && isfinite(b))
            return nothing
        end
        q = ep_row_tilt(w, c, b; iters = tilt_iters)
        if isnothing(q)
            return nothing
        end
        res = ep_rlvar(x, q, alpha, kappa; args = args, kwargs = kwargs, bracket = bracket)
        if abs(res.rlvar - rhs) <= tol * abs(rhs)
            return (; t = res.t, z = res.z, w = q)
        end
        t, z = res.t, res.z
    end
    return nothing
end
"""
    ep_rlvar_grid(x::VecNum, w::VecNum, alpha::Number, kappa::Number, op::Symbol,
                  rhs::Number, zstar::Number, pv::Number, pct::Number, K::Integer;
                  iters::Integer = 50, tol::Number = 1e-10, tilt_iters::Integer = 200,
                  args::Tuple = (), kwargs::NamedTuple = (;),
                  bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Build the grid of primal points a relativistic value-at-risk view is written on.

A view that carries an upper-bound half is centred on the pair [`ep_rlvar_anchor`](@ref) finds, and every shift is the one that minimises at the posterior that pair belongs to. A lower-bound view, and a view whose anchor does not converge, is centred on the prior's dual variable instead, and every shift is the one that minimises under the prior probabilities, less the distance from the prior value to the target. The relativistic value at risk and the shift that attains it are both translation-equivariant, so a posterior that moves the value to the target behaves, to first order, like translating every loss by that distance.

# Arguments

  - `x`: Loss series (`-returns`).
  - `w`: Prior probabilities, summing to one.
  - `alpha`: Significance level.
  - `kappa`: Deformation parameter, in `(0, 1)`.
  - `op`: Comparison operator of the view.
  - `rhs`: Target relativistic value at risk.
  - `zstar`: Dual variable that attains the prior RLVaR of the asset.
  - `pv`: Prior RLVaR of the asset.
  - `pct`: Half-width of the grid, as a fraction of the dual variable it is centred on.
  - `K`: Number of grid points.
  - `iters::Integer = 50`: Largest number of steps the anchor takes.
  - `tol::Number = 1e-10`: Relative distance from the target at which the anchor stops.
  - `tilt_iters::Integer = 200`: Largest number of bisection steps the tilt of one row takes (see [`ep_row_tilt`](@ref)).
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `bracket`: Spans of the searches, forwarded to [`ep_rlvar`](@ref) and [`ep_rlvar_shift`](@ref).

# Returns

  - `t::VecNum`: Shift variable of each grid point.
  - `z::VecNum`: Dual variable of each grid point.

# Algorithm

 1. Take the prior's pair as the centre, and the distance from the prior value to the target as the translation each shift carries.
 2. Where the view carries an upper-bound half, replace both with the pair and the posterior of [`ep_rlvar_anchor`](@ref), and drop the translation. Keep the prior's pair when the anchor does not converge.
 3. Span the dual variable from `zc * (1 - pct)` to `zc * (1 + pct)` in `K` points. `K` is odd, so the centre is a point of the grid, and a grid of one point is the centre alone.
 4. Minimise the objective over the shift at each point with [`ep_rlvar_shift`](@ref), and subtract the translation.

# Related

  - [`ep_rlvar_anchor`](@ref)
  - [`ep_rlvar_shift`](@ref)
  - [`ep_add_rlvar_view!`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
function ep_rlvar_grid(x::VecNum, w::VecNum, alpha::Number, kappa::Number, op::Symbol,
                       rhs::Number, zstar::Number, pv::Number, pct::Number, K::Integer;
                       iters::Integer = 50, tol::Number = 1e-10, tilt_iters::Integer = 200,
                       args::Tuple = (), kwargs::NamedTuple = (;),
                       bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    lnk = kappa_log(inv(alpha * length(x)), kappa)
    # RLVaR is translation-equivariant, and so is the shift that attains it: subtracting
    # `delta` from every loss subtracts `delta` from both. A posterior that moves the RLVaR
    # to the target behaves, to first order, like that translation.
    zc, wc, delta = zstar, w, pv - rhs
    # Issue #530. To first order is not enough for the upper-bound half, which reaches the
    # target only where the grid holds the pair the posterior itself attains. That pair is
    # not the prior's, and over the fixture it reaches 8.4e5 times the prior's dual
    # variable, which no `pct` spans. The anchor puts the centre of the grid on it.
    anc = if op == :geq
        nothing
    else
        ep_rlvar_anchor(x, w, alpha, kappa, rhs,
                        ep_rlvar_shift(x, w, kappa, lnk, zstar; args = args,
                                       kwargs = kwargs, bracket = bracket).t - delta, zstar;
                        iters = iters, tol = tol, tilt_iters = tilt_iters, args = args,
                        kwargs = kwargs, bracket = bracket)
    end
    if !isnothing(anc)
        zc, wc, delta = anc.z, anc.w, zero(delta)
    end
    # `K` is odd, so the centre is a grid point, and a grid of one point is the centre
    # alone. `range` refuses a single point between two ends that differ, so that case is
    # written out rather than left to raise from `Base`.
    z = if isone(K)
        [zc]
    else
        collect(range(zc * (one(pct) - pct), zc * (one(pct) + pct); length = K))
    end
    t = [ep_rlvar_shift(x, wc, kappa, lnk, zk; args = args, kwargs = kwargs,
                        bracket = bracket).t - delta for zk in z]
    return t, z
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the loss series and coefficient of every asset, the significance level and the target of a linear conditional value-at-risk view.

The view parser produces one of these per view that takes the linear formulation. [`add_ep_tail_view!`](@ref) then writes the dual representation of CVaR into the model from it, one block per asset. Every coefficient is positive: a view whose coefficients carry both signs takes another formulation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearConditionalValueatRiskViewConstraint(x, coef, alpha, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct LinearConditionalValueatRiskViewConstraint <:
                 AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_losses])
    """
    x
    """
    $(field_dict[:ep_view_coef])
    """
    coef
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the ordered tail window of every asset an integer conditional value-at-risk view names.

Each entry of `ord`, `x` pairs one asset named by the view with its coefficient in `coef`, so an absolute view carries one entry and a relative view carries two. The window is sorted ascending, so the largest loss is last and the tail the binary vector marks is a suffix of it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    IntegerConditionalValueatRiskViewConstraint(ord, x, coef, alpha, op, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct IntegerConditionalValueatRiskViewConstraint <:
                 AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_ord])
    """
    ord
    """
    $(field_dict[:ep_loss])
    """
    x
    """
    $(field_dict[:ep_view_coef])
    """
    coef
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_op])
    """
    op
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the loss series and coefficient of every asset, the significance level and the target of a conic entropic value-at-risk view.

The view parser produces one of these per view that takes the conic formulation. [`add_ep_tail_view!`](@ref) then writes the relative entropy cone that is the dual representation of EVaR from it, one per asset. Every coefficient is positive: a view whose coefficients carry both signs takes another formulation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConicEntropicValueatRiskViewConstraint(x, coef, alpha, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct ConicEntropicValueatRiskViewConstraint <: AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_losses])
    """
    x
    """
    $(field_dict[:ep_view_coef])
    """
    coef
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the grid of dual variables that an upper-bound or equality entropic value-at-risk view selects one point of.

A lower-bound grid view is a set of rows on the posterior probabilities alone, so it goes into the constraint dictionary and never reaches this carrier. An equality view emits both: the rows go into the dictionary and the selector block comes here.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GridEntropicValueatRiskViewConstraint(x, z, alpha, rhs, M)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct GridEntropicValueatRiskViewConstraint <: AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_loss])
    """
    x
    """
    $(field_dict[:ep_zgrid])
    """
    z
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:bigM])
    """
    M
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the loss series and coefficient of every asset, the significance level, the deformation parameter and the target of a conic relativistic value-at-risk view.

The view parser produces one of these per view that takes the conic formulation. [`add_ep_tail_view!`](@ref) then writes the power cones that are the dual representation of RLVaR from it, one pair per observation per asset. Every coefficient is positive: a view whose coefficients carry both signs takes another formulation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConicRelativisticValueatRiskViewConstraint(x, coef, alpha, kappa, rhs)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
@concrete struct ConicRelativisticValueatRiskViewConstraint <:
                 AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_losses])
    """
    x
    """
    $(field_dict[:ep_view_coef])
    """
    coef
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_kappa])
    """
    kappa
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the grid of primal points that an upper-bound or equality relativistic value-at-risk view selects one point of.

A lower-bound grid view is a set of rows on the posterior probabilities alone, so it goes into the constraint dictionary and never reaches this carrier. An equality view emits both: the rows go into the dictionary and the selector block comes here.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GridRelativisticValueatRiskViewConstraint(x, t, z, alpha, kappa, rhs, M)

Arguments correspond to the fields above.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
@concrete struct GridRelativisticValueatRiskViewConstraint <: AbstractEntropyPoolingTailView
    """
    $(field_dict[:ep_loss])
    """
    x
    """
    $(field_dict[:ep_rlvar_tgrid])
    """
    t
    """
    $(field_dict[:ep_rlvar_zgrid])
    """
    z
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_kappa])
    """
    kappa
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:rlvar_bigM])
    """
    M
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the carriers of the sequential convex tail view formulations.

A sequential carrier splits the assets of an oriented lower-bound view into a **dual side**, whose measures are concave in the posterior probabilities and take their exact dual blocks, and a **primal side**, whose measures enter with a negative coefficient and take a linear upper bound read at a fixed posterior. The three subtypes differ in the measure alone, and the two verbs that read a carrier, [`add_ep_tail_view!`](@ref) and [`ep_refine_tail_view`](@ref), have one method for the whole family. Each subtype supplies [`ep_tail_dual_block!`](@ref) and [`ep_tail_surrogate_row`](@ref) for its measure.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`SequentialConditionalValueatRiskViewConstraint`](@ref)
  - [`SequentialEntropicValueatRiskViewConstraint`](@ref)
  - [`SequentialRelativisticValueatRiskViewConstraint`](@ref)
  - [`ep_refine_tail_view`](@ref)
"""
abstract type AbstractSequentialTailViewConstraint <: AbstractEntropyPoolingTailView end
"""
$(DocStringExtensions.TYPEDEF)

Carries the two sides of a sequential conditional value-at-risk view, its surrogate row and its stopping rule.

The view parser produces one of these per view that takes [`SequentialConditionalValueatRiskView`](@ref), with the surrogate row read at the prior. [`ep_refine_tail_view`](@ref) then re-reads the row at each posterior [`entropy_pooling`](@ref) produces.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SequentialConditionalValueatRiskViewConstraint(xd, cd, xp, cp, c, b, alpha, rhs, iters, tol)

Arguments correspond to the fields above.

# Related

  - [`AbstractSequentialTailViewConstraint`](@ref)
  - [`SequentialConditionalValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`ep_refine_tail_view`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct SequentialConditionalValueatRiskViewConstraint <:
                 AbstractSequentialTailViewConstraint
    """
    $(field_dict[:ep_seq_xd])
    """
    xd
    """
    $(field_dict[:ep_seq_cd])
    """
    cd
    """
    $(field_dict[:ep_seq_xp])
    """
    xp
    """
    $(field_dict[:ep_seq_cp])
    """
    cp
    """
    $(field_dict[:ep_seq_row])
    """
    c
    """
    $(field_dict[:ep_seq_b])
    """
    b
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:ep_seq_iters])
    """
    iters
    """
    $(field_dict[:ep_seq_tol])
    """
    tol
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the two sides of a sequential entropic value-at-risk view, its surrogate row, its stopping rule, and the settings of the search for the dual variable.

The view parser produces one of these per view that takes [`SequentialEntropicValueatRiskView`](@ref), with the surrogate row read at the prior. [`ep_refine_tail_view`](@ref) then re-reads the row at each posterior [`entropy_pooling`](@ref) produces, which runs [`ep_evar`](@ref) once per asset of the primal side under `args`, `kwargs` and `zlo_frac`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SequentialEntropicValueatRiskViewConstraint(xd, cd, xp, cp, c, b, alpha, rhs, iters, tol,
                                                args, kwargs, zlo_frac)

Arguments correspond to the fields above.

# Related

  - [`AbstractSequentialTailViewConstraint`](@ref)
  - [`SequentialEntropicValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`ep_refine_tail_view`](@ref)
  - [`ep_evar`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
@concrete struct SequentialEntropicValueatRiskViewConstraint <:
                 AbstractSequentialTailViewConstraint
    """
    $(field_dict[:ep_seq_xd])
    """
    xd
    """
    $(field_dict[:ep_seq_cd])
    """
    cd
    """
    $(field_dict[:ep_seq_xp])
    """
    xp
    """
    $(field_dict[:ep_seq_cp])
    """
    cp
    """
    $(field_dict[:ep_seq_row])
    """
    c
    """
    $(field_dict[:ep_seq_b])
    """
    b
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:ep_seq_iters])
    """
    iters
    """
    $(field_dict[:ep_seq_tol])
    """
    tol
    """
    $(field_dict[:optargs]) They reach the search of [`ep_evar`](@ref) each re-read runs.
    """
    args
    """
    $(field_dict[:optkwargs]) They reach the same search `args` does.
    """
    kwargs
    """
    $(field_dict[:ep_tv_evar_zlo_frac])
    """
    zlo_frac
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the two sides of a sequential relativistic value-at-risk view, its surrogate row, its stopping rule, and the settings of the search for the primal pair.

The view parser produces one of these per view that takes [`SequentialRelativisticValueatRiskView`](@ref), with the surrogate row read at the prior. [`ep_refine_tail_view`](@ref) then re-reads the row at each posterior [`entropy_pooling`](@ref) produces, which runs [`ep_rlvar`](@ref) once per asset of the primal side under `args`, `kwargs` and `bracket`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SequentialRelativisticValueatRiskViewConstraint(xd, cd, xp, cp, c, b, alpha, kappa, rhs,
                                                    iters, tol, args, kwargs, bracket)

Arguments correspond to the fields above.

# Related

  - [`AbstractSequentialTailViewConstraint`](@ref)
  - [`SequentialRelativisticValueatRiskView`](@ref)
  - [`add_ep_tail_view!`](@ref)
  - [`ep_refine_tail_view`](@ref)
  - [`ep_rlvar`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
@concrete struct SequentialRelativisticValueatRiskViewConstraint <:
                 AbstractSequentialTailViewConstraint
    """
    $(field_dict[:ep_seq_xd])
    """
    xd
    """
    $(field_dict[:ep_seq_cd])
    """
    cd
    """
    $(field_dict[:ep_seq_xp])
    """
    xp
    """
    $(field_dict[:ep_seq_cp])
    """
    cp
    """
    $(field_dict[:ep_seq_row])
    """
    c
    """
    $(field_dict[:ep_seq_b])
    """
    b
    """
    $(field_dict[:ep_view_alpha])
    """
    alpha
    """
    $(field_dict[:ep_view_kappa])
    """
    kappa
    """
    $(field_dict[:ep_view_rhs])
    """
    rhs
    """
    $(field_dict[:ep_seq_iters])
    """
    iters
    """
    $(field_dict[:ep_seq_tol])
    """
    tol
    """
    $(field_dict[:optargs]) They reach the searches of [`ep_rlvar`](@ref) each re-read runs.
    """
    args
    """
    $(field_dict[:optkwargs]) They reach the same searches `args` does.
    """
    kwargs
    """
    $(field_dict[:ep_tv_bracket])
    """
    bracket
end
"""
    ep_tail_dual_block!(model::JuMP.Model, pw, tv::AbstractEntropyPoolingTailView, x::VecNum,
                        sc1::Number)

Add the dual block of one asset's risk measure to an entropy pooling JuMP model, and return the expression that attains the measure.

The dual representation of each measure is a maximum of ``\\boldsymbol{\\nu}^{\\intercal} \\boldsymbol{x}`` over a set of weights that depends on the posterior probabilities. `ep_tail_dual_block!` registers that set, and hands the linear expression back so the caller can bound it, on its own or in a coefficient-weighted sum over several assets. The carrier `tv` names the measure and carries its level, and the method is shared by the fixed carrier of the measure and its sequential one.

# JuMP formulation

## Variables

  - `pw`: $(math_dict[:ep_post_probs]) It is read from the caller, and every entry below is registered against it.
  - `nu`: $(math_dict[:ep_tail_nu]) The conditional value-at-risk method bounds it below by zero, and the two other methods bound it to ``[0, 1]``.
  - `tau`, `varsigma`: ``\\boldsymbol{\\tau}`` and ``\\boldsymbol{\\varsigma}``, ``T \\times 1`` each and bounded below by zero, created by the relativistic value-at-risk method. The cones and the budget already imply both bounds, and stating them is what turns a `SLOW_PROGRESS` report into an `OPTIMAL` one.

## Expressions

  - The method returns ``\\sum_{j=1}^{T} \\nu_{j} x_{j}``, registered under no name.

## Constraints

Every row is registered under no name. The conditional value-at-risk method, for [`LinearConditionalValueatRiskViewConstraint`](@ref) and [`SequentialConditionalValueatRiskViewConstraint`](@ref), registers two:

  - ``s_{c1} \\left(\\nu_{j} - \\dfrac{p_{j}}{\\alpha}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,T``.
  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\nu_{j} - 1\\right) = 0``.

The entropic value-at-risk method, for [`ConicEntropicValueatRiskViewConstraint`](@ref) and [`SequentialEntropicValueatRiskViewConstraint`](@ref), registers two:

  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\nu_{j} - 1\\right) = 0``.
  - ``\\left(s_{c1} \\ln\\left(\\dfrac{1}{\\alpha}\\right),\\, s_{c1} \\boldsymbol{p},\\, s_{c1} \\boldsymbol{\\nu}\\right) \\in \\mathcal{K}_{\\mathrm{re}}(2T+1)``.

The relativistic value-at-risk method, for [`ConicRelativisticValueatRiskViewConstraint`](@ref) and [`SequentialRelativisticValueatRiskViewConstraint`](@ref), registers four:

  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\nu_{j} - 1\\right) = 0``.
  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\dfrac{\\tau_{j} - \\varsigma_{j}}{2\\kappa} - \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\right) \\leq 0``.
  - ``\\left(s_{c1} \\tau_{j},\\, s_{c1} T p_{j},\\, s_{c1} \\nu_{j}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\left(\\dfrac{1}{1+\\kappa}\\right)``, ``\\forall\\, j = 1,\\ldots,T``.
  - ``\\left(s_{c1} \\nu_{j},\\, s_{c1} T p_{j},\\, s_{c1} \\varsigma_{j}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa)``, ``\\forall\\, j = 1,\\ldots,T``.

Where:

  - $(math_dict[:ep_sc1])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:ep_tail_nu])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - ``\\boldsymbol{\\tau}``, ``\\boldsymbol{\\varsigma}``: ``T \\times 1`` vectors that carry the Kaniadakis entropy budget of ``\\boldsymbol{\\nu}``.
  - ``\\mathcal{K}_{\\mathrm{re}}(1 + 2T) = \\{(u,\\, \\boldsymbol{v},\\, \\boldsymbol{s}) : u \\geq \\sum_{j=1}^{T} s_{j} \\ln(s_{j} / v_{j})\\}``: Relative entropy cone.
  - ``\\mathcal{K}_{\\mathrm{pow}}(\\pi) = \\{(a, b, c) : a^{\\pi} b^{1-\\pi} \\geq |c|,\\, a \\geq 0,\\, b \\geq 0\\}``: Power cone.

## Relaxation

$(val_dict[:relax])

The block is exact: the set it registers is the dual description of the measure, so the largest value the returned expression takes is the measure of the asset under `pw`.

# Arguments

  - `model`: Entropy pooling JuMP model.
  - `pw`: Vector of posterior probability variables.
  - `tv`: Tail view constraint. It names the measure and carries its level, and its deformation parameter for the relativistic method.
  - `x`: Loss series of the asset.
  - `sc1`: Constraint scaling factor.

# Returns

  - `expr::JuMP.AffExpr`: The expression ``\\sum_{j=1}^{T} \\nu_{j} x_{j}``.

# Related

  - [`add_ep_tail_view!`](@ref)
  - [`LinearConditionalValueatRiskViewConstraint`](@ref)
  - [`ConicEntropicValueatRiskViewConstraint`](@ref)
  - [`ConicRelativisticValueatRiskViewConstraint`](@ref)
  - [`AbstractSequentialTailViewConstraint`](@ref)

# References

  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
"""
function ep_tail_dual_block!(model::JuMP.Model, pw,
                             tv::Union{<:LinearConditionalValueatRiskViewConstraint,
                                       <:SequentialConditionalValueatRiskViewConstraint},
                             x::VecNum, sc1::Number)
    alpha = tv.alpha
    T = length(x)
    nu = JuMP.@variable(model, [1:T], lower_bound = 0)
    JuMP.@constraints(model, begin
                          [j = 1:T], sc1 * (nu[j] - pw[j] / alpha) <= 0
                          sc1 * (sum(nu) - one(alpha)) == 0
                      end)
    return LinearAlgebra.dot(nu, x)
end
function ep_tail_dual_block!(model::JuMP.Model, pw,
                             tv::Union{<:ConicEntropicValueatRiskViewConstraint,
                                       <:SequentialEntropicValueatRiskViewConstraint},
                             x::VecNum, sc1::Number)
    alpha = tv.alpha
    T = length(x)
    nu = JuMP.@variable(model, [1:T], lower_bound = 0, upper_bound = 1)
    JuMP.@constraints(model,
                      begin
                          sc1 * (sum(nu) - one(alpha)) == 0
                          [sc1 * log(inv(alpha)); sc1 * pw; sc1 * nu] in
                          JuMP.MOI.RelativeEntropyCone(2 * T + 1)
                      end)
    return LinearAlgebra.dot(nu, x)
end
function ep_tail_dual_block!(model::JuMP.Model, pw,
                             tv::Union{<:ConicRelativisticValueatRiskViewConstraint,
                                       <:SequentialRelativisticValueatRiskViewConstraint},
                             x::VecNum, sc1::Number)
    (; alpha, kappa) = tv
    T = length(x)
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik2 = inv(2 * kappa)
    lnk = kappa_log(inv(alpha * T), kappa)
    nu = JuMP.@variable(model, [1:T], lower_bound = 0, upper_bound = 1)
    # Both bounds are implied by the cones and the budget: the first slot of a power cone
    # is non-negative, and the budget is loosest at the largest `varsigma` the second cone
    # allows, which is non-negative. Stating them is what turns a `SLOW_PROGRESS` report
    # into an `OPTIMAL` one.
    tau = JuMP.@variable(model, [1:T], lower_bound = 0)
    varsigma = JuMP.@variable(model, [1:T], lower_bound = 0)
    JuMP.@constraints(model,
                      begin
                          sc1 * (sum(nu) - one(alpha)) == 0
                          sc1 * (sum(tau - varsigma) * ik2 - lnk) <= 0
                          [j = 1:T],
                          [sc1 * tau[j], sc1 * T * pw[j], sc1 * nu[j]] in
                          JuMP.MOI.PowerCone(inv(opk))
                          [j = 1:T],
                          [sc1 * nu[j], sc1 * T * pw[j], sc1 * varsigma[j]] in
                          JuMP.MOI.PowerCone(omk)
                      end)
    return LinearAlgebra.dot(nu, x)
end
"""
    ep_var_multiplier(x::VecNum, w::VecNum, alpha::Number)

Find the value at risk of a loss series under observation probabilities, as the minimiser of the primal of the conditional value at risk.

# Mathematical definition

The conditional value at risk is the minimum over ``\\eta`` of ``\\eta + \\dfrac{1}{\\alpha} \\sum_{j=1}^{T} w_{j} (x_{j} - \\eta)^{+}``, a convex piecewise-linear function whose kinks sit at the losses. Its minimiser is the loss at which the tail mass first reaches ``\\alpha``:

```math
\\eta^{\\star} = x_{(s)}\\,, \\quad s = \\min\\left\\{k : \\sum_{i=1}^{k} w_{(i)} \\geq \\alpha\\right\\}\\,,
```

with the losses sorted in descending order, ``x_{(1)} \\geq x_{(2)} \\geq \\ldots``.

Where:

  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

# Arguments

  - `x`: Loss series (`-returns`).
  - `w`: Observation probabilities. Normalised to sum to one.
  - `alpha`: Significance level.

# Returns

  - `eta::Number`: The value at risk, one of the losses of `x`.

# Related

  - [`ep_tail_surrogate_row`](@ref)
  - [`SequentialConditionalValueatRiskView`](@ref)
  - [`ValueatRisk`](@ref)
"""
function ep_var_multiplier(x::VecNum, w::VecNum, alpha::Number)
    o = sortperm(x; rev = true)
    sw = sum(w)
    cw = zero(promote_type(eltype(w), typeof(alpha)))
    for j in o
        cw += w[j] / sw
        if cw >= alpha
            return x[j]
        end
    end
    return x[o[end]]
end
"""
    ep_tail_surrogate_row(tv::AbstractSequentialTailViewConstraint, x::VecNum, w::VecNum)

Read the linear upper bound of one asset's risk measure at a posterior.

Each sequential formulation bounds the measure of an asset on its primal side by an affine function of the posterior probabilities, ``r_{0} + \\boldsymbol{r}^{\\intercal} \\boldsymbol{p}``, that is tight at the probabilities `w` it is read at. `ep_tail_surrogate_row` returns that function, so the value ``r_{0} + \\boldsymbol{r}^{\\intercal} \\boldsymbol{w}`` is the measure of the asset under `w`.

# Mathematical definition

The conditional value-at-risk method reads the primal at the value at risk ``\\eta`` of [`ep_var_multiplier`](@ref):

```math
r_{0} = \\eta\\,, \\quad r_{j} = \\dfrac{(x_{j} - \\eta)^{+}}{\\alpha}\\,.
```

The entropic value-at-risk method reads the dual variable ``z`` and the value of [`ep_evar`](@ref), and takes the tangent of the concave primal at `w`:

```math
r_{0} = \\mathrm{EVaR}_{\\alpha}(X) - z\\,, \\quad r_{j} = \\dfrac{z e^{x_{j}/z}}{\\sum_{k=1}^{T} w_{k} e^{x_{k}/z}}\\,.
```

The relativistic value-at-risk method reads the pair ``(t, z)`` of [`ep_rlvar`](@ref), at which the primal is linear in the probabilities:

```math
r_{0} = t + z \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\,, \\quad r_{j} = T \\varphi_{\\kappa}(t - x_{j},\\, z)\\,.
```

Where:

  - $(math_dict[:rlvar_loss])
  - $(math_dict[:rlvar_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:rlvar_phi])
  - $(math_dict[:evar_stat])
  - ``\\eta``: Value at risk of the loss series under ``\\boldsymbol{w}``.
  - ``z``: Dual variable that attains the measure under ``\\boldsymbol{w}``.
  - ``t``: Shift that attains the relativistic value at risk under ``\\boldsymbol{w}``.

# Arguments

  - `tv`: Sequential tail view constraint. It names the measure, and carries its level and the settings of its search.
  - `x`: Loss series of the asset (`-returns`).
  - `w`: Observation probabilities the bound is read at, summing to one.

# Returns

  - `r::VecNum`: Coefficients of the bound, one per observation.
  - `r0::Number`: Constant of the bound.

# Related

  - [`ep_refine_tail_view`](@ref)
  - [`ep_var_multiplier`](@ref)
  - [`ep_evar`](@ref)
  - [`ep_rlvar`](@ref)
  - [`AbstractSequentialTailViewConstraint`](@ref)

# References

  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
"""
function ep_tail_surrogate_row(tv::SequentialConditionalValueatRiskViewConstraint,
                               x::VecNum, w::VecNum)
    alpha = tv.alpha
    eta = ep_var_multiplier(x, w, alpha)
    return max.(x .- eta, zero(eta)) ./ alpha, eta
end
function ep_tail_surrogate_row(tv::SequentialEntropicValueatRiskViewConstraint, x::VecNum,
                               w::VecNum)
    (; alpha, args, kwargs, zlo_frac) = tv
    res = ep_evar(x, w, alpha; args = args, kwargs = kwargs, zlo_frac = zlo_frac)
    z = res.z
    # The exponentials are shifted by the largest loss, so the ratio does not overflow at
    # a small dual variable.
    r = exp.((x .- maximum(x)) ./ z)
    r .*= z / LinearAlgebra.dot(w, r)
    return r, res.evar - z
end
function ep_tail_surrogate_row(tv::SequentialRelativisticValueatRiskViewConstraint,
                               x::VecNum, w::VecNum)
    (; alpha, kappa, args, kwargs, bracket) = tv
    res = ep_rlvar(x, w, alpha, kappa; args = args, kwargs = kwargs, bracket = bracket)
    T = length(x)
    lnk = kappa_log(inv(alpha * T), kappa)
    return T .* ep_rlvar_tail.(res.t .- x, res.z, kappa), res.t + res.z * lnk
end
"""
    ep_refine_tail_view(tv::AbstractSequentialTailViewConstraint, w::VecNum)

Re-read the surrogate row of a sequential tail view at a posterior, and say whether the row it held was already tight there.

# Algorithm

 1. Return `tv` and `true` where the primal side is empty. The view is then convex, its dual blocks are exact, and there is nothing to re-read.
 2. Clamp `w` below at zero and normalise it to sum to one. A conic solver returns a posterior whose smallest entries sit a rounding error below zero, and the searches behind the rows refuse a negative probability.
 3. For each asset of the primal side, read its bound at `w` with [`ep_tail_surrogate_row`](@ref), and accumulate the coefficient-weighted sum of the bounds into a new row `c`, `b`. The value of the new row at `w` is the coefficient-weighted sum of the measures there.
 4. Read the gap between the row `tv` held and the new one at `w`. The old row bounds the same sum from the same side, so the gap is the slack the last solve left.
 5. Return the carrier with the new row, and whether the gap is within `tol` of the larger of the target and the largest loss the primal side names.

# Arguments

  - `tv`: Sequential tail view constraint.
  - `w`: Posterior probabilities of the last solve.

# Returns

  - `tv::AbstractSequentialTailViewConstraint`: The carrier with its row re-read at `w`.
  - `tight::Bool`: Whether the row `tv` held before the call was tight at `w`.

# Related

  - [`entropy_pooling`](@ref)
  - [`ep_tail_surrogate_row`](@ref)
  - [`AbstractSequentialTailViewConstraint`](@ref)
"""
function ep_refine_iters(tv::AbstractSequentialTailViewConstraint)
    return tv.iters
end
function ep_refine_tail_view(tv::AbstractSequentialTailViewConstraint, w::VecNum)
    (; xp, cp, c, b, rhs, tol) = tv
    if isempty(xp)
        return tv, true
    end
    # A conic solver returns a posterior with entries a rounding error below zero, which
    # the searches behind the rows refuse, so the posterior is clamped before it is read.
    wi = max.(w, zero(eltype(w)))
    wi ./= sum(wi)
    cn = zero(c)
    bn = zero(b)
    scale = abs(rhs)
    for (x, ci) in zip(xp, cp)
        r, r0 = ep_tail_surrogate_row(tv, x, wi)
        cn .+= ci .* r
        bn += ci * r0
        scale = max(scale, maximum(abs, x))
    end
    gap = abs((LinearAlgebra.dot(cn, wi) + bn) - (LinearAlgebra.dot(c, wi) + b))
    return Accessors.setproperties(tv, (; c = cn, b = bn)), gap <= tol * scale
end
"""
    ep_sequential_start(tv::AbstractSequentialTailViewConstraint, w::VecNum)

Read the first surrogate row of a sequential tail view at probabilities under which the row can meet the view.

A linear upper bound is tight where it is read, but it can fall only as far as its smallest value over the simplex, and a target further below the prior than that leaves the first solve with no feasible point. The relativistic measure is the one this bites: its bound at the prior's pair has a floor at the shift plus the deformed logarithm, and only the tail term above it can move. `ep_sequential_start` walks the multipliers toward the target with a chain of exponential tilts before any solver runs, which is what the anchors of [`GridEntropicValueatRiskView`](@ref) and [`GridRelativisticValueatRiskView`](@ref) do to centre their grids. It calls no solver.

# Algorithm

 1. Normalise `w` to sum to one, and read the row at it with [`ep_refine_tail_view`](@ref). Return the carrier where its primal side is empty, because the view is then convex and the row is empty.
 2. Read `need`, the value the row must reach: the target, less the row's constant, less the coefficient-weighted sum of the measures of the dual side under the current probabilities, each read through [`ep_tail_surrogate_row`](@ref).
 3. Return the carrier where the largest coefficient of the row is at least `need`. A probability vector then meets the row, and the solve can start.
 4. Otherwise tilt the probabilities with [`ep_row_tilt`](@ref) so the row reaches nine tenths of the way from its current value to its largest coefficient, re-read the row there, and return to step 2. Take at most `iters` steps, and return the last carrier where the tilt does not exist.

# Arguments

  - `tv`: Sequential tail view constraint, with any row.
  - `w`: Prior probability weights.

# Returns

  - `tv::AbstractSequentialTailViewConstraint`: The carrier with its first row.

# Related

  - [`ep_refine_tail_view`](@ref)
  - [`ep_tail_surrogate_row`](@ref)
  - [`ep_row_tilt`](@ref)
  - [`ep_evar_anchor`](@ref)
  - [`ep_rlvar_anchor`](@ref)
  - [`AbstractSequentialTailViewConstraint`](@ref)
"""
function ep_sequential_start(tv::AbstractSequentialTailViewConstraint, w::VecNum)
    (; xd, cd, xp, iters) = tv
    wi = w ./ sum(w)
    tv, _ = ep_refine_tail_view(tv, wi)
    if isempty(xp)
        return tv
    end
    for _ in 1:iters
        (; c, b, rhs) = tv
        need = rhs - b
        for (x, ci) in zip(xd, cd)
            r, r0 = ep_tail_surrogate_row(tv, x, wi)
            need -= ci * (r0 + LinearAlgebra.dot(r, wi))
        end
        cmax = maximum(c)
        if need <= cmax
            return tv
        end
        v = LinearAlgebra.dot(c, wi)
        q = ep_row_tilt(wi, c, v + 0.9 * (cmax - v))
        if isnothing(q)
            return tv
        end
        wi = q
        tv, _ = ep_refine_tail_view(tv, wi)
    end
    return tv
end
"""
    add_ep_tail_view!(model::JuMP.Model, pw, tv::AbstractEntropyPoolingTailView,
                      sc1::Number)

Add the variables and constraints of one tail view to an entropy pooling JuMP model.

`add_ep_tail_view!` is the one seam through which a conditional, entropic or relativistic value-at-risk view reaches the model. Each formulation has its own method, dispatched on the constraint carrier the view parser produced. The three dual carriers share one method, and the three sequential carriers share another: both write one dual block per asset with [`ep_tail_dual_block!`](@ref), and differ in the row that bounds the sum.

# JuMP formulation

The section covers the five methods, and every entry each of them registers. Each entry is anonymous: one model carries one block per view, so a name would collide on the second view of a family, and nothing reads these entries back by name.

## Variables

  - `pw`: $(math_dict[:ep_post_probs]) It is read from the caller, and every entry below is registered against it.
  - `nu`, `tau`, `varsigma`: created once per asset of the view by [`ep_tail_dual_block!`](@ref), for the dual carriers and for the dual side of the sequential ones. Its `# JuMP formulation` names them.
  - `y`, `q`: ``\\boldsymbol{y}`` and ``\\boldsymbol{q}``, ``\\bar{s} \\times 1`` each, created once per asset by the [`IntegerConditionalValueatRiskViewConstraint`](@ref) method. `y` is binary, and `q` is bounded below by zero.
  - `y`: ``\\boldsymbol{y}``, ``K \\times 1`` and binary, created by the [`GridEntropicValueatRiskViewConstraint`](@ref) and [`GridRelativisticValueatRiskViewConstraint`](@ref) methods. It selects the grid point the view is met at.

## Expressions

  - ``\\varepsilon``: Left hand side of the view, built once per method and registered under no name. It is the coefficient-weighted sum of the per-asset expressions [`ep_tail_dual_block!`](@ref) returns for the dual carriers, of the per-asset tail sums for the integer carrier, and of the dual-side expressions plus the surrogate row ``b + \\boldsymbol{c}^{\\intercal} \\boldsymbol{p}`` for the sequential carriers.

## Constraints

The method of the three dual carriers, [`LinearConditionalValueatRiskViewConstraint`](@ref), [`ConicEntropicValueatRiskViewConstraint`](@ref) and [`ConicRelativisticValueatRiskViewConstraint`](@ref), registers one block of [`ep_tail_dual_block!`](@ref) per asset the view names, and one row:

  - ``s_{c1} \\left(\\bar{c} - \\sum_{i} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j}\\right) \\leq 0``.

The method of the three sequential carriers, [`SequentialConditionalValueatRiskViewConstraint`](@ref), [`SequentialEntropicValueatRiskViewConstraint`](@ref) and [`SequentialRelativisticValueatRiskViewConstraint`](@ref), registers one block of [`ep_tail_dual_block!`](@ref) per asset of the dual side, and one row over both sides, divided by the largest coefficient of the surrogate row where that exceeds one, so the row's coefficients sit in ``[-1, 1]`` however small the dual variable of a relativistic measure is:

  - ``s_{c1} \\left(\\bar{c} - \\sum_{i \\in \\mathcal{P}} \\gamma_{i} \\sum_{j=1}^{T} \\nu_{i,\\,j} x_{i,\\,j} - b - \\sum_{j=1}^{T} c_{j} p_{j}\\right) \\Big/ \\max\\left(1, \\lVert \\boldsymbol{c} \\rVert_{\\infty}\\right) \\leq 0``.

The [`IntegerConditionalValueatRiskViewConstraint`](@ref) method registers five rows per asset the view names, over that asset's window of the ``\\bar{s}`` largest losses:

  - ``s_{c1} \\left(q_{j} - y_{j}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,\\bar{s}``.
  - ``s_{c1} \\left(q_{j} - p_{[j]}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,\\bar{s}``.
  - ``s_{c1} \\left(p_{[j]} - (1 - y_{j-1}) - q_{j}\\right) \\leq 0``, ``\\forall\\, j = 2,\\ldots,\\bar{s}``.
  - ``s_{c1} \\left(y_{j} - y_{j+1}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,\\bar{s}-1``.
  - ``s_{c1} \\left(\\sum_{j=1}^{\\bar{s}} q_{j} - \\alpha\\right) = 0``.

and one further row on ``\\varepsilon = \\sum_{i} \\dfrac{\\gamma_{i}}{\\alpha} \\sum_{j=1}^{\\bar{s}} q_{i,\\,j} x_{i,\\,[j]}``, the view's operator picking which of the three:

  - ``s_{c1} \\left(\\varepsilon - \\bar{c}\\right) = 0`` under `:eq`.
  - ``s_{c1} \\left(\\bar{c} - \\varepsilon\\right) \\leq 0`` under `:geq`.
  - ``s_{c1} \\left(\\varepsilon - \\bar{c}\\right) \\leq 0`` under `:leq`.

The [`GridEntropicValueatRiskViewConstraint`](@ref) and [`GridRelativisticValueatRiskViewConstraint`](@ref) methods register two rows each, over the row of each grid point divided by its bound:

  - ``s_{c1} \\left(\\sum_{k=1}^{K} y_{k} - 1\\right) = 0``.
  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\dfrac{c_{k,\\,j}}{b_{k}} p_{j} - 1 - M M_{k} (1 - y_{k})\\right) \\leq 0``, ``\\forall\\, k = 1,\\ldots,K``, with ``M_{k} = \\max_{j} c_{k,\\,j} / b_{k} - 1``.

Where:

  - $(math_dict[:ep_sc1])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:cvar_target])
  - ``x_{i,\\,j}``: Loss of asset ``i`` at observation ``j``, the negated return.
  - ``\\boldsymbol{\\nu}_{i}``: ``T \\times 1`` vector of weights that attains the measure of asset ``i``, from its block of [`ep_tail_dual_block!`](@ref).
  - ``\\bar{c}``: Target of the view, whichever measure it is stated on. A sequential carrier holds the view oriented as a lower bound, so its target carries the sign of that orientation.
  - ``\\bar{s}``: Length of one asset's window of largest losses, from [`ep_sbar`](@ref).
  - ``x_{[j]}``, ``p_{[j]}``: Loss and posterior probability of the observation in position ``j`` of that window, which is sorted ascending.
  - ``\\gamma_{i}``: Coefficient the view gives asset ``i``.
  - ``\\mathcal{P}``: Assets on the dual side of a sequential carrier.
  - ``b``, ``\\boldsymbol{c}``: Surrogate row a sequential carrier holds, from [`ep_tail_surrogate_row`](@ref).
  - ``\\boldsymbol{y}``: Binary vector. It marks the tail of one asset's window in the integer conditional value-at-risk method, and selects one grid point in the two grid methods.
  - ``\\boldsymbol{q}``: ``\\bar{s} \\times 1`` vector that carries the tail mass of each observation of the window: ``p_{[j]}`` above the lowest marked observation, a part of ``p_{[j]}`` at it, and zero below it.
  - ``\\varepsilon``: Left hand side of an integer conditional value-at-risk view, the coefficient-weighted sum of the per-asset posterior CVaRs.
  - ``K``: Number of grid points the carrier holds.
  - ``c_{k,\\,j}``: Scaled coefficient of observation ``j`` at grid point ``k``, from [`ep_evar_grid_row`](@ref) or [`ep_rlvar_grid_row`](@ref).
  - ``b_{k}``: Scaled bound of grid point ``k``, from those same two functions. It is ``\\alpha`` times the reciprocal that [`ep_evar_grid_row`](@ref) returns, and the target that [`ep_rlvar_grid_row`](@ref) returns.
  - ``M_{k}``: Smallest big-M constant that releases the row of grid point ``k``.
  - ``M``: Big-M multiplier the grid carrier holds.

## Relaxation

$(val_dict[:relax])

The two grid methods and the sequential method bound the statistic. The dual method is exact. The integer method is exact over the posteriors whose tail of mass ``\\alpha`` lies inside the window of ``\\bar{s}`` largest losses, which is every posterior when ``\\bar{s} = T``.

The sequential method is a **restriction** of the view, tightened by re-solves.

 1. **Direction.** The surrogate row bounds each measure of the primal side from above, and each of them carries a negative coefficient, so the row's value sits at or below the view's left hand side. A posterior that meets the row meets the view.

 2. **Quantity.** The coefficient-weighted sum of the posterior measures the view names, a statistic of `pw`.

 3. **Tightness.** The row is tight at the posterior it was read at. [`entropy_pooling`](@ref) re-reads it at each posterior with [`ep_refine_tail_view`](@ref) and solves again, until the slack is within the carrier's `tol` or its `iters` re-solves are spent, so the view is met to that tolerance at the fixed point and over-met before it.

 4. **Direction.** Every grid point is a feasible point of the primal programme of the statistic, so its row bounds the statistic from above. The block asks one grid point to hold, so the posterior statistic lies at or below the target. The encoding is a **restriction**: it can only be tighter than the view asks, and the view is never violated.

 5. **Quantity.** The posterior entropic value at risk under the [`GridEntropicValueatRiskViewConstraint`](@ref) method, and the posterior relativistic value at risk under the [`GridRelativisticValueatRiskViewConstraint`](@ref) method. Both are statistics of `pw`.

 6. **Tightness.** The bound is tight where the grid holds the point the posterior itself attains. [`ep_evar_anchor`](@ref) and [`ep_rlvar_anchor`](@ref) put the centre of the grid on that point. Where the anchor does not converge the grid falls back to the prior's point, and the posterior statistic can land strictly below the target. Widen `pct` or raise `K` there.

A row reads against one because its bound ``b_{k}`` falls far below a solver's feasibility tolerance where the dual variable is small. Read at that scale, the whole row is inside the tolerance, the solver takes it as met by any posterior, and the selector can pick that grid point with the view unmet.

``M M_{k}`` releases the rows of the grid points the selector does not pick. ``\\boldsymbol{p}`` sums to one, so the left hand side of row ``k`` never exceeds ``\\max_{j} c_{k,\\,j} / b_{k}``, and ``M_{k}`` is the smallest constant that clears it. An ``M`` below one would cut off a posterior the view admits, so ``M \\geq 1``. An ``M`` above one gives the released rows headroom for a posterior that sums to one only to the solver's tolerance, but it also widens the slack that the tolerance on integrality opens on the selected row, so the default is ``M = 1``.

The other half of a grid view is a **relaxation**, and it does not reach this function. A lower-bound view asks at ``K`` points a condition that must hold everywhere, so the posterior statistic holds at the grid points and can fall short between them. [`ep_add_evar_view!`](@ref) and [`ep_add_rlvar_view!`](@ref) file those rows into the entropy pooling constraint dictionary rather than into the model.

# Arguments

  - `model`: Entropy pooling JuMP model.
  - `pw`: Vector of posterior probability variables.
  - `tv`: Tail view constraint.
  - `sc1`: Constraint scaling factor.

# Returns

  - `nothing`: The function mutates `model` in-place.

# Related

  - [`AbstractEntropyPoolingTailView`](@ref)
  - [`ep_tail_dual_block!`](@ref)
  - [`LinearConditionalValueatRiskViewConstraint`](@ref)
  - [`IntegerConditionalValueatRiskViewConstraint`](@ref)
  - [`ConicEntropicValueatRiskViewConstraint`](@ref)
  - [`GridEntropicValueatRiskViewConstraint`](@ref)
  - [`ConicRelativisticValueatRiskViewConstraint`](@ref)
  - [`GridRelativisticValueatRiskViewConstraint`](@ref)
  - [`AbstractSequentialTailViewConstraint`](@ref)
  - [`entropy_pooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
"""
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::Union{<:LinearConditionalValueatRiskViewConstraint,
                                     <:ConicEntropicValueatRiskViewConstraint,
                                     <:ConicRelativisticValueatRiskViewConstraint},
                           sc1::Number)
    (; x, coef, rhs) = tv
    expr = JuMP.AffExpr()
    for (xi, ci) in zip(x, coef)
        JuMP.add_to_expression!(expr, ci, ep_tail_dual_block!(model, pw, tv, xi, sc1))
    end
    JuMP.@constraint(model, sc1 * (rhs - expr) <= 0)
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw, tv::AbstractSequentialTailViewConstraint,
                           sc1::Number)
    (; xd, cd, c, b, rhs) = tv
    expr = JuMP.AffExpr(b)
    for (xi, ci) in zip(xd, cd)
        JuMP.add_to_expression!(expr, ci, ep_tail_dual_block!(model, pw, tv, xi, sc1))
    end
    JuMP.add_to_expression!(expr, LinearAlgebra.dot(c, pw))
    # The tail function of the relativistic measure can reach the thousands at a small
    # dual variable, so the row is divided by its largest coefficient, as the grid rows are.
    # Below one the row is left alone, so a CVaR row keeps its natural units.
    s = max(one(eltype(c)), maximum(abs, c; init = zero(eltype(c))))
    JuMP.@constraint(model, sc1 * (rhs - expr) / s <= 0)
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::IntegerConditionalValueatRiskViewConstraint, sc1::Number)
    (; ord, x, coef, alpha, op, rhs) = tv
    expr = JuMP.AffExpr()
    for (ordi, xi, ci) in zip(ord, x, coef)
        sb = length(ordi)
        y = JuMP.@variable(model, [1:sb], binary = true)
        q = JuMP.@variable(model, [1:sb], lower_bound = 0)
        JuMP.@constraints(model,
                          begin
                              [j = 1:sb], sc1 * (q[j] - y[j]) <= 0
                              [j = 1:sb], sc1 * (q[j] - pw[ordi[j]]) <= 0
                              [j = 2:sb],
                              sc1 * (pw[ordi[j]] - (one(alpha) - y[j - 1]) - q[j]) <= 0
                              [j = 1:(sb - 1)], sc1 * (y[j] - y[j + 1]) <= 0
                              sc1 * (sum(q) - alpha) == 0
                          end)
        JuMP.add_to_expression!(expr, ci / alpha, LinearAlgebra.dot(q, xi))
    end
    if op == :eq
        JuMP.@constraint(model, sc1 * (expr - rhs) == 0)
    elseif op == :geq
        JuMP.@constraint(model, sc1 * (rhs - expr) <= 0)
    else
        JuMP.@constraint(model, sc1 * (expr - rhs) <= 0)
    end
    return nothing
end
