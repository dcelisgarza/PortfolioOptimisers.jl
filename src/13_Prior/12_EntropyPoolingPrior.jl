"""
    ep_evar(x::VecNum, w::VecNum, alpha::Number; args::Tuple = (),
            kwargs::NamedTuple = (;), zlo::Option{<:Number} = nothing)

Compute the sample entropic value at risk of a loss series and the dual variable that attains it.

`ep_evar` minimises the scalar convex objective of the sample EVaR formula with [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). It is used by the entropy pooling view machinery, which needs both the value (to compare a view against its prior) and the minimiser (to centre the grid of [`GridEntropicValueatRiskView`](@ref)).

# Mathematical definition

```math
\\mathrm{EVaR}_{\\alpha}(X) = \\min_{z > 0} \\; z \\ln\\left(\\dfrac{\\sum_{j=1}^{T} w_{j} \\exp(x_{j}/z)}{\\alpha}\\right)\\,.
```

# Algorithm

 1. Normalise the observation probabilities in the logarithmic domain, giving `lw`.
 2. Bracket the dual variable. The upper end `hi` is `(maximum(x) - dot(w, x)) / log(inv(alpha))`, replaced by `eps` of its own type where that is not positive, and the lower end is `hi * zlo`.
 3. Minimise the objective over the bracket with [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl). Each evaluation goes through `LogExpFunctions.logsumexp`, so a small `z` does not overflow.
 4. Return the minimum as `evar`, and the minimiser as `z`.

# Arguments

  - `x`: Loss series (`-returns`).
  - `w`: Observation probabilities. Normalised to sum to one.
  - `alpha`: Significance level.
  - $(arg_dict[:optargs]) Left empty it takes `Optim.Brent()`, which is what `Optim.optimize` selects for a bracketed scalar minimisation.
  - $(arg_dict[:optkwargs])
  - `zlo`: Lower end of the bracket of the dual variable, as a fraction of the upper end. `nothing` takes `sqrt(eps(T))` for the element type `T`, which the caller cannot state because the type follows from the data. The upper end is `(maximum(x) - dot(w, x)) / log(inv(alpha))`, above which the objective already exceeds `maximum(x)`, which bounds the EVaR from above. That is a proof, so the upper end is not a knob and only the lower one is.

# Validation

  - `0 < zlo < 1`.
  - The search converges. It is a bracketed scalar minimisation of a convex function, so it fails only under `args` or `kwargs` that stop it early.

# Returns

  - `res::@NamedTuple{evar::Number, z::Number}`: The value and the dual variable that attains it.

# Related

  - [`GridEntropicValueatRiskView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`EntropicValueatRiskView`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
function ep_evar(x::VecNum, w::VecNum, alpha::Number; args::Tuple = (),
                 kwargs::NamedTuple = (;), zlo::Option{<:Number} = nothing)
    lw = log.(w)
    lw .-= LogExpFunctions.logsumexp(lw)
    ila = -log(alpha)
    f = function (z)
        return z * (LogExpFunctions.logsumexp(lw .+ x ./ z) + ila)
    end
    hi = (maximum(x) - LinearAlgebra.dot(exp.(lw), x)) / ila
    ehi = eps(float(typeof(hi)))
    hi = ifelse(hi > zero(hi), hi, ehi)
    # The default lower end is the one the element type states, which a caller that holds no
    # data cannot, so `nothing` resolves here rather than in the view that carries it.
    zlo = isnothing(zlo) ? sqrt(ehi) : zlo
    @argcheck(zero(zlo) < zlo < one(zlo), DomainError(zlo, "zlo must be in (0, 1)"))
    lo = hi * zlo
    res = Optim.optimize(f, lo, hi, args...; kwargs...)
    @argcheck(Optim.converged(res),
              ErrorException("The search for the sample EVaR did not converge. Relax the `args` and `kwargs` of the view group, or leave them empty to take the defaults."))
    return (; evar = Optim.minimum(res), z = Optim.minimizer(res))
end
"""
    ep_evar_grid_row(x::VecNum, ebar::Number, z::Number)

Build one scaled row of the grid formulation of an entropic value-at-risk view.

`ep_evar_grid_row` returns the coefficients of `exp((x - ebar) / z)` divided by their largest entry, together with the reciprocal of that entry, which the right hand side must be multiplied by. Scaling the row keeps the coefficients in `(0, 1]` however small `z` is, which is what lets the big-M constant of [`GridEntropicValueatRiskView`](@ref) be a plain number rather than a function of the data.

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
  - `bracket`: Spans of the searches, or `nothing` to take the one [`RelativisticValueatRiskViewBracket`](@ref) states. This function reads `zlo` and `zhi`, the ends of the bracket of the logarithm of the dual variable, as offsets from the logarithm of the loss range. They are a margin, not a proof, so widen one where the minimising dual variable lands on an end of the bracket. [`ep_rlvar_shift`](@ref) reads `tspan`.

# Validation

  - Both searches converge. Each is a bracketed scalar minimisation of a convex function, so one fails only under `args` or `kwargs` that stop it early. A minimiser that lands on an end of a bracket does not fail it: `Optim` reports that end as converged.

# Returns

  - `res::@NamedTuple{rlvar::Number, t::Number, z::Number}`: The value and the primal pair that attains it.

# Algorithm

 1. Minimise over the logarithm of the dual variable with [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl), over a bracket running from `exp(zlo)` to `exp(zhi)` times the loss range, which is about `2e-9` to about `2e4` under the default bracket. The objective is convex in the pair, so the partial minimum over the shift is convex in the dual variable, and the logarithm is increasing, so the outer minimisation sees a unimodal function.
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
    zlo, zhi = bkt.zlo, bkt.zhi
    T = length(x)
    wi = w ./ sum(w)
    lnk = kappa_log(inv(alpha * T), kappa)
    xmin, xmax = extrema(x)
    span = xmax - xmin
    span = ifelse(span > zero(span), span, max(abs(xmax), one(xmax)))
    lspan = log(span)
    res = Optim.optimize(u -> ep_rlvar_shift(x, wi, kappa, lnk, exp(u); args = args,
                                             kwargs = kwargs, bracket = bracket).risk,
                         lspan + zlo, lspan + zhi, args...; kwargs...)
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

`ep_rlvar_grid_row` returns the coefficients `T * phi(t - x, z)` divided by their largest entry, together with the target of the row divided by that same entry. Scaling the row keeps the coefficients in `(0, 1]` however small `z` is, which is what lets the big-M constant of [`GridRelativisticValueatRiskView`](@ref) be a plain number rather than a function of the data.

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

The row's value under the tilt falls strictly as ``\\theta`` rises, from ``\\max_{j} c_{j}`` to ``\\min_{j} c_{j}``, so the tilt exists exactly when ``b`` sits strictly inside that range.

# Arguments

  - `w`: Prior probabilities, summing to one.
  - `c`: Coefficients of the row.
  - `b`: Value the row is to take.
  - `iters::Integer = 200`: Largest number of bisection steps. The bisection stops on its own when the midpoint stops moving, which for `Float64` happens near step 64, so this binds only a type of higher precision.

# Validation

  - `iters >= 1`.

# Returns

  - `q::Option{VecNum}`: The tilted probabilities, or `nothing` when `b` sits outside the range of `c` and no probability vector attains it.

# Algorithm

 1. Return `nothing` when `b` sits outside the open range of `c`.
 2. Bracket the root by doubling the tilt away from zero until the row's value crosses `b`.
 3. Bisect the bracket to the resolution of the floating-point type, or for `iters` steps, whichever comes first.

# Related

  - [`ep_rlvar_anchor`](@ref)
  - [`ep_rlvar_grid_row`](@ref)

# References

  - $(ref_dict[:EPRLVaR])
"""
function ep_row_tilt(w::VecNum, c::VecNum, b::Number; iters::Integer = 200)
    @argcheck(iters >= one(iters), DomainError(iters, "iters must be >= 1"))
    lo, hi = extrema(c)
    if !(lo < b < hi)
        return nothing
    end
    q = Vector{float(promote_type(eltype(w), eltype(c), typeof(b)))}(undef, length(w))
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
    return q ./ sum(q)
end
"""
    ep_evar_anchor(x::VecNum, w::VecNum, alpha::Number, rhs::Number, z::Number;
                   iters::Integer = 50, tol::Number = 1e-10, tilt_iters::Integer = 200,
                   args::Tuple = (), kwargs::NamedTuple = (;),
                   zlo::Option{<:Number} = nothing)

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
  - `zlo`: Lower end of the bracket, forwarded to [`ep_evar`](@ref).

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
                        zlo::Option{<:Number} = nothing)
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
        res = ep_evar(x, q, alpha; args = args, kwargs = kwargs, zlo = zlo)
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
                 kwargs::NamedTuple = (;), zlo::Option{<:Number} = nothing)

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
  - `zlo`: Lower end of the bracket, forwarded to [`ep_evar`](@ref).

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
                      kwargs::NamedTuple = (;), zlo::Option{<:Number} = nothing)
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
                       tilt_iters = tilt_iters, args = args, kwargs = kwargs, zlo = zlo)
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

Carries the loss series, the significance level and the target of a linear conditional value-at-risk view.

The view parser produces one of these per view that takes the linear formulation. [`add_ep_tail_view!`](@ref) then writes the dual representation of CVaR into the model from it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearConditionalValueatRiskViewConstraint(x, alpha, rhs)

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
    $(field_dict[:ep_loss])
    """
    x
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

Carries the loss series, the significance level and the target of a conic entropic value-at-risk view.

The view parser produces one of these per view that takes the conic formulation. [`add_ep_tail_view!`](@ref) then writes the relative entropy cone that is the dual representation of EVaR from it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConicEntropicValueatRiskViewConstraint(x, alpha, rhs)

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
    $(field_dict[:ep_loss])
    """
    x
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

Carries the loss series, the significance level, the deformation parameter and the target of a conic relativistic value-at-risk view.

The view parser produces one of these per view that takes the conic formulation. [`add_ep_tail_view!`](@ref) then writes the power cones that are the dual representation of RLVaR from it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConicRelativisticValueatRiskViewConstraint(x, alpha, kappa, rhs)

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
    $(field_dict[:ep_loss])
    """
    x
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
    add_ep_tail_view!(model::JuMP.Model, pw, tv::AbstractEntropyPoolingTailView,
                      sc1::Number)

Add the variables and constraints of one tail view to an entropy pooling JuMP model.

`add_ep_tail_view!` is the one seam through which a conditional, entropic or relativistic value-at-risk view reaches the model. Each formulation has its own method, dispatched on the constraint carrier the view parser produced.

# JuMP formulation

The section covers the six methods, and every entry each of them registers. Each entry is anonymous: one model carries one block per view, so a name would collide on the second view of a family, and nothing reads these entries back by name.

## Variables

  - `pw`: $(math_dict[:ep_post_probs]) It is read from the caller, and every entry below is registered against it.
  - `nu`: ``\\boldsymbol{\\nu}``, ``T \\times 1``, created by the [`LinearConditionalValueatRiskViewConstraint`](@ref), [`ConicEntropicValueatRiskViewConstraint`](@ref) and [`ConicRelativisticValueatRiskViewConstraint`](@ref) methods. The first bounds it below by zero, and the other two bound it to ``[0, 1]``.
  - `y`, `q`: ``\\boldsymbol{y}`` and ``\\boldsymbol{q}``, ``\\bar{s} \\times 1`` each, created once per asset by the [`IntegerConditionalValueatRiskViewConstraint`](@ref) method. `y` is binary, and `q` is bounded below by zero.
  - `y`: ``\\boldsymbol{y}``, ``K \\times 1`` and binary, created by the [`GridEntropicValueatRiskViewConstraint`](@ref) and [`GridRelativisticValueatRiskViewConstraint`](@ref) methods. It selects the grid point the view is met at.
  - `tau`, `varsigma`: ``\\boldsymbol{\\tau}`` and ``\\boldsymbol{\\varsigma}``, ``T \\times 1`` each and bounded below by zero, created by the [`ConicRelativisticValueatRiskViewConstraint`](@ref) method. The cones and the budget already imply both bounds, and stating them is what turns a `SLOW_PROGRESS` report into an `OPTIMAL` one.

## Constraints

The [`LinearConditionalValueatRiskViewConstraint`](@ref) method registers three rows:

  - ``s_{c1} \\left(\\nu_{j} - \\dfrac{p_{j}}{\\alpha}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,T``.
  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\nu_{j} - 1\\right) = 0``.
  - ``s_{c1} \\left(\\bar{c} - \\sum_{j=1}^{T} \\nu_{j} x_{j}\\right) \\leq 0``.

The [`ConicEntropicValueatRiskViewConstraint`](@ref) method registers three rows:

  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\nu_{j} - 1\\right) = 0``.
  - ``s_{c1} \\left(\\bar{e} - \\sum_{j=1}^{T} \\nu_{j} x_{j}\\right) \\leq 0``.
  - ``\\left(s_{c1} \\ln\\left(\\dfrac{1}{\\alpha}\\right),\\, s_{c1} \\boldsymbol{p},\\, s_{c1} \\boldsymbol{\\nu}\\right) \\in \\mathcal{K}_{\\mathrm{re}}(2T+1)``.

The [`IntegerConditionalValueatRiskViewConstraint`](@ref) method registers five rows per asset the view names, over that asset's window of the ``\\bar{s}`` largest losses:

  - ``s_{c1} \\left(q_{j} - y_{j}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,\\bar{s}``.
  - ``s_{c1} \\left(q_{j} - p_{[j]}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,\\bar{s}``.
  - ``s_{c1} \\left(p_{[j]} - (1 - y_{j}) - q_{j}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,\\bar{s}``.
  - ``s_{c1} \\left(y_{j} - y_{j+1}\\right) \\leq 0``, ``\\forall\\, j = 1,\\ldots,\\bar{s}-1``.
  - ``s_{c1} \\left(\\sum_{j=1}^{\\bar{s}} q_{j} - \\alpha\\right) = 0``.

and one further row on ``\\varepsilon = \\sum_{i} \\dfrac{\\gamma_{i}}{\\alpha} \\sum_{j=1}^{\\bar{s}} q_{i,\\,j} x_{i,\\,[j]}``, the view's operator picking which of the three:

  - ``s_{c1} \\left(\\varepsilon - \\bar{c}\\right) = 0`` under `:eq`.
  - ``s_{c1} \\left(\\bar{c} - \\varepsilon\\right) \\leq 0`` under `:geq`.
  - ``s_{c1} \\left(\\varepsilon - \\bar{c}\\right) \\leq 0`` under `:leq`.

The [`GridEntropicValueatRiskViewConstraint`](@ref) method registers two rows:

  - ``s_{c1} \\left(\\sum_{k=1}^{K} y_{k} - 1\\right) = 0``.
  - ``s_{c1} \\left(\\sum_{j=1}^{T} c_{k,\\,j} p_{j} - \\alpha \\iota_{k} - M (1 - y_{k})\\right) \\leq 0``, ``\\forall\\, k = 1,\\ldots,K``.

The [`ConicRelativisticValueatRiskViewConstraint`](@ref) method registers five rows:

  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\nu_{j} - 1\\right) = 0``.
  - ``s_{c1} \\left(\\bar{\\vartheta} - \\sum_{j=1}^{T} \\nu_{j} x_{j}\\right) \\leq 0``.
  - ``s_{c1} \\left(\\sum_{j=1}^{T} \\dfrac{\\tau_{j} - \\varsigma_{j}}{2\\kappa} - \\ln_{\\kappa}\\left(\\dfrac{1}{\\alpha T}\\right)\\right) \\leq 0``.
  - ``\\left(s_{c1} \\tau_{j},\\, s_{c1} T p_{j},\\, s_{c1} \\nu_{j}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\left(\\dfrac{1}{1+\\kappa}\\right)``, ``\\forall\\, j = 1,\\ldots,T``.
  - ``\\left(s_{c1} \\nu_{j},\\, s_{c1} T p_{j},\\, s_{c1} \\varsigma_{j}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa)``, ``\\forall\\, j = 1,\\ldots,T``.

The [`GridRelativisticValueatRiskViewConstraint`](@ref) method registers two rows:

  - ``s_{c1} \\left(\\sum_{k=1}^{K} y_{k} - 1\\right) = 0``.
  - ``s_{c1} \\left(\\sum_{j=1}^{T} c_{k,\\,j} p_{j} - b_{k} - M (1 - y_{k})\\right) \\leq 0``, ``\\forall\\, k = 1,\\ldots,K``.

Where:

  - $(math_dict[:ep_sc1])
  - $(math_dict[:ep_post_probs])
  - $(math_dict[:ep_tail_nu])
  - $(math_dict[:rlvar_loss])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:cvar_target])
  - $(math_dict[:evar_target])
  - $(math_dict[:rlvar_target])
  - ``\\bar{s}``: Length of one asset's window of largest losses, from [`ep_sbar`](@ref).
  - ``x_{[j]}``, ``p_{[j]}``: Loss and posterior probability of the observation in position ``j`` of that window, which is sorted ascending.
  - ``\\gamma_{i}``: Coefficient the view gives asset ``i``.
  - ``\\boldsymbol{y}``: Binary vector. It marks the tail of one asset's window in the integer conditional value-at-risk method, and selects one grid point in the two grid methods.
  - ``\\boldsymbol{q}``: ``\\bar{s} \\times 1`` vector that carries the product ``q_{j} = p_{[j]} y_{j}``.
  - ``\\boldsymbol{\\tau}``, ``\\boldsymbol{\\varsigma}``: ``T \\times 1`` vectors that carry the Kaniadakis entropy budget of ``\\boldsymbol{\\nu}``.
  - ``\\varepsilon``: Left hand side of an integer conditional value-at-risk view, the coefficient-weighted sum of the per-asset posterior CVaRs.
  - ``K``: Number of grid points the carrier holds.
  - ``c_{k,\\,j}``: Scaled coefficient of observation ``j`` at grid point ``k``, from [`ep_evar_grid_row`](@ref) or [`ep_rlvar_grid_row`](@ref).
  - ``\\iota_{k}``, ``b_{k}``: Scaled target of grid point ``k``, from those same two functions.
  - ``M``: Big-M constant the grid carrier holds.
  - ``\\mathcal{K}_{\\mathrm{re}}(1 + 2T) = \\{(u,\\, \\boldsymbol{v},\\, \\boldsymbol{s}) : u \\geq \\sum_{j=1}^{T} s_{j} \\ln(s_{j} / v_{j})\\}``: Relative entropy cone.
  - ``\\mathcal{K}_{\\mathrm{pow}}(\\pi) = \\{(a, b, c) : a^{\\pi} b^{1-\\pi} \\geq |c|,\\, a \\geq 0,\\, b \\geq 0\\}``: Power cone.

## Relaxation

$(val_dict[:relax])

The two grid methods bound the statistic. The four other methods are exact.

 1. **Direction.** Every grid point is a feasible point of the primal programme of the statistic, so its row bounds the statistic from above. The block asks one grid point to hold, so the posterior statistic lies at or below the target. The encoding is a **restriction**: it can only be tighter than the view asks, and the view is never violated.
 2. **Quantity.** The posterior entropic value at risk under the [`GridEntropicValueatRiskViewConstraint`](@ref) method, and the posterior relativistic value at risk under the [`GridRelativisticValueatRiskViewConstraint`](@ref) method. Both are statistics of `pw`.
 3. **Tightness.** The bound is tight where the grid holds the point the posterior itself attains. [`ep_evar_anchor`](@ref) and [`ep_rlvar_anchor`](@ref) put the centre of the grid on that point. Where the anchor does not converge the grid falls back to the prior's point, and the posterior statistic can land strictly below the target. Widen `pct` or raise `K` there.

``M`` releases the rows of the grid points the selector does not pick. A row's coefficients sit in ``(0, 1]`` and ``\\boldsymbol{p}`` sums to one, so the left hand side never exceeds one and the default ``M`` of both carriers clears it. An ``M`` below that bound restricts the model further, in the same direction.

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
  - [`LinearConditionalValueatRiskViewConstraint`](@ref)
  - [`IntegerConditionalValueatRiskViewConstraint`](@ref)
  - [`ConicEntropicValueatRiskViewConstraint`](@ref)
  - [`GridEntropicValueatRiskViewConstraint`](@ref)
  - [`ConicRelativisticValueatRiskViewConstraint`](@ref)
  - [`GridRelativisticValueatRiskViewConstraint`](@ref)
  - [`entropy_pooling`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
"""
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::LinearConditionalValueatRiskViewConstraint, sc1::Number)
    (; x, alpha, rhs) = tv
    T = length(x)
    nu = JuMP.@variable(model, [1:T], lower_bound = 0)
    JuMP.@constraints(model, begin
                          [j = 1:T], sc1 * (nu[j] - pw[j] / alpha) <= 0
                          sc1 * (sum(nu) - one(alpha)) == 0
                          sc1 * (rhs - LinearAlgebra.dot(nu, x)) <= 0
                      end)
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::ConicEntropicValueatRiskViewConstraint, sc1::Number)
    (; x, alpha, rhs) = tv
    T = length(x)
    nu = JuMP.@variable(model, [1:T], lower_bound = 0, upper_bound = 1)
    JuMP.@constraints(model,
                      begin
                          sc1 * (sum(nu) - one(alpha)) == 0
                          sc1 * (rhs - LinearAlgebra.dot(nu, x)) <= 0
                          [sc1 * log(inv(alpha)); sc1 * pw; sc1 * nu] in
                          JuMP.MOI.RelativeEntropyCone(2 * T + 1)
                      end)
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
                              [j = 1:sb],
                              sc1 * (pw[ordi[j]] - (one(alpha) - y[j]) - q[j]) <= 0
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
function add_ep_tail_view!(model::JuMP.Model, pw, tv::GridEntropicValueatRiskViewConstraint,
                           sc1::Number)
    (; x, z, alpha, rhs, M) = tv
    K = length(z)
    y = JuMP.@variable(model, [1:K], binary = true)
    JuMP.@constraint(model, sc1 * (sum(y) - one(alpha)) == 0)
    for (k, zk) in pairs(z)
        c, isc = ep_evar_grid_row(x, rhs, zk)
        JuMP.@constraint(model,
                         sc1 *
                         (LinearAlgebra.dot(c, pw) - alpha * isc - M * (one(alpha) - y[k])) <=
                         0)
    end
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::ConicRelativisticValueatRiskViewConstraint, sc1::Number)
    (; x, alpha, kappa, rhs) = tv
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
                          sc1 * (rhs - LinearAlgebra.dot(nu, x)) <= 0
                          sc1 * (sum(tau - varsigma) * ik2 - lnk) <= 0
                          [j = 1:T],
                          [sc1 * tau[j], sc1 * T * pw[j], sc1 * nu[j]] in
                          JuMP.MOI.PowerCone(inv(opk))
                          [j = 1:T],
                          [sc1 * nu[j], sc1 * T * pw[j], sc1 * varsigma[j]] in
                          JuMP.MOI.PowerCone(omk)
                      end)
    return nothing
end
function add_ep_tail_view!(model::JuMP.Model, pw,
                           tv::GridRelativisticValueatRiskViewConstraint, sc1::Number)
    (; x, t, z, alpha, kappa, rhs, M) = tv
    K = length(z)
    y = JuMP.@variable(model, [1:K], binary = true)
    JuMP.@constraint(model, sc1 * (sum(y) - one(alpha)) == 0)
    for k in 1:K
        c, b = ep_rlvar_grid_row(x, rhs, t[k], z[k], alpha, kappa)
        JuMP.@constraint(model,
                         sc1 * (LinearAlgebra.dot(c, pw) - b - M * (one(alpha) - y[k])) <=
                         0)
    end
    return nothing
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:evar}, alpha::Number,
                 w::Option{<:ObsWeights} = nothing, args::Tuple = (),
                 kwargs::NamedTuple = (;), zlo::Option{<:Number} = nothing)

Extract the Entropic Value-at-Risk (EVaR) for asset `i` from a prior result.

`get_pr_value` computes the EVaR at confidence level `alpha` for the asset indexed by `i` from the prior result object `pr`, by minimising the scalar objective of the sample EVaR formula with [`ep_evar`](@ref). The observations carry `w`, the weights the initial prior result was read at. A `w` of `nothing` leaves them uniform.

# Arguments

  - `pr`: Prior result containing asset return information. Only its returns matrix is read, under the weights `w` names.
  - `i`: Index of the asset.
  - `::Val{:evar}`: Dispatch tag for EVaR extraction.
  - `alpha`: Confidence level (e.g. `0.05` for 5% EVaR).
  - $(arg_dict[:oow])
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `zlo`: Lower end of the bracket, forwarded to [`ep_evar`](@ref).

# Returns

  - `evar::Number`: Entropic Value-at-Risk for asset `i` at level `alpha`.

# Related

  - [`ep_evar`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:evar}, alpha::Number,
                      w::Option{<:ObsWeights} = nothing, args::Tuple = (),
                      kwargs::NamedTuple = (;), zlo::Option{<:Number} = nothing)
    T = size(pr.X, 1)
    iT = inv(T)
    w = isnothing(w) ? range(iT, iT; length = T) : w
    return ep_evar(-view(pr.X, :, i), w, alpha; args = args, kwargs = kwargs, zlo = zlo).evar
end
"""
    get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:rlvar}, alpha::Number,
                 kappa::Number, w::Option{<:ObsWeights} = nothing, args::Tuple = (),
                 kwargs::NamedTuple = (;),
                 bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Extract the Relativistic Value-at-Risk (RLVaR) for asset `i` from a prior result.

`get_pr_value` computes the RLVaR at confidence level `alpha` and deformation parameter `kappa` for the asset indexed by `i` from the prior result object `pr`, by minimising the primal objective of the sample RLVaR with [`ep_rlvar`](@ref). The observations carry `w`, the weights the initial prior result was read at, on the reasoning the entropic value at risk method above gives.

# Arguments

  - `pr`: Prior result containing asset return information. Only its returns matrix is read, under the weights `w` names.
  - `i`: Index of the asset.
  - `::Val{:rlvar}`: Dispatch tag for RLVaR extraction.
  - `alpha`: Confidence level (e.g. `0.05` for 5% RLVaR).
  - `kappa`: Deformation parameter, in `(0, 1)`.
  - $(arg_dict[:oow])
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `bracket`: Spans of the searches, forwarded to [`ep_rlvar`](@ref) and [`ep_rlvar_shift`](@ref).

# Returns

  - `rlvar::Number`: Relativistic Value-at-Risk for asset `i` at level `alpha` and deformation `kappa`.

# Related

  - [`ep_rlvar`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`get_pr_value`](@ref)
"""
function get_pr_value(pr::AbstractPriorResult, i::Integer, ::Val{:rlvar}, alpha::Number,
                      kappa::Number, w::Option{<:ObsWeights} = nothing, args::Tuple = (),
                      kwargs::NamedTuple = (;),
                      bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    T = size(pr.X, 1)
    iT = inv(T)
    w = isnothing(w) ? range(iT, iT; length = T) : w
    return ep_rlvar(-view(pr.X, :, i), w, alpha, kappa; args = args, kwargs = kwargs,
                    bracket = bracket).rlvar
end
"""
    ep_view_terms(res::ParsingResult, sets::UniverseSets, X::MatNum; strict::Bool = false)

Resolve one parsed tail view into the assets it names, their coefficients, its operator and its target.

`ep_view_terms` routes a [`ParsingResult`](@ref) through [`get_linear_constraints`](@ref), which resolves the variable names against the universe and reports the ones it cannot place, then undoes the sign flip that entry point applies to a `>=` equation so the operator survives. The linear view machinery never needs the operator back, because a row of `A x <= b` carries it; a tail view does, because each operator picks a different formulation.

# Algorithm

 1. Resolve the view against the universe with [`get_linear_constraints`](@ref), giving `lc`. Return `nothing` where it places no name of the view.
 2. Read the sign `sgn` and the inequality flag of the view's operator with [`comparison_sign_ineq_flag`](@ref), and pick from them the operator `op` the view carries and the block `blk` it landed in.
 3. Scale the row `A` and the target `rhs` by `sgn`, which undoes the flip [`get_linear_constraints`](@ref) applies to a `>=` equation.
 4. Return the indices of the non-zero entries of `A`, the coefficients at those indices, `op` and `rhs`.

# Arguments

  - `res`: Parsed view constraint.
  - `sets`: Asset set mapping asset names to indices.
  - `X`: Asset returns matrix, read for its element type.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: If no name in the view could be placed in the universe.
  - `terms::@NamedTuple{idx::VecInt, coef::VecNum, op::Symbol, rhs::Number}`: The assets the view names, their coefficients, its operator (`:eq`, `:geq` or `:leq`) and its target.

# Related

  - [`ep_tail_views!`](@ref)
  - [`get_linear_constraints`](@ref)
  - [`comparison_sign_ineq_flag`](@ref)
"""
function ep_view_terms(res::ParsingResult, sets::UniverseSets, X::MatNum;
                       strict::Bool = false)
    lc = get_linear_constraints([res], sets; datatype = eltype(X), strict = strict)
    if isnothing(lc)
        return nothing
    end
    sgn, flag = comparison_sign_ineq_flag(res.op)
    op, blk = if !flag
        :eq, lc.eq
    elseif sgn == -1
        :geq, lc.ineq
    else
        :leq, lc.ineq
    end
    # `get_linear_constraints` negates a `>=` equation to file it as a `<=` row, scaling by
    # the same sign, so scaling by it again undoes the flip.
    d = eltype(X)(sgn)
    A = vec(blk.A) * d
    rhs = blk.B[1] * d
    idx = findall(!iszero, A)
    return (; idx = idx, coef = A[idx], op = op, rhs = rhs)
end
"""
    ep_normalise_view_term(coef::Number, op::Symbol, rhs::Number)

Divide a single-asset tail view by its coefficient.

Returns the operator and target of the equivalent view whose coefficient is one, flipping the operator when the coefficient is negative.

# Algorithm

 1. Divide the target by the coefficient, giving `rhs`.
 2. Where the coefficient is negative, exchange `:geq` and `:leq`, giving `op`. An equality is unchanged, because dividing both sides by a negative number preserves it.

# Arguments

  - `coef`: Coefficient the view gives the asset's risk measure.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.

# Returns

  - `op::Symbol`: Operator of the normalised view.
  - `rhs::Number`: Target of the normalised view.

# Related

  - [`ep_view_terms`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_normalise_view_term(coef::Number, op::Symbol, rhs::Number)
    rhs /= coef
    if coef < zero(coef)
        op = if op == :geq
            :leq
        elseif op == :leq
            :geq
        else
            op
        end
    end
    return op, rhs
end
"""
    ep_view_formulations(alg, N::Integer, key::Symbol)

Spread the tail view formulation setting of an entropy pooling prior over its views.

A single formulation applies to every view, a vector supplies one per view, and `nothing` leaves the choice to the view.

# Arguments

  - `alg`: Formulation setting.
  - `N`: Number of views.
  - `key`: Field name, used in the error message.

# Validation

  - If `alg` is a vector, `length(alg) == N`.

# Returns

  - `algs::AbstractVector`: One entry per view.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_view_formulations(alg::Option{<:AbstractEntropyPoolingViewFormulation},
                              N::Integer, ::Symbol)
    return fill(alg, N)
end
function ep_view_formulations(alg::AbstractVector, N::Integer, key::Symbol)
    @argcheck(length(alg) == N,
              DimensionMismatch("length($key) ($(length(alg))) must match the number of views ($N)"))
    return alg
end
"""
    ep_sbar(sbar, T::Integer, alpha::Number, w::VecNum, ord::VecInt)

Resolve the number of largest losses the integer conditional value-at-risk formulation considers.

# Algorithm

 1. An `Integer` `sbar` is a count. Return it, capped at `T`.
 2. Any other number is a fraction of `T`. Return `ceil(Int, sbar * T)`, never below one and capped at `T`.
 3. `nothing` takes the rule of thumb of [EPTail](@cite). Walk the losses from the largest down, accumulating the prior probabilities, and stop at the position `s` at which they first reach `alpha`.
 4. Return twice `s`, never below `ceil(Int, 2 * alpha * T)`, never below one and capped at `T`. A view above the prior CVaR moves mass into the tail and needs about the position `s`; a view below it moves mass out and needs more.

# Arguments

  - `sbar`: Setting held by [`IntegerConditionalValueatRiskView`](@ref). An `Integer` is a count, a fraction in `(0, 1)` is a fraction of `T`, and `nothing` applies the rule of thumb of [EPTail](@cite).
  - `T`: Number of observations.
  - `alpha`: Significance level of the view.
  - `w`: Prior probability weights.
  - `ord`: Indices of the losses in ascending order, so the largest loss is last.

# Returns

  - `sbar::Int`: Number of largest losses, in `1:T`.

# Related

  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ep_tail_views!`](@ref)

# References

  - $(ref_dict[:EPTail])
"""
function ep_sbar(sbar::Nothing, T::Integer, alpha::Number, w::VecNum, ord::VecInt)
    cw = zero(eltype(w))
    s = T
    for (j, o) in enumerate(Iterators.reverse(ord))
        cw += w[o]
        if cw >= alpha
            s = j
            break
        end
    end
    return min(T, max(2 * s, ceil(Int, 2 * alpha * T), 1))
end
function ep_sbar(sbar::Integer, T::Integer, args...)
    return min(T, sbar)
end
function ep_sbar(sbar::Number, T::Integer, args...)
    return min(T, max(1, ceil(Int, sbar * T)))
end
"""
    ep_assert_reachable_view(op::Symbol, rhs::Number, x::VecNum, eqn::AbstractString,
                             name::AbstractString)

Reject a tail view no reweighting of the sample can reach.

A tail risk measure of a reweighted sample lies between the smallest and the largest loss the sample holds, so a view outside that band is infeasible however the probabilities move.

# Algorithm

 1. Where `op` asks the statistic to reach or exceed `rhs`, read the largest loss of `x` and raise unless `rhs` sits below it.
 2. Where `op` asks the statistic to reach or fall below `rhs`, read the smallest loss of `x` and raise unless `rhs` sits above it.

# Arguments

  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `x`: Loss series of the asset the view names.
  - `eqn`: Equation of the view, used in the error message.
  - `name`: Name of the view family, used in the error message.

# Validation

  - If `op` is `:geq` or `:eq`, `rhs < maximum(x)`.
  - If `op` is `:leq` or `:eq`, `rhs > minimum(x)`.

# Returns

  - `nothing`.

# Related

  - [`ep_tail_views!`](@ref)
"""
function ep_assert_reachable_view(op::Symbol, rhs::Number, x::VecNum, eqn::AbstractString,
                                  name::AbstractString)
    if op == :geq || op == :eq
        @argcheck(rhs < maximum(x),
                  DomainError(rhs,
                              "View `$(eqn)` is too extreme: the largest $(name) any reweighting of this sample reaches is its worst realisation, $(maximum(x)). Lower the view, raise alpha, or use a prior with fatter tails."))
    end
    if op == :leq || op == :eq
        @argcheck(rhs > minimum(x),
                  DomainError(rhs,
                              "View `$(eqn)` is too extreme: the smallest $(name) any reweighting of this sample reaches is its best realisation, $(minimum(x)). Raise the view, lower alpha, or use a prior with a thinner tail."))
    end
    return nothing
end
"""
    ep_cvar_formulation(alg::Option{<:AbstractConditionalValueatRiskViewFormulation}, single::Bool,
                        op::Symbol, rhs::Number, pv::Number)

Pick the formulation of one conditional value-at-risk view.

A stated formulation is returned unchanged. `nothing` takes [`LinearConditionalValueatRiskView`](@ref) wherever it expresses the view exactly, and [`IntegerConditionalValueatRiskView`](@ref) otherwise, which is every view the linear formulation cannot express: a relative view, an upper bound, and an equality below the prior CVaR.

The branch each input takes:

| `alg`     | `single` | `op`   | `rhs` against `pv` | Branch                                      |
|:--------- |:-------- |:------ |:------------------ |:------------------------------------------- |
| stated    | any      | any    | any                | `alg`, unchanged                            |
| `nothing` | `true`   | `:geq` | any                | [`LinearConditionalValueatRiskView`](@ref)  |
| `nothing` | `true`   | `:eq`  | `rhs >= pv`        | [`LinearConditionalValueatRiskView`](@ref)  |
| `nothing` | `true`   | `:eq`  | `rhs < pv`         | [`IntegerConditionalValueatRiskView`](@ref) |
| `nothing` | `true`   | `:leq` | any                | [`IntegerConditionalValueatRiskView`](@ref) |
| `nothing` | `false`  | any    | any                | [`IntegerConditionalValueatRiskView`](@ref) |

# Arguments

  - `alg`: Stated formulation, or `nothing`.
  - `single`: Whether the view names one asset.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `pv`: Prior value of the view's left hand side.

# Returns

  - `alg::AbstractConditionalValueatRiskViewFormulation`: The formulation to use.

# Related

  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_cvar_formulation(alg::AbstractConditionalValueatRiskViewFormulation, args...)
    return alg
end
function ep_cvar_formulation(::Nothing, single::Bool, op::Symbol, rhs::Number, pv::Number)
    return if single && (op == :geq || op == :eq && rhs >= pv)
        LinearConditionalValueatRiskView()
    else
        IntegerConditionalValueatRiskView()
    end
end
"""
    ep_evar_formulation(alg::Option{<:AbstractEntropicValueatRiskViewFormulation}, op::Symbol,
                        rhs::Number, pv::Number)

Pick the formulation of one entropic value-at-risk view.

A stated formulation is returned unchanged. `nothing` takes [`ConicEntropicValueatRiskView`](@ref) wherever it expresses the view exactly, and [`GridEntropicValueatRiskView`](@ref) otherwise, which is an upper bound and an equality below the prior EVaR.

The branch each input takes:

| `alg`     | `op`   | `rhs` against `pv` | Branch                                 |
|:--------- |:------ |:------------------ |:-------------------------------------- |
| stated    | any    | any                | `alg`, unchanged                       |
| `nothing` | `:geq` | any                | [`ConicEntropicValueatRiskView`](@ref) |
| `nothing` | `:eq`  | `rhs >= pv`        | [`ConicEntropicValueatRiskView`](@ref) |
| `nothing` | `:eq`  | `rhs < pv`         | [`GridEntropicValueatRiskView`](@ref)  |
| `nothing` | `:leq` | any                | [`GridEntropicValueatRiskView`](@ref)  |

# Arguments

  - `alg`: Stated formulation, or `nothing`.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `pv`: Prior EVaR of the asset the view names.

# Returns

  - `alg::AbstractEntropicValueatRiskViewFormulation`: The formulation to use.

# Related

  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_evar_formulation(alg::AbstractEntropicValueatRiskViewFormulation, args...)
    return alg
end
function ep_evar_formulation(::Nothing, op::Symbol, rhs::Number, pv::Number)
    return if op == :geq || op == :eq && rhs >= pv
        ConicEntropicValueatRiskView()
    else
        GridEntropicValueatRiskView()
    end
end
"""
    ep_rlvar_formulation(alg::Option{<:AbstractRelativisticValueatRiskViewFormulation},
                         op::Symbol, rhs::Number, pv::Number)

Pick the formulation of one relativistic value-at-risk view.

A stated formulation is returned unchanged. `nothing` takes [`ConicRelativisticValueatRiskView`](@ref) wherever it expresses the view exactly, and [`GridRelativisticValueatRiskView`](@ref) otherwise, which is an upper bound and an equality below the prior RLVaR.

The branch each input takes:

| `alg`     | `op`   | `rhs` against `pv` | Branch                                     |
|:--------- |:------ |:------------------ |:------------------------------------------ |
| stated    | any    | any                | `alg`, unchanged                           |
| `nothing` | `:geq` | any                | [`ConicRelativisticValueatRiskView`](@ref) |
| `nothing` | `:eq`  | `rhs >= pv`        | [`ConicRelativisticValueatRiskView`](@ref) |
| `nothing` | `:eq`  | `rhs < pv`         | [`GridRelativisticValueatRiskView`](@ref)  |
| `nothing` | `:leq` | any                | [`GridRelativisticValueatRiskView`](@ref)  |

# Arguments

  - `alg`: Stated formulation, or `nothing`.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `pv`: Prior RLVaR of the asset the view names.

# Returns

  - `alg::AbstractRelativisticValueatRiskViewFormulation`: The formulation to use.

# Related

  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_rlvar_formulation(alg::AbstractRelativisticValueatRiskViewFormulation, args...)
    return alg
end
function ep_rlvar_formulation(::Nothing, op::Symbol, rhs::Number, pv::Number)
    return if op == :geq || op == :eq && rhs >= pv
        ConicRelativisticValueatRiskView()
    else
        GridRelativisticValueatRiskView()
    end
end
"""
    ep_add_cvar_view!(tvs::AbstractVector, alg::AbstractConditionalValueatRiskViewFormulation, X::MatNum,
                      idx::VecInt, coef::VecNum, op::Symbol, rhs::Number, alpha::Number,
                      w::VecNum, pv::Number, eqn::AbstractString)

Lower one conditional value-at-risk view into the tail view constraint its formulation needs.

# Algorithm

 1. [`LinearConditionalValueatRiskView`](@ref) checks the three preconditions below, then appends one [`LinearConditionalValueatRiskViewConstraint`](@ref) carrying the loss series of the single asset the view names, `alpha` and `rhs`.
 2. [`IntegerConditionalValueatRiskView`](@ref) sorts the loss series of each asset the view names, giving the ascending order `o`, and resolves the length `sb` of that asset's tail window with [`ep_sbar`](@ref).
 3. It keeps the last `sb` positions of `o` as `ord[k]`, and the losses at those positions as `x[k]`.
 4. It appends one [`IntegerConditionalValueatRiskViewConstraint`](@ref) carrying those windows, `coef`, `alpha`, `op` and `rhs`.

# Arguments

  - `tvs`: Tail view constraints, appended to.
  - `alg`: Formulation of the view.
  - `X`: Asset returns matrix.
  - `idx`: Indices of the assets the view names.
  - `coef`: Coefficient the view gives each asset's CVaR.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `alpha`: Significance level of the view.
  - `w`: Prior probability weights.
  - `pv`: Prior value of the view's left hand side.
  - `eqn`: Equation of the view, used in the error messages.

# Validation

  - [`LinearConditionalValueatRiskView`](@ref) needs one asset, an operator other than `<=`, and, for an equality, a target at or above the prior CVaR.

# Returns

  - `nothing`: The function mutates `tvs` in-place.

# Related

  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_add_cvar_view!(tvs::AbstractVector, ::LinearConditionalValueatRiskView,
                           X::MatNum, idx::VecInt, coef::VecNum, op::Symbol, rhs::Number,
                           alpha::Number, w::VecNum, pv::Number, eqn::AbstractString)
    @argcheck(isone(length(idx)),
              ArgumentError("View `$(eqn)` names $(length(idx)) assets. `LinearConditionalValueatRiskView` writes the CVaR of a single asset; a relative view needs `IntegerConditionalValueatRiskView`."))
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `LinearConditionalValueatRiskView` bounds the CVaR from below only; use `IntegerConditionalValueatRiskView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior CVaR $(pv). `LinearConditionalValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `IntegerConditionalValueatRiskView`."))
    push!(tvs, LinearConditionalValueatRiskViewConstraint(-X[:, idx[1]], alpha, rhs))
    return nothing
end
function ep_add_cvar_view!(tvs::AbstractVector, alg::IntegerConditionalValueatRiskView,
                           X::MatNum, idx::VecInt, coef::VecNum, op::Symbol, rhs::Number,
                           alpha::Number, w::VecNum, pv::Number, eqn::AbstractString)
    T = size(X, 1)
    N = length(idx)
    ord = Vector{Vector{Int}}(undef, N)
    x = Vector{Vector{eltype(X)}}(undef, N)
    for (k, j) in pairs(idx)
        xj = -X[:, j]
        o = sortperm(xj)
        sb = ep_sbar(alg.sbar, T, alpha, w, o)
        ord[k] = o[(T - sb + 1):T]
        x[k] = xj[ord[k]]
    end
    push!(tvs, IntegerConditionalValueatRiskViewConstraint(ord, x, coef, alpha, op, rhs))
    return nothing
end
"""
    ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector,
                      alg::AbstractEntropicValueatRiskViewFormulation, x::VecNum, alpha::Number,
                      op::Symbol, rhs::Number, w::VecNum, zstar::Number, pv::Number,
                      eqn::AbstractString; args::Tuple = (), kwargs::NamedTuple = (;),
                      zlo::Option{<:Number} = nothing)

Lower one entropic value-at-risk view into the constraints its formulation needs.

[`ConicEntropicValueatRiskView`](@ref) produces one tail view constraint. [`GridEntropicValueatRiskView`](@ref) produces linear rows on the posterior probabilities for the lower-bound half of the view, and a tail view constraint for the upper-bound half, so an equality view produces both.

# Algorithm

 1. [`ConicEntropicValueatRiskView`](@ref) checks the two preconditions below, then appends one [`ConicEntropicValueatRiskViewConstraint`](@ref) carrying `x`, `alpha` and `rhs`.
 2. [`GridEntropicValueatRiskView`](@ref) normalises `w` to sum to one, giving `wi`.
 3. It builds the grid `z` of dual variables with [`ep_evar_grid`](@ref).
 4. It keeps the points whose row is finite, giving `keep`, and raises where `keep` is empty.
 5. For the lower-bound half of the view, it builds the row of each kept point with [`ep_evar_grid_row`](@ref), and adds it to `epc` under `:ineq` with [`add_ep_constraint!`](@ref), negated so the row reads as the `<=` sense that key states.
 6. For the upper-bound half of the view, it appends one [`GridEntropicValueatRiskViewConstraint`](@ref) carrying `x`, the kept grid, `alpha`, `rhs` and the big-M constant `M`.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `alg`: Formulation of the view. A grid formulation is also where the number of steps and the tolerance of the anchor live.
  - `x`: Loss series of the asset the view names.
  - `alpha`: Significance level of the view.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `w`: Prior probability weights. They start the search for the dual variable the grid is centred on.
  - `zstar`: Dual variable that attains the prior EVaR of the asset.
  - `pv`: Prior EVaR of the asset.
  - `eqn`: Equation of the view, used in the error messages.
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `zlo`: Lower end of the bracket, forwarded to [`ep_evar`](@ref).

# Validation

  - [`ConicEntropicValueatRiskView`](@ref) needs an operator other than `<=`, and, for an equality, a target at or above the prior EVaR.
  - [`GridEntropicValueatRiskView`](@ref) needs at least one grid point whose row is finite. [`ep_evar_grid_row`](@ref) overflows at a dual variable near zero. The grid sits there when `pct` approaches one, and wholly there when `alpha * T` falls below one, because [`ep_evar`](@ref)'s minimiser is then at the end of its bracket. The points it overflows at are dropped, and a grid that keeps none of them raises.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`ep_evar_grid`](@ref)
  - [`ep_evar_grid_row`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector,
                           ::ConicEntropicValueatRiskView, x::VecNum, alpha::Number,
                           op::Symbol, rhs::Number, w::VecNum, zstar::Number, pv::Number,
                           eqn::AbstractString; args::Tuple = (), kwargs::NamedTuple = (;),
                           zlo::Option{<:Number} = nothing)
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `ConicEntropicValueatRiskView` bounds the EVaR from below only; use `GridEntropicValueatRiskView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior EVaR $(pv). `ConicEntropicValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `GridEntropicValueatRiskView`."))
    push!(tvs, ConicEntropicValueatRiskViewConstraint(x, alpha, rhs))
    return nothing
end
function ep_add_evar_view!(epc::AbstractDict, tvs::AbstractVector,
                           alg::GridEntropicValueatRiskView, x::VecNum, alpha::Number,
                           op::Symbol, rhs::Number, w::VecNum, zstar::Number, pv::Number,
                           eqn::AbstractString; args::Tuple = (), kwargs::NamedTuple = (;),
                           zlo::Option{<:Number} = nothing)
    (; pct, K, M, iters, tol, tilt_iters) = alg
    wi = w ./ sum(w)
    z = ep_evar_grid(x, wi, alpha, op, rhs, zstar, pct, K; iters = iters, tol = tol,
                     tilt_iters = tilt_iters, args = args, kwargs = kwargs, zlo = zlo)
    function row(zk)
        c, isc = ep_evar_grid_row(x, rhs, zk)
        return c, alpha * isc
    end
    # `exp((x - rhs) / z)` overflows at a dual variable near zero. The grid sits there when
    # `pct` approaches one, and wholly there when `alpha * T` falls below one, because the
    # minimiser is then at the end of its bracket.
    z = ep_add_grid_tail_view!(epc, z, op, row,
                               () -> "View `$(eqn)` builds no finite grid point. The row of every dual variable the grid spans overflows, which happens when `alpha` ($(alpha)) leaves fewer than one observation in the tail, and when `pct` ($(pct)) approaches one. Raise `alpha`, or narrow `pct`.")
    if op == :leq || op == :eq
        push!(tvs, GridEntropicValueatRiskViewConstraint(x, z, alpha, rhs, M))
    end
    return nothing
end
"""
    ep_add_grid_tail_view!(epc::AbstractDict, grid::AbstractVector, op::Symbol, row, msg)

Keep the finite points of a tail view grid, and add the rows of its lower-bound half.

`ep_add_grid_tail_view!` is the scaffold shared by [`GridEntropicValueatRiskView`](@ref) and [`GridRelativisticValueatRiskView`](@ref). Both build a grid of points, drop the points whose row is not finite, and add one linear row per kept point. They differ in what a point is and in how its row is built, and both reach the scaffold through `row`.

A point whose row is not finite is not a grid point, because a non-finite coefficient reaches the solver as `NaN * x[j]`. The caller's `msg` names the setting that put the whole grid there.

# Algorithm

 1. Build the row of every point with `row`, and keep the points whose coefficients and whose right-hand side are all finite.
 2. Raise with `msg` where no point is kept.
 3. Where the view carries a lower-bound half, add the row of each kept point to `epc` under `:ineq` with [`add_ep_constraint!`](@ref), negated so the row reads as the `<=` sense that key states.
 4. Return the kept points, which the caller carries into the tail view constraint of the upper-bound half.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `grid`: Points of the grid.
  - `op`: Comparison operator of the view.
  - `row`: Function taking one point to the coefficients and the right-hand side of its row.
  - `msg`: Function of no arguments giving the message of the error raised where no point is kept. It is called only where the grid keeps no point.

# Validation

  - At least one point of the grid has a finite row. A grid that keeps no point raises an `ArgumentError` carrying `msg()`.

# Returns

  - `grid::AbstractVector`: Points of the grid whose row is finite.

# Related

  - [`ep_add_evar_view!`](@ref)
  - [`ep_add_rlvar_view!`](@ref)
  - [`add_ep_constraint!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_add_grid_tail_view!(epc::AbstractDict, grid::AbstractVector, op::Symbol, row,
                                msg)
    keep = filter(eachindex(grid)) do k
        c, b = row(grid[k])
        return all(isfinite, c) && isfinite(b)
    end
    @argcheck(!isempty(keep), ArgumentError(msg()))
    grid = grid[keep]
    if op == :geq || op == :eq
        for g in grid
            c, b = row(g)
            add_ep_constraint!(epc, reshape(-c, 1, :), [-b], :ineq)
        end
    end
    return grid
end
"""
    ep_add_rlvar_view!(epc::AbstractDict, tvs::AbstractVector,
                       alg::AbstractRelativisticValueatRiskViewFormulation, x::VecNum,
                       alpha::Number, kappa::Number, op::Symbol, rhs::Number, w::VecNum,
                       zstar::Number, pv::Number, eqn::AbstractString; args::Tuple = (),
                       kwargs::NamedTuple = (;), bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)

Lower one relativistic value-at-risk view into the constraints its formulation needs.

[`ConicRelativisticValueatRiskView`](@ref) produces one tail view constraint. [`GridRelativisticValueatRiskView`](@ref) produces linear rows on the posterior probabilities for the lower-bound half of the view, and a tail view constraint for the upper-bound half, so an equality view produces both.

# Algorithm

 1. [`ConicRelativisticValueatRiskView`](@ref) checks the two preconditions below, then appends one [`ConicRelativisticValueatRiskViewConstraint`](@ref) carrying `x`, `alpha`, `kappa` and `rhs`.
 2. [`GridRelativisticValueatRiskView`](@ref) normalises `w` to sum to one, giving `wi`.
 3. It builds the grid `t`, `z` of primal points with [`ep_rlvar_grid`](@ref).
 4. It keeps the points whose row is finite, giving `keep`, and raises where `keep` is empty.
 5. For the lower-bound half of the view, it builds the row of each kept point with [`ep_rlvar_grid_row`](@ref), and adds it to `epc` under `:ineq` with [`add_ep_constraint!`](@ref), negated so the row reads as the `<=` sense that key states.
 6. For the upper-bound half of the view, it appends one [`GridRelativisticValueatRiskViewConstraint`](@ref) carrying `x`, the kept grid, `alpha`, `kappa`, `rhs` and the big-M constant `M`.

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `alg`: Formulation of the view. A grid formulation is also where the number of steps and the tolerance of the anchor live.
  - `x`: Loss series of the asset the view names.
  - `alpha`: Significance level of the view.
  - `kappa`: Deformation parameter of the view.
  - `op`: Comparison operator of the view.
  - `rhs`: Target value of the view.
  - `w`: Prior probability weights. They start the search for the point the grid is centred on, and they pin the shift of each grid point where that search does not converge.
  - `zstar`: Dual variable that attains the prior RLVaR of the asset.
  - `pv`: Prior RLVaR of the asset. With `rhs` it fixes the translation a grid centred on the prior carries.
  - `eqn`: Equation of the view, used in the error messages.
  - $(arg_dict[:optargs])
  - $(arg_dict[:optkwargs])
  - `bracket`: Spans of the searches, forwarded to [`ep_rlvar`](@ref) and [`ep_rlvar_shift`](@ref).

# Validation

  - [`ConicRelativisticValueatRiskView`](@ref) needs an operator other than `<=`, and, for an equality, a target at or above the prior RLVaR.
  - [`GridRelativisticValueatRiskView`](@ref) needs at least one grid point whose row is finite. [`ep_rlvar_tail`](@ref) overflows at a dual variable near zero, which is where the grid sits when `kappa` approaches one; the points it overflows at are dropped, and a grid that keeps none of them raises.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`ep_rlvar_grid`](@ref)
  - [`ep_rlvar_grid_row`](@ref)
  - [`ep_rlvar_tail`](@ref)
  - [`ep_tail_views!`](@ref)
"""
function ep_add_rlvar_view!(epc::AbstractDict, tvs::AbstractVector,
                            ::ConicRelativisticValueatRiskView, x::VecNum, alpha::Number,
                            kappa::Number, op::Symbol, rhs::Number, w::VecNum,
                            zstar::Number, pv::Number, eqn::AbstractString;
                            args::Tuple = (), kwargs::NamedTuple = (;),
                            bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    @argcheck(op != :leq,
              ArgumentError("View `$(eqn)` is an upper bound. `ConicRelativisticValueatRiskView` bounds the RLVaR from below only; use `GridRelativisticValueatRiskView`."))
    @argcheck(op != :eq || rhs >= pv,
              ArgumentError("View `$(eqn)` targets $(rhs), below the prior RLVaR $(pv). `ConicRelativisticValueatRiskView` writes an equality as a lower bound, which is slack at the prior and would leave the view unmet; use `GridRelativisticValueatRiskView`."))
    push!(tvs, ConicRelativisticValueatRiskViewConstraint(x, alpha, kappa, rhs))
    return nothing
end
function ep_add_rlvar_view!(epc::AbstractDict, tvs::AbstractVector,
                            alg::GridRelativisticValueatRiskView, x::VecNum, alpha::Number,
                            kappa::Number, op::Symbol, rhs::Number, w::VecNum,
                            zstar::Number, pv::Number, eqn::AbstractString;
                            args::Tuple = (), kwargs::NamedTuple = (;),
                            bracket::Option{<:RelativisticValueatRiskViewBracket} = nothing)
    (; pct, K, M, iters, tol, tilt_iters) = alg
    wi = w ./ sum(w)
    t, z = ep_rlvar_grid(x, wi, alpha, kappa, op, rhs, zstar, pv, pct, K; iters = iters,
                         tol = tol, tilt_iters = tilt_iters, args = args, kwargs = kwargs,
                         bracket = bracket)
    function row(g)
        return ep_rlvar_grid_row(x, rhs, g[1], g[2], alpha, kappa)
    end
    # `ep_rlvar_tail` overflows at a dual variable near zero, which is where the grid sits
    # when `kappa` approaches one.
    grid = ep_add_grid_tail_view!(epc, collect(zip(t, z)), op, row,
                                  () -> "View `$(eqn)` builds no finite grid point at `kappa = $(kappa)`. The tail function overflows at every dual variable the grid spans; state the view at a smaller `kappa`.")
    if op == :leq || op == :eq
        push!(tvs,
              GridRelativisticValueatRiskViewConstraint(x, first.(grid), last.(grid), alpha,
                                                        kappa, rhs, M))
    end
    return nothing
end
"""
    ep_tail_view_prior_args(tail_views::ConditionalValueatRiskView, w::VecNum)
    ep_tail_view_prior_args(tail_views::EntropicValueatRiskView, w::VecNum)
    ep_tail_view_prior_args(tail_views::RelativisticValueatRiskView, w::VecNum)

Give the trailing arguments a `prior(...)` reference of this tail view resolves under.

`ep_tail_view_prior_args` is one of the two kernels [`ep_tail_views!`](@ref) takes a measure from. It names the statistic [`get_pr_value`](@ref) reads, and it carries the level, the prior probabilities, and whatever settings the search for that statistic needs. [`replace_prior_views`](@ref) forwards the tuple unchanged, so a measure is added by adding a method here rather than by copying the lowering.

# Arguments

  - `tail_views`: Tail view group whose settings the tuple carries.
  - `w`: Prior probability weights.

# Returns

  - `args::Tuple`: Trailing arguments of [`replace_prior_views`](@ref), starting with the tag of the statistic.

# Related

  - [`ep_tail_views!`](@ref)
  - [`replace_prior_views`](@ref)
  - [`get_pr_value`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_tail_view_prior_args(tail_views::ConditionalValueatRiskView, w::VecNum)
    return (:cvar, tail_views.alpha, StatsBase.pweights(w))
end
function ep_tail_view_prior_args(tail_views::EntropicValueatRiskView, w::VecNum)
    return (:evar, tail_views.alpha, StatsBase.pweights(w), tail_views.args,
            tail_views.kwargs, tail_views.zlo)
end
function ep_tail_view_prior_args(tail_views::RelativisticValueatRiskView, w::VecNum)
    return (:rlvar, tail_views.alpha, tail_views.kappa, StatsBase.pweights(w),
            tail_views.args, tail_views.kwargs, tail_views.bracket)
end
"""
    ep_assert_absolute_view(idx::VecInt, eqn::AbstractString, name::AbstractString)

Reject a tail view that names more than one asset.

[EPTail](@cite) and [EPRLVaR](@cite) give a formulation for a view on the risk measure of one asset alone, so an entropic or relativistic value-at-risk view over several assets has nothing to lower into. A group name expands to its members before this check, so a group of one member names one asset and passes.

# Arguments

  - `idx`: Indices of the assets the view names.
  - `eqn`: Equation of the view, used in the error message.
  - `name`: Name of the risk measure, used in the error message.

# Validation

  - The view names exactly one asset. A view over more than one asset raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`ep_view_terms`](@ref)
  - [`ep_add_tail_view!`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
"""
function ep_assert_absolute_view(idx::VecInt, eqn::AbstractString, name::AbstractString)
    @argcheck(isone(length(idx)),
              ArgumentError("View `$(eqn)` names $(length(idx)) assets. An $(name) view names one asset: there is no formulation for a relative $(name) view."))
    return nothing
end
"""
    ep_normalise_absolute_view(terms::NamedTuple, X::MatNum, eqn::AbstractString,
                               name::AbstractString)

Normalise a single-asset tail view and read the loss series it is stated on.

The three steps below are shared by every tail view that names one asset, which is every entropic and relativistic value-at-risk view, and a conditional value-at-risk view that names one asset.

# Algorithm

 1. Divide the view by the coefficient it gives the asset with [`ep_normalise_view_term`](@ref), which flips the operator where that coefficient is negative.
 2. Read the loss series of the asset, `x`, as the negated returns column.
 3. Reject a target no reweighting of the sample reaches with [`ep_assert_reachable_view`](@ref).

# Arguments

  - `terms`: Resolved terms of the view, as [`ep_view_terms`](@ref) returns them.
  - `X`: Matrix of asset returns.
  - `eqn`: Equation of the view, used in the error messages.
  - `name`: Name of the risk measure, used in the error messages.

# Returns

  - `x::VecNum`: Loss series of the asset the view names.
  - `op::Symbol`: Operator of the normalised view.
  - `rhs::Number`: Target of the normalised view.

# Related

  - [`ep_view_terms`](@ref)
  - [`ep_normalise_view_term`](@ref)
  - [`ep_assert_reachable_view`](@ref)
  - [`ep_add_tail_view!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_normalise_absolute_view(terms::NamedTuple, X::MatNum, eqn::AbstractString,
                                    name::AbstractString)
    (; idx, coef, op, rhs) = terms
    op, rhs = ep_normalise_view_term(coef[1], op, rhs)
    x = -X[:, idx[1]]
    ep_assert_reachable_view(op, rhs, x, eqn, name)
    return x, op, rhs
end
"""
    ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                      tail_views::ConditionalValueatRiskView, alg, X::MatNum,
                      terms::NamedTuple, eqn::AbstractString, w::VecNum)
    ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                      tail_views::EntropicValueatRiskView, alg, X::MatNum,
                      terms::NamedTuple, eqn::AbstractString, w::VecNum)
    ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                      tail_views::RelativisticValueatRiskView, alg, X::MatNum,
                      terms::NamedTuple, eqn::AbstractString, w::VecNum)

Lower one resolved tail view into the constraints its measure and its formulation need.

`ep_add_tail_view!` is the second of the two kernels [`ep_tail_views!`](@ref) takes a measure from. [`ep_tail_views!`](@ref) parses the group, expands its groups, and resolves its prior references and its terms; this verb carries everything past that point, which is everything the measures do not share. A measure is added by adding a method here and to [`ep_tail_view_prior_args`](@ref).

# Algorithm

 1. A [`ConditionalValueatRiskView`](@ref) admits a relative view. Where the view names one asset, normalise it with [`ep_normalise_absolute_view`](@ref) and set its coefficient to one. Read `pv`, the prior value of the left hand side, as the coefficient-weighted sum of the per-asset CVaRs under `w`. Pick the formulation with [`ep_cvar_formulation`](@ref), and append with [`ep_add_cvar_view!`](@ref).
 2. An [`EntropicValueatRiskView`](@ref) names one asset. Check that with [`ep_assert_absolute_view`](@ref), normalise it with [`ep_normalise_absolute_view`](@ref), and read the prior EVaR and the dual variable that attains it with [`ep_evar`](@ref). Pick the formulation with [`ep_evar_formulation`](@ref), and append with [`ep_add_evar_view!`](@ref).
 3. A [`RelativisticValueatRiskView`](@ref) names one asset. Check and normalise it as the entropic view is, and read the prior RLVaR and the primal pair that attains it with [`ep_rlvar`](@ref). Pick the formulation with [`ep_rlvar_formulation`](@ref), and append with [`ep_add_rlvar_view!`](@ref).

# Arguments

  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `tail_views`: Tail view group the view belongs to. It carries the level, the formulation setting, and the settings of the searches.
  - `alg`: Formulation of this view, as [`ep_view_formulations`](@ref) spread it.
  - `X`: Matrix of asset returns.
  - `terms`: Resolved terms of the view, as [`ep_view_terms`](@ref) returns them.
  - `eqn`: Equation of the view, used in the error messages.
  - `w`: Prior probability weights.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ep_tail_views!`](@ref)
  - [`ep_tail_view_prior_args`](@ref)
  - [`ep_add_cvar_view!`](@ref)
  - [`ep_add_evar_view!`](@ref)
  - [`ep_add_rlvar_view!`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
"""
function ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                           tail_views::ConditionalValueatRiskView, alg, X::MatNum,
                           terms::NamedTuple, eqn::AbstractString, w::VecNum)
    (; idx, coef, op, rhs) = terms
    alpha = tail_views.alpha
    single = isone(length(idx))
    if single
        _, op, rhs = ep_normalise_absolute_view(terms, X, eqn, "CVaR")
        coef = [one(eltype(coef))]
    end
    rm = ConditionalValueatRisk(; alpha = alpha, w = StatsBase.pweights(w))
    pv = sum(ci * rm(view(X, :, j)) for (j, ci) in zip(idx, coef))
    alg = ep_cvar_formulation(alg, single, op, rhs, pv)
    ep_add_cvar_view!(tvs, alg, X, idx, coef, op, rhs, alpha, w, pv, eqn)
    return nothing
end
function ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                           tail_views::EntropicValueatRiskView, alg, X::MatNum,
                           terms::NamedTuple, eqn::AbstractString, w::VecNum)
    (; alpha, args, kwargs, zlo) = tail_views
    ep_assert_absolute_view(terms.idx, eqn, "EVaR")
    x, op, rhs = ep_normalise_absolute_view(terms, X, eqn, "EVaR")
    evr = ep_evar(x, w, alpha; args = args, kwargs = kwargs, zlo = zlo)
    pv, zstar = evr.evar, evr.z
    alg = ep_evar_formulation(alg, op, rhs, pv)
    ep_add_evar_view!(epc, tvs, alg, x, alpha, op, rhs, w, zstar, pv, eqn; args = args,
                      kwargs = kwargs, zlo = zlo)
    return nothing
end
function ep_add_tail_view!(epc::AbstractDict, tvs::AbstractVector,
                           tail_views::RelativisticValueatRiskView, alg, X::MatNum,
                           terms::NamedTuple, eqn::AbstractString, w::VecNum)
    (; alpha, kappa, args, kwargs, bracket) = tail_views
    ep_assert_absolute_view(terms.idx, eqn, "RLVaR")
    x, op, rhs = ep_normalise_absolute_view(terms, X, eqn, "RLVaR")
    rlv = ep_rlvar(x, w, alpha, kappa; args = args, kwargs = kwargs, bracket = bracket)
    alg = ep_rlvar_formulation(alg, op, rhs, rlv.rlvar)
    ep_add_rlvar_view!(epc, tvs, alg, x, alpha, kappa, op, rhs, w, rlv.z, rlv.rlvar, eqn;
                       args = args, kwargs = kwargs, bracket = bracket)
    return nothing
end
"""
    ep_tail_views!(tail_views::Nothing, args...; kwargs...)

No-op pass-through for tail view constraints when none are specified.

# Arguments

  - `tail_views::Nothing`: Indicates that no tail view constraints are specified.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `nothing`.

# Related

  - [`ep_tail_views!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_tail_views!(tail_views::Nothing, args...; kwargs...)
    return nothing
end
"""
    ep_tail_views!(tail_views::AbstractVector{<:AbstractEntropyPoolingTailViewEstimator},
                   args...; kwargs...)

Lower each group of tail views under its own settings.

Every [`AbstractEntropyPoolingTailViewEstimator`](@ref) in the vector is lowered in turn, so the groups accumulate into the same constraint set and one entropy pooling solve answers all of them.

# Algorithm

 1. Lower each group of `tail_views` in turn, forwarding `args...` and `kwargs...` to each call.
 2. Return `nothing`. Each call has already written its constraints into `epc` and `tvs`.

# Arguments

  - `tail_views`: Groups of tail views.
  - `args...`: Additional positional arguments forwarded to [`ep_tail_views!`](@ref).
  - `kwargs...`: Additional keyword arguments forwarded to [`ep_tail_views!`](@ref).

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ep_tail_views!`](@ref)
  - [`EntropyPoolingPrior`](@ref)
"""
function ep_tail_views!(tail_views::AbstractVector{<:AbstractEntropyPoolingTailViewEstimator},
                        args...; kwargs...)
    for tail_view in tail_views
        ep_tail_views!(tail_view, args...; kwargs...)
    end
    return nothing
end
"""
    ep_tail_views!(tail_views::AbstractEntropyPoolingTailViewEstimator, epc::AbstractDict,
                   tvs::AbstractVector, pr::AbstractPriorResult, sets::UniverseSets,
                   w::VecNum; strict::Bool = false)

Parse a group of tail views and lower them into entropy pooling constraints.

`ep_tail_views!` is the one lowering of the tail view family. It parses the view equations of a [`LinearConstraintEstimator`](@ref), replaces prior references with their values, resolves the asset names against the universe, picks a formulation for each view, and appends the constraints that formulation needs. Unlike the recursive algorithm of [`MeucciEntropyPoolingPrior`](@ref), nothing is solved here: the views become part of the one entropy pooling problem [`entropy_pooling`](@ref) solves.

It accepts `==`, `>=` and `<=`. A group name expands to its members, each carrying the coefficient the group carried, so a view on a group constrains the *sum* of the members' risk measures and not their average, and a group of more than one member is a relative view.

The two kernels below carry everything that differs between the conditional, the entropic and the relativistic measure, so a fourth measure supplies two methods rather than a fourth copy of this verb:

  - [`ep_tail_view_prior_args`](@ref) names the statistic a `prior(...)` reference resolves to, and the settings its search takes.
  - [`ep_add_tail_view!`](@ref) checks the shape of the view, reads its prior value, picks its formulation, and appends its constraints.

# Algorithm

 1. Parse the view equations of `tail_views.views.val`, giving one [`ParsingResult`](@ref) per view.
 2. Replace every group name by the assets it spans, giving one term per member.
 3. Replace every `prior(...)` reference by the prior value of the measure, through [`replace_prior_views`](@ref) under the tuple [`ep_tail_view_prior_args`](@ref) gives.
 4. Spread the formulation setting over the views with [`ep_view_formulations`](@ref), giving `algs`.
 5. For each view in turn, resolve its terms with [`ep_view_terms`](@ref), and drop the view where no name of it is placed in the universe.
 6. Lower the view with [`ep_add_tail_view!`](@ref).

# Arguments

  - `tail_views`: Tail view constraints.
  - `epc`: Dictionary of entropy pooling constraints, mapping keys to `(lhs, rhs)` pairs.
  - `tvs`: Tail view constraints, appended to.
  - `pr`: Prior result containing asset return information.
  - `sets`: Asset set mapping asset names to indices.
  - `w`: Prior probability weights.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

# Returns

  - `nothing`: The function mutates `epc` and `tvs` in-place.

# Related

  - [`ep_tail_view_prior_args`](@ref)
  - [`ep_add_tail_view!`](@ref)
  - [`ConditionalValueatRiskView`](@ref)
  - [`EntropicValueatRiskView`](@ref)
  - [`RelativisticValueatRiskView`](@ref)
  - [`EntropyPoolingPrior`](@ref)

# References

  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
"""
function ep_tail_views!(tail_views::AbstractEntropyPoolingTailViewEstimator,
                        epc::AbstractDict, tvs::AbstractVector, pr::AbstractPriorResult,
                        sets::UniverseSets, w::VecNum; strict::Bool = false)
    X = pr.X
    views = parse_equation(tail_views.views.val; ops1 = ("==", ">=", "<="),
                           ops2 = (:call, :(==), :(>=), :(<=)), datatype = eltype(X))
    views = replace_group_by_assets(views, sets, false, true, false)
    views = replace_prior_views(views, pr, sets, ep_tail_view_prior_args(tail_views, w)...;
                                strict = strict)
    if !isa(views, AbstractVector)
        views = [views]
    end
    algs = ep_view_formulations(tail_views.alg, length(views), :alg)
    for (res, algi) in zip(views, algs)
        terms = ep_view_terms(res, sets, X; strict = strict)
        if isnothing(terms)
            continue
        end
        ep_add_tail_view!(epc, tvs, tail_views, algi, X, terms, res.eqn, w)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Reweights the observations of a prior so that its moments and its tails meet a set of views.

`EntropyPoolingPrior` is a low order prior estimator that computes the mean and covariance of asset returns using entropy pooling. It supports views on the mean, the variance, the covariance, the correlation, the skewness and the kurtosis, views on the value at risk, the conditional and entropic value at risk views of [EPTail](@cite), and the relativistic value at risk views of [EPRLVaR](@cite).

The tail views are the difference with [`MeucciEntropyPoolingPrior`](@ref). There, a CVaR view is a target the recursive algorithm of Meucci et al. hunts by re-solving the whole entropy pooling problem for each candidate value at risk level, which supports equalities alone. Here each tail view is written as constraints of the single entropy pooling problem, so one solve answers every view, and the operators `==`, `>=` and `<=` are all available, along with relative CVaR views and views on the entropic and the relativistic value at risk.

!!! warning

    An infeasible view set is not raised on by the [`OptimEntropyPooling`](@ref) route. The dual of an infeasible set is unbounded, so the minimiser runs away, the posterior collapses onto one observation, and `Optim` reports the solve as converged. A grossly infeasible view overflows instead, and the non-finite weights reach the moment estimators as an `ArgumentError` naming Infs or NaNs. Read the result rather than the flag: `ens` falls to a handful out of the number of observations, one weight sits near one, `kld` is large, and the posterior statistic the view named is far from its target. [`entropy_pooling`](@ref) states the mechanism. The [`JuMPEntropyPooling`](@ref) route does not share it: the solver reports an infeasible model itself.

# Algorithm

The constructor derives the prior probabilities, and validates everything else.

 1. When `w` is `nothing`, derive nothing. [`prior`](@ref) builds the uniform weights `1/T` at solve time, one per observation.
 2. When `w` is not `nothing`, normalise it to sum to one, giving the prior probabilities the pooling starts from. A mutable `w.values` is normalised in place with `LinearAlgebra.normalize!`, and an immutable one is replaced by a new `StatsBase.pweights` over the normalised values.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropyPoolingPrior(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
        mu_views::Option{<:LinearConstraintEstimator} = nothing,
        var_views::Option{<:VV_VecVV} = nothing,
        cvar_views::Option{<:CVV_VecCVV} = nothing,
        evar_views::Option{<:EVV_VecEVV} = nothing,
        rlvar_views::Option{<:RVV_VecRVV} = nothing,
        sigma_views::Option{<:LinearConstraintEstimator} = nothing,
        sk_views::Option{<:LinearConstraintEstimator} = nothing,
        kt_views::Option{<:LinearConstraintEstimator} = nothing,
        cov_views::Option{<:LinearConstraintEstimator} = nothing,
        rho_views::Option{<:LinearConstraintEstimator} = nothing,
        sets::Option{<:UniverseSets} = nothing,
        opt::NonCVaREP = OptimEntropyPooling(),
        w::Option{<:StatsBase.ProbabilityWeights} = nothing,
        alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling()
    ) -> EntropyPoolingPrior

Keywords correspond to the struct's fields.

## Validation

  - If any view constraint is not `nothing`, `sets` must not be `nothing`.
  - If `cvar_views` is not `nothing`, `opt` must be a [`JuMPEntropyPooling`](@ref).
  - If a view field is a vector, it must not be empty.
  - If `w` is not `nothing`, it must be non-empty and match the number of observations.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pe`: Recursively updated via [`factory`](@ref).
  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pe`: Recursively viewed via [`port_opt_view`](@ref).
  - `sets`: Sliced to the selected indices via [`port_opt_view`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `pe`: Recursively indexed via [`obs_weights_view`](@ref).
  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# View comparison operators

The comparison operators accepted in each view's constraint strings depend on the moment being constrained. An unsupported operator raises a `ParseError` listing the operators allowed for that view.

  - `mu_views`, `sigma_views`, `sk_views`, `kt_views`, `cov_views`, `rho_views` accept `==`, `>=` and `<=`.
  - `var_views` (Value at Risk) accepts only `==` and `>=`.
  - `cvar_views`, `evar_views` and `rlvar_views` accept `==`, `>=` and `<=`.

# Tail views

A tail view needs auxiliary variables, so it is expressed in the JuMP model rather than reduced to rows that multiply the posterior probabilities. Two consequences follow.

  - `opt` must be a [`JuMPEntropyPooling`](@ref) whenever `cvar_views` is set, and whenever an `evar_views` or `rlvar_views` entry is anything other than a lower bound under [`GridEntropicValueatRiskView`](@ref) or [`GridRelativisticValueatRiskView`](@ref), which are the tail formulations that are linear in the posterior probabilities alone.
  - A view that needs a binary variable — every [`IntegerConditionalValueatRiskView`](@ref), and an upper bound or equality under [`GridEntropicValueatRiskView`](@ref) or [`GridRelativisticValueatRiskView`](@ref) — needs a solver that handles mixed-integer exponential cone programs.
  - [`ConicRelativisticValueatRiskView`](@ref) writes power cones, so its solver must handle the power cone alongside the exponential cone the objective needs.

The `alg` field of a view group picks the formulation. A single formulation applies to every view in that group, a vector supplies one per view, and `nothing` lets each view take the cheapest formulation that expresses it exactly: [`LinearConditionalValueatRiskView`](@ref), [`ConicEntropicValueatRiskView`](@ref) and [`ConicRelativisticValueatRiskView`](@ref) where they apply, [`IntegerConditionalValueatRiskView`](@ref), [`GridEntropicValueatRiskView`](@ref) and [`GridRelativisticValueatRiskView`](@ref) otherwise.

# Tail views at several significance levels

A significance level is part of the statistic, not a detail of the solve: the conditional value at risk at 1% and at 10% are different numbers on the same series. So the level lives on the view rather than on the estimator. `var_views`, `cvar_views`, `evar_views` and `rlvar_views` each take one [`ValueatRiskView`](@ref), [`ConditionalValueatRiskView`](@ref), [`EntropicValueatRiskView`](@ref) or [`RelativisticValueatRiskView`](@ref), or a vector of them, and each group carries the `alpha` its equations are read under. A [`RelativisticValueatRiskView`](@ref) carries a `kappa` as well, on the same reasoning: the deformation parameter is part of the statistic. A `prior(...)` reference inside a group is replaced by the prior value at *that* group's level.

A tail view group also carries `alg`, the formulation. For [`EntropicValueatRiskView`](@ref) that is where the grid of dual variables and the big-M constant live, so a [`GridEntropicValueatRiskView`](@ref) there gives one group its own `pct`, `K` and `M`. [`ValueatRiskView`](@ref) has no such field: a value at risk view is linear in the posterior probabilities, so there is no formulation to choose.

# Examples

```jldoctest
julia> EntropyPoolingPrior(;
                           sets = UniverseSets(; xkey = \"nx\",
                                               dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"])),
                           mu_views = LinearConstraintEstimator(;
                                                                val = [\"A == 0.03\",
                                                                       \"B + C == 0.04\"]))
EntropyPoolingPrior
           pe ┼ EmpiricalPrior
              │        ce ┼ PortfolioOptimisersCovariance
              │           │   ce ┼ Covariance
              │           │      │    me ┼ SimpleExpectedReturns
              │           │      │       │   w ┴ nothing
              │           │      │    ce ┼ GeneralCovariance
              │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
              │           │      │       │    w ┴ nothing
              │           │      │   alg ┼ FullMoment()
              │           │      │     w ┴ nothing
              │           │   mp ┼ MatrixProcessing
              │           │      │     pdm ┼ Posdef
              │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
              │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
              │           │      │      dn ┼ nothing
              │           │      │      dt ┼ nothing
              │           │      │     alg ┼ nothing
              │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
              │        me ┼ SimpleExpectedReturns
              │           │   w ┴ nothing
              │   horizon ┴ nothing
     mu_views ┼ LinearConstraintEstimator
              │   val ┼ Vector{String}: ["A == 0.03", "B + C == 0.04"]
              │   key ┴ nothing
    var_views ┼ nothing
   cvar_views ┼ nothing
   evar_views ┼ nothing
  rlvar_views ┼ nothing
  sigma_views ┼ nothing
     sk_views ┼ nothing
     kt_views ┼ nothing
    cov_views ┼ nothing
    rho_views ┼ nothing
         sets ┼ UniverseSets
              │    xkey ┼ String: "nx"
              │   uxkey ┼ String: "ux"
              │    fkey ┼ String: "nf"
              │   ufkey ┼ String: "uf"
              │    zkey ┼ String: "nz"
              │    dict ┴ Dict{String, Vector{String}}: Dict("nx" => ["A", "B", "C"])
          opt ┼ OptimEntropyPooling
              │     args ┼ Tuple{}: ()
              │   kwargs ┼ @NamedTuple{}: NamedTuple()
              │      sc1 ┼ Int64: 1
              │      sc2 ┼ Float64: 1000.0
              │      alg ┼ ExpEntropyPooling()
              │      err ┴ nothing
            w ┼ nothing
          alg ┴ H1_EntropyPooling()
```

# Related

  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`LinearConditionalValueatRiskView`](@ref)
  - [`IntegerConditionalValueatRiskView`](@ref)
  - [`ConicEntropicValueatRiskView`](@ref)
  - [`GridEntropicValueatRiskView`](@ref)
  - [`ConicRelativisticValueatRiskView`](@ref)
  - [`GridRelativisticValueatRiskView`](@ref)
  - [`JuMPEntropyPooling`](@ref)
  - [`OptimEntropyPooling`](@ref)
  - [`AbstractEntropyPoolingAlgorithm`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`obs_weights_view`](@ref)

# References

  - $(ref_dict[:meucci2008])
  - $(ref_dict[:vorobets2021])
  - $(ref_dict[:EPTail])
  - $(ref_dict[:EPRLVaR])
"""
@propagatable @concrete struct EntropyPoolingPrior <: AbstractLowOrderPriorEstimator_AF
    """
    $(field_dict[:pe])
    """
    @fprop @vprop pe
    """
    $(field_dict[:mu_views])
    """
    mu_views
    """
    $(field_dict[:var_views])
    """
    var_views
    """
    $(field_dict[:cvar_views])
    """
    cvar_views
    """
    $(field_dict[:evar_views])
    """
    evar_views
    """
    $(field_dict[:rlvar_views])
    """
    rlvar_views
    """
    $(field_dict[:sigma_views])
    """
    sigma_views
    """
    $(field_dict[:sk_views])
    """
    sk_views
    """
    $(field_dict[:kt_views])
    """
    kt_views
    """
    $(field_dict[:cov_views])
    """
    cov_views
    """
    $(field_dict[:rho_views])
    """
    rho_views
    """
    $(field_dict[:sets])
    """
    @vprop sets
    """
    $(field_dict[:opt_ep])
    """
    opt
    """
    $(field_dict[:ep_w])
    """
    @wprop w
    """
    $(field_dict[:epalg])
    """
    alg
    function EntropyPoolingPrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                                 mu_views::Option{<:LinearConstraintEstimator},
                                 var_views::Option{<:VV_VecVV},
                                 cvar_views::Option{<:CVV_VecCVV},
                                 evar_views::Option{<:EVV_VecEVV},
                                 rlvar_views::Option{<:RVV_VecRVV},
                                 sigma_views::Option{<:LinearConstraintEstimator},
                                 sk_views::Option{<:LinearConstraintEstimator},
                                 kt_views::Option{<:LinearConstraintEstimator},
                                 cov_views::Option{<:LinearConstraintEstimator},
                                 rho_views::Option{<:LinearConstraintEstimator},
                                 sets::Option{<:UniverseSets}, opt::NonCVaREP,
                                 w::Option{<:StatsBase.ProbabilityWeights},
                                 alg::AbstractEntropyPoolingAlgorithm)
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
            if ismutable(w.values)
                LinearAlgebra.normalize!(w, 1)
            else
                w = StatsBase.pweights(LinearAlgebra.normalize(w, 1))
            end
        end
        if !isnothing(mu_views) ||
           !isnothing(var_views) ||
           !isnothing(cvar_views) ||
           !isnothing(evar_views) ||
           !isnothing(rlvar_views) ||
           !isnothing(sigma_views) ||
           !isnothing(sk_views) ||
           !isnothing(kt_views) ||
           !isnothing(cov_views) ||
           !isnothing(rho_views)
            @argcheck(!isnothing(sets), IsNothingError("sets cannot be nothing"))
        end
        if !isnothing(cvar_views)
            @argcheck(isa(opt, JuMPEntropyPooling),
                      ArgumentError("A CVaR view needs auxiliary variables, which the dual formulation `OptimEntropyPooling` solves has no room for. Use `JuMPEntropyPooling` in `opt`."))
        end
        if isa(var_views, AbstractVector)
            @argcheck(!isempty(var_views), IsEmptyError("var_views cannot be empty"))
        end
        if isa(cvar_views, AbstractVector)
            @argcheck(!isempty(cvar_views), IsEmptyError("cvar_views cannot be empty"))
        end
        if isa(evar_views, AbstractVector)
            @argcheck(!isempty(evar_views), IsEmptyError("evar_views cannot be empty"))
        end
        if isa(rlvar_views, AbstractVector)
            @argcheck(!isempty(rlvar_views), IsEmptyError("rlvar_views cannot be empty"))
        end
        return new{typeof(pe), typeof(mu_views), typeof(var_views), typeof(cvar_views),
                   typeof(evar_views), typeof(rlvar_views), typeof(sigma_views),
                   typeof(sk_views), typeof(kt_views), typeof(cov_views), typeof(rho_views),
                   typeof(sets), typeof(opt), typeof(w), typeof(alg)}(pe, mu_views,
                                                                      var_views, cvar_views,
                                                                      evar_views,
                                                                      rlvar_views,
                                                                      sigma_views, sk_views,
                                                                      kt_views, cov_views,
                                                                      rho_views, sets, opt,
                                                                      w, alg)
    end
end
function EntropyPoolingPrior(; pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                             mu_views::Option{<:LinearConstraintEstimator} = nothing,
                             var_views::Option{<:VV_VecVV} = nothing,
                             cvar_views::Option{<:CVV_VecCVV} = nothing,
                             evar_views::Option{<:EVV_VecEVV} = nothing,
                             rlvar_views::Option{<:RVV_VecRVV} = nothing,
                             sigma_views::Option{<:LinearConstraintEstimator} = nothing,
                             sk_views::Option{<:LinearConstraintEstimator} = nothing,
                             kt_views::Option{<:LinearConstraintEstimator} = nothing,
                             cov_views::Option{<:LinearConstraintEstimator} = nothing,
                             rho_views::Option{<:LinearConstraintEstimator} = nothing,
                             sets::Option{<:UniverseSets} = nothing,
                             opt::NonCVaREP = OptimEntropyPooling(),
                             w::Option{<:StatsBase.ProbabilityWeights} = nothing,
                             alg::AbstractEntropyPoolingAlgorithm = H1_EntropyPooling())::EntropyPoolingPrior
    return EntropyPoolingPrior(pe, mu_views, var_views, cvar_views, evar_views, rlvar_views,
                               sigma_views, sk_views, kt_views, cov_views, rho_views, sets,
                               opt, w, alg)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties EntropyPoolingPrior begin
    forward(pe, me, ce)
end
"""
    const VecEP = AbstractVector{<:Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}}

Alias for an abstract vector of entropy pooling prior estimators of either family.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
"""
const VecEP = AbstractVector{<:Union{<:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior}}
"""
    prior(pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
          dims::Int = 1, strict::Bool = false, kwargs...)

Compute the entropy pooling prior of asset returns with tail views.

`prior` orients the data and forwards the estimator's algorithm as a value to [`ep_prior`](@ref), which enforces the views in stages or in one optimisation (ADR 0064).

# Algorithm

 1. Orient `X` and `F` along `dims` with [`dims_oriented`](@ref), so the observations lie in the rows.
 2. Forward `pe.alg` as a value to [`ep_prior`](@ref), and return the [`LowOrderPrior`](@ref) it builds.

# Arguments

  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix.
  - `F`: Optional factor returns matrix.
  - `dims`: Dimension along which the observations lie.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.
  - `kwargs...`: Additional keyword arguments forwarded to the wrapped prior estimator.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `pr::LowOrderPrior`: Prior result carrying the posterior probability weights.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`ep_prior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function prior(pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
    X, F = dims_oriented(dims, X, F)
    return ep_prior(pe.alg, pe, X, F; strict = strict, kwargs...)
end
"""
    ep_prior(alg::StagedEP, pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum};
             strict::Bool = false, kwargs...)

Compute entropy pooling prior moments with tail views, enforcing the views in stages.

`ep_prior` accumulates the views of each stage into one constraint set and solves once per stage, so a stage's views hold alongside every view of the stages before it. The mean of an asset a later stage constrains is pinned to the value the earlier stage produced, so a higher moment view does not silently move it.

# Algorithm

 1. Build the prior probabilities `w0`. They are the uniform weights `1/T` where `pe.w` is `nothing`, and `pe.w` itself otherwise, whose length must match `T`.
 2. Fit the wrapped prior estimator at `w0`, giving `pr`.
 3. Stage one holds the mean, value at risk, conditional, entropic and relativistic value at risk views. Accumulate them into the constraint dictionary `epc` and the tail view vector `tvs`. Where either is non-empty, solve from `w0` with [`entropy_pooling`](@ref), giving `w1`, and refit `pr` at `w1`.
 4. Stage two holds the variance and covariance views, with the mean of every asset they name pinned by [`fix_mu!`](@ref). Solve from `w0` under [`H1_EntropyPooling`](@ref), or from the previous `w1` under [`H2_EntropyPooling`](@ref), and refit `pr` at the new `w1`.
 5. Stage three holds the skewness, kurtosis and correlation views, with the mean and the variance of every asset they name pinned by [`fix_mu!`](@ref) and [`fix_sigma!`](@ref). Solve from the same start step 4 takes, and refit `pr` at the new `w1`.
 6. Compute `ens`, the effective number of scenarios of `w1`, and `kld`, the divergence of `w1` from `w0`.
 7. Return a [`LowOrderPrior`](@ref) carrying the refit moments, `w1`, `ens` and `kld`. The feature matrix and the factor block come from `pr` unchanged.

# Arguments

  - `alg`: Staged entropy pooling algorithm.

      + `::H1_EntropyPooling`: Each stage re-solves from the original prior weights.
      + `::H2_EntropyPooling`: Each stage re-solves from the previous stage's weights.

  - `pe`: Entropy pooling prior estimator.

  - `X`: Asset returns matrix, already oriented.

  - `F`: Optional factor returns matrix, already oriented.

  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.

  - `kwargs...`: Additional keyword arguments forwarded to the wrapped prior estimator.

# Returns

  - `pr::LowOrderPrior`: Prior result carrying the posterior probability weights.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`H1_EntropyPooling`](@ref)
  - [`H2_EntropyPooling`](@ref)
  - [`entropy_pooling`](@ref)
"""
function ep_prior(alg::StagedEP, pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum};
                  strict::Bool = false, kwargs...)
    T, N = size(X)
    w1 = w0 = if isnothing(pe.w)
        iT = inv(T)
        StatsBase.pweights(range(iT, iT; length = T))
    else
        @argcheck(length(pe.w) == T,
                  DimensionMismatch("length(pe.w) ($(length(pe.w))) must match T ($T)"))
        pe.w
    end
    fixed = falses(N, 2)
    epc = Dict{Symbol, Tuple{<:MatNum, <:VecNum}}()
    tvs = Vector{AbstractEntropyPoolingTailView}(undef, 0)
    # mu, VaR, CVaR, EVaR and RLVaR
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets, w0; strict = strict)
    ep_tail_views!(pe.cvar_views, epc, tvs, pr, pe.sets, w0; strict = strict)
    ep_tail_views!(pe.evar_views, epc, tvs, pr, pe.sets, w0; strict = strict)
    ep_tail_views!(pe.rlvar_views, epc, tvs, pr, pe.sets, w0; strict = strict)
    if !isempty(epc) || !isempty(tvs)
        w1 = entropy_pooling(w0, epc, tvs, pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    if !isnothing(pe.sigma_views) || !isnothing(pe.cov_views)
        # sigma
        if !isnothing(pe.sigma_views)
            to_fix = ep_sigma_views!(pe.sigma_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
        end
        # cov
        if !isnothing(pe.cov_views)
            to_fix = ep_cov_views!(pe.cov_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
        end
        w1 = entropy_pooling(ifelse(isa(alg, H1_EntropyPooling), w0, w1), epc, tvs, pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    if !isnothing(pe.rho_views) || !isnothing(pe.sk_views) || !isnothing(pe.kt_views)
        # skew
        if !isnothing(pe.sk_views)
            to_fix = ep_sk_views!(pe.sk_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        # kurtosis
        if !isnothing(pe.kt_views)
            to_fix = ep_kt_views!(pe.kt_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        # rho
        if !isnothing(pe.rho_views)
            to_fix = ep_rho_views!(pe.rho_views, epc, pr, pe.sets; strict = strict)
            fix_mu!(epc, view(fixed, :, 1), to_fix, pr)
            fix_sigma!(epc, view(fixed, :, 2), to_fix, pr)
        end
        w1 = entropy_pooling(ifelse(isa(alg, H1_EntropyPooling), w0, w1), epc, tvs, pe.opt)
        pe = factory(pe, w1)
        pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    end
    # Entropy pooling reweights observations without touching either axis of `Z`, so the
    # wrapped prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole, on the same reasoning as the
    # note at the same seam in `MeucciEntropyPoolingPrior`'s `ep_prior`.
    (; X, o_X, mu, sigma, chol, rr, fpr, Z) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, kld = kld, rr = rr, fpr = fpr, Z = Z)
end
"""
    ep_prior(alg::H0_EntropyPooling, pe::EntropyPoolingPrior, X::MatNum,
             F::Option{<:MatNum}; strict::Bool = false, kwargs...)

Compute entropy pooling prior moments with tail views, enforcing every view in one optimisation.

`ep_prior` builds every view constraint against the same prior and solves once. It is faster than the staged algorithms and pins nothing, so a higher moment view is free to move a lower moment.

# Algorithm

 1. Build the prior probabilities `w0`. They are the uniform weights `1/T` where `pe.w` is `nothing`, and `pe.w` itself otherwise, whose length must match `T`.
 2. Fit the wrapped prior estimator at `w0`, giving `pr`.
 3. Build every view against that one `pr`: the mean, value at risk, conditional, entropic and relativistic value at risk, variance, covariance, skewness, kurtosis and correlation views. Each row that is linear in the posterior probabilities reaches the constraint dictionary `epc`, and each tail view that needs auxiliary variables reaches the tail view vector `tvs`. No asset's mean or variance is pinned.
 4. Solve once from `w0` with [`entropy_pooling`](@ref), giving `w1`, and refit `pr` at `w1`.
 5. Compute `ens`, the effective number of scenarios of `w1`, and `kld`, the divergence of `w1` from `w0`.
 6. Return a [`LowOrderPrior`](@ref) carrying the refit moments, `w1`, `ens` and `kld`. The feature matrix and the factor block come from `pr` unchanged.

# Arguments

  - `alg`: Single-shot entropy pooling algorithm.
  - `pe`: Entropy pooling prior estimator.
  - `X`: Asset returns matrix, already oriented.
  - `F`: Optional factor returns matrix, already oriented.
  - `strict`: If `true`, throws error for missing assets; otherwise, issue warnings.
  - `kwargs...`: Additional keyword arguments forwarded to the wrapped prior estimator.

# Returns

  - `pr::LowOrderPrior`: Prior result carrying the posterior probability weights.

# Related

  - [`EntropyPoolingPrior`](@ref)
  - [`H0_EntropyPooling`](@ref)
  - [`entropy_pooling`](@ref)
"""
function ep_prior(alg::H0_EntropyPooling, pe::EntropyPoolingPrior, X::MatNum,
                  F::Option{<:MatNum}; strict::Bool = false, kwargs...)
    T = size(X, 1)
    w0 = if isnothing(pe.w)
        iT = inv(T)
        StatsBase.pweights(range(iT, iT; length = T))
    else
        @argcheck(length(pe.w) == T,
                  DimensionMismatch("length(pe.w) ($(length(pe.w))) must match T ($T)"))
        pe.w
    end
    epc = Dict{Symbol, Tuple{<:MatNum, <:VecNum}}()
    tvs = Vector{AbstractEntropyPoolingTailView}(undef, 0)
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    # mu, VaR, CVaR, EVaR and RLVaR
    ep_mu_views!(pe.mu_views, epc, pr, pe.sets; strict = strict)
    ep_var_views!(pe.var_views, epc, pr, pe.sets, w0; strict = strict)
    ep_tail_views!(pe.cvar_views, epc, tvs, pr, pe.sets, w0; strict = strict)
    ep_tail_views!(pe.evar_views, epc, tvs, pr, pe.sets, w0; strict = strict)
    ep_tail_views!(pe.rlvar_views, epc, tvs, pr, pe.sets, w0; strict = strict)
    # sigma
    if !isnothing(pe.sigma_views)
        ep_sigma_views!(pe.sigma_views, epc, pr, pe.sets; strict = strict)
    end
    # cov
    if !isnothing(pe.cov_views)
        ep_cov_views!(pe.cov_views, epc, pr, pe.sets; strict = strict)
    end
    # skew
    if !isnothing(pe.sk_views)
        ep_sk_views!(pe.sk_views, epc, pr, pe.sets; strict = strict)
    end
    # kurtosis
    if !isnothing(pe.kt_views)
        ep_kt_views!(pe.kt_views, epc, pr, pe.sets; strict = strict)
    end
    # rho
    if !isnothing(pe.rho_views)
        ep_rho_views!(pe.rho_views, epc, pr, pe.sets; strict = strict)
    end
    w1 = entropy_pooling(w0, epc, tvs, pe.opt)
    pe = factory(pe, w1)
    pr = prior(pe.pe, X, F; strict = strict, kwargs...)
    # Entropy pooling reweights observations without touching either axis of `Z`, so the
    # wrapped prior's feature matrix is forwarded unchanged (see [`LowOrderPrior`](@ref)).
    # The factor block is the refit prior's, forwarded whole, on the same reasoning as the
    # note at the same seam in `MeucciEntropyPoolingPrior`'s `ep_prior`.
    (; X, o_X, mu, sigma, chol, rr, fpr, Z) = pr
    ens = exp(StatsBase.entropy(w1))
    kld = StatsBase.kldivergence(w1, w0)
    return LowOrderPrior(; X = X, o_X = o_X, mu = mu, sigma = sigma, chol = chol, w = w1,
                         ens = ens, kld = kld, rr = rr, fpr = fpr, Z = Z)
end
function factor_residual_config(pe::EntropyPoolingPrior)
    return factor_residual_config(pe.pe)
end

export EntropyPoolingPrior
