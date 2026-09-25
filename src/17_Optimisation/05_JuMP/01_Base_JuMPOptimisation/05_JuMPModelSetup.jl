"""
    set_initial_w!(args...)
    set_initial_w!(w::VecNum, wi::VecNum)

Set the start values of the weight variables of the JuMP model.

The method for two vectors sets the start value of each weight variable to the matching entry of `wi`. Every other argument, `nothing` or a vector of weight vectors among them, reaches the method that does nothing.

# Arguments

  - `w::VecNum`: Vector of JuMP weight variables.
  - `wi::VecNum`: Vector of start values.

# Validation

  - `length(wi) == length(w)`. Otherwise a `DimensionMismatch` names both lengths.

# Returns

  - `nothing`.

# Related

  - [`set_w!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_initial_w!(args...)
    return nothing
end
function set_initial_w!(w::VecNum, wi::VecNum)
    @argcheck(length(wi) == length(w),
              DimensionMismatch("wi ($(length(wi))) must match w ($(length(w)))"))
    JuMP.set_start_value.(w, wi)
    return nothing
end
"""
    set_w!(model::JuMP.Model, X::MatNum, wi::Option{<:VecNum_VecVecNum})

Register the portfolio weight variables in the JuMP model, and set their start values from `wi`.

A vector `wi` sets one start value for each weight through [`set_initial_w!`](@ref). A vector of weight vectors, which a [`NearOptimalCentering`](@ref) sweep passes as its anchors, and `nothing` set no start value.

# JuMP formulation

## Variables

  - `w`: ``\\boldsymbol{w}``, created free, with one entry for each column of `X`.

Where:

  - $(math_dict[:w_port])

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix, ``T \\times N``. The function reads only its number of columns.
  - `wi`: Start values of the weights, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_initial_w!`](@ref)
  - [`get_w`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_w!(model::JuMP.Model, X::MatNum, wi::Option{<:VecNum_VecVecNum})
    JuMP.@variable(model, w[1:size(X, 2)])
    set_initial_w!(w, wi)
    return nothing
end
"""
    process_model(model::JuMP.Model, retcode::OptimisationSuccess)
    process_model(model::JuMP.Model, retcode::OptimisationFailure)

Read the portfolio weights of a solved JuMP model into a [`JuMPOptimisationSolution`](@ref).

After a success the weights are the solved `w` divided by the solved `k`. A ratio objective solves for ``k \\boldsymbol{w}``, so the division gives weights on the scale of the budget. When `k` solves to zero the weights are the solved `w` unchanged. After a failure the weights are a vector of `NaN`, one for each weight, in the value type of the model.

# Arguments

  - `model::JuMP.Model`: The optimised JuMP model.
  - `retcode`: The return code, [`OptimisationSuccess`](@ref) or [`OptimisationFailure`](@ref), that selects the method.

# Returns

  - `sol::JuMPOptimisationSolution`: The solution that holds the weights.

# Related

  - [`optimise_JuMP_model!`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function process_model(model::JuMP.Model, ::OptimisationSuccess)
    k = JuMP.value(model[:k])
    ik = !iszero(k) ? inv(k) : 1
    w = JuMP.value.(model[:w]) * ik
    return JuMPOptimisationSolution(; w = w)
end
function process_model(model::JuMP.Model, ::OptimisationFailure)
    return JuMPOptimisationSolution(;
                                    w = fill(convert(JuMP.value_type(typeof(model)), NaN),
                                             length(model[:w])))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Solve the JuMP model with each solver of `opt.opt.slv` in turn, until one gives an acceptable solution.

A solution is acceptable when JuMP reports it solved and feasible under the `check_sol` keywords of the solver, every weight is finite, and at least one weight is not zero. The fallback of the optimiser is not tried here.

# Algorithm

 1. For each solver of `opt.opt.slv`, in order:
     1. Attach the solver to `model`. When that raises, record the error under the name of the solver and go to the next solver.
     2. Set the attributes of the solver from its `settings`.
     3. Solve. When that raises, record the error and go to the next solver.
     4. Check the solution with `JuMP.assert_is_solved_and_feasible`, and read the weights. When every weight is finite and one weight is not zero, stop the loop.
     5. Otherwise record the settings, the error of the check if there is one, and the solution summary under the name of the solver.
 2. When a solver stopped the loop, the return code is an [`OptimisationSuccess`](@ref). Otherwise it is an [`OptimisationFailure`](@ref), and a warning points to its `res`.
 3. Read the weights with [`process_model`](@ref).

# Returns

  - `(retcode, sol)`: The return code and the [`JuMPOptimisationSolution`](@ref). The `res` field of the return code is a `Dict` from the name of each solver that failed to what went wrong. The solver that succeeds adds no entry, so the `res` of a success on the first solver is empty.

# Related

  - [`JuMPOptimisationEstimator`](@ref)
  - [`JuMPOptimisationSolution`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
  - [`process_model`](@ref)
"""
function optimise_JuMP_model!(model::JuMP.Model, opt::JuMPOptimisationEstimator,
                              datatype::DataType = Float64)
    trials = Dict()
    success = false
    for solver in opt.opt.slv
        try
            JuMP.set_optimizer(model, solver.solver; add_bridges = solver.add_bridges)
        catch err
            trials[solver.name] = Dict(:set_optimizer => err)
            continue
        end
        set_solver_attributes(model, solver.settings)
        try
            JuMP.optimize!(model)
        catch err
            trials[solver.name] = Dict(:optimize! => err)
            continue
        end
        trial = Dict{Symbol, Any}(:settings => solver.settings)
        try
            JuMP.assert_is_solved_and_feasible(model; solver.check_sol...)
            wv = JuMP.value.(model[:w])
            all_finite_weights = all(isfinite, wv)
            all_non_zero_weights = !all(x -> isapprox(x, zero(datatype)), wv)
            if all_finite_weights && all_non_zero_weights
                success = true
                break
            end
        catch err
            trial[:assert_is_solved_and_feasible] = err
        end
        trial[:err] = JuMP.solution_summary(model)
        trials[solver.name] = trial
    end
    retcode = if success
        OptimisationSuccess(; res = trials)
    else
        @warn("Failed to solve optimisation problem.\nCheck `result.retcode.res` on the returned result for per-solver diagnostics.")
        OptimisationFailure(; res = trials)
    end
    return retcode, process_model(model, retcode)
end
"""
    set_portfolio_returns!(model::JuMP.Model, X::MatNum; prefix::Symbol = Symbol(""))

Register the portfolio return series ``\\mathbf{X} \\boldsymbol{w}`` in the JuMP model, and return it.

A second call returns the registered series and builds nothing.

# JuMP formulation

## Variables

  - `w`: ``\\boldsymbol{w}``, read under `prefix` with [`get_w`](@ref).

## Expressions

  - `X` under `prefix`: ``\\mathbf{X} \\boldsymbol{w}``, a ``T \\times 1`` vector.

Where:

  - $(math_dict[:X_returns])
  - $(math_dict[:w_port])

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix.
  - `prefix::Symbol`: Model State namespace of the build. The empty default gives the bare key.

# Returns

  - The portfolio return series.

# Related

  - [`set_net_portfolio_returns!`](@ref)
  - [`get_X`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_portfolio_returns!(model::JuMP.Model, X::MatNum; prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :X))
        return model[Symbol(prefix, :X)]
    end
    w = get_w(model, prefix)
    return state_set!(model, prefix, :X, JuMP.@expression(model, X * w))
end
"""
    set_net_portfolio_returns!(model::JuMP.Model, X::MatNum; prefix::Symbol = Symbol(""))

Register the portfolio return series net of fees in the JuMP model, and return it.

The model's `:fees` expression holds the per period terms `l`, `s`, `tn` and `lq`. They are rates for each period, so each observation pays them. The model's `:one_time_fees` expression holds the one-off terms `fl`, `fs` and `flq`, which the holding period pays one time, and `:fee_fa` names the clock they fall on. A `nothing` or [`FirstObservationFees`](@ref) clock takes them from the first observation alone, and an [`AmortisedFees`](@ref) clock spreads them evenly over the observation count of the fit. [`charge_fees`](@ref) states the same rule for values, and [`charge_one_time_fees`](@ref) applies it here. A second call returns the registered series and builds nothing. `:fees` and `:one_time_fees` are shared entries, so a nested build reads them under the bare key.

# Algorithm

 1. When the series is registered under `prefix`, return it.
 2. Build the portfolio return series with [`set_portfolio_returns!`](@ref), giving `Xe`.
 3. When the model holds no `:fees`, register `Xe` as the net series and return it.
 4. Subtract `:fees` from each entry of `Xe`, giving `net`.
 5. When the model holds `:one_time_fees` and `net` is not empty, charge them on the clock `:fee_fa` with [`charge_one_time_fees`](@ref), over the count [`get_T`](@ref).
 6. Register `net` and return it.

# JuMP formulation

## Expressions

  - `net_X` under `prefix`: ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w} - f_r - c_t f_o``.

Where:

  - $(math_dict[:x_t_obs])
  - $(math_dict[:w_port])
  - ``f_r``: The per period fee, `:fees`, or zero when the model holds none.
  - ``f_o``: The one-off fee, `:one_time_fees`, or zero when the model holds none.
  - ``c_t``: The share of the one-off fee that observation ``t`` pays. Under a `nothing` or [`FirstObservationFees`](@ref) clock ``c_1 = 1`` and ``c_t = 0`` for ``t > 1``. Under an [`AmortisedFees`](@ref) clock ``c_t = 1 / T``.
  - $(math_dict[:T])

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix.
  - `prefix::Symbol`: Model State namespace of the build. The empty default gives the bare key.

# Returns

  - The net portfolio return series.

# Related

  - [`set_portfolio_returns!`](@ref)
  - [`get_net_X`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_net_portfolio_returns!(model::JuMP.Model, X::MatNum;
                                    prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :net_X))
        return model[Symbol(prefix, :net_X)]
    end
    Xe = set_portfolio_returns!(model, X; prefix = prefix)
    # `:fees` and `:one_time_fees` are shared and not recreated by a nested build, so both
    # are read bare.
    if !haskey(model, :fees)
        return state_set!(model, prefix, :net_X, JuMP.@expression(model, Xe))
    end
    fees = model[:fees]
    net = JuMP.@expression(model, Xe .- fees)
    if haskey(model, :one_time_fees) && !isempty(net)
        # The clock the fee states decides where the two fixed terms land, exactly as
        # `charge_fees` decides it at the value level.
        net = charge_one_time_fees(model, net, model[:one_time_fees], get_T(model),
                                   model[:fee_fa])
    end
    return state_set!(model, prefix, :net_X, net)
end
"""
    set_asset_returns_plus_one!(model::JuMP.Model, X::MatNum; prefix::Symbol = Symbol(""))

Register the gross asset returns ``\\mathbf{X} + 1`` in the JuMP model, and return them.

The value is the constant matrix `X .+ 1`, not a JuMP expression. The distributionally robust conditional value at risk measures the transport cost of its Wasserstein ambiguity ball against it. The gain side of that measure passes `-X` under a prefix of its own. A second call returns the registered matrix and builds nothing.

# JuMP formulation

## Expressions

  - `Xap1` under `prefix`: ``\\mathbf{X} + 1``, entry by entry.

Where:

  - $(math_dict[:X_returns])

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix.
  - `prefix::Symbol`: Model State namespace of the build. The empty default gives the bare key.

# Returns

  - The gross asset returns matrix.

# Related

  - [`get_Xap1`](@ref)
  - [`set_asset_neg_returns_plus_one!`](@ref)
  - [`set_portfolio_drawdowns_plus_one!`](@ref)
"""
function set_asset_returns_plus_one!(model::JuMP.Model, X::MatNum;
                                     prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :Xap1))
        return model[Symbol(prefix, :Xap1)]
    end
    return state_set!(model, prefix, :Xap1, JuMP.@expression(model, X .+ one(eltype(X))))
end
"""
    set_asset_neg_returns_plus_one!(model::JuMP.Model, X::MatNum; prefix::Symbol = Symbol(""))

Register the negated gross asset returns ``1 - \\mathbf{X}`` in the JuMP model, and return them.

The value is the constant matrix `-X .+ 1`, not a JuMP expression. A second call returns the registered matrix and builds nothing.

# JuMP formulation

## Expressions

  - `nXap1` under `prefix`: ``1 - \\mathbf{X}``, entry by entry.

Where:

  - $(math_dict[:X_returns])

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix.
  - `prefix::Symbol`: Model State namespace of the build. The empty default gives the bare key.

# Returns

  - The negated gross asset returns matrix.

# Related

  - [`set_asset_returns_plus_one!`](@ref)
"""
function set_asset_neg_returns_plus_one!(model::JuMP.Model, X::MatNum;
                                         prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :nXap1))
        return model[Symbol(prefix, :nXap1)]
    end
    return state_set!(model, prefix, :nXap1, JuMP.@expression(model, -X .+ one(eltype(X))))
end
"""
    set_portfolio_drawdowns_plus_one!(model::JuMP.Model, X::MatNum; prefix::Symbol = Symbol(""))

Register the drawdowns of each asset plus one in the JuMP model, and return them.

[`absolute_drawdown_arr`](@ref) takes the drawdown of each column of `X` on the cumulative sum of its returns, with the running peak seeded at zero. The value is a constant matrix, not a JuMP expression. The distributionally robust conditional drawdown at risk measures the transport cost of its Wasserstein ambiguity ball against it. A second call returns the registered matrix and builds nothing.

# Mathematical definition

```math
\\begin{align}
c_{t,i} &= \\sum_{s=1}^{t} x_{s,i}\\,, \\\\
D_{t,i} &= 1 + c_{t,i} - \\max\\left(0,\\, \\max_{s \\leq t} c_{s,i}\\right)\\,.
\\end{align}
```

Where:

  - ``x_{s,i}``: The return of asset ``i`` at observation ``s``, an entry of `X`.
  - ``c_{t,i}``: The cumulative sum of the returns of asset ``i`` up to observation ``t``.
  - ``D_{t,i}``: The drawdown of asset ``i`` at observation ``t``, plus one. It is at most one.

# JuMP formulation

## Expressions

  - `ddap1` under `prefix`: the matrix of ``D_{t,i}``, ``T \\times N``.

Where:

  - $(math_dict[:T])
  - $(math_dict[:N])

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix, ``T \\times N``.
  - `prefix::Symbol`: Model State namespace of the build. The empty default gives the bare key.

# Returns

  - The matrix of asset drawdowns plus one.

# Related

  - [`get_ddap1`](@ref)
  - [`absolute_drawdown_arr`](@ref)
  - [`set_asset_returns_plus_one!`](@ref)
"""
function set_portfolio_drawdowns_plus_one!(model::JuMP.Model, X::MatNum;
                                           prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :ddap1))
        return model[Symbol(prefix, :ddap1)]
    end
    _ddap1 = absolute_drawdown_arr(X) .+ one(eltype(X))
    return state_set!(model, prefix, :ddap1, JuMP.@expression(model, _ddap1))
end
"""
    scalarise_risk_expression!(model::JuMP.Model, sca)

Reduce the entries of `model[:risk_vec]` to the one risk expression `model[:risk]`.

This is the declaration of the generic function. Each scalariser defines a method in the file of the risk constraints.

# Related

  - [`SumScalariser`](@ref)
  - [`get_risk`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function scalarise_risk_expression! end
"""
    set_risk_constraints!(model::JuMP.Model, i, r, opt, pr, args...; kwargs...)

Add the variables, expressions and rows of the risk measure `r` to the JuMP model.

This is the declaration of the generic function. Each risk measure defines a method in the file of its constraints. `i` is the index of the measure in its vector, and it keeps the entries of two measures of one type apart.

# Related

  - [`scalarise_risk_expression!`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
function set_risk_constraints! end
