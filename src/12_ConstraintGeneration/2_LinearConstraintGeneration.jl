struct PartialLinearConstraintResult{T1 <: AbstractMatrix, T2 <: AbstractVector} <:
       AbstractConstraintResult
    A::T1
    B::T2
end
function PartialLinearConstraintResult(; A::AbstractMatrix, B::AbstractVector)
    @smart_assert(!isempty(A) && !isempty(B))
    return PartialLinearConstraintResult{typeof(A), typeof(B)}(A, B)
end
struct LinearConstraintResult{T1 <: Union{Nothing, <:PartialLinearConstraintResult},
                              T2 <: Union{Nothing, <:PartialLinearConstraintResult}} <:
       AbstractConstraintResult
    ineq::T1
    eq::T2
end
function LinearConstraintResult(;
                                ineq::Union{Nothing, <:PartialLinearConstraintResult} = nothing,
                                eq::Union{Nothing, <:PartialLinearConstraintResult} = nothing)
    @smart_assert(isnothing(ineq) ⊼ isnothing(eq))
    return LinearConstraintResult{typeof(ineq), typeof(eq)}(ineq, eq)
end
function Base.getproperty(obj::LinearConstraintResult, sym::Symbol)
    return if sym == :A_ineq
        isnothing(obj.ineq) ? nothing : obj.ineq.A
    elseif sym == :B_ineq
        isnothing(obj.ineq) ? nothing : obj.ineq.B
    elseif sym == :A_eq
        isnothing(obj.eq) ? nothing : obj.eq.A
    elseif sym == :B_eq
        isnothing(obj.eq) ? nothing : obj.eq.B
    else
        getfield(obj, sym)
    end
end

# Recursively evaluate numeric functions
function _eval_numeric_functions(expr)
    return if expr isa Expr
        if expr.head == :call
            fname = expr.args[1]
            # Only evaluate if all arguments are numeric
            args = [_eval_numeric_functions(arg) for arg in expr.args[2:end]]
            if all(x -> isa(x, Number), args)
                Base.invokelatest(getfield(Base, fname), args...)
            else
                Expr(:call, fname, args...)
            end
        else
            Expr(expr.head, map(_eval_numeric_functions, expr.args)...)
        end
    elseif expr isa Symbol && expr == :Inf
        Inf
    else
        expr
    end
end
# Collect terms: returns vector of (coefficient, variable::Union{String,Nothing})
function _collect_terms(expr)
    terms = []
    _collect_terms!(expr, 1.0, terms)
    return terms
end
function _collect_terms!(expr, coeff, terms)
    if expr isa Number
        push!(terms, (coeff * float(expr), nothing))
    elseif expr isa Symbol
        push!(terms, (coeff, string(expr)))
    elseif expr isa Expr
        if expr.head == :call && expr.args[1] == :*
            # Multiplication: find numeric and variable part
            a, b = expr.args[2], expr.args[3]
            if a isa Number
                _collect_terms!(b, coeff * float(a), terms)
            elseif b isa Number
                _collect_terms!(a, coeff * float(b), terms)
            else
                # e.g. x*y, treat as variable
                push!(terms, (coeff, string(expr)))
            end
        elseif expr.head == :call && expr.args[1] == :/
            a, b = expr.args[2], expr.args[3]
            if b isa Number
                _collect_terms!(a, coeff / float(b), terms)
            else
                # e.g. x/y, treat as variable
                push!(terms, (coeff, string(expr)))
            end
        elseif expr.head == :call && expr.args[1] == :+
            for i in 2:length(expr.args)
                # Collect terms from addition
                _collect_terms!(expr.args[i], coeff, terms)
            end
        elseif expr.head == :call && expr.args[1] == :-
            for i in 2:(length(expr.args) - 1)
                # Collect terms from addition
                _collect_terms!(expr.args[i], coeff, terms)
            end
            _collect_terms!(expr.args[length(expr.args)], -coeff, terms)
        else
            # treat as variable (e.g. sin(x))
            push!(terms, (coeff, string(expr)))
        end
    end
end
function _format_term(coeff, var)
    return if coeff == 1.0
        var
    elseif coeff == -1.0
        "-$var"
    else
        "$(coeff)*$var"
    end
end
function _rethrow_parse_error(::Any, side = :lhs)
    return nothing
end
function _rethrow_parse_error(::Nothing, side = :lhs)
    @warn("$(side) of equation is empy, assuming zero")
    return nothing
end
function _rethrow_parse_error(expr::Expr, side = :lhs)
    if expr.head == :incomplete
        throw(Meta.ParseError("$side is an incomplete expression.\n$expr"))
    end
    return nothing
end
struct ParsingResult{T1, T2, T3, T4, T5}
    vars::T1
    coef::T2
    op::T3
    rhs::T4
    eqn::T5
end
Base.length(res::ParsingResult) = 1
Base.iterate(res::ParsingResult, state = 1) = state > 1 ? nothing : (res, state + 1)
struct AssetSets{T1 <: AbstractString, T2 <: AbstractDict}
    key::T1
    dict::T2
end
function AssetSets(; key::AbstractString = "nx", dict::AbstractDict)
    @smart_assert(!isempty(dict))
    @smart_assert(haskey(dict, key))
    return AssetSets(key, dict)
end
function asset_sets_view(sets::AssetSets, i::AbstractVector)
    dict = Dict(k => v for (k, v) in sets.dict if k != sets.key)
    dict[sets.key] = view(sets.dict[sets.key], i)
    return AssetSets(; key = sets.key, dict = dict)
end
function _parse_equation(lhs, opstr::AbstractString, rhs, datatype::DataType = Float64)
    # 3. Evaluate numeric functions on both sides
    lexpr = _eval_numeric_functions(lhs)
    _rethrow_parse_error(lexpr, :lhs)
    rexpr = _eval_numeric_functions(rhs)
    _rethrow_parse_error(rexpr, :rhs)

    # 4. Move all terms to LHS: lhs - rhs == 0
    diff_expr = :($lexpr - ($rexpr))

    # 5. Expand and collect like terms
    terms = _collect_terms(diff_expr)

    # 6. Separate variables and constant
    varmap = Dict{String, Float64}()
    constant::datatype = 0.0
    for (coeff, var) in terms
        if var === nothing
            constant += coeff
        else
            varmap[var] = get(varmap, var, 0.0) + coeff
        end
    end

    # 7. Move constant to RHS, variables to LHS
    variables = collect(keys(varmap))
    coefficients = [varmap[v] for v in variables]
    rhs_val = -constant

    # 8. Format the simplified expression
    lhs_str = join([_format_term(coeff, var)
                    for (coeff, var) in zip(coefficients, variables)], " + ")
    lhs_str = replace(lhs_str, "+ -" => "-", "  " => " ")
    rhs_str = string(rhs_val)
    formatted = strip("$lhs_str $opstr $rhs_str")

    return ParsingResult(variables, coefficients, opstr, rhs_val, formatted)
end
function parse_equation(eqn::AbstractString; datatype::DataType = Float64)
    if occursin("++", eqn)
        throw(Meta.ParseError("Invalid operator '++' detected in equation."))
    end
    # 1. Identify the comparison operator
    ops = ("==", "<=", ">=")
    op = findfirst(op -> occursin(op, eqn), ops)
    if isnothing(op)
        error("Equation must contain a comparison operator (==, <=, >=).\n$(eqn)")
    end
    opstr = ops[op]
    parts = split(eqn, opstr)
    if length(parts) != 2
        error("Equation must have exactly one comparison operator.\n$(eqn)")
    end
    lhs, rhs = strip.(parts)
    # 2. Parse both sides into Julia expressions
    lexpr = Meta.parse(lhs)
    _rethrow_parse_error(lexpr, :lhs)
    rexpr = Meta.parse(rhs)
    _rethrow_parse_error(rexpr, :rhs)
    return _parse_equation(lexpr, opstr, rexpr, datatype)
end
function parse_equation(expr::Expr; datatype::DataType = Float64)
    # 1. Identify the comparison operator in the expression
    ops = (:call, :(==), :(<=), :(>=))
    if expr.head != :call || !(expr.args[1] in (:(==), :(<=), :(>=)))
        error("Expression must be a comparison (==, <=, >=):\n$expr")
    end
    opstr = string(expr.args[1])
    lhs, rhs = expr.args[2], expr.args[3]
    return _parse_equation(lhs, opstr, rhs, datatype)
end
function parse_equation(eqn::Union{<:AbstractVector{<:AbstractString},
                                   <:AbstractVector{Expr},
                                   <:AbstractVector{<:Union{<:AbstractString, Expr}}};
                        datatype::DataType = Float64)
    return parse_equation.(eqn; datatype = datatype)
end
function replace_group_by_assets(res::ParsingResult, sets::AssetSets, flag::Bool = false)
    variables, coeffs = res.vars, res.coef
    variables_new = copy(variables)
    coeffs_new = copy(coeffs)
    variables_tmp = Vector{eltype(variables)}(undef, 0)
    coeffs_tmp = Vector{eltype(coeffs)}(undef, 0)
    idx_rm = Vector{Int}(undef, 0)
    prior_pattern = r"prior\(([^()]*)\)"
    corr_pattern = r"\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"
    for (i, (v, c)) in enumerate(zip(variables, coeffs))
        m = match(prior_pattern, v)
        if isnothing(m)
            n = match(corr_pattern, v)
            if isnothing(n)
                asset = get(sets.dict, v, nothing)
                if isnothing(asset)
                    continue
                end
                c = coeffs[i]
                append!(variables_tmp, asset)
                append!(coeffs_tmp, Iterators.repeated(c, length(asset)))
                push!(idx_rm, i)
            else
                if !flag
                    throw(ArgumentError("`prior(a)` and `(a, b)` can only be used in entropy pooling."))
                end
                assets12 = n.captures[1]
                assets12 = split(assets12, ", ")
                asset1 = get(sets.dict, assets12[1], nothing)
                asset2 = get(sets.dict, assets12[2], nothing)
                if isnothing(asset1) && isnothing(asset2)
                    continue
                end
                @smart_assert(!isnothing(asset1) && !isnothing(asset2))
                @smart_assert(length(asset1) == length(asset2))
                variables_new[i] = "($(asset1[1]), $(asset2[1]))"
            end
        else
            if !flag
                throw(ArgumentError("`prior(a)` and `(a, b)` can only be used in entropy pooling."))
            end
            n = match(corr_pattern, v)
            if isnothing(n)
                asset = get(sets.dict, v[7:(end - 1)], nothing)
                if isnothing(asset)
                    continue
                end
                c = coeffs[i]
                append!(variables_tmp, ["prior($a)" for a in asset])
                append!(coeffs_tmp, Iterators.repeated(c, length(asset)))
                push!(idx_rm, i)
            else
                asset1 = n.captures[1]
                asset2 = n.captures[2]
                asset1 = get(sets.dict, asset1, nothing)
                asset2 = get(sets.dict, asset2, nothing)
                if isnothing(asset1) && isnothing(asset2)
                    continue
                end
                @smart_assert(!isnothing(asset1) && !isnothing(asset2))
                @smart_assert(length(asset1) == length(asset2))
                variables_new[i] = "prior($(asset1[1]), $(asset2[1]))"
            end
        end
    end
    if isempty(variables_tmp)
        return res
    end
    deleteat!(variables_new, idx_rm)
    deleteat!(coeffs_new, idx_rm)
    append!(variables_new, variables_tmp)
    append!(coeffs_new, coeffs_tmp)
    eqn = replace(join(string.(coeffs_new) .* "*" .* variables_new, " + "), "+ -" => "-",
                  "  " => " ")
    return ParsingResult(variables_new, coeffs_new, res.op, res.rhs,
                         "$(eqn) $(res.op) $(res.rhs)")
end
function replace_group_by_assets(res::AbstractVector{<:ParsingResult}, sets::AssetSets,
                                 flag::Bool = false)
    return replace_group_by_assets.(res, Ref(sets), flag)
end
function get_linear_constraints(lcs::Union{<:ParsingResult,
                                           <:AbstractVector{<:ParsingResult}},
                                sets::AssetSets; datatype::DataType = Float64,
                                strict::Bool = false)
    if isa(lcs, AbstractVector)
        @smart_assert(!isempty(lcs))
    end
    A_ineq = Vector{datatype}(undef, 0)
    B_ineq = Vector{datatype}(undef, 0)
    A_eq = Vector{datatype}(undef, 0)
    B_eq = Vector{datatype}(undef, 0)
    nx = sets.dict[sets.key]
    At = Vector{datatype}(undef, length(nx))
    for lc in lcs
        fill!(At, zero(eltype(At)))
        for (v, c) in zip(lc.vars, lc.coef)
            Ai = (nx .== v)
            if !any(isone, Ai)
                msg = "$(v) is not found in $(nx)."
                strict ? throw(ArgumentError(msg)) : @warn(msg)
                continue
            end
            At += Ai * c
        end
        d = ifelse(lc.op == ">=", -1, 1)
        flag = d == -1 || lc.op == "<="
        A = At .* d
        B = lc.rhs * d
        if flag
            append!(A_ineq, A)
            append!(B_ineq, B)
        else
            append!(A_eq, A)
            append!(B_eq, B)
        end
    end
    ineq_flag = !isempty(A_ineq)
    eq_flag = !isempty(A_eq)
    ineq = nothing
    eq = nothing
    if ineq_flag
        A_ineq = transpose(reshape(A_ineq, length(nx), :))
        ineq = PartialLinearConstraintResult(; A = A_ineq, B = B_ineq)
    end
    if eq_flag
        A_eq = transpose(reshape(A_eq, length(nx), :))
        eq = PartialLinearConstraintResult(; A = A_eq, B = B_eq)
    end
    return if !ineq_flag && !eq_flag
        nothing
    else
        LinearConstraintResult(; ineq = ineq, eq = eq)
    end
end
function linear_constraints(lcs::LinearConstraintResult, args...; kwargs...)
    return lcs
end
function linear_constraints(::Nothing, args...; kwargs...)
    return nothing
end
function linear_constraints(eqn::Union{<:AbstractString, Expr,
                                       <:AbstractVector{<:AbstractString},
                                       <:AbstractVector{Expr},
                                       <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                            sets::AssetSets; datatype::DataType = Float64,
                            strict::Bool = false)
    lcs = parse_equation(eqn; datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets)
    return get_linear_constraints(lcs, sets; datatype = datatype, strict = strict)
end
function get_weight_bounds_constraints(lcs::Union{<:ParsingResult,
                                                  <:AbstractVector{<:ParsingResult}},
                                       sets::AssetSets; datatype::DataType = Float64,
                                       strict::Bool = false)
    nx = sets.dict[sets.key]
    lb = zeros(promote_type(eltype(datatype), eltype(datatype)), length(nx))
    ub = ones(promote_type(eltype(datatype), eltype(datatype)), length(nx))
    At = falses(length(nx))
    for lc in lcs
        fill!(At, false)
        for v in lc.vars
            Ai = (nx .== v)
            if !any(isone, Ai)
                msg = "$(v) is not found in $(nx)."
                strict ? throw(ArgumentError(msg)) : @warn(msg)
                continue
            end
            At .= At .|| Ai
        end
        if lc.op == "<="
            ub[At] .= lc.rhs
        elseif lc.op == ">="
            lb[At] .= lc.rhs
        else
            ub[At] .= lc.rhs
            lb[At] .= lc.rhs
        end
    end
    return WeightBoundsResult(; lb = lb, ub = ub)
end
function weight_bounds_constraints(eqn::Union{<:AbstractString, Expr,
                                              <:AbstractVector{<:AbstractString},
                                              <:AbstractVector{Expr},
                                              <:AbstractVector{<:Union{<:AbstractString,
                                                                       Expr}}},
                                   sets::AssetSets; datatype::DataType = Float64,
                                   strict::Bool = false)
    lcs = parse_equation(eqn; datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets)
    return get_weight_bounds_constraints(lcs, sets; datatype = datatype, strict = strict)
end
function get_risk_budget_constraints(lcs::Union{<:ParsingResult,
                                                <:AbstractVector{<:ParsingResult}},
                                     sets::AssetSets; datatype::DataType = Float64,
                                     strict::Bool = false)
    nx = sets.dict[sets.key]
    rb = fill(inv(length(nx)), length(nx))
    At = falses(length(nx))
    for lc in lcs
        fill!(At, false)
        for v in lc.vars
            Ai = (nx .== v)
            if !any(isone, Ai)
                msg = "$(v) is not found in $(nx)."
                strict ? throw(ArgumentError(msg)) : @warn(msg)
                continue
            end
            At .= At .|| Ai
        end
        if lc.rhs < 0
            msg = "$(lc.eqn) has a negative risk budget."
            strict ? throw(ArgumentError(msg)) : @warn(msg)
            continue
        end
        rb[At] .= lc.rhs
    end
    return rb / sum(rb)
end
function risk_budget_constraints(eqn::Union{<:AbstractString, Expr,
                                            <:AbstractVector{<:AbstractString},
                                            <:AbstractVector{Expr},
                                            <:AbstractVector{<:Union{<:AbstractString,
                                                                     Expr}}},
                                 sets::AssetSets; datatype::DataType = Float64,
                                 strict::Bool = false)
    lcs = parse_equation(eqn; datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets)
    return get_risk_budget_constraints(lcs, sets; datatype = datatype, strict = strict)
end
function asset_sets_matrix(smtx::Union{Nothing, Symbol, <:AbstractString}, args...;
                           kwargs...)
    return smtx
end
function asset_sets_matrix(smtx::AbstractString, sets::AssetSets; strict::Bool = false)
    @smart_assert(haskey(sets.dict, smtx))
    nx = sets.dict[sets.key]
    set_keys = unique!(sets.dict[smtx])
    A = BitMatrix(undef, length(set_keys), length(nx))
    for (i, key) in pairs(set_keys)
        group = sets.dict[key]
        for a in group
            idx = findfirst(x -> x == a, nx)
            if isnothing(idx)
                msg = "$a is not found in $nx.\nIt belongs to $key, $group.\nWhich is referenced by $smtx, $set_keys."
                strict ? throw(ArgumentError(msg)) : @warn(msg)
                continue
            end
            A[i, idx] = true
        end
    end
    return A
end
function asset_sets_matrix_view(smtx::Union{Nothing, Symbol, <:AbstractString}, ::Any;
                                kwargs...)
    return smtx
end
function asset_sets_matrix_view(smtx::AbstractMatrix, i::AbstractVector; kwargs...)
    return view(smtx, :, i)
end

export AssetSets, PartialLinearConstraintResult, LinearConstraintResult, parse_equation,
       replace_group_by_assets, linear_constraints, asset_sets_matrix
