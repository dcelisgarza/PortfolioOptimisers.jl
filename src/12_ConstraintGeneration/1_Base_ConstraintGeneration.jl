abstract type AbstractConstraintSide <: AbstractEstimator end
abstract type AbstractConstraint <: AbstractEstimator end
abstract type AbstractConstraintResult <: AbstractResult end
abstract type ComparisonOperators end
abstract type EqualityComparisonOperators <: ComparisonOperators end
abstract type InequalityComparisonOperators <: ComparisonOperators end
struct EQ <: EqualityComparisonOperators end
struct LEQ <: InequalityComparisonOperators end
struct GEQ <: InequalityComparisonOperators end
function comparison_sign_ineq_flag(::EQ)
    return 1, false
end
function comparison_sign_ineq_flag(::LEQ)
    return 1, true
end
function comparison_sign_ineq_flag(::GEQ)
    return -1, true
end
# Recursively evaluate numeric functions
function _eval_numeric_functions(expr)
    if expr isa Number
        return expr
    elseif expr isa Symbol
        return expr
    elseif expr isa Expr
        if expr.head == :call
            fname = expr.args[1]
            # Only evaluate if all arguments are numeric
            args = [_eval_numeric_functions(arg) for arg in expr.args[2:end]]
            if all(x -> x isa Number, args)
                return Base.invokelatest(getfield(Base, fname), args...)
            else
                return Expr(:call, fname, args...)
            end
        else
            return Expr(expr.head, map(_eval_numeric_functions, expr.args)...)
        end
    else
        return expr
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
    if coeff == 1.0
        return var
    elseif coeff == -1.0
        return "-$var"
    else
        return "$(coeff)*$var"
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

function parse_equation(eqn::AbstractString, datatype::DataType = Float64)
    if occursin("++", eqn)
        throw(Meta.ParseError("Invalid operator '++' detected in equation."))
    end
    # 1. Identify the comparison operator
    ops = ("==", "<=", ">=")
    op = findfirst(op -> occursin(op, eqn), ops)
    if isnothing(op)
        error("Equation must contain a comparison operator (==, <=, >=).")
    end
    opstr = ops[op]
    parts = split(eqn, opstr)
    if length(parts) != 2
        error("Equation must have exactly one comparison operator.")
    end
    lhs, rhs = strip.(parts)

    # 2. Parse both sides into Julia expressions
    lexpr = Meta.parse(lhs)
    _rethrow_parse_error(lexpr, :lhs)
    rexpr = Meta.parse(rhs)
    _rethrow_parse_error(rexpr, :rhs)

    # 3. Evaluate numeric functions on both sides
    lexpr = _eval_numeric_functions(lexpr)
    rexpr = _eval_numeric_functions(rexpr)

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
    lhs_str = replace(lhs_str, "+ -" => " - ")
    rhs_str = string(rhs_val)
    formatted = "$lhs_str $opstr $rhs_str"

    return variables, coefficients, opstr, rhs_val, formatted
end

export EQ, LEQ, GEQ, parse_equation
