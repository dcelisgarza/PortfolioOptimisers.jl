abstract type ComparisonOperators end
abstract type EqualityComparisonOperators <: ComparisonOperators end
abstract type InequalityComparisonOperators <: ComparisonOperators end
struct EQ <: EqualityComparisonOperators end
struct LEQ <: InequalityComparisonOperators end
struct GEQ <: InequalityComparisonOperators end
abstract type AbstractConstraintResult <: AbstractResult end
function comparison_sign_ineq_flag(::EQ)
    return 1, false
end
function comparison_sign_ineq_flag(::LEQ)
    return 1, true
end
function comparison_sign_ineq_flag(::GEQ)
    return -1, true
end
abstract type ConstraintSide end
abstract type A_Constraint <: ConstraintSide end
struct A_LinearConstraint{T1, T2, T3 <: Union{<:Real, <:AbstractVector{<:Real}}} <:
       A_Constraint
    group::T1
    name::T2
    coef::T3
end
function A_LinearConstraint(; group, name,
                            coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if group_flag || name_flag || coef_flag
        @smart_assert(group_flag && name_flag && coef_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return A_LinearConstraint{typeof(group), typeof(name), typeof(coef)}(group, name, coef)
end
struct A_CardinalityConstraint{T1, T2} <: A_Constraint
    group::T1
    name::T2
end
function A_CardinalityConstraint(; group, name)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    if group_flag || name_flag
        @smart_assert(group_flag && name_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return A_CardinalityConstraint{typeof(group), typeof(name)}(group, name)
end
function split_equation(equation_str::AbstractString)
    # Remove whitespace for easier parsing
    eq_clean = replace(equation_str, r"\s+" => "")

    invalid = collect(eachmatch(r"(=>|=<)", eq_clean))
    if !isempty(invalid)
        throw(ArgumentError("Invalid comparison operators found in equation: $invalid"))
    end

    # Find the comparison operator
    comp_match = collect(eachmatch(r"(==|<=|>=|=)", eq_clean))

    # Check for multiple comparison operators
    if length(comp_match) > 1
        throw(ArgumentError("Multiple comparison operators found in equation: $comp_match"))
    end

    if isempty(comp_match)
        throw(ArgumentError("No valid comparison operator found in equation: $eq_clean"))
    end

    comp_match = comp_match[1]
    comp_op = comp_match.captures[1]
    comp_pos = comp_match.offset

    # Split into left and right sides
    lhs = eq_clean[1:(comp_pos - 1)]
    rhs = eq_clean[(comp_pos + length(comp_op)):end]

    return Dict("lhs" => lhs, "rhs" => rhs, "comparison" => comp_op)
end
function parse_constraint_equation(equation_str::AbstractString, strict::Bool = true)
    # Split the equation using the existing function
    split_eq = split_equation(equation_str)
    lhs = split_eq["lhs"]
    rhs = split_eq["rhs"]
    comp_op = split_eq["comparison"]

    # Standardize = to ==
    if comp_op == "="
        comp_op = "=="
    end

    # Function to tokenize and parse terms from a side of the equation
    function parse_side(side)
        # Add leading + if needed for consistent parsing
        if !isempty(side) && !(startswith(side, "+") || startswith(side, "-"))
            side = "+" * side
        end

        # Find term boundaries (looking for +/- not inside scientific notation)
        terms = String[]
        term_start = 1
        i = 2
        while i <= length(side)
            if (side[i] == '+' || side[i] == '-') &&
               !(i > 1 && side[i - 1] == 'e' && isdigit(side[i - 2]))
                push!(terms, side[term_start:(i - 1)])
                term_start = i
            end
            i += 1
        end
        # Add the last term
        push!(terms, side[term_start:end])

        # Process each term
        variable_terms = Tuple{Float64, String}[]
        constant_value = 0.0
        pevious_factor = 1.0
        for term ∈ terms
            # Skip empty terms
            if isempty(term)
                continue
            end

            # Get the sign
            if term == "-"
                pevious_factor = -pevious_factor
                continue
            end
            sign_factor = startswith(term, "-") ? -pevious_factor : pevious_factor
            pevious_factor = 1.0 # Reset previous factor.
            term = term[2:end]  # Remove sign

            # Skip empty terms after sign removal
            if isempty(term)
                continue
            end

            # Check if it's a numeric constant
            if occursin(r"^[0-9]+\.?[0-9]*$", term)
                constant_value += sign_factor * parse(Float64, term)
                continue
            end

            # Case 1: coefficient*variable[/denominator]
            m = match(r"^(\d*\.?\d*)\*([a-zA-Z][a-zA-Z0-9_]*)(?:\/(\d+))?$", term)
            if !isnothing(m)
                coef = isnothing(m[1]) || isempty(m[1]) ? 1.0 : parse(Float64, m[1])
                var = m[2]
                denom = isnothing(m[3]) || isempty(m[3]) ? 1.0 : parse(Float64, m[3])
                push!(variable_terms, (sign_factor * coef / denom, var))
                continue
            end

            # Case 2: variable*coefficient[/denominator]
            m = match(r"^([a-zA-Z][a-zA-Z0-9_]*)\*(\d*\.?\d+)(?:\/(\d+))?$", term)
            if !isnothing(m)
                var = m[1]
                coef = parse(Float64, m[2])
                denom = isnothing(m[3]) || isempty(m[3]) ? 1.0 : parse(Float64, m[3])
                push!(variable_terms, (sign_factor * coef / denom, var))
                continue
            end

            # Case 3: variable*coefficient[/denominator]
            m = match(r"^(\d*\.?\d+)(?:\/(\d+))?\*([a-zA-Z][a-zA-Z0-9_]*)$", term)
            if !isnothing(m)
                coef = parse(Float64, m[1])
                denom = isnothing(m[2]) || isempty(m[2]) ? 1.0 : parse(Float64, m[2])
                var = m[3]
                push!(variable_terms, (sign_factor * coef / denom, var))
                continue
            end

            # Case 4: variable/denominator
            m = match(r"^([a-zA-Z][a-zA-Z0-9_]*)\/(\d+)$", term)
            if !isnothing(m)
                var = m[1]
                denom = parse(Float64, m[2])
                push!(variable_terms, (sign_factor / denom, var))
                continue
            end

            # Case 5: pure variable
            m = match(r"^([a-zA-Z][a-zA-Z0-9_]*)$", term)
            if !isnothing(m)
                push!(variable_terms, (sign_factor, m[1]))
                continue
            end

            if strict
                throw(ArgumentError("Could not parse term: $term"))
            else
                @warn("Could not parse term: $term")
            end
        end

        return variable_terms, constant_value
    end

    # Parse both sides
    lhs_terms, lhs_const = parse_side(lhs)
    rhs_terms, rhs_const = parse_side(rhs)

    # Combine like terms - put all variables on left side
    var_coeffs = Dict{String, Float64}()

    # Add left-side variables
    for (coef, var) ∈ lhs_terms
        var_coeffs[var] = get(var_coeffs, var, 0.0) + coef
    end

    # Subtract right-side variables
    for (coef, var) ∈ rhs_terms
        var_coeffs[var] = get(var_coeffs, var, 0.0) - coef
    end

    # Move constants to right side
    right_const = rhs_const - lhs_const

    # Prepare the output variables and coefficients
    sorted_vars = sort(collect(keys(var_coeffs)))
    vars = String[]
    coefs = Float64[]

    for var ∈ sorted_vars
        coef = var_coeffs[var]
        if !isapprox(coef, 0.0; atol = 1e-10)  # Filter out essentially zero coefficients
            push!(vars, var)
            push!(coefs, coef)
        end
    end

    # Format the equation string
    terms = String[]
    for (i, var) ∈ enumerate(vars)
        coef = coefs[i]

        if isapprox(abs(coef), 1.0; atol = 1e-10)
            term = coef > 0 ? var : "-$var"
        else
            term = coef > 0 ? "$(coef)*$var" : "$(coef)*$var"
        end

        push!(terms, term)
    end

    # Handle empty left side
    left_str = if isempty(terms)
        "0"
    else
        # Join terms with proper spacing and signs
        result = terms[1]
        for i ∈ 2:length(terms)
            if startswith(terms[i], "-")
                result *= " " * terms[i]
            else
                result *= " + " * terms[i]
            end
        end
        result
    end

    # Create the final equation
    equation = "$left_str $comp_op $right_const"

    return Dict("equation" => equation, "variables" => vars, "coefficients" => coefs,
                "comparison" => comp_op, "constant" => right_const)
end

export EQ, LEQ, GEQ, A_LinearConstraint, A_CardinalityConstraint, split_equation,
       parse_constraint_equation
