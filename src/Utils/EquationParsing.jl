#=
const EqualityOperators = ("==", "=")
const InequalityOperators = ("<=", ">=")
const ComparisonOperators = union(EqualityOperators, InequalityOperators)
const AdditionSubtractionOperators = ("+", "-")
const MultiplicationOperators = ("*",)
const NonMultiplicationOperators = union(ComparisonOperators, AdditionSubtractionOperators)
const Operators = union(NonMultiplicationOperators, MultiplicationOperators)
const ComparisonOperatorSigns = Dict(">=" => -1, "<=" => 1, "==" => 1, "=" => 1)
const AdditionSubtractionOperatorSigns = Dict("+" => 1, "-" => -1)
function split_equation_string(eqstr::AbstractString)
    valid_pattern = Regex("(?=$(join([".+\\" * e  for e in ComparisonOperators], fill("|", length(ComparisonOperators)-2)...)))")
    valid_match = match(valid_pattern, eqstr)
    if isnothing(valid_match)
        throw(ArgumentError("No comparison operator found in equation string: $eqstr.\nThe string must contain a comparison operator: $(ComparisonOperators)"))
    end

    invalid_pattern = r"(?<!<)(?<!<=)>(?!=)|(?<!>)<(?!=)"
    invalid_match = match(invalid_pattern, eqstr)
    if !isnothing(invalid_match)
        throw(ArgumentError("Invalid comparison operators found in equation string: $eqstr.\nValid comparison operators are: $(ComparisonOperators)"))
    end

    valid_operators = Regex("((?:$(join(["\\" * e  for e in sort!(Operators, rev=true)], fill("|", length(ComparisonOperators)-2)...))))")

    idx = findall(valid_operators, eqstr)
    split_eq = String[]
    start = 1
    for (j, i) ∈ enumerate(idx)
        if isone(i[1])
            push!(split_eq, strip(eqstr[i]))
        else
            push!(split_eq, strip(eqstr[start:(i[1] - 1)]))
            start = i[1]
            push!(split_eq, strip(eqstr[start:i[end]]))
            if j == length(idx) && i[end] < length(eqstr)
                start = i[end] + 1
                push!(split_eq, strip(eqstr[start:end]))
            end
        end
        start = i[end] + 1
    end
    return [x for x ∈ split_eq if !isempty(x)]
end
function string_to_equation(equation::AbstractString) end
function parse_equations_to_matrix(groups, equations::AbstractVector{<:AbstractString},
                                   normalise::Bool = true;
                                   error_if_missing_group::Bool = false,
                                   names::Tuple = ("names", "equations"),
                                   type::DataType = Float64)
    A_eq = Matrix{DataType}(undef, 0, 0)
    B_eq = Vector{DataType}(undef, 0, 0)
    A_ineq = Matrix{DataType}(undef, 0, 0)
    B_ineq = Vector{DataType}(undef, 0, 0)

    for equation ∈ equations
        try
            A, B, is_ineq = string_to_equation(equation)
            if is_ineq
                A_ineq = vcat(A_ineq, A)
                append!(B_ineq, B)
            else
                A_eq = vcat(A_eq, A)
                append!(B_eq, B)
            end
        catch err
            if error_if_missing_group
                throw(err)
            else
                @warn("$err")
            end
        end
    end

    return A_eq, B_eq, A_ineq, B_ineq
end

export split_equation_string, parse_equations_to_matrix, EqualityOperators,
       InequalityOperators, ComparisonOperators, AdditionSubtractionOperators,
       MultiplicationOperators, NonMultiplicationOperators, Operators,
       ComparisonOperatorSigns, AdditionSubtractionOperatorSigns
       =#
