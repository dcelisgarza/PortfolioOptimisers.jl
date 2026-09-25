"""
    _parse_equation(lhs, opstr::AbstractString, rhs, datatype::DataType = Float64)

Parse and canonicalise a linear constraint equation from Julia expressions.

The function takes both sides of an equation as expressions, and the comparison operator as a string. It evaluates the numeric calls, moves every term to the left-hand side, sums the coefficient of each variable, and returns the equation in canonical form as a [`ParsingResult`](@ref).

# Algorithm

 1. Fold the constant subexpressions of both sides with [`eval_numeric_functions`](@ref), giving `lexpr` and `rexpr`, and check each with [`rethrow_parse_error`](@ref).
 2. Build `diff_expr`, the expression `lexpr - (rexpr)`. This moves every term of the equation to the left-hand side.
 3. Walk `diff_expr` with [`_collect_terms`](@ref), giving `terms`, one `(coefficient, variable)` pair per term.
 4. Accumulate `terms` into `varmap`, which holds the summed coefficient of each variable name, and into `constant`, the sum of the coefficients that carry no variable.
 5. Read `variables` and `coefficients` off `varmap`, and take `rhs_val` as the negated `constant`. This moves the constant to the right-hand side.
 6. Render each pair with [`format_term`](@ref), join the renderings with `+`, and fold `+ -` into `-`, giving the canonical string `formatted`.
 7. Return the [`ParsingResult`](@ref) built from `variables`, `coefficients`, `opstr`, `rhs_val` and `formatted`.

# Arguments

  - `lhs`: Left-hand side of the equation as a Julia expression.
  - `opstr`: Comparison operator as a string.
  - `rhs`: Right-hand side of the equation as a Julia expression.
  - `datatype`: Numeric type for coefficients and right-hand side.

# Returns

  - `res::ParsingResult`: The variables, coefficients, operator, right-hand side and formatted string of the equation in canonical form. The order of `vars` is the order the variable map iterates in, and it is not the order the equation was written in.

# Related

  - [`ParsingResult`](@ref)
  - [`parse_equation`](@ref)
  - [`eval_numeric_functions`](@ref)
  - [`_collect_terms`](@ref)
  - [`format_term`](@ref)
"""
function _parse_equation(lhs, opstr::AbstractString, rhs,
                         datatype::DataType = Float64)::ParsingResult
    # 3. Evaluate numeric functions on both sides
    lexpr = eval_numeric_functions(lhs, datatype)
    rethrow_parse_error(lexpr, :lhs)
    rexpr = eval_numeric_functions(rhs, datatype)
    rethrow_parse_error(rexpr, :rhs)

    # 4. Move all terms to LHS: lhs - rhs == 0
    diff_expr = :($lexpr - ($rexpr))

    # 5. Expand and collect like terms
    terms = _collect_terms(diff_expr, datatype)

    # 6. Separate variables and constant
    varmap = Dict{String, datatype}()
    constant = zero(datatype)
    for (coeff, var) in terms
        if isnothing(var)
            constant += coeff
        else
            varmap[var] = get(varmap, var, zero(datatype)) + coeff
        end
    end

    # 7. Move constant to RHS, variables to LHS
    variables = collect(keys(varmap))
    coefficients = [varmap[v] for v in variables]
    rhs_val = -constant

    # 8. Format the simplified expression
    lhs_str = join([format_term(coeff, var)
                    for (coeff, var) in zip(coefficients, variables)], " + ")
    lhs_str = replace(lhs_str, "+ -" => "-", "  " => " ")
    rhs_str = string(rhs_val)
    formatted = strip("$lhs_str $opstr $rhs_str")
    return ParsingResult(variables, coefficients, opstr, rhs_val, formatted)
end
"""
    parse_equation(eqn::EqnType;
                   ops1::Tuple = ("==", "<=", ">="), ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                   datatype::DataType = Float64, kwargs...)

Parse a linear constraint equation from a string into a structured [`ParsingResult`](@ref).

An equation string can come from outside the program, so both forms of input meet a limit of `EQUATION_LIMITS[]` before any recursive walk. The function caps the length of the string form before `Meta.parse` runs, and no length limit applies to the `Expr` form. It then caps the depth of the expression tree of both forms, so one number bounds the recursion for either form.

Julia parses the equation, so the number syntax of Julia applies. `2e1` is the number `20.0` and `2f1` is the number `20.0f0`, and neither is a coefficient times a variable `e1` or `f1`. Write `2*f1` for a variable whose name starts like an exponent.

# Algorithm

The method that Julia selects is the algorithm, and one method handles each shape of `eqn`.

 1. When `eqn` is a vector, apply this function to each element, and return the vector of results.
 2. When `eqn` is a string, check its length against `EQUATION_LIMITS[].max_length`, and refuse the pattern `++`.
 3. Find the first operator of `ops1` that occurs in the string, giving `opstr`, and split the string on it into `lhs` and `rhs`.
 4. Parse both parts with `Meta.parse`, giving `lexpr` and `rexpr`, check each with [`rethrow_parse_error`](@ref), and check the depth of each against `EQUATION_LIMITS[].max_depth` with [`_expr_depth_exceeds`](@ref).
 5. When `eqn` is an `Expr`, check its depth against `EQUATION_LIMITS[].max_depth` with [`_expr_depth_exceeds`](@ref), and refuse a `++` pattern with [`has_invalid_plus`](@ref).
 6. Check that the head of the expression is a call and is exactly one operator of `ops2`, giving `opstr`, and read `lhs` and `rhs` off the arguments of the call.
 7. Give `opstr` and the two sides to [`_parse_equation`](@ref), which puts them in canonical form and builds the [`ParsingResult`](@ref).

# Arguments

  - `eqn`: The equation string to parse.

      + `eqn::AbstractVector`: Each element needs to meet the criteria below.

      + `eqn::AbstractString`: Must contain exactly one comparison operator from `ops1`.

          * `ops1`: Tuple of valid comparison operators as strings.

      + `eqn::Expr`: Must contain exactly one comparison operator from `ops2`.

          * `ops2`: Tuple of valid comparison operator expressions.

  - `datatype`: The numeric type to use for coefficients and right-hand side.

  - `kwargs...`: Additional keyword arguments, ignored.

# Validation

  - `length(eqn) <= EQUATION_LIMITS[].max_length`, for the string form. A `Meta.ParseError` naming both lengths is thrown otherwise.
  - The expression tree of `eqn` is no deeper than `EQUATION_LIMITS[].max_depth`, for both forms. The string form is checked after `Meta.parse`, on each side of the operator. A `Meta.ParseError` naming the limit is thrown otherwise.
  - `eqn` holds no `++` pattern.
  - `eqn` holds exactly one comparison operator, from `ops1` for the string form and from `ops2` for the `Expr` form.
  - The head of the `Expr` form is a call.
  - Neither side of the equation is empty or incomplete, which [`rethrow_parse_error`](@ref) checks.

# Returns

  - If `eqn::Str_Expr`:

      + `res::ParsingResult`: Structured parsing result.

  - If `eqn::AbstractVector`:

      + `res::Vector{ParsingResult}`: Vector of structured parsing results.

# Examples

```jldoctest
julia> parse_equation(\"w_A + 2w_B <= 1\")
ParsingResult
  vars ┼ Vector{String}: ["w_A", "w_B"]
  coef ┼ Vector{Float64}: [1.0, 2.0]
    op ┼ String: "<="
   rhs ┼ Float64: 1.0
   eqn ┴ SubString{String}: "w_A + 2.0*w_B <= 1.0"
```

# Related

  - [`ParsingResult`](@ref)
  - [`_parse_equation`](@ref)
  - [`rethrow_parse_error`](@ref)
  - [`has_invalid_plus`](@ref)
  - [`_expr_depth_exceeds`](@ref)
  - [`replace_group_by_assets`](@ref)
"""
function parse_equation(eqn::AbstractString; ops1::Tuple = ("==", "<=", ">="),
                        datatype::DataType = Float64, kwargs...)::ParsingResult
    # Trust boundary: cap the untrusted string length before `Meta.parse`, so a deeply
    # nested string cannot exhaust the stack. The length bounds the achievable AST depth
    # at about half the character count, which is looser than `max_depth`, so the parsed
    # tree meets the depth cap directly, as the `Expr` form does.
    lim = EQUATION_LIMITS[]
    @argcheck(length(eqn) <= lim.max_length,
              Meta.ParseError("Equation string is too long ($(length(eqn)) > $(lim.max_length) characters)."))
    @argcheck(!occursin("++", eqn),
              Meta.ParseError("Invalid operator '++' detected in equation."))
    # 1. Identify the comparison operator
    op = findfirst(op -> occursin(op, eqn), ops1)
    @argcheck(!isnothing(op),
              Meta.ParseError("Equation must contain a valid comparison operator $(join(ops1,", ")) .\n$(eqn)"))
    opstr = ops1[op]
    parts = split(eqn, opstr)
    @argcheck(length(parts) == 2,
              Meta.ParseError("Equation must have exactly one comparison operator.\n$(eqn)"))
    lhs, rhs = strip.(parts)
    # 2. Parse both sides into Julia expressions
    lexpr = Meta.parse(lhs)
    rethrow_parse_error(lexpr, :lhs)
    rexpr = Meta.parse(rhs)
    rethrow_parse_error(rexpr, :rhs)
    # 3. Hold the parsed tree to the depth cap, before the recursive walks below run.
    @argcheck(!(_expr_depth_exceeds(lexpr, lim.max_depth) ||
                _expr_depth_exceeds(rexpr, lim.max_depth)),
              Meta.ParseError("Equation expression is too deeply nested (exceeds depth $(lim.max_depth))."))
    return _parse_equation(lexpr, opstr, rexpr, datatype)
end
"""
    has_invalid_plus(expr)

Check whether a Julia expression contains the operator `++` anywhere in its tree.

It is the `Expr` counterpart of the `++` check that the string form of [`parse_equation`](@ref) runs on the raw text.

# Algorithm

 1. Return `false` when `expr` is not a call, because only a call can carry the head this function refuses.
 2. Return `true` when the head of the call is the `++` operator.
 3. Apply this function to every argument of the call that is itself an expression, and return `true` when any of them does.

# Arguments

  - `expr`: Julia expression to check.

# Returns

  - `Bool`: `true` if the tree holds a `++` call, `false` otherwise.

# Related

  - [`parse_equation`](@ref)
  - [`_expr_depth_exceeds`](@ref)
"""
function has_invalid_plus(expr)::Bool
    if !(isa(expr, Expr) && expr.head == :call)
        return false
    end
    # A `++` call anywhere in the tree is refused, as the string form refuses the text.
    if expr.args[1] == :++
        return true
    end
    # Recurse into sub-expressions
    return any(has_invalid_plus(arg) for arg in expr.args[2:end] if isa(arg, Expr))
end
"""
    _expr_depth_exceeds(x, limit::Integer) -> Bool

Return `true` if the expression tree `x` is deeper than `limit`.

It protects the `Expr` form of [`parse_equation`](@ref) from a deeply nested tree that no
string length limit covers. The depth counts the levels of `Expr` along the deepest path, so
`:(a + b)` has depth one. The check recurses at most `limit + 1` frames deep and stops at the
first branch that passes the limit, so it cannot use up the stack that it protects.

# Algorithm

 1. Return `true` when `limit` is negative, because the walk has already gone one level past the cap.
 2. Return `false` when `x` is not an expression, because a leaf adds no depth.
 3. Apply this function to every argument of `x`, with `limit` lowered by one, and return `true` when any of them does. The scan stops at the first argument that answers `true`.

# Arguments

  - `x`: The expression tree to measure.
  - `limit`: The greatest depth the tree may have.

# Returns

  - `Bool`: `true` when the tree is deeper than `limit`, `false` otherwise.

# Related

  - [`parse_equation`](@ref)
  - [`has_invalid_plus`](@ref)
"""
function _expr_depth_exceeds(x, limit::Integer)::Bool
    if limit < 0
        return true
    end
    return isa(x, Expr) && any(_expr_depth_exceeds(a, limit - 1) for a in x.args)
end
function parse_equation(expr::Expr; ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                        datatype::DataType = Float64, kwargs...)::ParsingResult
    # Trust-boundary defence for the pre-built-AST form (no string length cap applies):
    # reject an over-deep tree before the recursive walks below can exhaust the stack.
    lim = EQUATION_LIMITS[]
    @argcheck(!_expr_depth_exceeds(expr, lim.max_depth),
              Meta.ParseError("Equation expression is too deeply nested (exceeds depth $(lim.max_depth))."))
    # Recursively check for invalid "++" pattern in the expression tree
    @argcheck(!has_invalid_plus(expr),
              Meta.ParseError("Invalid operator pattern '++' detected in equation expression:\n$expr"))
    # Ensure the expression is a call to a valid comparison operator
    @argcheck(expr.head == :call,
              Meta.ParseError("Expression must be a function call (comparison operator expected):\n$expr"))
    # Count how many valid operators are present
    op_count = count(op -> expr.args[1] == op, Iterators.drop(ops2, 1))
    @argcheck(op_count == 1,
              Meta.ParseError("Expression must contain a valid comparison operator $(join(ops2[2:end], ", ")) .\n$expr"))
    opstr = string(expr.args[1])
    lhs, rhs = expr.args[2], expr.args[3]
    return _parse_equation(lhs, opstr, rhs, datatype)
end
function parse_equation(eqn::VecStr_Expr; ops1::Tuple = ("==", "<=", ">="),
                        ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                        datatype::DataType = Float64)
    return parse_equation.(eqn; ops1 = ops1, ops2 = ops2, datatype = datatype)
end
"""
    replace_group_by_assets(res::PR_VecPR, sets::UniverseSets, bl_flag::Bool = false,
                            ep_flag::Bool = false, rho_flag::Bool = false;
                            ledger::Option{<:AbstractVector} = nothing)

Expand group or special variable references in a [`ParsingResult`](@ref) to their corresponding asset names.

A variable name of `res` can be a group name, a `prior(...)` expression or a correlation view such as `(A, B)`. The function replaces each of them with the asset names that the [`UniverseSets`](@ref) holds. It gives the Black-Litterman mean expansion, the `prior(...)` views of entropy pooling and the correlation views. A vector of results gives a vector, one result per element.

# Mathematical definition

```math
\\begin{align}
c\\, g &\\to \\sum_{j=1}^{k} c\\, m_j\\,, \\\\
c\\, g &\\to \\sum_{j=1}^{k} \\frac{c}{k}\\, m_j\\,.
\\end{align}
```

Where:

  - ``g``: A group name written in the equation.
  - ``m_j``: The ``j``-th member of the group ``g``.
  - ``k``: The number of members of the group ``g``.
  - ``c``: The coefficient the group name carries.

The two lines are different operations. The first repeats the coefficient on every member, so the expanded row constrains the sum over the group. The second divides the coefficient by the member count, so the expanded row constrains the mean over the group. A group of one member is the only case in which the two agree.

# Algorithm

 1. Copy `res.vars` and `res.coef` into `variables_new` and `coeffs_new`, and open the empty accumulators `variables_tmp`, `coeffs_tmp` and `idx_rm`.
 2. For each variable name of `res.vars`, match it against the prior pattern `prior(...)` and against the correlation pattern `(a, b)`. The four combinations of the two matches select steps 3 to 6.
 3. A name matching neither pattern, with `rho_flag` false, is a plain name. Look it up in `sets.dict`, and leave it where it stands when the dictionary does not hold it, because a name that is not a group is already the name of one column. A group name sheds its departed members with [`shed_departed_members`](@ref), then expands to what survived, each member carrying the coefficient the mathematics above gives over the surviving count, and the index of the group joins `idx_rm`. A group that shed every member keeps its first member, so the row names a departed asset, and [`get_linear_constraints`](@ref) drops that row in silence. A group that holds no member expands to nothing, and its index joins `idx_rm` all the same.
 4. A name matching the correlation pattern expands to one entry naming the two member lists, and that entry carries the coefficient of the view unchanged. A correlation view is one row over a pair of universes, so no coefficient is spread over members. The two lists shed jointly, so a pair survives only when both of its names did.
 5. A name matching the prior pattern expands the name inside `prior(...)` exactly as step 3 does, and wraps each member back in `prior(...)`.
 6. A name matching both patterns expands as step 4 does, and wraps each of the two member lists in `prior(...)`.
 7. Return `res` unchanged when nothing was struck, so an equation written in asset names costs no allocation.
 8. Delete the entries at `idx_rm` from `variables_new` and `coeffs_new`, append the two accumulators to them, and render the expanded equation string.
 9. Return the [`ParsingResult`](@ref) built from the new names and coefficients, together with the operator and the right-hand side of `res`, which the expansion leaves untouched.

# Arguments

  - `res`: A [`ParsingResult`](@ref) object containing variables and coefficients to be expanded.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `bl_flag`: Selects which of the two expansions above runs. `false` takes the first, which constrains the sum over the group. `true` takes the second, the Black-Litterman-style expansion, which constrains the mean.
  - `ep_flag`: If `true`, the function expands `prior(...)` expressions, for entropy pooling.
  - `rho_flag`: If `true`, the function expands correlation views `(A, B)`, for entropy pooling.
  - `ledger`: The door's ledger of departure casualties, or `nothing` when nobody is collecting. A shed group is recorded into it through [`record_group_shed!`](@ref).

# Validation

The three flags are not independent, and five guards hold the grammar they describe.

  - `bl_flag` can only be `true` if both `ep_flag` and `rho_flag` are `false`.
  - `rho_flag` can only be `true` if `ep_flag` is also `true`.
  - The pattern `(a, b)` can only be used when `ep_flag` and `rho_flag` are both `true`.
  - The pattern `prior(a)` can only be used when `ep_flag` is `true`.
  - The pattern `prior(a, b)` can only be used when `rho_flag` is `true`.

Two further guards hold the shape of a correlation view.

  - A correlation view is written `(a, b)`, and a correlation view prior is written `prior(a, b)`.
  - Both sides of a correlation view name a group that `sets.dict` holds, and the two groups have the same number of members. A view whose two sides are both absent from `sets.dict` is skipped instead of raised on.

# Returns

  - `res::ParsingResult`: A new [`ParsingResult`](@ref) with all group and special variable references expanded to asset names.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\",
                           dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"], \"group1\" => [\"A\", \"B\"]));

julia> res = parse_equation(\"group1 + 2C == 1\")
ParsingResult
  vars ┼ Vector{String}: ["group1", "C"]
  coef ┼ Vector{Float64}: [1.0, 2.0]
    op ┼ String: "=="
   rhs ┼ Float64: 1.0
   eqn ┴ SubString{String}: "group1 + 2.0*C == 1.0"

julia> replace_group_by_assets(res, sets)
ParsingResult
  vars ┼ Vector{String}: ["C", "A", "B"]
  coef ┼ Vector{Float64}: [2.0, 1.0, 1.0]
    op ┼ String: "=="
   rhs ┼ Float64: 1.0
   eqn ┴ String: "2.0*C + A + B == 1.0"
```

# Related

  - [`UniverseSets`](@ref)
  - [`ParsingResult`](@ref)
  - [`parse_equation`](@ref)
  - [`get_linear_constraints`](@ref)
  - [`linear_constraints`](@ref)
  - [`shed_departed_members`](@ref)
  - [`record_group_shed!`](@ref)
"""
function replace_group_by_assets(res::ParsingResult, sets::UniverseSets,
                                 bl_flag::Bool = false, ep_flag::Bool = false,
                                 rho_flag::Bool = false;
                                 ledger::Option{<:AbstractVector} = nothing)::ParsingResult
    @argcheck(!(bl_flag && (rho_flag || ep_flag)),
              ArgumentError("bl_flag can only be true if ep_flag and rho_flag are false. Got\nbl_flag => $(bl_flag)\nep_flag => $(ep_flag)\nrho_flag => $(rho_flag)."))
    @argcheck(!(rho_flag && !ep_flag),
              ArgumentError("rho_flag can only be true if ep_flag is also true. Got\nrho_flag => $rho_flag\nep_flag => $ep_flag"))
    # A group is a description the data resolves, so it sheds the members that left the
    # Investable Mask *before* the coefficient is spread: the mean then divides by the
    # surviving count, and the row still computes what its right-hand side asserts. A name
    # written out in the equation is the caller pointing at one asset, and that one takes
    # its row with it, one door later. See ADR 0125.
    other = counterpart_axis_names(sets, sets.xkey)
    variables, coeffs = res.vars, res.coef
    variables_new, coeffs_new = copy(variables), copy(coeffs)
    variables_tmp = Vector{eltype(variables)}(undef, 0)
    coeffs_tmp = Vector{eltype(coeffs)}(undef, 0)
    idx_rm = Vector{Int}(undef, 0)
    prior_pattern = r"prior\(([^()]*)\)"
    corr_pattern = r"\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"
    for (i, (v, c)) in enumerate(zip(variables, coeffs))
        m = match(prior_pattern, v)
        if isnothing(m)
            n = match(corr_pattern, v)
            if isnothing(n) && !rho_flag
                asset = get(sets.dict, v, nothing)
                if isnothing(asset)
                    continue
                end
                asset = shed_departed_members(asset, other, ledger, v, res.eqn)
                push!(idx_rm, i)
                c = !bl_flag ? coeffs[i] : coeffs[i] / length(asset)
                append!(variables_tmp, asset)
                append!(coeffs_tmp, Iterators.repeated(c, length(asset)))
            else
                @argcheck(ep_flag && rho_flag,
                          ArgumentError("The pattern '(a, b)' can only be used for rho_views (rho_flag is true) in entropy pooling (ep_flag is true). Got\nep_flag => $(ep_flag)\nrho_flag => $(rho_flag)."))
                @argcheck(!isnothing(n),
                          ArgumentError("Correlation views can only be of the form '(a, b)'. Got\nv => $v"))
                asset1 = n.captures[1]
                asset2 = n.captures[2]
                asset1 = get(sets.dict, asset1, nothing)
                asset2 = get(sets.dict, asset2, nothing)
                if isnothing(asset1) && isnothing(asset2)
                    continue
                end
                @argcheck(!isnothing(asset1), IsNothingError)
                @argcheck(!isnothing(asset2), IsNothingError)
                @argcheck(length(asset1) == length(asset2), DimensionMismatch)
                asset1, asset2 = shed_departed_members(asset1, asset2, other, ledger,
                                                       "($(n.captures[1]), $(n.captures[2]))",
                                                       res.eqn)
                push!(idx_rm, i)
                push!(variables_tmp, "([$(join(asset1, ", "))], [$(join(asset2, ", "))])")
                push!(coeffs_tmp, coeffs[i])
            end
        else
            @argcheck(ep_flag,
                      ArgumentError("The pattern 'prior(a)' can only be used in entropy pooling (ep_flag is true). Got\nep_flag => $(ep_flag)."))
            n = match(corr_pattern, v)
            if isnothing(n) && !rho_flag
                grp = @view v[7:(end - 1)]
                asset = get(sets.dict, grp, nothing)
                if isnothing(asset)
                    continue
                end
                asset = shed_departed_members(asset, other, ledger, grp, res.eqn)
                push!(idx_rm, i)
                c = !bl_flag ? coeffs[i] : coeffs[i] / length(asset)
                append!(variables_tmp, ["prior($a)" for a in asset])
                append!(coeffs_tmp, Iterators.repeated(c, length(asset)))
            else
                @argcheck(rho_flag,
                          ArgumentError("The pattern 'prior(a, b)' can only be used for rho_views (rho_flag is true) in entropy pooling (ep_flag is true). Got\nep_flag => $(ep_flag)\nrho_flag => $(rho_flag)."))
                @argcheck(!isnothing(n),
                          ArgumentError("Correlation views prior can only be of the form 'prior(a, b)'. Got\nv => $v"))
                asset1 = n.captures[1]
                asset2 = n.captures[2]
                asset1 = get(sets.dict, asset1, nothing)
                asset2 = get(sets.dict, asset2, nothing)
                if isnothing(asset1) && isnothing(asset2)
                    continue
                end
                @argcheck(!isnothing(asset1), IsNothingError)
                @argcheck(!isnothing(asset2), IsNothingError)
                @argcheck(length(asset1) == length(asset2), DimensionMismatch)
                asset1, asset2 = shed_departed_members(asset1, asset2, other, ledger,
                                                       "prior($(n.captures[1]), $(n.captures[2]))",
                                                       res.eqn)
                push!(idx_rm, i)
                push!(variables_tmp,
                      "prior([$(join(asset1, ", "))], [$(join(asset2, ", "))])")
                push!(coeffs_tmp, coeffs[i])
            end
        end
    end
    # `idx_rm`, not `variables_tmp`: a group that holds no member expands to nothing
    # and leaves the second accumulator empty, and returning `res` there would put the
    # group name back into a row that no longer names anything.
    if isempty(idx_rm)
        return res
    end
    deleteat!(variables_new, idx_rm)
    deleteat!(coeffs_new, idx_rm)
    append!(variables_new, variables_tmp)
    append!(coeffs_new, coeffs_tmp)
    # Render through the same `format_term` the unexpanded string uses, so one constraint
    # prints one way whether or not a group expanded.
    eqn = replace(join(format_term.(coeffs_new, variables_new), " + "), "+ -" => "-",
                  "  " => " ")
    return ParsingResult(variables_new, coeffs_new, res.op, res.rhs,
                         "$(eqn) $(res.op) $(res.rhs)")
end
function replace_group_by_assets(res::VecPR, sets::UniverseSets, args...; kwargs...)
    return [replace_group_by_assets(resi, sets, args...; kwargs...) for resi in res]
end
"""
    universe_axis(sets::UniverseSets, key::AbstractString) -> String

Return the name of the axis of the universe under `key`, read from the key itself.

A key that starts with `tfkey` or `cfkey` gives `"factor"`, and any other key gives `"asset"`. [`unknown_variable_msg`](@ref) and [`empty_row_msg`](@ref) read it to name the axis that the user wrote in.

Both callers resolve names against `sets.dict[key]` alone, so the axis of that universe is the axis where a lookup failed, and the key is the right source. [`get_black_litterman_views`](@ref) takes the key from the estimator that owns the views, and [`get_linear_constraints`](@ref) takes it from the constraint space. [`FactorSpace`](@ref) resolves at the factor axis that [`factor_axis_key`](@ref) reads from the loadings, `sets.tfkey` for the time-series family and `sets.cfkey` for the cross-sectional family. The re-basis is the wrong source. A wrapped estimator with its own `key` replaces the key of the space, so a re-based row can resolve against a universe that the loadings do not use, and the message must name the universe that the lookup searched.

The test is on the prefix and not on equality, so a factor group key such as `"nf_sector"` or `"ncf_sector"` also gives the factor axis. [`UniverseSets`](@ref) refuses a key prefix that starts with another, so the answer is unique.

# Algorithm

 1. Return `"factor"` when `key` starts with `sets.tfkey` or with `sets.cfkey`.
 2. Return `"asset"` in every other case.

# Arguments

  - `sets`: The [`UniverseSets`](@ref) whose `tfkey` and `cfkey` name the two factor axes.
  - `key`: The key the names were resolved against.

# Returns

  - `axis::String`: `"factor"` or `"asset"`, the word a diagnostic message uses to name the axis.

# Related

  - [`get_linear_constraints`](@ref)
  - [`get_black_litterman_views`](@ref)
  - [`UniverseSets`](@ref)
  - [`factor_axis_key`](@ref)
"""
function universe_axis(sets::UniverseSets, key::AbstractString)::String
    return startswith(key, sets.tfkey) || startswith(key, sets.cfkey) ? "factor" : "asset"
end
"""
    constraint_row_length(rr, nx::VecStr) -> Int

Return the length of an assembled constraint row.

Without a re-basis, the length is the size of the universe that the names resolve against. With a re-basis, it is the number of assets that the loadings project onto, because the assembly of the row applies the projection and gives an ordinary asset-space row.

# Algorithm

The method that Julia selects is the algorithm, and the re-basis selects it.

 1. When `rr` is `nothing`, return the length of `nx`, the universe the names resolve against.
 2. When `rr` is a regression result, return the number of rows of `rr.M`, which is the number of assets the loadings project onto.

# Arguments

  - `rr`: Loadings to re-base through, or `nothing` for an ordinary asset-space row.
  - `nx`: The universe the names resolve against.

# Returns

  - `N::Int`: The number of entries one assembled row has.

# Related

  - [`get_linear_constraints`](@ref)
  - [`constraint_row_term`](@ref)
"""
function constraint_row_length(::Nothing, nx::VecStr)::Int
    return length(nx)
end
function constraint_row_length(rr::AbstractLoadingsRegressionResult, ::VecStr)::Int
    return size(rr.M, 1)
end
"""
    constraint_row_term(::Nothing, Ai, c)
    constraint_row_term(rr::AbstractLoadingsRegressionResult, Ai, c)

Return the contribution of one matched variable to a constraint row.

Without a re-basis, the contribution is the indicator `Ai` times the coefficient `c`. With a re-basis, it is the sum of the columns of the loadings that `Ai` selects, times `c`. The function sums the columns and does not take only the first match, so a duplicated name in a factor universe contributes every column with that name, as a duplicated asset name does on the asset path.

The function reads `rr.M` and never `rr.L`. The columns of `M` are the named original factors, and a constraint must use names that a user can write in an equation. Risk decomposition reads `L` instead.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{a}^\\intercal \\boldsymbol{w}_f &= \\boldsymbol{a}^\\intercal \\mathbf{M}^\\intercal \\boldsymbol{w}_a = (\\mathbf{M} \\boldsymbol{a})^\\intercal \\boldsymbol{w}_a\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{a}``: A constraint row written in factor names.
  - ``\\boldsymbol{w}_f``: The factor weights that row is written against.
  - ``\\boldsymbol{w}_a``: The asset weights the optimiser holds.
  - ``\\mathbf{M}``: The factor loadings, one column per named factor and one row per asset.

The identity lets a row written in factor names bind the asset weights with no change of variables. The re-based row is an ordinary asset-space row over ``\\boldsymbol{w}_a``.

# Algorithm

The method that Julia selects is the algorithm, and the re-basis selects it.

 1. When `rr` is `nothing`, return `Ai` times `c`, one entry per name of the universe.
 2. When `rr` is a regression result, sum the columns of `rr.M` that `Ai` selects, multiply the sum by `c`, and return it. The value has one entry per asset, whatever names the row uses.

# Arguments

  - `rr`: Loadings to re-base through, or `nothing` for an ordinary asset-space row.
  - `Ai`: The indicator of the matched name over the universe the names resolve against.
  - `c`: The coefficient the matched name carries.

# Returns

  - The contribution of this term to the row, of the length [`constraint_row_length`](@ref) gives.

# Related

  - [`get_linear_constraints`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`constraint_row_length`](@ref)
"""
function constraint_row_term(::Nothing, Ai, c)
    return Ai * c
end
function constraint_row_term(rr::AbstractLoadingsRegressionResult, Ai, c)
    return vec(sum(view(rr.M, :, Ai); dims = 2)) * c
end
"""
    get_linear_constraints(lcs::PR_VecPR, sets::UniverseSets,
                           key::Option{<:AbstractString} = nothing;
                           datatype::DataType = Float64, strict::Bool = false,
                           rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                           ledger::Option{<:AbstractVector} = nothing)

Convert parsed linear constraint equations into a `LinearConstraint` object.

The function resolves the variable names of one or more [`ParsingResult`](@ref)s, as [`parse_equation`](@ref) returns them, against the universe of the [`UniverseSets`](@ref). It assembles the coefficient matrices and the right-hand sides, and returns a [`LinearConstraint`](@ref) that holds the equality rows and the inequality rows.

A row takes one of two shapes. Without `rr`, it runs over the universe that the names resolve against. With `rr`, it runs over the assets, because the loadings re-base each term during the assembly and the result is an ordinary asset-space row.

The function drops whole rows and never single terms. A row is one statement over several names with one right-hand side, so a name that the function cannot resolve drops the whole row. `a + c == 0.05` without `c` fits `a == 0.05`, which is a different and stronger statement than the caller wrote. The cause of the failure decides only whether the function reports the drop. The function drops a row that names the counterpart axis in silence, whatever `strict` is. [`counterpart_axis_names`](@ref) reads that axis, which in practice is the Non-Investable Axis that a door wrote. The name was correct over the universe that the caller had, and the data moved it, so the door reports the departure once. A name on neither axis is a typo, and the function reports it.

# Algorithm

 1. Take `k` as `key`, or `sets.xkey` when `key` is `nothing`, read the universe `nx` from `sets.dict` under it, name the axis with [`universe_axis`](@ref), and read the counterpart axis with [`counterpart_axis_names`](@ref).
 2. Take `N`, the row length, from [`constraint_row_length`](@ref), and allocate the working row `At` of that length.
 3. Zero `At` for each parsing result, and start that result not dropped.
 4. Build the indicator of each variable name of the result over `nx`. A name that matches no entry marks the row dropped, and is reported through [`strict_diagnostic`](@ref) unless it names the counterpart axis. Every name of the row is still visited, so a row carrying two typos names both.
 5. Add the contribution that [`constraint_row_term`](@ref) gives for the name and its coefficient to `At`. With `rr` the contribution is already projected, so `At` has one entry per asset.
 6. Move to the next result when the row was marked dropped.
 7. Report the row through [`strict_diagnostic`](@ref) and drop it when `At` is still zero. Every name resolved before this step, so the message says that the row sums to zero, because of the loadings under `rr`, because its coefficients cancel, or because it holds no name. It never says that a name has a typo.
 8. Read the sign and the inequality flag of the operator from [`comparison_sign_ineq_flag`](@ref), and scale the row and its right-hand side by the sign. That negates a `>=` row, so both senses of an inequality are written in the `<=` sense, which is the convention [`LinearConstraint`](@ref) states.
 9. Append the row to the inequality accumulator when the flag is `true`, and to the equality accumulator when it is `false`.
10. Reshape each accumulator that holds a row into a matrix of `N` columns, and build the [`PartialLinearConstraint`](@ref) of that half.
11. Return the [`LinearConstraint`](@ref) holding the halves that were built, or `nothing` when neither half holds a row.

# Arguments

  - `lcs`: A single [`ParsingResult`](@ref) or a vector of such objects, representing parsed constraint equations.
  - `sets`: A [`UniverseSets`](@ref) object specifying the universes and groupings.
  - `key`: Key naming the universe the variables resolve against. Defaults to `sets.xkey`; a re-based constraint passes the factor axis key that [`factor_axis_key`](@ref) reads off the loadings.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, a variable name that the universe does not hold throws. If `false`, it warns.
  - `rr`: Loadings to re-base through, or `nothing` for an ordinary asset-space constraint. Callers do not pass it directly, and [`ExposureConstraintEstimator`](@ref) passes it.
  - `ledger`: The door's ledger of departure casualties, or `nothing` when nobody is collecting. A row dropped for a name on the counterpart axis is recorded into it through [`record_non_investable_drop!`](@ref).

# Validation

  - `lcs` is non-empty, when it is a vector.
  - A variable name that matches no entry of the universe and none of the counterpart axis raises when `strict` is `true`, and issues a warning otherwise. The row is dropped either way.
  - A variable name on the counterpart axis drops its row in silence, under both settings of `strict`.
  - A row whose terms all fall away raises when `strict` is `true`, and issues a warning otherwise. The row is dropped either way.
  - Each `op` is one of `"=="`, `"<="` or `">="`, which [`comparison_sign_ineq_flag`](@ref) enforces.

# Returns

  - `lcs::LinearConstraint`: An object containing the assembled equality and inequality constraints, or `nothing` if no constraints are present.

# Related

  - [`ParsingResult`](@ref)
  - [`LinearConstraint`](@ref)
  - [`parse_equation`](@ref)
  - [`replace_group_by_assets`](@ref)
  - [`constraint_row_term`](@ref)
  - [`constraint_row_length`](@ref)
  - [`universe_axis`](@ref)
  - [`comparison_sign_ineq_flag`](@ref)
  - [`counterpart_axis_names`](@ref)
  - [`shed_departed_members`](@ref)
"""
function get_linear_constraints(lcs::PR_VecPR, sets::UniverseSets,
                                key::Option{<:AbstractString} = nothing;
                                datatype::DataType = Float64, strict::Bool = false,
                                rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                                ledger::Option{<:AbstractVector} = nothing)
    if isa(lcs, AbstractVector)
        @argcheck(!isempty(lcs), IsEmptyError)
    end
    A_ineq = Vector{datatype}(undef, 0)
    B_ineq = Vector{datatype}(undef, 0)
    A_eq = Vector{datatype}(undef, 0)
    B_eq = Vector{datatype}(undef, 0)
    k = ifelse(isnothing(key), sets.xkey, key)
    nx = sets.dict[k]
    axis = universe_axis(sets, k)
    other = counterpart_axis_names(sets, k)
    N = constraint_row_length(rr, nx)
    At = Vector{datatype}(undef, N)
    for lc in lcs
        fill!(At, zero(eltype(At)))
        dropped = false
        for (v, c) in zip(lc.vars, lc.coef)
            Ai = (nx .== v)
            if !any(isone, Ai)
                # A view is a **row**: `a + c == 0.05` fitted without `c` asserts
                # `a == 0.05`, which is a different and stronger claim than the caller
                # wrote. So the row goes whole, whatever the name's failure was. A name on
                # the counterpart axis goes in silence — it was correct over the universe
                # the caller was handed, and the data moved it — and a name on neither axis
                # is still a typo and is still reported. See ADR 0125.
                if v ∉ other
                    msg = unknown_variable_msg(v, nx, k; axis = axis,
                                               consequence = "row dropped")
                    strict_diagnostic(msg, strict)
                elseif !dropped
                    # Once per row, not once per departed name in it: the row is the unit
                    # that went, and `ni` already names every asset that left.
                    record_non_investable_drop!(ledger, "the row `$(lc.eqn)`")
                end
                dropped = true
                continue
            end
            At .+= constraint_row_term(rr, Ai, c)
        end
        if dropped
            continue
        end
        if !any(!iszero, At)
            # Every name of the row resolved — one that did not took the row with it above —
            # so what is left here is a row that resolved and still summed to zero: the
            # loadings annihilated it, or its own coefficients cancelled. Reporting a typo
            # for either would send a user hunting for one that is not there.
            msg = if !isnothing(rr)
                empty_projected_row_msg(lc.eqn, nx, k, N)
            else
                empty_row_msg(lc.eqn, nx, k; axis = axis)
            end
            strict_diagnostic(msg, strict)
            continue
        end
        d, flag = comparison_sign_ineq_flag(lc.op)
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
        A_ineq = transpose(reshape(A_ineq, N, :))
        ineq = PartialLinearConstraint(; A = A_ineq, B = B_ineq)
    end
    if eq_flag
        A_eq = transpose(reshape(A_eq, N, :))
        eq = PartialLinearConstraint(; A = A_eq, B = B_eq)
    end
    return if ineq_flag || eq_flag
        LinearConstraint(; ineq = ineq, eq = eq)
    else
        nothing
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the linear constraint equations to parse, and the universe key their names resolve against.

[`linear_constraints`](@ref) parses `val` and assembles the coefficient matrices of a [`LinearConstraint`](@ref) from it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LinearConstraintEstimator(;
        val::EqnType,
        key::Option{<:AbstractString} = nothing
    ) -> LinearConstraintEstimator

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(val)`, when `val` is a string or a vector.
  - `!isempty(key)`, when `key` is not `nothing`.

# Examples

```jldoctest
julia> lce = LinearConstraintEstimator(; val = [\"w_A + w_B == 1\", \"w_A >= 0.1\"]);

julia> sets = UniverseSets(; xkey = \"nx\", dict = Dict(\"nx\" => [\"w_A\", \"w_B\"]));

julia> linear_constraints(lce, sets)
LinearConstraint
  ineq ┼ PartialLinearConstraint
       │   A ┼ 1×2 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [-0.1]
    eq ┼ PartialLinearConstraint
       │   A ┼ 1×2 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [1.0]
```

# Related

  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`parse_equation`](@ref)
  - [`linear_constraints`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.1.
"""
@concrete struct LinearConstraintEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:lce_val])
    """
    val
    """
    $(field_dict[:ekey])
    """
    key
    function LinearConstraintEstimator(val::EqnType,
                                       key::Option{<:AbstractString} = nothing)::LinearConstraintEstimator
        if isa(val, Str_Vec)
            @argcheck(!isempty(val), IsEmptyError("val cannot be empty"))
        end
        if !isnothing(key)
            @argcheck(!isempty(key), IsEmptyError("key cannot be empty"))
        end
        return new{typeof(val), typeof(key)}(val, key)
    end
end
function LinearConstraintEstimator(; val::EqnType,
                                   key::Option{<:AbstractString} = nothing)::LinearConstraintEstimator
    return LinearConstraintEstimator(val, key)
end
"""
    const LcE_Lc = Union{<:LinearConstraintEstimator, <:LinearConstraint}

An unparsed [`LinearConstraintEstimator`](@ref), or an assembled [`LinearConstraint`](@ref). The group exists because a constraint slot accepts both. [`linear_constraints`](@ref) parses the first and returns the second unchanged, so a caller can give equations or a block that it built earlier.

# Related

  - [`LinearConstraintEstimator`](@ref)
  - [`LinearConstraint`](@ref)
  - [`linear_constraints`](@ref)
"""
const LcE_Lc = Union{<:LinearConstraintEstimator, <:LinearConstraint}
"""
    const VecLcE_Lc = AbstractVector{<:LcE_Lc}

Every abstract vector whose elements are [`LcE_Lc`](@ref)s. The group exists so that one slot may hold a mixed list of equations still to parse and constraints already assembled.

# Related

  - [`LcE_Lc`](@ref)
  - [`LcE_Lc_VecLcE_Lc`](@ref)
  - [`linear_constraints`](@ref)
"""
const VecLcE_Lc = AbstractVector{<:LcE_Lc}
"""
    const VecLcE = AbstractVector{<:LinearConstraintEstimator}

Every abstract vector whose elements are [`LinearConstraintEstimator`](@ref)s. The group is narrower than [`VecLcE_Lc`](@ref), because every element must still be parsed. [`linear_constraints`](@ref) maps over it and gives one constraint per element.

# Related

  - [`LinearConstraintEstimator`](@ref)
  - [`LcE_VecLcE`](@ref)
  - [`VecLcE_Lc`](@ref)
  - [`linear_constraints`](@ref)
"""
const VecLcE = AbstractVector{<:LinearConstraintEstimator}
"""
    const LcE_Lc_VecLcE_Lc = Union{<:LcE_Lc, <:VecLcE_Lc}

One [`LcE_Lc`](@ref), or a vector of them. It is the widest linear-constraint group that the library declares. It names every shape that a user can write into a linear-constraint field, so the type bound of that field uses it.

# Related

  - [`LcE_Lc`](@ref)
  - [`VecLcE_Lc`](@ref)
  - [`linear_constraints`](@ref)
"""
const LcE_Lc_VecLcE_Lc = Union{<:LcE_Lc, <:VecLcE_Lc}
"""
    const LcE_VecLcE = Union{<:LinearConstraintEstimator, <:VecLcE}

One [`LinearConstraintEstimator`](@ref), or a vector of them. The group excludes an assembled [`LinearConstraint`](@ref), so a method that dispatches on it knows that every element still carries equations to parse.

# Related

  - [`LinearConstraintEstimator`](@ref)
  - [`VecLcE`](@ref)
  - [`LcE_Lc`](@ref)
  - [`linear_constraints`](@ref)
"""
const LcE_VecLcE = Union{<:LinearConstraintEstimator, <:VecLcE}
"""
    linear_constraints(lcs::Option{<:LinearConstraint}, args...; kwargs...)
    linear_constraints(lcs::AbstractVector{<:LinearConstraint}, ::Nothing, args...; kwargs...)

Return an assembled `LinearConstraint`, `nothing`, or a vector of assembled constraints unchanged.

A constraint slot can hold equations or an assembled [`LinearConstraint`](@ref), so a caller can call [`linear_constraints`](@ref) on the slot with no check of its type.

The vector method takes a `nothing` universe and no other. A vector of assembled constraints needs no [`UniverseSets`](@ref), and a [`Pipeline`](@ref) gives an optimiser this shape when more than one constraint step ran. With a real `UniverseSets`, a wider vector method takes the call and applies this method to each element.

# Algorithm

 1. Return `lcs`. Neither method reads its further positional arguments or its keywords.

# Arguments

  - `lcs`: An existing [`LinearConstraint`](@ref) object, `nothing`, or a vector of constraints.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `lcs`: The input, unchanged.

# Related

  - [`LinearConstraint`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(lcs::Option{<:LinearConstraint}, args...;
                            kwargs...)::Option{<:LinearConstraint}
    return lcs
end
"""
    port_opt_view(lc::LinearConstraint, i, args...) -> LinearConstraint

Return a precomputed [`LinearConstraint`](@ref) unchanged under an asset sub-selection.

A row over the full universe does not mean the same thing over a subset. The method returns the constraint unchanged all the same, because a slice of `A` changes what each row states, and a matrix holds no names that a view can re-resolve. A [`NestedClustered`](@ref) inner solve refuses a bare precomputed constraint for this reason. [`Stacking`](@ref) and [`SubsetResampling`](@ref) have no such guard.

A constraint that reaches a meta-optimiser through an [`ExposureConstraintEstimator`](@ref) is a different case. Its `A` has one column per factor, and the optimiser projects it again against the loadings of the viewed prior, so the view applies to the basis and not to the row.

# Algorithm

 1. Return `lc`. The method reads neither the index nor the tail that follows it.

# Arguments

  - `lc`: The precomputed [`LinearConstraint`](@ref).
  - `::Any`: The asset index selection (ignored).
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `lc::LinearConstraint`: The input, unchanged.

# Related

  - [`port_opt_view`](@ref)
  - [`LinearConstraint`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
"""
function port_opt_view(lc::LinearConstraint, ::Any, args...)::LinearConstraint
    return lc
end
"""
    assert_investable_constraint_width(lcs::Nothing, N::Integer, slot::AbstractString)
    assert_investable_constraint_width(lc::LinearConstraint, N::Integer,
                                       slot::AbstractString)
    assert_investable_constraint_width(lcs::VecLc, N::Integer, slot::AbstractString)

Refuse a precomputed [`LinearConstraint`](@ref) whose rows are wider than the investable universe, with a message that states why.

A name-keyed estimator stays valid after a reduction to the Investable Mask. It resolves against the [`UniverseSets`](@ref) that the door gives it, and a name that left resolves on the Non-Investable Axis and is not refused. A precomputed constraint cannot do this. Its `A` is a matrix, and the position is the only link between a column and an asset, so there is no name to resolve again and no correct way to narrow the matrix. A dropped column changes what `Ax ≤ B` means, and for that reason [`port_opt_view`](@ref)`(::LinearConstraint, i)` is the identity.

The row therefore passes the door at its original width and meets a shorter weight vector. Without this check, the model raises a bare `DimensionMismatch` between two numbers, and nothing links either number to the asset that delisted. This function raises once, before the model, and its message states what the caller did and what the caller can do instead.

The repair is always the same. State the constraint as a [`LinearConstraintEstimator`](@ref), which resolves its names again over the universe that the door leaves.

# Algorithm

 1. Return when there is nothing to check, which is a `nothing` slot or a `nothing` half of a [`LinearConstraint`](@ref).
 2. Otherwise compare `size(A, 2)` of each half against `N` and throw a `DimensionMismatch` naming the slot, the two widths and the repair when they disagree.

# Arguments

  - `lcs`: The resolved constraint, a vector of them, or `nothing`.
  - `N`: The number of investable assets the optimisation runs over.
  - `slot`: Names the field the constraint came from in the message, for example `"lcse"`.

# Validation

  - Every half of every precomputed constraint has one column per investable asset.

# Returns

  - `nothing`.

# Related

  - [`LinearConstraint`](@ref)
  - [`LinearConstraintEstimator`](@ref)
  - [`port_opt_view`](@ref)
  - [`non_investable_sets`](@ref)
"""
function assert_investable_constraint_width(::Nothing, ::Integer, ::AbstractString)::Nothing
    return nothing
end
function assert_investable_constraint_width(lc::LinearConstraint, N::Integer,
                                            slot::AbstractString)::Nothing
    for half in (lc.ineq, lc.eq)
        if isnothing(half)
            continue
        end
        @argcheck(size(half.A, 2) == N,
                  DimensionMismatch("the precomputed linear constraint in `$slot` is written over $(size(half.A, 2)) assets, but this optimisation runs over $N. An asset left the investable universe, and a precomputed constraint cannot follow it: its `A` is bound to its columns by position, so no column can be dropped without changing what the constraint means. State it as a LinearConstraintEstimator, which is resolved by name against whatever universe the door leaves."))
    end
    return nothing
end
function assert_investable_constraint_width(lcs::VecLc, N::Integer,
                                            slot::AbstractString)::Nothing
    for lc in lcs
        assert_investable_constraint_width(lc, N, slot)
    end
    return nothing
end
function linear_constraints(lcs::AbstractVector{<:LinearConstraint}, ::Nothing, args...;
                            kwargs...)::AbstractVector{<:LinearConstraint}
    return lcs
end
"""
    linear_constraints(eqn::EqnType, sets::UniverseSets,
                       key::Option{<:AbstractString} = nothing;
                       ops1::Tuple = ("==", "<=", ">="),
                       ops2::Tuple = (:call, :(==), :(<=), :(>=)), datatype::DataType = Float64,
                       strict::Bool = false, bl_flag::Bool = false,
                       rr::Option{<:AbstractLoadingsRegressionResult} = nothing)

Parse and convert one or more linear constraint equations into a [`LinearConstraint`](@ref) object.

The function parses one or more equations, given as strings, as expressions or as a vector of them. It expands the group names through the [`UniverseSets`](@ref), builds the coefficient matrices, and returns a [`LinearConstraint`](@ref) that holds the equality rows and the inequality rows.

# Algorithm

This method is the whole pipeline, and each step names the stage that owns it.

 1. Parse `eqn` with [`parse_equation`](@ref), giving `lcs`, one [`ParsingResult`](@ref) per equation. Each result carries the equation in canonical form.
 2. Expand every group name of `lcs` into its members with [`replace_group_by_assets`](@ref), giving results written in names of the universe. `bl_flag` selects which of the two expansions runs.
 3. Assemble the coefficient matrices and the right-hand sides from `lcs` with [`get_linear_constraints`](@ref), which resolves each name against the universe `key` names and separates the equality rows from the inequality rows.
 4. Return what [`get_linear_constraints`](@ref) gives, which is a [`LinearConstraint`](@ref), or `nothing` when no row survived.

# Arguments

  - `eqn`: A single constraint equation (as `AbstractString` or `Expr`), or a vector of such equations.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `ops1`: Tuple of valid comparison operators as strings.
  - `ops2`: Tuple of valid comparison operators as expression heads.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, a variable name that the universe does not hold throws. If `false`, it warns.
  - `bl_flag`: If `true`, a group expands to the mean over its members, as the Black-Litterman views need. If `false`, it expands to the sum.
  - `key`: Key naming the universe the variables resolve against. Defaults to `sets.xkey`.
  - `rr`: Loadings to re-base through, or `nothing` for an ordinary asset-space constraint.

# Validation

  - Every stage validates its own input: [`parse_equation`](@ref) the equation text, [`replace_group_by_assets`](@ref) the flag grammar, and [`get_linear_constraints`](@ref) the names against the universe.

# Returns

  - `lcs::LinearConstraint`: An object containing the assembled equality and inequality constraints, or `nothing` if no constraints are present.

# Examples

```jldoctest
julia> sets = UniverseSets(; xkey = \"nx\", dict = Dict(\"nx\" => [\"w_A\", \"w_B\", \"w_C\"]));

julia> linear_constraints([\"w_A + w_B == 1\", \"w_A >= 0.1\"], sets)
LinearConstraint
  ineq ┼ PartialLinearConstraint
       │   A ┼ 1×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [-0.1]
    eq ┼ PartialLinearConstraint
       │   A ┼ 1×3 LinearAlgebra.Transpose{Float64, Matrix{Float64}}
       │   B ┴ Vector{Float64}: [1.0]
```

# Related

  - [`parse_equation`](@ref)
  - [`replace_group_by_assets`](@ref)
  - [`PartialLinearConstraint`](@ref)
  - [`LinearConstraint`](@ref)
  - [`UniverseSets`](@ref)
  - [`linear_constraints`](@ref)
"""
function linear_constraints(eqn::EqnType, sets::UniverseSets,
                            key::Option{<:AbstractString} = nothing;
                            ops1::Tuple = ("==", "<=", ">="),
                            ops2::Tuple = (:call, :(==), :(<=), :(>=)),
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false,
                            rr::Option{<:AbstractLoadingsRegressionResult} = nothing)::Option{<:LinearConstraint}
    lcs = parse_equation(eqn; ops1 = ops1, ops2 = ops2, datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets, bl_flag)
    return get_linear_constraints(lcs, sets, key; datatype = datatype, strict = strict,
                                  rr = rr)
end
"""
    linear_constraints(lcs::LinearConstraintEstimator, sets::UniverseSets;
                       datatype::DataType = Float64, strict::Bool = false,
                       bl_flag::Bool = false,
                       rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                       rd::Option{<:ReturnsResult} = nothing)
    linear_constraints(lcs::VecLcE, sets::UniverseSets;
                       datatype::DataType = Float64, strict::Bool = false,
                       bl_flag::Bool = false,
                       rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                       rd::Option{<:ReturnsResult} = nothing)

Parse the equations a [`LinearConstraintEstimator`](@ref) carries, against the universe key that estimator names.

The method reads `val` and `key` from the estimator and gives both to the equation method, so a single estimator and a vector of estimators have one interface. A vector gives a vector of the same length, one result per element.

The method takes `rr`, so that a caller that holds loadings, such as [`processed_jump_optimiser_attributes`](@ref), can pass them to the value of `lcse` with no check of its type. A bare [`LinearConstraintEstimator`](@ref) ignores them. The asset frame is the absence of a re-basis, and an estimator that re-based itself because loadings were available makes the space depend on the prior and not on what the user wrote. Only a wrapper [`ExposureConstraintEstimator`](@ref) asks for a re-basis. The method takes `rd` for the same reason and ignores it for a stronger one. Only a constraint space can ask for a refit, and a bare estimator has no space.

# Algorithm

 1. Read `val` and `key` from `lcs`.
 2. Ignore `rr` and `rd`, for the reason that the paragraph above gives.
 3. Return the [`LinearConstraint`](@ref) that the equation method builds from `val`, `sets` and `key`.
 4. Apply steps 1 to 3 to each element, and return the vector of results, when `lcs` is a vector. `rr` and `rd` reach every element, and every element ignores them.

# Arguments

  - `lcs`: The [`LinearConstraintEstimator`](@ref) to parse, or a vector of them.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, a variable name that the universe does not hold throws. If `false`, it warns.
  - `bl_flag`: If `true`, a group expands to the mean over its members, as the Black-Litterman views need. If `false`, it expands to the sum.
  - `rr`: Accepted and dropped. A bare estimator never re-bases.
  - `rd`: Accepted and dropped. A bare estimator never asks for a refit.

# Returns

  - `lcs::Option{<:LinearConstraint}`: The assembled constraint, or `nothing` when no row survived. A vector input gives one such value per element.

# Related

  - [`linear_constraints`](@ref)
  - [`ExposureConstraintEstimator`](@ref)
  - [`LinearConstraintEstimator`](@ref)
  - [`UniverseSets`](@ref)
"""
function linear_constraints(lcs::LinearConstraintEstimator, sets::UniverseSets;
                            datatype::DataType = Float64, strict::Bool = false,
                            bl_flag::Bool = false,
                            rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                            rd::Option{<:ReturnsResult} = nothing)::Option{<:LinearConstraint}
    return linear_constraints(lcs.val, sets, lcs.key; datatype = datatype, strict = strict,
                              bl_flag = bl_flag)
end
function linear_constraints(lcs::VecLcE, sets::UniverseSets; datatype::DataType = Float64,
                            strict::Bool = false, bl_flag::Bool = false,
                            rr::Option{<:AbstractLoadingsRegressionResult} = nothing,
                            rd::Option{<:ReturnsResult} = nothing)
    return [linear_constraints(lc, sets; datatype = datatype, strict = strict,
                               bl_flag = bl_flag, rr = rr, rd = rd) for lc in lcs]
end

export LinearConstraintEstimator, parse_equation, replace_group_by_assets,
       linear_constraints
