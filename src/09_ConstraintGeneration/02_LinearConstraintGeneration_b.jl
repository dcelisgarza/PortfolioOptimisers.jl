"""
    _parse_equation(lhs, opstr::AbstractString, rhs; datatype::DataType = Float64)

Parse and canonicalise a linear constraint equation from Julia expressions.

`_parse_equation` takes the left-hand side (`lhs`) and right-hand side (`rhs`) of a constraint equation, both as Julia expressions, and a comparison operator string (`opstr`). It evaluates numeric functions, moves all terms to the left-hand side, collects coefficients and variables, and returns a [`ParsingResult`](@ref) with the canonicalised equation.

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

  - `res::ParsingResult`: Structured result with canonicalised variables, coefficients, operator, right-hand side, and formatted equation. The order of `vars` is the order the variable map iterates in, and it is not the order the equation was written in.

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
    constant::datatype = 0.0
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

An equation string crosses a trust boundary, so both entry shapes carry a limit from `EQUATION_LIMITS[]` before any recursive walk runs. The string form is capped on length before `Meta.parse` runs, and no length applies to the pre-built `Expr` form. Both forms are then capped on the depth of the expression tree, so one number bounds the recursion whichever shape the input takes.

# Algorithm

The method that Julia selects is the algorithm, and one method answers each shape of `eqn`.

 1. `eqn` is a vector: apply this function to each element, and return the vector of results.
 2. `eqn` is a string: check its length against `EQUATION_LIMITS[].max_length`, and refuse the pattern `++`.
 3. Find the first operator of `ops1` that occurs in the string, giving `opstr`, and split the string on it into `lhs` and `rhs`.
 4. Parse both parts with `Meta.parse`, giving `lexpr` and `rexpr`, check each with [`rethrow_parse_error`](@ref), and check the depth of each against `EQUATION_LIMITS[].max_depth` with [`_expr_depth_exceeds`](@ref).
 5. `eqn` is an `Expr`: check its depth against `EQUATION_LIMITS[].max_depth` with [`_expr_depth_exceeds`](@ref), and refuse a `++` pattern with [`has_invalid_plus`](@ref).
 6. Check that the head of the expression is a call and is exactly one operator of `ops2`, giving `opstr`, and read `lhs` and `rhs` off the arguments of the call.
 7. Hand `opstr` and the two sides to [`_parse_equation`](@ref), which canonicalises them and builds the [`ParsingResult`](@ref).

# Arguments

  - `eqn`: The equation string to parse.

      + `eqn::AbstractVector`: Each element needs to meet the criteria below.

      + `eqn::AbstractString`: Must contain exactly one comparison operator from `ops1`.

          * `ops1`: Tuple of valid comparison operators as strings.

      + `eqn::Expr`: Must contain exactly one comparison operator from `ops1`.

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

Check whether a Julia expression contains an invalid `+` operator in a constraint context.

Internal helper used during linear constraint parsing to detect unsupported `+` operator usage in constraint expressions. It is the `Expr` counterpart of the `++` check the string form of [`parse_equation`](@ref) runs on the raw text.

# Algorithm

 1. Return `false` when `expr` is not a call, because only a call can carry the head this function refuses.
 2. Return `true` when the head of the call is the `++` operator.
 3. Apply this function to every argument of the call that is itself an expression, and return `true` when any of them does.

# Arguments

  - `expr`: Julia expression to check.

# Returns

  - `Bool`: `true` if the expression contains an invalid `+`, `false` otherwise.

# Related

  - [`parse_equation`](@ref)
  - [`_expr_depth_exceeds`](@ref)
"""
function has_invalid_plus(expr)::Bool
    if !(isa(expr, Expr) && expr.head == :call)
        return false
    end
    # Check for nested :+ calls (e.g., :(+(+(a, b), c))) or more than two arguments
    if expr.args[1] == :++
        # If any argument is itself a :+ call, that's suspicious (from "++")
        return true
    end
    # Recurse into sub-expressions
    return any(has_invalid_plus(arg) for arg in expr.args[2:end] if isa(arg, Expr))
end
"""
    _expr_depth_exceeds(x, limit::Integer) -> Bool

Return `true` if the expression tree `x` is deeper than `limit`.

Guards the `Expr` form of [`parse_equation`](@ref) against a deeply nested AST that no
string length cap covers. The check itself recurses at most `limit + 1` frames deep and
short-circuits the moment the limit is breached, so it cannot exhaust the stack it protects.

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
                            ep_flag::Bool = false, rho_flag::Bool = false)

Expand group or special variable references in a [`ParsingResult`](@ref) to their corresponding asset names.

This function takes a [`ParsingResult`](@ref) containing variable names (which may include group names, `prior(...)` expressions, or correlation views like `(A, B)`), and replaces these with the actual asset names from the provided [`UniverseSets`](@ref). It supports Black-Litterman-style group expansion, entropy pooling prior views, and correlation view parsing for advanced constraint generation. When `res` is a vector of [`ParsingResult`](@ref) objects, the function is applied to each element of the vector.

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

The two lines are different operations. The first repeats the coefficient on every member, so the expanded row constrains the **sum** over the group. The second divides the coefficient by the member count, so the expanded row constrains the **mean** over the group. A group of one member is the only case in which the two agree.

# Algorithm

 1. Copy `res.vars` and `res.coef` into `variables_new` and `coeffs_new`, and open the empty accumulators `variables_tmp`, `coeffs_tmp` and `idx_rm`.
 2. For each variable name of `res.vars`, match it against the prior pattern `prior(...)` and against the correlation pattern `(a, b)`. The four combinations of the two matches select steps 3 to 6.
 3. A name matching neither pattern, with `rho_flag` false, is a plain name. Look it up in `sets.dict`, and leave it where it stands when the dictionary does not hold it, because a name that is not a group is already the name of one column. A group name sheds its departed members with [`shed_departed_members`](@ref), then expands to what survived, each member carrying the coefficient the mathematics above gives over the **surviving** count, and the index of the group joins `idx_rm`. A group that shed every member expands to nothing and its index joins `idx_rm` all the same.
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
  - `ep_flag`: If `true`, enables expansion of `prior(...)` expressions for entropy pooling.
  - `rho_flag`: If `true`, enables expansion of correlation views `(A, B)` for entropy pooling.
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
    # `idx_rm`, not `variables_tmp`: a group that sheds its every member expands to nothing
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

Name of the axis the universe stored under `key` belongs to, read off the key itself: `"factor"` for anything carrying the `tfkey` or the `cfkey` prefix, `"asset"` otherwise. It exists only so [`unknown_variable_msg`](@ref) and [`empty_row_msg`](@ref) can name the axis the user wrote in.

The **key** is the evidence, for both callers, and the reason is that both resolve names against `sets.dict[key]` and nothing else: whatever axis that universe belongs to is the axis a failed lookup failed on. [`get_black_litterman_views`](@ref) takes the key from the estimator that owns the views, and [`get_linear_constraints`](@ref) from the constraint space — [`FactorSpace`](@ref) resolving at the factor axis [`factor_axis_key`](@ref) reads off the loadings, `sets.tfkey` for the time-series family and `sets.cfkey` for the cross-sectional one. Reading it off the *re-basis* instead would be a second encoding of the same fact, and a worse one: a wrapped estimator carrying its own `key` overrides the space's, so a re-based row can legitimately resolve against a universe the loadings are not written in, and the message must name the universe that was searched.

The **prefix** rather than equality is what makes a factor group key (`"nf_sector"`, `"ncf_sector"`) resolve as the factor axis too, and the disjoint-prefix rule [`UniverseSets`](@ref) enforces at construction is what makes that unambiguous.

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

Length of the assembled constraint row. Without a re-basis this is the size of the universe the names resolve against; with one it is the number of *assets* the loadings project onto, because the projection is applied while the row is assembled and what leaves is an ordinary asset-space row.

# Algorithm

The method that Julia selects is the algorithm, and the re-basis selects it.

 1. `rr` is `nothing`: return the length of `nx`, the universe the names resolve against.
 2. `rr` is a regression result: return the number of rows of `rr.M`, which is the number of assets the loadings project onto.

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

Contribution of one matched variable to a constraint row.

Without a re-basis the contribution is the indicator `Ai` scaled by the coefficient `c`. With one it is the columns of the loadings that `Ai` selects, summed and scaled. The columns are **summed** rather than indexed by `findfirst`, so a factor universe carrying a duplicated name contributes every column bearing it, matching how the asset path treats a duplicated asset name.

`rr.M` is used, never `rr.L`: `M`'s columns are the named original factors, and a constraint must be *written* in names a user can put in an equation. Risk decomposition wants `L` and is right to.

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

The identity is what lets a row written in factor names bind asset weights with no change of variables: the re-based row is an ordinary asset-space row over ``\\boldsymbol{w}_a``.

# Algorithm

The method that Julia selects is the algorithm, and the re-basis selects it.

 1. `rr` is `nothing`: return `Ai` scaled by `c`, one entry per name of the universe.
 2. `rr` is a regression result: sum the columns of `rr.M` that `Ai` selects, scale the sum by `c`, and return it. The value is asset-length whatever the row was written in.

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
                           rr::Option{<:AbstractLoadingsRegressionResult} = nothing)

Convert parsed linear constraint equations into a `LinearConstraint` object.

`get_linear_constraints` takes one or more [`ParsingResult`](@ref) objects (as produced by [`parse_equation`](@ref)), expands variable names using the provided [`UniverseSets`](@ref), and assembles the corresponding constraint matrices and right-hand side vectors. The result is a [`LinearConstraint`](@ref) object containing both equality and inequality constraints, suitable for use in portfolio optimisation routines.

A row takes one of two shapes. Without `rr` it runs over the universe the names resolve against. With `rr` it runs over the assets, because the loadings re-base each term as the row is assembled and what leaves the function is an ordinary asset-space row.

**A row is the unit of a drop.** A row is a joint statement over several names with one right-hand side, so a name this function cannot resolve takes the whole row with it rather than only its own term: `a + c == 0.05` assembled without `c` would fit `a == 0.05`, a different and stronger claim than the caller wrote. What the name's failure was decides only whether the drop is *reported*. A name on the **counterpart axis** — read with [`counterpart_axis_names`](@ref), and in practice the Non-Investable Axis a door minted — is dropped in silence under both settings of `strict`, because it was a correct name over the universe the caller was handed and the data moved it; the departure is announced once, by the door. A name on neither axis is a typo, and is reported exactly as before.

# Algorithm

 1. Take `k` as `key`, or `sets.xkey` when `key` is `nothing`, read the universe `nx` from `sets.dict` under it, name the axis with [`universe_axis`](@ref), and read the counterpart axis with [`counterpart_axis_names`](@ref).
 2. Take `N`, the row length, from [`constraint_row_length`](@ref), and allocate the working row `At` of that length.
 3. Zero `At` for each parsing result, and start that result not dropped.
 4. Build the indicator of each variable name of the result over `nx`. A name that matches no entry marks the row dropped, and is reported through [`strict_diagnostic`](@ref) unless it names the counterpart axis. Every name of the row is still visited, so a row carrying two typos names both.
 5. Add the contribution [`constraint_row_term`](@ref) gives for the name and its coefficient to `At`. With `rr` the contribution arrives already projected, so `At` is asset-length while it is accumulated.
 6. Move to the next result when the row was marked dropped.
 7. Report the row through [`strict_diagnostic`](@ref) and drop it when `At` is still zero. Every name resolved to get here, so the message says the row was annihilated — by the loadings under `rr`, by its own cancelling coefficients otherwise, or by there being no name in it at all — and never that a name was mistyped.
 8. Read the sign and the inequality flag of the operator from [`comparison_sign_ineq_flag`](@ref), and scale the row and its right-hand side by the sign. That negates a `>=` row, so both senses of an inequality are written in the `<=` sense, which is the convention [`LinearConstraint`](@ref) states.
 9. Append the row to the inequality accumulator when the flag is `true`, and to the equality accumulator when it is `false`.
10. Reshape each accumulator that holds a row into a matrix of `N` columns, and build the [`PartialLinearConstraint`](@ref) of that half.
11. Return the [`LinearConstraint`](@ref) holding the halves that were built, or `nothing` when neither half holds a row.

# Arguments

  - `lcs`: A single [`ParsingResult`](@ref) or a vector of such objects, representing parsed constraint equations.
  - `sets`: A [`UniverseSets`](@ref) object specifying the universes and groupings.
  - `key`: Key naming the universe the variables resolve against. Defaults to `sets.xkey`; a re-based constraint passes `sets.tfkey`.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.
  - `rr`: Loadings to re-base through, or `nothing` for an ordinary asset-space constraint. See [`ExposureConstraintEstimator`](@ref) — callers do not pass this directly.
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

  - `!isempty(val)`.

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

An unparsed [`LinearConstraintEstimator`](@ref), or an assembled [`LinearConstraint`](@ref). The group exists because a constraint slot accepts both: [`linear_constraints`](@ref) parses the first and passes the second through untouched, so a caller may hand over equations or a block it built earlier.

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

Every abstract vector whose elements are [`LinearConstraintEstimator`](@ref)s. The group is narrower than [`VecLcE_Lc`](@ref) on purpose: every element still has to be parsed, so [`linear_constraints`](@ref) is broadcast over it and answers one constraint per element.

# Related

  - [`LinearConstraintEstimator`](@ref)
  - [`LcE_VecLcE`](@ref)
  - [`VecLcE_Lc`](@ref)
  - [`linear_constraints`](@ref)
"""
const VecLcE = AbstractVector{<:LinearConstraintEstimator}
"""
    const LcE_Lc_VecLcE_Lc = Union{<:LcE_Lc, <:VecLcE_Lc}

One [`LcE_Lc`](@ref), or a vector of them. The group is the widest linear-constraint slot the library declares: it names every shape a user may write into such a field, so it is what the type bound of that field is written against.

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

No-op fallback for returning an existing `LinearConstraint` object, `nothing`, or a vector of them.

This method is used to pass through an already constructed [`LinearConstraint`](@ref) object or `nothing` without modification. It enables composability and uniform interface handling in constraint generation workflows, allowing functions to accept either raw equations or pre-built constraint objects.

The vector arity is narrowed to a `nothing` universe on purpose. A vector needs no [`UniverseSets`](@ref) precisely because every element is already assembled, and that is the shape a [`Pipeline`](@ref) hands an optimiser when more than one constraint step ran; with a real `UniverseSets` the broader vector methods take over and map this one over the elements.

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

The identity is deliberate, and it is **not** the claim that a full-universe row means the same thing over a subset — it does not. It is what the `lcse` slot already did: the slot was passed unviewed until a constraint space gained a basis a view has to follow, and slicing `A` here would change the behaviour of a path this method exists only to leave alone. A [`NestedClustered`](@ref) inner solve refuses a bare precomputed constraint outright for exactly this reason; [`Stacking`](@ref) and [`SubsetResampling`](@ref) carry no such guard, and that gap pre-dates the view.

A constraint reaching a meta-optimiser through an [`ExposureConstraintEstimator`](@ref) is a different case and is handled: its `A` is factor-width and is re-projected against the viewed prior's loadings, so the view it needs is of the *basis*, not of the row.

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

Refuse a **precomputed** [`LinearConstraint`](@ref) whose rows are wider than the investable universe, and say why.

A name-keyed estimator survives a reduction to the Investable Mask: it resolves against the [`UniverseSets`](@ref) the door hands it, and a name that left resolves on the Non-Investable Axis instead of being refused. A precomputed constraint cannot. Its `A` is a matrix, and position is the only link between a column and an asset, so there is no name to re-resolve and no honest way to narrow it — dropping a column silently changes what `Ax ≤ B` means, and [`port_opt_view`](@ref)`(::LinearConstraint, i)` is deliberately the identity for that reason.

So the row survives the door at its original width and meets a shorter weight vector. Left alone that surfaces inside the model as a bare `DimensionMismatch` between two numbers, with nothing to connect either to the asset that delisted. This says it once, at the seam, in terms of what the caller did and what they can do instead.

The repair is always the same: state the constraint as a [`LinearConstraintEstimator`](@ref). A name-keyed constraint is re-resolved over whatever universe the door leaves, which is the whole point of stating it by name.

# Algorithm

 1. Return when there is nothing to check: a `nothing` slot, or a `nothing` half of a [`LinearConstraint`](@ref).
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
    linear_constraints(eqn::EqnType,
                       sets::UniverseSets; ops1::Tuple = ("==", "<=", ">="),
                       key::Option{<:AbstractString} = nothing;
                       ops2::Tuple = (:call, :(==), :(<=), :(>=)), datatype::DataType = Float64,
                       strict::Bool = false, bl_flag::Bool = false)

Parse and convert one or more linear constraint equations into a [`LinearConstraint`](@ref) object.

This function parses one or more constraint equations (as strings, expressions, or vectors thereof), replaces group or asset references using the provided [`UniverseSets`](@ref), and constructs the corresponding constraint matrices. The result is a [`LinearConstraint`](@ref) object containing both equality and inequality constraints, suitable for use in portfolio optimisation routines.

# Algorithm

This method is the whole pipeline, and each step names the stage that owns it.

 1. Parse `eqn` with [`parse_equation`](@ref), giving `lcs`, one [`ParsingResult`](@ref) per equation. Each result carries the equation in canonical form.
 2. Expand every group name of `lcs` into its members with [`replace_group_by_assets`](@ref), giving results written in names of the universe. `bl_flag` selects which of the two expansions runs.
 3. Assemble the coefficient matrices and the right-hand sides from `lcs` with [`get_linear_constraints`](@ref), which resolves each name against the universe `key` names and separates the equality rows from the inequality rows.
 4. Return what [`get_linear_constraints`](@ref) gives: a [`LinearConstraint`](@ref), or `nothing` when no row survived.

# Arguments

  - `eqn`: A single constraint equation (as `AbstractString` or `Expr`), or a vector of such equations.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `ops1`: Tuple of valid comparison operators as strings.
  - `ops2`: Tuple of valid comparison operators as expression heads.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion.
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

The method reads `val` and `key` off the estimator and hands both to the equation method, which gives one uniform interface for a single constraint estimator and for a vector of them. A vector is answered element by element, and the result is a vector of the same length.

`rr` is accepted so that a caller holding loadings — [`processed_jump_optimiser_attributes`](@ref) does — can pass them uniformly to whatever sits in `lcse`, without inspecting its type first. A bare [`LinearConstraintEstimator`](@ref) **drops** them: the asset frame is the absence of a re-basis, and an estimator that quietly re-based itself because loadings happened to be available would make the space depend on the prior rather than on what the user wrote. A re-basis is asked for by wrapping in an [`ExposureConstraintEstimator`](@ref) and by nothing else. `rd` rides along for the same reason and is dropped for a stronger one: only a space can ask for a refit, and a bare estimator has no space.

# Algorithm

 1. Read `val` and `key` off `lcs`.
 2. Drop `rr` and `rd`, for the reason the paragraph above gives.
 3. Return the [`LinearConstraint`](@ref) that the equation method builds from `val`, `sets` and `key`.
 4. Apply steps 1 to 3 to each element, and return the vector of results, when `lcs` is a vector. `rr` and `rd` reach every element, and every element drops them.

# Arguments

  - `lcs`: The [`LinearConstraintEstimator`](@ref) to parse, or a vector of them.
  - `sets`: A [`UniverseSets`](@ref) object specifying the asset universe and groupings.
  - `datatype`: Numeric type for coefficients and right-hand side.
  - `strict`: If `true`, throws an error if a variable or group is not found in `sets`; if `false`, issues a warning.
  - `bl_flag`: If `true`, enables Black-Litterman-style group expansion.
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
