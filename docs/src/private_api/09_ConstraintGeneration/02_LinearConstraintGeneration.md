```@meta
Description = "Linear Constraints, private API of PortfolioOptimisers.jl: VecLc, Lc_VecLc, LcE_Lc, VecLcE, LcE_VecLcE, VecLcE_Lc, LcE_Lc_VecLcE_Lc, VecPR, PR_VecPR, …"
```

# Linear Constraints: private API

```@docs
VecLc
Lc_VecLc
LcE_Lc
VecLcE
LcE_VecLcE
VecLcE_Lc
LcE_Lc_VecLcE_Lc
VecPR
PR_VecPR
AbstractParsingResult
allowed_functions
merge_partial_linear_constraints
merge_linear_constraints
get_linear_constraints(lcs::PR_VecPR, sets::UniverseSets, key::Option{<:AbstractString} = nothing; datatype::DataType = Float64, strict::Bool = false, rr::Option{<:AbstractRegressionResult} = nothing)
prefixed_sets_keys
unclaimed_sets_keys
assert_factor_partition
assert_factor_unique_group
universe_axis
constraint_row_length
constraint_row_term
name_to_val!
_parse_equation
rethrow_parse_error
format_term
collect_terms!
_collect_terms
eval_numeric_functions
has_invalid_plus
factor_universe
factor_axis_key
_expr_depth_exceeds
assert_investable_constraint_width
non_investable_sets
non_investable_names
record_non_investable_drop!
record_group_shed!
announce_non_investable
counterpart_axis_names
shed_departed_members
```
