# Linear Constraints

```@docs
PartialLinearConstraint
LinearConstraint
VecLc
Lc_VecLc
LinearConstraintEstimator
LcE_Lc
VecLcE
LcE_VecLcE
VecLcE_Lc
LcE_Lc_VecLcE_Lc
ParsingResult
VecPR
PR_VecPR
replace_group_by_assets
estimator_to_val
parse_equation
linear_constraints
get_linear_constraints(lcs::PR_VecPR, sets::UniverseSets,
                                key::Option{<:AbstractString} = nothing;
                                datatype::DataType = Float64, strict::Bool = false,
                                rr::Option{<:AbstractRegressionResult} = nothing)
prefixed_sets_keys
unclaimed_sets_keys
universe_axis
constraint_row_length
constraint_row_term
AbstractParsingResult
name_to_val!
_parse_equation
rethrow_parse_error
format_term
collect_terms!
_collect_terms
allowed_functions
eval_numeric_functions
has_invalid_plus
port_opt_view(sets::UniverseSets, i, args...)
port_opt_view(lc::LinearConstraint, ::Any, args...)
factor_universe
feature_universe
_expr_depth_exceeds
```
