# Linear Constraints

```@docs
PartialLinearConstraint
LinearConstraint
Base.getproperty(obj::LinearConstraint, sym::Symbol)
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
get_linear_constraints(lcs::PR_VecPR, sets::AssetSets,
                                key::Option{<:AbstractString} = nothing;
                                datatype::DataType = Float64, strict::Bool = false)
AbstractParsingResult
group_to_val!
_parse_equation
_rethrow_parse_error
_format_term
_collect_terms!
_collect_terms
_eval_numeric_functions
_has_invalid_plus
port_opt_view
```
