struct FNPDM_NoFix <: FixNonPositiveDefiniteMatrix end
function fix_non_positive_definite_matrix!(::FNPDM_NoFix, ::AbstractMatrix)
    return nothing
end
