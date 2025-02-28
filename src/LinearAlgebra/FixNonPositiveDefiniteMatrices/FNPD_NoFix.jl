struct FNPD_NoFix <: FixNonPositiveDefiniteMatrix end
function fix_non_positive_definite_matrix!(::FNPD_NoFix, ::AbstractMatrix)
    return nothing
end

export FNPD_NoFix
