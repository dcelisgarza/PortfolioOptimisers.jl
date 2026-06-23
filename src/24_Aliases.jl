# Literature acronym aliases — each is a `const` pointing at the canonical long-form type.
# These are exported so callers who know the acronym can discover and use it directly via
# autocomplete without having to look up the full spelling.  Each alias is unambiguous in
# CONTEXT.md; ambiguous abbreviations (e.g. MAD ↔ Mean vs Median, RB ↔ several families)
# are intentionally omitted.

# ── Clustering / hierarchical optimisers ──────────────────────────────────────
"""
Alias for [`HierarchicalRiskParity`](@ref).
"""
const HRP = HierarchicalRiskParity
"""
Alias for [`HierarchicalEqualRiskContribution`](@ref).
"""
const HERC = HierarchicalEqualRiskContribution
"""
Alias for [`SchurComplementHierarchicalRiskParity`](@ref).
"""
const SCHRP = SchurComplementHierarchicalRiskParity

# ── Meta-optimisers ───────────────────────────────────────────────────────────
"""
Alias for [`NestedClustered`](@ref).
"""
const NCO = NestedClustered

# ── JuMP-based optimisers ─────────────────────────────────────────────────────
"""
Alias for [`RelaxedRiskBudgeting`](@ref).
"""
const RRB = RelaxedRiskBudgeting
"""
Alias for [`FactorRiskContribution`](@ref).
"""
const FRC = FactorRiskContribution
"""
Alias for [`NearOptimalCentering`](@ref).
"""
const NOC = NearOptimalCentering

# ── Risk measures — Value-at-Risk family ──────────────────────────────────────
"""
Alias for [`ConditionalValueatRisk`](@ref) (CVaR / Expected Shortfall).
"""
const CVaR = ConditionalValueatRisk
"""
Alias for [`EntropicValueatRisk`](@ref).
"""
const EVaR = EntropicValueatRisk
"""
Alias for [`RelativisticValueatRisk`](@ref).
"""
const RVaR = RelativisticValueatRisk
"""
Alias for [`PowerNormValueatRisk`](@ref).
"""
const PNVaR = PowerNormValueatRisk

# ── Risk measures — Drawdown-at-Risk family ───────────────────────────────────
"""
Alias for [`ConditionalDrawdownatRisk`](@ref).
"""
const CDaR = ConditionalDrawdownatRisk
"""
Alias for [`EntropicDrawdownatRisk`](@ref).
"""
const EDaR = EntropicDrawdownatRisk
"""
Alias for [`RelativisticDrawdownatRisk`](@ref).
"""
const RDDaR = RelativisticDrawdownatRisk
"""
Alias for [`PowerNormDrawdownatRisk`](@ref).
"""
const PNDaR = PowerNormDrawdownatRisk

export HRP, HERC, SCHRP, NCO, RRB, FRC, NOC, CVaR, EVaR, RVaR, PNVaR, CDaR, EDaR, RDDaR,
       PNDaR
