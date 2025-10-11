@interface(AbstractReturnsResultInterface, AbstractReturnsResult,
           (mandatory = (;
                         returns_results_view = ("returns result view" => a::Arguments -> returns_result_view(a.rd,
                                                                                                              a.i) isa
                                                                                          AbstractReturnsResult),
                         prior = ("prior" => a::Arguments -> prior(a.pr, a.rd;
                                                                   a.kwargs...) isa
                                                             AbstractReturnsResult),
                         ucs = ("ucs" => a::Arguments -> ucs(a.uc, a.rd; a.kwargs...) isa
                                                         AbstractUncertaintySetResult),
                         mu_ucs = ("mu ucs" => a::Arguments -> mu_ucs(a.uc, a.rd;
                                                                      a.kwargs...) isa
                                                               AbstractUncertaintySetResult),
                         sigma_ucs = ("sigma ucs" => a::Arguments -> sigma_ucs(a.uc, a.rd;
                                                                               a.kwargs...) isa
                                                                     AbstractUncertaintySetResult),
                         factor_risk_contribution = ("factor risk contribution" => a::Arguments -> factor_risk_contribution(a.args...;
                                                                                                                            rd = a.rd,
                                                                                                                            a.kwargs...) isa
                                                                                                   AbstractVector),
                         optimise = ("optimise" => a::Arguments -> optimise(a.opt, a.rd;
                                                                            a.kwargs...) isa
                                                                   OptimisationResult),
                         set_return_constraints! = ("set return constraints!" => a::Arguments -> set_return_constraints!(a.args...;
                                                                                                                         rd = a.rd,
                                                                                                                         a.kwargs...) ===
                                                                                                 nothing),
                         processed_jump_optimiser_attributes = ("processed jump optimiser attributes" => a::Arguments -> processed_jump_optimiser_attributes(a.opt,
                                                                                                                                                             a.rd;
                                                                                                                                                             a.kwargs...) isa
                                                                                                                         ProcessedJuMPOptimiserAttributes),
                         processed_jump_optimiser = ("processed jump optimiser" => a::Arguments -> processed_jump_optimiser(a.opt,
                                                                                                                            a.rd;
                                                                                                                            a.kwargs...) isa
                                                                                                   JuMPOptimiser),
                         set_factor_risk_contribution_constraints! = ("set factor risk contribution constraints!" => a::Arguments -> set_factor_risk_contribution_constraints!(a.model,
                                                                                                                                                                               a.re,
                                                                                                                                                                               a.rd,
                                                                                                                                                                               a.args...) ===
                                                                                                                                     nothing),
                         near_optimal_centering_setup = ("near optimal centering setup" => a::Arguments -> near_optimal_centering_setup(a.opt,
                                                                                                                                        a.rd;
                                                                                                                                        a.kwargs...) isa
                                                                                                           NearOptimalSetup),
                         set_risk_budgeting_constraints! = ("set risk budgeting constraints!" => a::Arguments -> set_risk_budgeting_constraints!(a.model,
                                                                                                                                                 a.rb,
                                                                                                                                                 a.sets,
                                                                                                                                                 a.pr,
                                                                                                                                                 a.wb,
                                                                                                                                                 a.rd) isa
                                                                                                                 ProcessedFactorRiskBudgetingAttributes),
                         set_risk_constraints! = ("set risk constraints!" => a::Arguments -> set_risk_constraints!(a.args...;
                                                                                                                   rd = a.rd,
                                                                                                                   a.kwargs...) ===
                                                                                             nothing),
                         regression = ("`regression(re::Union{<:RegressionEstimator, <:Regression}, rd::AbstractReturnsResult)`" => a::Arguments -> regression(a.re,
                                                                                                                                                               a.rd) isa
                                                                                                                                                    Regression)),
            optional = (;)), """Structures meant to contain returns data.""")
@implements(AbstractReturnsResultInterface, ReturnsResult, ReturnsResult())

export AbstractReturnsResultInterface
