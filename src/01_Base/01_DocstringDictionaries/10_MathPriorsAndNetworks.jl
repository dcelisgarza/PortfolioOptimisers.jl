# `math_dict`: entropy pooling, phylogeny, centrality, clusters, preselection and the compact
# covariance set.
# `01_Tables.jl` declares the table. `unique_key_dict!` refuses a key that another
# entry holds with a different description, in this file or in another one.
unique_key_dict!(math_dict, :math_dict,
                 # Entropy pooling tail views.
                 :cvar_stat => "``\\mathrm{CVaR}_{\\alpha}(X)``: Conditional value at risk of the loss series ``\\boldsymbol{x}`` at level ``\\alpha``.",#
                 :cvar_target => "``\\bar{c}``: Target conditional value at risk of the view.",#
                 :evar_stat => "``\\mathrm{EVaR}_{\\alpha}(X)``: Entropic value at risk of the loss series ``\\boldsymbol{x}`` at level ``\\alpha``.",#
                 :evar_target => "``\\bar{e}``: Target entropic value at risk of the view.",#
                 :ep_tail_nu => "``\\boldsymbol{\\nu}``: ``T \\times 1`` vector of weights that attains the risk measure, the variable of its dual representation.",#
                 # Entropy pooling.
                 :ep_prior_probs => "``\\boldsymbol{q}``: ``T \\times 1`` prior probabilities of the observations, summing to one.",#
                 :ep_post_probs => "``\\boldsymbol{p}``: ``T \\times 1`` posterior probabilities of the observations, summing to one. They are the unknown of the entropy pooling problem.",#
                 :ep_mu_prior_i => "``\\mu_{i}``: Prior mean of asset ``i``. It is a constant of the view, and a lower moment view or a fixing row holds the posterior mean at it.",#
                 :ep_sigma2_prior_i => "``\\sigma_{i}^{2}``: Prior variance of asset ``i``. It is a constant of the view, and a lower moment view or a fixing row holds the posterior variance at it.",#
                 :ep_sc1 => "``s_{c1}``: Constraint scale of the entropy pooling optimiser. It multiplies both sides of a row, so a positive value leaves the feasible set unchanged.",#
                 :ep_sc2 => "``s_{c2}``: Slack penalty of the fixed equality rows. It weights the norm of the slack in the objective, so a larger value holds those rows tighter.",#
                 :ep_so => "``s_{o}``: Objective scale of the entropy pooling optimiser. It multiplies the objective, so a positive value leaves the argument of the optimum unchanged.",#
                 # The rows of a phylogeny, and the factor weights they can act on.
                 :A_iplg => "``\\mathbf{A}``: Row set of an [`IntegerPhylogeny`](@ref), one row for each distinct neighbourhood or cluster, with one column for each position that the MIP space gates.",#
                 :B_iplg => "``\\boldsymbol{B}``: Largest number of held positions in each row of ``\\mathbf{A}``. A scalar applies to every row.",#
                 :i_plg => "``i``: Position of the entry in `plgs`, or `1` for a single entry.",#
                 :B_loadings => "``\\mathbf{B}``: Loading matrix, of size ``N \\times N_{f}``.",#
                 :w_1_factor => "``\\boldsymbol{w}_1``: Factor weights of [`FactorRiskContribution`](@ref), one entry for each of the ``N_f`` factors. The asset weights are ``\\mathbf{B}_1 \\boldsymbol{w}_1``, with ``\\mathbf{B}_1`` the pseudoinverse of ``\\mathbf{B}^\\intercal``, plus an off-factor block when `flag = true`.",#
                 :w_2_off_factor => "``\\boldsymbol{w}_2``: Off-factor weights, one entry for each of the ``N - N_f`` directions the loadings do not span.",#
                 :lu_factor => "``\\boldsymbol{\\ell}_{f}``, ``\\boldsymbol{u}_{f}``: Factor weight bounds, the lower and the upper, one entry for each factor. [`factor_weight_bounds`](@ref) derives them from the asset weight bounds.",#
                 # Separation decay. The four members of
                 # `AbstractSeparationDecayAlgorithm` each state a closed form over the
                 # same separation, so `01_Base_Phylogeny.jl` shares this symbol
                 # between four Units.
                 :d_sep => "``d``: Separation between two assets.",#
                 # The proximity matrix of a separation. `LinearDecay` reads the budget,
                 # and `Proximity` and `phylogeny_features` in
                 # `10_Prior/09_AssetPanelEstimators.jl` state the matrix over the four
                 # symbols below.
                 :dmax_sep => "``d_{\\mathrm{max}}``: Separation budget in scope.",#
                 :D_sep => "``\\mathbf{D}``: Separation matrix, of size ``N \\times N``. Entry ``D_{i,\\,k}`` is the separation of assets ``i`` and ``k`` in the structure, ``D_{i,\\,i} = 0``, and ``D_{i,\\,k} = +\\infty`` when no path joins the two assets.",#
                 :f_sep_decay => "``f``: Separation decay, the score that [`separation_decay`](@ref) gives a separation under the decay algorithm in scope.",#
                 :Z_prox => "``\\mathbf{Z}``: Proximity matrix, of size ``N \\times N``. Entry ``Z_{i,\\,k}`` scores how close asset ``k`` is to asset ``i``, and a larger entry means a closer pair.",#
                 # Network centrality. The eight members of
                 # `AbstractCentralityAlgorithm` each state a closed form over the
                 # same network, so `01_Centrality.jl` shares these symbols between
                 # eight Units.
                 :A_network => "``\\mathbf{A}``: Adjacency matrix of the network. It is binary on the unweighted route, and carries the edge weights of its own branch where the algorithm declares a polarity.",#
                 :n_network => "``n``: Number of assets, which is the number of vertices of the network.",#
                 :lambda_max_network => "``\\lambda_{\\mathrm{max}}``: Largest eigenvalue of ``\\mathbf{A}``.",#
                 :ell_ij_path => "``\\ell_{i,\\,j}``: Length of a shortest path between assets ``i`` and ``j``. It counts the edges on an unweighted network, and sums the edge weights on a weighted one.",#
                 :sigma_st_paths => "``\\sigma_{s,\\,t}``: Number of shortest paths between assets ``s`` and ``t``.",#
                 :sigma_st_i_paths => "``\\sigma_{s,\\,t}(i)``: Number of the shortest paths between assets ``s`` and ``t`` that pass through asset ``i``.",#
                 # Optimal number of clusters. The two members of
                 # `AbstractOptimalNumberClustersAlgorithm` each maximise a score over
                 # the same candidate counts, so `02_Clusters.jl` shares this symbol
                 # between two Units.
                 :c_star_clusters => "``c^{\\star}``: Selected number of clusters.",#
                 # Preselection. `20_AssetSelection.jl` states the admitted set of every
                 # selection rule and of every redundancy algorithm, so the four
                 # symbols below are each shared by two or more Units of that file.
                 :s_i_score => "``s_{i}``: Score of asset ``i``, the risk measure evaluated on that asset's own return series.",#
                 :K_keep_set => "``\\mathcal{K}``: Set of the assets a selector keeps.",#
                 :k_tail_count => "``k``: Number of assets taken from one end of the score ordering.",#
                 :t_corr_threshold => "``t``: Correlation at or above which two assets are redundant.",#
                 # The online step of the prior family. `12_PriorPartialFit.jl` states each
                 # read-out as the batch prior of the rows it folded.
                 :P_batch_prior => "``\\mathcal{P}(\\cdot)``: Batch prior, the result that the estimator's batch verb gives on the matrices in the parentheses.",#
                 :P_fold_prior => "``\\mathcal{P}_T``: Folded prior, the result that the estimator's read-out gives after it folds the observations ``\\boldsymbol{x}_1, \\ldots, \\boldsymbol{x}_T`` in order, one at a time or in blocks of any size.",#
                 # The compact covariance set and the rules that size its radius.
                 # `10_CompactRadiusRules.jl` states both rules over these symbols, and
                 # `09_OrthogonalUncertaintySets.jl` builds the set from them.
                 :kappa_cpt => "``\\kappa \\geq 0``: Radius of the compact covariance set, the multiplier of its penalty.",#
                 :C_cpt => "``\\mathbf{C}``: Diagonal metric square root of the covariance set.",#
                 :Q_cpt => "``\\mathbf{Q}``: Orthonormal basis of the weighted factor span.",#
                 # The mean set of the orthogonal uncertainty sets and its two scalings.
                 # `09_OrthogonalUncertaintySets.jl` states the estimator, both scaling
                 # tags and their verb over these symbols.
                 :G_orth => "``\\mathbf{G}``: Basis of the Orthogonal Subspace, ``N \\times r``, with orthonormal columns.",#
                 :r_orth => "``r``: Dimension of the Orthogonal Subspace.",#
                 :Lambda_orth => "``\\mathbf{\\Lambda}``: Scaling of the mean set inside the Orthogonal Subspace, ``r \\times r``, symmetric and positive semi-definite.",#
                 :D_orth => "``\\mathbf{D}``: Covariance of the idiosyncratic returns that the loadings block carries, ``N \\times N``. It is diagonal when the block stores a variance vector.")
