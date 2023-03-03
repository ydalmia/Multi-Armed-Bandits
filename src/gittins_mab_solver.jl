MABState = Tuple{Vararg{Int64}}
BanditState = Int64
GittinsIndexValue = Float64
GittinsIndexTable = Array{GittinsIndexValue}

struct GittinsIndexSolver <: Solver end

struct GittinsIndexPolicy <: Policy
    n_arms::Int64
    indices::Tuple{Vararg{GittinsIndexTable}}
end

function POMDPs.action(gittins_index_policy::GittinsIndexPolicy, s::MABState)
    gi_each_arm = [
        gittins_index_policy.indices[arm][s[arm]] 
        for arm in 1:gittins_index_policy.n_arms
    ]
    (best_gi, best_arm) = findmax(gi_each_arm)
    return best_arm
end

function POMDPs.solve(solver::GittinsIndexSolver, mab::MAB)
    n_arms = mab.n

    gittins_index_table_for_each_arm = Array{GittinsIndexTable}(undef, n_arms)
    for arm in 1:n_arms
        P = mab.T[arm]
        r = mab.r[arm] 
        m = mab.m[arm] # state space cardinality
        gittins_index_table::GittinsIndexTable = Array{GittinsIndexValue}(undef, m)
        for s₀ in 1:m
            gi_s₀ = solve(
                ChenKatehakisLinearProgramming(
                    BanditProcess(m, P, r, s₀, mab.γ)
                )
            )
            gittins_index_table[s₀] = gi_s₀
        end
        gittins_index_table_for_each_arm[arm] = gittins_index_table
    end
    return GittinsIndexPolicy(
        n_arms, 
        Tuple(gittins_index_table_for_each_arm)
    )
end
