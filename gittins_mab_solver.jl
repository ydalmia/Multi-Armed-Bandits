include("gittins_index_online_algorithms.jl")
include("mab.jl")

import POMDPs
using POMDPs: Solver, Policy
import DiscreteValueIteration
using DiscreteValueIteration: ValueIterationSolver

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
                Chen_Katehakis_Linear_Programming(
                    Bandit_Process(m, P, r, s₀, mab.γ)
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


function test_gittins_index_solver()
    mab = RandomMAB(3, (2, 4, 7), 0.7, rng=MersenneTwister(1))
    
    gittins_index_policy = POMDPs.solve(
        GittinsIndexSolver(),
        mab,
    )
	value_iteration_policy = POMDPs.solve(
        ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true),
        mab,
    )

    Random.seed!(1234)
	gittins_index_history = simulate(
		HistoryRecorder(max_steps=10), 
		mab, 
		gittins_index_policy,
	)
    Random.seed!(1234)
    value_iteration_history = simulate(
		HistoryRecorder(max_steps=10), 
		mab, 
		value_iteration_policy,
	)

    for (gittins_index_step, value_iteration_step) in zip(eachstep(gittins_index_history), eachstep(value_iteration_history))
		fields = [:s, :a, :r, :sp]
        for field in fields
            @assert gittins_index_step[field] == value_iteration_step[field]
        end
        s, a, r, sp = gittins_index_step[fields]
        println("reward $(r) received when state $(sp) was reached after action $(a) was taken in state $(s)")
	end
end 

test_gittins_index_solver()
