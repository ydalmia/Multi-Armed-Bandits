include("../src/MultiArmedBandits.jl")
using .MultiArmedBandits
using POMDPs
using POMDPTools
using Test
using Random
import DiscreteValueIteration
using DiscreteValueIteration: ValueIterationSolver

let 
    function test_gi_computation_approx_equal()
        bp = BanditProcess(
            4,
            [
                0.1 0 0.8 0.1; 
                0.5 0 0.1 0.4; 
                0.2 0.6 0 0.2; 
                0 0.8 0 0.2
            ],
            [16.0, 19.0, 30.0, 4.0],
            2,
            0.75
        )
    
        sol_chen_katehakis = MultiArmedBandits.solve(ChenKatehakisLinearProgramming(bp))
        sol_katehakis_veinott = MultiArmedBandits.solve(KatehakisVeinottRestartFormulation(bp))
        @test sol_katehakis_veinott ≈ sol_chen_katehakis
    end
    test_gi_computation_approx_equal()

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
                @test gittins_index_step[field] == value_iteration_step[field]
            end
            s, a, r, sp = gittins_index_step[fields]
            println("reward $(r) received when state $(sp) was reached after action $(a) was taken in state $(s)")
        end
    end 

    test_gittins_index_solver()
end