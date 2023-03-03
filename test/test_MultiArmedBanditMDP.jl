include("../src/MultiArmedBandits.jl")
using .MultiArmedBandits
using POMDPs:simulate
using POMDPTools
using Test
using Random

let
    function test_MAB_compiles()
        random_mab = RandomMAB(3, (2, 4, 7), 0.7, rng=MersenneTwister(1))
        return random_mab
    end

    function test_MAB_consistent_distributions()
        mab = RandomMAB(3, (2, 4, 7), 0.7, rng=MersenneTwister(1))
        @test has_consistent_distributions(mab)
    end

    function test_MAB_valid_transitions()
        n = 3 # number of arms
        m = (2, 4, 7) # state space for each arm
        γ = 0.7
        mdp = RandomMAB(n, m, γ, rng=MersenneTwister(1))
        
        println("Results of MAB Simulation: ")
        println("---------------------------")
        # run a simulation
        history = simulate(
            HistoryRecorder(max_steps=10), 
            mdp, 
            RandomPolicy(mdp, rng=MersenneTwister(2))
        )

        function validMABTransition(s, a, sp)
            return (
                (s[1:a-1] == sp[1:a-1]) 
                && (s[a+1:end] == sp[a+1:end])
            )
        end

        all_transitions_valid = true
        for (s, a, r, sp) in eachstep(history, "(s, a, r, sp)")
            all_transitions_valid = validMABTransition(s, a, sp) && all_transitions_valid 
            println("reward $(r) received when state $(sp) was reached after action $(a) was taken in state $(s)")
        end

        @test all_transitions_valid
    end

    test_MAB_compiles()
    test_MAB_consistent_distributions()
    test_MAB_valid_transitions()

end