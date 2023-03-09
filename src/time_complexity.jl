### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ b914e64a-eb74-4027-9395-acc1353d6e5b
begin
	include("gittins_index_online_algorithms.jl")
	include("MultiArmedBanditMDP.jl")
	include("gittins_mab_solver.jl")
	
end

# ╔═╡ 3f4d1e2e-b953-11ed-1c5f-cd674e849dc5
function time_compiler_n()
    # Simulate problems to collect execution times
    # Define value collections
    n_values = 1:10
    gittins_times = []
    value_iter_times = []
    # Iterate to fill arrays above
    for n_arms in n_values
        # Generate a new MAB problem with the specified number of arms
        mab = RandomMAB(n_arms, (10 for _ in 1:n_arms), 0.7, rng=MersenneTwister(1))

        # Time the execution of the Gittins Index algorithm
        Random.seed!(1234)
        t1 = @elapsed begin
            # Solve the problem using Gittins Index
            gittins_index_policy = POMDPs.solve(GittinsIndexSolver(),mab)
        end
        push!(gittins_times, t1)

        # Time the execution of the Value Iteration algorithm
        Random.seed!(1234)
        t2 = @elapsed begin
            # Solve the problem using Value Iteration
            value_iteration_policy = POMDPs.solve(ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true), mab)
        end
        push!(value_iter_times, t2)
    end

    # Create a line graph of n versus time to solve for each problem
    plot(n_values, gittins_times, label="Gittins Index")
    plot!(n_values, value_iter_times, label="Value Iteration")
    xlabel!("Number of Arms (n)")
    ylabel!("Time to Solve (seconds)")
    title!("Time to Solve: Gittins Index and Value Iteration")
end

# ╔═╡ a5794661-302d-423d-89ef-fcc503ed21e9
function time_compiler_m()
    # Simulate problems to collect execution times
    # Define value collections
    m_values = 1:10
    gittins_times = []
    value_iter_times = []
    # Iterate to fill arrays above
    for m_states in m_values
        mab = RandomMAB(10, (m_states for _ in 1:10), 0.7, rng=MersenneTwister(1))

        # Time the execution of the Gittins Index algorithm
        Random.seed!(1234)
        t1 = @elapsed begin
            # Solve the problem using Gittins Index
            gittins_index_policy = POMDPs.solve(GittinsIndexSolver(),mab)
        end
        push!(gittins_times, t1)

        # Time the execution of the Value Iteration algorithm
        Random.seed!(1234)
        t2 = @elapsed begin
            # Solve the problem using Value Iteration
            value_iteration_policy = POMDPs.solve(ValueIterationSolver(max_iterations=100, belres=1e-6, verbose=true), mab)
        end
        push!(value_iter_times, t2)
    end

    # Create a line graph of m versus time to solve for each algorithm
    plot(m_values, gittins_times, label="Gittins Index")
    plot!(m_values, value_iter_times, label="Value Iteration")
    xlabel!("State Space of Each Arm (m)")
    ylabel!("Time to Solve (seconds)")
    title!("Time to Solve: Gittins Index vs Value Iteration")
end

# ╔═╡ 974444d3-b7ef-4772-b04f-b8283ccf96be
time_compiler_n()

# ╔═╡ 1747516d-58cf-4444-afb2-80a5f602b0ee
time_compiler_m()

# ╔═╡ Cell order:
# ╠═b914e64a-eb74-4027-9395-acc1353d6e5b
# ╠═3f4d1e2e-b953-11ed-1c5f-cd674e849dc5
# ╠═a5794661-302d-423d-89ef-fcc503ed21e9
# ╠═974444d3-b7ef-4772-b04f-b8283ccf96be
# ╠═1747516d-58cf-4444-afb2-80a5f602b0ee
