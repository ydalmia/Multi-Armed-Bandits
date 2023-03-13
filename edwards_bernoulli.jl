### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 922f6ab8-3e58-4045-a2dc-b8b9724490ec
begin
	import Statistics
	using Statistics: mean
	using BenchmarkTools
	using ProgressLogging
end

# ╔═╡ a3b6d034-3909-4ce2-9b51-431d74d0a9e1
function calibrate_arm(f, lb, ub, tol, other_args...)
	while (ub - lb) > tol
		lambda = lb + (ub - lb) / 2
		if f(lambda, other_args...) > 0
			lb = lambda
		else
			ub = lambda
		end
	end
	return [lb, ub]
end

# ╔═╡ c5c0bd92-1a56-42de-b408-3c05b0e725d8
function bmab_giplus_value(lambda, Sigma::Int64, n::Int64, gamma::Float64)
	mu = Sigma / n
	mu_success = (Sigma + 1) / (n + 1)
	H = gamma / (1 - gamma)
	
	function f_continue(x) 		
		return pdf(Beta(Sigma + 1, n - Sigma), x) * x
	end	

	integral, err = quadgk(f_continue, lambda, 1)
	value_success = H * (integral + lambda * cdf(Beta(Sigma + 1, n - Sigma), lambda))
	
	value_fail = H * lambda
	
	return mu + mu * value_success + (1 - mu) * value_fail - lambda / (1 - gamma)
end

# ╔═╡ 328d314b-d160-4eb2-9f45-5981fa689c90
function bmab_giplus(
	Sigma::Int64, 
	n::Int64, 
	gamma::Float64, 
	tol::Float64, 
	upper::Bool = false,
)
	interval = calibrate_arm(
		bmab_giplus_value, 
		lb = Sigma / n, 
		ub = 1, 
		tol, 
		Sigma, 
		n, 
		gamma,
	)
	
  	if upper
		return interval[2]
	end
	
	return mean(interval)
end

# ╔═╡ fe452b4b-2278-4fa7-9e73-5a6d7405e90b
function bmab_kgi(
	Sigma::Union{Array{Int64}, Int64}, 
	n::Union{Array{Int64}, Int64},
	gamma::Float64,
)
  	mu = Sigma ./ n
  	H = gamma / (1 - gamma)
	
	return (mu .+ H .* mu .* (Sigma .+ 1) ./ (n .+ 1)) ./ (1 .+ H .* mu)
  	
end

# ╔═╡ bb157aa8-89ec-41d3-a6c5-56baa0be4be7
function bmab_gi_value(lambda, Sigma, n, gamma, N)
	h = N + 1
	
	n_vec = collect(n : (n + N))
	s_vec = collect(Sigma : (Sigma + N))
	
	mu = s_vec ./ transpose(n_vec)
	
	value_mat = Array{Union{Missing, Float64}, 2}(missing, h, h)
	
	# Values of end states
	value_mat[:, h] = max.(mu[:, h], lambda) .* gamma .^ N / (1 - gamma)
	safe_reward = lambda * gamma .^ ((1:N) .- 1) ./ (1 - gamma)
	
	# Run DP to get values of other states
	for i in N:-1:1
		j = i + 1
		  
		risky_reward = mu[1 : i, i] .* (gamma ^ (i - 1) .+ value_mat[2 : j, j]) .+ (1 .- mu[1 : i, i]) .* value_mat[1 : i, j]
		  
		value_mat[1 : i, i] = max.(risky_reward, safe_reward[i])
  	end
	
	return (value_mat[1, 1] - lambda / (1 - gamma))
end

# ╔═╡ 4b4c696a-9f78-45f0-8ae9-7acd6aaa2f1c
function bmab_gi(;
	Sigma::Int64, 
	n::Int64, 
	gamma::Float64, 
	N::Int64, 
	tol::Float64,
	lb::Union{Float64, Int64, Nothing} = nothing, 
	ub::Union{Float64, Int64, Nothing} = nothing,
)
	if isnothing(lb)
		lb = bmab_kgi(Sigma, n, gamma)
	end
	
	if isnothing(ub)
    	ub = bmab_giplus(Sigma=Sigma, n=n, gamma=gamma, tol=tol, upper=true)
	end
	
  	return mean(calibrate_arm(bmab_gi_value, lb, ub, tol, Sigma, n, gamma, N))
end

# ╔═╡ 406faf3c-7cba-48e1-a42c-4fa695c22d06
function bmab_gi_ab(;
	alpha::Int64, 
	beta::Int64, 
	gamma::Float64,
	tol::Float64, 
	N::Int64, 
	lb::Union{Float64, Int64, Nothing} = nothing, 
	ub::Union{Float64, Int64, Nothing} = nothing,
)
	return bmab_gi(
		Sigma = alpha, 
		n = alpha + beta, 
		gamma = gamma, 
		tol = tol,
		N = N,
		lb = lb, 
		ub = ub,
	)
end

# ╔═╡ 0f4b0858-c0e5-11ed-253c-d16f07ac1e8b
function bmab_gi_multiple_ab(;
	alpha_start::Int64, 
	beta_start::Int64, 
	gamma::Float64, 
	N::Int64, 
	num_actions::Int64, 
	tol::Float64,
)	
	GI = Array{Union{Float64, Missing}, 2}(missing, num_actions, num_actions)
  	alpha_range = collect(alpha_start:(alpha_start + num_actions - 1))
	beta_range = collect(beta_start:(beta_start + num_actions - 1))
  	mu = alpha_start ./ (alpha_start .+ beta_range)
  	lb_vec = bmab_kgi(alpha_start, alpha_start .+ beta_range, gamma)
	
  	@progress for a in 1:num_actions
		ub = 1
		for b in 1:(num_actions - a + 1)
			GI[b, a] = bmab_gi_ab(
				alpha = alpha_range[a], 
				beta = beta_range[b], 
				gamma = gamma,
				tol = tol, 
				N = N, 
				lb = lb_vec[b], 
				ub = ub,
			)
		  	ub = GI[b, a]
		end
		lb_vec = GI[:, a]
	end
	return GI
end

# ╔═╡ 134da00e-5f77-49e3-a3e6-e39ee7279e12
begin
	bmab_gi_multiple_ab(
		alpha_start = 1, 
		beta_start = 1, 
		gamma = 0.9, 
		N = 200, 
		num_actions = 1000, 
		tol = 1e-3,
	)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
BenchmarkTools = "~1.3.2"
ProgressLogging = "~0.1.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "44ea86739a048c0ef0e0de1ae36f1880457670e1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╠═922f6ab8-3e58-4045-a2dc-b8b9724490ec
# ╠═a3b6d034-3909-4ce2-9b51-431d74d0a9e1
# ╠═0f4b0858-c0e5-11ed-253c-d16f07ac1e8b
# ╠═406faf3c-7cba-48e1-a42c-4fa695c22d06
# ╠═4b4c696a-9f78-45f0-8ae9-7acd6aaa2f1c
# ╠═328d314b-d160-4eb2-9f45-5981fa689c90
# ╠═c5c0bd92-1a56-42de-b408-3c05b0e725d8
# ╠═fe452b4b-2278-4fa7-9e73-5a6d7405e90b
# ╠═bb157aa8-89ec-41d3-a6c5-56baa0be4be7
# ╠═134da00e-5f77-49e3-a3e6-e39ee7279e12
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
