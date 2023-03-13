### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 2894b34d-9353-46a1-bcc0-5b3616fc60c6
begin
	import Distributions
	using Distributions: Normal, pdf, cdf
	import Statistics
	using Statistics: mean
end

# ╔═╡ bcdc1e01-6e13-4269-84ec-5913ffaf3b5b
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

# ╔═╡ 2c1e0288-1995-4db6-9cb3-b1e809abc8ab
function nmab_giplus_value(lambda, mu, n, gamma)
	sd = sqrt(1 / n)
	return mu + gamma * pdf(Normal(), lambda / sd) * sd / (1 - gamma * cdf(Normal(), lambda / sd)) - lambda
end

# ╔═╡ 90eba303-4eb6-4ecf-aca8-b2c9501f8f6e
function nmab_giplus(;Sigma, n, gamma, tol, ub = NA, upper = false)
	if isnothing(ub)
		ub = gamma / (1 - gamma) / sqrt(n)
	end

	lb = 0
	interval = Sigma / n .+ calibrate_arm(
		nmab_giplus_value, lb, ub, tol, 0, n, gamma
	)
	
	if upper
		return interval[2]
	end
	
  	return mean(interval)
end

# ╔═╡ 409ffcf4-64ab-4efe-84a8-39c494783707
function nmab_kgi_value(lambda, mu, n, gamma, tau)
	sigt = sqrt(1 / n - 1 / (n + tau))
	z = (mu - lambda) / sigt
	v = sigt * (z * cdf(Normal(), z) + pdf(Normal(), z))
  	return v * gamma / (1 - gamma) + mu - lambda
end

# ╔═╡ 752c6b9b-6cfc-4af8-a5d9-e861c4792f46
function nmab_kgi(;Sigma, n, gamma, tau, tol, ub = NA, lower=false)
	if isnothing(ub)
		ub = gamma / (1 - gamma) / sqrt(n)
	end

	lb = 0
	interval = Sigma / n .+ calibrate_arm(
		nmab_kgi_value, lb, ub, tol, 0, n, gamma, tau,
	)
	
	if lower
		return(interval[1])
	end
	
	return mean(interval)
end

# ╔═╡ d44fe738-8255-480a-a998-fde7da172440
function nmab_risky_reward(;
	mu, 
	y_lo_scaled, 
	y_hi_scaled, 
	tn_scaled, 
	tau, 
	s, 
	value_vec, 
	discount,
)
  yhi = y_hi_scaled .- mu .* tn_scaled
  ylo = y_lo_scaled .- mu .* tn_scaled
  p = cdf(Normal(mu, s), yhi) - cdf(Normal(mu, s), ylo)
	
  return discount * mu .+ (
	  cdf(Normal(mu, s), ylo[1]) * value_vec[1] .+ (1 - cdf(Normal(mu, s), yhi[end])) * value_vec[end] .+ sum(p .* value_vec)
  )
end

# ╔═╡ f1d00fb8-5132-46dd-ab32-2a365b83e972
function nmab_gi_value(lambda, n, gamma, tau, N, xi, delta)
	extra = 1 # number of extra xi used for new states
	h = N + 1
	delta = delta / sqrt(n) # adjust delta so number of states is constant with n
	mu_range = 0 : delta : ((xi + extra) * sqrt(1 / n))
	mu_range = collect(mu_range)
	lr = length(mu_range)
	lr2 = 0 : delta : (xi * sqrt(1 / n))
	lr2 = length(collect(lr2))
	
	value = zeros(Float64, h, lr)
	
	# Value of end states (at stage N)
	value[h, :] = max.(mu_range, lambda) .* (gamma ^ N / (1 - gamma))
	rr = gamma .^ collect((0 : N)) ./ (1 - gamma)
	value[:, lr2 : lr] = rr * transpose(mu_range[lr2 : lr])
  
	lo = mu_range .- delta / 2 # lower end of discrete approximation to mu
  	hi = mu_range .+ delta / 2
	for j in N : -1: 2
		t = j - 1
		tn = n + tau * (j - 1)
		# the next 3 variables are used for speed-up only
		y_hi_scaled = hi * (tn + tau) / tau
		y_lo_scaled = lo * (tn + tau) / tau
		tn_scaled = tn / tau
		s = sqrt(1 / tn + 1 / tau) #sd of y
		discount = gamma ^ t
		safe_reward = lambda * discount / ( 1 - gamma)
		value_vec = value[j + 1, :]
		for i in (lr2 - 1): -1 : 1
			risky_reward = nmab_risky_reward(
				mu = mu_range[i], 
				y_lo_scaled = y_lo_scaled, 
				y_hi_scaled = y_hi_scaled, 
				tn_scaled = tn_scaled, 
				tau = tau, 
				s = s, 
				value_vec = value_vec, 
				discount = discount,
			)
			if risky_reward > safe_reward
				value[j, i] = risky_reward
			else
				value[j, 1 : i] .= safe_reward
				break
			end
		end
		# If risky arm preferred in [j, 1] then it will be preferred in starting state
		if value[j, 1] > safe_reward
			return (value[j, 1] - safe_reward)
		end
	end
	
	# Value of risky arm in starting state at time 0
	s = sqrt(1 / n + 1 / tau)
	value_vec = value[2, :]
	risky_reward = nmab_risky_reward(
		mu = mu_range[1], 
		y_lo_scaled = y_lo_scaled = lo * (n + tau) / tau, 
		y_hi_scaled = y_hi_scaled = hi * (n + tau) / tau,
		tn_scaled = tn_scaled = n / tau, 
		tau = tau, 
		s = s, 
		value_vec = value_vec, 
		discount = 1,
	)
	return (risky_reward - lambda / (1 - gamma))
end

# ╔═╡ ffa52a65-d6e6-469b-9c64-70833517233f
function nmab_gi(;Sigma, n, gamma, tau, tol, N, xi, delta, lb=nothing, ub=nothing)
	if isnothing(lb)
		lb = nmab_kgi(
			Sigma = 0, 
			n = n,
			gamma = gamma, 
			tau = tau, 
			tol = tol, 
			ub = ub, 
			lower=true,
		)
	end
	
	if isnothing(ub)
		ub = nmab_giplus(
			Sigma = 0, 
			n = n, 
			gamma = gamma, 
			tol = tol, 
			ub = ub, 
			upper=true,
		)
	end
	
	interval = Sigma / n .+ calibrate_arm(
		nmab_gi_value, lb, ub, tol, n, gamma, tau, N, xi, delta,
	)
	
	return mean(interval)
end

# ╔═╡ 4d10600a-c155-11ed-0d1f-539cc5b8aec5
function nmab_gi_multiple(;n_range, gamma, tau, tol, N, xi, delta)
	nn = length(n_range)
	gi_vec = zeros(nn)
	ubbl = gamma / (1 - gamma) / sqrt(n_range[1])
	ub = nmab_giplus(
		Sigma = 0, 
		n = n_range[1], 
		gamma = gamma, 
		tol = tol, 
		ub = ubbl, 
		upper = true,
	)
	
	for i in 1 : nn
		gi_vec[i] = nmab_gi(
			Sigma = 0, 
			n = n_range[i], 
			gamma = gamma, 
			tau = tau, 
			tol = tol, 
			N = N, 
			xi = xi, 
			delta = delta, 
			ub=ub,
		)
		ub = gi_vec[i]
	end
	
	return gi_vec
end

# ╔═╡ 8d017268-6d61-4744-981e-3f12d361854b
begin
	function trial()
		n_range = 1 : 20
		n = collect(n_range)
		gis = nmab_gi_multiple(
			n_range = n_range, 
			gamma = 0.9, 
			tau = 1, 
			tol = 5e-5, 
			N = 30, 
			xi = 3, 
			delta = 0.02
		)
		table_vals = n .* ((1 - 0.9)^ 0.5) .* gis
	end
	trial()
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Distributions = "~0.25.86"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "f154f27ec555893334bb624d542e3ed3aa0d2fd7"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "da9e1a9058f8d3eec3a8c9fe4faacfb89180066b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.86"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "d3ba08ab64bdfd27234d3f61956c966266757fe6"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.7"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "786efa36b7eff813723c4849c90456609cf06661"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═2894b34d-9353-46a1-bcc0-5b3616fc60c6
# ╠═bcdc1e01-6e13-4269-84ec-5913ffaf3b5b
# ╠═4d10600a-c155-11ed-0d1f-539cc5b8aec5
# ╠═ffa52a65-d6e6-469b-9c64-70833517233f
# ╠═90eba303-4eb6-4ecf-aca8-b2c9501f8f6e
# ╠═752c6b9b-6cfc-4af8-a5d9-e861c4792f46
# ╠═2c1e0288-1995-4db6-9cb3-b1e809abc8ab
# ╠═409ffcf4-64ab-4efe-84a8-39c494783707
# ╠═d44fe738-8255-480a-a998-fde7da172440
# ╠═f1d00fb8-5132-46dd-ab32-2a365b83e972
# ╠═8d017268-6d61-4744-981e-3f12d361854b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002