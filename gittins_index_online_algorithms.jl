using NLsolve
using JuMP
using HiGHS
using LinearAlgebra
using Test

struct BanditProcess
	m::Int64 # cardinality of state space
	P::Matrix{Float64} # transition matrix
	r::Vector{Float64} # reward vector
	s₀::Int64 # initial state
	β::Float64 # discount factor
end

### 24.4.2 Linear Programming Formulation (Chen and Katehakis)
struct ChenKatehakisLinearProgramming
	lp::Model # linear programming model
	
	function ChenKatehakisLinearProgramming(bp::BanditProcess)
		m, P, r, α, β = bp.m, bp.P, bp.r, bp.s₀, bp.β

		h = ones(m)
		h[α] -= 1.0
		c = hcat((1.0 - β) * transpose(ones(m)), [m])
		M = hcat(diagm(h) - β * P, ones(m, 1))

		lp = Model()
		@variable(lp, y[1:m] >= 0.0)
		@variable(lp, z)
		Z = transpose(hcat(transpose(y), z))
		@constraint(lp, M * Z .>= r)
		@objective(lp, Min, dot(c, Z))
		
		return new(lp)
	end
end

function solve(formulation::ChenKatehakisLinearProgramming)
	set_optimizer(formulation.lp, HiGHS.Optimizer)
	optimize!(formulation.lp)
	gi = value(formulation.lp[:z])
	return gi
end


### 24.1.1 Restart Formulation (Katehakis and Veinott)
struct KatehakisVeinottRestartFormulation
	Q⁰::Matrix{Float64} # transition matrix for restart option
	r⁰::Vector{Float64} # instaneous reward for restart option
	Q¹::Matrix{Float64} # transition matrix for continuation option
	r¹::Vector{Float64} # instaneous reward for continuation option
	β::Float64 # discount factor
	α::Int64 # initial start
	
	function KatehakisVeinottRestartFormulation(bp::BanditProcess)
		P, r, α, β = bp.P, bp.r, bp.s₀, bp.β
		
		Q⁰ = similar(P)
		Q⁰ .= P[α:α, :] # each row of Q0 equal to a-th row of P

		r⁰ = similar(r)
		r⁰ .= r[α] # each element of r0 equal to a-th element of r

		Q¹ = P

		r¹ = r
		
		return new(Q⁰, r⁰, Q¹, r¹, β, α)
	end
end

function solve(kv::KatehakisVeinottRestartFormulation)
	function f(v::Vector{Float64}, kv::KatehakisVeinottRestartFormulation)
		return max.(
			kv.r⁰ + kv.β * kv.Q⁰ * v, 
			kv.r¹ + kv.β * kv.Q¹ * v,
		) # element-wise max
	end
	
	f′(v) = f(v, kv)
	v₀ = [50.0, 50.0, 50.0, 50.0] # TODO: appropriate initialization
	sol = fixedpoint(f′, v₀)
	println(sol)
	v = sol.zero
	v = (1.0 - kv.β) * v
	gi = v[kv.α]
	return gi
end


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

	sol_chen_katehakis = solve(ChenKatehakisLinearProgramming(bp))
	sol_katehakis_veinott = solve(KatehakisVeinottRestartFormulation(bp))
	@assert sol_katehakis_veinott ≈ sol_chen_katehakis
end

# test_gi_computation_approx_equal()