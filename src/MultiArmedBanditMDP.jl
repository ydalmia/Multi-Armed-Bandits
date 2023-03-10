State = Tuple{Vararg{Int64}}
Action = Int64

struct MAB <: MDP{State, Action}
	n::Int64 # number of arms
	m::Tuple{Vararg{Int64}} # state space cardinality for each arm
	T::Tuple{Vararg{Array{Float64, 2}}} # bandit process for each arm, n x Sp x S
	r::Tuple{Vararg{Array{Float64}}} # reward vector for each arm, n x S
	γ::Float64 # Discount Factor
end

function POMDPs.states(p::MAB) 
	# m^n total states given by prod of m (this is why gittins index is useful!)
	# each state is a tuple indicating the state of every bandit
	return (convert(Tuple, CartesianIndices(p.m)[i]) for i in 1:prod(p.m))
end

POMDPs.actions(p::MAB) = 1:p.n # choose to pull exactly 1 of n arms

POMDPs.stateindex(p::MAB, s::State) = LinearIndices(p.m)[s...]
POMDPs.actionindex(p::MAB, a::Action) = a

POMDPs.discount(p::MAB) = p.γ

function POMDPs.transition(p::MAB, s::State, a::Action) 	
	function evolve_arm(s::State, a::Action, i::Int64) 
		# evolve arm a to state i
		sp = collect(s)
		sp[a] = i
		return Tuple(sp)
	end
	
	states = [evolve_arm(s, a, i) for i in 1:p.m[a]]
	probabilities = p.T[a][:, s[a]]
	return SparseCat(states, probabilities)
end
	
function POMDPs.reward(p::MAB, s::State, a::Action)
	return p.r[a][s[a]] # draw reward from arm based upon the arm's state
end

function POMDPs.initialstate(p::MAB)
	return Deterministic(Tuple(ones(Int64, p.n))) # s0 = (1, 1, ..., 1)
end 

function RandomMAB(
	n::Int64, m::Tuple{Vararg{Int64}}, γ::Float64; rng::AbstractRNG=Random.GLOBAL_RNG
)	
	T = [rand(rng, m, m) for m in m] # random transition dynamics
	r = [rand(rng, m) .- 0.5 for m in m] # random rewards [-0.5, 0.5]
	for i in 1:n
		# normalize arm transition dynamics
		for j in 1:m[i]
			T[i][:,j] /= sum(T[i][:,j])
		end
	end
	T = Tuple(T)
	r = Tuple(r)
	
	return MAB(n, m, T, r, γ)
end