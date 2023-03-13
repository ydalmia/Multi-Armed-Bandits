using Distributions

State = Int64
Action = Int64

struct MultiArmedBandit <: MDP{State, Action}
	k::Int64 # number of arms
	arms::Array{DistributionMarkovProcess}
	γ::Float64 # Discount Factor
end

mutable struct DistributionMarkovProcess
	dist::Distribution
	r::Union{Float64, Nothing}
	DistributionMarkovProcess(dist) = new(dist, nothing)
end

function evolve!(dist_markov_process::DistributionMarkovProcess)
	dist_markov_process.r = rand(dist_markov_process.dist)
end

function recieve_reward!(dist_markov_process::DistributionMarkovProcess)
	r = dist_markov_process.r
	dist_markov_process.r = nothing
	return r
end


function POMDPs.states(p::MultiArmedBandit) 
	return [1] # a single degenerate state.
end

POMDPs.actions(p::MultiArmedBandit) = 1:p.k # choose to pull exactly 1 of k arms

POMDPs.stateindex(p::MultiArmedBandit, s::State) = s
POMDPs.actionindex(p::MultiArmedBandit, a::Action) = a

POMDPs.discount(p::MultiArmedBandit) = p.γ

function POMDPs.transition(p::MultiArmedBandit, s::State, a::Action) 	
	# pull an arm, transition tjhe arm's underlying markov process.
	evolve!(p.arms[a])

	# remain in the degenerate state.
	return Deterministic(s)
end
	
function POMDPs.reward(p::MultiArmedBandit, s::State, a::Action)
	r = recieve_reward!(p.arms[a])
	return r
end

function POMDPs.initialstate(p::MultiArmedBandit)
	return Deterministic(1)
end 