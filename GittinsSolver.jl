include("DistributionMultiArmedBandit.jl")

using POMDPs
using Distributions

struct GittinsSolver <: Solver end

struct GittinsPlanner <: Policy
    multi_armed_bandit::MultiArmedBandit
end

function POMDPs.solve(sol::GittinsSolver, multi_armed_bandit::MultiArmedBandit)
    return MonteCarloGreedyPlanner(multi_armed_bandit)
end

function POMDPs.action(planner::MonteCarloGreedyPlanner, s)
    gis = []
    for arm in planner.multi_armed_bandit.arms:
        if typeof(arm) == DistributionMarkovProcess
            dist = arm.dist
            if typeof(dist) == Normal
                break
            elseif typeof(dist) == Bernoulli
                break
            end
        end
    end
end