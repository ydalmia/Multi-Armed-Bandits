"""
This module implements the MultiArmedBandit MDP and the corresponding Gittins index solver 
using the interface defined in POMDPs.jl

"""

module MultiArmedBandits

using POMDPs
using POMDPTools
using NLsolve
using JuMP
using HiGHS
using Random
using LinearAlgebra
using Test

include("gittins_index_online_algorithms.jl")
include("MultiArmedBanditMDP.jl")
include("gittins_mab_solver.jl")
export
    MAB,
    RandomMAB,
    BanditProcess,
    ChenKatehakisLinearProgramming,
    KatehakisVeinottRestartFormulation,
    GittinsIndexPolicy,
    GittinsIndexSolver,
    solve
end