include("../src/srht.jl")
include("../src/io.jl")

using TimerOutputs
using LinearAlgebra
using DelimitedFiles
using Test

@testset "srht local" begin
  to = TimerOutput()

  k = 50
  l = 2^6
  m = 2^10

  A = permutedims(Matrix(read_darray("generated_matrices/matrix_6x15_k50", procs=[2,3,4,5], dims=[1,4])))

  A_sketched = srht(A, l, MersenneTwister(), to)

  S = readdlm("generated_matrices/matrix_6x15_k50/original_singular_values.csv")[1:k]
  S2 = svd(A_sketched).S

  @test S ≈ S2[1:k] rtol=0.4
end
