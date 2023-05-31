include("test_utils.jl")
setnworkers(16)

include("../src/block_srht.jl")
include("../src/eval.jl")
include("../src/io.jl")

using TimerOutputs
using Test

@testset "srht" begin
  to = TimerOutput()

  k = 10
  l = 2^7

  A = read_darray("generated_matrices/matrix_6x15_k50", dims=[1,nworkers()])
  S = readdlm("generated_matrices/matrix_6x15_k50/original_singular_values.csv")[1:k]
  
  # Test 2 times for correct use of RNG

  A_sketched = sketch_right(A, l, to)
  S1 = sketch_right_svd_eval(A, A_sketched, k=k, method="svd")
  @test S ≈ S1[1:k] rtol = 0.25

  A_sketched = sketch_right(A, l, to)
  S1 = sketch_right_svd_eval(A, A_sketched, k=k, method="svd")
  @test S ≈ S1[1:k] rtol = 0.25

  close(A)
end

@testset "2D sketching" begin
  to = TimerOutput()

  k = 10
  l = 10

  @assert l >= k

  # Test: square matrix
  A = read_darray("generated_matrices/matrix_12x12_k50")
  S = readdlm("generated_matrices/matrix_12x12_k50/original_singular_values.csv")[1:k]

  A_sketched = sketch_2D(A, (l,l), to)
  A_sketched_eachblock = sketch_2D_eachblock(A, (l,l), to)
  S1 = sketch_right_svd_eval(A, A_sketched, k=k, method="svd")
  S1_eachblock = sketch_right_svd_eval(A, A_sketched_eachblock, k=k, method="svd")
  @test S ≈ S1[1:k] rtol = 0.001
  @test S ≈ S1_eachblock[1:k] rtol = 0.001
end
