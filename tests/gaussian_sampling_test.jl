using TimerOutputs
using DelimitedFiles
using Test

include("test_utils.jl")
setnworkers(4)

include("../src/gaussian_sampling.jl")
include("../src/eval.jl")
include("../src/io.jl")



@testset "gaussian 1D" begin
  to = TimerOutput()

  k = 10
  l = 2^7
  m = 2^10

  A = read_darray("generated_matrices/matrix_6x15_k50", dims=[1,nworkers()])
  S = readdlm("generated_matrices/matrix_6x15_k50/original_singular_values.csv")[1:k]

  A_sketched = sketch_right_gaussian(A, l, to)
  S1 = sketch_right_svd_eval(A, A_sketched, k=k, method="svd")
  @test S ≈ S1[1:k] rtol = 0.2

  A_sketched = sketch_right_gaussian(A, l, to)
  S1 = sketch_right_svd_eval(A, A_sketched, k=k, method="svd")
  @test S ≈ S1[1:k] rtol = 0.2

  close(A)
end


@testset "gaussian 2D" begin
  to = TimerOutput()

  k = 10
  l = 2^7
  m = 2^10

  A = read_darray("generated_matrices/matrix_10x10_k50")
  S = readdlm("generated_matrices/matrix_10x10_k50/original_singular_values.csv")[1:k]

  A_sketched = sketch_left_2D_gaussian(A, l, to)
  _,S1,_ = svd(Matrix(A_sketched))
  @test S ≈ S1[1:k] rtol = 0.2

  close(A)
end

@testset "gaussian sampling conserves symmetry" begin
  A = rand(128,128)
  A = A*A'
  r = rand(UInt)
  B = sketch_right_gaussian(copy(A), 32, seed=r)
  C = sketch_right_gaussian(copy(B'), 32, seed=r)
  @test norm(C - C') ≈ 0 atol=1e-10
end

@testset "1D block gaussian sampling conserves symmetry" begin
  A = rand(Float64, 128,128)
  X = distribute(A*A', dist=[1,nworkers()])
  r = rand(UInt, nworkers())
  B = sketch_right_gaussian(X, 32, TimerOutput(), block_seeds=r)
  Bt = distribute(Matrix(B)', dist=[1, nworkers()])
  C = sketch_right_gaussian(Bt, 32, TimerOutput(), block_seeds=r)
  @test norm(C - C') ≈ 0 atol=1e-10
  @test reduce(min, eigvals(C)) >= 0.0
end

@testset "2D left and 1D right block gaussian sampling conserves symmetry" begin
  A = rand(Float64, 128,128)
  r = rand(UInt, nworkers())
  X = distribute(A)
  B = sketch_left_2D_gaussian(X, 2, TimerOutput(), block_seeds=r)
  X = distribute(A', dist=[1, size(procs(B))[2]])
  C = sketch_right_gaussian(X, 2, TimerOutput(), block_seeds=r)
  @test norm(C' - B) ≈ 0 atol=1e-13
end


@testset "2D left and 1D right block gaussian sampling conserves symmetry" begin
  A = rand(Float64, 128,128)
  X = distribute(A*A')
  r = rand(UInt, nworkers())
  B = sketch_left_2D_gaussian(X, 32, TimerOutput(), block_seeds=r)
  C = sketch_right_gaussian(B, 32, TimerOutput(), block_seeds=r)
  @test norm(C - C') ≈ 0 atol=1e-10
  @test reduce(min, eigvals(C)) >= 0.0
end

@testset "2D left block gaussian sampling 2 times" begin
  A = rand(Float64, 128,128)
  X = distribute(A*A')
  r = rand(UInt, nworkers())
  B = sketch_left_2D_gaussian(X, 2, TimerOutput(), block_seeds=r)
  C = sketch_left_2D_gaussian(X, 2, TimerOutput(), block_seeds=r)
  @test norm(Matrix(C) - Matrix(B)) == 0.0
end

@testset "2D left and right block gaussian sampling conserves symmetry" begin
  A = rand(Float64, 128,128)
  X = distribute(A*A')
  r = rand(UInt, nworkers())
  B = sketch_left_2D_gaussian(X, 2, TimerOutput(), block_seeds=r)
  Bt = distribute(Matrix(B)')
  Cp = sketch_left_2D_gaussian(Bt, 2, TimerOutput(), block_seeds=r)
  C = Matrix(Cp)
  @test norm(C - C') ≈ 0 atol=1e-10
  @test reduce(min, eigvals(C)) >= 0.0
end
