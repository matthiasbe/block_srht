include("test_utils.jl")
setnworkers(8)

include("../src/random_matrix.jl")

using Test

@testset "orthogonality" begin
  # Testing orthonormalize()
  U = drand((2^10,2^10), workers(), [1,nworkers()])

  @assert abs(det(Matrix(U))) > 10e-10

  orthonormalize(U)

  for p in workers()
    U_p = @fetchfrom p U[:l]
    for p2 in workers()
      U_p2 = @fetchfrom p2 U[:l]
      if p == p2
        @test norm(U_p' * U_p2, Inf) ≈ 1 atol=10e-10
      else
        @test norm(U_p' * U_p2, Inf) ≈ 0 atol=10e-10
      end
    end
  end

  close(U)

  # Testing generate_exp_decay_distributed()
  V,S = generate_exp_decay_distributed(10, 2^10)
  @test S ≈ LinearAlgebra.svd(convert(Array, V)).S atol=10e-10
  close(V)

  # Testing generate_exp_decay_distributed()
  V,S = generate_exp_decay_distributed(10, 2^11, n=2^10)
  @test S ≈ LinearAlgebra.svd(convert(Array, V)).S atol=10e-10
  close(V)

  ## Testing generate_exp_decay_distributed()
  V,S = generate_exp_decay_distributed(10, 2^10, n=2^11)
  @test S ≈ LinearAlgebra.svd(convert(Array, V)).S atol=10e-10
  close(V)
end

@testset "transpose" begin
  A = drand((1000,2000), workers(), [1,nworkers()])
  B = darray_transpose_layout(A)

  @test procs(B) == procs(A)'
  @test Matrix(A) == Matrix(B)
end
