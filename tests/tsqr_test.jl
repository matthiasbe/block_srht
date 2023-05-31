include("../src/tsqr.jl")

using Test

@testset "TSQR" begin
  # Valid test
  A = drand((1000,250), workers(), [nworkers(), 1])
  R_tsqr = tsqr(A)
  Q = Matrix(A) / R_tsqr
  @test norm(I - Q'*Q) ≈ 0 atol=10e-10

  # Valid test : tall blocks
  A = drand((2000,250), workers(), [nworkers(), 1])
  R_tsqr = tsqr(A)
  Q = Matrix(A) / R_tsqr
  @test norm(I - Q'*Q) ≈ 0 atol=10e-10

  # Unvalid test : Square distribution grid
  P = Int(floor(nworkers() / 2))
  A = drand((1000,250), workers(), [P,P])
  @test_throws AssertionError tsqr(A)

  # Unvalid test : blocks are a tiny bit flat
  A = drand((1000,251), workers(), [nworkers(), 1])
  @test_throws AssertionError tsqr(A)
end
