using Plots, DelimitedFiles

include("../src/rbs_kernel.jl")
include("../src/block_srht.jl")
include("../src/gaussian_sampling.jl")

function main(data_name::String, l::Int)
  if (data_name == "mnist")
    dataset_filename = "/home/mbeauper/randomization/datasets/mnist/mnist_pot12_780"
    σ::Float64 = 100
    Ap = generate_darray(dataset_filename, σ, sep=' ', proc_grid_dims=[nworkers(),1], global_size=[4096,1024])
  elseif (data_name == "rand")
    Ap = drand((4096,1024), workers(), (4,1))
  elseif (data_name == "heat")
    io = open("/home/mbeauper/code/matrix_generation/binaries/heat.bin", "r")
    A = Array{Float64,2}(undef, (1000, 1000))
    read!(io, A)
  end

  plot_size::Int = 30

  Bg = sketch_left_2D_gaussian(Ap, l, TimerOutput())
  Bs = sketch_left_2D(Ap, l, TimerOutput())


  pairs = [rand(1:size(Bg)[2], 2) for i in 1:plot_size]

  A_pairs = [abs(dot(Ap[:,x[1]], Ap[:,x[2]])) for x in pairs]
  Bg_pairs = [abs(dot(Bg[:,x[1]], Bg[:,x[2]])) for x in pairs]
  Bs_pairs = [abs(dot(Bs[:,x[1]], Bs[:,x[2]])) for x in pairs]

  f = mean(A_pairs) / mean(Bg_pairs)
  fs = mean(A_pairs) / mean(Bs_pairs)

  println("difference : $f (gaussian) and $fs (bsrht)")

  writedlm("data/original_$(data_name).csv", [1:plot_size A_pairs], ',')
  writedlm("data/l_$(l)_gaussian_$(data_name).csv", [1:plot_size Bg_pairs * f], ',')
  writedlm("data/l_$(l)_bsrht_$(data_name).csv", [1:plot_size Bs_pairs * fs], ',')

  plot([A_pairs Bg_pairs .* f Bs_pairs .* fs], labels=["original" "gaussian" "bsrht"])
  savefig("innerproducts_l$(l)_$(data_name).pdf")
end

for d in ["mnist" "rand"]
  for l in [20 50 100 200 500]
    main(d, l)
  end
end
