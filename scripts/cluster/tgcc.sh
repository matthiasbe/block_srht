
#MSUB -o runs/%I.out
#MSUB -e runs/%I.out
#MSUB -r srht
#MSUB -q skylake
#MSUB -x  # Exclusif
#MSUB -T 7200 #time_limit

julia_exe=/ccc/cont003/dsku/blanchet/home/user/unisorbo/beauperm/julia-1.7.3/bin/julia

set -x
cd ${BRIDGE_MSUB_PWD}

$julia_exe scripts/nystrom.jl
