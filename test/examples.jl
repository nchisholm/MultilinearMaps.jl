@inline _skew(v1, v2, v3, v4) = ((v1⋅v3)*(v2⋅v4) - (v1⋅v4)*(v2⋅v3))
@inline _skew(vs::NTuple{4}) = _skew(vs...)

# The Kronecker delta, δ_{ij}, as a function of indices
inner_ixfn(i, j) = (i==j)
@inline inner_ixfn(args::Dims{2}) = inner_ixfn(args...)
@inline inner_ixfn(I::CartesianIndex{2}) = inner_ixfn(Tuple(I)...)

# A 4th order isotropic tensor, δ_{ik} δ_{jl} - δ_{il} δ_{jk}
skew_ixfn(i, j, k, l) = (i==k) * (j==l) - (i==l) * (j==k)
@inline skew_ixfn(args::Dims{4}) = skew_ixfn(args...)
@inline skew_ixfn(I::CartesianIndex{4}) = skew_ixfn(Tuple(I)...)
