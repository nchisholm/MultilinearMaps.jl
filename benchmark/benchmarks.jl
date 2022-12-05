using BenchmarkTools
using LinearAlgebra
using Static
using StaticArrays
using Base.Cartesian
using MultilinearMaps


# A 4th order isotropic tensor, δ_{ik} δ_{jl} - δ_{il} δ_{jk}
skew_ixfn(i, j, k, l) = (i==k) * (j==l) - (i==l) * (j==k)
@inline skew_ixfn(args::Dims{4}) = skew_ixfn(args...)
@inline skew_ixfn(I::CartesianIndex{4}) = skew_ixfn(Tuple(I)...)

@inline _skew(v1, v2, v3, v4) = ((v1⋅v3)*(v2⋅v4) - (v1⋅v4)*(v2⋅v3))
@inline _skew(vs::NTuple{4}) = _skew(vs...)

@inline inds2uvecs(inds::Vararg{Int}) = map(i -> StandardUnitVector(i, 3), inds)

@generated function fillfn_nloops!(arr::AbstractArray{<:Any,N}, f::F) where {N,F}
    quote
        @nloops $N i arr begin
            @inbounds (@nref $N arr i) = (@ncall $N f i)
        end
        arr
    end
end

@generated function fill_nloops!(arr::AbstractArray{<:Any,N}, src::F) where {N,F}
    quote
        @nloops $N i arr begin
            @inbounds (@nref $N arr i) = (@nref $N src i)
        end
        arr
    end
end

const onlytrue = MultilinearMap(() -> true, ())
const inner = MultilinearMap(dot, static((3,3)))
const skew = MultilinearMap(_skew, static((3,3,3,3)))

# println("Function of indices")
# @btime fillfn_nloops!(A, skew_ixfn) setup=(A = MArray{NTuple{4,3},Int64}(undef))
# println("(Bare) function of unit vectors")
# @btime(fillfn_nloops!(A, _skew ∘ inds2uvecs), setup=(A = MArray{NTuple{4,3},Int64}(undef)))
# println("Function of unit vectors")
# @btime(fillfn_nloops!(A, Base.splat(skew) ∘ inds2uvecs), setup=(A = MArray{NTuple{4,3},Int64}(undef)))
# println("Function of unit vectors (indexing)")
# @btime(fill_nloops!(A, skew), setup=(A = MArray{NTuple{4,3},Int64}(undef)))
# println("sacollect")
# @btime sacollect(SArray{NTuple{4,3}}, skew)

const SUITE = BenchmarkGroup(
    [],
    "Baseline (function of indices)" =>
        @benchmarkable(fillfn_nloops!(A, skew_ixfn), setup=(A = MArray{NTuple{4,3},Int64}(undef))),
    "Function of `StandardUnitVector`s" =>
        @benchmarkable(fillfn_nloops!(A, _skew ∘ inds2uvecs), setup=(A = MArray{NTuple{4,3},Int64}(undef))),
    "MultilinearMap (loops, indexing)" =>
        @benchmarkable(fill_nloops!(A, skew), setup=(A = MArray{NTuple{4,3},Int64}(undef))),
    "MultilinearMap (materialize)" =>
        @benchmarkable(materialize!(A, skew), setup=(A = MArray{NTuple{4,3},Int64}(undef))),
    "MultilinearMap (iteration)" =>
        @benchmarkable(StaticArrays.sacollect(SArray{NTuple{4,3}}, skew))
)
