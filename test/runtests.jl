using Test

using MultilinearMaps

import ArrayInterface as Arr
using LinearAlgebra
using StaticArrays

const MM = MultilinearMaps


@testset "Unit Vectors" begin
    e = StdUnitVector
    @testset "Construction" begin
        @test Arr.size(e{2}(1)) isa MM.SizeS{1}
        @test length(e{2}(1)) == only(size(e{2}(1)))
        @test_throws DomainError e{2}(3)
        @test_throws DomainError e{1}(0)
        @test only(e{1}(1))
    end
    @testset "Equality" begin
        @test e{2}(1) == e{2}(1)
        @test e{2}(1) !== e{2}(2)
        @test e{2}(1) !== e{3}(1)
        @test e{2}(1) == Bool[true, false]
        @test e{2}(1) !== Bool[true, false, false]
    end
    @testset "Dot product" begin
        @test @inferred e{1}(1) ⋅ e{1}(1)
        @test e{2}(1) ⋅ e{2}(1)
        @test !(e{2}(1) ⋅ e{2}(2))
        @test !(e{2}(2) ⋅ e{2}(1))
        @test e{2}(1) ⋅ [1,2] == [1,2] ⋅ e{2}(1) == 1
        @test e{2}(2) ⋅ [1,2] == [1,2] ⋅ e{2}(2) == 2
        @test e{2}(1) ⋅ SVector(1,2) == SVector(1,2) ⋅ e{2}(1) == 1
        @test e{2}(2) ⋅ [1,2] == [1,2] ⋅ e{2}(2) == 2
        @test_throws DimensionMismatch e{2}(1) ⋅ e{1}(1)
        @test_throws DimensionMismatch SVector(1,2) ⋅ e{1}(1)
        @test_throws DimensionMismatch [1,2] ⋅ e{1}(1)
    end
    # Other
    @test @inferred(e{2}(2) + [1,0]) == ones(2)
    @test SVector{2}(e{2}(1) + e{2}(2)) === ones(SVector{2,eltype(true+true)})
    @test_broken SVector(e{2}(1) + e{2}(2)) === ones(SVector{2,eltype(true+true)})
end

@testset "ApplyMode Trait" begin
    @test MM.ApplyMode() === MM.FullApply()
    @test MM.ApplyMode([1,2], [3,4,5], [5,6]) === MM.FullApply()
    @test MM.ApplyMode([1,2,3,4], :, [5,6]) === MM.PartialApply()
end

include("examples.jl")

const ê = StdUnitVector  # For convenience
_just_true() = true
const solo = @inferred MultilinearForm{0}(_just_true)
const inner = @inferred MultilinearForm{2,3}(dot)
const skew = @inferred MultilinearForm{4,3}(_skew)

@testset "AtomicMultilinearMap" begin
    u = StdUnitVector{2}(1) # SVector(1., 0.)
    v = StdUnitVector{2}(2) # SVector(0., 1.)
    inner2d = @inferred MultilinearForm{2,2}(dot)
    skew2d = @inferred MultilinearForm{4,2}(_skew)
    @test solo() == true
    @test_throws ArgumentError solo(u)
    @test inner2d(u,u) == 1
    @test inner2d(u,v) == 0
    @test inner2d(v,u) == 0
    @test_throws ArgumentError inner2d(u)        # too few
    @test_throws ArgumentError inner2d(u, v, u)  # too many
    @test skew2d(u,u,v,v) == 0
    @test skew2d(u,v,u,v) == 1
    @test skew2d(u,v,v,u) == -1
end

@testset "Vector Space Algebra" begin
    @testset "Equality" begin
        e = StdBasis{3}(Real)
        @test inner == inner
        @test inner != skew && skew != inner
        @test skew(e[1], :, e[2], :) != inner && inner != skew(e[1], :, e[2], :)
    end
    @testset "Scalar Multiples" begin
        @test MM.ScalarMultiple(inner, 0.5) == 0.5 * inner == inner / 2
        @test inner !== inner / 2
        @test MM.ScalarMultiple(inner, 1//2) == inner // 2 == 1//2 * inner
    end
    @testset "Sums" begin
        # Associativity
        @test (inner + inner) + inner == inner + (inner + inner) == inner + inner + inner
        # Can't add maps of unequal sizes (should probably give a more helpful exception)
        @test_throws DimensionMismatch inner + skew
    end;
    @testset "Linear Combinations" begin
        @test all(==(0), skew - skew)
        @test inner + inner == 2 * inner
        @test inner + inner + inner == 2*inner + inner == inner + 2*inner == 3*inner
        @test 2*(skew + skew) / 2 == 2*skew
    end
end;

@testset "PartialMap" begin
    @testset "Component equality" begin
        skew_components = SArray(skew)  # Materialize the whole tensor
        # Now, slice the component array and compare it to tensor contraction
        # with the unit vectors
        @test SArray(skew(ê{3}(1), :, ê{3}(2), :)) == skew_components[1,:,2,:]
        @test SArray(skew(:, :, ê{3}(3), ê{3}(2))) == skew_components[:,:,3,2]
    end
    @testset "2d" begin
        e = StdUnitVector{2}(1)
        inner = @inferred MultilinearForm{2,2}(dot)
        @test_throws ArgumentError inner(:)      # too few
        @test_throws ArgumentError inner(:,:,:)  # too many
        @test inner(:,:) === inner
        @inferred inner(e,:)
        @test inner(:,e) == inner(e,:)           # symmetry of `inner`
    end
    @testset "3d" begin
        e = StdBasis{3}(Real)
        (u,v,w,x) = ntuple(_ -> rand(SVector{3,Float64}), Val(4))
        # inner = @inferred MultilinearForm{2,3}(dot)
        # skew = @inferred MultilinearForm{4,3}(_skew)
        @inferred skew(u,v,w,:)
        @inferred skew(u,v,w,:)(x)
        @test inner(u,v) == inner(u,:)(v) == inner(:,u)(v) == inner(:,:)(u,v)
        @test skew(u,v,w,x) ≈ skew(u,v,w,:)(x) ≈ skew(u,v,:,:)(w,x) ≈
            skew(u,:,:,:)(v,w,x) ≈ skew(:,v,w,x)(u)
        @test eltype(inner(e[1], :)) == eltype(e[1])
        @test eltype(inner(u, :)) == eltype(u)
        @test eltype(skew(e[1], e[2], :, e[3])) == Int
    end
end

# FIXME
# @testset "Sizes" begin
#     SA_like_inner = StaticArray{Tuple{3,3}}
#     @test MM._size(SA_like_inner) === MM._size(inner)
#     @test MM.samesize(SA_like_inner, inner) === Arr.size(inner) ==
#         Arr.known_size(SA_like_inner)
#     @test_throws DimensionMismatch MM.samesize(skew, SA_like_inner)
# end;

@testset "StaticArrays traits" begin
    @test StaticArrays.Length(inner) == StaticArrays.Length(3^2)
    @test StaticArrays.Length(skew) == StaticArrays.Length(3^4)
    @test StaticArrays.Size(inner) == StaticArrays.Size(3,3)
    @test StaticArrays.Size(skew) == StaticArrays.Size(3,3,3,3)
end


include("harmonics.jl")

"""Test (recursively) if an array is traceless in every pair of indices"""
istraceless(A::AbstractArray{<:Any, 0}, _::Int) = true
istraceless(A::AbstractArray{<:Any, 1}, _::Int) = true
istraceless(A::AbstractArray{<:Any, 2}, _::Int) =
    ≈(tr(A), 0, atol=√(eps(eltype(A))))
istraceless(A::AbstractArray, dim::Int) =
    all(istraceless(B) for B in eachslice(A, dims=dim))
    # For dim = 1, does
    # all(≈(tr(out[i,:,:]), 0, atol=eps(eltype(out))) for i ∈ axes(out, 1))
istraceless(A::AbstractArray) = all(istraceless(A, dim) for dim ∈ 1:ndims(A))

_issymmetric(A::AbstractArray{<:Any, 0}) = true
_issymmetric(A::AbstractArray{<:Any, 1}) = true
_issymmetric(A::AbstractArray{<:Any, 2}) =
    all(≈(A[i,j] - A[j,i], 0, atol=√(eps(eltype(A)))) for i ∈ axes(A,1), j ∈ axes(A,2))
# _issymmetric(A::AbstractArray, dim) = all(issymmetric(B) for B in eachslice(A, dims=dim))
# _issymmetric(A::AbstractArray) = all(issymmetric(A, dim) for dim in 1:ndims(A))

@testset "Harmonics" begin
    x = normalize(rand(SVector{3,Float64}))
    ê = StdBasis{3}(Real)
    @testset "Traceless" begin
        for formfield in (sphharm30, sphharm31, sphharm32, sphharm33)
            form = formfield(x)
            K = ndims(form)
            D = Arr.size(form, 1)
            out = SArray(form)
            @test ndims(out) == K
            @test all(==(D), size(out))
            @test istraceless(out)
        end
    end
    @testset "Symmetric" begin
        @test issymmetric(SArray(sphharm32(x)))
        for i ∈ 1:3
            @test _issymmetric(SArray(sphharm33(x)(:,:, ê[i])))
            @test _issymmetric(SArray(sphharm33(x)(:, ê[i], :)))
            # Needed? I think implied by the previous two
            @test _issymmetric(SArray(sphharm33(x)(ê[i], :, :)))
        end
    end
end;
