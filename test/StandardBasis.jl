using Test
using MultilinearMaps
using LinearAlgebra
using StaticArrays

@testset "Unit Vectors" begin
    e1 = StandardBasis(1)
    e2 = StandardBasis(2)
    e3 = StandardBasis(3)

    @testset "Construction" begin
        @test size(e2[1]) == (2,)
        @test length(e2[1]) == only(size(e2[1]))
        @test_throws DomainError e1[0]
        @test_throws DomainError e2[3]
        @test only(e1[1])
    end
    @testset "Equality" begin
        @test e2[1] == e2[1]
        @test e2[1] !== e2[2]
        @test e2[1] !== e3[1]
        @test e2[1] == Bool[true, false]
        @test e2[1] !== Bool[true, false, false]
    end
    @testset "Dot product" begin
        @test @inferred e1[1] ⋅ e1[1]
        @test e2[1] ⋅ e2[1]
        @test !(e2[1] ⋅ e2[2])
        @test !(e2[2] ⋅ e2[1])
        @test e2[1] ⋅ [1,2] == [1,2] ⋅ e2[1] == 1
        @test e2[2] ⋅ [1,2] == [1,2] ⋅ e2[2] == 2
        @test e2[1] ⋅ SVector(1,2) == SVector(1,2) ⋅ e2[1] == 1
        @test e2[2] ⋅ [1,2] == [1,2] ⋅ e2[2] == 2
        @test_throws DimensionMismatch e2[1] ⋅ e1[1]
        @test_throws DimensionMismatch SVector(1,2) ⋅ e1[1]
        @test_throws DimensionMismatch [1,2] ⋅ e1[1]
    end
    # Other
    @test @inferred(e2[2] + [1,0]) == ones(2)
    @test SVector{2}(e2[1] + e2[2]) === ones(SVector{2,eltype(true+true)})
    @test_broken SVector(e2[1] + e2[2]) === ones(SVector{2,eltype(true+true)})
end;
