struct ScalarMultiple{K, D, S<:Number, M<:AbstractMultilinearForm{K,D}} <: AbstractMultilinearForm{K,D}
    scalar::S
    mf::M
end

const TupleN{T,N} = NTuple{N,T}

ScalarMultiple(s::Number, mf::AbstractMultilinearForm{K,D}) where {K,D} =
    ScalarMultiple{K, D, typeof(s), typeof(mf)}

@inline (smf::ScalarMultiple)(args...) = smf.scalar * smf.mf(args...)

*(s::Number, mf::AbstractMultilinearForm) = ScalarMultiple(s, mf)
*(mf::AbstractMultilinearForm, s::Number) = ScalarMultiple(s, mf)

struct LinearCombo{K, D, Ms<:TupleN{AbstractMultilinearForm{K,D}}} <: AbstractMultilinearForm{K,D}
    smf::Ms
end

LinearCombo(mfs::Vararg{AbstractMultilinearForm{K,D}}) where {K,D} =
    LinearCombo{K,D,typeof(mfs)}(mfs)

+(mfs::Vararg{Vararg{AbstractMultilinearForm{K,D}}}) where {K,D} = LinearCombo(mfs...)
