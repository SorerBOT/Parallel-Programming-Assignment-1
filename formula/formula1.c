#include <xmmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pmmintrin.h>

typedef float FLOAT;
typedef enum
{
        false,
        true
} boolean;
typedef struct
{
        float sqrt_sum;
        float pow_prod;
} state;

#define MAX_STR             255
#define CHUNK_SIZE          16
#define FLOATS_PER_CHUNK    (CHUNK_SIZE / (sizeof(FLOAT)))

#define IS_DEBUG_P true
/*
 * functions intended for testings, activated depending on IS_DEBUG_P
 */
float get_res_from_state(state s);
boolean formula1_compare_states(state first, state second);
state formula1_naive(const float *x, unsigned int length);

/*
 * List of functions and their respective use-cases:
 * - _mm_sqrt_ps    : performs a packed square root operation on single floats
 * - _mm_mul_ps     : performs a packed multiplication on single floats
 * - _mm_add_ps     : performs a packed addition on single floats
 * - _mm_hadd_ps    : sums pairs of adjacent single precision floats
 *                    in a packed vector (used in the sum helper function)
 * - _mm_cvtss_f32  : retrieves the 4 rightmost bytes of a vector, parsing them as a float
 */
float packed_horizontal_sum(const __m128 a)
{
    __m128 intermediary = _mm_hadd_ps(a, a);
    __m128 result       = _mm_hadd_ps(intermediary, intermediary);
    return                _mm_cvtss_f32(result);
}

/*
 * There is no equivalent function to _mm_hadd_ps and so I am
 *                       going to take the following approach :
 * 1. Given 'a':                (float_1, float_2,           float_3,                     float_4)
 * 2. Shuffle 'a' to get:       (float_2, float_1,           float_4,                     float_3)
 * 3. Multiply the two:         (XXX,     float_1 * float_2, YYY,               float_3 * float_4)
 * 4. Shuffle the result to get (XXX,     YYY,               float_3 * float_4, float_1 * float_2)
 * 5. Multiply the two:         (XXX,     YYY,               ZZZ,               float_1 * float_2 * float_3 * float_4)
 * 6. Retrieve the last float using _mm_cvtss_f32.
 */
float packed_horizontal_mul(const __m128 a)
{
    __m128 perm         = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 intermediary = _mm_mul_ps(a, perm);
    __m128 perm_2       = _mm_shuffle_ps(intermediary, intermediary, _MM_SHUFFLE(0, 1, 2, 3));
    __m128 result       = _mm_mul_ps(intermediary, perm_2);
    return                _mm_cvtss_f32(result);
}
float formula1(float *x, unsigned int length)
{
        const float* first_address      = x;
    const size_t iters                  = length / FLOATS_PER_CHUNK;
    float sqrt_sum                      = 0.f;
    float pow_prod                      = 1.f;

    const __m128 ones   = _mm_set_ps1(1.f);
    for (size_t i = 0; i < iters; ++i, x += FLOATS_PER_CHUNK)
    {
        const __m128 chunk      = _mm_loadu_ps(x);
        const __m128 sqrt       = _mm_sqrt_ps(chunk);
        sqrt_sum               += packed_horizontal_sum(sqrt);

        const __m128 pow        = _mm_mul_ps(chunk, chunk);
        const __m128 add        = _mm_add_ps(pow, ones);
        pow_prod               *= packed_horizontal_mul(add);

#ifdef IS_DEBUG_P
                size_t elements_passed = FLOATS_PER_CHUNK * (i + 1);
                if (formula1_compare_states((state) { .sqrt_sum = sqrt_sum, .pow_prod = pow_prod }, formula1_naive(first_address, elements_passed)))
                {
                                printf( "(intermediary) maximal gap after %lu elements, terminated calculation\n", elements_passed );
                                exit(1);
                }
#endif
    }

    const size_t remainder      = length % FLOATS_PER_CHUNK;
    if (remainder != 0)
    {
        float x_chunk_padded[FLOATS_PER_CHUNK] = { 0.f };
        for (size_t i = 0; i < remainder; ++i)
        {
            x_chunk_padded[i] = x[i];
        }
        const __m128 chunk      = _mm_loadu_ps(x_chunk_padded);
        const __m128 sqrt       = _mm_sqrt_ps(chunk);
        sqrt_sum               += packed_horizontal_sum(sqrt);

        const __m128 pow        = _mm_mul_ps(chunk, chunk);
        const __m128 add        = _mm_add_ps(pow, ones);
        pow_prod               *= packed_horizontal_mul(add);
    }

#ifdef IS_DEBUG_P
        size_t elements_passed = length;
        if (formula1_compare_states((state) { .sqrt_sum = sqrt_sum, .pow_prod = pow_prod }, formula1_naive(first_address, elements_passed)))
        {
                        printf( "(end) maximal gap after %lu elements, terminated calculation\n", elements_passed );
                        exit(1);
        }
#endif

    return                        sqrtf( 1 + cbrtf(sqrt_sum) / pow_prod );
}

state formula1_naive(const float *x, unsigned int length)
{
        float sqrt_sum = 0.f;
        float pow_prod = 1.f;
        for (size_t i = 0; i < length; ++i)
        {
                sqrt_sum += sqrtf(x[i]);
                pow_prod *= (1 + x[i] * x[i]);
        }
        return (state)
        {
                .sqrt_sum = sqrt_sum,
                .pow_prod = pow_prod
        };
}

float get_res_from_state(state s)
{
        float res = sqrtf(1 + cbrtf(s.sqrt_sum) / s.pow_prod);
        return res;
}

boolean formula1_compare_states(state first, state second)
{
        boolean failed = false;
        if (fabsf(first.sqrt_sum - second.sqrt_sum) > 0.01)
        {
                printf( "sqrt_sum max difference allowed exceeded.\n" );
                failed = true;
        }
        if (fabsf(first.pow_prod - second.pow_prod) > 0.01)
        {
                printf( "pow_prod max difference allowed exceeded.\n" );
                failed = true;
        }
        if (fabsf(get_res_from_state(first) - get_res_from_state(second)) > 0.01)
        {
                printf( "res max difference allowed exceeded.\n" );
                failed = true;
        }

        return  failed;
}

