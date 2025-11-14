#include <emmintrin.h>   // SSE2
#include <smmintrin.h>   // SSE4.1
#include <string.h>

#define MAX_STR 256
#define CHUNK_SIZE 16
#define MATCH_VALUE 0xFF

/*
 * Algorithm:
 * Consider the strings as sequences of 8-bit numbers:
 * String 1: cat - 01100011 01100001 01110100
 * String 2: car - 01100011 01100001 01110010
 *
 * Running _mm_cmpeq_epi8 on a sequence of 8-bit numbers-
 * will return a sequence of 0-numbers and 0xFF-numbers
 * where 0 indicates that the corresponding numbers were equal-
 * and 0xFF indicates that they were not equal.
 * _mm_cmpeq_epi8: 00000000 00000000 11111111
 * Now we'd want to count the number of 0xFF's. So we're
 * going to use _mm_sad_epu8, and then we'll get 0xFF * (match-count)
 * and then dividing by 0xFF will get us the result.
 */

size_t strlen(const char* str)
{
    const char* init_address = str;
    while (*str != '\0')
    {
        ++str;
    }
    return str - init_address;
}

int abs(int x)
{
    return x > 0
        ? x
        : -x;
}

int max(int x, int y)
{
    return x > y ? x : y;
}

int min(int x, int y)
{
    return x > y ? y : x;
}

int hamming_dist_chunk(const char* str_ptr1, const char* str_ptr2)
{
    const __m128i str_xmm1         = _mm_loadu_si128((const __m128i_u *) str_ptr1);
    const __m128i str_xmm2         = _mm_loadu_si128((const __m128i_u *) str_ptr2);
    const __m128i empty_buffer_xmm = _mm_setzero_si128();

    const __m128i res              = _mm_cmpeq_epi8(str_xmm1, str_xmm2);
    const __m128i sum_matches      = _mm_sad_epu8(res, empty_buffer_xmm);
    const size_t  sum_lower        = _mm_extract_epi16(sum_matches, 0x0);
    const size_t  sum_upper        = _mm_extract_epi16(sum_matches, 0x4);
    return CHUNK_SIZE - (sum_lower + sum_upper) / MATCH_VALUE;
}

int hamming_dist(char str1[MAX_STR], char str2[MAX_STR])
{
    const size_t len1           = strlen(str1);
    const size_t len2           = strlen(str2);
    const size_t len_min        = min(len1, len2);

    size_t dist                 = abs((int)len1 - (int)len2);
    const size_t iters          = len_min / CHUNK_SIZE; // in case there's a remainder

    char* str_ptr1              = str1;
    char* str_ptr2              = str2;

    for (size_t i = 0;
            i < iters;
            str_ptr1 += CHUNK_SIZE,
            str_ptr2 += CHUNK_SIZE,
            ++i)
    {
        dist += hamming_dist_chunk(str_ptr1, str_ptr2);
    }

    if (len_min % CHUNK_SIZE > 0)
    {
        /*
         * Removing garbage from strings
         */
        const size_t current_len1     = strlen(str_ptr1);
        const size_t current_len2     = strlen(str_ptr2);
        const size_t current_len_min  = min(current_len1, current_len2);

        char buffer_tmp1[CHUNK_SIZE] = { 0 };
        char buffer_tmp2[CHUNK_SIZE] = { 0 };
        memcpy(buffer_tmp1, str_ptr1, current_len_min);
        memcpy(buffer_tmp2, str_ptr2, current_len_min);
        dist += hamming_dist_chunk(buffer_tmp1, buffer_tmp2);
    }

    return dist;
}
