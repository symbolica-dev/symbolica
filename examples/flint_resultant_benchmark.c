#define _POSIX_C_SOURCE 200809L

#include <flint/flint.h>
#include <flint/fmpz_mpoly.h>
#include <flint/nmod_mpoly.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct
{
    const char * name;
    const char * a;
    const char * b;
    int samples;
} benchmark_case;

static const benchmark_case cases[] = {
    {
        "dense outer degrees 7/6",
        "1+(2+y^2+z^3)*x+(3+y^3+z^2)*x^2+(4+y+z)*x^3+(5+y^2+z^3)*x^4+(6+y^3+z^2)*x^5+(7+y+z)*x^6+(8+y^2+z^3)*x^7",
        "1+(3+y^3-z^2)*x+(5+y^2-z^3)*x^2+(7+y-z)*x^3+(9+y^3-z^2)*x^4+(11+y^2-z^3)*x^5+(13+y-z)*x^6",
        7,
    },
    {
        "lacunary outer degrees 18/11",
        "(y+1)*x^18+(z+2)*x^13+(y*z+3)*x^7+(y^2-z)*x^2+1",
        "(z+1)*x^11+(y-2)*x^8+(y+z)*x^3+2",
        7,
    },
    {
        "nonunit leading degrees 9/7",
        "(y+1)*x^9+(z^2+2)*x^8+(y*z+1)*x^5+(y^2+z)*x^2+3",
        "(z+1)*x^7+(y^2-1)*x^6+(y+z+1)*x^3+z*x+2",
        7,
    },
    {
        "large high-height degrees 14/10",
        "(1000000000039+y^3+z^2)*x^14+(1000000000061+y*z^2-z^3)*x^10+(1000000000091+y*z+y)*x^6+(1000000000163+y^2*z^2+z)*x^2+1000000000169+y+z",
        "(1000000000187+z^2+y)*x^10+(1000000000193+y^3-z^2)*x^7+(1000000000223+y^2*z-z^3)*x^4+(1000000000241+y^2+z^2)*x+1000000000271-z",
        3,
    },
};

static double now_seconds(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double) time.tv_sec + (double) time.tv_nsec * 1.0e-9;
}

static int compare_double(const void * left, const void * right)
{
    double a = *(const double *) left;
    double b = *(const double *) right;
    return (a > b) - (a < b);
}

static int benchmark_selected(const char * name)
{
    const char * filter = getenv("BENCHMARK_FILTER");
    return filter == NULL || strstr(name, filter) != NULL;
}

static void compare_multiplication_with_options(const char * name,
                                                const char * a_string, ulong a_power,
                                                int subtract_one_from_a,
                                                const char * b_string, ulong b_power,
                                                int subtract_one_from_b,
                                                const char ** variables, slong nvariables,
                                                int default_samples)
{
    if (!benchmark_selected(name))
        return;

    const char * sample_override = getenv("MULTIPLICATION_BENCH_SAMPLES");
    int samples = sample_override ? atoi(sample_override) : default_samples;
    if (samples < 1)
        samples = 1;

    fmpz_mpoly_ctx_t context;
    fmpz_mpoly_t a_base, b_base, a, b, product;
    fmpz_mpoly_ctx_init(context, nvariables, ORD_LEX);
    fmpz_mpoly_init(a_base, context);
    fmpz_mpoly_init(b_base, context);
    fmpz_mpoly_init(a, context);
    fmpz_mpoly_init(b, context);
    fmpz_mpoly_init(product, context);

    if (fmpz_mpoly_set_str_pretty(a_base, a_string, variables, context) != 0 ||
        fmpz_mpoly_set_str_pretty(b_base, b_string, variables, context) != 0 ||
        !fmpz_mpoly_pow_ui(a, a_base, a_power, context) ||
        !fmpz_mpoly_pow_ui(b, b_base, b_power, context))
    {
        fprintf(stderr, "Could not construct multiplication case: %s\n", name);
        exit(1);
    }
    if (subtract_one_from_a)
        fmpz_mpoly_sub_ui(a, a, 1, context);
    if (subtract_one_from_b)
        fmpz_mpoly_sub_ui(b, b, 1, context);

    fmpz_mpoly_mul(product, a, b, context);
    double * timings = flint_malloc((size_t) samples * sizeof(double));
    for (int sample = 0; sample < samples; sample++)
    {
        double start = now_seconds();
        fmpz_mpoly_mul(product, a, b, context);
        timings[sample] = now_seconds() - start;
    }
    qsort(timings, (size_t) samples, sizeof(double), compare_double);

    printf("%-32s MUL   %9.3f ms  lhs/rhs/product terms %ld/%ld/%ld\n",
           name, timings[samples / 2] * 1000.0,
           fmpz_mpoly_length(a, context), fmpz_mpoly_length(b, context),
           fmpz_mpoly_length(product, context));
    fflush(stdout);

    flint_free(timings);
    fmpz_mpoly_clear(a_base, context);
    fmpz_mpoly_clear(b_base, context);
    fmpz_mpoly_clear(a, context);
    fmpz_mpoly_clear(b, context);
    fmpz_mpoly_clear(product, context);
    fmpz_mpoly_ctx_clear(context);
}

static void compare_multiplication(const char * name,
                                   const char * a_string, ulong a_power,
                                   const char * b_string, ulong b_power,
                                   int default_samples)
{
    const char * variables[] = {"x", "y", "z"};
    compare_multiplication_with_options(name,
                                        a_string, a_power, 0,
                                        b_string, b_power, 0,
                                        variables, 3, default_samples);
}

static void compare_finite_field_multiplication_with_options(
    const char * name, ulong modulus,
    const char * a_string, ulong a_power, int subtract_one_from_a,
    const char * b_string, ulong b_power, int subtract_one_from_b,
    const char ** variables, slong nvariables, int default_samples)
{
    if (!benchmark_selected(name))
        return;

    const char * sample_override = getenv("MULTIPLICATION_BENCH_SAMPLES");
    int samples = sample_override ? atoi(sample_override) : default_samples;
    if (samples < 1)
        samples = 1;

    nmod_mpoly_ctx_t context;
    nmod_mpoly_t a_base, b_base, a, b, product;
    nmod_mpoly_ctx_init(context, nvariables, ORD_LEX, modulus);
    nmod_mpoly_init(a_base, context);
    nmod_mpoly_init(b_base, context);
    nmod_mpoly_init(a, context);
    nmod_mpoly_init(b, context);
    nmod_mpoly_init(product, context);

    if (nmod_mpoly_set_str_pretty(a_base, a_string, variables, context) != 0 ||
        nmod_mpoly_set_str_pretty(b_base, b_string, variables, context) != 0 ||
        !nmod_mpoly_pow_ui(a, a_base, a_power, context) ||
        !nmod_mpoly_pow_ui(b, b_base, b_power, context))
    {
        fprintf(stderr, "Could not construct finite-field multiplication case: %s\n", name);
        exit(1);
    }
    if (subtract_one_from_a)
        nmod_mpoly_sub_ui(a, a, 1, context);
    if (subtract_one_from_b)
        nmod_mpoly_sub_ui(b, b, 1, context);

    nmod_mpoly_mul(product, a, b, context);
    double calibration_start = now_seconds();
    nmod_mpoly_mul(product, a, b, context);
    double calibration = now_seconds() - calibration_start;
    double batch_estimate = 0.020 / (calibration > 1e-9 ? calibration : 1e-9);
    int batch_size = batch_estimate > 256.0 ? 256 : (int) batch_estimate;
    if (batch_size < 1)
        batch_size = 1;

    double * timings = flint_malloc((size_t) samples * sizeof(double));
    for (int sample = 0; sample < samples; sample++)
    {
        double start = now_seconds();
        for (int batch = 0; batch < batch_size; batch++)
            nmod_mpoly_mul(product, a, b, context);
        timings[sample] = (now_seconds() - start) / batch_size;
    }
    qsort(timings, (size_t) samples, sizeof(double), compare_double);

    printf("%-48s MUL   %9.3f ms  lhs/rhs/product terms %ld/%ld/%ld\n",
           name, timings[samples / 2] * 1000.0,
           nmod_mpoly_length(a, context), nmod_mpoly_length(b, context),
           nmod_mpoly_length(product, context));
    fflush(stdout);

    flint_free(timings);
    nmod_mpoly_clear(a_base, context);
    nmod_mpoly_clear(b_base, context);
    nmod_mpoly_clear(a, context);
    nmod_mpoly_clear(b, context);
    nmod_mpoly_clear(product, context);
    nmod_mpoly_ctx_clear(context);
}

static void compare_finite_field_dense_univariate(
    const char * name, ulong modulus, ulong left_degree, ulong right_degree,
    int default_samples)
{
    if (!benchmark_selected(name))
        return;

    const char * sample_override = getenv("MULTIPLICATION_BENCH_SAMPLES");
    int samples = sample_override ? atoi(sample_override) : default_samples;
    if (samples < 1)
        samples = 1;

    nmod_mpoly_ctx_t context;
    nmod_mpoly_t left, right, product;
    nmod_mpoly_ctx_init(context, 1, ORD_LEX, modulus);
    nmod_mpoly_init(left, context);
    nmod_mpoly_init(right, context);
    nmod_mpoly_init(product, context);

    for (ulong exponent = 0; exponent <= left_degree; exponent++)
    {
        ulong exponents[] = {exponent};
        nmod_mpoly_push_term_ui_ui(left, exponent % 16 + 1, exponents, context);
    }
    for (ulong exponent = 0; exponent <= right_degree; exponent++)
    {
        ulong exponents[] = {exponent};
        nmod_mpoly_push_term_ui_ui(right, (7 * exponent) % 16 + 1, exponents, context);
    }
    nmod_mpoly_sort_terms(left, context);
    nmod_mpoly_sort_terms(right, context);

    nmod_mpoly_mul(product, left, right, context);
    double calibration_start = now_seconds();
    nmod_mpoly_mul(product, left, right, context);
    double calibration = now_seconds() - calibration_start;
    double batch_estimate = 0.020 / (calibration > 1e-9 ? calibration : 1e-9);
    int batch_size = batch_estimate > 256.0 ? 256 : (int) batch_estimate;
    if (batch_size < 1)
        batch_size = 1;

    double * timings = flint_malloc((size_t) samples * sizeof(double));
    for (int sample = 0; sample < samples; sample++)
    {
        double start = now_seconds();
        for (int batch = 0; batch < batch_size; batch++)
            nmod_mpoly_mul(product, left, right, context);
        timings[sample] = (now_seconds() - start) / batch_size;
    }
    qsort(timings, (size_t) samples, sizeof(double), compare_double);

    printf("%-48s MUL   %9.3f ms  lhs/rhs/product terms %ld/%ld/%ld\n",
           name, timings[samples / 2] * 1000.0,
           nmod_mpoly_length(left, context), nmod_mpoly_length(right, context),
           nmod_mpoly_length(product, context));
    fflush(stdout);

    flint_free(timings);
    nmod_mpoly_clear(left, context);
    nmod_mpoly_clear(right, context);
    nmod_mpoly_clear(product, context);
    nmod_mpoly_ctx_clear(context);
}

static void compare_finite_field_suite(const char * label, ulong modulus)
{
    const char * variables3[] = {"x", "y", "z"};
    const char * variables5[] = {"x1", "x2", "x3", "x4", "x5"};
    const char * variables7[] = {"x1", "x2", "x3", "x4", "x5", "x6", "x7"};
    char name[128];

    snprintf(name, sizeof(name), "%s dense univariate degree-4912 multiplication", label);
    compare_finite_field_dense_univariate(name, modulus, 4912, 4911, 3);

    snprintf(name, sizeof(name), "%s dense large multiplication", label);
    compare_finite_field_multiplication_with_options(
        name, modulus, "1+x+y+z", 24, 0, "1+2*x-y+3*z", 23, 0,
        variables3, 3, 5);

    snprintf(name, sizeof(name), "%s dense very large multiplication", label);
    compare_finite_field_multiplication_with_options(
        name, modulus, "1+x+y+z", 40, 0, "1+2*x-y+3*z", 39, 0,
        variables3, 3, 3);

    snprintf(name, sizeof(name), "%s five-variable total-degree multiplication", label);
    compare_finite_field_multiplication_with_options(
        name, modulus,
        "1+x1+2*x2+3*x3+4*x4+5*x5", 13, 1,
        "1+2*x1-3*x2+5*x3-7*x4+11*x5", 12, 1,
        variables5, 5, 3);

    snprintf(name, sizeof(name), "%s sparse large multiplication", label);
    compare_finite_field_multiplication_with_options(
        name, modulus,
        "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83", 7, 0,
        "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83", 7, 0,
        variables3, 3, 1);

    snprintf(name, sizeof(name), "%s seven-variable power-minus-one multiplication", label);
    compare_finite_field_multiplication_with_options(
        name, modulus,
        "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7", 7, 1,
        "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7", 7, 1,
        variables7, 7, 1);
}

static void compare_power_minus_one_square(const char * name,
                                           const char * base_string, ulong power,
                                           int default_samples)
{
    const char * variables[] = {"x1", "x2", "x3", "x4", "x5", "x6", "x7"};
    compare_multiplication_with_options(name,
                                        base_string, power, 1,
                                        base_string, power, 1,
                                        variables, 7, default_samples);
}

int main(void)
{
    const char * variables[] = {"x", "y", "z"};
    const char * sample_override = getenv("RESULTANT_BENCH_SAMPLES");
    flint_set_num_threads(1);
    printf("FLINT %s\n", FLINT_VERSION);

    compare_multiplication("dense small multiplication",
                           "1+x+y+z", 12,
                           "1+2*x-y+3*z", 11, 25);
    compare_multiplication("dense high multiplication",
                           "1000000000039+x+y+z", 12,
                           "1000000000187+2*x-y+3*z", 11, 10);
    compare_multiplication("dense large multiplication",
                           "1+x+y+z", 24,
                           "1+2*x-y+3*z", 23, 7);
    compare_multiplication("dense very large multiplication",
                           "1+x+y+z", 40,
                           "1+2*x-y+3*z", 39, 3);
    compare_multiplication("dense high large multiplication",
                           "1000000000039+x+y+z", 20,
                           "1000000000187+2*x-y+3*z", 19, 3);
    compare_multiplication("sparse separated multiplication",
                           "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47", 7,
                           "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47", 7, 3);
    compare_multiplication("sparse large multiplication",
                           "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83", 7,
                           "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83", 7, 1);
    compare_power_minus_one_square("seven-variable power-minus-one multiplication",
                                   "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7", 7, 1);
    compare_finite_field_suite("GF(17)", 17);
    compare_finite_field_suite("GF(18446744073709551557)",
                               (ulong) 18446744073709551557ULL);

    for (size_t case_index = 0; case_index < sizeof(cases) / sizeof(cases[0]); case_index++)
    {
        const benchmark_case * benchmark = cases + case_index;
        if (!benchmark_selected(benchmark->name))
            continue;

        int samples = sample_override ? atoi(sample_override) : benchmark->samples;
        if (samples < 1)
            samples = 1;

        fmpz_mpoly_ctx_t context;
        fmpz_mpoly_t a, b, resultant;
        fmpz_mpoly_ctx_init(context, 3, ORD_LEX);
        fmpz_mpoly_init(a, context);
        fmpz_mpoly_init(b, context);
        fmpz_mpoly_init(resultant, context);

        if (fmpz_mpoly_set_str_pretty(a, benchmark->a, variables, context) != 0 ||
            fmpz_mpoly_set_str_pretty(b, benchmark->b, variables, context) != 0)
        {
            fprintf(stderr, "Could not parse benchmark case: %s\n", benchmark->name);
            return 1;
        }

        if (!fmpz_mpoly_resultant(resultant, a, b, 0, context))
        {
            fprintf(stderr, "FLINT resultant failed: %s\n", benchmark->name);
            return 1;
        }

        double * timings = flint_malloc((size_t) samples * sizeof(double));
        for (int sample = 0; sample < samples; sample++)
        {
            double start = now_seconds();
            if (!fmpz_mpoly_resultant(resultant, a, b, 0, context))
            {
                fprintf(stderr, "FLINT resultant failed: %s\n", benchmark->name);
                return 1;
            }
            timings[sample] = now_seconds() - start;
        }
        qsort(timings, (size_t) samples, sizeof(double), compare_double);

        printf("%-32s FLINT %9.3f ms  terms %ld\n", benchmark->name,
               timings[samples / 2] * 1000.0, fmpz_mpoly_length(resultant, context));
        fflush(stdout);

        flint_free(timings);
        fmpz_mpoly_clear(a, context);
        fmpz_mpoly_clear(b, context);
        fmpz_mpoly_clear(resultant, context);
        fmpz_mpoly_ctx_clear(context);
    }

    flint_cleanup();
    return 0;
}
