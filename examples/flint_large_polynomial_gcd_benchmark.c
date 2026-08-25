#define _POSIX_C_SOURCE 200809L

#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mpoly.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int (*gcd_backend)(fmpz_mpoly_t, const fmpz_mpoly_t,
                           const fmpz_mpoly_t, const fmpz_mpoly_ctx_t);

static double now_seconds(void)
{
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double) time.tv_sec + (double) time.tv_nsec * 1.0e-9;
}

static int compare_double(const void *left, const void *right)
{
    double a = *(const double *) left;
    double b = *(const double *) right;
    return (a > b) - (a < b);
}

static size_t append_checked(char *buffer, size_t capacity, size_t length,
                             const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    int written = vsnprintf(buffer + length, capacity - length, format, arguments);
    va_end(arguments);

    if (written < 0 || (size_t) written >= capacity - length)
    {
        fprintf(stderr, "Polynomial expression buffer is too small\n");
        exit(1);
    }
    return length + (size_t) written;
}

static void build_linear_expression(char *output, size_t capacity,
                                    const char *const *weights, const int *signs,
                                    slong variable_count)
{
    size_t length = 0;
    length = append_checked(output, capacity, length, "1");
    for (slong variable = 0; variable < variable_count; variable++)
    {
        length = append_checked(output, capacity, length, "%c%s*x%ld",
                                signs[variable] < 0 ? '-' : '+', weights[variable],
                                variable + 1);
    }
}

static void build_sparse_expression(char *output, size_t capacity,
                                    slong variable_count, ulong degree)
{
    static const ulong coefficients[] = {1, 2, 3, 5, 7, 11, 13, 17};
    size_t length = 0;
    length = append_checked(output, capacity, length, "1");
    for (slong variable = 0; variable < variable_count; variable++)
    {
        if (coefficients[variable] == 1)
            length = append_checked(output, capacity, length, "+x%ld^%lu",
                                    variable + 1, degree);
        else
            length = append_checked(output, capacity, length, "+%lu*x%ld^%lu",
                                    coefficients[variable], variable + 1, degree);
    }
}

static void construct_power(fmpz_mpoly_t output, fmpz_mpoly_t base,
                            const char *expression, ulong degree, slong constant,
                            const char **variables, const fmpz_mpoly_ctx_t context)
{
    if (fmpz_mpoly_set_str_pretty(base, expression, variables, context) != 0 ||
        !fmpz_mpoly_pow_ui(output, base, degree, context))
    {
        fprintf(stderr, "Could not construct polynomial\n");
        exit(1);
    }
    fmpz_mpoly_add_si(output, output, constant, context);
}

static void run_backend(const char *name, gcd_backend backend,
                        const fmpz_mpoly_t ag, const fmpz_mpoly_t bg,
                        const fmpz_mpoly_t expected, int samples,
                        const fmpz_mpoly_ctx_t context)
{
    fmpz_mpoly_t result;
    fmpz_mpoly_init(result, context);
    double *timings = flint_malloc((size_t) samples * sizeof(double));

    for (int sample = 0; sample < samples; sample++)
    {
        double start = now_seconds();
        int success = backend(result, ag, bg, context);
        timings[sample] = now_seconds() - start;
        if (!success || !fmpz_mpoly_equal(result, expected, context))
        {
            fprintf(stderr, "%s returned an incorrect GCD\n", name);
            exit(1);
        }
    }

    qsort(timings, (size_t) samples, sizeof(double), compare_double);
    printf("FLINT %-8s %10.3f ms  terms ag/bg/gcd %ld/%ld/%ld\n",
           name, timings[samples / 2] * 1000.0,
           fmpz_mpoly_length(ag, context), fmpz_mpoly_length(bg, context),
           fmpz_mpoly_length(result, context));

    flint_free(timings);
    fmpz_mpoly_clear(result, context);
}

static void run_product_benchmark(fmpz_mpoly_t ag, fmpz_mpoly_t bg,
                                  const fmpz_mpoly_t a, const fmpz_mpoly_t b,
                                  const fmpz_mpoly_t g, int samples,
                                  const fmpz_mpoly_ctx_t context)
{
    double *timings = flint_malloc((size_t) samples * sizeof(double));
    for (int sample = 0; sample < samples; sample++)
    {
        if (sample > 0)
        {
            fmpz_mpoly_clear(ag, context);
            fmpz_mpoly_clear(bg, context);
            fmpz_mpoly_init(ag, context);
            fmpz_mpoly_init(bg, context);
        }
        double start = now_seconds();
        fmpz_mpoly_mul(ag, a, g, context);
        fmpz_mpoly_mul(bg, b, g, context);
        timings[sample] = now_seconds() - start;
    }

    qsort(timings, (size_t) samples, sizeof(double), compare_double);
    printf("FLINT products %10.3f ms  terms ag/bg %ld/%ld\n",
           timings[samples / 2] * 1000.0,
           fmpz_mpoly_length(ag, context), fmpz_mpoly_length(bg, context));
    flint_free(timings);
}

int main(void)
{
    const char *variables[] = {"x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"};
    const char *benchmark_case = getenv("GCD_BENCH_CASE");
    if (benchmark_case == NULL)
        benchmark_case = "dense";
    if (strcmp(benchmark_case, "dense") != 0 &&
        strcmp(benchmark_case, "sparse") != 0 &&
        strcmp(benchmark_case, "high-gap") != 0 &&
        strcmp(benchmark_case, "high-height") != 0)
    {
        fprintf(stderr, "GCD_BENCH_CASE must be dense, sparse, high-gap, or high-height\n");
        return 1;
    }
    const char *sample_override = getenv("GCD_BENCH_SAMPLES");
    const char *backend = getenv("GCD_BENCH_BACKEND");
    if (backend == NULL)
        backend = "all";
    if (strcmp(backend, "all") != 0 &&
        strcmp(backend, "hensel") != 0 &&
        strcmp(backend, "zippel") != 0 &&
        strcmp(backend, "zippel2") != 0 &&
        strcmp(backend, "product") != 0)
    {
        fprintf(stderr, "GCD_BENCH_BACKEND must be all, hensel, zippel, zippel2, or product\n");
        return 1;
    }
    int samples = sample_override ? atoi(sample_override) : 1;
    if (samples < 1)
        samples = 1;
    const char *variable_override = getenv("GCD_BENCH_NVARS");
    slong variable_count = variable_override ? atol(variable_override) : 7;
    if (variable_count < 2 || variable_count > 8)
    {
        fprintf(stderr, "GCD_BENCH_NVARS must be between 2 and 8\n");
        return 1;
    }
    const char *degree_override = getenv("GCD_BENCH_DEGREE");
    long degree_value = degree_override ? atol(degree_override) : 7;
    if (degree_value < 1)
    {
        fprintf(stderr, "GCD_BENCH_DEGREE must be positive\n");
        return 1;
    }
    ulong degree = (ulong) degree_value;
    const char *coefficient_bits_override = getenv("GCD_BENCH_COEFFICIENT_BITS");
    long coefficient_bits_value = coefficient_bits_override
        ? atol(coefficient_bits_override) : 30;
    if (coefficient_bits_value < 8 || coefficient_bits_value > 1024)
    {
        fprintf(stderr, "GCD_BENCH_COEFFICIENT_BITS must be between 8 and 1024\n");
        return 1;
    }
    ulong coefficient_bits = (ulong) coefficient_bits_value;

    flint_set_num_threads(1);

    fmpz_mpoly_ctx_t context;
    fmpz_mpoly_ctx_init(context, variable_count, ORD_LEX);
    fmpz_mpoly_t base, a, b, g, ag, bg;
    fmpz_mpoly_init(base, context);
    fmpz_mpoly_init(a, context);
    fmpz_mpoly_init(b, context);
    fmpz_mpoly_init(g, context);
    fmpz_mpoly_init(ag, context);
    fmpz_mpoly_init(bg, context);
    ulong benchmark_gap = 0;

    const int high_height = strcmp(benchmark_case, "high-height") == 0;
    static const char *small_weights[] = {"3", "5", "7", "9", "11", "13", "15", "17"};
    static const char *thirty_bit_weights[] = {
        "1000000007", "1000000009", "1000000033", "1000000087",
        "1000000093", "1000000097", "1000000103", "1000000123"};
    static const ulong weight_offsets[] = {7, 9, 33, 87, 93, 97, 103, 123};
    char *generated_weights[] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    static const int b_signs[] = {-1, -1, -1, 1, -1, -1, 1, -1};
    int positive_signs[] = {1, 1, 1, 1, 1, 1, 1, 1};
    int gcd_signs[] = {1, 1, 1, 1, 1, 1, 1, 1};
    gcd_signs[variable_count - 1] = -1;
    const char *const *weights = small_weights;
    if (high_height && coefficient_bits == 30)
    {
        weights = thirty_bit_weights;
    }
    else if (high_height)
    {
        fmpz_t weight;
        fmpz_init(weight);
        for (slong variable = 0; variable < variable_count; variable++)
        {
            fmpz_one(weight);
            fmpz_mul_2exp(weight, weight, coefficient_bits - 1);
            fmpz_add_ui(weight, weight, weight_offsets[variable]);
            generated_weights[variable] = fmpz_get_str(NULL, 10, weight);
        }
        fmpz_clear(weight);
        weights = (const char *const *) generated_weights;
    }

    char a_expression[4096];
    char b_expression[4096];
    char g_expression[4096];
    build_linear_expression(a_expression, sizeof(a_expression), weights,
                            positive_signs, variable_count);
    build_linear_expression(b_expression, sizeof(b_expression), weights,
                            b_signs, variable_count);
    construct_power(a, base, a_expression, degree, -1, variables, context);
    construct_power(b, base, b_expression, degree, 1, variables, context);
    if (strcmp(benchmark_case, "dense") == 0 || high_height)
    {
        build_linear_expression(g_expression, sizeof(g_expression), weights,
                                gcd_signs, variable_count);
        construct_power(g, base, g_expression, degree, 3, variables, context);
    }
    else
    {
        if (strcmp(benchmark_case, "sparse") == 0)
        {
            build_sparse_expression(g_expression, sizeof(g_expression),
                                    variable_count, degree);
        }
        else
        {
            const char *gap_override = getenv("GCD_BENCH_GAP");
            long gap_value = gap_override ? atol(gap_override) : 10;
            if (gap_value < 1)
            {
                fprintf(stderr, "GCD_BENCH_GAP must be positive\n");
                return 1;
            }
            benchmark_gap = (ulong) gap_value;
            build_sparse_expression(g_expression, sizeof(g_expression),
                                    variable_count, benchmark_gap);
        }
        if (fmpz_mpoly_set_str_pretty(g, g_expression, variables, context) != 0)
        {
            fprintf(stderr, "Could not construct sparse GCD\n");
            return 1;
        }
    }
    printf("FLINT %s, case %s, variables %ld, degree %lu, samples %d\n",
           flint_version, benchmark_case, variable_count, degree, samples);
    if (high_height)
        printf("coefficient_bits %lu\n", coefficient_bits);
    if (benchmark_gap > 0)
        printf("gap %lu\n", benchmark_gap);
    run_product_benchmark(ag, bg, a, b, g, samples, context);
    if (strcmp(backend, "all") == 0 || strcmp(backend, "hensel") == 0)
        run_backend("hensel", fmpz_mpoly_gcd_hensel, ag, bg, g, samples, context);
    if (strcmp(backend, "all") == 0 || strcmp(backend, "zippel") == 0)
        run_backend("zippel", fmpz_mpoly_gcd_zippel, ag, bg, g, samples, context);
    if (strcmp(backend, "all") == 0 || strcmp(backend, "zippel2") == 0)
        run_backend("zippel2", fmpz_mpoly_gcd_zippel2, ag, bg, g, samples, context);

    fmpz_mpoly_clear(base, context);
    fmpz_mpoly_clear(a, context);
    fmpz_mpoly_clear(b, context);
    fmpz_mpoly_clear(g, context);
    fmpz_mpoly_clear(ag, context);
    fmpz_mpoly_clear(bg, context);
    fmpz_mpoly_ctx_clear(context);
    for (slong variable = 0; variable < variable_count; variable++)
        if (generated_weights[variable] != NULL)
            flint_free(generated_weights[variable]);
    flint_cleanup_master();
    return 0;
}
