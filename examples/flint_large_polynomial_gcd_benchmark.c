#define _POSIX_C_SOURCE 200809L

#include <flint/flint.h>
#include <flint/fmpz_mpoly.h>

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

static void construct_power(fmpz_mpoly_t output, fmpz_mpoly_t base,
                            const char *expression, slong constant,
                            const char **variables, const fmpz_mpoly_ctx_t context)
{
    if (fmpz_mpoly_set_str_pretty(base, expression, variables, context) != 0 ||
        !fmpz_mpoly_pow_ui(output, base, 7, context))
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

int main(void)
{
    const char *variables[] = {"x1", "x2", "x3", "x4", "x5", "x6", "x7"};
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
        strcmp(backend, "zippel2") != 0)
    {
        fprintf(stderr, "GCD_BENCH_BACKEND must be all, hensel, zippel, or zippel2\n");
        return 1;
    }
    int samples = sample_override ? atoi(sample_override) : 1;
    if (samples < 1)
        samples = 1;

    flint_set_num_threads(1);

    fmpz_mpoly_ctx_t context;
    fmpz_mpoly_ctx_init(context, 7, ORD_LEX);
    fmpz_mpoly_t base, a, b, g, ag, bg;
    fmpz_mpoly_init(base, context);
    fmpz_mpoly_init(a, context);
    fmpz_mpoly_init(b, context);
    fmpz_mpoly_init(g, context);
    fmpz_mpoly_init(ag, context);
    fmpz_mpoly_init(bg, context);
    int benchmark_gap = 0;

    const int high_height = strcmp(benchmark_case, "high-height") == 0;
    const char *a_expression = high_height
        ? "1+1000000007*x1+1000000009*x2+1000000033*x3+1000000087*x4+1000000093*x5+1000000097*x6+1000000103*x7"
        : "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7";
    const char *b_expression = high_height
        ? "1-1000000007*x1-1000000009*x2-1000000033*x3+1000000087*x4-1000000093*x5-1000000097*x6+1000000103*x7"
        : "1-3*x1-5*x2-7*x3+9*x4-11*x5-13*x6+15*x7";
    construct_power(a, base, a_expression, -1, variables, context);
    construct_power(b, base, b_expression, 1, variables, context);
    if (strcmp(benchmark_case, "dense") == 0 || high_height)
    {
        const char *g_expression = high_height
            ? "1+1000000007*x1+1000000009*x2+1000000033*x3+1000000087*x4+1000000093*x5+1000000097*x6-1000000103*x7"
            : "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6-15*x7";
        construct_power(g, base, g_expression, 3, variables, context);
    }
    else
    {
        char high_gap_expression[512];
        const char *g_expression;
        if (strcmp(benchmark_case, "sparse") == 0)
        {
            g_expression = "1+x1^7+2*x2^7+3*x3^7+5*x4^7+7*x5^7+11*x6^7+13*x7^7";
        }
        else
        {
            const char *gap_override = getenv("GCD_BENCH_GAP");
            benchmark_gap = gap_override ? atoi(gap_override) : 10;
            if (benchmark_gap < 1)
            {
                fprintf(stderr, "GCD_BENCH_GAP must be positive\n");
                return 1;
            }
            snprintf(high_gap_expression, sizeof(high_gap_expression),
                     "1+x1^%d+2*x2^%d+3*x3^%d+5*x4^%d+7*x5^%d+11*x6^%d+13*x7^%d",
                     benchmark_gap, benchmark_gap, benchmark_gap, benchmark_gap,
                     benchmark_gap, benchmark_gap, benchmark_gap);
            g_expression = high_gap_expression;
        }
        if (fmpz_mpoly_set_str_pretty(g, g_expression, variables, context) != 0)
        {
            fprintf(stderr, "Could not construct sparse GCD\n");
            return 1;
        }
    }
    fmpz_mpoly_mul(ag, a, g, context);
    fmpz_mpoly_mul(bg, b, g, context);

    printf("FLINT %s, case %s, samples %d\n", flint_version, benchmark_case, samples);
    if (benchmark_gap > 0)
        printf("gap %d\n", benchmark_gap);
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
    flint_cleanup_master();
    return 0;
}
