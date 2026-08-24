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
    if (strcmp(benchmark_case, "dense") != 0 && strcmp(benchmark_case, "sparse") != 0)
    {
        fprintf(stderr, "GCD_BENCH_CASE must be dense or sparse\n");
        return 1;
    }
    const char *sample_override = getenv("GCD_BENCH_SAMPLES");
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

    construct_power(a, base, "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7",
                    -1, variables, context);
    construct_power(b, base, "1-3*x1-5*x2-7*x3+9*x4-11*x5-13*x6+15*x7",
                    1, variables, context);
    if (strcmp(benchmark_case, "dense") == 0)
    {
        construct_power(g, base, "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6-15*x7",
                        3, variables, context);
    }
    else if (fmpz_mpoly_set_str_pretty(
                 g, "1+x1^7+2*x2^7+3*x3^7+5*x4^7+7*x5^7+11*x6^7+13*x7^7",
                 variables, context) != 0)
    {
        fprintf(stderr, "Could not construct sparse GCD\n");
        return 1;
    }
    fmpz_mpoly_mul(ag, a, g, context);
    fmpz_mpoly_mul(bg, b, g, context);

    printf("FLINT %s, case %s, samples %d\n", flint_version, benchmark_case, samples);
    run_backend("zippel", fmpz_mpoly_gcd_zippel, ag, bg, g, samples, context);
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
