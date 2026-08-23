#define _POSIX_C_SOURCE 200809L

#include <flint/flint.h>
#include <flint/fmpz_mpoly.h>

#include <stdio.h>
#include <stdlib.h>
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

static void compare_multiplication(const char * name,
                                   const char * a_string, ulong a_power,
                                   const char * b_string, ulong b_power,
                                   int default_samples)
{
    const char * variables[] = {"x", "y", "z"};
    const char * sample_override = getenv("MULTIPLICATION_BENCH_SAMPLES");
    int samples = sample_override ? atoi(sample_override) : default_samples;
    if (samples < 1)
        samples = 1;

    fmpz_mpoly_ctx_t context;
    fmpz_mpoly_t a_base, b_base, a, b, product;
    fmpz_mpoly_ctx_init(context, 3, ORD_LEX);
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

    for (size_t case_index = 0; case_index < sizeof(cases) / sizeof(cases[0]); case_index++)
    {
        const benchmark_case * benchmark = cases + case_index;
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
