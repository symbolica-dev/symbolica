"""Independent endpoint-one GPL quadrature; no Holder/series implementation.

G uses dt/(t-a), integration along the real unit interval. The square root
in r4=(-1-i*sqrt(3))/2 is positive real; logs use their principal branches.
A = G(0,0,r4,1;1)
  = 1/2 integral_0^1 log(t)^2 log(1-t)/(t-r4) dt.
B = G(0,1,1,r4;1)
  = integral_0^1 Li2(v)/v log(1-(1-v)/r4) dv
  = -integral_0^1 Li3(v)/(v+r4-1) dv.
The last equality follows by integration by parts with dLi3(v)=Li2(v)dv/v;
both boundary terms vanish. It provides a separately evaluated cross-check.
"""
import argparse
import json
import pathlib
import time
import mpmath as mp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=pathlib.Path, help="development evidence directory")
OUT = parser.parse_args().output
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for digits in (110, 230, 270):
    mp.mp.dps = digits
    r4 = (-1 - 1j*mp.sqrt(3))/2
    def a(t):
        if t == 0 or t == 1:
            return mp.mpc(0)
        return mp.log(t)**2 * mp.log1p(-t)/(2*(t-r4))
    def b(v):
        if v == 0:
            return mp.log1p(-1/r4)
        if v == 1:
            return mp.mpc(0)
        return mp.polylog(2, v)/v * mp.log1p(-(1-v)/r4)
    def b_parts(v):
        return -mp.polylog(3, v)/(v+r4-1)
    start = time.monotonic()
    values = {}
    errors = {}
    for name, fun in [('A', a), ('B', b), ('B_parts', b_parts)]:
        value, estimate = mp.quad(fun, [0, mp.mpf('0.25'), mp.mpf('0.5'), mp.mpf('0.75'), 1],
                                  method='tanh-sinh', error=True)
        values[name] = {'real': mp.nstr(value.real, digits), 'imaginary': mp.nstr(value.imag, digits)}
        errors[name] = mp.nstr(estimate, 8)
        print(f'dps={digits} {name} re={mp.nstr(value.real, 20)} im={mp.nstr(value.imag, 20)} estimated_error={estimate} elapsed={time.monotonic()-start:.3f}s', flush=True)
    row = {'working_decimal_digits': digits, 'values': values, 'quadrature_error_estimates': errors,
           'elapsed_seconds': time.monotonic()-start}
    rows.append(row)
    (OUT / 'endpoint-one-quadrature-raw.json').write_text(json.dumps({'mpmath_version': mp.__version__, 'runs': rows}, indent=2)+'\n')

mp.mp.dps = 280
for name in ('A', 'B'):
    for lower in (rows[0], rows[1]):
        low = lower['values'][name]
        high = rows[2]['values'][name]
        difference = abs(mp.mpc(low['real'], low['imaginary']) - mp.mpc(high['real'], high['imaginary']))
        print(f'{name}: dps{lower["working_decimal_digits"]} vs270 absolute_difference={mp.nstr(difference,12)}', flush=True)
for row in rows:
    values = row['values']
    difference = abs(mp.mpc(values['B']['real'], values['B']['imaginary']) - mp.mpc(values['B_parts']['real'], values['B_parts']['imaginary']))
    print(f'B integration-by-parts cross-check dps{row["working_decimal_digits"]}: {mp.nstr(difference,12)}', flush=True)
reference = {
 'convention': 'G(a1,...,an;1) with kernel dt/(t-a1), real path [0,1], principal logarithms',
 'r4': '(-1-i*sqrt(3))/2 with positive real sqrt(3)',
 'method': 'mpmath tanh-sinh quadrature of independent single-integral identities; B also integrated by parts to a Li3 integral',
 'mpmath_version': mp.__version__,
 'reference_decimal_digits': 210,
 'working_decimal_digits': [110,230,270],
 'values': {}
}
for name, word in [('A',['0','0','r4','1']), ('B',['0','1','1','r4'])]:
    values = rows[-1]['values'][name]
    reference['values'][name] = {'word':word, **{key:mp.nstr(mp.mpf(value),210) for key,value in values.items()}}
(OUT / 'endpoint-one-reference.json').write_text(json.dumps(reference,indent=2)+'\n')
print('wrote', OUT / 'endpoint-one-reference.json', flush=True)
