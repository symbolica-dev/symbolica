//! Compact repeated factors while preserving alias and function-map boundaries.

use super::*;
use crate::atom::{AliasedAtom, AtomQuery, AtomType};

impl FunctionMap {
    /// Share profitable repeated functions and powers in an owned alias DAG.
    ///
    /// Input expressions and existing alias handles remain opaque. Function tags
    /// keep their exact identity; ordinary arguments may contain algebra. Lazy
    /// `if` guards and `excluded_heads` remain exposed at their original sites.
    /// Newly shared factors are stored verbatim, without expanding definitions.
    pub fn share_factors<A: AtomCore>(
        &self,
        atom: AliasedAtom,
        inputs: &[A],
        excluded_heads: &[Symbol],
    ) -> AliasedAtom {
        let boundaries = inputs
            .iter()
            .map(AtomCore::as_atom_view)
            .chain(atom.aliases.keys().map(Atom::as_view))
            .collect::<HashSet<_>>();
        let max_bytes = boundaries
            .iter()
            .map(|key| key.get_byte_size())
            .max()
            .unwrap_or(0);
        let boundary =
            |view: AtomView<'_>| view.get_byte_size() <= max_bytes && boundaries.contains(&view);
        let factor = |view: AtomView<'_>| {
            matches!(view, AtomView::Pow(_) | AtomView::Fun(_))
                && !matches!(view, AtomView::Fun(call) if call.get_symbol() == Symbol::IF)
        };
        let mut counts = HashMap::default();
        for body in std::iter::once(&atom.root).chain(atom.aliases.values()) {
            body.visitor(&mut |view| {
                if boundary(view) {
                    return false;
                }
                if factor(view) {
                    *counts.entry(view).or_insert(0usize) += 1;
                    return false;
                }
                true
            });
        }
        let algebraic_query =
            AtomQuery::new().require_any_kind([AtomType::Add, AtomType::Mul, AtomType::Pow]);
        let algebraic = |view: AtomView<'_>| algebraic_query.matches(&view);
        let sensitive = boundaries
            .iter()
            .copied()
            .filter(|view| algebraic(*view))
            .collect::<HashSet<_>>();
        let mut candidates = counts
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(view, _)| *view);
        if candidates.is_empty() {
            return atom;
        }
        let symbol = AliasedAtom::fresh_alias_symbol();
        let mut replacements = HashMap::default();
        for (view, count) in candidates {
            let mut protected = false;
            view.visitor(&mut |part| {
                protected |= part.get_byte_size() <= max_bytes && sensitive.contains(&part);
                if let AtomView::Var(variable) = part {
                    protected |= excluded_heads.contains(&variable.get_symbol());
                }
                if let AtomView::Fun(call) = part {
                    let head = call.get_symbol();
                    protected |= head == Symbol::IF || excluded_heads.contains(&head);
                    let tag_count = self.tag.get(&head).copied().unwrap_or_else(|| {
                        if head.is_builtin() {
                            0
                        } else {
                            call.get_nargs()
                        }
                    });
                    protected |= call.iter().take(tag_count).any(algebraic);
                }
                !protected
            });
            if protected {
                continue;
            }
            let handle = symbol.call_args([Atom::num(replacements.len())]);
            if view.get_byte_size().saturating_mul(count - 1)
                > handle.as_view().get_byte_size().saturating_mul(count + 1)
            {
                replacements.insert(view, handle);
            }
        }
        if replacements.is_empty() {
            return atom;
        }
        let rewrite = |body: &Atom| {
            body.replace_map(|view, _, out| {
                if boundary(view) || factor(view) {
                    out.set_from_view(&replacements.get(&view).map_or(view, Atom::as_view));
                }
            })
        };
        let mut shared = AliasedAtom::from(rewrite(&atom.root));
        for (handle, body) in &atom.aliases {
            shared.register_alias(handle.clone(), rewrite(body));
        }
        for (body, handle) in replacements {
            shared.register_alias(handle, body.to_owned());
        }
        shared
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, symbol};

    #[test]
    fn forwarded_coefficients_keep_one_native_body_across_components() {
        let q = symbol!("shared_coefficient::q");
        let unused = symbol!("shared_coefficient::unused");
        let x = parse!("shared_coefficient::x");
        let y = parse!("shared_coefficient::y");
        let coefficient = symbol!("shared_coefficient::coefficient");
        let alias = parse!("shared_coefficient::alias");
        let options = FunctionRegistrationOptions::new()
            .inlining(InliningPolicy::Never)
            .prune_unused_arguments(true);
        for component_count in [4, 16] {
            let mut functions = FunctionMap::new();
            functions
                .add_function_with_options(
                    coefficient,
                    vec![q, unused],
                    Atom::add_many((1..=16).map(|index| (Atom::var(q) + index).pow(-1))),
                    options.clone(),
                )
                .unwrap();
            functions
                .add_aliases([(
                    alias.clone(),
                    coefficient.call_args([Atom::var(q), Atom::var(unused)]),
                )])
                .unwrap();
            let mut outputs = Vec::new();
            for component in 0..component_count {
                let function = symbol!(format!("shared_coefficient::component_{component}"));
                functions
                    .add_function_with_options(
                        function,
                        vec![q, unused],
                        &alias + component,
                        options.clone(),
                    )
                    .unwrap();
                outputs.push(function.call_args([x.clone(), y.clone()]));
            }
            let evaluator = Atom::evaluator_multiple(&outputs, &[x.clone(), y.clone()])
                .function_map(functions)
                .build()
                .unwrap();
            let exported = evaluator.export_instructions();
            assert_eq!(exported.sub_evaluators.len(), component_count);
            let mut unique = HashSet::default();
            let mut pending = vec![&exported];
            let mut inversions = 0;
            let mut coefficients = 0;
            while let Some(program) = pending.pop() {
                inversions += program
                    .instructions
                    .iter()
                    .filter(|instruction| matches!(instruction, Instruction::Pow(_, _, -1, _)))
                    .count();
                for body in &program.sub_evaluators {
                    if unique.insert(Arc::as_ptr(&body.instructions)) {
                        coefficients += usize::from(body.symbol == coefficient);
                        pending.push(body.instructions.as_ref());
                    }
                }
            }
            assert_eq!(coefficients, 1);
            assert_eq!(
                inversions, 16,
                "compiled coefficient work must not scale with component count"
            );
            let mut numeric = evaluator.map_coeff(&|coefficient| coefficient.re.to_f64());
            let mut actual = vec![0.; component_count];
            numeric.evaluate(&[2., 99.], &mut actual);
            let shared = (1..=16).map(|index| 1. / (2. + index as f64)).sum::<f64>();
            for (component, actual) in actual.into_iter().enumerate() {
                assert!((actual - (shared + component as f64)).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn factor_sharing_keeps_owned_dags_and_lazy_poles() {
        let x = parse!("shared_factors::x");
        let condition = parse!("shared_factors::condition");
        let a = parse!("shared_factors::a");
        let b = parse!("shared_factors::b");
        let inverse = Atom::add_many((1..=8).map(|power| x.pow(power))).pow(-1);
        let mut source =
            AliasedAtom::from(Symbol::IF.call_args([condition.clone(), &a + &b, Atom::num(3)]));
        source.register_alias(a.clone(), inverse.clone());
        source.register_alias(b, &a + &x * &inverse);
        let expected = source.clone().into_inner();
        let map = FunctionMap::new();
        let shared = map.share_factors(source, &[x.clone(), condition.clone()], &[]);
        assert_eq!(shared.get_aliases().len(), 3);
        assert_eq!(shared.clone().into_inner(), expected);
        let repeated = map.share_factors(shared.clone(), &[x.clone(), condition.clone()], &[]);
        assert_eq!(shared.get_root(), repeated.get_root());
        assert_eq!(shared.get_aliases(), repeated.get_aliases());
        let expressions = [shared];
        let mut evaluator = AliasedAtom::evaluator_multiple(&expressions, &[x, condition])
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[0., 0.]), 3.);
        assert_eq!(evaluator.evaluate_single(&[1., 1.]), 0.375);
    }

    #[test]
    fn factor_sharing_preserves_registered_tags_and_opaque_inputs() {
        let x = parse!("shared_factors_keys::x");
        let y = parse!("shared_factors_keys::y");
        let key = Atom::add_many((1..=8).map(|power| x.pow(power))).pow(-1);
        let free = Atom::add_many((1..=8).map(|power| y.pow(power))).pow(-1);
        let f = symbol!("shared_factors_keys::f");
        let call = f.call_args([key.clone()]);
        let expression = Atom::add_many((1..=4).map(|i| (&x + i) * call.pow(2) + (&y + i) * &free));
        let mut map = FunctionMap::new();
        map.add_tagged_function(f, vec![key], Vec::<Indeterminate>::new(), Atom::num(7))
            .unwrap();
        let shared = map.share_factors(expression.clone().into(), &[x.clone(), y.clone()], &[]);
        assert!(shared.get_aliases().values().any(|body| body == &free));
        assert!(
            !shared
                .get_aliases()
                .values()
                .any(|body| body == &call.pow(2))
        );
        assert_eq!(shared.clone().into_inner(), expression);
        let expressions = [shared];
        let mut evaluator = Atom::evaluator_multiple(&[expressions[0].get_root()], &[x, y])
            .function_map(map)
            .add_aliases(
                expressions[0]
                    .get_aliases()
                    .iter()
                    .map(|(a, b)| (a.clone(), b.clone())),
            )
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[1., 1.]), 687.75);
    }
}
