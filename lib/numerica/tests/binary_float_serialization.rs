use numerica::domains::float::{Float, FloatLike, SingleFloat};

const PRECISIONS: &[u32] = &[2, 7, 8, 9, 53, 63, 64, 65, 256, 4096];

fn assert_roundtrip(value: &Float) {
    let encoded = value.serialize();
    assert_eq!(value.serialized_size(), encoded.len());
    let decoded = Float::try_deserialize(&encoded, value.prec()).unwrap();
    assert_eq!(decoded, *value);
    assert_eq!(decoded.prec(), value.prec());
    assert_eq!(decoded.serialize(), encoded);
    assert_eq!(Float::deserialize(&encoded, value.prec()), decoded);
    assert_eq!(Float::serialized_is_zero(&encoded), value.is_zero());
    assert_eq!(Float::serialized_is_finite(&encoded), value.is_finite());

    let mut appended = b"prefix".to_vec();
    value.serialize_into(&mut appended);
    value.serialize_into(&mut appended);
    assert_eq!(&appended[..6], b"prefix");
    assert_eq!(&appended[6..6 + encoded.len()], encoded);
    assert_eq!(&appended[6 + encoded.len()..], encoded);
}

// Only boundary tests use this schema-level fixture builder. Ordinary values
// below have literal expected bytes, independent of the production encoder.
fn finite_fixture(exponent: i64, significand: &[u8]) -> Vec<u8> {
    let mut bytes = vec![1];
    let mut magnitude = exponent.unsigned_abs();
    magnitude = if exponent < 0 {
        magnitude.wrapping_mul(2).wrapping_sub(1)
    } else {
        magnitude * 2
    };
    while magnitude >= 128 {
        bytes.push((magnitude % 128) as u8 + 128);
        magnitude /= 128;
    }
    bytes.push(magnitude as u8);
    bytes.extend_from_slice(significand);
    bytes
}

#[test]
fn binary_values_match_portable_wire_fixtures() {
    for precision in PRECISIONS.iter().copied().filter(|&p| p >= 3) {
        for (value, expected) in [
            (1., &[1, 2, 0x80][..]),
            (-1., &[2, 2, 0x80][..]),
            (1.25, &[1, 2, 0xa0][..]),
            (-1.25, &[2, 2, 0xa0][..]),
            (0.5, &[1, 0, 0x80][..]),
            (0.25, &[1, 1, 0x80][..]),
            (7., &[1, 6, 0xe0][..]),
            (128., &[1, 16, 0x80][..]),
            (256., &[1, 18, 0x80][..]),
        ] {
            let value = Float::with_val(precision, value);
            assert_eq!(value.serialize(), expected);
            assert_roundtrip(&value);
        }
    }

    for &precision in PRECISIONS {
        let large = Float::with_val(precision, 2).pow(4000);
        let small = large.inv();
        assert_eq!(large.serialize(), [1, 0xc2, 0x3e, 0x80]);
        assert_eq!(small.serialize(), [1, 0xbd, 0x3e, 0x80]);
        assert_roundtrip(&large);
        assert_roundtrip(&small);
    }
}

#[test]
fn binary_roundtrips_preserve_finite_values_and_requested_precision() {
    for &precision in PRECISIONS {
        for text in ["0.1", "-13.125", "1e-1000", "-1e1000"] {
            assert_roundtrip(&Float::parse(text, Some(precision)).unwrap());
        }
        let mut seed = 0x7da2_03ef_154b_c9d1u64;
        for _ in 0..64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let exponent = ((seed >> 32) % 2046 + 1) << 52;
            let value = f64::from_bits((seed & !(0x7ff << 52)) | exponent);
            assert_roundtrip(&Float::with_val(precision, value));
        }
    }
}

#[test]
fn equal_values_have_identical_payloads_despite_precision_padding() {
    for exponent in [
        -4000i32, -65, -64, -63, -8, -1, 0, 1, 7, 8, 63, 64, 65, 4000,
    ] {
        for mantissa in [1., -1., 1.25, -1.25, 127.5] {
            let mut canonical = None;
            for precision in [8, 9, 53, 63, 64, 65, 256, 4096] {
                let mut scale = Float::with_val(precision, 2).pow(exponent.unsigned_abs().into());
                if exponent < 0 {
                    scale = scale.inv();
                }
                let value = Float::with_val(precision, mantissa) * scale;
                let bytes = value.serialize();
                if let Some(expected) = &canonical {
                    assert_eq!(&bytes, expected);
                } else {
                    canonical = Some(bytes);
                }
            }
        }
    }
}

#[test]
fn binary_comparison_matches_decoded_values_across_precisions_and_classes() {
    let mut values = Vec::new();
    for precision in [8, 53, 65, 256] {
        for text in [
            "-1e1000",
            "-256",
            "-1.00390625",
            "-1",
            "-0.1",
            "-1e-1000",
            "0",
            "1e-1000",
            "0.1",
            "1",
            "1.00390625",
            "256",
            "1e1000",
        ] {
            values.push(Float::parse(text, Some(precision)).unwrap());
        }
        for value in [f64::NEG_INFINITY, f64::INFINITY, f64::NAN, -0.] {
            values.push(Float::with_val(precision, value));
        }
    }
    // Equal high-byte prefixes with different low-byte lengths must compare
    // by the remaining fractional bits, in both sign directions.
    for significand in [&[0x80][..], &[1, 0x80], &[1, 0, 0x80], &[0xff, 0x80]] {
        let value = Float::try_deserialize(&finite_fixture(1, significand), 256).unwrap();
        values.push(-value.clone());
        values.push(value);
    }
    let bytes: Vec<_> = values.iter().map(Float::serialize).collect();
    for (left, left_bytes) in values.iter().zip(&bytes) {
        for (right, right_bytes) in values.iter().zip(&bytes) {
            assert_eq!(
                Float::serialized_partial_cmp(left_bytes, right_bytes),
                left.partial_cmp(right),
                "comparison differs for {left_bytes:?} and {right_bytes:?}"
            );
        }
    }
    for malformed in [&[][..], &[6], &[1, 0x80, 0, 0x80], &[1, 0, 0, 0x80]] {
        assert_eq!(Float::serialized_partial_cmp(malformed, &[0]), None);
        assert_eq!(Float::serialized_partial_cmp(&[0], malformed), None);
    }
}

#[cfg(feature = "float-mpfr")]
#[test]
fn binary_mpfr_imports_exactly_across_native_word_and_stack_thresholds() {
    let word_bytes = std::mem::size_of::<std::os::raw::c_ulong>();
    let limb_bytes = std::mem::size_of::<gmp_mpfr_sys::gmp::limb_t>();
    let mut counts = vec![word_bytes - 1, word_bytes, word_bytes + 1];
    for limbs in 1..=4 {
        let bytes = limbs * limb_bytes;
        counts.extend([bytes - 1, bytes, bytes + 1]);
    }
    counts.sort_unstable();
    counts.dedup();
    for count in counts {
        let significand: Vec<_> = (0..count).map(|i| 0xa1 + i as u8).collect();
        let precision = 8 * count as u32;
        // A radix-two string supplies an independent exact-value reference.
        let digits: String = significand
            .iter()
            .rev()
            .map(|b| format!("{b:08b}"))
            .collect();
        let raw = rug::Float::parse_radix(digits.as_str(), 2).unwrap();
        let expected = Float::from_raw(rug::Float::with_val(precision, raw));
        for negative in [false, true] {
            let mut bytes = finite_fixture(i64::from(precision), &significand);
            if negative {
                bytes[0] = 2;
            }
            let value = Float::try_deserialize(&bytes, precision).unwrap();
            assert_eq!(
                value,
                if negative {
                    -expected.clone()
                } else {
                    expected.clone()
                }
            );
            assert_eq!(value.serialize(), bytes);
            assert!(Float::try_deserialize(&bytes, precision - 1).is_err());
        }
    }
}

#[cfg(feature = "float-mpfr")]
#[test]
fn binary_mpfr_stack_import_preserves_sparse_limbs_and_low_padding_bits() {
    let limb_bytes = std::mem::size_of::<gmp_mpfr_sys::gmp::limb_t>();
    for count in [limb_bytes + 1, 4 * limb_bytes, 4 * limb_bytes + 1] {
        let mut significand = vec![0; count];
        significand[0] = 0xa0;
        significand[count - 1] = 0x80;
        let bits = 8 * count as u32;
        let precision = bits - 5;
        let digits: String = significand
            .iter()
            .rev()
            .map(|byte| format!("{byte:08b}"))
            .collect();
        for exponent in [-4000, 4000] {
            let raw = rug::Float::parse_radix(digits.as_str(), 2).unwrap();
            let expected =
                Float::from_raw(rug::Float::with_val(precision, raw) << (exponent - bits as i32));
            for negative in [false, true] {
                let mut bytes = finite_fixture(exponent.into(), &significand);
                if negative {
                    bytes[0] = 2;
                }
                let value = Float::try_deserialize(&bytes, precision).unwrap();
                assert_eq!(
                    value,
                    if negative {
                        -expected.clone()
                    } else {
                        expected.clone()
                    }
                );
                assert_eq!(value.serialize(), bytes);
                assert_eq!(value.prec(), precision);
                assert!(Float::try_deserialize(&bytes, precision - 1).is_err());
            }
        }
    }
}

#[test]
fn binary_special_values_preserve_class_precision_and_canonical_zero() {
    for &precision in PRECISIONS {
        for (value, tag) in [
            (0., 0),
            (-0., 0),
            (f64::INFINITY, 3),
            (f64::NEG_INFINITY, 4),
            (f64::NAN, 5),
        ] {
            let value = Float::with_val(precision, value);
            assert_eq!(value.serialize(), [tag]);
            assert_roundtrip(&value);
            let decoded = Float::try_deserialize(&[tag], precision).unwrap();
            assert_eq!(decoded.prec(), precision);
            assert_eq!(decoded.is_sign_negative(), tag == 4);
        }
    }
}

#[cfg(feature = "float-mpfr")]
#[test]
fn binary_mpfr_canonicalizes_negative_nan() {
    for &precision in PRECISIONS {
        let value = Float::from_raw(-rug::Float::with_val(precision, rug::float::Special::Nan));
        assert!(value.is_sign_negative());
        assert_eq!(value.serialize(), [5]);
        let decoded = Float::try_deserialize(&[5], precision).unwrap();
        assert!(!decoded.is_sign_negative());
        assert!(decoded.as_raw().is_nan());
        assert_roundtrip(&value);
    }
}

#[test]
fn binary_rejects_noncanonical_negative_nan_tag() {
    assert!(Float::try_deserialize(&[6], 53).is_err());
}

#[test]
fn binary_cross_backend_fixtures_preserve_every_significant_bit() {
    // The exact MPFR 53-bit nearest value of 1/3, with three low padding bits.
    let mpfr_third = [1, 1, 0xa8, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa];
    let decoded = Float::try_deserialize(&mpfr_third, 53).unwrap();
    assert_eq!(decoded, Float::with_val(53, 1. / 3.));
    assert_eq!(decoded.prec(), 53);
    assert_eq!(decoded.serialize(), mpfr_third);

    // Astro arithmetic retains all 64 bits although its requested precision
    // remains 53. A backend unable to preserve them must reject, not round.
    let astro_third = [1, 1, 0xab, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa];
    #[cfg(feature = "float-astro")]
    {
        let decoded = Float::try_deserialize(&astro_third, 53).unwrap();
        let expected = Float::with_val(53, 1) / Float::with_val(53, 3);
        assert_eq!(decoded, expected);
        assert_eq!(decoded.prec(), 53);
        assert_eq!(decoded.serialize(), astro_third);
    }
    #[cfg(feature = "float-mpfr")]
    assert!(Float::try_deserialize(&astro_third, 53).is_err());
    assert_roundtrip(&Float::try_deserialize(&astro_third, 64).unwrap());
}

#[test]
fn binary_decode_rejects_malformed_or_noncanonical_payloads() {
    for bytes in [
        &[][..],
        &[6],
        &[7],
        &[255],
        &[0, 0],
        &[3, 0],
        &[4, 0],
        &[5, 0],
        &[6, 0],
        &[1],
        &[2],
        &[1, 0],
        &[1, 0, 0],
        &[1, 0, 0x7f],
        &[1, 0, 0, 0x80],
        &[1, 0, 0x80, 0],
        &[1, 0x80, 0, 0x80],
        &[1, 0x82, 0, 0x80],
        // More than 64 exponent bits, or an unterminated ten-byte exponent.
        &[
            1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 2, 0x80,
        ],
        &[
            1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        ],
        b"0",
        b"1.4",
        b"-1.4@+100",
        b"inf",
        b"NaN",
    ] {
        assert!(
            Float::try_deserialize(bytes, 53).is_err(),
            "accepted {bytes:?}"
        );
    }
    assert!(Float::try_deserialize(&[0], 0).is_err());
    for exponent in [i64::MIN, i64::MAX] {
        assert!(Float::try_deserialize(&finite_fixture(exponent, &[0x80]), 53).is_err());
    }
    #[cfg(feature = "float-mpfr")]
    if let Some(precision) = rug::float::prec_max().checked_add(1) {
        assert!(Float::try_deserialize(&[0], precision).is_err());
    }
}

#[cfg(feature = "float-mpfr")]
#[test]
fn binary_mpfr_roundtrips_current_exponent_endpoints_without_global_changes() {
    use gmp_mpfr_sys::mpfr;

    // Bounds are read from this test thread; no global/thread-local exponent
    // configuration is changed. Construct endpoint powers directly to avoid
    // overflow in an intermediate unscaled integer.
    let (minimum, maximum) = unsafe { (mpfr::get_emin(), mpfr::get_emax()) };
    for exponent in [minimum, maximum] {
        let mut raw = rug::Float::new(53);
        let rounding =
            unsafe { mpfr::set_ui_2exp(raw.as_raw_mut(), 1, exponent - 1, mpfr::rnd_t::RNDN) };
        assert_eq!(rounding, 0);
        let value = Float::from_raw(raw);
        assert_eq!(value.serialize(), finite_fixture(exponent as i64, &[0x80]));
        assert_roundtrip(&value);
    }
    for exponent in [minimum as i64 - 1, maximum as i64 + 1] {
        assert!(Float::try_deserialize(&finite_fixture(exponent, &[0x80]), 53).is_err());
    }
}

#[cfg(feature = "float-astro")]
#[test]
fn binary_astro_roundtrips_subnormal_payloads_and_exponent_endpoints() {
    for precision in [2, 53, 65, 256] {
        let storage_bits =
            (precision as usize).div_ceil(astro_float::WORD_BIT_SIZE) * astro_float::WORD_BIT_SIZE;
        let minimum = i64::from(astro_float::EXPONENT_MIN);
        let maximum = i64::from(astro_float::EXPONENT_MAX);
        for (exponent, significand) in [
            (minimum - storage_bits as i64 + 1, vec![0x80]),
            (minimum - storage_bits as i64 + 2, vec![0xc0]),
            (minimum, vec![0x80]),
            (maximum, vec![0xff; storage_bits / 8]),
        ] {
            let bytes = finite_fixture(exponent, &significand);
            let value = Float::try_deserialize(&bytes, precision).unwrap();
            assert_eq!(value.serialize(), bytes);
            assert_roundtrip(&value);
        }
        for exponent in [minimum - storage_bits as i64, maximum + 1] {
            assert!(Float::try_deserialize(&finite_fixture(exponent, &[0x80]), precision).is_err());
        }
    }
    let bytes = finite_fixture(
        i64::from(astro_float::EXPONENT_MIN) - 3,
        &[1, 2, 3, 4, 5, 6, 7, 8, 0x80],
    );
    let value = Float::try_deserialize(&bytes, 128).unwrap();
    assert_eq!(value.serialize(), bytes);
    assert_roundtrip(&value);
}

#[cfg(feature = "bincode")]
#[test]
fn binary_bincode_roundtrips_values_and_rejects_legacy_hex_payloads() {
    let config = bincode::config::standard();
    for &precision in PRECISIONS {
        for value in [
            0.,
            -0.,
            1.25,
            -1.25,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let value = Float::with_val(precision, value);
            let encoded = bincode::encode_to_vec(&value, config).unwrap();
            let (decoded, used): (Float, usize) =
                bincode::decode_from_slice(&encoded, config).unwrap();
            assert_eq!(used, encoded.len());
            assert_eq!(decoded, value);
            assert_eq!(decoded.prec(), precision);
            assert_eq!(decoded.serialize(), value.serialize());
        }
    }
    for bytes in [b"0".as_slice(), b"1.4", b"-1.4@+100", b"NaN"] {
        let legacy = bincode::encode_to_vec((53u32, bytes.to_vec()), config).unwrap();
        assert!(bincode::decode_from_slice::<Float, _>(&legacy, config).is_err());
    }
}

#[cfg(feature = "serde")]
#[test]
fn binary_serde_preserves_exact_values_precision_and_canonical_specials() {
    assert_eq!(
        serde_json::to_string(&Float::with_val(53, 1.25)).unwrap(),
        "[53,[1,2,160]]"
    );
    for &precision in PRECISIONS {
        let mut values: Vec<_> = [
            0.,
            -0.,
            1.25,
            -1.25,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ]
        .into_iter()
        .map(|value| Float::with_val(precision, value))
        .collect();
        values.push(Float::parse("0.1", Some(precision)).unwrap());
        values.push(Float::parse("-1e1000", Some(precision)).unwrap());
        values.push(Float::with_val(precision, 1) / Float::with_val(precision, 3));
        #[cfg(feature = "float-mpfr")]
        values.push(Float::from_raw(-rug::Float::with_val(
            precision,
            rug::float::Special::Nan,
        )));
        for value in values {
            let encoded = serde_json::to_string(&value).unwrap();
            let decoded: Float = serde_json::from_str(&encoded).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(decoded.prec(), precision);
            assert_eq!(decoded.serialize(), value.serialize());
            assert_eq!(
                decoded.is_sign_negative(),
                !value.is_zero() && !value.as_raw().is_nan() && value.is_sign_negative()
            );
            assert_eq!(serde_json::to_string(&decoded).unwrap(), encoded);
        }
    }
}

#[cfg(feature = "serde")]
#[test]
fn binary_serde_rejects_malformed_and_legacy_backend_payloads() {
    for bytes in [
        vec![],
        vec![6],
        vec![7],
        vec![0, 0],
        vec![1, 0],
        vec![1, 0, 0, 0x80],
        vec![1, 0x80, 0, 0x80],
        finite_fixture(i64::MAX, &[0x80]),
        b"1.4".to_vec(),
    ] {
        let json = serde_json::to_string(&(53u32, bytes)).unwrap();
        assert!(serde_json::from_str::<Float>(&json).is_err());
    }
    for json in ["[]", "[53]", "[53,[0],0]", "[0,[0]]", "[4294967296,[0]]"] {
        assert!(serde_json::from_str::<Float>(json).is_err());
    }
    assert!(serde_json::from_str::<Float>("[53,[6]]").is_err());

    // The former Float newtype delegated directly to its backend's serde
    // representation: a Rug object or an Astro (precision, decimal string).
    let value = Float::with_val(53, 1.25);
    let legacy_backend = serde_json::to_string(value.as_raw()).unwrap();
    assert!(serde_json::from_str::<Float>(&legacy_backend).is_err());
    let legacy_hex = serde_json::to_string(&(53u32, "1.4")).unwrap();
    assert!(serde_json::from_str::<Float>(&legacy_hex).is_err());
}
