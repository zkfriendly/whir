//! Quadratic sumcheck protocol.

use std::fmt;

use ark_ff::Field;
use ark_std::rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::{
    algebra::{
        dot,
        sumcheck::{compute_sumcheck_polynomial, fold, fold_and_compute_polynomial},
        univariate_evaluate,
    },
    protocols::proof_of_work,
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
    type_info::Type,
    utils::{chunks_exact_or_empty, zip_strict},
    verify,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Config<F>
where
    F: Field,
{
    pub field: Type<F>,
    pub initial_size: usize,
    pub round_pow: proof_of_work::Config,
    pub num_rounds: usize,
    pub mask_length: usize,
}

impl<F: Field> Config<F> {
    pub fn final_size(&self) -> usize {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        if self.initial_size == 0 || self.num_rounds == 0 {
            self.initial_size
        } else {
            self.initial_size.next_power_of_two() >> self.num_rounds
        }
    }

    /// Runs the quadratic sumcheck protocol as configured.
    ///
    /// It reduces a claim of the form `dot(a, b) == sum` to an exponentially
    /// smaller claim `dot(a', b') == sum'` where `a'` is `a` folded in place
    /// and similarly for `b`.
    ///
    /// This function:
    /// - Samples random values to progressively reduce the polynomial.
    /// - Applies proof-of-work grinding if required.
    /// - Returns the sampled folding randomness values used in each reduction step.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn prove<H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        a: &mut Vec<F>,
        b: &mut Vec<F>,
        sum: &mut F,
        masks: &[F],
    ) -> (Vec<F>, F, F)
    where
        H: DuplexSpongeInterface,
        R: CryptoRng + RngCore,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        assert!(self.mask_length == 0 || self.mask_length >= 3);
        assert_eq!(a.len(), self.initial_size);
        assert_eq!(b.len(), self.initial_size);
        debug_assert_eq!(dot(a, b), *sum);
        assert_eq!(masks.len(), self.num_rounds * self.mask_length);

        // Send mask sum and get combination randomness.
        let mask_rlc = if !masks.is_empty() {
            // f(0) + f(1) = f_0 + f_0 + f_1 + ⋯ + f_d.
            let mask_sum = masks
                .chunks_exact(self.mask_length)
                .map(eval_01) // s(0) + s(1)
                .sum::<F>()
                * F::from(2).pow([self.num_rounds.saturating_sub(1) as u64]);
            prover_state.prover_message(&mask_sum);
            let mask_rlc: F = prover_state.verifier_message();
            Some(mask_rlc)
        } else {
            None
        };

        // We do a staggered Sumcheck loop so we can merge the inner fold+compute loops.
        let mut res = Vec::with_capacity(self.num_rounds);
        let mut folding_randomness = None;
        for round in 0..self.num_rounds {
            // Fold and compute sumcheck polynomial in one pass.
            let (c0, c2) = if let Some(w) = folding_randomness {
                fold_and_compute_polynomial(a, b, w)
            } else {
                compute_sumcheck_polynomial(a, b)
            };
            let c1 = *sum - c0.double() - c2;

            // Optionally mask with univariate
            if let Some(mask_rlc) = mask_rlc {
                let mut mask_sum = F::ZERO;
                let sum_multiple =
                    F::from(2).pow([self.num_rounds.saturating_sub(round + 1) as u64]);
                let future_multiple =
                    F::from(2).pow([self.num_rounds.saturating_sub(round + 2) as u64]);
                for (j, mask) in masks.chunks_exact(self.mask_length).enumerate() {
                    if j < round {
                        mask_sum += univariate_evaluate(mask, res[j]) * sum_multiple;
                    }
                    if j > round {
                        mask_sum += eval_01(mask) * future_multiple;
                    }
                }
                let mask = masks.chunks_exact(self.mask_length).nth(round).unwrap();
                let mut univariate = Vec::new();
                for (i, m) in mask.iter().enumerate() {
                    let mut coeff = *m * sum_multiple;
                    if i == 0 {
                        coeff += mask_sum;
                    }
                    if let Some(&c) = [c0, c1, c2].get(i) {
                        coeff += mask_rlc * c;
                    }
                    univariate.push(coeff);
                }
                dbg!(&univariate);
                prover_state.prover_messages(&univariate);
            } else {
                prover_state.prover_messages(&[c0, c2]);
            }

            // Receive the random evaluation point and update the sum
            self.round_pow.prove(prover_state);
            let r = prover_state.verifier_message::<F>();
            res.push(r);
            *sum = (c2 * r + c1) * r + c0;
            folding_randomness = Some(r);
        }
        if let Some(w) = folding_randomness {
            // Final fold of the inputs (no polynomial computation)
            fold(a, w);
            fold(b, w);
        }

        // Add now constant mask offset to sum.
        let (mask_sum, mask_rlc) = if let Some(mask_rlc) = mask_rlc {
            let masked_sum = zip_strict(&res, masks.chunks_exact(self.mask_length))
                .map(|(x, c)| univariate_evaluate(c, *x))
                .sum::<F>();
            *sum = masked_sum + mask_rlc * *sum;
            (masked_sum, mask_rlc)
        } else {
            (F::ZERO, F::ONE)
        };

        (res, mask_sum, mask_rlc)
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn verify<H>(
        &self,
        verifier_state: &mut VerifierState<H>,
        sum: &mut F,
    ) -> VerificationResult<Vec<F>>
    where
        H: DuplexSpongeInterface,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
    {
        assert!(
            self.num_rounds == 0 || self.initial_size.next_power_of_two() >= 1 << self.num_rounds
        );
        assert!(self.mask_length == 0 || self.mask_length >= 3);

        if self.mask_length > 0 && self.num_rounds > 0 {
            let mask_sum: F = verifier_state.prover_message()?;
            let mask_rlc: F = verifier_state.verifier_message();
            *sum = mask_sum + mask_rlc * *sum;
        };

        let mut univariate = vec![F::ZERO; self.mask_length.max(3)];
        let mut res = Vec::with_capacity(self.num_rounds);
        for _ in 0..self.num_rounds {
            if self.mask_length > 0 {
                for c in &mut univariate {
                    *c = verifier_state.prover_message()?;
                }
                dbg!(&univariate);
                // Check h(0) + h(1) = sum
                verify!(eval_01(&univariate) == *sum);
            } else {
                // Receive sumcheck polynomial c0 and c2, derive c1.
                univariate[0] = verifier_state.prover_message()?;
                univariate[2] = verifier_state.prover_message()?;
                univariate[1] = *sum - univariate[0].double() - univariate[2];
            }

            // Check proof of work (if any)
            self.round_pow.verify(verifier_state)?;

            // Receive the random evaluation point
            let folding_randomness = verifier_state.verifier_message::<F>();
            res.push(folding_randomness);

            // Update the sum
            *sum = univariate_evaluate(&univariate, folding_randomness);
        }
        Ok(res)
    }
}

impl<F: Field> fmt::Display for Config<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "size {} rounds {} pow {:.2}",
            self.initial_size,
            self.num_rounds,
            self.round_pow.difficulty()
        )
    }
}

// Evaluated a univariate as p(0) + p(1)
fn eval_01<F: Field>(coefficients: &[F]) -> F {
    if coefficients.is_empty() {
        return F::ZERO;
    }
    coefficients[0] + coefficients.iter().sum::<F>()
}

#[cfg(test)]
mod tests {

    // TODO: Proptest based tests checking invariants and post conditions.
    use ark_std::rand::{
        distributions::{Distribution, Standard},
        rngs::StdRng,
        SeedableRng,
    };
    use proptest::{prelude::Just, prop_oneof, proptest, strategy::Strategy};
    #[cfg(feature = "tracing")]
    use tracing::instrument;

    use super::*;
    use crate::{
        algebra::{
            fields::{self, Field64},
            multilinear_extend, random_vector,
        },
        transcript::DomainSeparator,
    };

    impl<F: Field> Config<F> {
        pub fn arbitrary() -> impl Strategy<Value = Self> {
            let mask_length = prop_oneof![
                3 => Just(0_usize),
                7 => 3_usize..100,
            ];
            (0_usize..(1 << 12), 0_usize..12, mask_length).prop_map(
                |(initial_size, num_rounds, mask_length)| {
                    let num_rounds =
                        num_rounds.min(initial_size.next_power_of_two().trailing_zeros() as usize);
                    Self {
                        field: Type::new(),
                        initial_size,
                        num_rounds,
                        round_pow: proof_of_work::Config::none(),
                        mask_length,
                    }
                },
            )
        }
    }

    #[cfg_attr(feature = "tracing", instrument)]
    fn test_config<F>(seed: u64, config: &Config<F>)
    where
        F: Field + Codec,
        Standard: Distribution<F>,
    {
        // Pseudo-random Instance
        let instance = U64(seed);
        let ds = DomainSeparator::protocol(config)
            .session(&format!("Test at {}:{}", file!(), line!()))
            .instance(&instance);
        let mut rng = StdRng::seed_from_u64(seed);
        let initial_vector = random_vector(&mut rng, config.initial_size);
        let initial_covector = random_vector(&mut rng, config.initial_size);
        let initial_sum = dot(&initial_vector, &initial_covector);
        let masks = random_vector(&mut rng, config.mask_length * config.num_rounds);

        // Prover
        let mut vector = initial_vector.clone();
        let mut covector = initial_covector.clone();
        let mut sum = initial_sum;
        let mut prover_state = ProverState::new_std(&ds);
        let (point, mask_sum, mask_rlc) = config.prove(
            &mut prover_state,
            &mut vector,
            &mut covector,
            &mut sum,
            &masks,
        );
        let expected_mask_sum = zip_strict(
            chunks_exact_or_empty(&masks, config.mask_length, config.num_rounds),
            &point,
        )
        .map(|(m, x)| univariate_evaluate(m, *x))
        .sum::<F>();
        assert_eq!(vector.len(), config.final_size());
        assert_eq!(covector.len(), config.final_size());
        assert_eq!(mask_sum, expected_mask_sum);
        assert_eq!(mask_sum + mask_rlc * dot(&vector, &covector), sum);
        if config.final_size() == 1 {
            assert_eq!(multilinear_extend(&initial_vector, &point), vector[0]);
            assert_eq!(multilinear_extend(&initial_covector, &point), covector[0]);
        } else {
            // TODO: Check correct folding.
        }
        let proof = prover_state.proof();

        // Verifier
        let mut verifier_sum = initial_sum;
        let mut verifier_state = VerifierState::new_std(&ds, &proof);
        let verifier_point = config
            .verify(&mut verifier_state, &mut verifier_sum)
            .unwrap();
        assert_eq!(verifier_point, point);
        assert_eq!(verifier_sum, sum);
        verifier_state.check_eof().unwrap();
    }

    fn test_sumcheck<F>()
    where
        F: Field + Codec,
        Standard: Distribution<F>,
    {
        crate::tests::init();
        proptest!(|(seed: u64, config in Config::arbitrary())| {
            test_config(seed, &config);
        });
    }

    #[test]
    fn test_single_round() {
        test_config(
            0,
            &Config::<Field64> {
                field: Type::new(),
                initial_size: 2,
                round_pow: proof_of_work::Config::none(),
                num_rounds: 1,
                mask_length: 3,
            },
        );
    }

    #[test]
    fn test_two_rounds() {
        test_config(
            0,
            &Config::<Field64> {
                field: Type::new(),
                initial_size: 3,
                round_pow: proof_of_work::Config::none(),
                num_rounds: 2,
                mask_length: 3,
            },
        );
    }

    #[test]
    fn test_three_rounds() {
        test_config(
            0,
            &Config::<Field64> {
                field: Type::new(),
                initial_size: 5,
                round_pow: proof_of_work::Config::none(),
                num_rounds: 3,
                mask_length: 3,
            },
        );
    }

    #[test]
    fn test_field64_1() {
        test_sumcheck::<fields::Field64>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_2() {
        test_sumcheck::<fields::Field64_2>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field64_3() {
        test_sumcheck::<fields::Field64_3>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field128() {
        test_sumcheck::<fields::Field128>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field192() {
        test_sumcheck::<fields::Field192>();
    }

    #[test]
    #[ignore = "Somewhat expensive and redundant"]
    fn test_field256() {
        test_sumcheck::<fields::Field256>();
    }
}
