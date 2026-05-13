use std::{any::Any, borrow::Cow, mem, sync::Arc, time::Instant};

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;

use super::{Config, Witness};
use crate::{
    algebra::{
        dot,
        embedding::Identity,
        eq_weights,
        linear_form::{Covector, Evaluate, LinearForm, UnivariateEvaluation},
        tensor_product, univariate_evaluate,
    },
    hash::Hash,
    protocols::{
        geometric_challenge::geometric_challenge,
        irs_commit, sumcheck,
        whir::FinalClaim,
        whir_accelerator::{whir_prover_accelerator, DeviceVector, WhirProverAccelerator},
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
    utils::{chunks_exact_or_empty, zip_strict},
};

enum RoundWitness<'a, F: Field> {
    Initial(Vec<Cow<'a, irs_commit::Witness<F, F>>>),
    Round(irs_commit::Witness<F, F>),
}

const GPU_SUMCHECK_MIN_SIZE: usize = 1 << 20;

impl<F: Field + 'static> Config<Identity<F>> {
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn prove_accelerated<'a, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: Vec<Cow<'a, [F]>>,
        witnesses: Vec<Cow<'a, Witness<F, Identity<F>>>>,
        linear_forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: Cow<'a, [F]>,
    ) -> FinalClaim<F>
    where
        Standard: Distribution<F>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.prove_accelerated_inner(
            prover_state,
            vectors,
            witnesses,
            linear_forms,
            evaluations,
            None,
            None,
        )
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn prove_accelerated_with_device_vector<'a, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vector: Cow<'a, [F]>,
        device_vector: Box<dyn DeviceVector<F>>,
        witness: Cow<'a, Witness<F, Identity<F>>>,
        linear_forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: Cow<'a, [F]>,
    ) -> FinalClaim<F>
    where
        Standard: Distribution<F>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(device_vector.len(), self.initial_size());
        self.prove_accelerated_with_device_vectors(
            prover_state,
            vec![vector],
            vec![device_vector],
            vec![witness],
            linear_forms,
            evaluations,
        )
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn prove_accelerated_with_device_vectors<'a, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: Vec<Cow<'a, [F]>>,
        device_vectors: Vec<Box<dyn DeviceVector<F>>>,
        witnesses: Vec<Cow<'a, Witness<F, Identity<F>>>>,
        linear_forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: Cow<'a, [F]>,
    ) -> FinalClaim<F>
    where
        Standard: Distribution<F>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.prove_accelerated_inner(
            prover_state,
            vectors,
            witnesses,
            linear_forms,
            evaluations,
            Some(device_vectors),
            None,
        )
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn prove_accelerated_with_device_vectors_and_first_linear_form<'a, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: Vec<Cow<'a, [F]>>,
        device_vectors: Vec<Box<dyn DeviceVector<F>>>,
        witnesses: Vec<Cow<'a, Witness<F, Identity<F>>>>,
        linear_forms: Vec<Box<dyn LinearForm<F>>>,
        first_linear_form: Box<dyn DeviceVector<F>>,
        evaluations: Cow<'a, [F]>,
    ) -> FinalClaim<F>
    where
        Standard: Distribution<F>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        self.prove_accelerated_inner(
            prover_state,
            vectors,
            witnesses,
            linear_forms,
            evaluations,
            Some(device_vectors),
            Some(first_linear_form),
        )
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn prove_accelerated_inner<'a, H, R>(
        &self,
        prover_state: &mut ProverState<H, R>,
        vectors: Vec<Cow<'a, [F]>>,
        witnesses: Vec<Cow<'a, Witness<F, Identity<F>>>>,
        linear_forms: Vec<Box<dyn LinearForm<F>>>,
        evaluations: Cow<'a, [F]>,
        device_vectors: Option<Vec<Box<dyn DeviceVector<F>>>>,
        first_linear_form: Option<Box<dyn DeviceVector<F>>>,
    ) -> FinalClaim<F>
    where
        Standard: Distribution<F>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let Some(accelerator) = whir_prover_accelerator::<F>() else {
            return self.prove(prover_state, vectors, witnesses, linear_forms, evaluations);
        };

        let num_vectors = vectors.len();
        assert_eq!(
            num_vectors,
            witnesses.len() * self.initial_committer.num_vectors
        );
        assert_eq!(evaluations.len(), num_vectors * linear_forms.len());
        for vector in &vectors {
            assert_eq!(vector.len(), self.initial_size());
        }
        for linear_form in &linear_forms {
            assert_eq!(linear_form.size(), self.initial_size());
        }
        #[cfg(debug_assertions)]
        for (linear_form, evaluations) in
            zip_strict(linear_forms.iter(), evaluations.chunks_exact(num_vectors))
        {
            let covector = Covector::from(&**linear_form);
            for (vector, evaluation) in zip_strict(&vectors, evaluations) {
                debug_assert_eq!(covector.evaluate(self.embedding(), vector), *evaluation);
            }
        }
        if vectors.is_empty() {
            return FinalClaim::default();
        }

        let (oods_evals, oods_matrix) = {
            let mut oods_evals = Vec::new();
            let mut oods_matrix = Vec::new();
            let mut vector_offset = 0;
            for witness in &witnesses {
                for (oods_eval, oods_row) in zip_strict(
                    witness.out_of_domain().evaluators(self.initial_size()),
                    witness.out_of_domain().rows(),
                ) {
                    for (j, vector) in vectors.iter().enumerate() {
                        if j >= vector_offset && j < oods_row.len() + vector_offset {
                            debug_assert_eq!(
                                oods_row[j - vector_offset],
                                oods_eval.evaluate(self.embedding(), vector)
                            );
                            oods_matrix.push(oods_row[j - vector_offset]);
                        } else {
                            let eval = oods_eval.evaluate(self.embedding(), vector);
                            prover_state.prover_message(&eval);
                            oods_matrix.push(eval);
                        }
                    }
                    oods_evals.push(oods_eval);
                }
                vector_offset += witness.num_vectors();
            }
            (oods_evals, oods_matrix)
        };

        let mut vector_rlc_coeffs: Vec<F> = geometric_challenge(prover_state, num_vectors);
        assert_eq!(vector_rlc_coeffs[0], F::ONE);
        let mut vector_inputs = vectors.into_iter();
        let first = vector_inputs.next().expect("non-empty");
        let mut vector = if let Some(mut device_vectors) = device_vectors {
            assert_eq!(device_vectors.len(), num_vectors);
            let mut vector = device_vectors.remove(0);
            assert_eq!(vector.len(), first.len());
            for (rlc_coeff, (input_vector, device_vector)) in zip_strict(
                &vector_rlc_coeffs[1..],
                zip_strict(vector_inputs, device_vectors),
            ) {
                assert_eq!(device_vector.len(), input_vector.len());
                accelerator.add_assign_scaled_device_vector(
                    &mut *vector,
                    *rlc_coeff,
                    &*device_vector,
                );
            }
            vector
        } else {
            let mut vector = accelerator.upload(&first);
            for (rlc_coeff, input_vector) in zip_strict(&vector_rlc_coeffs[1..], vector_inputs) {
                accelerator.add_assign_scaled_slice(&mut *vector, *rlc_coeff, &input_vector);
            }
            vector
        };

        let mut prev_witness: RoundWitness<'a, F> = RoundWitness::Initial(witnesses);

        let constraint_rlc_coeffs: Vec<F> =
            geometric_challenge(prover_state, linear_forms.len() + oods_evals.len());
        let has_constraints = !constraint_rlc_coeffs.is_empty();
        let (initial_forms_rlc_coeffs, oods_rlc_coeffs) =
            constraint_rlc_coeffs.split_at(linear_forms.len());

        let mut first_linear_form = first_linear_form;
        let mut covector_cpu = vec![];
        let mut linear_forms = linear_forms;
        let mut covector = if let Some((first, linear_forms)) = linear_forms.split_first_mut() {
            debug_assert_eq!(initial_forms_rlc_coeffs[0], F::ONE);
            if let Some(mut covector) = first_linear_form.take() {
                assert_eq!(covector.len(), self.initial_size());
                assert_eq!(first.size(), covector.len());
                for (rlc_coeff, linear_form) in
                    zip_strict(&initial_forms_rlc_coeffs[1..], linear_forms)
                {
                    if let Some(covector_form) =
                        (linear_form.as_mut() as &mut dyn Any).downcast_mut::<Covector<F>>()
                    {
                        accelerator.add_assign_scaled_slice(
                            &mut *covector,
                            *rlc_coeff,
                            &covector_form.vector,
                        );
                    } else {
                        covector_cpu.clear();
                        covector_cpu.resize(self.initial_size(), F::ZERO);
                        linear_form.accumulate(&mut covector_cpu, *rlc_coeff);
                        accelerator.add_assign_scaled_slice(&mut *covector, F::ONE, &covector_cpu);
                    }
                }
                covector
            } else {
                if let Some(covector_form) =
                    (first.as_mut() as &mut dyn Any).downcast_mut::<Covector<F>>()
                {
                    mem::swap(&mut covector_cpu, &mut covector_form.vector);
                } else {
                    covector_cpu.resize(self.initial_size(), F::ZERO);
                    first.accumulate(&mut covector_cpu, F::ONE);
                }
                for (rlc_coeff, linear_form) in
                    zip_strict(&initial_forms_rlc_coeffs[1..], linear_forms)
                {
                    linear_form.accumulate(&mut covector_cpu, *rlc_coeff);
                }
                accelerator.upload(&covector_cpu)
            }
        } else if has_constraints {
            accelerator.upload_zeroes(self.initial_size())
        } else {
            accelerator.upload_zeroes(self.initial_size())
        };
        drop(linear_forms);
        drop(covector_cpu);

        let mut the_sum: F = zip_strict(
            initial_forms_rlc_coeffs,
            evaluations.chunks_exact(num_vectors),
        )
        .map(|(poly_coeff, row)| *poly_coeff * dot(&vector_rlc_coeffs, row))
        .sum();
        drop(evaluations);

        accelerator.accumulate_univariate_evaluations(
            &mut *covector,
            &oods_points(&oods_evals),
            oods_rlc_coeffs,
        );
        the_sum += zip_strict(oods_rlc_coeffs, oods_matrix.chunks_exact(num_vectors))
            .map(|(poly_coeff, row)| *poly_coeff * dot(&vector_rlc_coeffs, row))
            .sum::<F>();
        drop(oods_evals);
        drop(oods_matrix);

        let mut folding_randomness = if has_constraints {
            accelerated_sumcheck(
                &self.initial_sumcheck,
                prover_state,
                &accelerator,
                &mut vector,
                &mut covector,
                &mut the_sum,
                &[],
            )
            .0
        } else {
            let folding_randomness = (0..self.initial_sumcheck.num_rounds)
                .map(|_| prover_state.verifier_message())
                .collect::<Vec<_>>();
            self.initial_skip_pow.prove(prover_state);
            for &f in &folding_randomness {
                accelerator.fold(&mut *vector, f);
            }
            covector = accelerator.upload_zeroes(self.initial_sumcheck.final_size());
            folding_randomness
        };
        let mut evaluation_point = folding_randomness.clone();

        for (round_index, round_config) in self.round_configs.iter().enumerate() {
            let new_witness = accelerated_commit(
                prover_state,
                &accelerator,
                &round_config.irs_committer,
                &*vector,
            );

            round_config.pow.prove(prover_state);

            let in_domain = match prev_witness {
                RoundWitness::Initial(init_witnesses) => {
                    let witness_refs: Vec<&_> = init_witnesses.iter().map(|c| &**c).collect();
                    self.initial_committer
                        .open(prover_state, &witness_refs)
                        .lift(self.embedding())
                }
                RoundWitness::Round(old_witness) => {
                    let prev_round_config = &self.round_configs[round_index - 1];
                    prev_round_config
                        .irs_committer
                        .open(prover_state, &[&old_witness])
                }
            };

            let stir_challenges = new_witness
                .out_of_domain()
                .evaluators(round_config.initial_size())
                .chain(in_domain.evaluators(round_config.initial_size()))
                .collect::<Vec<_>>();
            let stir_evaluations = new_witness
                .out_of_domain()
                .values(&[F::ONE])
                .chain(in_domain.values(&tensor_product(
                    &vector_rlc_coeffs,
                    &eq_weights(&folding_randomness),
                )))
                .collect::<Vec<_>>();
            let stir_rlc_coeffs = geometric_challenge(prover_state, stir_challenges.len());
            accelerator.accumulate_univariate_evaluations(
                &mut *covector,
                &oods_points(&stir_challenges),
                &stir_rlc_coeffs,
            );
            the_sum += dot(&stir_rlc_coeffs, &stir_evaluations);

            folding_randomness = accelerated_sumcheck(
                &round_config.sumcheck,
                prover_state,
                &accelerator,
                &mut vector,
                &mut covector,
                &mut the_sum,
                &[],
            )
            .0;

            evaluation_point.extend(folding_randomness.iter().copied());
            prev_witness = RoundWitness::Round(new_witness);
            vector_rlc_coeffs = vec![F::ONE];
        }

        assert_eq!(vector.len(), self.final_sumcheck.initial_size);
        for coeff in accelerator.download(&*vector) {
            prover_state.prover_message(&coeff);
        }

        self.final_pow.prove(prover_state);

        match prev_witness {
            RoundWitness::Initial(init_witnesses) => {
                let witness_refs: Vec<&_> = init_witnesses.iter().map(|c| &**c).collect();
                let _in_domain = self.initial_committer.open(prover_state, &witness_refs);
            }
            RoundWitness::Round(old_witness) => {
                let prev_config = self.round_configs.last().unwrap();
                let _in_domain = prev_config
                    .irs_committer
                    .open(prover_state, &[&old_witness]);
            }
        }

        let final_folding_randomness = accelerated_sumcheck(
            &self.final_sumcheck,
            prover_state,
            &accelerator,
            &mut vector,
            &mut covector,
            &mut the_sum,
            &[],
        )
        .0;
        evaluation_point.extend(final_folding_randomness.iter().copied());

        FinalClaim {
            evaluation_point,
            rlc_coefficients: initial_forms_rlc_coeffs.to_vec(),
            linear_form_rlc: F::ZERO,
        }
    }
}

fn oods_points<F: Field>(evaluators: &[UnivariateEvaluation<F>]) -> Vec<F> {
    evaluators.iter().map(|e| e.point).collect()
}

fn accelerated_commit<H, R, F>(
    prover_state: &mut ProverState<H, R>,
    accelerator: &Arc<dyn WhirProverAccelerator<F>>,
    config: &irs_commit::Config<Identity<F>>,
    vector: &dyn DeviceVector<F>,
) -> irs_commit::Witness<F, F>
where
    Standard: Distribution<F>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    F: Field + Codec<[H::U]> + 'static,
    Hash: ProverMessage<[H::U]>,
{
    assert_eq!(config.num_vectors, 1);
    assert_eq!(vector.len(), config.vector_size);

    let masks = crate::algebra::random_vector(
        prover_state.rng(),
        config.mask_length * config.num_messages(),
    );
    let committed = accelerator
        .commit_device_vector(
            vector,
            &masks,
            config.codeword_length,
            config.interleaving_depth,
            &config.matrix_commit,
        )
        .expect("WHIR accelerator failed to commit device vector");
    prover_state.prover_message(&committed.root);

    let oods_points: Vec<F> = prover_state.verifier_message_vec(config.out_domain_samples);
    let oods_matrix = accelerator.evaluate_univariate_many(vector, &oods_points);
    for value in &oods_matrix {
        prover_state.prover_message(value);
    }

    irs_commit::Witness {
        masks,
        rows: committed.rows,
        matrix_witness: committed.matrix_witness,
        out_of_domain: irs_commit::Evaluations {
            points: oods_points,
            matrix: oods_matrix,
        },
    }
}

fn accelerated_sumcheck<H, R, F>(
    config: &sumcheck::Config<F>,
    prover_state: &mut ProverState<H, R>,
    accelerator: &Arc<dyn WhirProverAccelerator<F>>,
    a: &mut Box<dyn DeviceVector<F>>,
    b: &mut Box<dyn DeviceVector<F>>,
    sum: &mut F,
    masks: &[F],
) -> (Vec<F>, F, F)
where
    H: DuplexSpongeInterface,
    R: CryptoRng + RngCore,
    F: Field + Codec<[H::U]> + 'static,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
{
    assert!(
        config.num_rounds == 0 || config.initial_size.next_power_of_two() >= 1 << config.num_rounds
    );
    assert!(config.mask_length == 0 || config.mask_length >= 3);
    assert_eq!(a.len(), config.initial_size);
    assert_eq!(b.len(), config.initial_size);
    assert_eq!(masks.len(), config.num_rounds * config.mask_length);

    if config.num_rounds > 0 && config.initial_size < GPU_SUMCHECK_MIN_SIZE {
        let timing = std::env::var_os("PROVEKIT_WHIR_GPU_TIMING").is_some();
        let total_start = timing.then(Instant::now);
        let download_start = timing.then(Instant::now);
        let mut a_cpu = accelerator.download(&**a);
        let mut b_cpu = accelerator.download(&**b);
        let download_us = download_start.map_or(0, |start| start.elapsed().as_micros());

        let prove_start = timing.then(Instant::now);
        let result = config.prove(prover_state, &mut a_cpu, &mut b_cpu, sum, masks);
        let prove_us = prove_start.map_or(0, |start| start.elapsed().as_micros());

        let upload_start = timing.then(Instant::now);
        *a = accelerator.upload(&a_cpu);
        *b = accelerator.upload(&b_cpu);
        let upload_us = upload_start.map_or(0, |start| start.elapsed().as_micros());

        if let Some(total_start) = total_start {
            eprintln!(
                "WHIR_CPU_SUMCHECK len={} rounds={} download_us={} prove_us={} upload_us={} total_us={}",
                config.initial_size,
                config.num_rounds,
                download_us,
                prove_us,
                upload_us,
                total_start.elapsed().as_micros(),
            );
        }

        return result;
    }

    let half = F::from(2).inverse().unwrap();

    let mut mask_sum = F::ZERO;
    let mut mask_rlc = F::ONE;
    if !masks.is_empty() {
        let sum_multiple = F::from(1 << config.num_rounds.saturating_sub(1));
        mask_sum = masks
            .chunks_exact(config.mask_length)
            .map(eval_01)
            .sum::<F>()
            * sum_multiple;
        prover_state.prover_message(&mask_sum);
        mask_rlc = prover_state.verifier_message();
    }

    let mut univariate = Vec::new();
    let mut res = Vec::with_capacity(config.num_rounds);
    let mut folding_randomness = None;
    let timing = std::env::var_os("PROVEKIT_WHIR_GPU_TIMING").is_some();
    for (round, mask) in
        chunks_exact_or_empty(masks, config.mask_length, config.num_rounds).enumerate()
    {
        let round_start = Instant::now();
        let fold_start = Instant::now();
        if let Some(w) = folding_randomness {
            accelerator.fold(&mut **a, w);
            accelerator.fold(&mut **b, w);
        }
        let fold_elapsed = fold_start.elapsed();
        let poly_start = Instant::now();
        let (c0, c2) = accelerator.sumcheck_polynomial(&**a, &**b);
        let poly_elapsed = poly_start.elapsed();
        let transcript_start = Instant::now();
        let c1 = *sum - c0.double() - c2;

        if mask.is_empty() {
            prover_state.prover_messages(&[c0, c2]);
        } else {
            univariate.clear();
            let sum_multiple = F::from(1 << config.num_rounds.saturating_sub(round + 1));
            univariate.extend(mask.iter().map(|m| sum_multiple * *m));
            univariate[0] += (mask_sum - sum_multiple * eval_01(mask)) * half;
            univariate[0] += mask_rlc * c0;
            univariate[1] += mask_rlc * c1;
            univariate[2] += mask_rlc * c2;
            prover_state.prover_message(&univariate[0]);
            prover_state.prover_messages(&univariate[2..]);
        }

        config.round_pow.prove(prover_state);
        let r = prover_state.verifier_message::<F>();
        res.push(r);
        *sum = (c2 * r + c1) * r + c0;
        if !masks.is_empty() {
            let masked_sum = univariate_evaluate(&univariate, r);
            mask_sum = masked_sum - mask_rlc * *sum;
        }
        folding_randomness = Some(r);
        let transcript_elapsed = transcript_start.elapsed();
        if timing {
            eprintln!(
                "WHIR_GPU_SUMCHECK round={round} len={} fold_us={} polynomial_us={} transcript_us={} total_us={}",
                a.len(),
                fold_elapsed.as_micros(),
                poly_elapsed.as_micros(),
                transcript_elapsed.as_micros(),
                round_start.elapsed().as_micros(),
            );
        }
    }
    let final_fold_start = Instant::now();
    if let Some(w) = folding_randomness {
        accelerator.fold(&mut **a, w);
        accelerator.fold(&mut **b, w);
    }
    if timing {
        eprintln!(
            "WHIR_GPU_SUMCHECK final_fold len={} fold_us={}",
            a.len(),
            final_fold_start.elapsed().as_micros(),
        );
    }

    *sum = mask_sum + mask_rlc * *sum;
    (res, mask_sum, mask_rlc)
}

fn eval_01<F: Field>(coefficients: &[F]) -> F {
    if coefficients.is_empty() {
        return F::ZERO;
    }
    coefficients[0] + coefficients.iter().sum::<F>()
}
