//! NTT and related algorithms.

mod cooley_tukey;
mod matrix;
mod transpose;
mod utils;
mod wavelet;

use std::{
    fmt::Debug,
    sync::{Arc, LazyLock},
};

use ark_ff::Field;
use static_assertions::assert_obj_safe;
#[cfg(feature = "tracing")]
use tracing::instrument;

use self::matrix::MatrixMut;
pub use self::{
    cooley_tukey::NttEngine,
    transpose::transpose,
    wavelet::{inverse_wavelet_transform, wavelet_transform},
};
use crate::{
    algebra::fields,
    type_map::{self, TypeMap},
};

pub static NTT: LazyLock<TypeMap<NttFamily>> = LazyLock::new(|| {
    let map = TypeMap::new();
    map.insert(
        Arc::new(NttEngine::<fields::Field64>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>
    );
    map.insert(
        Arc::new(NttEngine::<fields::Field128>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>
    );
    map.insert(
        Arc::new(NttEngine::<fields::Field192>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>
    );
    map.insert(
        Arc::new(NttEngine::<fields::Field256>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>
    );
    map.insert(
        Arc::new(NttEngine::<fields::Field64_2>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>,
    );
    map.insert(
        Arc::new(NttEngine::<fields::Field64_3>::new_from_fftfield()) as Arc<dyn ReedSolomon<_>>,
    );
    map.insert(Arc::new(
        NttEngine::<<fields::Field64_2 as Field>::BasePrimeField>::new_from_fftfield(),
    ) as Arc<dyn ReedSolomon<_>>);
    map.insert(Arc::new(
        NttEngine::<<fields::Field64_3 as Field>::BasePrimeField>::new_from_fftfield(),
    ) as Arc<dyn ReedSolomon<_>>);
    map
});

#[derive(Default)]
pub struct NttFamily;

impl type_map::Family for NttFamily {
    type Dyn<F: 'static> = dyn ReedSolomon<F>;
}

/// Trait for a Reed-Solomon encoder implementation for a given field `F`.
pub trait ReedSolomon<F>: Debug + Send + Sync {
    /// Returns the next supported order equal or larger than `size`.
    ///
    /// The result will be an NTT-smooth number suitable for `codeword_length`.
    ///
    /// Returns `None` if `size` exceeds the largest supported order.
    fn next_order(&self, size: usize) -> Option<usize>;

    /// Returns the `indext`th evaluation point for a `order` sized codeword.
    ///
    /// Returns `None` if the `order` is not supported or `index ≥ order`.
    fn evaluation_point(&self, order: usize, index: usize) -> Option<F>;

    /// Compute a batch NTT.
    ///
    /// The function signature is designed specifically for [`irs_commit`].
    ///
    /// `coeffs` are `num_vector` slices of `vector_length` elements.
    /// `mask` is a `num_messages` × `mask_length` matrix.
    /// `codeword_length` must be an NTT-smooth number >= `message_length`
    /// `interleaving_depth` must be at least `1`.
    /// returns an `codeword_length × num_messages` matrix.
    ///
    /// where
    ///
    /// `message_length = vector_length / interleaving_depth + mask_length`.
    /// `num_messages = num_vectors ·  interleaving_depth`.
    fn interleaved_encode(
        &self,
        interleaved_coeffs: &[&[F]],
        mask: &[F],
        codeword_length: usize,
        interleaving_depth: usize,
    ) -> Vec<F>;
}

assert_obj_safe!(ReedSolomon<crate::algebra::fields::Field256>);

pub fn next_order<F: 'static>(size: usize) -> Option<usize> {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .next_order(size)
}

pub fn evaluation_point<F: 'static>(order: usize, index: usize) -> Option<F> {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .evaluation_point(order, index)
}

pub fn interleaved_rs_encode<F: 'static>(
    interleaved_coeffs: &[&[F]],
    mask: &[F],
    codeword_length: usize,
    interleaving_depth: usize,
) -> Vec<F> {
    NTT.get::<F>()
        .expect("Unsupported NTT field.")
        .interleaved_encode(
            interleaved_coeffs,
            mask,
            codeword_length,
            interleaving_depth,
        )
}

#[cfg(test)]
mod tests {
    use std::iter;

    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, Rng, SeedableRng,
    };
    use proptest::{collection, prelude::Just, proptest, sample::select, strategy::Strategy};

    use super::*;
    use crate::{algebra::univariate_evaluate, utils::zip_strict};

    fn valid_codeword_lengths<F: 'static>(size: usize, count: usize) -> Vec<usize> {
        let ntt = NTT.get::<F>().expect("No NTT engine for field.");
        iter::successors(ntt.next_order(size), |size| ntt.next_order(*size + 1))
            .take(count)
            .collect()
    }

    fn test<F: Field>(ntt: &dyn ReedSolomon<F>)
    where
        Standard: Distribution<F>,
    {
        let cases = (0_usize..3, 0_usize..(1 << 10), 1_usize..=8, 1_usize..=32).prop_flat_map(
            |(num_vectors, message_length, interleaving_depth, sample_size)| {
                let valid_codeword_lengths = valid_codeword_lengths::<F>(message_length, 6);
                select(valid_codeword_lengths).prop_flat_map(move |codeword_length| {
                    let sample_size = sample_size.min(codeword_length.max(1));
                    (
                        Just(num_vectors),
                        Just(message_length),
                        Just(codeword_length),
                        Just(interleaving_depth),
                        collection::vec(0..codeword_length, sample_size),
                    )
                })
            },
        );
        proptest!(|(
            seed: u64,
            (num_vectors, message_length, codeword_length, interleaving_depth, sampled_indices) in cases
        )| {
            let block_length = interleaving_depth * num_vectors;
            let mut rng = StdRng::seed_from_u64(seed);
            let vector = (0..num_vectors).map(|_| (0..message_length * interleaving_depth)
                .map(|_| rng.gen::<F>())
                .collect::<Vec<_>>()).collect::<Vec<_>>();
            let vector_refs = vector.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
            let codeword = ntt.interleaved_encode(
                &vector_refs,
                &[],
                codeword_length,
                interleaving_depth,
            );

            // Output must be the right size.
            assert_eq!(codeword.len(), codeword_length * block_length);

            // Output valus are polynomial evaluations in the evaluation points.
            let mut evaluation_points = Vec::new();
            for &index in &sampled_indices {
                let evaluation_point = ntt.evaluation_point(codeword_length, index).unwrap();
                evaluation_points.push(evaluation_point);
                let evaluations = &codeword[index * block_length.. (index + 1) * block_length];
                if message_length > 0 {
                    for (coeffs, value) in vector.iter().flat_map(|v| v.chunks_exact(message_length)).zip(evaluations) {
                        assert_eq!(*value, univariate_evaluate(coeffs, evaluation_point));
                    }
                } else {
                    assert!(evaluations.iter().all(|v| *v == F::ZERO));
                }
            }

            // Evaluation points are unique.
            let mut sample_indices = sampled_indices;
            sample_indices.sort_unstable();
            sample_indices.dedup();
            evaluation_points.sort_unstable();
            evaluation_points.dedup();
            assert_eq!(sample_indices.len(), evaluation_points.len());
        });
    }

    #[test]
    fn test_field64_1() {
        test::<fields::Field64>(NTT.get().unwrap().as_ref());
    }
}
