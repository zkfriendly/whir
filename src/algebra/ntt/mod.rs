//! NTT and related algorithms.

mod cooley_tukey;
mod matrix;
mod transpose;
mod utils;
mod wavelet;

use std::fmt::Debug;

use static_assertions::assert_obj_safe;

use self::matrix::MatrixMut;
pub use self::{
    cooley_tukey::NttEngine,
    transpose::transpose,
    wavelet::{inverse_wavelet_transform, wavelet_transform},
};

/// Trait for a Reed-Solomon encoder implementation for a given field `F`.
pub trait ReedSolomon<F>: Debug + Send + Sync {
    /// Returns the next supported order equal or larger than `size`.
    ///
    /// The result will be an NTT-smooth number suitable for `codeword_length`.
    ///
    /// Returns `None` if `size` exceeds the largest supported order.
    fn next_order(&self, size: usize) -> Option<usize>;

    fn generator(&self, codeword_length: usize) -> F;

    /// Returns the `index`th evaluation point.
    ///
    /// `masked_message_length`: the total message length including any mask values.
    ///
    /// # Panics
    ///
    /// Panics if any of the indices are `>= codeword_length` or `order` is not supported.
    fn evaluation_points(
        &self,
        masked_message_length: usize,
        codeword_length: usize,
        indices: &[usize],
    ) -> Vec<F>;

    /// Compute a masked interleaved Reed-Solomon encoding.
    ///
    /// `messages` are `num_messages` slices of `message_length` elements.
    /// `masks` is a `num_messages` × `mask_length` matrix of blinding coefficients.
    /// `codeword_length` must be an NTT-smooth number >= `message_length + mask_length`.
    /// returns an `codeword_length × num_messages` matrix.
    ///
    /// Each output value is the univariate polynomial evaluation in the evaluation point
    /// corresponding with the index of a coefficient list formed by concatenating message and mask.
    ///
    fn interleaved_encode(&self, messages: &[&[F]], masks: &[F], codeword_length: usize) -> Vec<F>;
}

assert_obj_safe!(ReedSolomon<crate::algebra::fields::Field256>);

#[cfg(test)]
mod tests {
    use std::iter;

    use ark_ff::FftField;
    use ark_std::rand::{
        distributions::Standard, prelude::Distribution, rngs::StdRng, SeedableRng,
    };
    use proptest::{collection, prelude::Just, proptest, sample::select, strategy::Strategy};

    use super::*;
    use crate::{
        algebra::{random_vector, univariate_evaluate},
        utils::{chunks_exact_or_empty, zip_strict},
    };

    fn valid_codeword_lengths<F: FftField>(size: usize, count: usize) -> Vec<usize> {
        let ntt = NttEngine::<F>::new_from_fftfield();
        iter::successors(ntt.next_order(size), |size| ntt.next_order(*size + 1))
            .take(count)
            .collect()
    }

    fn test<F: FftField>(ntt: &dyn ReedSolomon<F>)
    where
        Standard: Distribution<F>,
    {
        let cases = (
            0_usize..10,
            0_usize..(1 << 10),
            0_usize..(1 << 10),
            1_usize..=32,
        )
            .prop_flat_map(|(num_messages, message_length, mask_length, sample_size)| {
                let valid_codeword_lengths =
                    valid_codeword_lengths::<F>(message_length + mask_length, 6);
                select(valid_codeword_lengths).prop_flat_map(move |codeword_length| {
                    let sample_size = sample_size.min(codeword_length.max(1));
                    (
                        Just(num_messages),
                        Just(message_length),
                        Just(mask_length),
                        Just(codeword_length),
                        collection::vec(0..codeword_length, sample_size),
                    )
                })
            });
        proptest!(|(
            seed: u64,
            (num_messages, message_length, mask_length, codeword_length, sampled_indices) in cases
        )| {
            let mut rng = StdRng::seed_from_u64(seed);
            let messages = (0..num_messages)
                .map(|_| random_vector(&mut rng, message_length))
                .collect::<Vec<_>>();
            let masks = random_vector(&mut rng, mask_length * num_messages);
            let message_refs = messages.iter().map(|v| v.as_slice()).collect::<Vec<_>>();
            let codeword = ntt.interleaved_encode(
                &message_refs,
                &masks,
                codeword_length,
            );

            // Output must be the right size.
            assert_eq!(codeword.len(), codeword_length * num_messages);

            // Output values are polynomial evaluations in the evaluation points.
            let mut evaluation_points = ntt.evaluation_points(message_length + mask_length, codeword_length, &sampled_indices);
            for (&index, &evaluation_point) in zip_strict(&sampled_indices, &evaluation_points) {
                let evaluations = &codeword[index * num_messages.. (index + 1) * num_messages];
                let masks = chunks_exact_or_empty(&masks, mask_length, num_messages);
                for ((message, mask), value) in zip_strict(zip_strict(&messages, masks), evaluations) {
                    assert_eq!(*value,
                        univariate_evaluate(message, evaluation_point)
                        + evaluation_point.pow([message_length as u64])
                        * univariate_evaluate(mask, evaluation_point));
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
        test::<crate::algebra::fields::Field64>(
            &NttEngine::<crate::algebra::fields::Field64>::new_from_fftfield(),
        );
    }
}
