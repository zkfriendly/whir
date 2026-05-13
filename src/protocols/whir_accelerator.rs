use std::{
    any::Any,
    sync::{Arc, LazyLock},
};

use ark_ff::Field;

use crate::{
    protocols::{irs_commit::IrsCommitArtifact, matrix_commit},
    type_map::{self, TypeMap},
};

pub static WHIR_PROVER_ACCELERATORS: LazyLock<TypeMap<WhirProverAcceleratorFamily>> =
    LazyLock::new(TypeMap::new);

#[derive(Default)]
pub struct WhirProverAcceleratorFamily;

impl type_map::Family for WhirProverAcceleratorFamily {
    type Dyn<F: 'static> = dyn WhirProverAccelerator<F>;
}

pub trait DeviceVector<F: Field>: Any + Send {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn len(&self) -> usize;
}

pub trait WhirProverAccelerator<F: Field>: Send + Sync {
    fn upload(&self, values: &[F]) -> Box<dyn DeviceVector<F>>;
    fn upload_zeroes(&self, len: usize) -> Box<dyn DeviceVector<F>>;
    fn download(&self, vector: &dyn DeviceVector<F>) -> Vec<F>;

    fn add_assign_scaled_slice(
        &self,
        accumulator: &mut dyn DeviceVector<F>,
        scalar: F,
        values: &[F],
    );
    fn add_assign_scaled_device_vector(
        &self,
        accumulator: &mut dyn DeviceVector<F>,
        scalar: F,
        values: &dyn DeviceVector<F>,
    );
    fn accumulate_univariate_evaluations(
        &self,
        accumulator: &mut dyn DeviceVector<F>,
        points: &[F],
        scalars: &[F],
    );

    fn dot(&self, a: &dyn DeviceVector<F>, b: &dyn DeviceVector<F>) -> F;
    fn fold(&self, vector: &mut dyn DeviceVector<F>, weight: F);
    fn sumcheck_polynomial(&self, a: &dyn DeviceVector<F>, b: &dyn DeviceVector<F>) -> (F, F);
    fn evaluate_univariate_many(&self, vector: &dyn DeviceVector<F>, points: &[F]) -> Vec<F>;
    #[allow(clippy::too_many_arguments)]
    fn evaluate_gamma_block(
        &self,
        blinding_vectors: &dyn DeviceVector<F>,
        gammas: &[F],
        masking_challenge: F,
        blinding_challenge: F,
        tau2: F,
        num_polynomials: usize,
        num_witness_variables: usize,
        num_blinding_variables: usize,
    ) -> Option<(Vec<F>, Box<dyn DeviceVector<F>>)> {
        let _ = (
            blinding_vectors,
            gammas,
            masking_challenge,
            blinding_challenge,
            tau2,
            num_polynomials,
            num_witness_variables,
            num_blinding_variables,
        );
        None
    }

    fn commit_device_vector(
        &self,
        vector: &dyn DeviceVector<F>,
        masks: &[F],
        codeword_length: usize,
        interleaving_depth: usize,
        matrix_commit: &matrix_commit::Config<F>,
    ) -> Option<IrsCommitArtifact<F>>;
}

pub fn whir_prover_accelerator<F: Field + 'static>() -> Option<Arc<dyn WhirProverAccelerator<F>>> {
    WHIR_PROVER_ACCELERATORS.get::<F>()
}
