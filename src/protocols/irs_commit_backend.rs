use std::sync::{Arc, LazyLock};

use ark_ff::{FftField, Field};

use crate::{
    algebra::{
        fields,
        ntt::{NttEngine, ReedSolomon},
    },
    hash::Hash,
    protocols::{matrix_commit, merkle_tree},
    type_map::{self, TypeMap},
};

pub static IRS_COMMITTERS: LazyLock<TypeMap<IrsCommitterFamily>> = LazyLock::new(|| {
    let map = TypeMap::new();
    register_cpu_committer::<fields::Field64>(&map);
    register_cpu_committer::<fields::Field128>(&map);
    register_cpu_committer::<fields::Field192>(&map);
    register_cpu_committer::<fields::Field256>(&map);
    register_cpu_committer::<fields::Field64_2>(&map);
    register_cpu_committer::<fields::Field64_3>(&map);
    register_cpu_committer::<<fields::Field64_2 as Field>::BasePrimeField>(&map);
    register_cpu_committer::<<fields::Field64_3 as Field>::BasePrimeField>(&map);
    map
});

#[derive(Default)]
pub struct IrsCommitterFamily;

impl type_map::Family for IrsCommitterFamily {
    type Dyn<F: 'static> = dyn IrsCommitter<F>;
}

pub trait MatrixRows<F: Field>: Send + Sync {
    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;
    fn read_rows(&self, indices: &[usize]) -> Vec<F>;
}

pub trait IrsCommitter<F: Field>: ReedSolomon<F> + Send + Sync {
    fn commit(
        &self,
        messages: &[&[F]],
        masks: &[F],
        codeword_length: usize,
        matrix_commit: &matrix_commit::Config<F>,
    ) -> IrsCommitArtifact<F>;
}

pub struct IrsCommitArtifact<F: Field> {
    pub root: Hash,
    pub rows: Arc<dyn MatrixRows<F>>,
    pub matrix_witness: Arc<dyn merkle_tree::WitnessTrait + Send + Sync>,
}

#[derive(Clone)]
pub struct HostMatrixRows<F: Field> {
    pub matrix: Vec<F>,
    pub num_rows: usize,
    pub num_cols: usize,
}

impl<F: Field> MatrixRows<F> for HostMatrixRows<F> {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_cols(&self) -> usize {
        self.num_cols
    }

    fn read_rows(&self, indices: &[usize]) -> Vec<F> {
        let mut out = Vec::with_capacity(indices.len() * self.num_cols);
        for &index in indices {
            let start = index * self.num_cols;
            let end = start + self.num_cols;
            out.extend_from_slice(&self.matrix[start..end]);
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct CpuIrsCommitter<F: Field> {
    pub rs: Arc<dyn ReedSolomon<F>>,
}

impl<F: Field + 'static> CpuIrsCommitter<F> {
    pub fn new(rs: Arc<dyn ReedSolomon<F>>) -> Self {
        Self { rs }
    }
}

impl<F: Field + 'static> IrsCommitter<F> for CpuIrsCommitter<F> {
    fn commit(
        &self,
        messages: &[&[F]],
        masks: &[F],
        codeword_length: usize,
        matrix_commit: &matrix_commit::Config<F>,
    ) -> IrsCommitArtifact<F> {
        let matrix = self.interleaved_encode(messages, masks, codeword_length);
        let matrix_witness = matrix_commit.build_witness(&matrix);
        let root = merkle_tree::WitnessTrait::root(&matrix_witness);
        IrsCommitArtifact {
            root,
            rows: Arc::new(HostMatrixRows {
                matrix,
                num_rows: codeword_length,
                num_cols: matrix_commit.num_cols,
            }),
            matrix_witness: Arc::new(matrix_witness),
        }
    }
}

impl<F: Field + 'static> ReedSolomon<F> for CpuIrsCommitter<F> {
    fn next_order(&self, size: usize) -> Option<usize> {
        self.rs.next_order(size)
    }

    fn generator(&self, codeword_length: usize) -> F {
        self.rs.generator(codeword_length)
    }

    fn evaluation_points(
        &self,
        masked_message_length: usize,
        codeword_length: usize,
        indices: &[usize],
    ) -> Vec<F> {
        self.rs
            .evaluation_points(masked_message_length, codeword_length, indices)
    }

    fn interleaved_encode(&self, messages: &[&[F]], masks: &[F], codeword_length: usize) -> Vec<F> {
        self.rs.interleaved_encode(messages, masks, codeword_length)
    }
}

pub fn irs_committer<F: Field + 'static>() -> Arc<dyn IrsCommitter<F>> {
    IRS_COMMITTERS.get::<F>().expect("Unsupported IRS field.")
}

fn register_cpu_committer<F: FftField + 'static>(map: &TypeMap<IrsCommitterFamily>) {
    let rs = Arc::new(NttEngine::<F>::new_from_fftfield()) as Arc<dyn ReedSolomon<F>>;
    map.insert(Arc::new(CpuIrsCommitter::new(rs)) as Arc<dyn IrsCommitter<F>>);
}
