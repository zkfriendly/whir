use ark_crypto_primitives::{
    crh::CRHScheme,
    merkle_tree::{Config, LeafParam, MerkleTree, MultiPath, TwoToOneParam},
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::{ProofError, ProofResult};

use crate::{
    crypto::merkle_tree::proof::FullMultiPath,
    parameters::MerkleProofStrategy,
    whir::utils::{HintDeserialize, HintSerialize},
};

/// Extension trait for `Config` that allows batch computation of leaf digests.
///
/// Override `batch_build_leaf_digests` to provide optimized batch hashing
/// (e.g., SIMD-accelerated Skyscraper hashing). The default implementation
/// hashes each leaf individually using `LeafHash::evaluate`.
pub trait BatchLeafDigest: Config {
    fn batch_build_leaf_digests<L: AsRef<Self::Leaf> + Send + Sync>(
        leaf_hash_param: &LeafParam<Self>,
        leaves: &[L],
    ) -> Result<Vec<Self::LeafDigest>, ark_crypto_primitives::Error>
    where
        Self: Sized,
    {
        #[cfg(feature = "parallel")]
        let iter = leaves.par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = leaves.iter();

        iter.map(|input| Self::LeafHash::evaluate(leaf_hash_param, input.as_ref()))
            .collect()
    }
}

/// Build a Merkle tree using batch leaf digest computation.
///
/// Calls `MC::batch_build_leaf_digests` to compute leaf digests, then
/// delegates to `MerkleTree::new_with_leaf_digest` for inner node construction.
pub fn build_merkle_tree<MC: BatchLeafDigest, L: AsRef<MC::Leaf> + Send + Sync>(
    leaf_hash_param: &LeafParam<MC>,
    two_to_one_param: &TwoToOneParam<MC>,
    leaves: Vec<L>,
) -> Result<MerkleTree<MC>, ark_crypto_primitives::Error> {
    let leaf_digests = MC::batch_build_leaf_digests(leaf_hash_param, &leaves)?;
    MerkleTree::new_with_leaf_digest(leaf_hash_param, two_to_one_param, leaf_digests)
}

/// Prover-side state for emitting Merkle proofs using a fixed strategy.
pub struct ProverMerkleState {
    strategy: MerkleProofStrategy,
}

impl ProverMerkleState {
    pub const fn new(strategy: MerkleProofStrategy) -> Self {
        Self { strategy }
    }

    pub fn write_proof_hint<ProverState, MerkleConfig>(
        &self,
        tree: &MerkleTree<MerkleConfig>,
        indices: &[usize],
        prover_state: &mut ProverState,
    ) -> ProofResult<()>
    where
        MerkleConfig: Config,
        ProverState: HintSerialize,
    {
        match self.strategy {
            MerkleProofStrategy::Compressed => {
                let merkle_proof = tree
                    .generate_multi_proof(indices.to_vec())
                    .expect("indices sampled from transcript must be valid for the tree");
                prover_state.hint::<MultiPath<MerkleConfig>>(&merkle_proof)?;
            }
            MerkleProofStrategy::Uncompressed => {
                let proofs = indices
                    .iter()
                    .map(|&index| {
                        tree.generate_proof(index)
                            .expect("indices sampled from transcript must be valid for the tree")
                    })
                    .collect::<Vec<_>>();
                let merkle_proof: FullMultiPath<MerkleConfig> = proofs.into();

                prover_state.hint::<FullMultiPath<MerkleConfig>>(&merkle_proof)?;
            }
        }

        Ok(())
    }
}

/// Verifier-side state for consuming Merkle proofs using a fixed strategy and hash parameters.
pub struct VerifierMerkleState<'a, MerkleConfig>
where
    MerkleConfig: Config,
{
    strategy: MerkleProofStrategy,
    leaf_hash_params: &'a LeafParam<MerkleConfig>,
    two_to_one_params: &'a TwoToOneParam<MerkleConfig>,
}

impl<'a, MerkleConfig> VerifierMerkleState<'a, MerkleConfig>
where
    MerkleConfig: Config,
{
    pub const fn new(
        strategy: MerkleProofStrategy,
        leaf_hash_params: &'a LeafParam<MerkleConfig>,
        two_to_one_params: &'a TwoToOneParam<MerkleConfig>,
    ) -> Self {
        Self {
            strategy,
            leaf_hash_params,
            two_to_one_params,
        }
    }

    pub fn read_and_verify_proof<VerifierState, L>(
        &self,
        verifier_state: &mut VerifierState,
        indices: &[usize],
        root: &MerkleConfig::InnerDigest,
        leaves: impl IntoIterator<Item = L>,
    ) -> ProofResult<()>
    where
        VerifierState: HintDeserialize,
        L: Clone + std::borrow::Borrow<MerkleConfig::Leaf>,
    {
        let leaves: Vec<L> = leaves.into_iter().collect();

        match self.strategy {
            MerkleProofStrategy::Compressed => {
                let merkle_proof: MultiPath<MerkleConfig> = verifier_state.hint()?;
                if merkle_proof.leaf_indexes != indices {
                    return Err(ProofError::InvalidProof);
                }

                let correct = merkle_proof
                    .verify(
                        self.leaf_hash_params,
                        self.two_to_one_params,
                        root,
                        leaves.iter().cloned(),
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
                if !correct {
                    return Err(ProofError::InvalidProof);
                }
            }
            MerkleProofStrategy::Uncompressed => {
                let merkle_proof: FullMultiPath<MerkleConfig> = verifier_state.hint()?;
                if merkle_proof.indices() != indices {
                    return Err(ProofError::InvalidProof);
                }

                let correct = merkle_proof
                    .verify(
                        self.leaf_hash_params,
                        self.two_to_one_params,
                        root,
                        leaves.iter().cloned(),
                    )
                    .map_err(|_| ProofError::InvalidProof)?;
                if !correct {
                    return Err(ProofError::InvalidProof);
                }
            }
        }

        Ok(())
    }
}
