use {
    super::utils::BlindingPolynomials,
    ark_bn254::{Fr, FrConfig},
    ark_ff::{AdditiveGroup, BigInt, FftField, Field, Fp, MontBackend},
    cudarc::{
        driver::{
            CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, DriverError,
            LaunchConfig, PushKernelArg, ValidAsZeroBits,
        },
        nvrtc::{compile_ptx_with_opts, CompileOptions},
    },
    std::{
        any::TypeId,
        collections::hash_map::DefaultHasher,
        env, fs,
        hash::{Hash as _, Hasher},
        marker::PhantomData,
        mem::{size_of, ManuallyDrop},
        path::PathBuf,
        sync::{Arc, Mutex, OnceLock},
    },
};

const DEFAULT_MIN_OPS: usize = 1 << 20;
const BLOCK_DIM: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct GpuField {
    limbs: [u64; 4],
}

unsafe impl DeviceRepr for GpuField {}
unsafe impl ValidAsZeroBits for GpuField {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct EvalParams {
    half_size:             u64,
    num_polynomials:       u64,
    num_witness_variables: u64,
    num_gammas:            u64,
    num_blinding_variables: u32,
    _padding:              u32,
    one_plus_rho:          GpuField,
    tau2:                  GpuField,
}

unsafe impl DeviceRepr for EvalParams {}

struct CudaGammaEngine {
    ctx:        Arc<CudaContext>,
    dot_kernel: CudaFunction,
    beq_kernel: CudaFunction,
    workspaces: Mutex<Vec<Workspace>>,
}

struct Workspace {
    stream:   Arc<CudaStream>,
    partials: Option<CudaSlice<GpuField>>,
    beq_half: Option<CudaSlice<GpuField>>,
}

struct WorkspaceLease<'a> {
    engine:    &'a CudaGammaEngine,
    workspace: Option<Workspace>,
}

static ENGINE: OnceLock<Result<Arc<CudaGammaEngine>, String>> = OnceLock::new();

pub(super) fn try_evaluate_gamma_block<F: FftField + 'static>(
    blinding_polynomials: &[BlindingPolynomials<F>],
    h_gammas: &[F],
    masking_challenge: F,
    blinding_challenge: F,
    tau2: F,
    num_blinding_variables: usize,
    num_witness_variables: usize,
) -> Option<(Vec<F>, Vec<F>)> {
    if env::var_os("WHIR_DISABLE_CUDA_GAMMA_BLOCK").is_some() {
        return None;
    }
    if TypeId::of::<F>() != TypeId::of::<Fr>() {
        return None;
    }

    let half_size = 1usize << num_blinding_variables;
    let op_count = half_size
        .saturating_mul(h_gammas.len())
        .saturating_mul(blinding_polynomials.len())
        .saturating_mul(num_witness_variables.saturating_add(1));
    let min_ops = env::var("WHIR_CUDA_GAMMA_BLOCK_MIN_OPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MIN_OPS);
    if op_count < min_ops {
        return None;
    }

    let blinding_polynomials =
        unsafe { cast_slice::<BlindingPolynomials<F>, BlindingPolynomials<Fr>>(blinding_polynomials) };
    let h_gammas = unsafe { cast_slice::<F, Fr>(h_gammas) };

    match evaluate_gamma_block_bn254(
        blinding_polynomials,
        h_gammas,
        unsafe { cast_scalar::<F, Fr>(masking_challenge) },
        unsafe { cast_scalar::<F, Fr>(blinding_challenge) },
        unsafe { cast_scalar::<F, Fr>(tau2) },
        num_blinding_variables,
        num_witness_variables,
    ) {
        Ok((eval_results, beq_weight_accum)) => {
            trace_event(format_args!(
                "gamma block cuda success gammas={} polynomials={} witness_vars={} half_size={} ops={}",
                h_gammas.len(),
                blinding_polynomials.len(),
                num_witness_variables,
                half_size,
                op_count,
            ));
            Some(unsafe {
                (
                    cast_vec::<Fr, F>(eval_results),
                    cast_vec::<Fr, F>(beq_weight_accum),
                )
            })
        }
        Err(err) => {
            trace_event(format_args!("gamma block fallback: {err}"));
            #[cfg(feature = "tracing")]
            tracing::warn!("CUDA gamma block fallback: {err}");
            None
        }
    }
}

fn evaluate_gamma_block_bn254(
    blinding_polynomials: &[BlindingPolynomials<Fr>],
    h_gammas: &[Fr],
    masking_challenge: Fr,
    blinding_challenge: Fr,
    tau2: Fr,
    num_blinding_variables: usize,
    num_witness_variables: usize,
) -> Result<(Vec<Fr>, Vec<Fr>), String> {
    let engine = engine()?;
    let num_polynomials = blinding_polynomials.len();
    let num_gammas = h_gammas.len();
    let half_size = 1usize << num_blinding_variables;
    let stride_per_poly = num_witness_variables + 2;
    let stride_per_gamma = num_polynomials * stride_per_poly;

    if num_polynomials == 0 || num_gammas == 0 {
        return Ok((Vec::new(), vec![Fr::ZERO; half_size * 2]));
    }

    let one_plus_rho = Fr::ONE + masking_challenge;
    let neg_rho = -masking_challenge;

    let mut folded_m_polys = Vec::with_capacity(num_polynomials * half_size);
    let mut g_hats = Vec::with_capacity(num_polynomials * num_witness_variables * half_size);
    for bp in blinding_polynomials {
        for j in 0..half_size {
            folded_m_polys.push(fr_to_gpu(
                one_plus_rho * bp.m_poly[2 * j] + neg_rho * bp.m_poly[2 * j + 1],
            ));
        }
        for g_hat in &bp.g_hats {
            g_hats.extend(g_hat.iter().copied().map(fr_to_gpu));
        }
    }

    let gammas: Vec<_> = h_gammas.iter().copied().map(fr_to_gpu).collect();
    let params = EvalParams {
        half_size: half_size as u64,
        num_polynomials: num_polynomials as u64,
        num_witness_variables: num_witness_variables as u64,
        num_gammas: num_gammas as u64,
        num_blinding_variables: num_blinding_variables as u32,
        _padding: 0,
        one_plus_rho: fr_to_gpu(one_plus_rho),
        tau2: fr_to_gpu(tau2),
    };

    let mut workspace = engine.checkout_workspace()?;
    let total_partials = num_gammas * num_polynomials * (num_witness_variables + 1);
    workspace.ensure_partials(total_partials)?;
    workspace.ensure_beq_half(half_size)?;
    let stream = Arc::clone(workspace.stream());

    let folded_m_dev = stream.clone_htod(&folded_m_polys).map_err(driver_err)?;
    let g_hats_dev = stream.clone_htod(&g_hats).map_err(driver_err)?;
    let gammas_dev = stream.clone_htod(&gammas).map_err(driver_err)?;

    {
        let partials = workspace.partials();
        let mut launch = stream.launch_builder(&engine.dot_kernel);
        launch.arg(&folded_m_dev);
        launch.arg(&g_hats_dev);
        launch.arg(&gammas_dev);
        launch.arg(&mut *partials);
        launch.arg(&params);
        unsafe { launch.launch(grid_dot(num_polynomials, num_gammas, num_witness_variables + 1)) }
            .map_err(driver_err)?;
    }

    {
        let beq_half = workspace.beq_half();
        let mut launch = stream.launch_builder(&engine.beq_kernel);
        launch.arg(&gammas_dev);
        launch.arg(&mut *beq_half);
        launch.arg(&params);
        unsafe { launch.launch(grid_1d(half_size)) }.map_err(driver_err)?;
    }

    stream.synchronize().map_err(driver_err)?;

    let mut partials_host = vec![GpuField::default(); total_partials];
    let mut beq_half_host = vec![GpuField::default(); half_size];
    let partials = workspace.partials();
    stream
        .memcpy_dtoh(&*partials, &mut partials_host)
        .map_err(driver_err)?;
    let beq_half = workspace.beq_half();
    stream
        .memcpy_dtoh(&*beq_half, &mut beq_half_host)
        .map_err(driver_err)?;
    stream.synchronize().map_err(driver_err)?;

    let mut eval_results = vec![Fr::ZERO; num_gammas * stride_per_gamma];
    for (gi, &gamma) in h_gammas.iter().enumerate() {
        for poly_idx in 0..num_polynomials {
            let partial_base = (gi * num_polynomials + poly_idx) * (num_witness_variables + 1);
            let off = gi * stride_per_gamma + poly_idx * stride_per_poly;
            let m_eval = gpu_to_fr(partials_host[partial_base]);
            eval_results[off] = m_eval;
            let mut h = m_eval;
            let mut bp_pow = blinding_challenge;
            let mut gp = gamma;
            for j in 0..num_witness_variables {
                let g_eval = gpu_to_fr(partials_host[partial_base + 1 + j]);
                eval_results[off + 1 + j] = g_eval;
                h += bp_pow * gp * g_eval;
                bp_pow *= blinding_challenge;
                gp = gp.square();
            }
            eval_results[off + 1 + num_witness_variables] = h;
        }
    }

    let mut beq_weight_accum = vec![Fr::ZERO; half_size * 2];
    for (j, value) in beq_half_host.into_iter().enumerate() {
        let half = gpu_to_fr(value);
        beq_weight_accum[2 * j] = one_plus_rho * half;
        beq_weight_accum[2 * j + 1] = neg_rho * half;
    }

    Ok((eval_results, beq_weight_accum))
}

fn engine() -> Result<&'static Arc<CudaGammaEngine>, String> {
    match ENGINE.get_or_init(|| CudaGammaEngine::new().map(Arc::new)) {
        Ok(engine) => Ok(engine),
        Err(err) => Err(err.clone()),
    }
}

impl CudaGammaEngine {
    fn new() -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(driver_err)?;
        let (major, minor) = ctx.compute_capability().map_err(driver_err)?;
        let ptx = compile_cached_ptx(
            "whir-gamma",
            CUDA_GAMMA_SOURCE,
            arch_for_compute_capability(major, minor),
        )?;
        let module = ctx.load_module(ptx).map_err(driver_err)?;
        Ok(Self {
            ctx,
            dot_kernel: module.load_function("dot_gamma_poly").map_err(driver_err)?,
            beq_kernel: module
                .load_function("accumulate_beq_half")
                .map_err(driver_err)?,
            workspaces: Mutex::new(Vec::new()),
        })
    }

    fn checkout_workspace(&self) -> Result<WorkspaceLease<'_>, String> {
        let workspace = self.workspaces.lock().unwrap().pop();
        let workspace = match workspace {
            Some(workspace) => workspace,
            None => Workspace::new(Arc::clone(&self.ctx))?,
        };
        Ok(WorkspaceLease {
            engine: self,
            workspace: Some(workspace),
        })
    }
}

impl Workspace {
    fn new(ctx: Arc<CudaContext>) -> Result<Self, String> {
        Ok(Self {
            stream: ctx.new_stream().map_err(driver_err)?,
            partials: None,
            beq_half: None,
        })
    }

    fn ensure_partials(&mut self, len: usize) -> Result<(), String> {
        ensure_buffer(&self.stream, &mut self.partials, len).map(|_| ())
    }

    fn ensure_beq_half(&mut self, len: usize) -> Result<(), String> {
        ensure_buffer(&self.stream, &mut self.beq_half, len).map(|_| ())
    }
}

impl Drop for WorkspaceLease<'_> {
    fn drop(&mut self) {
        if let Some(workspace) = self.workspace.take() {
            self.engine.workspaces.lock().unwrap().push(workspace);
        }
    }
}

impl WorkspaceLease<'_> {
    fn ensure_partials(&mut self, len: usize) -> Result<(), String> {
        self.workspace.as_mut().unwrap().ensure_partials(len)
    }

    fn ensure_beq_half(&mut self, len: usize) -> Result<(), String> {
        self.workspace.as_mut().unwrap().ensure_beq_half(len)
    }

    fn partials(&mut self) -> &mut CudaSlice<GpuField> {
        self.workspace.as_mut().unwrap().partials.as_mut().unwrap()
    }

    fn beq_half(&mut self) -> &mut CudaSlice<GpuField> {
        self.workspace.as_mut().unwrap().beq_half.as_mut().unwrap()
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.workspace.as_ref().unwrap().stream
    }
}

fn ensure_buffer<'a, T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    slot: &'a mut Option<CudaSlice<T>>,
    len: usize,
) -> Result<&'a mut CudaSlice<T>, String> {
    let needs_alloc = slot.as_ref().is_none_or(|buffer| buffer.len() < len);
    if needs_alloc {
        *slot = Some(stream.alloc_zeros::<T>(len).map_err(driver_err)?);
    }
    Ok(slot.as_mut().unwrap())
}

fn grid_1d(work_items: usize) -> LaunchConfig {
    let grid = (work_items as u32).div_ceil(BLOCK_DIM);
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn grid_dot(num_polynomials: usize, num_gammas: usize, num_items: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (num_polynomials as u32, num_gammas as u32, num_items as u32),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: (BLOCK_DIM as usize * size_of::<GpuField>()) as u32,
    }
}

fn arch_for_compute_capability(major: i32, minor: i32) -> Option<&'static str> {
    match (major, minor) {
        (5, 0) => Some("compute_50"),
        (5, 2) => Some("compute_52"),
        (6, 0) => Some("compute_60"),
        (6, 1) => Some("compute_61"),
        (7, 0) => Some("compute_70"),
        (7, 5) => Some("compute_75"),
        (8, 0) => Some("compute_80"),
        (8, 6) => Some("compute_86"),
        (8, 9) => Some("compute_89"),
        (9, 0) => Some("compute_90"),
        _ => None,
    }
}

fn compile_cached_ptx(
    tag: &str,
    source: &str,
    arch: Option<&'static str>,
) -> Result<cudarc::nvrtc::Ptx, String> {
    let cache_path = ptx_cache_path(tag, source, arch);
    if let Some(path) = cache_path.as_ref() {
        if let Ok(ptx) = fs::read_to_string(path) {
            return Ok(cudarc::nvrtc::Ptx::from_src(ptx));
        }
    }
    let ptx = compile_ptx_with_opts(
        source,
        CompileOptions {
            arch,
            ..Default::default()
        },
    )
    .map_err(|err| err.to_string())?;
    if let Some(path) = cache_path {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(path, ptx.to_src());
    }
    Ok(ptx)
}

fn ptx_cache_path(tag: &str, source: &str, arch: Option<&str>) -> Option<PathBuf> {
    let cache_root = env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    arch.unwrap_or("generic").hash(&mut hasher);
    source.hash(&mut hasher);
    Some(
        cache_root
            .join("provekit")
            .join("cuda")
            .join(format!("{tag}-{}-{:016x}.ptx", arch.unwrap_or("generic"), hasher.finish())),
    )
}

fn driver_err(err: DriverError) -> String {
    err.to_string()
}

fn trace_event(args: std::fmt::Arguments<'_>) {
    if env::var_os("WHIR_CUDA_GAMMA_TRACE").is_some() {
        eprintln!("[whir-cuda-gamma] {args}");
    }
}

fn fr_to_gpu(value: Fr) -> GpuField {
    GpuField { limbs: value.0 .0 }
}

fn gpu_to_fr(value: GpuField) -> Fr {
    Fp::<MontBackend<FrConfig, 4>, 4>(BigInt(value.limbs), PhantomData)
}

unsafe fn cast_slice<T, U>(slice: &[T]) -> &[U] {
    debug_assert_eq!(size_of::<T>(), size_of::<U>());
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
    std::slice::from_raw_parts(slice.as_ptr().cast::<U>(), slice.len())
}

unsafe fn cast_scalar<T, U>(value: T) -> U {
    debug_assert_eq!(size_of::<T>(), size_of::<U>());
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
    let value = ManuallyDrop::new(value);
    std::ptr::read((&*value as *const T).cast::<U>())
}

unsafe fn cast_vec<T, U>(vec: Vec<T>) -> Vec<U> {
    debug_assert_eq!(size_of::<T>(), size_of::<U>());
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
    let mut vec = ManuallyDrop::new(vec);
    Vec::from_raw_parts(vec.as_mut_ptr().cast::<U>(), vec.len(), vec.capacity())
}

const CUDA_GAMMA_SOURCE: &str = r#"
typedef unsigned int uint;
typedef unsigned long long ulong;

struct Fe {
    ulong limbs[4];
};

struct EvalParams {
    ulong half_size;
    ulong num_polynomials;
    ulong num_witness_variables;
    ulong num_gammas;
    uint num_blinding_variables;
    uint _padding;
    Fe one_plus_rho;
    Fe tau2;
};

__device__ __constant__ ulong MODULUS[4] = {
    0x43e1f593f0000001ull,
    0x2833e84879b97091ull,
    0xb85045b68181585dull,
    0x30644e72e131a029ull
};

__device__ __constant__ ulong N0_INV = 0xc2e1f593efffffffull;
__device__ __constant__ Fe FE_ONE = {{1ull, 0ull, 0ull, 0ull}};
__device__ __constant__ Fe FE_MONT_ONE = {{
    0xac96341c4ffffffbull,
    0x36fc76959f60cd29ull,
    0x666ea36f7879462eull,
    0x0e0a77c19a07df2full
}};

__device__ __forceinline__ Fe zero_fe() {
    Fe out = {{0ull, 0ull, 0ull, 0ull}};
    return out;
}

__device__ __forceinline__ bool geq_mod(Fe a) {
    for (int i = 3; i >= 0; --i) {
        if (a.limbs[i] > MODULUS[i]) {
            return true;
        }
        if (a.limbs[i] < MODULUS[i]) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ Fe sub_modulus(Fe a) {
    Fe out;
    ulong borrow = 0ull;
    for (uint i = 0; i < 4; ++i) {
        ulong tmp = a.limbs[i] - MODULUS[i] - borrow;
        borrow = (a.limbs[i] < MODULUS[i] + borrow) ? 1ull : 0ull;
        out.limbs[i] = tmp;
    }
    return out;
}

__device__ __forceinline__ Fe add_mod(Fe a, Fe b) {
    Fe out;
    ulong carry = 0ull;
    for (uint i = 0; i < 4; ++i) {
        ulong sum = a.limbs[i] + b.limbs[i];
        ulong c1 = sum < a.limbs[i] ? 1ull : 0ull;
        ulong sum2 = sum + carry;
        ulong c2 = sum2 < sum ? 1ull : 0ull;
        out.limbs[i] = sum2;
        carry = c1 + c2;
    }
    if (carry != 0ull || geq_mod(out)) {
        out = sub_modulus(out);
    }
    return out;
}

__device__ __forceinline__ Fe sub_mod(Fe a, Fe b) {
    Fe out;
    ulong borrow = 0ull;
    for (uint i = 0; i < 4; ++i) {
        ulong tmp = a.limbs[i] - b.limbs[i] - borrow;
        ulong next_borrow = (a.limbs[i] < b.limbs[i] + borrow) ? 1ull : 0ull;
        out.limbs[i] = tmp;
        borrow = next_borrow;
    }
    if (borrow != 0ull) {
        ulong carry = 0ull;
        for (uint i = 0; i < 4; ++i) {
            ulong sum = out.limbs[i] + MODULUS[i];
            ulong c1 = sum < out.limbs[i] ? 1ull : 0ull;
            ulong sum2 = sum + carry;
            ulong c2 = sum2 < sum ? 1ull : 0ull;
            out.limbs[i] = sum2;
            carry = c1 + c2;
        }
    }
    return out;
}

__device__ __forceinline__ Fe mont_mul(Fe a, Fe b) {
    ulong t[5] = {0ull, 0ull, 0ull, 0ull, 0ull};

    for (uint i = 0; i < 4; ++i) {
        ulong carry = 0ull;
        for (uint j = 0; j < 4; ++j) {
            ulong lo = a.limbs[j] * b.limbs[i];
            ulong hi = __umul64hi(a.limbs[j], b.limbs[i]);

            ulong sum = t[j] + lo;
            hi += (sum < t[j]) ? 1ull : 0ull;

            ulong sum2 = sum + carry;
            hi += (sum2 < sum) ? 1ull : 0ull;

            t[j] = sum2;
            carry = hi;
        }
        t[4] = carry;

        ulong m = t[0] * N0_INV;
        carry = 0ull;

        {
            ulong lo = m * MODULUS[0];
            ulong hi = __umul64hi(m, MODULUS[0]);
            ulong sum = t[0] + lo;
            hi += (sum < t[0]) ? 1ull : 0ull;
            ulong sum2 = sum + carry;
            hi += (sum2 < sum) ? 1ull : 0ull;
            carry = hi;
        }

        for (uint j = 1; j < 4; ++j) {
            ulong lo = m * MODULUS[j];
            ulong hi = __umul64hi(m, MODULUS[j]);
            ulong sum = t[j] + lo;
            hi += (sum < t[j]) ? 1ull : 0ull;
            ulong sum2 = sum + carry;
            hi += (sum2 < sum) ? 1ull : 0ull;
            t[j - 1] = sum2;
            carry = hi;
        }

        ulong sum = t[4] + carry;
        ulong c = (sum < t[4]) ? 1ull : 0ull;
        t[3] = sum;
        t[4] = c;
    }

    Fe out;
    out.limbs[0] = t[0];
    out.limbs[1] = t[1];
    out.limbs[2] = t[2];
    out.limbs[3] = t[3];
    if (t[4] != 0ull || geq_mod(out)) {
        out = sub_modulus(out);
    }
    return out;
}

__device__ __forceinline__ Fe eq_weight(Fe gamma, ulong index, uint num_blinding_variables) {
    Fe weight = FE_MONT_ONE;
    Fe gamma_power = gamma;
    for (uint i = 0; i < num_blinding_variables; ++i) {
        uint bit = num_blinding_variables - 1u - i;
        weight = mont_mul(
            weight,
            (index & (1ull << bit)) ? gamma_power : sub_mod(FE_MONT_ONE, gamma_power)
        );
        gamma_power = mont_mul(gamma_power, gamma_power);
    }
    return weight;
}

extern "C" __global__ void dot_gamma_poly(
    const Fe* folded_m_polys,
    const Fe* g_hats,
    const Fe* gammas,
    Fe* partials,
    EvalParams params
) {
    ulong poly_idx = (ulong)blockIdx.x;
    ulong gamma_idx = (ulong)blockIdx.y;
    ulong item_idx = (ulong)blockIdx.z;
    uint tid = threadIdx.x;
    if (poly_idx >= params.num_polynomials ||
        gamma_idx >= params.num_gammas ||
        item_idx > params.num_witness_variables) {
        return;
    }

    Fe gamma = gammas[gamma_idx];
    Fe acc = zero_fe();
    for (ulong idx = tid; idx < params.half_size; idx += (ulong)blockDim.x) {
        Fe eq = eq_weight(gamma, idx, params.num_blinding_variables);
        Fe coeff = item_idx == 0
            ? folded_m_polys[poly_idx * params.half_size + idx]
            : g_hats[((poly_idx * params.num_witness_variables + (item_idx - 1)) * params.half_size) + idx];
        acc = add_mod(acc, mont_mul(eq, coeff));
    }

    extern __shared__ Fe scratch[];
    scratch[tid] = acc;
    __syncthreads();

    for (uint offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] = add_mod(scratch[tid], scratch[tid + offset]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        Fe out = scratch[0];
        if (item_idx != 0) {
            out = mont_mul(params.one_plus_rho, out);
        }
        partials[((gamma_idx * params.num_polynomials + poly_idx) * (params.num_witness_variables + 1)) + item_idx] = out;
    }
}

extern "C" __global__ void accumulate_beq_half(
    const Fe* gammas,
    Fe* beq_half,
    EvalParams params
) {
    ulong idx = (ulong)blockIdx.x * (ulong)blockDim.x + (ulong)threadIdx.x;
    if (idx >= params.half_size) {
        return;
    }

    Fe acc = zero_fe();
    Fe tau2_pow = FE_MONT_ONE;
    for (ulong gamma_idx = 0; gamma_idx < params.num_gammas; ++gamma_idx) {
        Fe eq = eq_weight(gammas[gamma_idx], idx, params.num_blinding_variables);
        acc = add_mod(acc, mont_mul(tau2_pow, eq));
        tau2_pow = mont_mul(tau2_pow, params.tau2);
    }
    beq_half[idx] = acc;
}
"#;

#[cfg(test)]
mod tests {
    use super::evaluate_gamma_block_bn254;
    use crate::protocols::whir_zk::utils::BlindingPolynomials;
    use ark_bn254::Fr;
    use ark_ff::{AdditiveGroup, Field};
    use ark_std::{
        UniformRand,
        rand::{SeedableRng, rngs::StdRng},
    };

    fn eval_cpu(
        blinding_polynomials: &[BlindingPolynomials<Fr>],
        h_gammas: &[Fr],
        masking_challenge: Fr,
        blinding_challenge: Fr,
        tau2: Fr,
        num_blinding_variables: usize,
        num_witness_variables: usize,
    ) -> (Vec<Fr>, Vec<Fr>) {
        let num_polynomials = blinding_polynomials.len();
        let half_size = 1usize << num_blinding_variables;
        let weight_size = 1usize << (num_blinding_variables + 1);
        let one_plus_rho = Fr::ONE + masking_challenge;
        let neg_rho = -masking_challenge;
        let folded_m_polys: Vec<Vec<Fr>> = blinding_polynomials
            .iter()
            .map(|bp| {
                (0..half_size)
                    .map(|j| one_plus_rho * bp.m_poly[2 * j] + neg_rho * bp.m_poly[2 * j + 1])
                    .collect()
            })
            .collect();
        let mut tau2_powers = Vec::with_capacity(h_gammas.len());
        let mut p = Fr::ONE;
        for _ in h_gammas {
            tau2_powers.push(p);
            p *= tau2;
        }
        let stride_per_poly = num_witness_variables + 2;
        let stride_per_gamma = num_polynomials * stride_per_poly;
        let mut eval_results = vec![Fr::ZERO; h_gammas.len() * stride_per_gamma];
        let mut beq_half = vec![Fr::ZERO; half_size];

        for (gi, &gamma) in h_gammas.iter().enumerate() {
            let mut eq = vec![Fr::ZERO; half_size];
            eq[0] = Fr::ONE;
            let mut gamma_power = gamma;
            for i in 0..num_blinding_variables {
                for j in (0..(1usize << i)).rev() {
                    eq[2 * j + 1] = eq[j] * gamma_power;
                    eq[2 * j] = eq[j] - eq[2 * j + 1];
                }
                gamma_power = gamma_power.square();
            }
            for (acc, value) in beq_half.iter_mut().zip(eq.iter()) {
                *acc += tau2_powers[gi] * *value;
            }
            for (poly_idx, bp) in blinding_polynomials.iter().enumerate() {
                let off = gi * stride_per_gamma + poly_idx * stride_per_poly;
                let m_eval: Fr = eq
                    .iter()
                    .zip(folded_m_polys[poly_idx].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                eval_results[off] = m_eval;
                let mut h = m_eval;
                let mut beta_pow = blinding_challenge;
                let mut gp = gamma;
                for (j, g_hat) in bp.g_hats.iter().enumerate() {
                    let g_eval: Fr = eq.iter().zip(g_hat.iter()).map(|(&a, &b)| a * b).sum();
                    let g_eval = one_plus_rho * g_eval;
                    eval_results[off + 1 + j] = g_eval;
                    h += beta_pow * gp * g_eval;
                    beta_pow *= blinding_challenge;
                    gp = gp.square();
                }
                eval_results[off + 1 + num_witness_variables] = h;
            }
        }

        let mut beq = vec![Fr::ZERO; weight_size];
        for j in 0..half_size {
            beq[2 * j] = one_plus_rho * beq_half[j];
            beq[2 * j + 1] = neg_rho * beq_half[j];
        }
        (eval_results, beq)
    }

    #[test]
    fn cuda_gamma_matches_cpu() {
        if std::env::var_os("WHIR_DISABLE_CUDA_GAMMA_BLOCK").is_some() {
            return;
        }
        let mut rng = StdRng::seed_from_u64(7);
        let num_blinding_variables = 4;
        let num_witness_variables = 3;
        let polynomials: Vec<_> = (0..3)
            .map(|_| {
                BlindingPolynomials::sample(
                    &mut rng,
                    num_blinding_variables,
                    num_witness_variables,
                )
            })
            .collect();
        let h_gammas: Vec<_> = (0..6).map(|_| Fr::rand(&mut rng)).collect();
        let masking_challenge = Fr::rand(&mut rng);
        let blinding_challenge = Fr::rand(&mut rng);
        let tau2 = Fr::rand(&mut rng);

        let Ok((gpu_eval, gpu_beq)) = evaluate_gamma_block_bn254(
            &polynomials,
            &h_gammas,
            masking_challenge,
            blinding_challenge,
            tau2,
            num_blinding_variables,
            num_witness_variables,
        ) else {
            return;
        };
        let (cpu_eval, cpu_beq) = eval_cpu(
            &polynomials,
            &h_gammas,
            masking_challenge,
            blinding_challenge,
            tau2,
            num_blinding_variables,
            num_witness_variables,
        );
        assert_eq!(gpu_eval, cpu_eval);
        assert_eq!(gpu_beq, cpu_beq);
    }
}
