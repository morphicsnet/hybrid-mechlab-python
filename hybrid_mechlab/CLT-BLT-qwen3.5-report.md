# Native Core Portfolio Expansion for hybrid-mechlab as a Mathematical Transport/Topology Substrate

## Executive summary

The Native Core Portfolio Expansion plan is best framed as a **mathematical de-risking strategy**: one transport/topology substrate (family-aware schedule + IR + trace/evidence schema) that (i) can *faithfully* represent multiple hybrid long-context families, (ii) supports *auditable* comparative traces across “native vs adapter vs Liger-backed” backends, and (iii) yields reproducible “evidence bundles” that can be hashed, signed, and re-checked offline. This wedge aligns with the core empirical reality of modern long-context hybrids: they explicitly interleave **recurrent/linear-memory tracts** with **periodic global bridges** (often attention), and this tract/bridge decomposition is already first-class in public model specifications such as Qwen3.5’s declared 3:1 cadence. citeturn0search1

Concretely, the recommended positioning is: **hybrid-mechlab owns the transport/topology substrate as the single source of truth**, and any legacy or parallel tooling (e.g., BLT/TDA sidecar tooling) becomes a consumer of the new IR/contracts until the migration is complete. The first “transformer-competition” narrative should emphasize **long-context reasoning** where the measurable performance drivers are (a) retention under long contexts, (b) sparse global synchronization (bridge layers), and (c) controlled bridge dependence—precisely the axes exposed by hybrid designs that reduce KV-cache pressure while preserving periodic high-fidelity recall. citeturn0search1turn4search0turn0search3

This plan is also compatible with an “assurance tier” story: by encoding **bridge semantics, regime boundaries, and transport summaries** into a reproducible trace schema, one can offer progressively stronger assurance modes—ranging from *deterministic replay with signed digests* to *offline recomputation of exact topology artifacts*—even when full end-to-end certification of a neural model remains out of scope.

## README section from the attached BLT/TDA memo

The following section integrates the attached memo content as a README block (verbatim text with minimal Markdown normalization). Direct URL strings from the memo are replaced by `[link omitted]` to comply with environments that require citations rather than raw URLs.

<details>
<summary><strong>Click to expand: Block-Level Transcoder and TDA Sidecar for Hybrid 3:1 Architectures (attached memo content)</strong></summary>

**Block-Level Transcoder and TDA Sidecar for Hybrid 3:1 Architectures**

**Executive summary**

Qwen3.5-2B is a strong first prototype target for a Block-Level Transcoder (BLT) plus a Topological Data Analysis (TDA) sidecar because its model card exposes an explicit hybrid “3:1” rhythm—three consecutive (Gated DeltaNet → FFN) pairs followed by one (Gated Attention → FFN)—repeated across the network, with native 262,144-token context. [1] This layout directly matches the BLT design goal: instrument and summarize repeated micro-regimes (DeltaNet tracts) plus the periodic bridge (softmax attention) at a stable cadence, without inventing a new architecture first. [2]

The key architectural difference to exploit is that Gated DeltaNet–style linear attention replaces a growing KV cache with a fixed-size recurrent state; this creates natural “tract” objects (state updates) that are well-suited to topological summaries and to intervention protocols that vary (i) geometric scale and (ii) intervention strength. [3] The periodic Gated Attention layers can be treated as “caged softmax bridges”: sparse, auditable, and explicitly penalized when they become the dominant dependency path. The gating-after-SDPA formulation is directly supported by the Gated Attention paper, which finds head-specific sigmoid gating after SDPA improves stability and mitigates attention sink behavior. [4]

The BLT specification in this report is model-agnostic by construction (it is an IR + hook contract) but is optimized for hybrid recurrent/attention families (Qwen3.5, Qwen3-Next, OLMo Hybrid, Kimi Linear, MiniMax-Text-01). [5] The near-term recommendation is:
- Qwen3.5-2B: end-to-end BLT + TDA sidecar prototype and instrumentation hardening. [1]
- Qwen3.5-9B: first “publishable” mechanistic study (stronger circuit depth while retaining the same 3:1 scaffold), with long-context extension up to ~1,010,000 tokens stated on the 9B model card. [6]

Compute-wise, you can make early iterations cheap by storing sparse BLT/CLT-compatible codes and online topological sketches rather than full activation tapes; the CLT model card shows a concrete precedent: 20,480 features per layer with ~115 active features (L0) and sharded weights designed for tractable downstream tooling. [7]

**Architectural background**

**What changes relative to a standard Transformer**

A standard Transformer’s dominant long-context inference cost is tied to softmax attention and its growing KV cache (keys/values stored for all prior tokens). The hybrid families under discussion keep some full attention for recall/bridging but replace most token mixing with recurrent/linear mechanisms to reduce quadratic scaling. [8]

For Qwen3.5-2B specifically, the model card describes:
- 24 total layers and hidden dimension 2048. [1]
- A hidden layout of 6 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)), i.e., a repeated 3:1 cadence. [1]
- Native context length 262,144 tokens. [1]

For Qwen3.5-9B, the card similarly specifies:
- 32 layers and hidden dimension 4096. [9]
- 8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)). [9]
- Native 262,144 tokens and extension up to ~1,010,000 tokens. [6]

**Gated DeltaNet as tractable recurrent state transport**

The central property that makes Gated DeltaNet attractive for BLT+TDA is the fixed-size recurrent state semantics: the FPGA-focused GDN paper states directly that Gated DeltaNet replaces the growing KV cache with a fixed-size recurrent state, and that hybrid models use a 3:1 ratio where this dominates decode-time computation. [3] This suggests a natural decomposition into:
- Tracts: recurrent state update operators (per layer/head) that can be summarized as structured “transport.” [3]
- Bridges: periodic softmax attention layers that recover precise token-level recall beyond what a bounded recurrent state can retain. This “prevent information from getting stuck in a bounded recurrent state” motivation is stated explicitly in the OLMo Hybrid writeup for the same 3:1 pattern. [10]

**Gated Attention as a “caged softmax bridge”**

The Gated Attention paper (from the Qwen team) studies gating-augmented softmax attention and identifies a simple variant—head-specific sigmoid gating after SDPA output—as a consistent improvement with stability and attention-sink mitigation benefits. [4] This is the right structural interpretation for BLT: treat the periodic full-attention layers as auditable, gate-modulated bridges, and apply explicit penalties and topological diagnostics to keep them “minimal but sufficient” rather than ubiquitous. [4]

**BLT specification**

**BLT goals and non-goals**

BLT is an intermediate “replacement-model-capable” instrumentation layer that makes hybrid models legible at the block scale:
- Convert opaque tensor flows into sparse, inspectable feature codes and tract-level transport summaries.
- Preserve enough semantics to support circuit tracing, interventions, and reproducible audit trails—i.e., the same role cross-layer transcoders play in attribution-graph methods, but generalized to hybrid blocks. [11]

BLT is not a new base model architecture; it is a model-agnostic interface (hooks + IR) plus optional learned transcoders that can be trained for any supported backbone.

**Hook contract**

BLT attaches to any model that can expose a repeated “3× recurrent + 1× attention” rhythm, but it does not require that rhythm. For Qwen3.5-like hybrids, the minimum per-block hook set requested is:
· pre-D1: input residual stream to the first Gated DeltaNet sublayer in the local rhythm.
· post-D1
· post-D2
· post-D3
· post-attention: after the periodic full-attention sublayer in the rhythm.
· block output: after the final FFN/residual integration for the block.

For Qwen3.5-2B, “D1/D2/D3” correspond to the three (Gated DeltaNet → FFN) pairs specified in the hidden layout. [1]

**BLT IR design**

A compiler-style IR is the key to staying model-agnostic while still supporting “verification-grade” mechanistic tooling. The BLT IR should be multi-level (high-level semantic ops → lowered numeric ops), borrowing design principles from multi-level compiler IR systems such as MLIR (extensible dialects, progressive lowering) and StableHLO-style portability layers. [12]

Recommended BLT IR layers:
1. Semantic IR (S-IR): token-mixing primitives with explicit state semantics.
   1. RecurrentTransportUpdate(state, x -> state', y) for DeltaNet/HGRN2/RetNet/Hawk-like layers.
   2. SoftmaxBridge(q, k, v -> y) for full attention (optionally gated).
   3. FFN(x -> y) and ResidualAdd, Norm, Gate.
2. Graph IR (G-IR): acyclic dataflow graph for a prompt (prefill) or stepwise stream graph (decode), compatible with capture/transformation tools in the spirit of torch.fx / export IR. [13]
3. Numeric IR (N-IR): optional lowered forms for equivalence testing (e.g., blockwise linearizations, Jacobian-vector products, or fused kernels), aligned with the “progressive lowering” motivation in MLIR and deep-learning compiler stacks. [14]

**BLT external API**

A minimal but sufficient API surface for the prototype:
· Attach & configure
· blt = BLT.attach(model, hook_spec, feature_spec, storage_spec, mode={trace|serve|audit})
· Run & record
· trace = blt.run(prompts, capture={codes, sketches, minimal_acts}, interventions=None)
· Intervene
· trace2 = blt.run(prompts, interventions=[Intervention(target="D2.post", feature_id=..., strength=...)])
· Export
· trace.to_parquet(...) / trace.to_zarr(...)
· trace.to_attribution_graph(...) (Circuit-Tracer compatible form)

This “replacement model” framing is aligned with the circuit tracing methodology: replace part of the model with a more interpretable component trained to approximate it, then trace computation to build attribution graphs. [11]

**Data formats for tract sketches and CLT compatibility**

A pragmatic requirement is direct compatibility with existing CLT/circuit-tracing ecosystems. The published CLT model card provides a concrete reference format:
- Encoder maps each layer input  to sparse features ; decoder reconstructs MLP outputs using features from prior layers. [15]
- Weights are sharded by layer with clear tensor names and shapes, including decoder tensors shaped like (d_latent, n_out_layers, d_model). [15]

BLT should therefore emit two “first-class” objects:
1. Sparse feature codes (CLT-like):
   1. codes[hook_point] as a ragged list of (feature_id, value) per token, stored as:
      · uint32 feature_id, float16 value, plus optional sign and provenance fields.
2. Tract sketches (TDA-ready):
   1. Per hook point and per token window, store:
      · a small adjacency (kNN) or coactivation graph among active features,
      · plus incremental topological summaries (Section “TDA sidecar”).

This enables “cheap iteration” by minimizing full activation storage, while keeping exact-recompute pathways for selected segments.

```mermaid
flowchart LR
  A[Token stream + prompt] --> B[Hybrid model forward]
  subgraph Block i (3:1 rhythm)
    B --> H0[pre-D1 hook]
    H0 --> D1[Gated DeltaNet 1]
    D1 --> H1[post-D1 hook]
    H1 --> D2[Gated DeltaNet 2]
    D2 --> H2[post-D2 hook]
    H2 --> D3[Gated DeltaNet 3]
    D3 --> H3[post-D3 hook]
    H3 --> GA[Gated Attention bridge]
    GA --> H4[post-attention hook]
    H4 --> OUT[block output hook]
  end
  H0 --> E[BLT encoder]
  H1 --> E
  H2 --> E
  H3 --> E
  H4 --> E
  OUT --> E
  E --> IR[BLT IR: S-IR/G-IR/N-IR]
  IR --> CODES[Sparse codes (CLT-compatible)]
  IR --> SKETCH[Online TDA sketches]
  CODES --> TRACER[Attribution graph exporter]
  SKETCH --> OFFLINE[Offline exact TDA pipelines]
```

**TDA sidecar design**

Online sketches

The online sidecar is meant to run during normal forward passes and emit stable, low-footprint signals. The design should reuse known stable vectorizations of persistent homology such as persistence landscapes and persistence images when persistence diagrams are available, because these representations are designed for statistical use and stability. [16]

To align with the “signed / cancellation-aware” goal, online sketches should decompose transport into three complexes per hook window:
· Positive complex: edges/features whose contribution increases a chosen target statistic (e.g., logit margin, block output energy, or a feature-group activation).
· Negative complex: edges/features that decrease it.
· Cancellation complex: paired positive/negative contributions that largely cancel under composition, highlighting brittle “knife-edge” dependence.

These are sketches because full signed persistence is expensive; the core idea is to maintain separate filtrations for each sign channel and track coarse invariants (Betti-0/1 estimates, connected-component counts, cycle indicators) incrementally.

Offline exact pipelines

For audits, publications, and certification-grade workflows, BLT must support offline recomputation. The baseline algorithms to anchor are:
· Persistent homology on simplicial filtrations, using standard references for computing persistence. [17]
· Mapper to build multiresolution graph summaries of feature spaces. This is particularly aligned with “concept evolution” maps and “supernode” discovery. [18]
· Multiparameter persistence / bifiltration methods for interventions that vary jointly with scale and strength; the foundational limitation is that multidimensional persistence does not admit a complete discrete invariant like the 1D barcode, motivating the use of rank invariants / surrogate invariants. [19]

A concrete, implementable offline pipeline for the prototype:
1. Choose a windowed segment (tokens × blocks × hooks) from BLT traces.
2. Build a point cloud or graph:
   - points: (feature embedding vectors, coactivation vectors, or decoder vectors),
   - edges: kNN under cosine distance, or thresholded coactivation.
3. Compute persistence diagrams over a Vietoris–Rips filtration (or graph filtration), then render to vector signatures (landscapes or images) for comparison across interventions. [20]
4. Run Mapper on the same segment to obtain interpretable multiresolution graphs, then align those graphs across blocks and interventions. [18]

Intervention bifiltration protocol and metrics

The intervention protocol should explicitly support a bifiltration in:
- geometric scale : filtration parameter controlling neighborhood size / similarity threshold,
- intervention strength : magnitude of feature- or tract-level perturbation.

This is structurally consistent with multiparameter persistence framing. [21]

Intervention targets in BLT should include:
- feature activation clamping / scaling at any hook point,
- tract-state perturbations (DeltaNet recurrent state),
- bridge gating interventions on softmax attention outputs (“caged softmax” controls).

Two prototype metrics that map directly onto productizable “mechanistic assurance” claims:
1. Topological susceptibility
2. Bridge dependence

(Additional memo content continues in the attached document; this integrated block preserves the memo’s structure and notation, with direct links elided.)

</details>

**Seed memo reference map (selected)**  
The seed memo’s technical anchors correspond to widely available primary sources: Qwen3.5 hybrid layout and context length. citeturn0search1turn0search4 Gated DeltaNet as fixed-state linear attention and hybrid behavior. citeturn1search1turn5search2 Gated Attention’s post-SDPA sigmoid gating and attention-sink mitigation. citeturn5search1 OLMo Hybrid’s explicit delta/attention interleaving and mixed cache (KV + recurrent state). citeturn11search1 MLIR and StableHLO as exemplars of progressive lowering and portability-layer IR design. citeturn11search2turn12search0 TDA primitives: Mapper (Singh–Mémoli–Carlsson) and stable vectorizations of persistence diagrams (persistence images / landscapes). citeturn8search1turn8search10turn9search0

## Formal substrate and public interfaces

This section presents a math-heavy, verification-aligned interface layer for **family-aware transport**. The key modeling move is to treat a hybrid model not as “layers,” but as a **typed sequence of transport regimes** plus **bridge events** with explicit cadence and conformance evidence.

### Core objects as formal definitions

Let a *prompt execution* be indexed by token time \(t \in \{0,\dots,T-1\}\) and *structural position* \(p\) (block, layer, hookpoint). Define a directed acyclic “execution scaffold” graph \(G = (V,E)\) where each vertex \(v \in V\) is a hookpoint state, and each edge \(e \in E\) is a semantics-preserving transition (transport) between hookpoints.

We define a **family-aware schedule** as a labeled path decomposition of \(G\) into regimes that share semantics and invariants.

**Definition (TransportFamilyKind).**  
A finite tag set describing the *intended kernel semantics*, not vendor-specific parameterizations:
\[
\mathsf{TransportFamilyKind} \in \{
\textsf{GatedDeltaNet},\ \textsf{HGRN2},\ \textsf{RetNet},\ \textsf{Hawk},\ \textsf{TransNormerLLM},\ \textsf{Unknown}
\}.
\]
The tags correspond to families with distinct “memory law” primitives: gated delta-rule recurrences. citeturn5search2 Outer-product state expansion gated RNNs (HGRN2). citeturn10search4 Retention mechanisms with parallel/recurrent/chunkwise paradigms (RetNet). citeturn0search3 Gated linear recurrence with O(1) step cost (Hawk/Griffin). citeturn6search0 Linear-attention LLMs with normalization/positional schemes (TransNormerLLM). citeturn10search2

**Definition (TransportRegimeKind).**  
A regime is the semantics class of an edge-set in the execution scaffold:
\[
\mathsf{TransportRegimeKind} \in \{
\textsf{RecurrentTransport},\ \textsf{LinearAttentionTransport},\ \textsf{RetentionTransport},\ \textsf{SoftmaxBridge},\ \textsf{LocalAttentionBridge},\ \textsf{FFN},\ \textsf{Norm},\ \textsf{Residual},\ \textsf{Gate},\ \textsf{Other}
\}.
\]
This separates *transport* (stateful token mixing) from *integration* primitives (FFN/Norm/Residual).

**Definition (BridgeSpec).**  
A bridge is a designated “global synchronization” event with audit-relevant parameters. A bridge spec is:
\[
\mathsf{BridgeSpec} = (\mathsf{bridge\_kind},\ \mathsf{cadence},\ \mathsf{gate\_law},\ \mathsf{scope},\ \mathsf{constraints}),
\]
where
- \(\mathsf{bridge\_kind} \in \{\textsf{SoftmaxBridge},\textsf{LocalAttentionBridge}\}\),
- \(\mathsf{cadence}\) is a schedule constraint (e.g. “every 4th tract” as in 3:1 hybrids),
- \(\mathsf{gate\_law}\) describes whether the bridge output is gated (e.g., sigmoid gating after SDPA). citeturn5search1
- \(\mathsf{scope}\) defines which tokens/states are synchronizable (full, windowed, sparse),
- \(\mathsf{constraints}\) are measurable invariants (bridge-mask invariants, gate range, normalization behavior).

**Definition (FamilyDescriptor).**  
A family descriptor is the *portable* summary sufficient to (i) select a kernel, (ii) interpret traces, (iii) validate conformance:
\[
\mathsf{FamilyDescriptor} = (\mathsf{family},\ \mathsf{regimes},\ \mathsf{bridge},\ \mathsf{state\_shape},\ \mathsf{head\_schema},\ \mathsf{normalization\_law},\ \mathsf{position\_law},\ \mathsf{opaque\_fields}),
\]
with explicit “opaque fields” for vendor-specific, under-specified, or undisclosed internals.

**Definition (KernelConformanceReport).**  
A conformance report is an evidence object:
\[
\mathsf{KernelConformanceReport} = (\mathsf{descriptor},\ \mathsf{tests\_run},\ \mathsf{hashes},\ \mathsf{metrics},\ \mathsf{violations},\ \mathsf{build\_provenance},\ \mathsf{signatures}),
\]
where \(\mathsf{hashes}\) include a transport digest and trace-schema digest, and \(\mathsf{signatures}\) allow “evidence bundle” signing for controlled artifact promotion. (Certification is a policy decision; the substrate provides the necessary structure.)

### Rust `no_std` contracts as type sketches

The plan constraint “hm_core remains `no_std`” implies: allocate-free kernels by default, stable POD-serializable structures, and explicit buffer passing. The following are contracts (type sketches) intended for `hm_core`.

```rust
#![no_std]

pub enum TransportFamilyKind {
    GatedDeltaNet,
    Hgrn2,
    RetNet,
    Hawk,
    TransNormerLLM,
    Unknown,
}

pub enum TransportRegimeKind {
    RecurrentTransport,
    LinearAttentionTransport,
    RetentionTransport,
    SoftmaxBridge,
    LocalAttentionBridge,
    FFN,
    Norm,
    Residual,
    Gate,
    Other,
}

pub struct BridgeSpec {
    pub kind: TransportRegimeKind,        // must be *Bridge kind*
    pub cadence_num: u8,                  // e.g. 3 in 3:1
    pub cadence_den: u8,                  // e.g. 1 in 3:1
    pub gated: bool,
    pub gate_law_id: u32,                 // e.g. "post_sdpa_sigmoid"
    pub scope_id: u32,                    // e.g. full/global/windowed
}

pub struct FamilyDescriptor {
    pub family: TransportFamilyKind,
    pub regimes: &'static [TransportRegimeKind], // static schedule skeleton
    pub bridge: BridgeSpec,
    pub d_model: u32,
    pub state_bytes: u32,                  // canonical recurrent state size
    pub opaque_fields_hash: [u8; 32],      // vendor/unknown details hashed
}

/// Family-aware schedule: a concrete instantiation of the skeleton.
pub struct TransportSchedule {
    pub family: FamilyDescriptor,
    pub n_layers: u16,
    pub layer_regime: &'static [TransportRegimeKind], // length = n_layers
    pub bridge_mask: &'static [bool],                 // length = n_layers
}

/// A deterministic digest of transport behavior over a run segment.
pub struct TransportDigest {
    pub algo_id: u32,
    pub digest: [u8; 32],
    pub window_tokens: u32,
    pub window_layers: u16,
}

/// Sketch = compact signed topological/feature summary.
pub struct SignedSketch {
    pub sketch_id: u32,
    pub payload_hash: [u8; 32],
    pub signature: [u8; 64], // e.g. Ed25519; exact scheme chosen at workspace layer
}

/// Conformance report: core system emits structured facts, not prose.
pub struct KernelConformanceReport {
    pub family: FamilyDescriptor,
    pub schedule_digest: [u8; 32],
    pub transport_digest: TransportDigest,
    pub violations_mask: u64, // bitmask of failed invariants
}
```

### Native vs adapter vs migration backends

The substrate needs three execution lanes:

**NativeTransportKernel.** Canonical, family-specific kernel implementation (first is Gated DeltaNet). The interface is *stepwise*, enabling streaming decode and deterministic replay.

```rust
pub trait NativeTransportKernel {
    type Params;
    type State;    // fixed-size state, preferably repr(C)
    type Input;
    type Output;

    fn family_descriptor() -> FamilyDescriptor;

    fn init_state(params: &Self::Params, out_state: &mut Self::State) -> Result<(), KernelError>;

    fn step(
        params: &Self::Params,
        state: &mut Self::State,
        x_t: &Self::Input,
        y_t: &mut Self::Output,
        hooks: &mut dyn BridgeCrossingHooks,
    ) -> Result<(), KernelError>;

    fn summarize_state(state: &Self::State, digest_out: &mut TransportDigest) -> Result<(), KernelError>;
}

pub trait BridgeCrossingHooks {
    fn on_bridge_enter(&mut self, layer_idx: u16);
    fn on_bridge_exit(&mut self, layer_idx: u16);
}
```

**FamilyAdapter.** A mapping layer that reads vendor internals and emits the same IR events/traces. This is the correct lane for Qwen3.5 first because Qwen’s cadence is explicit and easily aligned with schedule/bridge semantics. citeturn0search1

**MigrationBackend.** A “runtime compatibility” lane whose job is configuration translation + execution bridging. In this plan, Liger is treated as workspace/runtime infrastructure (not `hm_core`) because it is a set of Triton kernel patches and model wrappers rather than a `no_std` kernel. citeturn2search3turn2search6

A minimal contract:

```rust
pub trait MigrationBackend {
    fn load_and_patch(model_id: &str, cfg_json: &str) -> Result<Self, BackendError>
    where
        Self: Sized;

    fn run_trace(&mut self, prompts_json: &str) -> Result<Vec<u8>, BackendError>; // returns trace bytes
}
```

## Mathematical semantics: partial sheaves, connections, and topology over transport

The expansion plan’s “transport/topology substrate” becomes mathematically rigorous when we treat the execution scaffold \(G\) as the base space for **cellular sheaves** whose stalks contain *local state summaries* and whose restriction maps encode *compatibility under transport*. This aligns with established cellular-sheaf frameworks: sheaves assign vector spaces to cells and linear restriction maps to incidences, with global sections representing consistent assignments. citeturn7search4turn7search7

### Partial sheaf model for hybrid execution

Let the hookpoint set \(V\) be vertices, and let transport edges \(E\) be directed edges. Define a (cellular) sheaf \(\mathcal{F}\) on \(G\) by:
- stalks \(\mathcal{F}(v)\): feature-code space, transport-digest space, or low-rank summary space at hookpoint \(v\),
- stalks \(\mathcal{F}(e)\): constraint space in which “agreement” is tested across an edge,
- restriction maps \(\rho_{v\to e}\): the transport-induced projection of local summaries into the constraint space.

A *partial sheaf* perspective is operationally: we only build \(\mathcal{F}\) over a subgraph \(H \subseteq G\) (e.g., a tract window), compute defects/consistency on \(H\), and glue subgraphs by bridge events.

This is directly analogous to the “knowledge sheaf” framing where embeddings are approximate global sections with consistency constraints and Laplacian-based defect measures. citeturn7search4turn7search6

### Connection and holonomy as bridge/cadence semantics

A tract (recurrent/linear regime) naturally defines a **discrete connection**: parallel transport along successive steps updates a state \(S_t\) via a family-specific law. For Gated DeltaNet this “memory law” is explicitly framed as a fixed-size recurrent state replacing the KV cache, with gating enabling controlled forgetting and delta updates enabling targeted writes. citeturn1search1turn5search2

A bridge introduces non-local synchronization. When bridges are gated (e.g., head-specific sigmoid gating after SDPA), we can interpret them as **controlled connection resets** or **sparse holonomy corrections**—measurable events that can be penalized when they dominate. citeturn5search1

Operationally, define:
- a tract transport operator \(\mathcal{T}_{i}\) for each tract step \(i\),
- a bridge operator \(\mathcal{B}_j\) for each bridge layer \(j\),
- a composed transport over a window \(W\): \(\Phi_W = \prod \mathcal{T}_{i} \prod \mathcal{B}_j\).

Bridge-cadence “2:1–4:1” schedules correspond to constraints on the word in the free monoid generated by \(\{\mathcal{T},\mathcal{B}\}\). Qwen3.5’s 3:1 block pattern is an explicit instance of such a constrained word. citeturn0search1 OLMo Hybrid similarly formalizes interleaving of attention and Gated DeltaNet layers and even exposes a mixed cache structure (KV + recurrent state), which makes this semantics mechanically hookable. citeturn11search1

### Persistent homology stubs as topology-on-traces

Given sparse feature codes or coactivation graphs per hook window, define a filtration \(\{K_\epsilon\}\) (e.g., Vietoris–Rips or graph threshold filtration). Then compute persistence diagrams \(D_k\) for dimensions \(k=0,1\) (and optionally \(2\)). Standard persistent homology computation reduces to algorithms on filtered chain complexes; canonical references include Zomorodian–Carlsson for algorithmic foundations. citeturn9search10turn9search11

For statistical and comparison use, represent diagrams with stable vectorizations:
- persistence images (provably stable mapping from diagrams to finite-dimensional vectors), citeturn8search10
- persistence landscapes (Banach-space-valued summaries with LLN/CLT results and stability). citeturn9search0

This supports the plan’s core requirement: *topology is a shared substrate*, so each family reuses identical topology APIs, while family-specific kernels only alter how the trace graph is generated.

## Implementation architecture: Rust `no_std` core, Python SDK, and Liger integration

### Repository snapshot and “source-of-truth” posture

In the shared project folder, the visible artifacts are:
- `Block-Level Transcoder and TDA Sidecar for Hybrid 3_1 Architectures.docx` (seed memo),
- `partial persistent knowledge sheafs.pdf` (sheaf-theoretic framing),
- `PLAN.md` (current milestone plan: topology-first exact persistence & research artifacts),
- `hybrid_mechlab_flattened_README.md` (current flattened spec),
- `Pasted text.txt` (auxiliary note content).

The expansion posture is to treat **hybrid-mechlab** as the contract owner: schedule/IR, topology interfaces, trace schema, and evidence-bundle format. Any BLT/sidecar tooling consumes these outputs rather than defining parallel schemas.

### Python binding surface (PyO3) and Python 3.13 constraints

PyO3 supports actively supported Python versions and provides version-specific cfg flags such as `Py_3_13`; it also documents how `abi3` wheels work and how runtime version checks may still be required. citeturn1search2turn1search3turn1search10 Because the plan explicitly targets Python 3.13 environments, the upgrade strategy should track modern PyO3 releases and avoid relying on forward-compat flags as the sole fix. PyO3’s release notes and changelog show ongoing packaging and free-threading-related changes, so pinning and CI testing against the actual interpreter variant remains necessary. citeturn3search3turn1search4turn1search9

A binding surface consistent with the plan:

```python
class HybridLab:
    def attach(self, model, *, family: str, backend: str, schedule=None, hooks=None) -> "HybridLab":
        ...

    def compare(self, left, right, *, metrics=None, trace_schema="hm.v1") -> "ComparisonReport":
        ...
```

and typed objects mirroring Rust:

```python
@dataclass(frozen=True)
class FamilyDescriptor:
    family: str
    regimes: list[str]
    bridge: "BridgeSpec"
    d_model: int
    state_bytes: int
    opaque_fields_hash: bytes
```

### Liger integration points

Liger Kernel is a Triton-kernel suite intended to patch or wrap transformer training stacks, and is integrated in some training ecosystems with simple flags (e.g., within TRL). citeturn2search0turn2search3turn2search6 It also exposes wrapper/model-specific patching APIs (e.g., AutoModel wrappers and monkey-patching functions), which makes it a plausible “migration backend” for *execution bridging* even if it is not itself a transport kernel. citeturn2search11turn2search12

The design constraint “Liger is workspace-level, not `hm_core`” is therefore coherent: `hm_core` remains a portable semantic-contract crate, while `hm_liger` sits in the runtime layer and handles:
- load + monkey patch,
- config translation,
- trace execution and capture,
- emitting the same trace schema as native/adapters.

### IR layering and compiler substrate analogy

The seed memo’s IR stance is well grounded in contemporary compiler practice:
- MLIR is explicitly designed to represent and progressively lower high-level dataflow graphs into target-specific code in a single framework. citeturn11search2
- StableHLO is explicitly a portability layer between ML frameworks and ML compilers. citeturn12search0
- `torch.fx` is explicitly a tracer + IR + transformation toolkit for PyTorch modules. citeturn11search0

In hybrid-mechlab terms, this supports the design of:
- Semantic IR: regime events + state semantics.
- Graph IR: hookpoint DAG for a run segment.
- Numeric IR: optional lowered equivalence artifacts (kernel conformance tools).

## Validation: conformance suites, formal properties, and benchmark metrics

### Formal properties to verify

The plan requires *conformance tests* that establish the substrate as a research-grade reference. The following properties are phrased so they can be checked mechanically:

**Replayability.** Given identical inputs and identical descriptor/schedule, the trace must be reproducible (bitwise for deterministic backends; distributionally for stochastic backends with fixed RNG seed and explicit nondeterminism reporting).

**Determinism (in the trace layer).** Even if underlying kernels use nondeterministic GPU primitives, the trace schema must record enough provenance (seed, device, kernel IDs) so that determinism can be asserted or falsified.

**Bridge-crossing invariants.** Bridge entry/exit events in the trace must match the `BridgeSpec` cadence mask, with no hidden global synchronization. For gated bridges (e.g., gated attention), gates must be within declared law ranges. citeturn5search1turn0search1

**Topology-preservation (stability under small perturbations).** For defined topology summaries, small perturbations of the input trace graph should induce bounded changes in the chosen topological representation. Persistence images and landscapes are designed explicitly as stable representations suitable for statistical comparison. citeturn8search10turn9search0

### Metrics defined as measurable quantities

The initial wedge requires metrics that align with the tract/bridge decomposition:

**Bridge dependence.** For a window \(W\), define
\[
\mathrm{BD}(W) = \frac{\sum_{j \in \mathrm{bridges}(W)} \| \Delta y_j \|}{\sum_{i \in \mathrm{tracts}(W)} \| \Delta y_i \| + \sum_{j \in \mathrm{bridges}(W)} \| \Delta y_j \|}
\]
where \(\Delta y\) is a chosen contribution proxy (activation delta norm, attribution mass, or logit-margin delta). For gated attention bridges, BD can be decomposed gate-weighted. citeturn5search1

**Tract retention.** A family-agnostic operational definition is to measure persistence of tracked entity/state features under long context by correlating early-window and late-window feature codes (or state summaries) under controlled interventions. This aligns with the motivation for hybrids: preserve recall without full KV growth. citeturn0search3turn4search0

**Topology drift.** For two traces \(A,B\), define drift via a stable vectorization \(v(\cdot)\) (persistence images or landscapes) and a norm:
\[
\mathrm{TD}(A,B)=\|v(D(A)) - v(D(B))\|_2.
\]
Stability and vector-space structure are precisely why images/landscapes are used. citeturn8search10turn9search0

**Replacement divergence.** For a replacement-model-capable system (BLT-style transcoders), define divergence as output deviation between original model and replaced-model runs under matched inputs. This is the core “faithfulness” measure the sidecar needs.

### Minimal reproducible long-context benchmark harness

A first harness should be *transport-heavy* and *bridge-sensitive*, e.g. “entity/state retention under adversarial distractors”:
- Build sequences where a small set of latent state variables evolves under deterministic rules (finite-state machine or simple algebraic transition).
- Insert long distractor spans to exceed typical attention windows.
- Evaluate correctness as a function of (a) bridge cadence, (b) bridge gating strength, (c) tract state perturbations.

This benchmark design is consistent with the known motivation for hybrids: recurrent/linear components are efficient but bounded; periodic bridges prevent information from being trapped in bounded recurrent state. citeturn11search1turn0search1

## Portfolio comparison and research agenda

### Native-core families vs reference profiles

The portfolio split is technically coherent because the families stress different transport laws:

- **Native-core families** (kernel-first): Gated DeltaNet (gated delta-rule recurrence). citeturn5search2 HGRN2 (state expansion without extra parameters). citeturn10search4 RetNet (parallel/recurrent/chunkwise retention). citeturn0search3 Hawk (gated linear recurrences). citeturn6search0 TransNormerLLM (linear attention with normalization/positional acceleration). citeturn10search2

- **Understanding-first reference profiles** (adapter-first): Qwen3.5 (explicit 3:1 delta/attention cadence). citeturn0search1 OLMo Hybrid (delta/attention interleaving + mixed cache). citeturn11search1 Kimi Linear (KDA + global attention, claims KV-cache reduction and throughput gains). citeturn4search0

A compact comparison table (interfaces are family-agnostic; semantics differ):

| Axis | Native-core kernels | Reference adapters |
|---|---|---|
| Purpose | Canonical transport semantics | Mapping fidelity + replay |
| Primary risk | Kernel correctness + conformance | Hook/IR mismatch + vendor drift |
| Evidence | Conformance suite + digests | Trace-schema parity + alignment errors |
| Best early target | Gated DeltaNet | Qwen3.5 |

### Recommended experiments and proof outlines

The research-grade claim is: “the IR abstraction is faithful enough to compare families and reason about bridge/tract behavior.” Suggested validations:

**Lemma (Schedule soundness).** If two executions share identical `TransportSchedule` and satisfy bridge-crossing invariants, then the induced event word over \(\{\mathcal{T},\mathcal{B}\}\) is identical. Proof outline: by construction, regime tags and bridge masks are deterministic functions of schedule; invariants eliminate hidden bridges.

**Lemma (Digest determinism).** For a deterministic kernel and fixed input, `TransportDigest` is invariant under replay. Proof outline: show digest update is a pure function of (state, input, params) and uses deterministic hashing only.

**Lemma (Topology stability under bounded perturbations).** If a trace-graph perturbation is bounded in the chosen metric (edge weight perturbation bounded), then the topology representation distance is bounded. Proof outline: reduce to stability results of the chosen representation (persistence images/landscapes) under bottleneck/Wasserstein perturbations. citeturn8search10turn9search0

For formal-methods integration, the substrate can be *specified* and partially verified with: TLA+ for protocol-level invariants (trace schema evolution, evidence bundle state machine). citeturn3search4 Lean for kernel-level proof re-checking workflows (e.g., checker replays). citeturn3search5 Rocq for extraction-style certified components when feasible (recognizing that typical extraction targets are OCaml/Haskell/Scheme, so Rust integration would be an additional compilation boundary). citeturn3search0turn3search2

## Roadmap and current plan snapshot

### Phased roadmap as acceptance criteria

The portfolio plan can be captured as a deliverables/acceptance table consistent with “substrate before performance”:

| Stage | Deliverable | Acceptance criteria |
|---|---|---|
| Foundation stabilization | Family-aware schedule/IR; trace schema v1; PyO3 3.13 build green | `cargo check` + Python import smoke; schedule conformance suite passes; Qwen3.5 adapter produces schema-valid trace |
| First canonical kernel | Native Gated DeltaNet kernel + conformance harness | Native trace matches Qwen-style cadence and bridge semantics; deterministic digests stable across runs |
| Native-core expansion | HGRN2 + RetNet, then Hawk + TransNormerLLM | One shared conformance suite passes across 5 families; no family-specific topology code introduced |
| Reference broadening | OLMo Hybrid + Kimi Linear adapters | Mapping fidelity tests pass; trace parity across adapter/native/liger |
| External wedge | Long-context reasoning harness + comparative evidence bundles | Reproducible reports: bridge dependence, tract retention, topology drift, replacement divergence |

### Current repo plan excerpt

The current plan document emphasizes a topology-first milestone: implement exact persistence offline in `hm_std`, keep `hm_core` trait/IR-only, and drive notebooks and JSON artifacts as the primary output path.

```text
# Topology-First Phase 2: Exact Persistence and Research Artifacts

- Keep the current portfolio/runtime scaffold intact and make the next milestone about real offline topology...
- Use Qwen3.5 and Gated DeltaNet as the canonical comparison pair...
- Implement exact persistence first in hm_std; defer real Mapper execution...
- Optimize outputs for reproducible notebooks backed by stable JSON artifacts...

Public Interfaces
- Rust additions: hm_std::exact_persistence::{ExactPersistenceInput, PersistenceDiagram, ...}
- Python changes: typed compute_persistence(trace) -> PersistenceReport
- Preserve: HybridLab.attach(... family=..., backend=...), shared trace schema
```

This “Topology-First Phase 2” is a coherent de-risk step because it upgrades the substrate’s most differentiating claim—**offline exact topology artifacts for mechanistic comparison**—without requiring full inference-parity work across all families. It also aligns with the stability/rigor motivation for topological summaries and their statistical vectorizations. citeturn9search0turn8search10