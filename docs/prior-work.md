# Prior-work survey for CASSANDRA Chorus

| | |
|---|---|
| Date | 2026-10-07 |
| Status | Draft, not reviewed by owner |
| Prepared by | A research subagent in the Claude Code session of 2026-10-07, using web search and a summarizing page-fetch tool (see the caveat below). Not independently verified, except for the spot-checks in the next row. |
| Spot-checked | On 2026-10-07, against raw data: the arXiv API abstracts of 2602.11543 (SPES), 2408.11304 (FedMoE), 2601.06857 (MoE-DisCo) and 2212.01548 (FedRolex), and the GitHub API record of `zjr2000/SPES` (Apache-2.0, created 2026-02-08, last push 2026-05-08, not archived). All consistent with this file. Claims taken from paper bodies (for example that SPES nodes store a full copy of the model) were not re-checked. |
| Decision references | Updated to SRS Draft 0.4 numbering (D1 to D9, O1 to O3) when the file was added to the repository |
| Scope | PLAN section 9 and SRS section 10 (assumptions on prior work and on Flower / Hivemind maintenance) |
| Method | Bounded web search (WebSearch tool, standard and extended modes) plus page fetching (WebFetch tool), arXiv abstract and HTML pages, two PDFs converted locally with `pdftotext`, and raw JSON from the PyPI and GitHub REST APIs fetched with `curl`. Roughly 27 search queries and 60 page or API fetches, all in one session on 2026-10-07. |
| Important caveat on the method | WebFetch returns a model-written summary of a page, not the raw page. Where a number matters it was cross-checked against raw text where possible (HeteroFL and FedRolex PDFs, PyPI JSON, GitHub API). One summary was demonstrably wrong (Flower release dates) and was replaced by the raw JSON. Numbers marked "per fetched summary" should be re-checked against the paper before any public citation. The GitHub MCP connector failed to connect in this session, so no GitHub code search was done. |

Source status labels used below:

- **read**: full text or a substantial part (method and experiments) was retrieved in this session.
- **abstract-only**: only the arXiv abstract page (or equivalent landing page) was opened.
- **secondary mention**: seen only in a search-result snippet or a third-party page; not opened at the source.

## Summary

1. The closest existing work is **SPES** (arXiv 2602.11543, Feb 2026, code at github.com/zjr2000/SPES, Apache-2.0): decentralized pretraining of MoE language models (2B and 7B from scratch, 9B upcycled) where each node trains the shared part plus its own subset of experts for H = 50 or 100 local steps, the shared part is averaged and each expert is taken from its owner. It differs from Chorus in that every node stores the full model, routing is not masked, and expert ownership is a fixed disjoint partition.
2. **MoE-DisCo** (arXiv 2601.06857, Jan 2026) trains "shared backbone plus one expert" dense submodels independently, averages the shared part, assembles the MoE, then fits the router in a short centralized fine-tune. One-shot, not iterative.
3. **FedMoE** (arXiv 2408.11304, 2024) already uses essentially the SRS S0-F-08 merge rule (unheld experts unchanged, single holder copied, multi-holder FedAvg, router rows following the same pattern), but for fine-tuning a Switch Transformer with 30 clients.
4. Sub-network training on transformers is published, from small (HeteroFL WikiText2, FedRolex Stack Overflow) up to 1B dense LLaMA pretraining (SDP, but with per-step sync) and 2B to 7B MoE (SPES). Assumption B is therefore contradicted.
5. The router worry is real and documented: "routing breakdown" after merging MoEs (HARC, arXiv 2606.03391), "gating misalignment" under federated averaging (FedAlign-MoE, arXiv 2603.21276), and BTX, MoE-DisCo and FlexOlmo all avoid averaging a jointly trained router. Two remedies not in the current Gate A list are worth adding: router rows averaged only among holders of that expert, and no local router training followed by a short central router fit.
6. DiPaCo has no official code (one small, unverified MIT reimplementation, OpenDiPaCo, June 2026). All recent public decentralized pretraining runs found (INTELLECT-1, Covenant-72B, Pluralis-8B via Agora, Psyche Consilience 40B) are dense; Agora gets "no single party holds the full weights" through pipeline parallelism instead.
7. Flower is very active (1.39.0 on 2026-09-28, roughly weekly releases). Hivemind is in low-activity maintenance (1.1.12 on 2026-01-03, last push 2026-01-11, mostly packaging fixes). Petals has had no push since 2024-09-07.

## Verdicts on the two assumptions

### A. "No complete open-source implementation of sliced local training for mixture-of-experts language models exists."

**Verdict: CONTRADICTED as worded. A narrower version remains unrefuted, but is not proven.**

Evidence against the assumption as worded:

- SPES (arXiv 2602.11543; repo https://github.com/zjr2000/SPES, Apache-2.0, created 2026-02-08, last push 2026-05-08, 8 commits) is an official, open implementation of local training for MoE language models in which "each node is responsible for training a distinct subset of experts, while keeping the remaining experts frozen during local updates", synchronizing every H = 50 or 100 steps; shared parameters (embeddings, attention, normalization, router) are averaged and each expert is taken from its owner node. The README describes training code built on OLMo and MegaBlocks, a parameter-server launcher, and Hugging Face checkpoints SPES-2B, SPES-7B and SPES-9B (checkpoint pages not opened). Source status: paper read (HTML, via fetch summaries), README read, GitHub API read.

What SPES does not do, which is exactly where Chorus differs (from the SRS section 3 definition of "slice"):

| Property | Chorus (SRS) | SPES (as read) |
|---|---|---|
| What a worker stores | Shared part plus a subset of experts only | Full model; the paper states "Although each node stores a full copy of the model, gradients and optimizer states are maintained only for the updated parameters" |
| Routing during local training | Masked to held experts (S0-F-07) | Not masked; tokens may be routed to frozen, locally stored copies of other experts |
| Expert overlap | Experts may be held by several workers and averaged among holders (S0-F-08) | Fixed disjoint partition, so no averaging of an expert across workers |
| Slice size | Configurable per worker (S0-F-05) | Uniform in the examples found (for example 16 experts over 16 nodes) |
| Outer optimizer | Optional, e.g. momentum (S0-F-10) | Plain averaging (FedAvg) per the fetched summary |

Other near misses:

- FedMoE (Mei et al., arXiv 2408.11304) implements the same partial-averaging rule but for federated fine-tuning; no code link was found in the version read.
- MoE-DisCo (arXiv 2601.06857) gives an anonymous code link (anonymous.4open.science, not opened); it is one-shot and trains no router locally.
- Hivemind's decentralized MoE (Learning@home, NeurIPS 2020) is open source and no single trainer holds every expert, but it is synchronous remote expert calls, not local training on a slice.

The narrower claim that survives the search: *no open-source implementation was found in which workers materialize only the shared part plus a variable-size subset of experts, train with routing masked to that subset for many local steps, and have overlapping experts averaged only among their holders.* This is a bounded search (no GitHub code search, no citation chasing on SPES or FedMoE, no Chinese-language venues), so absence here is not proof of absence. It is the most plausible candidate for what is distinctive about Chorus, and it should be described in those exact terms rather than as "first" or "novel".

### B. "Published sub-network training results are limited to small vision and feed-forward models."

**Verdict: CONTRADICTED.**

Counterexamples, all opened in this session:

| Work | Model | Scale | Setting |
|---|---|---|---|
| HeteroFL (ICLR 2021, arXiv 2010.01264) | Transformer on WikiText2 | up to 19.3M parameters (Table 3, read in PDF text) | Federated, width sub-models, 100 clients |
| FedRolex (NeurIPS 2022, arXiv 2212.01548) | 3-layer Transformer on Stack Overflow | 10,000-word vocab, 342,477 training clients (read in PDF text) | Federated, rolling sub-model extraction |
| TwIST (arXiv 2511.03983, Nov 2025) | GPT-2 | 124M, fine-tuning on WikiText-103, 4 workers | IST, subnetworks resampled every 15 batches |
| SDP (arXiv 2507.09029, v5 May 2026) | LLaMA-style | 134M, 500M and 1B pretraining on FineWeb (v5); v4 reported only 134M | Subnetworks per worker, but synchronized every step |
| FedMoE (arXiv 2408.11304, 2024) | Switch Transformer, 32 experts | fine-tuning on AG News, SQuAD, XSum | Federated sub-MoE per client |
| SPES (arXiv 2602.11543, 2026) | MoE LLM | 2B and 7B from scratch, 9B upcycled | Subset of experts trained per node, H = 50 or 100 |
| MoE-DisCo (arXiv 2601.06857, 2026) | Qwen1.5-MoE-2.7B, LLaMA-MoE-3.5B | 4 experts as configured | One expert per independent submodel |

Honest caveats: most transformer sub-network results with long local horizons are small (at or below about 124M) or are fine-tuning. The billion-scale dense result (SDP) synchronizes every step, so it says little about H in the hundreds. The billion-scale results with local steps are MoE-specific (SPES, MoE-DisCo). The project could defensibly say "few published results combine partial-model training, long local horizons and language-model pretraining", but not that results are limited to vision and feed-forward models.

## 1. Sub-network, federated-dropout, independent-subnetwork and partial-model training

| Work | Year/venue | Link | Source status | What it shows (model type, scale) | Relevance to Chorus |
|---|---|---|---|---|---|
| Federated Dropout (Caldas, Konečný, McMahan, Talwalkar), "Expanding the Reach of Federated Learning by Reducing Client Resource Requirements" | 2018, arXiv | https://arxiv.org/abs/1812.07210 | abstract-only | Clients train smaller sub-models; abstract claims up to 14x less server-to-client communication, 1.7x less local computation, 28x less upload, "without degrading the quality of the final model". Models not named on the page opened. | Origin of the "client trains a slice" idea. Random extraction; see FedRolex for its weakness. |
| HeteroFL (Diao, Ding, Tarokh) | ICLR 2021, arXiv 2010.01264 | https://arxiv.org/abs/2010.01264 ; code https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients (MIT, last push 2023-02-27) | read (experiments section, PDF text) | Width-sliced sub-models of a global model; CNN on MNIST, PreResNet18 on CIFAR10, Transformer on WikiText2 (largest 19.3M parameters, perplexity reported per mix of client sizes), 100 clients, 10% active per round. | Transformer language-model evidence at small scale. Parameters averaged only over clients whose slice contains them, which is the dense analogue of S0-F-08. |
| FedRolex (Alam, Liu, Yan, Zhang) | NeurIPS 2022, arXiv 2212.01548 | https://arxiv.org/abs/2212.01548 ; code https://github.com/AIoT-MLSys-Lab/FedRolex (Apache-2.0, last push 2024-08-27) | read (experiments section, PDF text) | Rolling (deterministic, shifting) sub-model extraction so all parts of the global model are trained evenly. 3-layer Transformer on Stack Overflow, client capacities 1 to 1/16. Next-word accuracy (Table 3, PDF table layout garbled, mapping consistent with authors' text): FedRolex 29.22, HeteroFL 27.21, Federated Dropout 23.46, homogeneous smallest 27.32, homogeneous largest 29.79. Authors: HeteroFL and Federated Dropout "perform even worse than the model homogeneous case using the smallest model". | Strong caution: random slice extraction can be worse than simply training a smaller model. Supports a rolling or coverage-balancing assignment policy (S0-F-06). |
| InclusiveFL, "No One Left Behind" (Liu et al.) | KDD 2022, arXiv 2202.08036 | https://arxiv.org/abs/2202.08036 | abstract-only | Different-size models per client with momentum knowledge distillation; abstract names BERT as motivating case. Exact experimental models not confirmed. | Heterogeneous capacity in federated transformers; distillation route, not slicing. |
| Towards a Better Theoretical Understanding of IST (Shulgin, Richtárik) | ICML 2024, arXiv 2306.16484 | https://arxiv.org/abs/2306.16484 | abstract-only | Theory of independent subnetwork training on quadratic models. | Theoretical grounding only. |
| Model Parallelism With Subnetwork Data Parallelism, SDP (Singh, Khalid, Cagnasso, Oyallon, Belilovsky) | arXiv 2507.09029, v1 Jul 2025, v5 May 2026 | https://arxiv.org/abs/2507.09029 | read (HTML v4 and v5) | Each worker trains a structured subnetwork (neuron-level or block-level), every parameter assigned to at least one worker, masked averaging among holders, synchronized every step. v5: LLaMA-style 134M, 500M, 1B on FineWeb, 8 workers; 1B best variant validation loss 2.437 vs DDP 2.452 with 28% lower peak memory (per fetched summary). v4 reported only 134M. Abstract also covers ResNet-18 on CIFAR. | Shows partial-model training of a dense LLM to 1B works when syncing every step. Says nothing about H in the hundreds. Masked "average among holders" is the same rule as S0-F-08. |
| TwIST (Menezes et al.) | arXiv 2511.03983, Nov 2025 | https://arxiv.org/abs/2511.03983 | read (HTML) | GPT-2 124M fine-tuned on WikiText-103, 4 P100 workers, subnetworks (attention heads, FFN blocks) resampled every 15 batches, "ensuring that every parameter is included", first and last two layers shared, shared parameters averaged among holders. Reports 23.14 PPL vs SparseGPT 31.64 at aggressive pruning. Code at an anonymous link (not opened). | IST on a language model with short local horizons; same coverage and holder-averaging rules as Chorus. |
| DEPT: Decoupled Embeddings for Pre-training Language Models (Iacob et al.) | ICLR 2025, arXiv 2410.05021 | https://arxiv.org/abs/2410.05021 | abstract-only (variant details from search snippet: secondary mention) | Federated pretraining where the transformer body is aggregated with an outer optimizer and token embeddings are trimmed or kept local per data source; claims "first vocabulary-agnostic federated pre-training of billion-scale models". | A partial-model (embedding-level) precedent at billion scale with outer optimizer. Same lab as Photon. |
| Evolving Subnetwork Training for LLMs (EST) | arXiv 2406.06962, 2024 | https://arxiv.org/abs/2406.06962 | secondary mention | Samples subnetworks (heads, MLP width, layers) per step during centralized LLM training to save compute. | Not distributed; listed only to avoid confusion with IST. |

## 2. DiPaCo and path or module composition

| Work | Year/venue | Link | Source status | What it shows (model type, scale) | Relevance to Chorus |
|---|---|---|---|---|---|
| DiPaCo: Distributed Path Composition (Douillard et al., Google DeepMind) | arXiv 2403.10616, Mar 2024 | https://arxiv.org/abs/2403.10616 | read (HTML v1) | Modules arranged in levels (e.g. 16x16 = 256 paths of 150M parameters each), routing at sequence level (k-means on features of the first 32 tokens, or a discriminative router), each worker trains one path with about 150 inner steps, module updates averaged over the paths that share the module with a Nesterov outer optimizer, outer-gradient norm rescaled by the square root of the number of paths through a module, shard-size weighting. Beats a 1B dense model on C4 at the cost of one 150M path at inference. Stated limits: "significantly less FLOP efficient per evaluation perplexity than a standard dense compute optimal model", one path-size scale, one dataset. No code mentioned. | This is the project's fallback ("modular paths"). Its per-module averaging and sqrt(paths) rescaling are directly reusable for per-expert averaging with varying holder counts. |
| OpenDiPaCo (MasonFlint44) | GitHub, created 2026-06-11 | https://github.com/MasonFlint44/OpenDiPaCo | read (README and GitHub API) | Independent MIT implementation, 63 commits, last push 2026-06-21, 0 stars, no stated affiliation with the authors. Self-reported C4 runs from 4 to 36 paths; README says paper-scale validation (256 paths x 150M, multi-GPU) "remains open". | Only open DiPaCo implementation found. Small and unverified; useful as reading material for the fallback, not as a dependency. |
| No Need to Talk: Asynchronous Mixture of Language Models, SMALLTALK LM (Filippova, Katharopoulos, Grangier, Collobert) | ICLR 2025, arXiv 2410.03529 | https://arxiv.org/abs/2410.03529 | abstract-only (router size from search snippet: secondary mention) | Independently trained expert LMs with a lightweight prefix-based sequence router; lower perplexity than dense at equal training FLOPs. | A simpler fallback than DiPaCo: whole-model experts, no shared modules, no sync. |
| Branch-Train-Merge (Li et al.) | arXiv 2208.03306, Aug 2022 | https://arxiv.org/abs/2208.03306 | abstract-only | Embarrassingly parallel expert LMs per domain; 64 domains, 192B tokens, 22.4B total parameters matching a dense transformer trained with 2.5x more compute. | Fully independent experts, no shared part; the extreme end of the design space. |

## 3. Federated or decentralized MoE training, and expert or router merging

| Work | Year/venue | Link | Source status | What it shows (model type, scale) | Relevance to Chorus |
|---|---|---|---|---|---|
| SPES: Pretraining A Large Language Model using Distributed GPUs: A Memory-Efficient Decentralized Paradigm (Zhang, Xiao, Wu, Zhang, Zhang) | arXiv 2602.11543, v1 Feb 2026, v3 Jun 2026 | https://arxiv.org/abs/2602.11543 ; code https://github.com/zjr2000/SPES (Apache-2.0, last push 2026-05-08) | read (HTML via fetch summaries; README; API) | MoE pretraining; each node trains shared part plus a disjoint subset of experts while storing the full model; H = 50 or 100; shared part averaged, experts taken from owners; losses: cross-entropy, z-loss, load-balancing loss; early "expert-merging warm-up" that blends each expert with its most cosine-similar experts with a decaying coefficient (ablation 48.99 to 50.95 average score per fetched summary). Abstract: 2B on "16 standalone 48GB GPUs" over internet, 7B from scratch, 9B upcycled, "competitive performance with centrally trained LLMs under similar computational budgets". Per fetched summary: DiLoCo needs 55 GB per node vs 35 GB for SPES. | Closest prior work. Chorus is strictly harder on memory (workers never store other experts) and on routing (masked). Reusable: z-loss and load-balancing loss, expert-merging warm-up, DiLoCo as baseline. |
| MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models (Ye, Cheng, Zhang, Zhang) | arXiv 2601.06857, Jan 2026 | https://arxiv.org/abs/2601.06857 | read (HTML v1) | Decomposes an MoE into dense submodels "shared backbone plus one expert", data split by unsupervised clustering, each trained independently "without any inter-device communication", shared parts averaged, gate "not utilized" during submodel training and learned in a brief global fine-tune on high-bandwidth GPUs. Qwen1.5-MoE-2.7B and LLaMA-MoE-3.5B (4 experts as configured), C4, WikiText-2, OpenWebText; claims 47.6% to 69.5% cost reduction at comparable quality. One-shot, not iterative. Whether initialization is random or from checkpoints is not stated clearly in what was read. | Shows the "no local router, central router fit" remedy works in at least one setting. Its one-shot design is the H equals infinity end of Chorus's sweep. |
| FedMoE: Personalized Federated Learning via Heterogeneous Mixture of Experts (Mei, Cai, Zhou, Wang, Xu) | arXiv 2408.11304, Aug 2024 | https://arxiv.org/abs/2408.11304 | read (HTML v1) | Switch Transformer with 32 experts per layer, 30 clients (18 to 24 GB), fine-tuning on AG News, SQuAD, XSum. Each client gets a sub-MoE chosen from activation statistics. "Modular aggregation": unactivated experts unchanged, single-client experts updated directly, shared experts FedAvg, router dimensions follow the same pattern. | Closest prior art for the S0-F-08 merge rule, including per-expert router rows. Fine-tuning only, personalized, no code found. |
| FLEX-MoE: Federated Mixture-of-Experts with Load-balanced Expert Assignment (Zhang et al.) | arXiv 2512.23070, Dec 2025 | https://arxiv.org/abs/2512.23070 | read (HTML v1) | ResNet with MoE layers, 8 experts, CIFAR-10, EMNIST, GTSRB. Clients store at most k_c experts; server solves a linear program over client-expert fitness scores with load upper and lower bounds and "historical deficit" tracking. Experts averaged among assigned clients weighted by usage; routers are per-client and not aggregated. Greedy top-k assignment gave severe imbalance (CV above 0.29 vs about 0.003). | Directly reusable for S0-F-05/S0-F-06 (capacity-aware, coverage-guaranteeing assignment). Sidesteps the shared-router problem rather than solving it. Vision only. |
| Learning to Specialize: Joint Gating-Expert Training for Adaptive MoEs in Decentralized Settings (DDOME; earlier title FedJETs) | arXiv 2306.08586, v1 Jun 2023, v3 Jun 2025 | https://arxiv.org/abs/2306.08586 | abstract-only | Federated MoE where a pretrained common expert informs the gate; clients use personalized expert subsets; image and text classification. | Evidence that gate training with expert subsets needs an anchor. Note the retitling when citing. |
| Federated Mixture of Experts, FedMix (Reisser, Louizos, Gavves, Welling) | arXiv 2107.06724, Jul 2021 | https://arxiv.org/abs/2107.06724 | abstract-only | Ensemble of specialists; each user adaptively selects and trains a subset of members. | Early "clients train only some experts" precedent; not a sparse transformer. |
| FLEx: Personalized Federated Learning for MoE LLMs via Expert Grafting (Liu et al.) | arXiv 2506.00965, Jun 2025 | https://arxiv.org/abs/2506.00965 | abstract-only | Pretrained MoE LLM; aggregates only shared non-expert parameters, pretrained experts frozen, personalized grafted expert and gate per client. | Opposite choice to Chorus: experts never averaged. |
| HFedMoE (Fang et al.) | arXiv 2601.00583, Jan 2026 | https://arxiv.org/abs/2601.00583 | abstract-only | LLM fine-tuning; client expert subsets fitted to compute budget; "sparsity-aware" aggregation of experts and gate with importance weights; names "misalignment during global aggregation when clients use different expert subsets" as a challenge. | Confirms the merge-misalignment risk in the subset setting. |
| Aggregation Alignment for Federated Learning with MoE under Data Heterogeneity, FedAlign-MoE (Fang et al.) | arXiv 2603.21276, Mar 2026 | https://arxiv.org/abs/2603.21276 | abstract-only | Identifies "gating misalignment" (averaged gates become "one-size-fits-none") and "expert semantic blurring" (same-index experts drift to different roles) under naive averaging. | Directly names the failure modes Chorus's Gate A should measure, especially under skewed data (S0-F-12). |
| When Model Merging Breaks Routing: Training-Free Calibration for MoE, HARC (Huang et al.) | arXiv 2606.03391, Jun 2026 | https://arxiv.org/abs/2606.03391 | abstract-only | Merging fine-tuned MoE LLMs causes "routing breakdown"; proposes a training-free Hessian-aware router recalibration solved with conjugate gradient. | Candidate post-merge remedy for Gate A; setting (merging fine-tunes of one base) differs from Chorus. |
| Branch-Train-MiX, BTX (Sukhbaatar et al., Meta FAIR) | arXiv 2403.07816, Mar 2024 | https://arxiv.org/abs/2403.07816 | abstract-only | Dense experts trained in parallel from a seed model; FFNs become MoE experts, other parameters averaged, then an MoE fine-tune "to learn token-level routing". | Averaging the non-expert part after independent training works when followed by router fine-tuning. |
| FlexOlmo: Open Language Models for Flexible Data Use (Shi et al., AI2) | arXiv 2507.07024, v1 Jul 2025, v4 Aug 2025 | https://arxiv.org/abs/2507.07024 | read (HTML v4) | Each data owner trains one expert FFN plus its own router embedding, with the public expert and shared attention frozen as an anchor; router rows initialized from domain embeddings; experts concatenated with no joint training, optional router tuning on a small public proxy set; models up to 37B (20B active); reports that BTX and model soups do poorly because experts "diverge too much from one another". | Strong evidence for two remedies: router rows that travel with their expert, and a frozen shared anchor. Maps onto "freeze the router after a central warm-up", extended to freezing more of the shared part. |
| Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts, Learning@home (Ryabinin, Gusev) | NeurIPS 2020, arXiv 2002.04013 | https://arxiv.org/abs/2002.04013 ; docs https://learning-at-home.readthedocs.io/en/stable/modules/client.html | abstract-only (paper); read (docs page) | Experts hosted by volunteers and found through a DHT; trainer holds a local gating function and calls remote experts; tolerates failures (k_min, timeouts, gradients "averaged without the missing experts"). | Volunteer MoE where no trainer holds all experts, but synchronous per-step remote calls rather than local slices. Implemented in Hivemind. |
| Decoupled DiLoCo for Resilient Distributed Pre-training (Douillard et al.) | arXiv 2604.21428, Apr 2026 | https://arxiv.org/abs/2604.21428 | abstract-only | Asynchronous independent learners with a central synchronizer; abstract states results "for both dense and mixture-of-expert architectures". | Full-model local averaging with MoE; relevant to S0-F-14 baseline with an MoE model. |

## 4. Low-communication distributed LLM training (full-model local-averaging baseline)

| Work | Year/venue | Link | Source status | What it shows (model type, scale) | Relevance to Chorus |
|---|---|---|---|---|---|
| DiLoCo (Douillard et al.) | arXiv 2311.08105, v1 Nov 2023, v3 Sep 2024 | https://arxiv.org/abs/2311.08105 | abstract-only | Federated-averaging variant with AdamW inner and Nesterov momentum outer; 8 workers match fully synchronous training "while communicating 500 times less". | Reference design for S0-F-10 and S0-F-14. |
| OpenDiLoCo (Jaghouar, Ong, Hagemann, Prime Intellect) | arXiv 2407.07852, Jul 2024 | https://arxiv.org/abs/2407.07852 ; code https://github.com/PrimeIntellect-ai/OpenDiloco (Apache-2.0, last push 2026-07-17) | abstract-only; repo via API | Open DiLoCo reproduction built on Hivemind; two continents, three countries, 90 to 95% compute utilization; FP16 pseudo-gradient compression without loss. | Open baseline code; shows Hivemind used for this purpose. |
| INTELLECT-1 Technical Report (Jaghouar et al.) | arXiv 2412.01152, Dec 2024 | https://arxiv.org/abs/2412.01152 ; framework https://github.com/PrimeIntellect-ai/prime (MIT, last push 2026-10-06) | abstract-only; repo via API | 10B dense model, 1T tokens, up to 14 concurrent nodes on 3 continents, 30 compute providers, DiLoCo with FSDP2 and int8 all-reduce, 400x less bandwidth than data parallel. | Largest documented DiLoCo-style run; dense. |
| Streaming DiLoCo with Overlapping Communication (Douillard et al.) | arXiv 2501.18512, Jan 2025 | https://arxiv.org/abs/2501.18512 | abstract-only | Synchronizes parameter subsets in sequence, overlaps communication, quantizes exchanges; "reducing required bandwidth by two orders of magnitude" at billion scale. | Partial (per-fragment) synchronization schedules are a precedent for "average the router more often than the experts". |
| Scaling Laws for DiLoCo (Charles et al.) | arXiv 2503.09799, Mar 2025 | https://arxiv.org/abs/2503.09799 | abstract-only | DiLoCo "scales both predictably and robustly with model size". | Baseline expectations for S0-F-14 at larger scale. |
| Photon: Federated LLM Pre-Training (Sani et al.) | MLSys 2025, arXiv 2411.02908 | https://arxiv.org/abs/2411.02908 | abstract-only | Federated pretraining up to 7B, better perplexity than centralized, 64x to 512x less communication, small client batches with high learning rates. | Federated (FedAvg-style) baseline at LLM scale; hyperparameter hints for local steps. |
| Covenant-72B (Lidin et al., Templar) | arXiv 2603.08163, Mar 2026 | https://arxiv.org/abs/2603.08163 | abstract-only | About 1.1T tokens, permissionless peers, SparseLoCo optimizer; claims parity with centralized models at comparable compute. Abstract does not describe an MoE. | Shows sparse pseudo-gradient methods at 72B over internet. |
| Can Model Merging Improve Aggregation in DiLoCo? (Horoi et al.) | arXiv 2607.03011, 2026 | https://arxiv.org/pdf/2607.03011 | secondary mention | Title suggests merging methods as alternatives to averaging in DiLoCo. Not opened. | Possibly relevant to merge rules; follow up. |

## 5. Decentralized and volunteer LLM training systems

| Work | Year/venue | Link | Source status | What it shows (model type, scale) | Relevance to Chorus |
|---|---|---|---|---|---|
| Distributed Deep Learning in Open Collaborations, DeDLOC (Diskin et al.) | NeurIPS 2021, arXiv 2106.10207 | https://arxiv.org/abs/2106.10207 | abstract-only | Volunteer collaborative training; SwAV and ALBERT pretraining; collaborative LM pretraining "with 40 participants". | Earliest documented volunteer LM pretraining; full-model replicas. |
| SWARM Parallelism (Ryabinin, Dettmers, Diskin, Borzunov) | ICML 2023, arXiv 2301.11913 | https://arxiv.org/abs/2301.11913 | abstract-only | Randomized, self-rebalancing pipelines on preemptible T4 GPUs with under 200 Mb/s; 1B shared (about 13B unshared) parameter transformer. | Model-parallel route to "no worker holds the whole model". |
| Petals (Borzunov et al.) | arXiv 2209.01188, 2022 (venue not confirmed on page opened) | https://arxiv.org/abs/2209.01188 ; https://github.com/bigscience-workshop/petals | abstract-only; README read; API read | Collaborative inference and fine-tuning of large models (BLOOM-176B about one step per second); README lists Llama 3.1, Mixtral, Falcon, BLOOM and a swarm monitor. Repo: MIT, last push 2024-09-07, last release v2.2.0 on 2023-09-06, 94 open issues and 20 open PRs. | Inference and fine-tuning, not pretraining. Appears unmaintained since 2024. |
| Learning@home and Hivemind | NeurIPS 2020 | see section 3 and 6 | see there | Decentralized MoE; library for volunteer training. | See section 6 for status. |
| INTELLECT-1 | Dec 2024 | see section 4 | abstract-only | 10B dense, 30 providers. | Dense DiLoCo run. |
| Agora: Collective and Permissionless Internet-Scale Pretraining of LLMs (Avraham et al., Pluralis Research) | arXiv 2607.13332, Jul 2026 | https://arxiv.org/abs/2607.13332 | abstract-only (run details from search snippets: secondary mention) | Pipeline-parallel sharding over internet links with fault-tolerant collectives; participants hold one stage so "no single party ever possesses the full weights". Snippets: Pluralis-8B, 8.6B dense, 500B FineWeb-Edu tokens, 40 days, 330 nodes mostly consumer GPUs, same-stage workers average 5% of parameters every 20 local steps (not verified at source). | The most recent public volunteer pretraining run found; achieves Chorus's "no worker holds the whole model" goal by pipeline parallelism on a dense model. |
| Covenant-72B | Mar 2026 | see section 4 | abstract-only | 72B, permissionless, Bittensor-based. | Dense; trustless participation (relevant to Stage 3). |
| Psyche / Consilience 40B (Nous Research) | Blog, September 2025 per fetched page | https://nousresearch.com/the-next-phase-of-psyche | read (blog post) | States the "Consilience 40B run was the largest distributed pre-training run ever" and that Nous is "moving from provability to performance". Architecture (MLA) and 20T-token target appear only in secondary snippets. | Shows a large volunteer-style run being wound down toward other priorities; dense. |

## 6. Maintenance status of Flower and Hivemind (with Petals for context)

This section uses a different table shape from sections 1 to 5 on purpose: the topic is repository health, not papers, so the columns are release, commit and issue data rather than "what it shows".

All values fetched on 2026-10-07 from the PyPI JSON API (`https://pypi.org/pypi/<name>/json`) and the GitHub REST API (`https://api.github.com/repos/...`, plus the issue search endpoint). Source status: read (raw JSON).

| Project | Repository | Licence | Latest release (date) | Release cadence | Last push / newest commit on default branch | Recent commit activity | Open issues / open PRs | Issue flow |
|---|---|---|---|---|---|---|---|---|
| Flower (`flwr`) | https://github.com/flwrlabs/flower (the old `adap/flower` path redirects) | Apache-2.0 | 1.39.0 (2026-09-28) | 1.31.0 2026-06-08, 1.32.0 2026-06-25, 1.32.1 2026-07-01, 1.33.0 2026-08-05, 1.34.0 2026-08-19, 1.35.0 2026-08-25, 1.36.0 2026-09-01, 1.37.0 2026-09-15, 1.38.0 2026-09-22, 1.39.0 2026-09-28 | push 2026-10-06; newest commit 2026-10-01 | The 100 most recent commits span 2026-09-15 to 2026-10-01 (about 100 commits in two weeks) | 48 issues / 354 PRs (repo-level counter 402 includes PRs) | 19 issues opened since 2026-07-01, 3 closed in that window. Very active development; the large open-PR count is worth noting but is typical of a busy repo. |
| Hivemind | https://github.com/learning-at-home/hivemind | MIT | 1.1.12 (2026-01-03) | 1.1.7 2023-03-31, 1.1.8 2023-05-01, 1.1.9 2023-07-23, 1.1.10.post2 2023-08-31, then 1.1.11 2025-04-20, 1.1.12 2026-01-03 | push 2026-01-11; newest commit 2026-01-04 ("Make Hivemind compatible with PEP 517 (#670)") | Commits per month in the last 100: 2025-03: 8, 2025-04: 3, 2025-05: 1, 2025-08: 2, 2025-10: 1, 2026-01: 1. Recent commits are build and packaging fixes. | 70 issues / 24 PRs | 2 issues opened since 2025-10-01, 1 closed. Low-activity maintenance: not archived, still released, but no feature work visible in the last year. |
| Petals | https://github.com/bigscience-workshop/petals | MIT | v2.2.0 (2023-09-06) | none since 2023 | push 2024-09-07; newest commit 2024-08-25 | 12 commits in 2024-07, 1 in 2024-08, none after | 94 issues / 20 PRs | Appears unmaintained since 2024. |

Implication for SRS open item O2 (Flower, Hivemind or custom networking): Flower is the only one of the three with active development. Hivemind remains usable and installable (PEP 517 fix in January 2026) but a Stage 1 dependency on it would carry maintenance risk. Prime Intellect's `prime` framework (MIT, push 2026-10-06) is an actively maintained alternative for DiLoCo-style runs; it was not evaluated further.

## Design implications for Stage 0

Each item cites the source it rests on. Items marked **(inference)** are this survey's reasoning, not a claim made by any source.

**Merge rule (S0-F-08, S0-F-09)**

1. The planned merge rule is already used by FedMoE ("modular aggregation", arXiv 2408.11304), HeteroFL-style partial averaging (arXiv 2010.01264), SDP's masked averaging (arXiv 2507.09029) and TwIST (arXiv 2511.03983). It is well precedented; unit tests in S0-N-06 can use those definitions as reference behaviour.
2. **Router rows should probably follow their expert, not the shared part.** FedMoE averages router dimensions with the same per-expert pattern as experts; FlexOlmo trains each expert's router embedding together with that expert (arXiv 2507.07024). **(inference)** If masking sets non-held logits to minus infinity, the router rows of non-held experts get zero gradient on that worker. Averaging the whole router over all N workers then scales the update to row j by n_j / N (n_j = number of holders), a silent learning-rate cut for rarely held experts. Averaging row j over its holders only removes that bias. This is a candidate change to S0-F-08, which currently puts the router in the shared part. Decoupled weight decay (AdamW) would also shrink non-held rows on every worker; exclude them from weight decay.
3. **(inference)** Masked routing changes the gate normalization: a softmax over the held subset gives larger weights than a softmax over all experts after the merge. Using top-k followed by softmax over the selected k (renormalized weights) keeps the combination scale consistent between masked training and full inference. Worth fixing in S0-F-02 before Task A, because otherwise a scale mismatch could be mistaken for a routing failure.
4. Experts held by nobody stay unchanged: same as FedMoE ("unactivated experts remain unchanged") and SPES (owner replacement). If an outer optimizer with momentum is used (S0-F-10), decide explicitly whether an unheld expert's momentum buffer is applied, frozen or decayed; the SRS rule "left unchanged" implies skipping the outer step for it. DiPaCo applies its outer update per module over the paths that touched it (arXiv 2403.10616).

**Outer optimizer (S0-F-10)**

5. DiLoCo's AdamW inner plus Nesterov outer (arXiv 2311.08105) is the standard choice. DiPaCo adds two rules that transfer directly to experts with varying holder counts: rescale each module's outer gradient norm by the square root of the number of paths through it, and weight outer gradients by shard size (arXiv 2403.10616). SPES used plain averaging and still matched centralized training at 2B (arXiv 2602.11543), so plain averaging is a reasonable default as the SRS states.

**Slice assignment (S0-F-05, S0-F-06)**

6. FedRolex (arXiv 2212.01548) found random sub-model extraction (Federated Dropout) worse than training the smallest model homogeneously on a transformer LM, while rolling extraction matched the full homogeneous model. Add a **rolling (round-robin) assignment** policy alongside random and guaranteed-coverage.
7. FLEX-MoE (arXiv 2512.23070) gives a ready formulation for capacity-limited, load-balanced assignment: a linear program with per-client capacity k_c, lower and upper expert-load bounds, and "historical deficit" tracking. This fits S0-F-05/S0-F-06 and also gives a principled way to guarantee coverage under heterogeneous slice sizes.
8. SRS decision D6 (same expert indices in every layer or independent per layer; decided 2026-10-07: both supported, same indices the default, S0-F-25): DiPaCo's path structure picks modules per level independently, and SDP's block-level variant drops whole blocks. Neither source settles the question for MoE; it remains an experiment.

**Router remedies (PLAN Gate A)**

9. The central worry is documented elsewhere: "routing breakdown" after merging MoE models (HARC, arXiv 2606.03391), "gating misalignment" and "expert semantic blurring" under federated averaging (FedAlign-MoE, arXiv 2603.21276), and merge misalignment with heterogeneous expert subsets (HFedMoE, arXiv 2601.00583). Gate A should measure expert semantic drift (same-index experts used for different maps across workers) in addition to router consistency, especially under skewed data (S0-F-12).
10. Remedies seen in the literature that the current Gate A list lacks:
    - Router rows averaged among holders only (FedMoE; item 2 above).
    - No router training on workers, then a short central router fit after the merge (MoE-DisCo, arXiv 2601.06857; BTX, arXiv 2403.07816). This is a stronger form of "freezing the router after a centrally trained warm-up".
    - A frozen shared anchor during local training, i.e. freezing attention and a common expert as well as the router (FlexOlmo, arXiv 2507.07024; DDOME/FedJETs common expert, arXiv 2306.08586).
    - Training-free router recalibration after each merge (HARC, arXiv 2606.03391).
    - An early expert-merging warm-up that blends similar experts with a decaying coefficient (SPES, arXiv 2602.11543), plus z-loss alongside the load-balancing loss.
    Adding these does not change the plan's order (try remedies one at a time, then fall back), but it lengthens the list and two of them (holder-only router averaging, top-k renormalization) are cheap enough to be defaults rather than remedies.

**Baselines (S0-F-13, S0-F-14)**

11. **(inference)** Add an intermediate baseline modelled on SPES: every worker stores the full model and routes over all experts, but updates only its assigned experts. Comparing full-model local averaging (S0-F-14), SPES-style partial updates, and Chorus sliced training separates the cost of partial expert updates from the cost of masked routing, which is exactly the question in PLAN section 2.
12. SPES compared against DiLoCo; Photon (arXiv 2411.02908) reports that small client batches with high learning rates help federated pretraining. These are useful starting hyperparameters for S0-F-14.

**Modular-paths fallback**

13. DiPaCo has no official code; OpenDiPaCo (MIT, June 2026) is small and self-described as not validated at paper scale. DiPaCo itself reports being less FLOP-efficient than a compute-optimal dense model. SMALLTALK LM (arXiv 2410.03529) is a simpler fallback with sequence-level routing and no shared modules. MoE-DisCo shows a middle ground (shared backbone, one expert per worker, central router fit). The fallback remains reasonable, but the survey suggests trying the MoE-DisCo or FlexOlmo style router handling before abandoning the MoE design.

**Framing**

14. Given SPES, MoE-DisCo, FedMoE and FLEX-MoE, any public description of Chorus should position it as a variant within an existing line of work (partial-expert local training for MoE language models), and state the specific differences: workers never store non-held experts, routing is masked, overlapping holders are averaged, slice sizes vary per worker, and the target is volunteer hardware.

## Search log

### Queries run (WebSearch)

1. `DiPaCo distributed path composition arXiv`
2. `federated dropout transformer language model sub-model training heterogeneous clients`
3. `federated mixture of experts clients hold subset of experts averaging gate`
4. `DiPaCo open source implementation github path composition`
5. `local SGD mixture of experts workers hold subset of experts low-communication decentralized training` (extended)
6. `FedJETs federated mixture of experts just-in-time personalization clients subset of experts`
7. `FedMoE personalized federated learning heterogeneous mixture of experts submodel clients arXiv 2408.11304`
8. `"Learning@home" decentralized mixture of experts volunteer Ryabinin`
9. `FlexOlmo independently trained experts merged mixture of experts router domain embeddings arXiv`
10. `"subnetwork data parallelism" model parallelism transformer training subnetworks`
11. `independent subnetwork training transformer language model IST`
12. `FedRolex rolling sub-model extraction transformer language model results` (returned nothing relevant; paper read directly instead)
13. `DEPT decoupled embeddings pre-training language models federated arXiv`
14. `InclusiveFL "No One Left Behind" inclusive federated learning heterogeneous devices BERT`
15. `DiLoCo mixture-of-experts sparse expert synchronization low-communication training 2025` (extended)
16. `federated pretraining mixture-of-experts language model from scratch clients train subset of experts partial expert aggregation` (extended)
17. `Branch-Train-MiX mixing expert LLMs into a mixture-of-experts arXiv 2403.07816`
18. `SMALLTALK LM asynchronous mixture of language models independently trained router`
19. `decentralized LLM pretraining run 2026 volunteer permissionless internet model released` (extended)
20. `Pluralis Node0 protocol learning pipeline parallel decentralized training 8B`
21. `Nous Psyche Consilience 40B decentralized pretraining DisTrO status`
22. `decentralized mixture-of-experts pretraining consumer GPUs each node trains subset of experts heterogeneous memory 2026 arXiv` (extended)
23. `router trained with masked subset of experts merged mixture of experts routing mismatch federated` (extended)
24. `heterogeneous sub-model federated learning large language model width depth pruning clients pretraining LLaMA` (extended)
25. `github mixture of experts federated averaging expert subset per client local training implementation pytorch`
26. `"mixture of experts" "federated" pretraining language model "from scratch" expert-sliced OR "expert sharding" clients only hold some experts`
27. `DiPaCo reproduction code modular paths DiLoCo open implementation 2025 2026`

API queries: PyPI JSON for `flwr` and `hivemind`; GitHub REST `repos`, `commits`, `releases` and `search/issues` for flwrlabs/flower, learning-at-home/hivemind, bigscience-workshop/petals, zjr2000/SPES, MasonFlint44/OpenDiPaCo, AIoT-MLSys-Lab/FedRolex, PrimeIntellect-ai/OpenDiloco, PrimeIntellect-ai/prime, and the HeteroFL repository.

### Not covered, or inconclusive

- **No GitHub code search.** The GitHub MCP connector failed to connect ("Authorization header is badly formatted"), and the unauthenticated REST API was used only for repository metadata. An implementation of Chorus-style slicing inside a larger codebase would not have been found.
- **No citation chasing** (Google Scholar, Semantic Scholar, connected-papers) on SPES, FedMoE, MoE-DisCo, FLEX-MoE or DiPaCo. Papers citing SPES are the most likely place for a closer match and should be checked before any public claim.
- **No search of Chinese-language venues** or non-arXiv workshop proceedings.
- **Opened only at abstract level:** DiLoCo, OpenDiLoCo, INTELLECT-1, Streaming DiLoCo, DiLoCo scaling laws, Photon, Decoupled DiLoCo, Covenant-72B, Agora, DeDLOC, SWARM, Petals, Learning@home, BTM, BTX, SMALLTALK LM, DEPT, InclusiveFL, Federated Dropout, IST theory, FedMix, DDOME/FedJETs, FLEx, HFedMoE, FedAlign-MoE, HARC.
- **Not opened at all (secondary mention or not pursued):** EST (2406.06962), "Can Model Merging Improve Aggregation in DiLoCo?" (2607.03011), MaskMoE (2407.09816), FlexMoRE (2602.08818), FedMoE-DA (2411.02115), the IJCAI 2025 server-MoE federated paper, FedSpaLLM, HeterMoE, Hexa-MoE, Lazarus, Moshpit SGD, IOTA (2507.17766), "Incentivizing Permissionless Distributed Learning of LLMs" (2505.21684), SparseLoCo, DisTrO/DeMo, MuLoCo, INTELLECT-2/3, Gensyn's work, FjORD, ScaleFL, NeFL, DepthFL, ResIST, Asynchronous Local-SGD.
- **Not opened:** the anonymous code links for MoE-DisCo and TwIST, and the SPES Hugging Face model cards.
- **Inconclusive details:** SPES per-model token counts (the fetched summary was internally inconsistent; the abstract gives none); whether MoE-DisCo starts from random initialization; whether SPES ever rotates expert ownership; the venue of SDP (an icml.cc 2025 page appeared in search results but was not opened) and of Petals.
- **Tool caveat:** most "read" entries were read through a summarizing fetch tool. Exact quotations in this file come from those summaries, which were asked for verbatim text but may still paraphrase. Re-verify quotations against the PDFs before external use.
