// Citation registry for operating-point provenance.
// Operating points reference these by key in their `sources` field.

export interface Source {
  title: string
  url: string
}

export const SOURCES = {
  'arxiv-2501-12084': {
    title: 'Dissecting the NVIDIA Hopper Architecture through Microbenchmarking',
    url: 'https://arxiv.org/abs/2501.12084'
  },
  'arxiv-2402-13499': {
    title: 'Benchmarking and Dissecting the NVIDIA Hopper GPU Architecture',
    url: 'https://arxiv.org/abs/2402.13499'
  },
  'arxiv-2510-27583': {
    title: 'AMD MI300X GPU Performance Analysis',
    url: 'https://arxiv.org/abs/2510.27583'
  },
  'arxiv-2512-02189': {
    title: "Microbenchmarking NVIDIA's Blackwell Architecture",
    url: 'https://arxiv.org/abs/2512.02189'
  },
  'arxiv-2502-05317': {
    title: 'Apple vs. Oranges: Evaluating Apple Silicon M-Series SoCs for HPC',
    url: 'https://arxiv.org/abs/2502.05317'
  },
  'mamf-finder': {
    title: 'stas00/ml-engineering — mamf-finder community table',
    url: 'https://github.com/stas00/ml-engineering/tree/master/compute/accelerator/benchmarks'
  },
  'nvbandwidth': {
    title: 'NVIDIA nvbandwidth',
    url: 'https://github.com/NVIDIA/nvbandwidth'
  },
  'amd-rocm-mafs': {
    title: 'AMD ROCm — Measuring Max-Achievable FLOPs',
    url: 'https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html'
  },
  'nvidia-cublas-12-0': {
    title: 'NVIDIA cuBLAS 12.0 Performance Blog',
    url: 'https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/'
  },
  // === Interconnect sources ===
  'nvidia-nvlink': {
    title: 'NVIDIA NVLink and NVLink Switch product page',
    url: 'https://www.nvidia.com/en-us/data-center/nvlink/'
  },
  'amd-cdna3-whitepaper': {
    title: 'AMD CDNA 3 Architecture White Paper',
    url: 'https://www.amd.com/system/files/documents/amd-cdna-3-white-paper.pdf'
  },
  'amd-mi350x-datasheet': {
    title: 'AMD Instinct MI350X GPU Datasheet',
    url: 'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-gpu-brochure.pdf'
  },
  'amd-mi355x-datasheet': {
    title: 'AMD Instinct MI355X GPU Datasheet',
    url: 'https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf'
  },
  'google-tpu-v5p-docs': {
    title: 'Google Cloud — TPU v5p system architecture',
    url: 'https://cloud.google.com/tpu/docs/v5p'
  },
  'google-tpu-v6e-docs': {
    title: 'Google Cloud — TPU v6e (Trillium) system architecture',
    url: 'https://cloud.google.com/tpu/docs/v6e'
  },
  // Architecture papers (schematic citations)
  'arxiv-2305-13245': {
    title: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints',
    url: 'https://arxiv.org/abs/2305.13245'
  },
  'arxiv-2310-06825': {
    title: 'Mistral 7B',
    url: 'https://arxiv.org/abs/2310.06825'
  },
  'arxiv-2002-05202': {
    title: 'GLU Variants Improve Transformer',
    url: 'https://arxiv.org/abs/2002.05202'
  },
  'arxiv-2503-19786': {
    title: 'Gemma 3 Technical Report',
    url: 'https://arxiv.org/abs/2503.19786'
  },
  'arxiv-2411-19146': {
    title: 'Puzzle: Distillation-Based NAS for Inference-Optimized LLMs',
    url: 'https://arxiv.org/abs/2411.19146'
  },
  'arxiv-2405-04434': {
    title: 'DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model',
    url: 'https://arxiv.org/abs/2405.04434'
  },
  // No arXiv listing for the DSA report; the official repo carries it.
  'deepseek-v3-2-exp': {
    title: 'DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention',
    url: 'https://github.com/deepseek-ai/DeepSeek-V3.2-Exp'
  },
  'arxiv-2606-13392': {
    title: 'MiniMax Sparse Attention',
    url: 'https://arxiv.org/abs/2606.13392'
  },
  'deepseek-v4-report': {
    title: 'DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence',
    url: 'https://arxiv.org/abs/2606.19348'
  },
  'arxiv-2510-26692': {
    title: 'Kimi Linear: An Expressive, Efficient Attention Architecture',
    url: 'https://arxiv.org/abs/2510.26692'
  },
  'arxiv-2412-06464': {
    title: 'Gated Delta Networks: Improving Mamba2 with Delta Rule',
    url: 'https://arxiv.org/abs/2412.06464'
  },
  'arxiv-2405-21060': {
    title: 'Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality',
    url: 'https://arxiv.org/abs/2405.21060'
  },
  'arxiv-2504-03624': {
    title: 'Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models',
    url: 'https://arxiv.org/abs/2504.03624'
  },
  'arxiv-2101-03961': {
    title: 'Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity',
    url: 'https://arxiv.org/abs/2101.03961'
  },
  'arxiv-2401-06066': {
    title: 'DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models',
    url: 'https://arxiv.org/abs/2401.06066'
  }
} as const satisfies Record<string, Source>

export type SourceKey = keyof typeof SOURCES
