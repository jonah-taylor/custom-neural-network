# Custom Neural Network

**A custom neural network made by Jonah Taylor**

This custom neural network was for the purpose of demystifying machine learning. This project was built over Fall 2024. The project was largely inspired by 3Blue1Brown's general view youtube series on machine learning.
---

## Build and run the program

```
Upload PDF
    │
    ▼
S3: protocols/{hash}.pdf
    │
    ▼  (S3 trigger)
Textract Lambda
    │  Kicks off async AWS Textract job with 15 targeted clinical queries
    │
    ▼  (SNS notification on completion)
Extract Lambda
    │  Structures extracted text into clinical sections (Design, Endpoints, Stats, etc.)
    │
    ▼
S3: textract_results/{hash}.json
    │
    ▼  (S3 trigger → AWS Batch)
Embeddings Job
    │  Encodes each section using BGE-M3 dense embeddings (GPU)
    │
    ▼
S3: embeddings/{hash}.pkl
    │
    ▼  (S3 trigger → AWS Batch)
Summary Job
    │  Hybrid retrieval (BGE-M3 + BM25) → cross-encoder reranking → Qwen-2.5-14B
    │  Generates CONSORT-aligned summary sections
    │
    ▼
S3: output/{hash}_summary.pdf
    │
    ▼
User downloads summary from frontend
```

---

## Developer Notes

### Layers

This neural network includes dense layers and convolutional layers. Convolutional layers are clunky because each convo-layer only has one convolution.
