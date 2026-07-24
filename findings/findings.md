# RAG Assignment Findings

## What I built

I built a basic local RAG system that answers questions from three university PDFs:

- an AM receiver project;
- a three-stage elevator system;
- notes about computer input/output modules.

The full pipeline is:

**PDFs → extract text → chunk → embed → store in ChromaDB → retrieve top 3 chunks → generate an answer**

I used `pypdf` to extract the text, LangChain's recursive text splitter for chunking, ChromaDB for vector storage and retrieval, and `qwen3.5:2b` for answer generation.

## Initial setup

| Component | Choice |
|---|---|
| Documents | 3 university PDFs |
| Readable pages | 54 |
| Initial chunk size | 1000 characters |
| Initial overlap | 200 characters |
| Initial number of chunks | 108 |
| Initial embedding model | `nomic-embed-text` |
| Vector database | ChromaDB |
| Retrieval | Top 3 by cosine distance |
| Generation model | `qwen3.5:2b` |
| Temperature | 0.2 |

I printed the retrieved chunks before generating the final answer. This made it possible to judge retrieval separately from generation instead of trusting the final answer blindly.

## Questions I tested

1. How does the elevator detect and respond to an overweight condition?
2. How does DMA reduce CPU involvement compared with programmed and interrupt-driven I/O?
3. What factors should be considered when selecting the resistor and capacitor values in the AM demodulator?
4. What motor model was used to move the elevator?

The first three questions had answers in the documents. The fourth was intentionally unsupported.

## Baseline results: `nomic-embed-text` with 1000/200 chunks

### Elevator overweight question

The retriever returned three chunks from `elevator_system.pdf`. They correctly described the potentiometer, the ATD reading, the threshold `0x66`, the buzzer, the blinking red LED, and the movement lockout.

The retrieval was strong, although one chunk started in the middle of a sentence. The answer was grounded in the correct document.

### DMA question

The retriever returned the correct I/O document and found the comparison between programmed I/O, interrupt-driven I/O, and DMA.

The information was split across multiple chunks. The first generated answer also made a technical mistake by connecting cycle stealing to interrupt-driven I/O. Cycle stealing is a DMA mode. This showed that relevant retrieval does not guarantee a perfect final answer.

### AM demodulator question

The retriever found the important rule from the AM receiver document:

- preserve the wanted 0–20 kHz baseband;
- reject the unwanted carrier beginning at 200 kHz.

The correct design condition is:

`20 kHz < fc < 200 kHz`

where:

`fc = 1 / (2πRC)`

The retrieval worked, but one of the top three chunks was only weakly relevant.

### Unsupported motor question

The system retrieved text about elevator motion and control, but the document did not name a motor model. The model correctly admitted that the information was unavailable instead of inventing a motor.

This test was important because semantic retrieval will still return related passages even when the exact answer does not exist.

# Experiment 1: changing the chunk size

I changed only the chunking settings so the comparison would be fair.

## Configuration A

- Chunk size: 1000 characters
- Overlap: 200 characters
- Number of chunks: 108

## Configuration B

- Chunk size: 500 characters
- Overlap: 100 characters
- Number of chunks: 210

## What changed

### Elevator question

The 500-character chunks were more focused and had strong similarity scores. However, the retrieved passages were more fragmented, and one began halfway through a sentence.

### DMA question

The smaller chunks worked well. They separated the key comparison clearly:

- programmed I/O uses constant polling;
- interrupt-driven I/O still makes the CPU move data;
- DMA lets the controller move the data while the CPU mainly performs setup.

The generated answer was cleaner than in the baseline run.

### AM demodulator question

This was the main failure of the smaller chunks.

The retriever returned three chunks from page 2 that mostly repeated the assignment question and described the circuit. It failed to retrieve the nearby page 3 chunk containing the actual cutoff-frequency rule.

Because the answer was not present in the retrieved context, Qwen correctly said there was not enough information.

### Hard question

The motor-model question was still handled correctly. The system did not invent a motor.

## Chunk-size conclusion

The smaller chunks improved precision for narrow questions such as DMA, but they also broke the surrounding context into smaller pieces. In the AM test, this caused the retriever to miss the chunk that contained the actual answer.

For this dataset, **1000 characters with 200 overlap was more reliable overall**. It preserved enough context to answer technical questions that crossed nearby sections or pages.

# Experiment 2: changing the embedding model

For the second experiment, I restored the original chunk settings and changed only the embedding model.

## Fixed settings

- Chunk size: 1000 characters
- Overlap: 200 characters
- Number of chunks: 108
- Top K: 3
- Generator: `qwen3.5:2b`

## Model A: `nomic-embed-text`

- Local Ollama model
- Approximately 274 MB on disk
- 768-dimensional embeddings

It worked well as a lightweight starting model, but some results were less focused or split across chunks.

## Model B: `mxbai-embed-large`

- Local Ollama model
- Approximately 669 MB on disk
- 1024-dimensional embeddings

It produced more focused retrieval overall:

- the elevator question returned complete overload-handling passages;
- the DMA question returned both the comparison and the explanation of how the CPU configures DMA before the controller takes over;
- the AM question returned the page containing the actual cutoff-frequency rule;
- the unsupported motor question was still handled correctly.

## Generation weakness

Even with better retrieval, the AM answer contained mistakes. It called the passive circuit “active-passive,” attached some citations to the wrong chunks, and wrote a malformed final inequality.

The correct information was in the retrieved context, but the generator expressed it incorrectly.

This was one of the most useful findings: **better retrieval improves the model's chances, but it does not remove the need to verify the generated answer.**

## Embedding-model conclusion

For these documents, **`mxbai-embed-large` gave better retrieval than `nomic-embed-text`**. Its passages were generally more complete and directly useful.

The tradeoff is that it is a larger model and uses more storage and memory. I did not formally measure indexing or retrieval time, so I am not claiming that one model was faster.

# Where the system worked well

- It consistently identified the correct PDF for direct questions.
- Semantic retrieval worked even when the question and source used different wording.
- The system correctly refused the unsupported motor-model question.
- Metadata made it easy to inspect the source file and page.
- `mxbai-embed-large` improved the quality of the top retrieved passages.

# Where the system broke

- Character-based chunks sometimes started or ended in the middle of an idea.
- Smaller chunks could miss nearby context needed for a complete answer.
- Top-3 retrieval sometimes included a weakly relevant result.
- Qwen occasionally assigned citations to the wrong retrieved source.
- Qwen could still add an incorrect technical statement even when retrieval was correct.
- A related retrieval result does not prove that the answer exists in the documents.

# Final configuration

My preferred final setup is:

```python
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "mxbai-embed-large"
GENERATION_MODEL = "qwen3.5:2b"
TOP_K = 3
```

# Final conclusion

The project successfully implemented the complete RAG pipeline: ingest, chunk, embed, store, retrieve, and generate.

The system performed well when the answer was clearly present in the documents. The experiments also showed that chunk size and embedding-model choice affect retrieval quality in different ways. Smaller chunks were sometimes more precise, but larger chunks preserved context more reliably. `mxbai-embed-large` was the stronger embedding model for this dataset.

The biggest lesson was that RAG reduces hallucination but does not eliminate it. The retrieved chunks and the final answer both need to be checked. A good RAG system is not only one that produces a convincing answer; it is one that makes the evidence visible enough to verify.
