# Comparative Chapter Generator

A robust Python utility that automates the creation of a cohesive, thesis‑ready comparative discussion chapter from multiple scientific PDF articles.

## Key Features

- **PDF Processing**: Recursively scans the `papers/` directory, extracting native text or falling back to OCR via pytesseract when needed.
- **Adaptive Chunking**: Detects available GPU memory (if `torch.cuda` is available) and computes an ideal character chunk size (≈ 80 000 chars per GB), with sensible min/max limits.
- **Summarization Pipeline**:
  1. **Chunking**: Splits each document into balanced pieces, preserving paragraph boundaries.
  2. **Chunk Summaries**: Uses an Ollama LLM (default `gemma3:27b`) to summarize each chunk in cultured Portuguese.
  3. **Article Synthesis**: Merges chunk-level summaries into a single, fluent summary for each paper.
  4. **Comparative Chapter**: Combines all article summaries into a single text chapter, highlighting:
     - Common themes across works
     - Theoretical or methodological conflicts
     - Key contributions of each study
     - Identified gaps and future research directions
- **Plain Text Output**: Produces clean, thesis-ready text (no Markdown), saved to `static/capitulo_comparativo.txt`.

## Advantages

- **GPU-Aware**: Automatically adjusts chunk sizes to available GPU RAM, accelerating LLM calls for large texts.
- **OCR Fallback**: Ensures maximum coverage by automatically applying OCR on scanned or image‑based PDFs.
- **Detailed Logging**: Timestamped logs (via `logging` and custom `log()`) for each processing step, easing debugging and performance monitoring.
- **Modularity**: Easily swap models, override chunk-size logic, or integrate into larger workflows.
- **Lightweight**: Dependencies are minimal and installable via `pip`.

## Dependencies

- Python 3.8+  
- PyMuPDF (`fitz`)  
- Pillow  
- pytesseract  
- ollama  
- torch (for GPU detection)

Install with:
```bash
pip install -r requirements.txt
```

## Usage

1. Create a `papers/` folder in the project root and place your PDF files there (e.g., `article1.pdf`, `article2.pdf`).
2. Run the comparison script:
   ```bash
   python compare_papers.py
   ```
3. Review the generated chapter at `static/capitulo_comparativo.txt`.

## Example

```bash
# Assume papers/article1.pdf and papers/article2.pdf exist
$ python compare_papers.py
2025-04-19 12:00:00 - Processing article1.pdf...
2025-04-19 12:00:05 - GPU detected: 8.0 GB – setting chunk size to 640000 chars.
2025-04-19 12:00:10 - article1.pdf: divided into 3 chunks.
2025-04-19 12:00:15 - Summarizing chunk 1/3...
...
2025-04-19 12:01:20 - Processing article2.pdf...
2025-04-19 12:01:40 - OCR fallback used on page 2 of article2.pdf.
...
2025-04-19 12:03:10 - Generating comparative chapter...
2025-04-19 12:03:30 - Chapter saved at static/capitulo_comparativo.txt
```

## Configuration

- **LLM Model**: Change the `model` argument in `summarize_chunk()` or pass a different model name.
- **Chunk Size**: Override `get_ideal_chunk_size()` or supply `max_chars` directly to `chunk_text()`.
- **Logging Level**: Modify `logging.basicConfig(level=...)` to adjust verbosity.

---

Feel free to adapt this tool to streamline your thesis or dissertation writing process.

