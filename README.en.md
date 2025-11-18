[English](README.md) | [简体中文](README.zh-CN.md)

# RAG Challenge Winner Solution - No GPU Required

## Acknowledgments & Project Description

### Project Background

This project originated from the need to learn RAG (Retrieval-Augmented Generation) technology. When we discovered this excellent RAG Challenge winning solution created by [Ilya Rice (Ilya Ryabov)](https://github.com/IlyaRice), we realized that the original project had high hardware requirements, particularly requiring a GPU for PDF parsing (the original author used an RTX 4090).

**To enable more learners without high-performance GPUs to run and study this excellent RAG project, we made necessary modifications:**

- **Replaced PDF Parsing Solution**: Changed from the original Mockling local parsing (requires GPU) to using [Mineru API](https://www.mineru.com/), allowing PDF parsing to be completed via API without requiring a local GPU.
- **Adapted Vector Model**: Uses Qwen's text-embedding-v3 model for vectorization, eliminating the need for local model deployment.
- **Integrated Qwen API**: Uses Qwen API for Q&A and reranking, further reducing local hardware requirements.

Special thanks to [Ilya Rice (Ilya Ryabov)](https://github.com/IlyaRice) for providing the solid foundation for this project, and to [Mineru](https://www.mineru.com/) for their powerful API services.

**Original Project Resources:**
- Russian: https://habr.com/ru/articles/893356/
- English: https://abdullin.com/ilya/how-to-build-best-rag/

---

## Project Overview

This repository contains the winning solution from the RAG Challenge that was nominated for two awards, modified to run without GPU requirements. The system achieves state-of-the-art results in answering questions about company annual reports, combining the following technologies:

- Custom PDF parsing using Mineru API with format adaptation.
- Vector search with parent document retrieval.
- LLM reranking for improved context relevance.
- Structured output prompting with chain-of-thought reasoning.
- Query routing for multi-company comparisons.
- Integrated Qwen API for question answering.
- Streamlit-based graphical user interface.

### Disclaimer

This is competition code - it's rough around the edges but works. Before you dive in, be aware:

- IBM Watson integration won't work (it was competition-specific).
- The code may have some rough edges and quirky workarounds.
- No tests, minimal error handling - you've been warned.
- You'll need your own API keys for OpenAI/Gemini, Mineru, and Qwen.
- ✨ **No GPU Required**: This modified version uses API services to replace local GPU-intensive processing.

If you're looking for production-ready code, this isn't it. But if you want to explore different RAG techniques and their implementation while learning without GPU requirements - check it out!

## Major Modifications

### PDF Parsing
- Original version uses Mockling for local PDF parsing with high hardware requirements (requires GPU)
- **Improvement**: Changed to use Mineru API with format adaptation for downstream processes

### Vector Model
- **Improvement**: Uses Qwen's text-embedding-v3 model via API calls, no local deployment needed

### LLM Reranking
- **Improvement**: Integrated Qwen model for ranking
- **Note**: When using Qwen model, must explicitly mention "JSON" and the expected JSON format in prompts


### Quick Start

Clone and setup:

```bash
git clone https://github.com/IlyaRice/RAG-Challenge-2.git
cd RAG-Challenge-2
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\Activate.ps1  # Windows (PowerShell)
pip install -e . -r requirements.txt
pip install streamlit # Add dependency for Streamlit UI
```

Rename `env` file to `.env` and add your API keys, including `OPENAI_API_KEY` / `GEMINI_API_KEY`, `MINERU_API_KEY`, and `QWEN_API_KEY`.

### Test Datasets

This repository contains two datasets:

1.  A small test set (in `data/test_set/`) with 5 annual reports and questions.
2.  The complete ERC2 competition dataset (in `data/erc2_set/`) with all competition questions and reports.

Each dataset directory contains its own `README` with specific setup instructions and available files. You can use either dataset to:

-   Study example questions, reports, and system outputs.
-   Run the pipeline from scratch using provided PDFs.
-   Jump directly to specific pipeline stages using preprocessed data.

See the respective `README` files for detailed dataset contents and setup instructions:

-   `data/test_set/README.md` - For the small test dataset.
-   `data/erc2_set/README.md` - For the complete competition dataset.

### Usage

You can run any part of the pipeline by uncommenting the methods you want to run in `src/pipeline.py` and executing:

```bash
python ./src/pipeline.py
```

You can also use `main.py` to run any pipeline stage, but you need to run it from the directory containing the data:

```bash
cd ./data/test_set/
python ../../main.py process-questions --config max_nst_o3m
```

### CLI Commands

Get help on available commands:

```bash
python main.py --help
```

Available commands:

-   `parse-pdfs` - Parse PDF reports using Mineru API with parallel processing (note this command has been modified if you still need local Docling parsing).
-   `serialize-tables` - Process tables from parsed reports.
-   `process-reports` - Run the complete pipeline on parsed reports.
-   `process-questions` - Process questions using specified configuration.

Each command has its own options. For example:

```bash
python main.py parse-pdfs --help
# Note: PDF parsing will be done via Mineru API. Processing time depends on Mineru's speed. In my tests, parsing 3 annual reports took 140 seconds.

python main.py process-reports --config ser_tab
# Process reports using serialized tables configuration
```

### Configuration Options

-   `max_nst_o3m` - Best performance configuration using OpenAI's o3-mini model.
-   `ibm_llama70b` - Alternative using IBM's Llama 70B model.
-   `gemini_thinking` - Full context Q&A using Gemini's massive context window. This is not actually RAG.
-   `qwen_config` - Configuration using Qwen API.

Check `pipeline.py` for more configurations and details.

### Graphical Interface

```bash
streamlit run streamlit_app.py
```

### License

MIT
