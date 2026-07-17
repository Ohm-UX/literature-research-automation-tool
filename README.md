# Literature Research Automation Tool

This project turns a large collection of paper metadata into a semantic search system. It reads JSON data, exports selected fields to Parquet, creates embeddings from titles and abstracts, stores those embeddings on disk, and lets you find papers that are semantically similar to a natural-language query.

## What the project does

The current workflow consists of three main steps:

1. Inspect and export metadata from a JSON dataset into Parquet.
2. Create semantic embeddings from the paper title and abstract using the Microsoft Harrier embedding model.
3. Search the saved embeddings for semantically similar papers and export the matches to CSV and Parquet.

This is useful for exploring large collections of scientific papers without manually reading every document.

## Repository structure

- [main.py](main.py): entry point that runs the full pipeline.
- [database_inspector.py](database_inspector.py): exports selected fields from a JSON file into Parquet.
- [database_embedder.py](database_embedder.py): loads the Parquet data, creates embeddings, stores them, builds a USearch vector index, and performs similarity search.
- [search_embedding.ipynb](search_embedding.ipynb): interactive notebook version of the embedding search workflow.
- [dataset/](dataset): example input data, including the ArXiv metadata snapshot.
- [output_data/](output_data): generated embedding files, metadata, vector indexes, and search result exports.
- [tests/](tests): basic test coverage for the embedding pipeline.

## Requirements

The project depends on the packages listed in [pyproject.toml](pyproject.toml), including:

- duckdb
- numpy
- pandas
- pyarrow
- torch
- transformers
- sentence-transformers
- usearch
- jupyter

## Installation

From the repository root, install the project and dependencies:

```bash
pip install -e .
```

If you use uv, the equivalent command is:

```bash
uv sync
```

## Quick start

1. Make sure the source dataset exists at [dataset/arxiv-metadata-oai-snapshot.json](dataset/arxiv-metadata-oai-snapshot.json).
2. Review the configuration values at the top of [main.py](main.py), especially:
   - `json_file`
   - `parquet_file`
   - `column_names`
   - `id_column`
   - `text_columns`
   - `model_name`
   - `query`
   - `top_k`
   - `similarity_threshold`
   - `chunk_size`
   - `batch_size`
3. Run the pipeline:

```bash
python main.py
```

The script will:

- create the Parquet export if it does not already exist,
- build embeddings and save them under [output_data/](output_data),
- create a USearch index for fast nearest-neighbor search,
- write similarity search results to CSV and Parquet files.

## Notebook usage

The notebook [search_embedding.ipynb](search_embedding.ipynb) offers an interactive way to:

- load the saved embeddings and vector index,
- run a semantic query against the indexed data,
- export matching rows to CSV and Parquet files.

## Output files

The pipeline writes the following artifacts into [output_data/](output_data):

- embedding arrays as `.npy` files,
- metadata as `.meta.json` files,
- vector indexes as `.usearch.index` files,
- search results as `.csv` and `.parquet` files.

## Notes

- The default configuration uses the model `microsoft/harrier-oss-v1-0.6b`.
- `test_mode` in [main.py](main.py) can be enabled for a smaller smoke test over a limited subset of rows.
- If your dataset is stored at a different location, update the paths in [main.py](main.py) before running the tool.
- Use the notebook for searching queries in the already embedded files. This is a lot faster then always running [main.py](main.py).

## License

This project is intended for research and experimentation. Please verify the licensing terms of any datasets or models you use with it.
