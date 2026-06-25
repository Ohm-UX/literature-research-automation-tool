# Literature Research Automation Tool

This project automates part of the literature research workflow by turning a large JSON dataset of papers into a searchable semantic index. It reads paper metadata, exports selected fields to Parquet, creates text embeddings from titles and abstracts, and then finds papers similar to a natural-language query.

## What the program does

The pipeline consists of three main steps:

1. Extract selected columns from a JSON dataset and export them to Parquet.
2. Build semantic embeddings from the paper title and abstract using a sentence-transformer model.
3. Search the embedded dataset for papers similar to a user-provided query.

This is useful for quickly exploring large collections of scientific papers without manually reading everything.

## Project structure

- main.py: entry point that runs the full pipeline.
- database_inspector.py: loads a JSON file and exports selected columns to Parquet.
- database_embedder.py: creates embeddings, saves them to disk, and performs similarity search.
- data/: generated outputs such as embeddings, metadata, and similarity results.

## Requirements

The project uses Python and the dependencies listed in pyproject.toml, including:

- duckdb
- numpy
- pandas
- pyarrow
- torch
- transformers
- sentence-transformers
- tqdm

## Installation

From the repository root, install the project and dependencies:

```bash
pip install -e .
```

If you use uv, this also works:

```bash
uv sync
```

## Configuration

The main settings are defined in main.py. Before running the tool, review the following values:

- json_file: path to the source JSON dataset
- parquet_file: output path for the Parquet export
- column_names: fields to extract from the JSON file
- id_column: identifier column used for matching results
- text_columns: fields used as input text for embedding
- model_name: embedding model to use
- query: natural-language search query
- top_k: number of results to return
- similarity_threshold: optional minimum similarity score

For a quick smoke test, the code sets test_mode=True, which embeds only a teh first batch. Set it to False to process the full dataset.

## Usage

Run the program from the repository root:

```bash
python main.py
```

The script will:

- create or update the Parquet file,
- generate embeddings and save them under data/,
- write similarity search results to CSV and Parquet files.

## Output files

The pipeline produces:

- Parquet export of selected paper metadata
- Embedding file in .npy format
- Metadata file in .meta.json format
- Similarity search results in CSV and Parquet formats

## Notes

The repository expects a local dataset path for the input JSON file. If your dataset is stored elsewhere, update the paths in main.py before running the script.

## License

This project is intended for research and experimentation. Please verify the licensing terms of any datasets or models you use with it.
