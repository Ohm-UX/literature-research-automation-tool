"""
Runs the file

Author: Christoph Ruff
"""

from database_embedder import DatabaseEmbedder
from database_inspector import JsonDatabaseInspector


def main():
    # =========================
    # User Configuration
    # =========================
    # File paths for the JSON and Parquet files.
    json_file = "../Arxiv_Dataset/arxiv-metadata-oai-snapshot.json"
    parquet_file = "../Arxiv_Dataset/arxiv_metadata.parquet"
    # Columns to be extracted from the JSON file.
    column_names = ["id", "title", "doi", "abstract", "authors_parsed"]
    # Column name for the unique identifier in the Parquet file.
    id_column = "id"
    # Columns to be merged into a single text input for embedding.
    text_columns = ["title", "abstract"]
    # Model name for the embedding model.
    model_name = "microsoft/harrier-oss-v1-0.6b"
    # Path prefix for saving embeddings and metadata.
    output_prefix = f"data/{model_name.replace('/', '_')}"
    # Query string for searching similar papers in the embedded dataset.
    query = (
        "The evolution of the Earth-Moon system based on the dark matter field "
        "fluid model. "
    )
    # Number of top similar results to retrieve. Set to None to ignore the top_k limit.
    top_k = 5
    # Similarity threshold for filtering results. Set to None to ignore the threshold.
    similarity_threshold = None  # Set to None to ignore the threshold

    print("Start Literature Research Automation Tool!")

    # Create an inspector object for the JSON file
    inspector = JsonDatabaseInspector(json_file)
    inspector.export_to_parquet(output_file=parquet_file, column_names=column_names)

    # columns_name = inspector.get_column_names()
    # columns = inspector.get_columns_by_name(column_names)
    # print("\nColumn names only:")
    # print(columns_name)
    # result = inspector.get_first_n_rows(n=5)
    # print(f"\nFirst {1} rows:\n")
    # print(result)

    # Always close the database connection when finished
    inspector.close()

    # Embed the Parquet dataset
    embedder = DatabaseEmbedder(
        parquet_file=parquet_file,
        output_prefix=output_prefix,
        id_column=id_column,
        text_columns=text_columns,
        model_name=model_name,
        test_mode=True,  # Set to False to embed the entire dataset
    )
    embedder.run()

    # Search for similar papers using the generated embeddings
    embedder.search_embeddings(
        query=query,
        output_file=f"{output_prefix}_similarity_search.json",
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


if __name__ == "__main__":
    main()
