"""
Embed a Parquet dataset using microsoft/harrier-oss-v1-0.6b.

This script:
1. Reads a Parquet file.
2. Joins selected text columns into one text per row.
3. Creates embeddings with SentenceTransformer.
4. Saves embeddings as .npy.
5. Saves metadata as .meta.json.

Authors: Christoph Ruff, ChatGPT
"""

import json
from pathlib import Path

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer


class DatabaseEmbedder:
    """Embed a Parquet dataset with a certain model."""

    def __init__(
        self,
        parquet_file: str,
        output_prefix: str = "data/model",
        id_column: str = "id",
        text_columns: list[str] | None = None,
        model_name: str = "microsoft/harrier-oss-v1-0.6b",
        batch_size: int = 8,
        test_mode: bool = False,
    ):
        self.parquet_file = Path(parquet_file)
        self.id_column = id_column

        # Use title and abstract as default text input because these usually
        # contain the most useful semantic information for literature search.
        self.text_columns = text_columns or ["title", "abstract"]
        self.model_name = model_name
        self.batch_size = batch_size

        # If test_mode is True, only the first batch is embedded.
        # This is useful to check whether the model, files, and dependencies work
        # before embedding the whole dataset.
        self.test_mode = test_mode

        self.model = SentenceTransformer(
            self.model_name,
            model_kwargs={"dtype": "auto"},
        )

        output_prefix_path = Path(output_prefix)

        # Create parent folders if they do not exist
        output_prefix_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = output_prefix_path.with_name(
            f"{output_prefix_path.stem}_embeddings.npy"
        )
        self.metadata_path = output_prefix_path.with_name(
            f"{output_prefix_path.stem}_metadata.meta.json"
        )

    def _read_rows(self, limit: int | None = None) -> list[tuple]:
        """Read selected rows from the Parquet file using DuckDB."""

        selected_columns = [self.id_column, *self.text_columns]
        columns_sql = ", ".join(f'"{column}"' for column in selected_columns)
        limit_sql = ""

        if limit is not None:
            limit_sql = f"LIMIT {limit}"

        query = f"""
            SELECT {columns_sql}
            FROM read_parquet('{self.parquet_file.as_posix()}')
            {limit_sql}
        """

        return duckdb.sql(query).fetchall()

    def _build_ids_and_texts(self, rows: list[tuple]) -> tuple[list[str], list[str]]:
        """Transform raw Parquet rows into IDs and combined text values."""

        selected_columns = [self.id_column, *self.text_columns]
        ids = []
        texts = []

        for row in rows:
            row_values = dict(zip(selected_columns, row, strict=True))

            ids.append(str(row_values[self.id_column]))

            parts = []
            for column in self.text_columns:
                value = row_values.get(column)

                if value is not None:
                    # Remove excessive whitespace and newlines from the text
                    value = " ".join(str(value).split())

                    if value:
                        parts.append(value)

            texts.append(". ".join(parts))

        return ids, texts

    def _read_texts(self) -> tuple[list[str], list[str]]:
        """Read IDs and combined text fields from the Parquet file."""

        if self.test_mode:
            print(f"Test mode active: reading only {self.batch_size} rows.")
            rows = self._read_rows(limit=self.batch_size)
        else:
            rows = self._read_rows()

        return self._build_ids_and_texts(rows)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Create normalized embeddings for a list of texts."""

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        print(f"Embedding matrix shape: {embeddings.shape}")

        return np.asarray(embeddings, dtype=np.float32)

    def _save_outputs(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: np.ndarray,
    ) -> None:
        """Save embeddings and metadata."""

        np.save(self.embeddings_path, embeddings)

        metadata = {
            "ids": ids,
            "texts": texts,
            "model": self.model_name,
            "text_columns": self.text_columns,
        }

        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False)

        print(f"Saved embeddings to: {self.embeddings_path}")
        print(f"Saved metadata to: {self.metadata_path}\n")

    def run(self) -> None:
        """Run the full embedding pipeline."""

        print("Reading Parquet file...")
        ids, texts = self._read_texts()

        print(f"Embedding {len(texts)} texts with {self.model_name}...")
        embeddings = self._embed_texts(texts)

        self._save_outputs(ids, texts, embeddings)

    def _save_matching_results(
        self,
        results: list[dict],
        output_parquet_path: Path,
        output_csv_path: Path,
    ) -> None:
        """Save full matching rows to Parquet and CSV using DuckDB."""

        con = duckdb.connect()

        try:
            con.execute(
                """
                CREATE TEMP TABLE search_results (
                    id VARCHAR,
                    similarity_score DOUBLE
                )
                """
            )

            con.executemany(
                """
                INSERT INTO search_results VALUES (?, ?)
                """,
                [(result["id"], result["similarity_score"]) for result in results],
            )

            # Add after p.* to remove \n and \r characters from title and abstract
            # columns
            # REPLACE(REPLACE(CAST(title AS VARCHAR),
            # chr(13), ' '), chr(10), ' ') AS title,
            # REPLACE(REPLACE(CAST(abstract AS VARCHAR),
            # chr(13), ' '), chr(10), ' ') AS abstract,
            query = f"""
                SELECT
                    p.*,
                    s.similarity_score
                FROM read_parquet('{self.parquet_file.as_posix()}') AS p
                JOIN search_results AS s
                    ON CAST(p.{self.id_column} AS VARCHAR) = s.id
                ORDER BY s.similarity_score DESC
            """

            con.execute(
                f"COPY ({query}) TO '{output_parquet_path.as_posix()}' (FORMAT PARQUET)"
            )
            con.execute(
                f"""
                COPY (
                    {query}
                )
                TO '{output_csv_path.as_posix()}'
                (
                    FORMAT CSV,
                    HEADER TRUE,
                    DELIMITER ';'
                )
                """
            )
        finally:
            con.close()

        print(f"Saved full matching rows to {output_parquet_path}")
        print(f"Saved full matching rows to {output_csv_path}\n")

    def search_embeddings(
        self,
        query: str,
        output_file: str,
        top_k: int | None = 5,
        similarity_threshold: float | None = None,
    ) -> None:
        """
        Search saved embeddings with a text query using cosine similarity and saves
        the results in a parquet and csv file with the matching parquet rows.
        """
        print(f"Start searching for similar results for query:\n {query}\n")
        embeddings = np.load(self.embeddings_path)
        rows = self._read_rows()
        ids, texts = self._build_ids_and_texts(rows)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        scores = embeddings @ query_embedding
        if similarity_threshold is not None:
            indices = np.where(scores >= similarity_threshold)[0]
            indices = indices[np.argsort(scores[indices])[::-1]]

            if top_k is not None:
                indices = indices[:top_k]
        else:
            indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in indices:
            results.append(
                {
                    "similarity_score": float(scores[index]),
                    "id": ids[index],
                    "text": texts[index],
                }
            )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_csv_path = output_path.with_suffix(".csv")
        output_parquet_path = output_path.with_suffix(".parquet")
        self._save_matching_results(results, output_parquet_path, output_csv_path)

        print("Finished searching for similar results.\n")
