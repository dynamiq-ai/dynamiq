import math
from datetime import datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document


class TimeWeightedDocumentRankerInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Parameter to provide list of documents.")


class TimeWeightedDocumentRanker(Node):
    """
    A document ranker node boosting the recent content more.

    This ranker adjusts the initial scores of documents based on their recency. The recency coefficient
    depends on the number of days from today. The initial score is multiplied by the recency coefficient,
    and the documents are re-ranked based on the adjusted score.

    The formula for the adjustment is:
        adjusted_score = score * recency_coefficient

    The recency coefficient is calculated as follows:
        min_coefficient <= coefficient <= 1 (if the same date)

    The coefficient is determined based on the number of days since the content was created.

    An exponential decay formula is used to ensure that the coefficient decreases as the number of days
    increases, but never goes below the specified minimum coefficient.

    The formula used is:
        coefficient = min_coefficient + (1 - min_coefficient) * exp(-3 * days / max_days)

    This ensures that:
        - If days <= 0, the coefficient is 1.0 (no decay).
        - If days >= max_days, the coefficient is min_coefficient (maximum decay).
        - For days in between, the coefficient smoothly transitions from 1.0 to min_coefficient.

    Attributes:
        group (Literal[NodeGroup.RANKERS]): The group this node belongs to.
        name (str): The name of the node.
        top_k (int): The number of top documents to return. Default is 5.
        max_days (int): The maximum number of days to consider for adjustment. Default is 3600.
        min_coefficient (float): The minimum coefficient for score adjustment. Default is 0.9.
        date_field (str): The field name in the metadata containing the date. Default is "date".
        date_format (str): The format of the date string. Default is "%d %B, %Y".

    Example:

        from dynamiq.nodes.rankers import TimeWeightedDocumentRanker
        from dynamiq.types import Document

        # Initialize the ranker
        ranker = TimeWeightedDocumentRanker()

        # Example input data
        input_data = {
            "query": "example query",
            "documents": [
                Document(content="Document content", score=0.8, metadata={"date": "01 January, 2022"}),
                Document(content="Document content", score=0.9, metadata={"date": "01 January, 2021"})
            ]
        }

        # Execute the ranker
        output = ranker.execute(input_data)

        # Output will be a dictionary with ranked documents
        print(output)
    """

    group: Literal[NodeGroup.RANKERS] = NodeGroup.RANKERS
    name: str = "Time Weighted Document Ranker"
    top_k: int = 5
    max_days: int = 3600
    min_coefficient: float = 0.9
    date_field: str = "date"
    date_format: str = "%d %B, %Y"
    input_schema: ClassVar[type[TimeWeightedDocumentRankerInputSchema]] = TimeWeightedDocumentRankerInputSchema

    def execute(
        self, input_data: TimeWeightedDocumentRankerInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the document ranking process.

        Args:
            input_data (TimeWeightedDocumentRankerInputSchema): The input data containing documents and query.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the original query and the ranked documents.

        Example:

            input_data = {
                "documents": [
                    Document(content="Document content", score=0.8, metadata={"date": "01 January, 2022"}),
                    Document(content="Document content", score=0.9, metadata={"date": "01 January, 2021"})
                ]
            }

            output = ranker.execute(input_data)

            # output will be a dictionary with ranked documents
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = input_data.documents

        ranked_documents = self.adjust_similarity_scores(
            documents,
            date_field=self.date_field,
            max_days=self.max_days,
            min_coefficient=self.min_coefficient,
            date_format=self.date_format,
        )

        return {
            "documents": ranked_documents,
        }

    @staticmethod
    def date_to_days(date_string: str, date_format: str = "%d %B, %Y") -> int:
        """
        Convert a date string to the number of days since that date.

        Args:
            date_string (str): Date in the format "dd Month, YYYY"
            date_format (str): The format of the date string (default: "%d %B, %Y").

        Returns:
            int: Number of days since the given date.

        Example:

            days = TimeWeightedDocumentRanker.date_to_days("01 January, 2022")

            # days will be the number of days since 01 January, 2022
        """
        date_object = datetime.strptime(date_string, date_format)
        current_date = datetime.now()
        return (current_date - date_object).days

    @staticmethod
    def days_to_coefficient(
        days: int, max_days: int = 3600, min_coefficient: float = 0.1
    ) -> float:
        """
        Transform number of days into a coefficient for score adjustment.

        The coefficient is calculated based on the number of days since the content was created.

        The function uses an exponential decay formula to ensure that the coefficient decreases
        as the number of days increases, but never goes below the specified minimum coefficient.

        The formula used is:
            coefficient = min_coefficient + (1 - min_coefficient) * exp(-3 * days / max_days)

        This ensures that:
            - If days <= 0, the coefficient is 1.0 (no decay).
            - If days >= max_days, the coefficient is min_coefficient (maximum decay).
            - For days in between, the coefficient smoothly transitions from 1.0 to min_coefficient.

        Args:
            days (int): Number of days since the content was created.
            max_days (int): Maximum number of days to consider (default: 3600, about 12 years).
            min_coefficient (float): Minimum coefficient value (default: 0.1).

        Returns:
            float: Coefficient between min_coefficient and 1.

        Example:

            coefficient = TimeWeightedDocumentRanker.days_to_coefficient(365)

            # coefficient will be a value between 0.1 and 1 based on the number of days
        """
        if days <= 0:
            return 1.0
        elif days >= max_days:
            return min_coefficient
        else:
            return min_coefficient + (1 - min_coefficient) * math.exp(
                -3 * days / max_days
            )

    @staticmethod
    def adjust_similarity_scores(
        candidates: list[Document],
        date_field: str = "date",
        max_days: int = 3600,
        min_coefficient: float = 0.9,
        date_format: str = "%d %B, %Y",
    ) -> list[Document]:
        """
        Adjust cosine similarity scores based on content recency.

        Args:
            candidates (list[Document]): List of Document objects containing candidates with 'score' and date fields.
            date_field (str): Name of the field containing the date string (default: 'date').
            max_days (int): Maximum number of days to consider for adjustment.
            min_coefficient (float): Minimum coefficient for score adjustment.
            date_format (str): The format of the date string (default: "%d %B, %Y").

        Returns:
            list[Document]: List of candidates with adjusted scores, sorted by the new scores.

        Example:

            candidates = [
                Document(content="Document content", score=0.5, metadata={"date": "01 January, 2022"}),
                Document(content="Document content", score=0.5, metadata={"date": "01 January, 2021"})
            ]

            adjusted_candidates = TimeWeightedDocumentRanker.adjust_similarity_scores(candidates)

            # adjusted_candidates will be sorted by adjusted scores
        """
        for candidate in candidates:
            if date := candidate.metadata.get(date_field):
                days = TimeWeightedDocumentRanker.date_to_days(
                    date,
                    date_format=date_format,
                )
                coefficient = TimeWeightedDocumentRanker.days_to_coefficient(
                    days, max_days=max_days, min_coefficient=min_coefficient
                )
                candidate.score = candidate.score * coefficient

        documents = [
            {"score": candidate.score, "document": candidate}
            for candidate in candidates
        ]

        sorted_documents = sorted(documents, key=lambda x: x["score"], reverse=True)

        return [document["document"] for document in sorted_documents]
