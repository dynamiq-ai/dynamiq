class VectorStoreException(Exception):
    """
    Base exception class for vector store related errors.

    This exception is raised when a general error occurs in the vector store operations.
    """

    pass


class VectorStoreDuplicateDocumentException(Exception):
    """
    Exception raised when attempting to add a duplicate document to the vector store.

    This exception is thrown when a document with the same identifier or content is already present
    in the vector store and an attempt is made to add it again.
    """

    pass


class VectorStoreFilterException(Exception):
    """
    Exception raised when there's an error in filtering operations on the vector store.

    This exception is thrown when an invalid filter is applied or when there's an issue with the
    filtering process in the vector store.
    """

    pass
