class GenericTokenizer:
    def tokenize(self, contents: list[str] | str):
        """Tokenizes a string of contents.

        Args:
            contents (typing.Union[typing.List[str], str]): Tokenizes either a list of UTF-8 strings or a single UTF-8 string.

        Raises:
            NotImplementedError: The method is not implemented for the current tokenizer.

        Notes:
            When implementing, take note that this functions will be called more than once across multiple processes / threads. As such, ensure that this function is threadsafe.
        """
        raise NotImplementedError
