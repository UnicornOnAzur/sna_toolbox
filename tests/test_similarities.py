# -*- coding: utf-8 -*-
"""
@author: UnicornOnAzur

"""

# Standard library
import typing
# Third party
import pytest
# Local
import sna_toolbox.src.similarities as similarities


def _test_no_match(method: typing.Callable[[typing.Set[typing.Any],
                                            typing.Set[typing.Any]],
                                           float]) -> None:
    """Test case for no match between two sets."""
    result = method(set(range(10)), set(range(10, 15)))
    assert result == 0


def _test_full_match(method: typing.Callable[[typing.Set[typing.Any],
                                              typing.Set[typing.Any]],
                                             float]) -> None:
    """Test case for full match between two sets."""
    result = method(set(range(10)), set(range(10)))
    assert round(result) == 1


class TestInputValidation:

    def test_invalid_option(self) -> None:
        """Test for invalid option in input validation."""
        with pytest.raises(
                ValueError,
                match="The provided option is incorrect; it can only be 'one'"
                ):
            @similarities.validate_input("two")
            def func() -> None:
                pass

    def test_invalid_input(self) -> None:
        """Test for invalid input types."""
        @similarities.validate_input()
        def func() -> None:
            pass
        with pytest.raises(TypeError,
                           match="All arguments must be sets!"):
            func(1, 1)
        with pytest.raises(TypeError,
                           match="All arguments must be sets!"):
            func({1}, 1)

    def test_one_empty_sets(self) -> None:
        """Test for one empty set."""
        @similarities.validate_input()
        def func(*args) -> int:
            return 0
        result = func({1}, set())
        assert result == 0

        @similarities.validate_input("one")
        def func() -> None:
            pass
        with pytest.warns() as record:
            result = func({1}, set())
        assert result is None
        assert str(record[0].message) ==\
            "At least one of the sets must be non-empty."

    def test_two_empty_sets(self) -> None:
        """Test for two empty sets."""
        @similarities.validate_input()
        def func() -> None:
            pass
        with pytest.warns() as record:
            result = func(set(), set())
        assert result == 0
        assert str(record[0].message) == "Both sets are empty!"

        @similarities.validate_input("one")
        def func() -> None:
            pass
        with pytest.warns() as record:
            result = func(set(), set())
        assert result is None
        assert str(record[0].message) ==\
            "At least one of the sets must be non-empty."

    def test_uneven_types(self) -> None:
        """Test for not the same types in sets."""
        @similarities.validate_input()
        def func(*args) -> None:
            return 0
        with pytest.raises(
                TypeError,
                match="Elements in the sets must be of the same type."):
            func({1, 2, 3}, {"f"})
        result = func({1, 2, 3}, {.5})
        assert result == 0

    def test_invalid_total_range(self) -> None:
        """Test for invalid total range in set inputs."""
        @similarities.validate_input()
        def func(*args) -> None:
            pass
        with pytest.warns() as record:
            result = func({1}, {2}, {3})
        assert result is None
        assert str(record[0].message) ==\
            "The total range provided is not a superset of the other two sets"


class TestOverlapCoefficient:

    def test_no_match(self) -> None:
        """Test for no match in overlap coefficient."""
        _test_no_match(similarities.overlap_coefficient)

    def test_full_match(self) -> None:
        """Test for full match in overlap coefficient."""
        _test_full_match(similarities.overlap_coefficient)

    def test_example_overlap(self) -> None:
        """Test the overlap coefficient with example sets.
        https://developer.nvidia.com/blog/similarity-in-graphs-jaccard-versus-the-overlap-coefficient/  # noqa: E501
        """
        # Example from Nefi Alcron
        result = similarities.overlap_coefficient({2, 3, 4, 5},
                                                  {1, 3, 4, 5})
        assert result == 0.75
        # Example from Nefi Alcron
        results = []
        for a, b in zip(range(100, 151, 10), range(0, 51, 10)):
            seta = set(range(a))
            setb = set(range(50+b, 150))
            results.append(round(similarities.overlap_coefficient(seta,
                                                                  setb),
                                 3)
                           )
        assert results == [0.5, 0.556, 0.625, 0.714, 0.833, 1]


class TestJaccardCoeffienct:

    def test_no_match(self) -> None:
        """Test for no match in jaccard similarity."""
        _test_no_match(similarities.jaccard_similarity)

    def test_full_match(self) -> None:
        """Test for full match in jaccard similarity."""
        _test_full_match(similarities.jaccard_similarity)

    def test_example_jaccard(self) -> None:
        """Test the Jaccard similarity with example sets.
        https://www.statology.org/jaccard-similarity/
        https://developer.nvidia.com/blog/similarity-in-graphs-jaccard-versus-the-overlap-coefficient/  # noqa: E501
        https://www.learndatasci.com/glossary/jaccard-similarity/
        https://people.revoledu.com/kardi/tutorial/Similarity/Jaccard.html
        """
        # Example from Zach Bobbitt
        result = similarities.jaccard_similarity({0, 1, 2, 5, 6, 8, 9},
                                                 {0, 2, 3, 4, 5, 7, 9})
        assert result == 0.4
        # Example from Nefi Alcron
        result = similarities.jaccard_similarity({2, 3, 4, 5},
                                                 {1, 3, 4, 5})
        assert result == 0.6
        # Example from Fatih Karabiber
        result = similarities.jaccard_similarity({0, 1, 2, 5, 6},
                                                 {0, 2, 3, 4, 5, 7, 9})
        assert round(result, 2) == 0.33
        # Example from Nefi Alcron
        results = []
        for a, b in zip(range(100, 151, 10), range(0, 51, 10)):
            seta = set(range(a))
            setb = set(range(50+b, 150))
            results.append(round(similarities.jaccard_similarity(seta,
                                                                 setb),
                                 3)
                           )
        assert results == [0.333] * 6
        # Example from Kardi Teknomo
        result = similarities.jaccard_similarity({7, 3, 2, 4, 1},
                                                 {4, 1, 9, 7, 5})
        assert round(result, 3) == 0.429


class TestDiceSorensen:

    def test_no_match(self) -> None:
        """Test for no match in dice sørensen coefficient."""
        _test_no_match(similarities.dice_sørensen_coefficient)

    def test_full_match(self) -> None:
        """Test for full match in dice sørensen coefficient."""
        _test_full_match(similarities.dice_sørensen_coefficient)

    def test_example_dice(self) -> None:
        """Test the Dice-Sørensen coefficient with example sets.
        https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient
        """
        # Example from Wikipedia
        result = similarities.dice_sørensen_coefficient(
             {"ni", "ig", "gh", "ht"},
             {"na", "ac", "ch", "ht"})
        assert result == 0.25


class TestCosineSimilarity:

    def test_no_match(self) -> None:
        """Test for no match in cosine similarity."""
        _test_no_match(similarities.cosine_similarity)

    def test_full_match(self) -> None:
        """Test for full match in cosine similarity."""
        _test_full_match(similarities.cosine_similarity)

    def test_example_cosine(self) -> None:
        """Test the cosine similarity with example sets.
        https://www.learndatasci.com/glossary/cosine-similarity/
        """
        # Example from Fatih Karabiber
        result = similarities.cosine_similarity(
            set('the best data science course'.split(" ")),
            set('data science is popular'.split(" ")))
        assert round(result, 5) == 0.44721


class TestSMC:

    def test_no_match(self):
        """Test for no match in simple matching coefficient."""
        _test_no_match(similarities.simple_matching_coefficient)

    def test_full_match(self):
        """Test for full match in simple matching coefficient."""
        _test_full_match(similarities.simple_matching_coefficient)

    def test_example_smc(self) -> None:
        """Test the simple matching coefficient with example sets.
        https://people.revoledu.com/kardi/tutorial/Similarity/SimpleMatching.html  # noqa: E501
        """
        # Example from Kardi Teknomo
        result = similarities.simple_matching_coefficient(
            {"a", "b", "c", "d"},
            {"b"})
        assert result == 0.25


class TestHammingDistance:

    def test_full_overlap(self):
        """Test for full overlap meaning no score on hamming distance."""
        result = similarities.hamming_distance(set(range(10)), set(range(10)))
        assert round(result) == 0

    def test_no_overlap(self):
        """Test for no match in hamming coefficient meaning no match."""
        result = similarities.hamming_distance(set(range(10)), set(range(10, 15)))
        assert result == 15


class TestHammingCoefficient:

    def test_no_match(self):
        """Test for full match in hamming coefficient meaning no match."""
        result = similarities.hamming_coefficient(set(range(10)), set(range(10)))
        assert round(result) == 0

    def test_full_match(self):
        """Test for no match in hamming coefficient meaning no match."""
        result = similarities.hamming_coefficient(set(range(10)), set(range(10, 15)))
        assert result == 1

    def test_example_hamming(self) -> None:
        """
        Test the Hamming coefficient calculation for various sets.

        This function tests the similarities.hamming_coefficient method
        by asserting the expected results for different sets of integers.
        It checks the Hamming coefficient for two pairs of sets and ensures
        that the output matches the expected values.
        """
        # Test case 1
        set_a = {1, 2, 3, 4}
        set_b = {2, 3, 4, 5, 6}
        result = similarities.hamming_coefficient(set_a, set_b)
        assert result == 0.5, f"Expected 0.5 but got {result}"

        # Test case 2
        set_c = {1, 2, 3}
        set_d = {3, 4}
        set_e = {1, 2, 3, 4, 5, 6}
        result = similarities.hamming_coefficient(set_c, set_d, set_e)
        assert result == 0.5, f"Expected 0.5 but got {result}"

        # Example from Kardi Teknomo
        result = similarities.hamming_coefficient(
            {"a", "b", "c", "d"},
            {"b"})
        assert result == 0.75


if __name__ == "__main__":
    pass
