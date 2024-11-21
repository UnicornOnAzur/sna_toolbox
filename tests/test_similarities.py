# -*- coding: utf-8 -*-
"""
@author: UnicornOnAzur

"""


# Third party
import pytest
# Local
import sna_toolbox.src.similarities as similarities


class TestOverlapCoefficient:
    def test_invalid_input(self) -> None:
        """Test for invalid input types."""
        with pytest.raises(TypeError) as excinfo:
            similarities.overlap_coefficient(1, 1)
        assert excinfo.type == TypeError
        assert str(excinfo.value) == "All arguments must be sets!"

    def test_two_empty_sets(self) -> None:
        """Test the overlap coefficient for two empty sets."""
        with pytest.warns() as record:
            result = similarities.overlap_coefficient(set(), set())
        assert result == 0
        assert str(record[0].message) == "Both sets are empty!"

    def test_one_empty_set(self) -> None:
        """Test the overlap coefficient when one set is empty."""
        with pytest.raises(ValueError) as excinfo:
            similarities.overlap_coefficient(set(), {1})
        assert excinfo.type == ValueError
        assert str(excinfo.value) ==\
            "At least one of the sets must be non-empty."

    def test_uneven_types(self) -> None:
        """Test the overlap coefficient when one set is empty."""
        with pytest.raises(TypeError) as excinfo:
            similarities.overlap_coefficient({1, 2, 3}, {"f"})
        assert excinfo.type == TypeError
        assert str(excinfo.value) ==\
            "Elements in the sets must be of the same type."
        result = similarities.overlap_coefficient({1, 2, 3}, {.5})
        assert result == 0

    def test_no_match(self) -> None:
        """Test the overlap coefficient for sets with no common elements."""
        result = similarities.overlap_coefficient(set(range(10)),
                                                  set(range(10, 15)))
        assert result == 0

    def test_full_match(self) -> None:
        """Test the overlap coefficient for identical sets."""
        result = similarities.overlap_coefficient(set(range(10)),
                                                  set(range(10)))
        assert result == 1

    def test_example(self) -> None:
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


class TestJaccardSimilarity:
    def test_invalid_input(self) -> None:
        """Test for invalid input types."""
        with pytest.raises(TypeError) as excinfo:
            similarities.jaccard_similarity(1, 1)
        assert excinfo.type == TypeError
        assert str(excinfo.value) == "All arguments must be sets!"

    def test_two_empty_sets(self) -> None:
        """Test the Jaccard similarity for two empty sets."""
        with pytest.warns() as record:
            result = similarities.jaccard_similarity(set(), set())
        assert result == 0
        assert str(record[0].message) == "Both sets are empty!"

    def test_one_empty_set(self) -> None:
        """Test the Jaccard similarity when one set is empty."""
        result = similarities.jaccard_similarity(set(), {1})
        assert result == 0

    def test_uneven_types(self) -> None:
        """Test the Jaccard similarity with sets of uneven types."""
        with pytest.raises(TypeError) as excinfo:
            similarities.jaccard_similarity({1, 2, 3}, {"f"})
        assert excinfo.type == TypeError
        assert str(excinfo.value) ==\
            "Elements in the sets must be of the same type."
        result = similarities.jaccard_similarity({1, 2, 3}, {.5})
        assert result == 0

    def test_no_match(self) -> None:
        """Test the Jaccard similarity for sets with no common elements."""
        result = similarities.jaccard_similarity(set(range(10)),
                                                 set(range(10, 15)))
        assert result == 0

    def test_full_match(self) -> None:
        """Test the Jaccard similarity for identical sets."""
        result = similarities.jaccard_similarity(set(range(10)),
                                                 set(range(10)))
        assert result == 1

    def test_example(self) -> None:
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


class TestDiceSørensenCoefficient:
    def test_invalid_input(self) -> None:
        """Test for invalid input types."""
        with pytest.raises(TypeError) as excinfo:
            similarities.dice_sørensen_coefficient(1, 1)
        assert excinfo.type == TypeError
        assert str(excinfo.value) == "All arguments must be sets!"

    def test_two_empty_sets(self) -> None:
        """Test the Dice-Sørensen coefficient for two empty sets."""
        with pytest.warns() as record:
            result = similarities.dice_sørensen_coefficient(set(), set())
        assert result == 0
        assert str(record[0].message) == "Both sets are empty!"

    def test_one_empty_set(self) -> None:
        """Test the Dice-Sørensen coefficient when one set is empty."""
        result = similarities.dice_sørensen_coefficient(set(), {1})
        assert result == 0

    def test_uneven_types(self) -> None:
        """Test the Dice-Sørensen coefficient with sets of uneven types."""
        with pytest.raises(TypeError) as excinfo:
            similarities.dice_sørensen_coefficient({1, 2, 3}, {"f"})
        assert excinfo.type == TypeError
        assert str(excinfo.value) ==\
            "Elements in the sets must be of the same type."
        result = similarities.dice_sørensen_coefficient({1, 2, 3}, {.5})
        assert result == 0

    def test_no_match(self) -> None:
        """Test the Dice-Sørensen coefficient for sets with no common
        elements."""
        result = similarities.dice_sørensen_coefficient(set(range(10)),
                                                        set(range(10, 15)))
        assert result == 0

    def test_full_match(self) -> None:
        """Test the Dice-Sørensen coefficient for identical sets."""
        result = similarities.dice_sørensen_coefficient(set(range(10)),
                                                        set(range(10)))
        assert result == 1

    def test_example(self) -> None:
        """Test the Dice-Sørensen coefficient with example sets.
        https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient
        """
        # Example from Wikipedia
        result = similarities.dice_sørensen_coefficient(
             {"ni", "ig", "gh", "ht"},
             {"na", "ac", "ch", "ht"})
        assert result == 0.25


class TestCosineCoefficient:
    def test_invalid_input(self) -> None:
        """Test for invalid input types."""
        with pytest.raises(TypeError) as excinfo:
            similarities.cosine_similarity(1, 1)
        assert excinfo.type == TypeError
        assert str(excinfo.value) == "All arguments must be sets!"

    def test_two_empty_sets(self) -> None:
        """Test the cosine similarity for two empty sets."""
        with pytest.warns() as record:
            result = similarities.cosine_similarity(set(), set())
        assert result == 0
        assert str(record[0].message) == "Both sets are empty!"

    def test_one_empty_set(self) -> None:
        """Test the cosine similarity when one set is empty."""
        with pytest.raises(ValueError) as excinfo:
            similarities.cosine_similarity(set(), {1})
        assert excinfo.type == ValueError
        assert str(excinfo.value) ==\
            "At least one of the sets must be non-empty."

    def test_uneven_types(self) -> None:
        """Test the cosine similarity with sets of uneven types."""
        with pytest.raises(TypeError) as excinfo:
            similarities.cosine_similarity({1, 2, 3}, {"f"})
        assert excinfo.type == TypeError
        assert str(excinfo.value) ==\
            "Elements in the sets must be of the same type."
        result = similarities.cosine_similarity({1, 2, 3}, {.5})
        assert result == 0

    def test_no_match(self) -> None:
        """Test the cosine similarity for sets with no common elements."""
        result = similarities.cosine_similarity(set(range(10)),
                                                set(range(10, 15)))
        assert result == 0

    def test_full_match(self) -> None:
        """Test the cosine similarity for identical sets."""
        result = similarities.cosine_similarity(set(range(10)),
                                                set(range(10)))
        assert round(result) == 1

    def test_example(self) -> None:
        """Test the cosine similarity with example sets.
        https://www.learndatasci.com/glossary/cosine-similarity/
        """
        # Example from Fatih Karabiber
        result = similarities.cosine_similarity(
            set('the best data science course'.split(" ")),
            set('data science is popular'.split(" ")))
        assert round(result, 5) == 0.44721


class TestSimpleMatchingCoefficient:
    def test_invalid_input(self) -> None:
        """Test for invalid input types."""
        with pytest.raises(TypeError) as excinfo:
            similarities.simple_matching_coefficient(1, 1)
        assert excinfo.type == TypeError
        assert str(excinfo.value) == "All arguments must be sets!"

    def test_two_empty_sets(self) -> None:
        """Test the simple matching coefficient for two empty sets."""
        with pytest.warns() as record:
            result = similarities.simple_matching_coefficient(set(), set())
        assert result == 0
        assert str(record[0].message) == "Both sets are empty!"

    def test_one_empty_set(self) -> None:
        """Test the simple matching coefficient when one set is empty."""
        with pytest.raises(ValueError) as excinfo:
            similarities.simple_matching_coefficient(set(), {1})
        assert excinfo.type == ValueError
        assert str(excinfo.value) ==\
            "At least one of the sets must be non-empty."

    def test_uneven_types(self) -> None:
        """Test the simple matching coefficient with sets of uneven types."""
        with pytest.raises(TypeError) as excinfo:
            similarities.simple_matching_coefficient({1, 2, 3}, {"f"})
        assert excinfo.type == TypeError
        assert str(excinfo.value) ==\
            "Elements in the sets must be of the same type."
        result = similarities.simple_matching_coefficient({1, 2, 3}, {.5})
        assert result == 0

    def test_no_match(self) -> None:
        """Test the simple matching coefficient for sets with no common
        elements."""
        result = similarities.simple_matching_coefficient(set(range(10)),
                                                          set(range(10, 15)))
        assert result == 0

    def test_full_match(self) -> None:
        """Test the simple matching coefficient for identical sets."""
        result = similarities.simple_matching_coefficient(set(range(10)),
                                                          set(range(10)))
        assert result == 1

    def test_example(self) -> None:
        """Test the simple matching coefficient with example sets.
        https://people.revoledu.com/kardi/tutorial/Similarity/SimpleMatching.html  # noqa: E501
        """
        # Example from Kardi Teknomo
        result = similarities.simple_matching_coefficient(
            {"a", "b", "c", "d"},
            {"a_", "b", "c_", "d_"})
        assert result == 0.25


if __name__ == "__main__":
    pass
