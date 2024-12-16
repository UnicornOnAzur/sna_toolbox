# -*- coding: utf-8 -*-
"""
@author: UnicornOnAzur

"""

# Third party
import pytest
# Local
import sna_toolbox.src.similarities as similarities


class TestSetSimilarity:
    def _test_invalid_input(self, method):
        with pytest.raises(TypeError, match="All arguments must be sets!"):
            method(1, 1)

    def _test_two_empty_sets(self, method):
        with pytest.warns() as record:
            result = method(set(), set())
        assert result == 0
        assert str(record[0].message) == "Both sets are empty!"

    def _test_one_empty_set(self, method):
        result = method({1}, set())
        assert result == 0

    def _test_one_empty_set_param_one(self, method):
        with pytest.warns() as record:
            result = method(set(), {1})
        assert result is None
        assert str(record[0].message) ==\
            "At least one of the sets must be non-empty."

    def _test_uneven_types(self, method):
        with pytest.raises(
                TypeError,
                match="Elements in the sets must be of the same type."):
            method({1, 2, 3}, {"f"})
        result = method({1, 2, 3}, {.5})
        assert result == 0

    def _test_no_match(self, method):
        result = method(set(range(10)), set(range(10, 15)))
        assert result == 0

    def _test_full_match(self, method):
        result = method(set(range(10)), set(range(10)))
        assert round(result) == 1

    def _test_example(self, method, example_sets, expected_results):
        for example_set, expected in zip(example_sets, expected_results):
            result = method(*example_set)
            assert result == expected

    def test_overlap_coefficient(self):
        self._test_invalid_input(similarities.overlap_coefficient)
        self._test_two_empty_sets(similarities.overlap_coefficient)
        self._test_one_empty_set_param_one(similarities.overlap_coefficient)
        self._test_uneven_types(similarities.overlap_coefficient)
        self._test_no_match(similarities.overlap_coefficient)
        self._test_full_match(similarities.overlap_coefficient)

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

    def test_jaccard_similarity(self):
        self._test_invalid_input(similarities.jaccard_similarity)
        self._test_two_empty_sets(similarities.jaccard_similarity)
        self._test_one_empty_set(similarities.jaccard_similarity)
        self._test_uneven_types(similarities.jaccard_similarity)
        self._test_no_match(similarities.jaccard_similarity)
        self._test_full_match(similarities.jaccard_similarity)

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

    def test_dice_sørensen_coefficient(self):
        self._test_invalid_input(similarities.dice_sørensen_coefficient)
        self._test_two_empty_sets(similarities.dice_sørensen_coefficient)
        self._test_one_empty_set(similarities.dice_sørensen_coefficient)
        self._test_uneven_types(similarities.dice_sørensen_coefficient)
        self._test_no_match(similarities.dice_sørensen_coefficient)
        self._test_full_match(similarities.dice_sørensen_coefficient)

    def test_example_dice(self) -> None:
        """Test the Dice-Sørensen coefficient with example sets.
        https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient
        """
        # Example from Wikipedia
        result = similarities.dice_sørensen_coefficient(
             {"ni", "ig", "gh", "ht"},
             {"na", "ac", "ch", "ht"})
        assert result == 0.25

    def test_cosine_coefficient(self):
        self._test_invalid_input(similarities.cosine_similarity)
        self._test_two_empty_sets(similarities.cosine_similarity)
        self._test_one_empty_set_param_one(similarities.cosine_similarity)
        self._test_uneven_types(similarities.cosine_similarity)
        self._test_no_match(similarities.cosine_similarity)
        self._test_full_match(similarities.cosine_similarity)

    def test_example_cosine(self) -> None:
        """Test the cosine similarity with example sets.
        https://www.learndatasci.com/glossary/cosine-similarity/
        """
        # Example from Fatih Karabiber
        result = similarities.cosine_similarity(
            set('the best data science course'.split(" ")),
            set('data science is popular'.split(" ")))
        assert round(result, 5) == 0.44721

    def test_simple_matching_coefficient(self):
        self._test_invalid_input(similarities.simple_matching_coefficient)
        self._test_two_empty_sets(similarities.simple_matching_coefficient)
        self._test_one_empty_set_param_one(similarities.simple_matching_coefficient)  # noqa: E501
        self._test_uneven_types(similarities.simple_matching_coefficient)
        self._test_no_match(similarities.simple_matching_coefficient)
        self._test_full_match(similarities.simple_matching_coefficient)

    def test_example_smc(self) -> None:
        """Test the simple matching coefficient with example sets.
        https://people.revoledu.com/kardi/tutorial/Similarity/SimpleMatching.html  # noqa: E501
        """
        # Example from Kardi Teknomo
        result = similarities.simple_matching_coefficient(
            {"a", "b", "c", "d"},
            {"b"})
        assert result == 0.25


if __name__ == "__main__":
    pass
