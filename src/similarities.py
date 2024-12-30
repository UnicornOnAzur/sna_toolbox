# -*- coding: utf-8 -*-
"""
@author: UnicornOnAzur

This module provides functions to calculate five similarity measures: Overlap
coefficient, Jaccard similarity, Dice-Sørensen coefficient, Cosine similarity,
simple matching coefficient, and Hamming coefficient. Input validation ensures that the provided
sets meet the required criteria for each calculation.
"""

# Standard library
import functools
import math
import numbers
import typing
import warnings


def validate_input(option: typing.Optional[str] = None) -> typing.Callable:
    """
    Decorator to validate input sets for the decorated function.

    Parameters:
    option: Optional parameter to specify validation rules. If "one", at least
            one set must be non-empty.

    Raises:
    ValueError: If option is provided and is not "one".

    Returns:
    function: The wrapped function with input validation.
    """
    if option and option != "one":
        raise ValueError(
            "The provided option is incorrect; it can only be 'one'")

    def decorator_validate_input(func: typing.Callable) -> typing.Callable:
        """
        Inner decorator to apply validation to the function.

        Parameters:
        func: The function to be decorated.

        Returns:
        function: The wrapper function with input validation.
        """
        # Preserves the metadata of the original function, such as name.
        @functools.wraps(func)
        def wrapper_validate_input(*args: tuple[typing.Set]) -> typing.Any:
            """
            Wrapper function that performs input validation before calling the
            function.

            Parameters:
            args: Variable length argument list representing the sets to be
                  validated.

            Raises:
            TypeError: If any argument is not a set or if elements are of
                       different types.

            Returns:
            The result of the decorated function if validation passes or 0 if
            both sets are empty.
            """
            # Validate that all arguments are sets
            if not all(isinstance(s, set) for s in args):
                raise TypeError("All arguments must be sets!")
            # Ensure at least one set is non-empty if option is "one"
            if option == "one" and any(not s for s in args[:2]):
                warnings.warn("At least one of the sets must be non-empty.")
                return None
            # Return 0 if all sets are empty
            if all(not s for s in args[:2]):
                warnings.warn("Both sets are empty!", UserWarning)
                return 0
            # Ensure all elements in the sets are of the same type
            if len({numbers.Number if isinstance(c, numbers.Number)
                    else type(c) for c in args[0] | args[1]}) != 1:
                raise TypeError(
                    "Elements in the sets must be of the same type.")
            # Ensure that the provided totalrange is a superset of the other
            # two sets, otherwise remove it from the arguments
            if len(args) == 3:
                set1, set2, total_range = args
                if not total_range >= (set1 | set2):
                    warnings.warn("The total range provided is not a superset of the other two sets",  # noqa E501
                                  UserWarning)
                    args = args[:2]
            return func(*args)
        return wrapper_validate_input
    return decorator_validate_input


@validate_input("one")
def overlap_coefficient(set1: set,
                        set2: set,
                        total_range: typing.Optional[set] = None,
                        /) -> float:
    """Calculate the overlap coefficient between two sets.

    Formula:
    |set1 ∩ set2| / min(|set1|, |set2|)

    Parameters:
    set1: The first set.
    set2: The second set.
    total_range: The entire set of all options, default is None. Unused in
    this function but there for consistency.

    Returns:
    float: The overlap coefficient between the two sets.
    """
    return len(set1 & set2) / min(len(set1), len(set2))


@validate_input()
def jaccard_similarity(set1: set,
                       set2: set,
                       total_range: typing.Optional[set] = None,
                       /) -> float:
    """Calculate the Jaccard coefficient between two sets.

    Formula:
    |set1 ∩ set2| / |set1 ∪ set2|

    Parameters:
    set1: The first set.
    set2: The second set.
    total_range: The entire set of all options, default is None. Unused in
    this function but there for consistency.

    Returns:
    float: The Jaccard coefficient between the two sets.
    """
    return len(set1 & set2) / len(set1 | set2)


@validate_input()
def dice_sørensen_coefficient(set1: set,
                              set2: set,
                              total_range: typing.Optional[set] = None,
                              /) -> float:
    """Calculate the Dice-Sørensen coefficient between two sets.

    Formula:
    2 * |set1 ∩ set2| / (|set1| + |set2|)

    Parameters:
    set1: The first set.
    set2: The second set.
    total_range: The entire set of all options, default is None. Unused in
    this function but there for consistency.

    Returns:
    float: The Dice-Sørensen coefficient between the two sets.
    """
    return 2 * len(set1 & set2) / (len(set1) + len(set2))


@validate_input("one")
def cosine_similarity(set1: set,
                      set2: set,
                      total_range: typing.Optional[set] = None,
                      /) -> float:
    """Calculate the cosine coefficient between two sets.

    Formula:
    |set1 . set2| / (||set1|| x ||set2||)

    Parameters:
    set1: The first set.
    set2: The second set.
    total_range: The entire set of all options, default is None. Unused in
    this function but there for consistency.

    Returns:
    float: The cosine coefficient between the two sets.
    """
    intersection: set = set1 | set2
    vector1: list[int] = [1 if i in set1 else 0 for i in intersection]
    vector2: list[int] = [1 if i in set2 else 0 for i in intersection]
    dot_product: int = sum(a * b for a, b in zip(vector1, vector2))
    norm_a: float = math.sqrt(sum(a ** 2 for a in vector1))
    norm_b: float = math.sqrt(sum(b ** 2 for b in vector2))
    magnitude: float = norm_a * norm_b
    return dot_product / magnitude


@validate_input("one")
def simple_matching_coefficient(set1: set,
                                set2: set,
                                total_range: typing.Optional[set] = None,
                                /) -> float:

    """Calculate the simple matching coefficient between two sets.

    Formula: (p + s) / (p + q + r + s)
    p: the total number of positions where both vectors are 1
    q: the total number of positions where vector1 is 1 and vector2 is 0
    r: the total number of positions where vector1 is 0 and vector2 is 1
    s: the total number of positions where both vectors are 0

    Parameters:
    set1: The first set.
    set2: The second set.
    total_range: The entire set of all options, default is None.

    Returns:
    float: The simple matching coefficient between the two sets.
    """
    all_: set = set1 | set2 if not total_range else total_range
    p: int = len(set1 & set2)
    q: int = len(set1 - set2)
    r: int = len(set2 - set1)
    s: int = len((set1 ^ all_) & (set2 ^ all_))
    return (p + s) / (p + q + r + s)


def hamming_distance(set1: set, set2: set) -> int:
    """
    Calculate the Hamming distance between two sets.

    The Hamming distance is defined as the number of elements that are 
    present in either of the sets but not in both. This function uses 
    the symmetric difference operator to compute the distance.

    Parameters:
    set1 : The first set of elements.
    set2 : The second set of elements.

    Returns:
    int: The Hamming distance between the two sets.
    """
    return len(set1 ^ set2)


@validate_input()
def hamming_coefficient(set1: set,
                        set2: set,
                        total_range: typing.Optional[set] = None,
                        /) -> float:
    """
    Calculate the Hamming coefficient between two sets.

    The Hamming coefficient is the Hamming distance divided by the length of
    the total range.

    Parameters:
    set1 : The first set.
    set2 : The second set.
    total_range : The entire set of all options, default is None.

    Returns:
    float: The Hamming coefficient, a value between 0 and 1.
    """
    all_: set = set1 | set2 if not total_range else total_range
    return hamming_distance(set1, set2) / len(all_)


def demo():
    # Example usage of the coefficients
    print("Demonstration of the coefficients")
    set_a: set = {1, 2, 3}
    set_b: set = {2, 3, 4}

    print(f"Set A: {set_a} and set B: {set_b}")
    print("Overlap Coefficient:", overlap_coefficient(set_a, set_b))
    print("Jaccard Similarity:", jaccard_similarity(set_a, set_b))
    print("Dice-Sørensen Coefficient:", dice_sørensen_coefficient(set_a, set_b))  # noqa: E501
    print("Cosine Similarity:", cosine_similarity(set_a, set_b))
    print("Simple Matching Coefficient:", simple_matching_coefficient(set_a, set_b))  # noqa: E501
    print("Hamming Coefficient:", hamming_coefficient(set_a, set_b))


if __name__ == "__main__":
    demo()
