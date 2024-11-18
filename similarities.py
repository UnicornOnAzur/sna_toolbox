# -*- coding: utf-8 -*-
"""
@author: UnicornOnAzur

"""

def validate_input(option="both"):
    def decorator_validate_input(func):
        def wrapper(*args):
            # print(args)
            if any(not isinstance(s, set) for s in args):
                raise TypeError("These are not all sets!")
            elif option == "both" and all(s == set() for s in args):
                raise ValueError("One or both sets are empty")
            elif option == "one" and any(s == set() for s in args):
                raise ValueError("One of the sets is empty")
            else:
                return func(*args)
        return wrapper
    return decorator_validate_input


@validate_input("one")
def overlap_coefficient(set1: set, set2: set):
    numerator = len(set1 & set2)
    denumerator = min((len(set1), len(set2)))
    return numerator / denumerator


@validate_input("both")
def jaccard_coefficient(set1: set, set2: set):
    numerator = len(set1 & set2)
    denumerator = len(set1 | set2)
    return numerator / denumerator


@validate_input()
def dice_sørensen_coefficient(set1: set, set2: set):
    """Dice-Sørensen"""
    numerator = 2 * len(set1 & set2)
    denumerator = len(set1) + len(set2)
    return numerator / denumerator


@validate_input()
def cosine_coefficient(set1: set, set2: set):
    pass

def demo():
    pass


if __name__ == "__main__":
    demo()
