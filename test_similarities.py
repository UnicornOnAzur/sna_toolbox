import pytest
import similarities


class TestOverlapCoefficient:
    def test_invalid_input(self):
        with pytest.raises(TypeError):
            similarities.overlap_coefficient(1,1)

class TestJaccardCoefficient:
    def test_invalid_input(self):
        with pytest.raises(TypeError):
            similarities.jaccard_coefficient(1,1)

class TestDiceSørensenCoefficient:
    def test_invalid_input(self):
        with pytest.raises(TypeError):
            similarities.dice_sørensen_coefficient(1,1)

class TestCosineCoefficient:
    def test_invalid_input(self):
        with pytest.raises(TypeError):
            similarities.cosine_coefficient(1,1)

if __name__ == "__main__":
    pass
