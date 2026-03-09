"""Tests for the cheatsheet module."""

import pytest

from tgp.poolers import pooler_map
from tgp.utils.cheatsheet import (
    extract_paper_links,
    get_pooler_cheatsheet,
    print_cheatsheet,
    supports_aux_loss,
    supports_sparse,
    supports_trainable,
)


class TestCheatsheetFunctions:
    """Test individual cheatsheet functions."""

    def test_supports_sparse(self):
        """Test the supports_sparse function."""
        # Test with a known sparse pooler
        assert isinstance(supports_sparse("topk"), bool)

        # Test with invalid pooler name
        assert supports_sparse("nonexistent_pooler") is False

    def test_supports_trainable(self):
        """Test the supports_trainable function."""
        # Test with a known pooler
        assert isinstance(supports_trainable("topk"), bool)

        # Test with invalid pooler name
        assert supports_trainable("nonexistent_pooler") is False

    def test_supports_aux_loss(self):
        """Test the supports_aux_loss function."""
        # Test with a known pooler
        assert isinstance(supports_aux_loss("topk"), bool)

        # Test with invalid pooler name
        assert supports_aux_loss("nonexistent_pooler") is False

    def test_extract_paper_links(self):
        """Test the extract_paper_links function."""

        # Test with a class that has no docstring
        class NoDocClass:
            pass

        assert extract_paper_links(NoDocClass) == []

        # Test with a class that has docstring with links
        class WithLinksClass:
            """This is a test class.

            Reference: `Paper <https://example.com/paper1>`_
            Another link: https://example.com/paper2
            """

        links = extract_paper_links(WithLinksClass)
        assert len(links) >= 1
        assert any("example.com" in link for link in links)

    def test_get_pooler_cheatsheet(self):
        """Test the main cheatsheet generation function."""
        cheatsheet_data = get_pooler_cheatsheet()

        # Check that we get data for all poolers
        assert len(cheatsheet_data) > 0
        assert len(cheatsheet_data) == len(pooler_map)

        # Check the structure of each entry
        for entry in cheatsheet_data:
            assert (
                len(entry) == 6
            )  # pooler_name, class_name, sparse, trainable, aux_loss, paper_links
            pooler_name, class_name, sparse, trainable, aux_loss, paper_links = entry

            # Check types
            assert isinstance(pooler_name, str)
            assert isinstance(class_name, str)
            assert isinstance(sparse, bool)
            assert isinstance(trainable, bool)
            assert isinstance(aux_loss, bool)
            assert isinstance(paper_links, list)

            # Check that pooler_name exists in pooler_map
            assert pooler_name in pooler_map

            # Check that class_name matches the actual class name
            assert class_name == pooler_map[pooler_name].__name__

        # Check that data is sorted by class name
        class_names = [entry[1] for entry in cheatsheet_data]
        assert class_names == sorted(class_names)

    def test_print_cheatsheet(self, capsys):
        """Test the print_cheatsheet function."""
        print_cheatsheet()

        captured = capsys.readouterr()
        output = captured.out

        # Check that output contains expected headers
        assert "TGP Pooler Cheatsheet" in output
        assert "Name" in output
        assert "Class" in output
        assert "Sparse" in output
        assert "Trainable" in output
        assert "Aux Loss" in output

        # Check that output contains at least one pooler
        pooler_names = list(pooler_map.keys())
        assert any(name in output for name in pooler_names)


class TestCheatsheetIntegration:
    """Test cheatsheet integration with actual poolers."""

    def test_all_poolers_covered(self):
        """Test that all poolers in pooler_map are covered in cheatsheet."""
        cheatsheet_data = get_pooler_cheatsheet()
        cheatsheet_poolers = {entry[0] for entry in cheatsheet_data}
        pooler_map_poolers = set(pooler_map.keys())

        assert cheatsheet_poolers == pooler_map_poolers

    def test_property_consistency(self):
        """Test that pooler properties are consistently detected."""
        # Test a few known poolers to ensure properties are detected correctly
        cheatsheet_data = get_pooler_cheatsheet()
        pooler_dict = {entry[0]: entry for entry in cheatsheet_data}

        # Test each pooler individually to catch any instantiation errors
        for pooler_name in pooler_map.keys():
            assert pooler_name in pooler_dict
            pooler_name, class_name, sparse, trainable, aux_loss, paper_links = (
                pooler_dict[pooler_name]
            )

            # These should not raise exceptions
            assert supports_sparse(pooler_name) == sparse
            assert supports_trainable(pooler_name) == trainable
            assert supports_aux_loss(pooler_name) == aux_loss


if __name__ == "__main__":
    pytest.main([__file__])
