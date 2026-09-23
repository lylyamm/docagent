"""Test de fumée : vérifie que l'environnement est bien installé."""

import pymupdf

import docagent


def test_package_importable():
    assert docagent.__version__


def test_pymupdf_available():
    assert pymupdf.__version__
