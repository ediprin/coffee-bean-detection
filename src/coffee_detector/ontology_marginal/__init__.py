"""Zero-parameter ontology-marginalized classification supervision."""

from .loss import OntologyMarginalConfig, OntologyMarginalizer
from .model import OntologyDetectionModel
from .trainer import make_ontology_marginal_trainer

__all__ = [
    "OntologyDetectionModel",
    "OntologyMarginalConfig",
    "OntologyMarginalizer",
    "make_ontology_marginal_trainer",
]
