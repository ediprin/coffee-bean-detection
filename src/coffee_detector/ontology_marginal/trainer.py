from __future__ import annotations

from typing import Any

from .loss import OntologyMarginalConfig
from .model import OntologyDetectionModel


def make_ontology_marginal_trainer(
    config: OntologyMarginalConfig | dict[str, Any],
):
    from ultralytics.models.yolo.detect import DetectionTrainer

    frozen = OntologyMarginalConfig.from_mapping(config)

    class OntologyMarginalTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = self.set_model_names_for_load(
                OntologyDetectionModel(
                    cfg,
                    nc=self.data["nc"],
                    ch=self.data["channels"],
                    verbose=verbose,
                    ontology_marginal=frozen,
                )
            )
            if weights:
                model.load(weights)
            return model

    OntologyMarginalTrainer.__name__ = (
        "OntologyMarginalTrainerSemantic"
        if frozen.mode == "semantic"
        else "OntologyMarginalTrainerIdentityControl"
    )
    return OntologyMarginalTrainer
