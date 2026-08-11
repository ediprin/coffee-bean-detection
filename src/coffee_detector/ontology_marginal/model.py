from __future__ import annotations

from typing import Any

from ultralytics.nn.tasks import DetectionModel

from .loss import OntologyDetectionLoss, OntologyMarginalConfig


class OntologyDetectionModel(DetectionModel):
    """YOLO detection graph with an unchanged forward path and custom loss only."""

    def __init__(
        self,
        cfg: str | dict = "yolo26.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        ontology_marginal: OntologyMarginalConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if int(self.model[-1].nc) != 21:
            raise ValueError("Ontology-marginal v1 hanya valid untuk SNI-21")
        self.ontology_marginal_config = OntologyMarginalConfig.from_mapping(
            ontology_marginal
        )

    def init_criterion(self):
        from ultralytics.utils.loss import E2ELoss

        if getattr(self, "end2end", False):
            return E2ELoss(self, loss_fn=OntologyDetectionLoss)
        return OntologyDetectionLoss(self)
