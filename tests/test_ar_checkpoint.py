from pathlib import Path

import torch

from chronopde.training.trainer import load_checkpoint, save_checkpoint


def test_checkpoint_restores_training_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=4,
        optimizer_steps=20,
        best_metric=0.2,
        patience_counter=3,
        model_name="test",
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    payload = load_checkpoint(path, model, optimizer, scheduler)
    assert payload["epoch"] == 4
    assert payload["optimizer_steps"] == 20
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
