from statsforecast.models import _TS  # type: ignore

from ..model.time_series import load_model, save_model, train_model


async def load_models() -> dict[str, _TS]:
    model_path = "time_series.pkl"
    models = load_model(model_path)

    # If model doesn't exist, train and save a new one
    if not models:
        print("Training new model...")
        models = train_model()
        save_model(models, model_path)
        print("Models trained and saved successfully")

    print(f"Models loaded successfully: {models}")

    return models
