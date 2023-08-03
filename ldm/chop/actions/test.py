import logging
import os
import pickle

import pytorch_lightning as pl
from ldm.chop.plt_wrapper import get_model_wrapper
from ldm.chop.tools.checkpoint_load import load_model
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

logger = logging.getLogger(__name__)


import logging
import os
import pickle

import pytorch_lightning as pl
from ldm.chop.plt_wrapper import get_model_wrapper
from ldm.chop.tools.checkpoint_load import load_model
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

logger = logging.getLogger(__name__)


def test(
    model_name,
    info,
    model,
    task,
    data_module,
    optimizer,
    learning_rate,
    plt_trainer_args,
    auto_requeue,
    save_path,
    load_name,
    load_type,
):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if save_path is not None:
        tb_logger = TensorBoardLogger(save_dir=save_path, name="logs_test-sw")
        plt_trainer_args["callbacks"] = []
        plt_trainer_args["logger"] = tb_logger

    # plugin
    if auto_requeue:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    plt_trainer_args["plugins"] = plugins

    wrapper_cls = get_model_wrapper(model_name, task)

    if load_name is not None:
        if isinstance(model, dict):
            model["model"] = load_model(
                load_name, load_type=load_type, model=model["model"]
            )
        else:
            model = load_model(load_name, load_type=load_type, model=model)
    plt_model = wrapper_cls(
        model, info=info, learning_rate=learning_rate, optimizer=optimizer
    )

    trainer = pl.Trainer(**plt_trainer_args)
    data_module.prepare_data()
    data_module.setup()
    if data_module.test_dataset is not None:
        trainer.test(plt_model, datamodule=data_module)
    elif data_module.pred_dataset is not None:
        predicted_results = trainer.predict(plt_model, datamodule=data_module)
        pred_save_name = os.path.join(save_path, "predicted_result.pkl")
        with open(pred_save_name, "wb") as f:
            pickle.dump(predicted_results, f)
        logging.info(f"Predicted results is saved to {pred_save_name}")
    else:
        raise RuntimeError(
            "Cannot run --test-sw because both the test dataset and pred dataset are None."
        )
