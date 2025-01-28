from .mosei_datamodule import MOSEIDataModule
from .moseiemo_datamodule import MOSEIEMODataModule

_datamodules = {
    "mosei": MOSEIDataModule,
    "moseiemo": MOSEIEMODataModule,
}
