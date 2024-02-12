__all__ = ["DatasetLoader", "NotConfiguredError", "UNSET"]
from functools import partial
from pathlib import Path
from shutil import rmtree
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Literal, Optional
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
import yaml

from .config import DOWNLOAD, DatasetConfig, ImageFormat, InvalidConfigError, Preprocessing
from .dataset import Dataset
from .utilities import Rect, RichProgress

#   === CONSTANTS ===
# Figshare public urls of the MAPLES DR dataset
MAPLES_DR_ADDITIONAL_URL = "https://figshare.com/ndownloader/files/43695822"
MAPLES_DR_DIAGNOSTIC_URL = "https://figshare.com/ndownloader/files/43654878"

#: Unset constant
UNSET = "UNSET"


class NotConfiguredError(Exception):
    """
    Exception raised when the dataset loader is not configured.
    """

    def __init__(self, message: str = "MAPLES-DR dataset is not configured yet.", *args):
        super().__init__(message, *args)


class DatasetLoader:
    """
    Loader for MAPLES-DR dataset.
    """

    def __init__(self):
        self._datasets_config = DatasetConfig(
            resize=1500,
            image_format="PIL",
            preprocessing="none",
            cache=None,
        )
        self.maples_dr_infos: Optional[dict] = None
        self._maples_dr_path: Optional[Path] = None
        self._diagnosis: Optional[pd.DataFrame] = None
        self._is_maples_dr_folder_temporary: bool = False

        self._messidor_paths: Optional[dict[str, str]] = None
        self._messidor_ROIs: Optional[dict[str, Rect]] = None

    def __del__(self):
        if self._is_maples_dr_folder_temporary:
            rmtree(self._maples_dr_path, ignore_errors=True)

    # === CONFIGURATION ===
    def configure(
        self,
        maples_dr_path: Optional[str] = UNSET,
        maples_dr_diagnosis_path: Optional[str] = UNSET,
        messidor_path: Optional[str] = UNSET,
        resize: Optional[int] = None,
        image_format: Optional[ImageFormat] = None,
        preprocessing: Optional[Preprocessing] = None,
        cache: Optional[str] = None,
        disable_check: bool = False,
    ):
        """Configure the default behavior of the MAPLES-DR dataset.

        Parameters
        ----------
        maples_dr_path : Optional[str], optional
            Path to the MAPLES-DR additional data. Must point to the directory or to the zip file.
            If None (by default), then the dataset is downloaded from figshare.
        maples_dr_diagnosis_path : Optional[str], optional
            Path to the MAPLES-DR diagnosis.xls.
            If None (by default), then the file is downloaded from figshare.
        messidor_path : Optional[str], optional
            Path to the MESSIDOR dataset.
            Must point to a directory containing the "Base11", "Base12", ... subdirectories or zip files.
        resize : Optional[int], optional
            Size of the generated images. By default, keep the original image size of 1500x1500.
        image_format : Optional[ImageFormat], optional
            Python format of the generated images. Must be either "PIL", "rgb" or "bgr".
            If "rgb" or "bgr" is selected, images will be formatted as numpy array of shape: (height, width, channel).
            By default, "PIL" is used.
        preprocessing : Optional[Preprocessing], optional
            Preprocessing aglorithm applied on the fundus images.
            Must be either "clahe", "median" or None (no preprocessing).
            By default, no preprocessing is applied.
        cache : Optional[str], optional
            Path to permanently cache the formatted dataset. If None (by default), then the cache is disabled.
        disable_check : bool, optional
            If True, disable the integrity check of the dataset.
        """
        # === Update the dataset configuration ===
        self._datasets_config.update(
            resize=resize,
            image_format=image_format,
            preprocessing=preprocessing,
            cache=cache,
        )

        # === Prepare Maples-DR ===
        # If configure is called the first time with no path, download the dataset.
        if maples_dr_path is UNSET and self._maples_dr_path is None:
            maples_dr_path = DOWNLOAD

        if maples_dr_path is not UNSET and self._maples_dr_path != maples_dr_path:
            # Set the path and download the dataset if needed.
            folder = self._change_maples_dr_path(maples_dr_path)

            # Load the dataset infos.
            with open(folder / "biomarkers-export-config.yaml", "r") as infos:
                self.maples_dr_infos = yaml.safe_load(infos)

            with open(folder / "MESSIDOR-rois.yaml", "r") as rois_yaml:
                rois = yaml.safe_load(rois_yaml)
                self._messidor_ROIs = {k.rsplit(".", 1)[0]: Rect(*v) for k, v in rois.items()}

            # Check the integrity of the dataset.
            if not disable_check:
                self.check_maples_dr_integrity()

        # If configure is called the first time with no diagnosis path, download the excel file.
        if maples_dr_diagnosis_path is UNSET and self._diagnosis is None:
            maples_dr_diagnosis_path = DOWNLOAD

        if maples_dr_diagnosis_path is not UNSET:
            if maples_dr_diagnosis_path is None:
                self._maples_dr_diagnosis = None
            else:
                self._diagnosis = self.load_maples_dr_diagnosis(maples_dr_diagnosis_path)

        # === Prepare MESSIDOR ===
        if messidor_path is not UNSET:
            if messidor_path is None:
                self._messidor_paths = None
            else:
                self.discover_messidor_images(self.image_names(extensions=False), messidor_path)

    @property
    def datasets_config(self) -> DatasetConfig:
        """
        Return the default configuration of the loaded dataset.
        """
        return self._datasets_config

    def is_configured(self) -> bool:
        """
        Check if the dataset is initialized.
        """
        return self.maples_dr_infos is not None

    def ensure_configured(self):
        """
        Ensure the dataset is initialized.
        """
        if not self.is_initialized():
            self.configure()

    # --- Maples-DR path configuration ---
    @property
    def maples_dr_folder(self) -> Path:
        """
        Return the path to the MAPLES-DR dataset folder.
        """
        if self._maples_dr_path is None:
            raise NotConfiguredError()
        return self._maples_dr_path

    def _change_maples_dr_path(self, path: Optional[str | Path] = None) -> Path:
        """
        Return the path to the MAPLES-DR dataset folder.
        """
        # If Maples-DR was previously downloaded, delete the temporary folder.
        if self._is_maples_dr_folder_temporary and self._maples_dr_path is not None:
            rmtree(self._maples_dr_path, ignore_errors=True)
            self._is_maples_dr_folder_temporary = False

        self._maples_dr_path = None

        # If no path is given, create a temporary folder to download the dataset
        if path is None or path is DOWNLOAD:
            path = mkdtemp()
            self._is_maples_dr_folder_temporary = True

        path = Path(path)

        if path.is_dir():
            # === If the path is a directory ===
            # Ensure it exists ...
            if not path.exists():
                path.mkdir(parents=True)
            # and check if the folder contains the dataset infos.
            if not (path / "biomarkers-export-config.yaml").exists():
                # If not, download the dataset.
                zip_path = path / "maples_dr.zip"
                download(MAPLES_DR_ADDITIONAL_URL, zip_path, "MAPLES-DR segmentation maps")
                with ZipFile(path / "maples_dr.zip", "r") as zip_file:
                    zip_file.extractall(path)
                (path / "maples_dr.zip").unlink()
                path = path / "AdditionalData"

        elif path.name.endswith(".zip"):
            # === If the path is a zip file, unzip it to a temporary folder ===
            zip_path = path
            # Create a temporary folder.
            path = Path(mkdtemp())
            # Unzip the dataset to the temporary folder.
            if not zip_path.exists():
                raise InvalidConfigError(f"Invalid Maples DR archive: {zip_path} doesn't exist.")
            try:
                with ZipFile(zip_path, "r") as zip_file:
                    zip_file.extractall(path)
                    path = path / "AdditionalData"
            except Exception as e:
                raise InvalidConfigError(
                    f"Invalid Maples DR archive: {zip_path}:" "\n\t the provided archive is impossible to unzip."
                ) from e
            else:
                # Test if the zip file contains maples_dr folder.
                if not (path / "biomarkers-export-config.yaml").exists():
                    raise InvalidConfigError(
                        f"Invalid Maples DR archive:  {path}:"
                        '\n\t the provided archive doesn\'t contains the file "biomarkers-export-config.yaml".'
                    )

        self._maples_dr_path = path
        return path

    def check_maples_dr_integrity(self):
        """
        Check if the MAPLES-DR dataset contains all segmentation maps.
        """
        maples_root = self.maples_dr_folder
        missing_images = 0
        for biomarker in self.maples_dr_infos["biomarkers"]:
            for img in self.maples_dr_infos["test"] + self.maples_dr_infos["train"]:
                if not (maples_root / "annotations" / biomarker / img).exists():
                    missing_images += 1
        if missing_images > 0:
            raise InvalidConfigError(
                f"The provided folder to the Maples-DR dataset is incomplete: "
                f"{missing_images} segmentation maps are missing."
            )

    @staticmethod
    def load_maples_dr_diagnosis(path: Optional[str | Path] = None) -> pd.DataFrame:
        """
        Load the MAPLES-DR diagnostic file.

        Parameters
        ----------
        path : str
            Path to the MAPLES-DR diagnostic file. If None, download the file from Figshare.
        """
        if path is None or path is DOWNLOAD:
            with NamedTemporaryFile() as tmp:
                download(MAPLES_DR_DIAGNOSTIC_URL, tmp.name, "MAPLES-DR diagnostic file")
                return DatasetLoader.load_maples_dr_diagnosis(tmp)

        dr_diagnosis = (
            pd.read_excel(path, sheet_name="DR", index_col=0)
            .rename(columns={"Consensus": "dr"} | {f"Retinologist{r}": f"dr_{r}" for r in "ABC"})
            .drop(columns="MajorityVoting")
        )
        me_diagnosis = (
            pd.read_excel(path, sheet_name="ME", index_col=0)
            .rename(columns={"Consensus": "me"} | {f"Retinologist{r}": f"me_{r}" for r in "ABC"})
            .drop(columns="MajorityVoting")
        )
        return dr_diagnosis.join(me_diagnosis)

    # --- MESSIDOR path configuration ---
    def discover_messidor_images(self, images: list[str], path: Optional[str | Path] = None):
        """
        Discover the MESSIDOR images corresponding to the given MAPLES-DR images.

        :param images: List of MAPLES-DR images names. The image name should not contain the extension.
        :param path: Path to the MESSIDOR dataset.
        """
        path = Path(path)
        self._messidor_paths = {}

        if not path.is_dir():
            raise InvalidConfigError(f"Invalid MESSIDOR path: {self.cfg.messidor_path} is not a folder.")

        # Scan the MESSIDOR subfolders and list images.

        # Scan MESSIDOR subfolders to find each MAPLES-DR images.
        missing_images = set(images)
        for jpg in path.glob("**/*.tif"):
            try:
                missing_images.remove(jpg.stem)
            except KeyError:
                pass
            else:
                self._messidor_paths[jpg.stem] = jpg.absolute()
                if len(missing_images) == 0:
                    return

        # If some images are missing, try to unzip the MESSIDOR subfolders.
        unzip_folder = path / "maples_dr"
        total_missing = len(missing_images)
        with RichProgress.iteration(
            "Retrieving MESSIDOR fundus from zip files...",
            total=total_missing,
            done_message=f"Extracted {total_missing} images in {'{t}'} second.",
        ) as progress:
            for zip_path in path.glob("**/*.zip"):
                with ZipFile(zip_path, "r") as zip_file:
                    # If the zip file contains a MESSIDOR subfolder, unzip it.
                    for zip_content in zip_file.namelist():
                        if zip_content.endswith(".tif"):
                            stem = (zip_content.rsplit("/", 1)[-1])[:-4]
                            try:
                                missing_images.remove(stem)
                            except KeyError:
                                pass
                            else:
                                unzip_folder.mkdir(parents=True, exist_ok=True)
                                unzip_path = unzip_folder / (stem + ".tif")
                                zip_file.extract(zip_content, unzip_folder)
                                self._messidor_paths[stem] = unzip_path.absolute()
                                progress.update(1)
                                if len(missing_images) == 0:
                                    return

        # If some images are still missing, raise an error.
        if len(missing_images) > 0:
            raise InvalidConfigError(
                f"The provided folder to the MESSIDOR dataset is incomplete: "
                f"{len(missing_images)} images included in MAPLES-DR are missing."
            )

    # === DATASETS FACTORIES ===
    def dataset(self, subset: Optional[Literal["train", "test"]] = None) -> Dataset:
        """
        Return the MAPLES-DR dataset.

        :param subset: Subset of the dataset to return. If None, return the whole dataset.
                       Must be either None, "train" or "test".
        """
        if not self.is_configured():
            self.configure()
        names = self.image_names(subset, extensions=False)
        paths = {}
        if self._messidor_paths is not None:
            paths["fundus"] = [self._messidor_paths[name] for name in names]
        for biomarker in self.maples_dr_infos["biomarkers"]:
            paths[biomarker] = [self.maples_dr_folder / "annotations" / biomarker / (name + ".png") for name in names]
        data = pd.DataFrame(paths, index=names).join(self._diagnosis)

        return Dataset(data, self.datasets_config, self._messidor_ROIs)

    # === UTILITIES ===
    def image_names(
        self, subset: Optional[Literal["train", "test", "duplicates"]] = None, extensions: bool = True
    ) -> list[str]:
        """
        Return the list of images names of the given subset.

        :param subset: Subset to return the images names from. If None, return all images names.
                       Must be either None, "train", "test" or "duplicates".
        :param no_extension: If True, return the images names without the extension.
        """
        if not self.is_configured():
            raise NotConfiguredError()
        names = []
        if subset is None or subset == "train":
            names += self.maples_dr_infos["train"]
        if subset is None or subset == "test":
            names += self.maples_dr_infos["test"]
        if subset is None or subset == "duplicates":
            names += list(self.maples_dr_infos["duplicates"].values())
        if not extensions:
            names = [name.rsplit(".", 1)[0] for name in names]
        return names


GLOBAL_LOADER = DatasetLoader()


def download(url: str, path: str | Path, description: Optional[str] = None):
    """
    Download the file at the given url to the given path.
    """
    response = urlopen(url)
    with RichProgress.download(description, byte_size=int(response.info()["Content-length"])) as progress:
        with open(path, "wb") as dest_file:
            for data in iter(partial(response.read, 32768), b""):
                dest_file.write(data)
                progress.update(len(data))
