import importlib.util
import runpy


def _run_module_by_name(module_name: str):
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Module {module_name} not found in installed package")
    runpy.run_path(spec.origin, run_name="__main__")


def database_viewer():
    """Run the database viewer script packaged in this project."""
    _run_module_by_name("Database_viewer_final")

def run_prediction_tcia():
    """Run the nnU-Net prediction pipeline for TCIA data."""
    _run_module_by_name("run_prediction_of_nnUNet_networks_on_TCIA_data_final")

def run_prediction_new():
    """Run the nnU-Net prediction pipeline for new NIfTI data."""
    _run_module_by_name("run_prediction_of_nnUNet_networks_on_new_data")

if __name__ == "__main__":
    database_viewer()
