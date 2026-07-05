from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


BASE_SCRIPT = Path(__file__).with_name("05_distinguish cell type.py")


def load_base_module():
    spec = spec_from_file_location("stage05_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base script: {BASE_SCRIPT}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    stage05 = load_base_module()

    stage05.PIPELINE_VARIANT = "05b"
    stage05.ABSOLUTE_FEATURE_NORMALIZATION = "per_image_robust"
    stage05.INFER_UMAP_MODE = "per_image_fit"
    stage05.OUTPUT_ROOT = (
        stage05.ANALYSIS_ROOT / f"{stage05.INFERENCE_SET_NAME}_supervised_prediction_xgb_relative"
    )

    print("============================================================")
    print("05-b relative batch-normalized mode")
    print("批次内相对化模式：对绝对尺度特征做每张图内鲁棒标准化")
    print("============================================================")

    stage05.main()


if __name__ == "__main__":
    main()
