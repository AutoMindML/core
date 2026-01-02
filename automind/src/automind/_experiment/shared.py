import subprocess
from pathlib import Path

from automind import logger
from automind.data.dataset import AvailableDataset, load_data

target_column = "HEALTHCARE_EXPENSES"
iterations = 5
gemini_code_dir = Path(__file__).parent.absolute() / "gemini_code"
io_dir = Path(__file__).parent.absolute() / "io"
output_dir = Path(__file__).parent.absolute() / "output"
gemini_report_path = (
    Path(__file__).parent.absolute() / "report/gemini_report.txt"
)
automind_report_path = (
    Path(__file__).parent.absolute() / "report/automind_report.txt"
)
llm_response_dir = Path(__file__).parent.absolute() / "llm_response"
visual_output_dir = Path(__file__).parent.absolute() / "visualizations"
visual_output_dir.mkdir(parents=True, exist_ok=True)

df_patients = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_patients"]
)

df_encounters = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_encounters"]
)

df_conditions = load_data(
    AvailableDataset.synthea_covid19_10k.datasets["slice_conditions"]
)


def init_gemini_result_csv():
    for i in range(iterations):
        logger.info(f"executing gemini_{i + 1}.py...")
        try:
            subprocess.run(
                ["uv", "run", f"{gemini_code_dir}/gemini_{i + 1}.py"]
            )
            logger.info(f"gemini_{i + 1}.py has finished")
        except Exception as e:
            logger.error("error occur: ", e)
