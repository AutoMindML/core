import subprocess
from automind import rc, logger


iterations = 5

if __name__ == "__main__":
    for i in range(iterations):
        logger.info(f"executing gemini_{i + 1}.py...")
        try:
            subprocess.run(
                ["uv", "run", f"./src/automind/_experiment/gemini_{i + 1}.py"]
            )
            logger.info(f"gemini_{i + 1}.py has finished")
        except Exception as e:
            logger.error("error occur: ", e)
