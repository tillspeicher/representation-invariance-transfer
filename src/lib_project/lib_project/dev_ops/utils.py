import subprocess


def run_command(command: str) -> bool:
    """Run a command as subprocess and return whether it was successful."""
    if len(command) == 0:
        return True
    # Run the command interactively
    process = subprocess.Popen(command, shell=True)
    try:
        # Wait for the process to finish or KeyboardInterrupt
        returncode = process.wait()
        return returncode == 0
    except KeyboardInterrupt:
        # Terminate the process when KeyboardInterrupt is raised
        process.terminate()
        return False
