import subprocess
import os

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Executed: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command}: {e}")

if __name__ == "__main__":
    global_path = f'{os.getenv("ANALYSIS_PATH")}/data/'
    commands = [
        f"rm {global_path}stdall_*",
        f"rm {global_path}htcondor_jobs_*.json",
        f"rm -r {global_path}jobs/tmp*",
        f"rm {global_path}logs/*"
    ]
    
    for cmd in commands:
        run_command(cmd)