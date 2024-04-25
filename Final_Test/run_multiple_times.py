import subprocess

number_of_runs = 10  # Set the desired number of iterations

for i in range(number_of_runs):
    command = (
        rf'python C:\Users\A490243\Desktop\Master_Thesis\Mid_Term_Test\test_complex_env_makeController.py {i}')
    try:

        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"iteration {i} output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:

        print(f"iteration {i} error:\n{e.stderr}")
        break  