import os

current_directory = os.getcwd()
target_directory = os.path.join(current_directory,"csv_results")
file_names = [f for f in os.listdir(target_directory)]

for file in file_names:
    os.remove(os.path.join(target_directory,file))
    print(f'{file} has been deleted')
