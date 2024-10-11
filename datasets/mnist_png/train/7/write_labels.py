import os
folder_files = os.listdir('.')
current_folder = os.getcwd()
current_folder = current_folder.split('\\')[-1]
with open('labels.txt', 'a') as f:
    for file in folder_files:
        if ".jpg" in file or ".png" in file or ".jpeg" in file:
            f.write(f'{file},{current_folder}\n')