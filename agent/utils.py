import os

def get_highest_number(folder_path):
    files = os.listdir(folder_path)
    numbers = []

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        file_n = file_name.split('.')[0]
        if os.path.isfile(file_path) and file_n.isdigit():
            numbers.append(int(file_n))

    if numbers:
        highest_number = max(numbers)
        return highest_number
    else:
        return 0