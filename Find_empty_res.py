res_file_path = 'C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\36k2\\36k2Res.txt'
item_file_path = 'C:\\Users\\linka\\OneDrive\\Desktop\\Code\\HuiFu\\36k2\\36k2-copy.csv'

with open(item_file_path, "r", encoding="utf-8") as file2, open(res_file_path, "r", encoding="utf-8") as file:
    for line_number, (line1, line2) in enumerate(zip(file, file2), 1):
        line1 = line1.strip()
        line2 = line2.strip()
        if not line1:
            print(f"Line {line_number} in file1 is empty. Corresponding item in file2: {line2}")
       
