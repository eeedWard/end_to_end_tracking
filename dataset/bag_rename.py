import os
bags_dir = "Mthesis/database/my_database/bags/"
os.chdir(bags_dir)

count = 335 #first number of the resulting bag

for root, dirs, files in os.walk("."):
    for filename in sorted(files):
        if "2020" in filename:
            rename = "pt_only_ss_%03d.bag"%count

            if os.path.exists(rename):
                print('FILE ALREADY EXISTS, ABORTING')
                exit()

            os.rename(bags_dir + filename, rename)
            print("DONE with ", rename)
            count += 1
