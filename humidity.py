import time as t

limit = 75
input_file = "/mnt/c/Users/Puppy/Desktop/AAG_SLD.dat"
output_file = "output.txt"

while True:
    t.sleep(2)
    with open(input_file, "r") as f:
        input_txt = f.read()

    print("I: " + input_txt)
    output_txt = input_txt
    
    if int(input_txt[56:58]) > limit:
        output_txt = input_txt[:103] + "1"
    print("O: " + output_txt)

    with open(output_file, "w") as f:
        f.write(output_txt)
