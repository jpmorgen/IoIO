import time as t

limit = 75
input_file = r"C:\Users\PLANETARY SCIENCE\Dropbox\AAG\AAG_SLD.dat"
output_file = r"C:\Users\PLANETARY SCIENCE\Dropbox\AAG\AAG_SLD_HUM.dat"
running_log = r"C:\Users\PLANETARY SCIENCE\Desktop\IoIO\data\AAG\AAG_SLD.log"

ooutput_txt = ""
while True:
    try:
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

        if output_txt != ooutput_txt:
            with open(running_log, "a") as f:
                f.write(output_txt + "\n")
        ooutput_txt = output_txt
            
    except Exception as e:
        print(f'{e}. Checking again in 5 seconds...')
        t.sleep(5)
        

