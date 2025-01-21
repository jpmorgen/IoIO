import time as t

hum_limit = 80 # Humidity limit
time_limit = 120 # time is s weather station can be offline before unsafe
input_file = r"C:\Users\IoIO\Desktop\IoIO\AAG\AAG_SLD.dat"
output_file = r"C:\Users\IoIO\Dropbox\AAG\AAG_SLD_HUM.dat"
output_file2 = r"C:\inetpub\wwwroot\AAG_SLD_HUM.dat"
running_log = r"C:\Users\IoIO\Desktop\IoIO\data\AAG\AAG_SLD.log"

ooutput_txt = ""
while True:
    try:
        t.sleep(5)
        with open(input_file, "r") as f:
            input_txt = f.read()
    
        print("I: " + input_txt)
        output_txt = input_txt
        
        # Fix humidity > 100% and dew point sudden drop to -20 with
        # simultanous humidity = 50%
        ambient = float(input_txt[33:40])
        humidity = float(input_txt[55:58])
        dew_point = float(input_txt[58:65])
        if 49.9 < humidity and humidity < 50.1 and dew_point < -19.9:
            humidity = 100
            dew_point = ambient
        humidity = min(humidity, 100)
        output_txt = output_txt[:54]                \
                      + '{:4.0f}'.format(humidity)  \
                      + '{:7.1f}'.format(dew_point) \
                      + output_txt[65:]

        # Check to make sure weather station is still online
        lasttime = t.strptime(input_txt[0:19], '%Y-%m-%d %H:%M:%S')
        dt = t.time() - t.mktime(lasttime)
        if (int(humidity) > hum_limit
            or dt > time_limit):
            output_txt = output_txt[:103] + "1"
        print("O: " + output_txt)
    
        with open(output_file, "w") as f:
            f.write(output_txt)

        with open(output_file2, "w") as f:
            f.write(output_txt)

        if output_txt != ooutput_txt:
            with open(running_log, "a") as f:
                f.write(output_txt + "\n")
        ooutput_txt = output_txt
            
    except Exception as e:
        print(f'{e}. Checking again in 20 seconds...')
        t.sleep(20)
        

