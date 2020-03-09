#!usr/bin/env python3
from pathlib import Path
from datetime import datetime as dt
import time

class WeatherFixer():
    def __init__(self):
        pass
    def DateFix(self, date):
        return(date.replace(' ', '0'))
    def ReadFoster(self, wstr):
        data = {}
        #print(wstr[0:10])
        #split = [wstr:3]
        #print(split)
        if wstr[77:78] == ' ':
            #print(wstr[77:])
            p2 = wstr[77:].replace(' ', '0', 1)
            p1 = wstr[:77]
            wstr = p1+p2
            #print(wstr)
            #print(wstr[:77])
        data['Date'] = wstr[0:10]
        data['Time'] = wstr[11:22]
        data['T'] = wstr[23:24]
        data['V'] = wstr[25:26]
        data['SkyT'] = wstr[27:31]
        data['AmbT'] = wstr[34:38]
        data['SenT'] = wstr[39:43]
        data['Wind'] = wstr[48:52]
        data['Hum'] = wstr[55:57]
        data['DewPt'] = wstr[59:64]
        data['Heat'] = wstr[66:69]
        data['R'] = wstr[70:71]
        data['W'] = wstr[72:73]
        data['Since'] = wstr[74:79]
        data['Now()'] = wstr[80:92]
        data['c'] = wstr[93:94]
        data['w'] = wstr[95:96]
        data['r'] = wstr[97:98]
        data['d'] = wstr[99:100]
        data['C'] = wstr[101:102]
        data['A'] = wstr[103:104]
        #data[]
        return(data)
    def FosterToClarityII(self, wstr):
        read = self.ReadFoster(wstr)
        read['Date'] = self.DateFix(read['Date'])
        d = float(read['Now()']) - 1104 # f = datetime.date(1966, 12, 24) # Foster Now()
#                                         l = datetime.date(1970, 1, 1) # Unix Now()
#                                         t = l-f # Subtracts them
#                                         t.days # gets 1104
        #print(d)
        #s = calendar.timegm(time.strptime(f'{read["Date"]} {read["Time"][:8]}', '%Y-%m-%d %H:%M:%S'))
        #m = y / 60
        #h = m / 60
        #d = h / 24
        if str(d)[5] == '.':
            d = f'0{d}'
        read['Now()'] = str(d)[:12]
        #print(read)
        for i in read:
            read[i] = read[i].lstrip()
        #print(read)
        fin = read['Date']
        for i in range(100):
            fin = fin + ' '
        fin = fin[:11] + read['Time'] + fin[11:]
        fin = fin[:23] + read['T'] + fin[23:]
        fin = fin[:25] + read['V'] + fin[25:]
        fin = fin[:27] + read['SkyT'] + fin[27:]
        fin = fin[:34] + read['AmbT'] + fin[34:]
        fin = fin[:40] + read['SenT'] + fin[40:]
        fin = fin[:48] + read['Wind'] + fin[48:]
        fin = fin[:55] + read['Hum'] + fin[55:]
        fin = fin[:59] + read['DewPt'] + fin[59:]
        fin = fin[:66] + read['Heat'] + fin[66:]
        fin = fin[:70] + read['R'] + fin[70:]
        fin = fin[:72] + read['W'] + fin[72:]
        fin = fin[:74] + read['Since'] + fin[74:]
        fin = fin[:80] + read['Now()'] + fin[80:]
        fin = fin[:93] + read['c'] + fin[93:]
        fin = fin[:95] + read['w'] + fin[95:]
        fin = fin[:97] + read['r'] + fin[97:]
        fin = fin[:99] + read['d'] + fin[99:]
        fin = fin[:101] + read['C'] + fin[101:]
        fin = fin[:103] + read['A'] #+ fin[103:]
        return(fin)

if __name__ == '__main__':
    w = WeatherFixer()
    #w.ReadFoster('2019- 1-16 19:41:53.00 C M -22     18   18      0      41  8      000 0 0 000 5 019016.82075 1 1 1 3 1 0')
    #print(w.DateFix('2019- 1-16'))
    #wstr = '2019- 1-16 19:41:53.00 C M -22     18   18      0      41  8      000 0 0 000 5 019016.82075 1 1 1 3 1 0'
    x = 0
    foster_dropbox = Path("C:/Users/jpmorgen/Dropbox/Foster/AstroAlert")
    # Maybe we want PureWindowsPath
    while True:
        try:
            print('\n \n')
            #    with open('/cygdrive/c/cygwin64/home/Puppy/JavaScript/F2CIIFile.txt', 'r') as f:
            with open(Path(foster_dropbox, 'OneLineFmt.txt'), 'r') as f:
                wstr = f.read()
            owstr = wstr
            while owstr == wstr:
                time.sleep(2)
                with open(Path(foster_dropbox, 'OneLineFmt.txt'), 'r') as f:
                    wstr = f.read()
                    print(f'I {wstr}')
            owstr = wstr
            read = w.ReadFoster(wstr)
            d = float(read['Now()']) - 1104
            if str(d)[5] == '.':
                d = f'0{d}'
                read['Now()'] = str(d)[:12]
                t = time.time()
            if t - (float(read['Now()'])*24*60*60) > 120: # (float(read['Now()'])*24*60*60)
                print('Waited too long. Giving error.')
                #print('1'.join(wstr.rsplit('0', 1)))
                c = w.FosterToClarityII('1'.join(wstr.rsplit('0', 1)))
            else:
                c = w.FosterToClarityII(wstr)
            print(f'O {c}')
            # --> make log file
            with open(Path(foster_dropbox, 'ClarityII.log'), 'w') as wf:
                wf.write(c + '\n')
        except Exception as e:
            print(f'{e}. Checking again in 5 seconds...')
            time.sleep(5)

