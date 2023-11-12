import os


for filename in os.listdir('.'):
    fp = open(filename, 'r')
    c = fp.read().replace('', '')
    fp.close()
    fp = open(filename, 'w')
    fp.write(c)
    fp.close()
    os.rename(filename, filename.replace('', ''))