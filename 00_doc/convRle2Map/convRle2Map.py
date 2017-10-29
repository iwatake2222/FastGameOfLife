# -*- coding: utf-8 -*-
""" main.py
Sample Code
"""
import sys
from re import sub


###
# Main
###
def main():
    """ main function
    """
    if len(sys.argv) == 2:
        filename_input = sys.argv[1]
    else:
        #filename_input = 'DJI_0001.DNG'
        print("invalid input filename")
        return

    file = open(filename_input)
    text = file.read()
    file.close()
    text = text.replace('\n','').replace('\r','')
    text = decode(text)
    text = text.replace('b', '.').replace('o', 'X').replace('$', '\n').replace('!', '\n')
    file = open(filename_input + '_map.txt', 'w')
    file.write(text)
    file.close()

    print('Exit')

def encode(text):
    return sub(r'(.)\1*', lambda m: str(len(m.group(0))) + m.group(1),
               text)

def decode(text):
    return sub(r'(\d+)(\D)', lambda m: m.group(2) * int(m.group(1)),
               text)

if __name__ == "__main__":
    main()
