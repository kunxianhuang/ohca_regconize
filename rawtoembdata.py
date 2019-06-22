#!/bin/env python3
#!*-* coding=utf-8 *-*
import string,os
import argparse


def clean(infname):
    
    with open(infname,encoding='utf-8') as dataf:
        for i,line in enumerate(dataf):
            if (i==0):
                continue
            lines=line.split(',')
            #context = lines[1:]
            context='.'.join(lines)
            yield [ct.strip("\"") for ct in context if ct is not " "]
            
def main():
    parser = argparse.ArgumentParser(prog='rawtoembdata.py')
    parser.add_argument('--inputfile',type=str,dest="inputfile",default='data/script_raw.csv')
    parser.add_argument('--outputfile',type=str,dest="outputfile",default='data/script_emb.txt')
    args = parser.parse_args()

#    print(clean(args.inputfile))
    with open(args.outputfile, 'w+',encoding='utf-8') as ofile:
        for wrstr in clean(args.inputfile):
            ofile.write(' '.join(wrstr))
    
    
    return



if __name__ == '__main__':
    main()
