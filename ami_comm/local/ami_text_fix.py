#!/usr/bin/env python3

import sys
import re
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Fix full stops in transcript1 in ami_text_prep.sh.')
    parser.add_argument('file_in', type=str,
        help='input file path, transcript1 in ami_text_prep.sh.')
    parser.add_argument('file_out', type=str,
        help='output file path after fix.')
    args = parser.parse_args()
    fix_full_stop(args.file_in, args.file_out)

def fix_full_stop(file_in, file_out):
    with open(file_in, 'r') as fin, open(file_out, 'w') as fout:
        for line in fin:
            pattern = re.compile(r"[^\s\d]\.|\.[^\s\d]")
            result = pattern.search(line)
            if result is None:
                fout.write(line)
            else:
                full_stop_pos = [pos for pos, char in enumerate(line) if char == '.']
                full_stop_pos_to_change = []
                full_stop_replacement = []
                for pos in full_stop_pos:
                    if pattern.search(line[pos-1 : pos+2]) is not None:
                        full_stop_pos_to_change.append(pos)
                        space_replace_pattern = re.compile(r"\w\.\w")
                        if space_replace_pattern.search(line[pos-1: pos+2]) is not None:
                            full_stop_replacement.append(' ')
                        else:
                            full_stop_replacement.append('')
                new_string = []
                for pos, char in enumerate(line):
                    if pos not in full_stop_pos_to_change:
                        new_string.append(char)
                    else:
                        new_string.append(full_stop_replacement.pop(0))
                assert not full_stop_replacement
                fout.write(''.join(new_string))

if __name__ == '__main__':
    main()
