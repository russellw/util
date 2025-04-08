import argparse
import os
import re
import subprocess
import shlex

text=''
ti=0
tok=''

def lex():
    global ti
    global tok
    while ti < len(text):
        # space
        if text[ti].isspace():
            ti += 1
            continue

        # comment
        elif text[ti] == "%":
            while text[ti] != "\n":
                ti += 1
            continue
        elif text[ti : ti + 2] == "/*":
            ti += 2
            while text[ti : ti + 2] != "*/":
                ti += 1
            ti += 2
            continue

        # word
        elif text[ti].isalnum() or text[ti] == "$" or text[ti] == "_":
            j = ti
            while text[ti].isalnum() or text[ti] == "$" or text[ti] == "_":
                ti += 1
            tok=text[j:ti]

        # quote
        elif text[ti] in ('"', "'"):
            j = ti
            q = text[ti]
            ti += 1
            while text[ti] != q:
                if text[ti] == "\\":
                    ti += 1
                ti += 1
            ti += 1
            tok=text[j:ti]

        # punctuation
        elif text[ti : ti + 3] in ("<=>", "<~>"):
            tok=text[ti : ti + 3]
            ti += 3
        elif text[ti : ti + 2] in ("!=", "<=" ,"=>" ,"~&" ,"~|"):
            tok=text[ti : ti + 2]
            ti += 2
        else:
            tok=text[ti]
            ti += 1
        return

    #eof
    tok=''

def eat(o):
    if tok == o:
        lex()
        return True

def expect(o):
    if not eat(o):
        err("Expected '" + o + "'")

parser = argparse.ArgumentParser(description="Verify TPTP format proof")
parser.add_argument("proof_file")
args = parser.parse_args()

text = open(args.proof_file).read()
lex()
print(tok)
