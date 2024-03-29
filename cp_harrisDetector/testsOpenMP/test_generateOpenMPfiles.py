import os
from time import sleep

# get template code in a string stored in variable lines
with open("testTemplate_harrisDetectorOpenMP.c") as f:
    lines = f.readlines()
    lines = "".join(lines)

# list of tests done in the first loop and second loop
tests = [
    "schedule(guided,8)",
    "schedule(guided,2)",
    "schedule(guided,30)",
    "schedule(static,8)",
    "schedule(dynamic,8)",

    "schedule(guided,8) collapse(2)",
    "schedule(guided,2) collapse(2)",
    "schedule(guided,30) collapse(2)",
    "schedule(static,8) collapse(2)",
    "schedule(dynamic,8) collapse(2)"
]

test_scripts = [f"test_harrisDetectorOpenMP_{tests[i].replace('(','').replace(')','').replace(',','').replace(' ','')}.c" for i in range(len(tests))]
test_results = ""

user_input = input("Do you want to compare the generated images after executing each file (y/n)? ")

# execute all tests
for i in range(len(tests)):

    # format template code, save it in a script 
    with open(test_scripts[i],"w") as f:
        f.write(lines.format(tests[i],tests[i]))

    # compile script
    os.system(f"g++ -fopenmp -O3 -ICommon {test_scripts[i]} -o {test_scripts[i][:-2]}")

    # execute code and print results
    print(tests[i])
    os.system(f"./{test_scripts[i][:-2]}")

    # compare images if user has requested it 
    if user_input.lower()=='y':
        os.system(f"../testDiffs ../resultChessBigOpenMP.pgm ../referenceChessBigOpenMP.pgm")

    print("\nwaiting 10 secs...")
    sleep(10)