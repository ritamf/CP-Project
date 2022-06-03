import os
from time import sleep

# get template code in a string stored in variable lines
with open("testTemplate_harrisDetectorCuda.cu") as f:
    lines = f.readlines()
    lines = "".join(lines)

# list of tests done in the first loop and second loop
tests = [
    "chess{}.pgm",
    "chessBig{}.pgm",
    "house{}.pgm",
    "chessRotate1{}.pgm",
    "chessL{}.pgm"
]

test_scripts = [f"test_harrisDetectorCuda_{tests[i][:-4].format('')}.cu" for i in range(len(tests))]
test_results = ""

# execute all tests
for i in range(len(tests)):

    # format template code, save it in a script 
    with open(test_scripts[i],"w") as f:

        imgInput = "../"+tests[i].format("")
        imgOutRes = "../"+tests[i].format("ResultsCuda")
        imgOutRef = "../"+tests[i].format("ReferenceCuda")
        
        f.write(lines.format(imgInput, imgOutRes, imgOutRef))

    # compile script
    os.system(f"nvcc -arch=sm_30 -O3 -ICommon {test_scripts[i]} -o {test_scripts[i][:-3]}")

    # execute code and print results
    print(tests[i].format(""))
    os.system(f"./{test_scripts[i][:-3]}")
    print("\nwaiting 10 secs...")

    sleep(10)


user_input = input("Do you want to compare the images (y/n)? ")

if user_input.lower()=='y':
    # execute testDiffs
    for img in tests:
        img1 = "../"+tests[i].format("ResultsCuda")
        img2 = "../"+tests[i].format("ReferenceCuda")

        print(f"Difference between {img1[3:-4]} and {img2[3:-4]}")
        os.system("../testDiffs {} {}".format(img1, img2))
