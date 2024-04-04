import subprocess
import os
from pathlib import WindowsPath

# print(subprocess.check_call(['wsl', 'dpkg', '-l', 'libopencv-dev']))

os.chdir(WindowsPath('\\\\wsl.localhost/Debian/home/colon/OpenCFU'))
subprocess.run(['wsl', 'pwd'])
# print(subprocess.run(['wsl', './opencfu', '-i', '/mnt/c/Users/colon/OneDrive/Documents/OpenCFU/samples/A.png','>', '/mnt/c/Users/colon/OneDrive/Documents/test.txt']))


petri_dish_name = 'p0'
cfu_coord_path = '../coords/'+ petri_dish_name + '.csv'
# subprocess.run(['wsl', 'mkdir', '../coords'])
subprocess.run(['wsl', './opencfu', '-i', '/mnt/c/Users/colon/Documents/computer_vision/red/petri_dish_0.jpg', '>', cfu_coord_path])


# subprocess.run(['wsl', 'mv', cfu_coord_path, '/mnt/c/Users/colon/OneDrive/Documents'])

# (subprocess.run(['wsl', './opencfu', '-i', '' '>', '/mnt/c/Users/colon/OneDrive/Documents/test.txt']))

# print(bing)









