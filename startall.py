import sys 
import os

num_processes = int(sys.argv[1])
num_episodes = int(sys.argv[2])
num_computers = int(sys.argv[3])

command = ""

for i in range(num_processes//num_computers):
    command += f"python mcts.py {num_processes} {num_episodes} & "
command += "ls"

os.system(command)