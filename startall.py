import sys 
import os

num_processes = int(sys.argv[1])
num_episodes = int(sys.argv[2])

command = ""

for i in range(num_processes):
    command += f"python mcts_cpu.py {num_processes} {num_episodes} & "
command += "ls"

os.system(command)