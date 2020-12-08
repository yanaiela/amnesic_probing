import spur
from tqdm import tqdm


env = {
    'USE_SIMPLE_THREADED_LEVEL3': '1',
    'OMP_NUM_THREADS': '1',
}
ts = 'PATH-TO-TASK-SPOOLER/ts'

# ┌──────────────────────┐
# │ connect to all nodes │
# └──────────────────────┘
nodes = [
    'nlp01',
    #'nlp02',
    'nlp03',
    'nlp04',
    'nlp05',
    'nlp06',
    'nlp07',
    'nlp08',
    'nlp09',
    'nlp10',
    'nlp11',
    'nlp12',
    'nlp13',
    'nlp14',
    'nlp15',
]

# assumes automatic connection w/o password
connections = [spur.SshShell(hostname=node, username="USERNAME") for node in nodes]

dargs = {}

for connection in tqdm(connections):

    connection.run(f"{ts} -C".split(), update_env=env)
    connection.run(f"{ts} -S 4".split(), update_env=env)
