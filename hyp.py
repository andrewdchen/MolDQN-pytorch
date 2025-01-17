start_molecule = None
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 2000
optimizer = "Adam"
polyak = 0.995
atom_types = ["C", "O", "N"]
max_steps_per_episode = 10
allow_removal = True
allow_no_modification = True
allow_bonds_between_rings = False
allowed_ring_sizes = [3, 4, 5, 6]
replay_buffer_size = 1000000
learning_rate = 1e-4
gamma = 0.95
fingerprint_radius = 3
fingerprint_length = 2048
discount_factor = 0.9
num_eval = 800
warm_start_dataset = "dataset/NSP15_6W01_A_3_H.negonly_unique_100.csv"
constrain_factor = 100.0
delta = 0.6 # 0.0, 0.2, 0.4, 0.6
gpu = 1
run_id = "106"
name = "moldqn_autodock_const_opt_0.6_10" # opt_0.0, opt_0.2, opt_0.4, opt_0.6
obabel_path = ""
adt_path = ""
receptor_file = ""