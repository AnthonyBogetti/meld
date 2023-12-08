import numpy as np
import meld
from meld import unit as u
from meld.vault import DataStore
from meld.system.scalers import LinearRamp
from meld.system.indexing import AtomIndex
import glob as glob

# Define global parameters
N_REPLICAS = 3
N_EXCHANGES = 10
BLOCK_SIZE = 5
nm = u.nanometer
ang = u.angstrom
kj = u.kilojoule/(u.mole*u.nanometer**2)

# Load in all pdb files and use the first to initialize system
templates = glob.glob('templates/*.pdb')
protein = meld.AmberSubSystemFromPdbFile(templates[0])

# Define force field and other MD options
build_options = meld.AmberOptions(forcefield="ff14sbside",
                                  implicit_solvent_model="gbNeck2",
                                  use_bigger_timestep=True,
                                  cutoff=1.0*nm)

# Build system
builder = meld.AmberSystemBuilder(build_options)
system = builder.build_system([protein]).finalize()

# Set temperature scaling options
system.temperature_scaler = meld.GeometricTemperatureScaler(0, 1, 300. * u.kelvin, 400. * u.kelvin)

# Set MELD restraint scaling options
scaler = system.restraints.create_scaler('constant')
percent = 1.0
dists = []

# Collect MELD restraints into a list
with open("contacts.dat") as cfile:
    rest_group = []
    lines = cfile.read().splitlines()
    for line in lines:
        if not line:
            print("noline")
            dists.append(system.restraints.create_restraint_group(rest_group, int(percent * len(rest_group))))
            rest_group = []
        else:
            atom_id = int(line.split()[0])
            x = float(line.split()[1])
            y = float(line.split()[2])
            z = float(line.split()[3])

            rest_group.append(system.restraints.create_restraint('cartesian', scaler, LinearRamp(0,100,0,1),
                                                                 atom_index=AtomIndex(atom_id), 
                                                                 x=x*ang, y=y*ang, z=z*ang, 
                                                                 delta=1*ang,
                                                                 force_const=250*kj))

# Add MELD restraints to the system
system.restraints.add_selectively_active_collection(dists, 1)

# Define no. of MD timesteps between exchanges (50 ps) and minimization length
#options = meld.RunOptions(timesteps = 25000,
#options = meld.RunOptions(timesteps = 11111,
options = meld.RunOptions(timesteps = 222, # every 1 ps
                          minimize_steps = 20000)

# Set up REMD with total of 500 ns
remd = meld.setup_replica_exchange(system, n_replicas=N_REPLICAS, n_steps=N_EXCHANGES)

# Store simulation output
store = DataStore(
        state_template=system.get_state_template(),
        n_replicas=N_REPLICAS,
        pdb_writer=system.get_pdb_writer(),
        block_size=BLOCK_SIZE)

store.initialize(mode="w")
store.save_system(system)
store.save_run_options(options)
store.save_remd_runner(remd.remd_runner)
store.save_communicator(remd.communicator)

# Add multiple starting states
def gen_state_templates(index,templates,rand):
#    n_templates = len(templates)
    random = rand[index]
#    protein = meld.AmberSubSystemFromPdbFile(templates[(n_templates-1)-(index%n_templates)])
    protein = meld.AmberSubSystemFromPdbFile(templates[random])
    build_options = meld.AmberOptions(forcefield="ff14sbside",
                                      implicit_solvent_model="gbNeck2",
                                      use_bigger_timestep=True,
                                      cutoff=1.0*nm)
    builder = meld.AmberSystemBuilder(build_options)
    system = builder.build_system([protein]).finalize()
    state = system.get_state_template()
    state.alpha = index / (N_REPLICAS-1)
    return state

#indices = np.concatenate((np.arange(0,len(templates)),np.arange(0,len(templates))))
indices = np.arange(0,len(templates))
np.random.shuffle(indices)

np.savetxt("indices.txt", indices)

states = [gen_state_templates(i,templates,indices) for i in range(N_REPLICAS)]
store.save_states(states, 0)
store.save_data_store()
