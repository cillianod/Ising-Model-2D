print("Cillian O'Donnell")
print('20333623')
import time
import numpy as np
import matplotlib.pyplot as plt

t = time.time()
# Class defining an Ising lattice object describing an L-site 1D lattice of classical spins
class IsingLattice2D():
    """Object describing a 2D Ising lattice, with size L, temperature T and magnetic field h.
    Includes methods to compute the magnetisation using the Metropolis Monte Carlo algorithm"""
    
    # Initialise the lattice for a given system size, temperature, and external field
    def __init__(self, L, T, h):
        self.size = L
        self.temp = T
        self.field = h
        self.state = np.random.choice([-1,1], (L,L))
        self.mag = self.get_mag()
        self.energy = self.get_energy()
        
    # Set the state
    def set_state(self, state):
        self.state = state
        self.mag = self.get_mag()
        self.energy = self.get_energy()
     
    # Compute the magnetisation
    def get_mag(self):
        M = np.sum(self.state)  
        return M
    
    # Compute the energy
    def get_energy(self):
        En = 0
        for i in range(self.size):
            for j in range(self.size):
                En += -1*(self.state[i-1,j] + self.state[i,j-1])
        return En
    
    # Print the observables
    def print_obs(self):
        print('Magnetisation = ', self.mag, '\nEnergy = ', self.energy)
        

    def local_field(self, site):
        i = site[0]
        j = site[1]
        L = self.size
        S = self.state
        #modulu used to loop counting for site > L-1
        return self.field + S[(i+1)%L][j] + S[(i-1)%L][j] + S[i][(j+1)%L] + S[i][(j-1)%L]

    # Sweep through the lattice site by site and update the state,
    # magnetisation and energy at each step
    def sweep(self):
        
        # Define parameters
        L, T = self.size, self.temp
        
        # Loop over the lattice
        for ii in range(L):
            for jj in range(L):
                r = np.linspace(0,L-1,L)
                ii = int(np.random.choice(r))
                jj = int(np.random.choice(r))
                # Change of local spin and energy
                ds = -2*self.state[ii,jj]
                dE = -ds*self.local_field((ii,jj))
               
                # Acceptance ratio
                acc = np.exp(-dE/T)
                if acc < 1:
                    r = np.random.rand()
                    if r > acc:
                        # Reject the change
                        continue
                # Accept the change and update the state and observables
                self.state[ii,jj] = -self.state[ii,jj]
                self.mag += ds
                self.energy += dE
    

    # Metropolis Monte Carlo sampling of Ising states via sweeps
    def metropolis(self, n_sweeps = 1000, init_sweeps = 200):
        L = self.size
        T = self.temp
        # Initialise the lattice
        for nn in range(init_sweeps):
            self.sweep()
            
        # Run sweeps and store observables at each step
        M, dM, E, dE = 0.0, 0.0, 0.0, 0.0
        for nn in range(n_sweeps):
            self.sweep()
            M += (self.mag)/(L**2)
            E += (self.energy)/(L**2)
            dM += (self.mag**2)/(L**4)
            dE += (self.energy**2)/(L**4)
        # Perform averages
        M, E = M/n_sweeps, E/n_sweeps
        dM = (dM/n_sweeps - M**2)
        dE = (dE/n_sweeps - E**2)
        dE = dE/(T**2)
        dM = dM/(T)
        
        # Return results
        return M, dM, E, dE 


Ts = np.linspace(0.1,4,40)
sizes = [5,10,15,20]

h = 0



for L in sizes:  
    results = [IsingLattice2D(L,T,h).metropolis() for T in Ts]
    
    plt.figure(1)
    plt.scatter(Ts,[abs(r[0]) for r in results],s=10,label=('L = '+str(L)))
    plt.figure(2)
    plt.scatter(Ts,[r[1] for r in results],s=10,label=('L = '+str(L)))
    plt.figure(3)
    plt.plot(Ts,[r[2] for r in results],linestyle='-.',label=('L = '+str(L)))
    plt.figure(4)
    plt.scatter(Ts,[r[3] for r in results],s=10,label=('L = '+str(L)))

plt.figure(3)
plt.title('Mean Energy vs Temperature for Randomised Updating')
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.legend()


plt.figure(1)
plt.title('Average Magnetism per Spin for Randomised Updating')
plt.xlabel('Temperature')
plt.ylabel('Magnetism')
plt.legend()


plt.figure(2)
plt.title('Magnetic Susceptibility vs Temperature for Randomised Updating')
plt.xlabel('Temperature')
plt.ylabel('Magnetic Susceptibility')
plt.legend()

plt.figure(4)
plt.title('Heat Capacity vs Temperature for Randomised Updating')
plt.ylabel('Heat Capacity')
plt.xlabel('Temperature')
plt.legend()


Ls = np.linspace(2,40,31)
Tc = 2.3

models = [IsingLattice2D(int(l),Tc,h).metropolis() for l in Ls]
Cs = [m[3] for m in models]

plt.figure(5)
plt.scatter(Ls,[Cs[i]*(Ls[i]**2) for i in range(len(Cs))],s=10)
plt.title('Heat Capacity against System Size for Randomised Updating')
plt.xlabel('System Size')
plt.ylabel('Heat Capacity')
plt.show()

plt.figure(6)
plt.scatter(np.log10(Ls),[Cs[i]*(Ls[i]**2) for i in range(len(Cs))],s=10)
plt.title('Heat Capacity against Log of System Size for Randomised Updating')
plt.xlabel('System Size')
plt.ylabel('Heat Capacity')
plt.show()


#Counter used for my own sanity - we did hit 80 mins at one point
t2 = time.time()
time = (t2-t)/60
print('Time elapsed = ' + str(time))