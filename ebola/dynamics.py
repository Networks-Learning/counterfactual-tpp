"""
epidemic_helper.py: Helper module to simulate continuous-time stochastic
SIR epidemics.
Copyright © 2018 — LCA 4
"""
import time
import bisect
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import scipy.optimize
import scipy as sp
import random as rd
import heapq
import collections
import itertools
import os
import copy
from counterfactual_tpp import sample_counterfactual, combine
from sampling_utils import thinning_T
# from . import maxcut
from settings import DATA_DIR


def sample_seeds(graph, delta, method='data', n_seeds=None, max_date=None, verbose=True):
    """
    Extract seeds from the Ebola cases datasets, by choosing either:
        * the first `n_seeds`.
        * the first seed until the date `max_date`.
    For each seed, we then simulate its recovery time and attribute it to a random node in the
    corresponding district. We then start the epidemic at the time of infection of the last seed.
    Note that some seeds may have already recovered at this time. In this case, they are just
    ignored from the simulation altogether.

    Arguments:
    ---------
    graph : nx.Graph
        The graph of individuals in districts. Nodes must have the attribute `district`.
    delta : float
        Recovery rate of the epidemic process. Used to sample recovery times of seeds.
    n_seeds : int
        Number of seeds to sample.
    max_date : str
        Maximum date to sample seeds (max_date is included in sampling).
    method : str ('data' or 'random')
        Method to sample the seeds. Can be one of:
            - 'data': Use the seeds from the dataset and sample recovery time
            - 'random': Sample random seeds along with their recovery time
    verbose : bool
        Indicate whether or not to print seed generation process.
    """
    assert (n_seeds is not None) or (max_date is not None), "Either `n_seeds` or `max_date` must be given"
    
    if method == 'data':

        # Load real data
        df = pd.read_csv(os.path.join(DATA_DIR, 'ebola', 'rstb20160308_si_001_cleaned.csv'))
        if n_seeds:
            df = df.sort_values('infection_timestamp').iloc[:n_seeds]
        elif max_date:
            df = df[df.infection_date <= max_date].sort_values('infection_timestamp')
        # Extract the seed disctricts
        seed_names = list(df['district'])
        # Extract district name for each node in the graph
        node_names = np.array([u for u, d in graph.nodes(data=True)])
        node_districts = np.array([d['district'] for u, d in graph.nodes(data=True)])
        # Get last infection time of seeds (this is time zero for the simulation)
        last_inf_time = df.infection_timestamp.max()
        # Init list of seed events
        init_event_list = list()
        for _, row in df.iterrows():
            inf_time = row['infection_timestamp']
            # Sample recovery time
            rec_time = inf_time + rd.expovariate(delta) - last_inf_time
            # Ignore seed if recovered before time zero
            if rec_time > 0:
                # Randomly sample one node for each seed in the corresponding district
                idx = np.random.choice(np.where(node_districts == row['district'])[0])
                node = node_names[idx]
                # Add infection event
                # node to node infection flags initial seeds in code
                init_event_list.append([(node, 'inf', node), 0.0])  # Gets infection at the start
                # Add recovery event
                init_event_list.append([(node, 'rec', None), rec_time])
                if verbose:
                    print(f'Add seed {node} from district {row["district"]} - inf: {0.0}, rec: {rec_time} ')
        return init_event_list

    elif method == 'random':

        if n_seeds is None:
            raise ValueError("`n_seeds` must be provided for method `random`")
        
        init_event_list = list()
        for _ in range(n_seeds):
            node = np.random.choice(graph.nodes())
            init_event_list.append([(node, 'inf', node), 0.0])
            rec_time = rd.expovariate(delta)
            init_event_list.append([(node, 'rec', None), rec_time])

        return init_event_list

    else:
        raise ValueError('Invalid method.')


class PriorityQueue(object):
    """
    PriorityQueue with O(1) update and deletion of objects
    """

    def __init__(self, initial=[], priorities=[]):

        self.pq = []
        self.entry_finder = {}               # mapping of tasks to entries
        self.removed = '<removed-task>'      # placeholder for a removed task
        self.counter = itertools.count()     # unique sequence count

        assert(len(initial) == len(priorities))
        for i in range(len(initial)):
            self.push(initial[i], priority=priorities[i])

    def push(self, task, priority=0):
        """Add a new task or update the priority of an existing task"""
        if task in self.entry_finder:
            self.delete(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def delete(self, task):
        """Mark an existing task as REMOVED.  Raise KeyError if not found."""
        entry = self.entry_finder.pop(task)
        entry[-1] = self.removed

    def remove_all_tasks_of_type(self, type):
        """Removes all existing tasks of a specific type (for SIRSimulation)"""
        keys = list(self.entry_finder.keys())
        for event in keys:
            u, type_, v = event
            if type_ == type:
                self.delete(event)

    def pop_priority(self):
        """
        Remove and return the lowest priority task with its priority value.
        Raise KeyError if empty.
        """
        while self.pq:
            priority, _, task = heapq.heappop(self.pq)
            if task is not self.removed:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def pop(self):
        """
        Remove and return the lowest priority task. Raise KeyError if empty.
        """
        task, _ = self.pop_priority()
        return task

    def priority(self, task):
        """Return priority of task"""
        if task in self.entry_finder:
            return self.entry_finder[task][0]
        else:
            raise KeyError('task not in queue')
        
    def __len__(self):
        return len(self.entry_finder)

    def __str__(self):
        return str(self.pq)

    def __repr__(self):
        return repr(self.pq)

    def __setitem__(self, task, priority):
        self.push(task, priority=priority)


class ProgressPrinter(object):
    """
    Helper object to print relevant information throughout the epidemic
    """
    PRINT_INTERVAL = 0.1
    _PRINT_MSG = ('{t:.2f} days elapsed '
                  '| '
                  '{S:.0f} sus., '
                  '{I:.0f} inf., '
                  '{R:.0f} rec., '
                  '{Tt:.0f} tre ({TI:.2f}% of inf) | '
                  # 'I(q): {iq} R(q): {rq} T(q): {tq} |q|: {lq} | '
                  'max_u {max_u:.2e}'
                  )
    _PRINTLN_MSG = ('Epidemic stopped after {t:.2f} days '
                    '| '
                    '{S:.0f} sus., '
                    '{I:.0f} inf., '
                    '{R:.0f} rec., '
                    '{Tt:.0f} tre ({TI:.2f}% of inf) | '
                    # 'I(q): {iq} R(q): {rq} T(q): {tq} |q|: {lq}'
                    'max_u {max_u:.2e}'
                    )

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.last_print = time.time()

    def print(self, sir_obj, epitime, end='', force=False):
        if not self.verbose:
            return
        if (time.time() - self.last_print > self.PRINT_INTERVAL) or force:
            S = np.sum(sir_obj.is_sus)
            I = np.sum(sir_obj.is_inf * (1 - sir_obj.is_rec))
            R = np.sum(sir_obj.is_rec)
            T = np.sum(sir_obj.is_tre)
            
            Tt = np.sum(sir_obj.is_tre)
            TI = 100. * T / I if I > 0 else np.nan

            iq = sir_obj.infs_in_queue
            rq = sir_obj.recs_in_queue
            tq = sir_obj.tres_in_queue
            lq = len(sir_obj.queue)

            print('\r', self._PRINT_MSG.format(t=epitime, S=S, I=I, R=R, Tt=Tt, TI=TI,
                                               max_u=sir_obj.max_total_control_intensity),
                  sep='', end='', flush=True)
            self.last_print = time.time()

    def println(self, sir_obj, epitime):
        if not self.verbose:
            return
        S = np.sum(sir_obj.is_sus)
        I = np.sum(sir_obj.is_inf * (1 - sir_obj.is_rec))
        R = np.sum(sir_obj.is_rec)
        T = np.sum(sir_obj.is_tre)
        
        Tt = np.sum(sir_obj.is_tre)
        TI = 100. * T / I if I > 0 else np.nan
        
        iq = sir_obj.infs_in_queue
        rq = sir_obj.recs_in_queue
        tq = sir_obj.tres_in_queue
        lq = len(sir_obj.queue)

        print('\r', self._PRINTLN_MSG.format(
              t=epitime, S=S, I=I, R=R, Tt=Tt, TI=TI,
              max_u=sir_obj.max_total_control_intensity),
              sep='', end='\n', flush=True)
        self.last_print = time.time()


class SimulationSIR(object):
    """
    Simulate continuous-time SIR epidemics with treatement, with exponentially distributed
    inter-event times.

    The simulation algorithm works by leveraging the Markov property of the model and rejection
    sampling. Events are treated in order in a priority queue. An event in the queue is a tuple
    the form
        `(node, event_type, infector_node)`
    where elements are as follows:
    `node` : is the node where the event occurs,
    `event_type` : is the type of event (i.e. infected 'inf', recovery 'rec', or treatement 'tre')
    `infector_node` : for infections only, the node of caused the infection.
    """

    AVAILABLE_LPSOLVERS = ['scipy', 'cvxopt']

    def __init__(self, G, beta, gamma, delta, rho, verbose=True):
        """
        Init an SIR cascade over a graph
        
        Arguments:
        ---------
        G : networkx.Graph()
                Graph over which the epidemic propagates
        beta : float
            Exponential infection rate (positive)
        gamma : float
            Reduction in infection rate by treatment
        delta : float
            Exponential recovery rate (non-negative)
        rho : float
            Increase in recovery rate by treatment
        verbose : bool (default: True)
            Indicate the print behavior, if set to False, nothing will be printed
        """
        if not isinstance(G, nx.Graph):
            raise ValueError('Invalid graph type, must be networkx.Graph')
        self.G = G
        self.A = sp.sparse.csr_matrix(nx.adjacency_matrix(self.G).toarray())

        # Cache the number of nodes
        self.n_nodes = len(G.nodes())
        self.max_deg = np.max([d for n, d in self.G.degree()])
        self.min_deg = np.min([d for n, d in self.G.degree()])
        self.idx_to_node = dict(zip(range(self.n_nodes), self.G.nodes()))
        self.node_to_idx = dict(zip(self.G.nodes(), range(self.n_nodes)))

        # Check parameters
        if isinstance(beta, (float, int)) and (beta > 0):
            self.beta = beta
        else:
            raise ValueError("`beta` must be a positive float")
        if isinstance(gamma, (float, int)) and (gamma >= 0) and (gamma <= beta):
            self.gamma = gamma
        else:
            raise ValueError(("`gamma` must be a positive float smaller than `beta`"))
        if isinstance(delta, (float, int)) and (delta >= 0):
            self.delta = delta
        else:
            raise ValueError("`delta` must be a non-negative float")
        if isinstance(rho, (float, int)) and (rho >= 0):
            self.rho = rho
        else:
            raise ValueError("`rho` must be a non-negative float")

        # Control pre-computations
        self.lrsr_initiated = False   # flag for initial LRSR computation
        self.mcm_initiated = False    # flag for initial MCM computation

        # Control statistics
        self.max_total_control_intensity = 0.0

        # Printer for logging
        self._printer = ProgressPrinter(verbose=verbose)

    def expo(self, rate):
        """Samples a single exponential random variable."""
        return rd.expovariate(rate)

    def nodes_at_time(self, status, time):
        """
        Get the status of all nodes at a given time
        """
        if status == 'S':
            return self.inf_occured_at > time
        elif status == 'I':
            return (self.rec_occured_at > time) * (self.inf_occured_at < time)
        elif status == 'T':
            return (self.tre_occured_at < time) * (self.rec_occured_at > time)
        elif status == 'R':
            return self.rec_occured_at < time
        else:
            raise ValueError('Invalid status.')

    def _init_run(self, init_event_list, max_time):
        """
        Initialize the run of the epidemic
        """

        # Max time of the run
        self.max_time = max_time

        # Priority queue of events by time
        # event invariant is ('node', event, 'node') where the second node is the infector if applicable
        self.queue = PriorityQueue()
        # Cache the number of ins, recs, tres in the queue
        self.infs_in_queue = 0
        self.recs_in_queue = 0
        self.tres_in_queue = 0

        # Susceptible nodes tracking: is_sus[node]=1 if node is currently susceptible)
        self.initial_seed = np.zeros(self.n_nodes, dtype='bool')
        self.is_sus = np.ones(self.n_nodes, dtype='bool')                    # True if u susceptible

        # Infection tracking: is_inf[node]=1 if node has been infected
        # (note that the node can be already recovered)
        self.inf_occured_at = np.inf * np.ones(self.n_nodes, dtype='float')  # time infection of u_idx occurred
        self.is_inf = np.zeros(self.n_nodes, dtype='bool')                   # True if u_idx infected
        self.infector = -1 * np.ones(self.n_nodes, dtype='int')              # index of node that infected u_idx (if -1, then no infector)
        self.num_child_inf = np.zeros(self.n_nodes, dtype='int')             # number of neighbors u_idx infected

        # Recovery tracking: is_rec[node]=1 if node is currently recovered
        self.rec_occured_at = np.inf * np.ones(self.n_nodes, dtype='float')  # time recovery of u_idx occured
        self.is_rec = np.zeros(self.n_nodes, dtype='bool')                   # True if u_idx recovered

        # Treatment tracking: is_tre[node]=1 if node is currently treated
        self.tre_occured_at = np.inf * np.ones(self.n_nodes, dtype='float')  # time treatment of u_idx occured
        self.is_tre = np.zeros(self.n_nodes, dtype='bool')                   # True if u_idx treated

        # Conrol tracking
        self.old_lambdas = np.zeros(self.n_nodes, dtype='float')             # control intensity of prev iter
        self.max_interventions_reached = False

        # Add the initial events to priority queue
        for event, time in init_event_list:
            u, event_type, _ = event
            u_idx = self.node_to_idx[u]
            self.initial_seed[u_idx] = True
            if event_type == 'inf':
                # Initial infections have infections from NaN to u
                self.queue.push(event, priority=time)
                self.infs_in_queue += 1
            elif event_type == 'rec':
                self.queue.push(event, priority=time)
                self.recs_in_queue += 1
            else:
                raise ValueError('Invalid Event Type for initial seeds.')

    def _process_infection_event(self, u, time, w):
        """
        Mark node `u` as infected at time `time`
        Sample its recovery time and its neighbors infection times and add to the queue
        """
        # Get node index
        u_idx = self.node_to_idx[u]
        # Handle infection event
        self.is_inf[u_idx] = True
        self.is_sus[u_idx] = False
        self.inf_occured_at[u_idx] = time
        if self.initial_seed[u_idx]:
            # Handle initial seeds
            self.infector[u_idx] = -1
        else:
            w_idx = self.node_to_idx[w]
            self.infector[u_idx] = w_idx
            self.num_child_inf[w_idx] += 1
            recovery_time_u = time + self.expo(self.delta)
            self.queue.push((u, 'rec', None), priority=recovery_time_u)
            self.recs_in_queue += 1
        # Set neighbors infection events
        for v in self.G.neighbors(u):
            v_idx = self.node_to_idx[v]
            if self.is_sus[v_idx]:
                infection_time_v = time + self.expo(self.beta)
                self.queue.push((v, 'inf', u), priority=infection_time_v)
                self.infs_in_queue += 1

    def _process_recovery_event(self, u, time):
        """
        Mark node `node` as recovered at time `time`
        """
        # Get node index
        u_idx = self.node_to_idx[u]
        # Handle recovery event
        self.rec_occured_at[u_idx] = time
        self.is_rec[u_idx] = True

    def _process_treatment_event(self, u, time):
        """
        Mark node `u` as treated at time `time`
        Update its recovery time and its neighbors infection times and the queue
        """
        # Get node index
        u_idx = self.node_to_idx[u]
        # Handle treatement event
        self.tre_occured_at[u_idx] = time
        self.is_tre[u_idx] = True
        # Update own recovery event with rejection sampling
        assert(self.rho <= 0)
        if rd.random() < - self.rho / self.delta:
            # reject previous event
            self.queue.delete((u, 'rec', None))
            # re-sample
            new_recovery_time_u = time + self.expo(self.delta + self.rho)
            self.queue.push((u, 'rec', None), priority=new_recovery_time_u)
        # Update neighbors infection events triggered by u
        for v in self.G.neighbors(u):
            v_idx = self.node_to_idx[v]
            if self.is_sus[v_idx]:
                if rd.random() < self.gamma / self.beta:
                    # reject previous event
                    self.queue.delete((v, 'inf', u))
                    # re-sample
                    if self.beta - self.gamma > 0:
                        new_infection_time_v = time + self.expo(self.beta - self.gamma)
                    else:
                        # Avoid DivisionByZeroError if beta = gamma
                        # i.e., if no infectivity under treatement, then set infection time to inf
                        # We still set an event to make the algo easier and avoid bugs
                        new_infection_time_v = np.inf
                    self.queue.push((v, 'inf', u), priority=new_infection_time_v)

    def _control(self, u, time, policy='NO'):
        # Get node index
        u_idx = self.node_to_idx[u]
        # Check if max interventions were reached (for FL)
        if '-FL' in policy:
            max_interventions = self.policy_dict['front-loading']['max_interventions']
            current_interventions = np.sum(self.is_tre)
            if current_interventions > max_interventions:
                # End interventions for this simulation
                self.max_interventions_reached = True
                self.queue.remove_all_tasks_of_type('tre')
                print('All treatments ended')
                return
        # Compute control intensity
        self.new_lambda = self._compute_lambda(u, time, policy=policy)
        # Sample treatment event
        delta = self.new_lambda - self.old_lambdas[u_idx]
        if delta < 0:
            # Update treatment event with rejection sampling as intensity was reduced
            if rd.random() < 1 - self.new_lambda / self.old_lambdas[u_idx]:
                # reject previous event
                self.queue.delete((u, 'tre', None))
                if self.new_lambda > 0:
                    # re-sample
                    new_treatment_time_u = time + self.expo(self.new_lambda)
                    self.queue.push((u, 'tre', None), priority=new_treatment_time_u)
        elif delta > 0:
            # Sample new/additional treatment event with the superposition principle
            new_treatment_time_u = time + self.expo(delta)
            self.queue.push((u, 'tre', None), priority=new_treatment_time_u)
            self.tres_in_queue += 1
        # store lambda
        self.old_lambdas[u_idx] = self.new_lambda

    def _compute_lambda(self, u, time, policy='NO'):
        """Computes control intensity of the respective policy"""

        if policy == 'NO':
            return 0.0

        elif policy == 'TR':
            # lambda = const.
            return self.policy_dict['TR']

        elif policy == 'TR-FL':
            return self._frontloadPolicy(
                self.policy_dict['TR'],
                self.policy_dict['TR'], time)

        elif policy == 'MN':
            # lambda ~ deg(u)
            return self.G.degree(u) * self.policy_dict['MN']

        elif policy == 'MN-FL':
            return self._frontloadPolicy(
                self.G.degree(u) * self.policy_dict['MN'], 
                self.max_deg * self.policy_dict['MN'], time)

        elif policy == 'LN':
            # lambda ~ (maxdeg - deg(u) + 1)
            return (self.max_deg - self.G.degree(u) + 1) * self.policy_dict['LN']

        elif policy == 'LN-FL':
            return self._frontloadPolicy(
                (self.max_deg - self.G.degree(u) + 1) * self.policy_dict['LN'],
                (self.max_deg - self.min_deg + 1) * self.policy_dict['LN'], time)

        elif policy == 'LRSR':
            # lambda ~ 1/rank
            # where rank is order of largest reduction in spectral radius of A
            intensity, _ = self._compute_LRSR_lambda(u, time)
            return intensity

        elif policy == 'LRSR-FL':
            intensity, max_intensity = self._compute_LRSR_lambda(u, time)
            return self._frontloadPolicy(
                intensity, max_intensity, time)

        elif policy == 'MCM':
            # lambda ~ 1/rank
            # where rank is MCM heuristic ranking
            intensity, _ = self._compute_MCM_lambda(u, time)
            return intensity

        elif policy == 'MCM-FL':
            intensity, max_intensity = self._compute_MCM_lambda(u, time)
            return self._frontloadPolicy(
                intensity, max_intensity, time)

        elif policy == 'SOC':
            return self._compute_SOC_lambda(u, time)

        else:
            raise KeyError('Invalid policy code.')

        
    def launch_epidemic(self, init_event_list, max_time=np.inf, policy='NO', policy_dict={}, stop_criteria=None):
        """
        Run the epidemic, starting from initial event list, for at most `max_time` units of time
        """

        self._init_run(init_event_list, max_time)
        self.policy = policy
        self.policy_dict = policy_dict
        
        # Set SOC control parameters
        # TODO: Handle policy parameters better
        if policy == 'SOC':
            self.eta = policy_dict['eta']
            self.q_x = policy_dict['q_x']
            self.q_lam = policy_dict['q_lam']
            if policy_dict.get('lpsolver') in self.AVAILABLE_LPSOLVERS:
                self.lpsolver = policy_dict['lpsolver']
            else:
                raise ValueError("Invalid `lpsolver`")

        time = 0.0

        while self.queue:
            # Get the next event to process
            (u, event_type, w), time = self.queue.pop_priority()
            
            # Update queue cache
            if event_type == 'inf':
                self.infs_in_queue -= 1
            elif event_type == 'rec':
                self.recs_in_queue -= 1
            elif event_type == 'tre':
                self.tres_in_queue -= 1
            
            # Get node index
            u_idx = self.node_to_idx[u]
            
            # Stop at the end of the observation window
            if time > self.max_time:
                time = self.max_time
                break
            
            # Process the event
            # Check validity of infection event (node u is not infected yet)
            if (event_type == 'inf') and (not self.is_inf[u_idx]):
                assert self.is_sus[u_idx], f"Node `{u}` should be susceptible to be infected"
                w_idx = self.node_to_idx[w]
                if self.initial_seed[u_idx] or (not self.is_rec[w_idx]):
                    self._process_infection_event(u, time, w)
            # Check validity of recovery event (node u is not recovered yet)
            elif (event_type == 'rec') and (not self.is_rec[u_idx]):
                assert self.is_inf[u_idx], f"Node `{u}` should be infected to be recovered"
                self._process_recovery_event(u, time)
            # Check validity of treatement event (node u is not treated yet, and not recovered)
            elif (event_type == 'tre') and (not self.is_tre[u_idx]) and (not self.is_rec[u_idx]):
                assert self.is_inf[u_idx], f"Node `{u}` should be infected to be treated"
                self._process_treatment_event(u, time)

            # If no-one is infected, the epidemic is finished. Stop the simulation.
            if np.sum(self.is_inf * (1 - self.is_rec)) == 0:
                break

            if stop_criteria:
                if stop_criteria(self):
                    break

            # Update Control for infected nodes still untreated
            if not self.max_interventions_reached:
                controlled_nodes = np.where(self.is_inf * (1 - self.is_rec) * (1 - self.is_tre))[0]
                if self.policy == 'SOC':
                    self._update_LP_sol()
                for u_idx in controlled_nodes:
                    self._control(self.idx_to_node[u_idx], time, policy=self.policy)
                self.max_total_control_intensity = max(
                    self.max_total_control_intensity, self.old_lambdas.sum())

            self._printer.print(self, time)

        self._printer.println(self, time)

        # Free memory
        del self.queue

    def calculate_counterfactual(self, new_beta, G_prime, intervention_time = 0, vaccine = np.array([])):
        """
        Calculate the counterfactual infection and recovery times for the given intervention.
        input:
            new_beta: the new beta value (for the vaccination policy)
            G_prime: the graph for which the counterfactual infection and recovery times are calculated 
            (Can be the same as initial graph or different in graph isolation)
            intervention_time: the time at which the intervention is assumed to have happened
            vaccine: the vaccine policy
        """
    # initializing the counterfactual queue with seed nodes
        seeds = []
        seed_priorities = []
        processed = np.zeros(self.n_nodes, dtype = 'bool')
        for node in self.G.nodes():
            node_id = self.node_to_idx[node]
            if self.inf_occured_at[node_id] <= intervention_time and self.rec_occured_at[node_id] > intervention_time:
                seeds.append(node)
                seed_priorities.append(intervention_time)
            if self.rec_occured_at[node_id] <= intervention_time:
                processed[node_id] = True
        seeds = np.array(seeds)
        seed_priorities = np.array(seed_priorities)
        self.cf_queue = PriorityQueue(initial=seeds, priorities=seed_priorities)
        general_processed = copy.deepcopy(processed)
        beta_max = max(self.beta, new_beta)
        # initializing the counterfactual infection and recovery times
        self.cf_inf = np.zeros(self.n_nodes)
        self.cf_rec = np.zeros(self.n_nodes)
        self.cf_infector = np.full(self.n_nodes, -1)
        for node in self.G.nodes():
            node_id = self.node_to_idx[node]
            if general_processed[node_id]:
                self.cf_inf[node_id] = self.inf_occured_at[node_id]
                self.cf_rec[node_id] = self.rec_occured_at[node_id]
            else:
                if node in seeds:
                    self.cf_inf[node_id] = intervention_time
                    self.cf_rec[node_id] = self.cf_inf[node_id] + self.expo(self.delta)
                else:
                    self.cf_inf[node_id]= np.inf
        # processing nodes in order and calculate their counterfctual inf and rec times
        while self.cf_queue:
            node, _ = self.cf_queue.pop_priority()
            node_id = self.node_to_idx[node]
            if not processed[node_id]:
                processed[node_id] = True
                for neighbor in G_prime.neighbors(node):
                    neighbor_id = self.node_to_idx[neighbor]
                    if general_processed[neighbor_id]:
                        continue
                    def gamma2(t):
                        if len(vaccine) == 0 or vaccine[neighbor_id] == True:
                            g_beta = new_beta
                        else:
                            g_beta = self.beta
                        if t == self.cf_inf[node_id] or t == self.cf_rec[node_id]: return 1 * g_beta
                        return g_beta * np.heaviside(t - self.cf_inf[node_id], 1) - g_beta * np.heaviside(t - self.cf_rec[node_id], 0)
                    if self.infector[neighbor_id] == node_id:
                        t_inf = self.inf_occured_at[neighbor_id]
                        def gamma1(t):
                            if t == self.cf_inf[node_id] or t ==  min(self.cf_rec[node_id], t_inf): return 1 * self.beta
                            return self.beta * np.heaviside(t - self.cf_inf[node_id], 1) - self.beta * np.heaviside(t - min(self.cf_rec[node_id], t_inf), 0)   
                        ############# CF
                        gamma_bar = lambda t: beta_max - gamma1(t)
                        t_rejected= thinning_T(start=self.cf_inf[node_id], intensity=gamma_bar, lambda_max=beta_max, T= min(self.cf_rec[node_id], t_inf))
                        sample, lambdas, indicators = combine(np.array([t_inf]), np.array([gamma1(t_inf)]), t_rejected, gamma1)
                        counterfactuals, _ = sample_counterfactual(sample, lambdas, beta_max, indicators, gamma2)
                        ############# CF
                        if len(counterfactuals) != 0:
                            t_cf_inf = np.min(counterfactuals)
                        else:
                            t_cf_inf = np.inf
                    else:
                        H = thinning_T(start=0, intensity=gamma2, lambda_max=beta_max, T= self.max_time)
                        if len(H) != 0:
                            t_cf_inf = np.min(H)
                        else:
                            t_cf_inf = np.inf
                    if self.cf_inf[neighbor_id] > t_cf_inf:
                        self.cf_inf[neighbor_id] = t_cf_inf
                        self.cf_rec[neighbor_id] = t_cf_inf + self.expo(self.delta)
                        self.cf_infector[neighbor_id] = node_id
                        self.cf_queue.push(neighbor, priority=t_cf_inf)
        
        for rec_time in self.cf_rec:
            if rec_time > self.max_time:
                rec_time = np.inf
        # check this later.
        for node in self.G.nodes():
            if node in seeds:
                node_id = self.node_to_idx[node]
                self.cf_inf[node_id] = self.inf_occured_at[node_id]