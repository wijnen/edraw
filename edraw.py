#!/usr/bin/python3
# Python module for drawing electrical schematics.

# Imports. {{{
import sys
import os
import shutil
import math
import tempfile
import subprocess
import locale
locale.setlocale(locale.LC_NUMERIC, locale.getlocale())
# }}}

'''The purpose of this module is to make it easy to create high quality schematics for use in teaching materials.
This includes animations for educational videos.
'''

'''Animation: {{{
Selected properties can be animated by setting them to a function instead of a value. A list is below.

When animating, it is also possible to visualize charges.
The energy of a charge is the potential at its location.
The distance between charges is inversely proportional to the energy.
In order to limit changes to only positive, an offset to the potential is used.
Therefore 0V does not correspond to inifinite distance.
This offset and scale are specified as dist, which is the distance between charges at two potentials.

r = C / (V + V0)
r(d0) = C / (v0 + V0) => V0 = C / d0 - v0
r(d1) = C / (v1 + V0) => V0 = C / d1 - v1
	=> C * (d1 - d0) = v0 - v1
	=> C = (v0 - v1) / (d1 - d0)
}}}'''

# Directions are: {{{
# 0: right
# 1: up
# 2: left
# 3: down
# }}}

epsilon = 1e-5

active_debug = {}
#active_debug = {'compute', 'U'}
def debug(*args, **kwargs): # {{{
	if 'type' in kwargs and kwargs.pop('type') not in active_debug:
		return
	print(file = sys.stderr, *args, **kwargs)
# }}}

class Schematic: # {{{
	def __init__(self): # {{{
		self.terminals = {}
		self.components = {}
		self.wires = {}
		self.name_idx = {}
		self.current_time = 0
		self.scale = None	# mm per unit if number, output pixel size if tuple or list. Default is 50.
		self.dist = {0: 3, 12: 1}	# distance between charges at two voltages.
	# }}}
	def animate(self, total_time, target, fps = 1): # {{{
		'''Generate total_time images into the video file given by target.'''
		# Compute initial state and number of charges in the system.
		self.current_time = 0
		self.compute()
		(v0, d0), (v1, d1) = list(self.getanim(self.dist).items())
		C = (v0 - v1) / (d1 - d0)
		V0 = C / d0 - v0
		balance = [0]
		num_charges = 0
		def init_wire(wire, A, B, V): # {{{
			A = self.getanim(A)
			B = self.getanim(B)
			if V is None or math.isnan(V):
				V = 0
			r = C / (V + V0)
			length = sum((B[c] - A[c]) ** 2 for c in range(2)) ** .5
			rawnum = length / r + balance[0]
			num = round(rawnum)
			balance[0] = rawnum - num
			wire.charge = [((n + .5) / num, num_charges + n) for n in range(num)]
			return num
		# }}}
		for w in self.wires:
			wire = self.wires[w]
			A, B = [self.getanim(t.position) for t in wire.terminal]
			num_charges += init_wire(wire, A, B, wire.terminal[0].computed_V)
		tempdir = tempfile.mkdtemp(prefix='edraw-')
		path = os.path.join(tempdir, '%06d.svg')
		try:
			for frame in range(total_time * fps):
				self.current_time = frame / fps
				self.compute()
				# Recompute V0. {{{
				# First compute number of charges if V0 == 0 (or the minimum V in the setup if that is lower).
				min_V0 = max(1, min(terminal.computed_V for terminal in self.terminals.values()) + 1)
				num = 0
				lensum = 0
				for w, wire in self.wires.items():
					A, B = [self.getanim(terminal.position) for terminal in wire.terminal]
					r = C / ((wire.terminal[0].computed_V + wire.terminal[1].computed_V) / 2 + min_V0)
					length = sum((B[c] - A[c]) ** 2 for c in range(2)) ** .5
					num += length / r
					lensum += length
				# num_charges = sum(length[i]*V[i]/C) + sum(length[i]*V0/C)
				# lensum / C * V0 = num_charges - num
				V0 = (num_charges - num) / (lensum / C)
				# }}}
				# Move all charges. {{{
				for terminal in self.terminals.values():
					terminal.pending = []
				for component in self.components.values():
					component.pending = []
				def handle_wire(wire):
					#debug('handling wire', wire.name, wire.terminal[0].name, wire.terminal[0].computed_V)
					r = C / ((wire.terminal[0].computed_V + wire.terminal[1].computed_V) / 2 + V0)
					A, B = tuple(self.getanim(terminal.position) for terminal in wire.terminal)
					wire.length = sum((B[c] - A[c]) ** 2 for c in range(2)) ** .5
					wire.num_per_wire = wire.length / r
					wires_per_s = wire.computed_I / wire.num_per_wire
					wire.wires_per_frame = wires_per_s / fps
					#debug('speed', wire.name, wire.wires_per_frame)
					if wire.wires_per_frame == 0:
						return
					new_charges = []
					for i, (pos, n) in enumerate(wire.charge):
						# adjust speed for wire population.
						if 0 < i < len(wire.charge) - 1:
							mid = (wire.charge[i - 1][0] + wire.charge[i + 1][0]) / 2
						elif i == 0 and len(wire.charge) == 2:
							mid = max(0, wire.charge[1][0] - r)
						elif i == 1 and len(wire.charge) == 2:
							mid = min(1, wire.charge[0][0] + r)
						else:
							mid = pos
						pos += (mid - pos) / 4
						target_charge = pos + wire.wires_per_frame
						if target_charge < 0:
							wire.terminal[0].pending.append((-target_charge / wire.wires_per_frame, n))
						elif target_charge >= 1:
							wire.terminal[1].pending.append(((target_charge - 1) / wire.wires_per_frame, n))
						else:
							new_charges.append((target_charge, n))
					wire.charge = new_charges
				for wire in self.wires.values():
					handle_wire(wire)
				# }}}
				# Handle pending charges. {{{
				must_run = True
				while must_run:
					must_run = False
					for terminal in self.terminals.values():
						terminal.pending.sort()
						while len(terminal.pending) > 0:
							charge, n = terminal.pending.pop()
							wire = max(terminal.connection.values(), key = lambda x: x.hunger(terminal))
							target_charge = charge * wire.wires_per_frame
							if wire.terminal[0] is terminal:
								if len(wire.charge) == 0:
									if target_charge >= wire.length:
										wire.get_other(terminal).pending.append(((target_charge - 1) / wire.wires_per_frame), n)
										must_run = True
									else:
										wire.charge.append((target_charge, n))
								elif wire.charge[0][0] <= target_charge:
									wire.charge.insert(0, (wire.charge[0][0] * .9, n))
								else:
									wire.charge.insert(0, (target_charge, n))
							else:
								assert wire.terminal[1] is terminal
								if len(wire.charge) == 0:
									if target_charge >= 1:
										wire.get_other(terminal).pending.append(((target_charge - 1) / wire.wires_per_frame, n))
										must_run = True
									else:
										wire.charge.append((1 - target_charge, n))
								elif wire.charge[-1][0] >= 1 - target_charge:
									wire.charge.append((1 - (1 - wire.charge[-1][0]) * .9, n))
								else:
									wire.charge.append((1 - target_charge, n))
				# }}}
				# Write output. {{{
				dirname = os.path.dirname(path.format(0))
				os.makedirs(dirname, exist_ok = True)
				with open(path % frame, 'w') as f:
					f.write(self.svg())
				# }}}
			subprocess.run(('ffmpeg', '-y', '-r', str(fps), '-i', path, target))
		finally:
			shutil.rmtree(tempdir)
	# }}}
	def getanim(self, target): # {{{
		'''Get value of animatable property.
		Animatable properties are:
		Terminal.V
		Terminal.position
		Wire.I
		Wire.draw_I
		Wire.draw_V
		Component.position
		Component.angle
		Dipole.U
		Dipole.I
		Dipole.R
		Dipole.draw_U
		Dipole.draw_I
		Resistor.draw_R
		Switch.open
		'''
		if callable(target):
			return target(self.current_time)
		else:
			return target
	# }}}
	def mkname(self, type): # {{{
		if type not in self.name_idx:
			self.name_idx[type] = 0
		self.name_idx[type] += 1
		return '_auto_{}_{}'.format(type, self.name_idx[type])
	# }}}
	def Terminal(self, position, name = None): # {{{
		if name is None:
			name = self.mkname('terminal')
		assert name not in self.terminals
		self.terminals[name] = Terminal(self, position)
		self.terminals[name].name = name
		return self.terminals[name]
	# }}}
	def Wire(self, *connections, name = None): # {{{
		'''Create a wire through zero or more terminals.
		connections must have at least length 2.
		The first and last connection must be existing terminals or names of terminals.
		Other connections are coordinates.
		New terminals are created at those coordinates.
		This function creates n-1 Wire objects.
		It returns all the newly created terminals, not any Wire.'''
		assert len(connections) >= 2
		ret = []
		def mkwirename(i): # {{{
			if name is None:
				wname = self.mkname('wire')
			else:
				wname = name + '_{}'.format(i)
			assert wname not in self.wires
			return wname
		# }}}
		firstwire = mkwirename(0)
		A = connections[0] if isinstance(connections[0], Terminal) else self.terminals[connections[0]]
		B = connections[-1] if isinstance(connections[-1], Terminal) else self.terminals[connections[-1]]
		if len(connections) == 2:
			self.wires[firstwire] = Wire(self, A, B)
			ret = []
		else:
			self.wires[firstwire] = Wire(self, A, self.Terminal(connections[1]))
			ret = [self.wires[firstwire].terminal[1]]
		self.wires[firstwire].name = firstwire
		for i in range(1, len(connections) - 2):
			wname = mkwirename(i)
			self.wires[wname] = Wire(self, ret[-1], self.Terminal(connections[i + 1]))
			self.wires[wname].name = wname
			ret.append(self.wires[wname].terminal[1])
		if len(connections) > 2:
			wname = mkwirename(len(connections) - 1)
			self.wires[wname] = Wire(self, ret[-1], B)
			self.wires[wname].name = wname
		return ret
	# }}}
	def Add(self, component_class, position, *args, **kwargs): # {{{
		if 'name' not in kwargs:
			name = self.mkname(component_class.typename)
		else:
			name = kwargs.pop('name')
		assert name not in self.components
		if 'angle' not in kwargs:
			angle = 0
		else:
			angle = kwargs.pop('angle')
		if len(self.components) == 0 and 'V' not in kwargs:
			kwargs['V'] = 0
		self.components[name] = component_class(self, position, angle, *args, **kwargs)
		self.components[name].name = name
		return self.components[name]
	# }}}
	def compute(self): # {{{
		# Compute electrical quantities.
		# First reset all computed values.
		debug('restart computation', type = 'compute')
		for c in self.components:
			self.components[c].handle_time()
		for t in self.terminals:
			self.terminals[t].computed_V = self.getanim(self.terminals[t].V)
		for w in self.wires:
			self.wires[w].computed_I = self.getanim(self.wires[w].I)
			if self.wires[w].component is not None:
				self.wires[w].computed_U = None
			debug('wire I {} set to {}'.format(w, self.wires[w].computed_I), type = 'compute')
		# Compute all values based on possibly new settings.
		terminals = {self.terminals[t] for t in self.terminals if self.terminals[t].computed_V is None}
		wires = {self.wires[w] for w in self.wires if self.wires[w].computed_I is None}
		num = len(terminals) + len(wires)
		did_something = True
		while num > 0:
			did_something = False
			for c in self.components:
				self.components[c].handle_time()
			# Copy potentials over R=0 wires.
			for t in terminals.copy():
				t.compute_V()
				if t.computed_V is not None:
					terminals.remove(t)
			# Compute currents in wires.
			for w in wires.copy():
				old_U = w.computed_U
				w.compute_I()
				if w.computed_U != old_U:
					did_something = True
				if w.computed_I is not None:
					wires.remove(w)
			for c in self.components:
				self.components[c].compute_I()
			# Compute I of wires.
			# In a branch of series wires and components, with known V at ends, compute total R and total U.
			# Replace parallel branches of only R with replacement.
			# Compute I for the wires in the series.
			nodes = {}
			branches = {}
			for t, terminal in self.terminals.items():
				if t not in nodes:
					# All terminals in a node have the same V.
					nodes[t] = {'name': t, 'V': self.getanim(terminal.V), 'branch': set(), 'usable': True}
				for b in terminal.connection.values():
					o = b.get_other(terminal)
					if o is None:
						nodes[t]['usable'] = False
						continue
					o = o.name
					if b.terminal[0] is not terminal:
						# Wire in opposite direction; handle from other side.
						continue
					if o not in nodes:
						nodes[o] = {'name': o, 'V': self.getanim(self.terminals[o].V), 'branch': set()}
					key = frozenset({t, o})
					if key not in branches:
						branches[key] = {'key': key, 'node': [t, o], 'I': b.computed_I, 'wire': {(b.name, 1)}, 'R': self.getanim(b.R), 'U': b.computed_U}
						debug('U, R for {} set to {}, {}'.format(branches[key]['node'], branches[key]['U'], branches[key]['R']), type = 'compute')
					nodes[t]['branch'].add(key)
					nodes[o]['branch'].add(key)
			did_something2 = True
			while did_something2:
				did_something2 = False
				# Remove zero-length branches
				def remove_branch(b): # {{{
					did_something2 = True
					del branches[b]
					for n, node in nodes.items():
						if b in node['branch']:
							node['branch'].remove(b)
				# }}}
				def remove_circular(branch): # {{{
					debug('removing circular branch', branch['node'], type = 'compute')
					if branch['I'] is not None:
						I = branch['I']
					elif branch['R'] is not None and branch['U'] is not None:
						if abs(branch['R']) < epsilon:
							#self.dump()
							#debug(branch)
							assert abs(branch['U']) < epsilon
							return
						I = branch['U'] / branch['R']
					else:
						raise ValueError('Cannot compute circular branch')
					debug('removing circular', branch, type = 'compute')
					for w, sign in branch['wire']:
						if w in self.wires:
							debug('set I for circular', w, sign, I, branch['U'], branch['R'], type = 'compute')
							if self.wires[w].computed_I is None:
								did_something = True
								self.wires[w].computed_I = I * sign
								debug('circular I {} set to {}'.format(w, self.wires[w].computed_I), type = 'compute')
							else:
								#self.dump()
								#debug(w, self.wires[w].computed_I, I, sign)
								assert abs(self.wires[w].computed_I - I * sign) < epsilon
					remove_branch(branch['key'])
				# }}}
				for b in tuple(branches):
					branch = branches[b]
					if branch['node'][0] == branch['node'][1]:
						remove_circular(branch)
				def flip_branch(b): # {{{
					# invert branch
					b['node'] = b['node'][::-1]
					if b['I'] is not None:
						b['I'] *= -1
					if b['U'] is not None:
						b['U'] *= -1
					b['wire'] = {(w, -sign) for w, sign in b['wire']}
				# }}}
				# Merge branches on nodes with 2 connections.
				for n, node in tuple(nodes.items()):
					if len(node['branch']) != 2:
						continue
					did_something2 = True
					debug('merge serial branches', node['branch'], type = 'compute')
					A, B = (branches[x] for x in node['branch'])
					for i, b in enumerate((A, B)):
						if b['node'][i] is node['name']:
							flip_branch(b)
						assert b['node'][1 - i] is node['name']
					debug('merging', A, B, type = 'compute')
					if A['I'] is None and B['I'] is not None:
						A['I'] = B['I']
					elif B['I'] is None and A['I'] is not None:
						B['I'] = A['I']
					else:
						#self.dump()
						#debug(A, B)
						assert A['I'] is None or abs(A['I'] - B['I']) < epsilon
					if A['R'] is not None and B['R'] is not None:
						A['R'] += B['R']
					else:
						A['R'] = None
					if A['U'] is not None and B['U'] is not None:
						A['U'] += B['U']
					else:
						A['U'] = None
					A['node'][1] = B['node'][1]
					A['wire'].update(B['wire'])
					target = nodes[B['node'][1]]
					target['branch'].remove(B['key'])
					debug('merged', A, target, type = 'compute')
					remove_branch(B['key'])
					circular = A['key'] in target['branch']
					target['branch'].add(A['key'])
					if all(nodes[n]['V'] is not None for n in A['node']):
						I = (nodes[A['node'][0]]['V'] - nodes[A['node'][1]]['V']) / b['R']
						for w, sign in b['wire']:
							if self.wires[w].computed_I is None:
								did_something = True
								self.wires[w].computed_I = I * sign
								debug('set I from UR', w, I * sign, type = 'compute')
							else:
								assert abs(self.wires[w].computed_I - I * sign) < epsilon
					nodes.pop(n)
					if circular:
						remove_circular(A)
				# Remove branches of 0A from graph.
				for b in tuple(branches):
					if branches[b]['I'] == 0:
						remove_branch(b)
				# Merge branches connecting the same nodes
				for Akey in tuple(branches):
					if Akey not in branches:
						# Branch has been removed in previous iteration.
						continue
					A = branches[Akey]
					parI = set()	# parallel branches with valid I
					parR = set()	# parallel branches with valid R and no valid I
					parZ = set()	# parallel branches with R = 0
					io = set()	# branches connecting to the parallel branches, but which aren't parallel
					U = A['U']	# U of parallel stretch, if known.
					I = 0		# Current from parI that flows from first node of A
					I_in = 0
					I_out = 0
					all_IR = True	# set to False if any parallel branch is found with no I and no R.
					G = 0		# total conductivity of all branches in parR.
					for Bkey in tuple(branches):
						B = branches[Bkey]
						if not any(node in B['node'] for node in A['node']):
							# This branch is not connected to A.
							continue
						debug('merge attempt', B, type = 'U')
						if not all(node in B['node'] for node in A['node']):
							# This branch is connected to start or end only.
							# Update I_in or I_out.
							if A['node'][0] in B['node']:
								# update I_in.
								debug('I_in', B)
								if I_in is not None:
									if B['I'] is None:
										I_in = None
									elif A['node'][0] is B['node'][0]:
										I_in -= B['I']
									else:
										I_in += B['I']
							else:
								# update I_out.
								debug('I_out', B)
								if I_in is not None:
									if B['I'] is None:
										I_out = None
									elif A['node'][1] is B['node'][0]:
										I_out += B['I']
									else:
										I_out -= B['I']
							continue
						# This branch is parallel.
						if A['node'][0] != B['node'][0]:
							# Flip B.
							flip_branch(B)
						if B['U'] is not None:
							if U is None:
								U = B['U']
							else:
								debug('U', A['node'], Akey, Bkey, B['U'], U, type = 'compute')
								#self.dump()
								#debug('branches', branches)
								assert abs(B['U'] - U) < epsilon
						if B['I'] is not None:
							# Branch has valid I
							debug('add {} to parI'.format(B), type = 'U')
							parI.add(B['key'])
							if B['node'][0] is A['node'][0]:
								I += B['I']
							else:
								I -= B['I']
							continue
						if B['R'] is None:
							# No valid I and not valid R.
							debug('no all_IR because {}'.format(B), type = 'U')
							all_IR = False
							I_in = None
							I_out = None
							continue
						# Valid R.
						if abs(B['R']) < epsilon:
							parZ.add(B['key'])
							debug('short circuit in parallel system', B, type = 'compute')
						else:
							debug('add {} to parR'.format(B), type = 'U')
							parR.add(B['key'])
							G += 1 / B['R']
					assert I_in is None or I_out is None or abs(I_in - I_out) < epsilon
					if len(parZ) > 0:
						assert len(parI) == 0
						debug('handle short circuit', type = 'compute')
						# There is a short circuit. Set I to 0 for all branches with R != 0.
						for b in parR:
							branches[b]['I'] = 0
							for w in branches[b]['wire']:
								if wires[w]['I'] is None:
									did_something = True
									wires[w]['I'] = 0
							did_something2 = True
						# Push current through all wires.
						if I_in is not None or I_out is not None:
							for b in parZ:
								branches[b]['I'] = (I_in or I_out) / len(parZ)
								did_something2 = True
						continue
					# If all_IR is True, set I for all branches in parR.
					if all_IR:
						debug('setting I for branches to', I_in, I, G, type = 'compute')
						for b in parR:
							debug('branch', b, type = 'compute')
							Gb = 1 / branches[b]['R']
							branches[b]['I'] = (I_in - I) * (Gb / G)
							for (w, sign) in branches[b]['wire']:
								self.wires[w].computed_I = sign * branches[b]['I']
							parI.add(b)
							did_something2 = True
					# Merge all branches with valid I.
					if A['I'] is not None:
						debug('parallel merge', A, parI, I_in, U, type = 'U')
						A['I'] = sum(branches[b]['I'] for b in parI)
						A['U'] = U
						if U is None:
							A['R'] = None
						else:
							A['R'] = 0
						A['wire'] = set()
						for b in parI:
							if branches[b] is not A:
								remove_branch(b)
								did_something2 = True
						#debug(A)
					else:
						debug('no parallel merge', A, parI, type = 'U')
			if num == len(terminals) + len(wires) and not did_something:
				debug('nodes', '\n'.join('{}: {}'.format(n, nodes[n]) for n in nodes))
				debug('branches', '\n'.join('{}: {}'.format(b, branches[b]) for b in branches))
				self.dump()
				print('Warning: currents cannot be computed', file = sys.stderr)
				for t in terminals:
					assert t.computed_V is None
					t.computed_V = math.nan
				for w in wires:
					if w.computed_I is None:
						w.computed_I = math.nan
						debug('reset I {} set to {}'.format(w.name, w.computed_I), type = 'compute')
				break
			num = len(terminals) + len(wires)
	# }}}
	def dump(self): # {{{
		'''Dump all information about the schematic to screen.'''
		debug('Terminals:')
		for t in self.terminals:
			debug('\t{}: {} V; {}'.format(t, self.terminals[t].computed_V, ', '.join('{}: {} A'.format(w, self.wires[w].get_I(self.terminals[t])) for w in self.wires if self.terminals[t] in self.wires[w].terminal))) 
		debug('Components:')
		for c in self.components:
			debug('\t{}: {} A; {}'.format(c, self.components[c].computed_I, ', '.join(t for t in self.terminals if self.terminals[t] in self.components[c].terminal))) 
		debug('Wires:')
		for w in self.wires:
			debug('\t{}: {} A; {}'.format(w, self.wires[w].computed_I, ', '.join('{} ({} V)'.format(t.name, t.computed_V) for t in self.wires[w].terminal)))
	# }}}
	def display_value(self, value, digits, symbol): # {{{
		if digits is None:
			digits = 3
		if value == 0:
			# Special case.
			return ('{:#.%dn} {}' % digits).format(0., symbol)
		# 1 <= abs(value) < 10 ** digits: use value
		# 1 <= abs(value) * 1e3 < 10 ** digits: use k
		# 1 <= abs(value) * 1e6 < 10 ** digits: use M
		# 1 <= abs(value) * 1e-3 < 10 ** digits: use m
		# 1 <= abs(value) * 1e-6 < 10 ** digits: use μ
		# use scientific notation.
		v = abs(value)
		if math.isnan(v):
			return 'NaN'
		sign = '-' if value < 0 else ''
		for option in (('', 0), ('k', 3), ('M', 6), ('m', -3), ('μ', -6)):
			if .95 <= v / (10 ** option[1]) < 10 ** digits - .5:
				base = (option[0], 10 ** option[1])
				break
		else:
			# scientific notation.
			d = math.floor(math.log10(v)) + 1
			base = v / (10 ** (d - 1))
			superscript = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
			return ('{:#.%dn}·10{} {}' % digits).format(base, '{}'.format(d - 1).translate(superscript), symbol)
		d = math.floor(math.log10(v / base[1])) + 1
		comma = '#' if d < digits else ''
		return ('{}{:%s.%dn} {}{}' % (comma, digits)).format(sign, v / base[1], base[0], symbol)
	# }}}
	def svg(self): # {{{
		bbox = [[None, None], [None, None]]
		ret = ''
		def check_bbox(position, size): # {{{
			for c in range(2):
				target = position[c] - size[c] / 2
				if bbox[0][c] is None or target < bbox[0][c]:
					bbox[0][c] = target
				target = position[c] + size[c] / 2
				if bbox[1][c] is None or target > bbox[1][c]:
					bbox[1][c] = target
		# }}}
		for c in self.components:
			pos = self.getanim(self.components[c].position)
			angle = self.getanim(self.components[c].angle)
			if angle == 0:
				before, after = '', ''
			else:
				before, after = '<g transform="rotate({} {} {})">'.format(angle, pos[0], pos[1]), '</g>'
			ret += before + self.components[c].svg() + after
			check_bbox(pos, self.components[c].size)
		for w in self.wires:
			if self.wires[w].component is None:
				ret += self.wires[w].svg()
		for t in self.terminals:
			if self.terminals[t].component is None:
				pos = self.getanim(self.terminals[t].position)
				if len(self.terminals[t].connection) >= 3:
					ret += '<circle cx="{}" cy="{}" r=".1" stroke="none" fill="black"/>'.format(pos[0], pos[1])
				check_bbox(pos, (1, 1))
		# Draw all charges.
		for w in self.wires:
			wire = self.wires[w]
			A, B = [self.getanim(t.position) for t in wire.terminal]
			for c, n in wire.charge:
				pos = [A[t] + c * (B[t] - A[t]) for t in range(2)]
				ret += '<circle cx="{}" cy="{}" r=".1" stroke="none" fill="{}"/>'.format(pos[0], pos[1], '#0f08' if n % 10 == 0 else '#00f8')
		border = 1
		w = bbox[1][0] - bbox[0][0] + 2 * border
		h = bbox[1][1] - bbox[0][1] + 2 * border
		x = bbox[0][0] - border
		y = bbox[0][1] - border
		ymid = (bbox[0][1] + bbox[1][1]) / 2
		#debug(x, y, w, h, bbox)
		if self.scale is None:
			scale = 50
		elif isinstance(self.scale, (list, tuple)):
			# 90 pixels per inch hardcoded.
			# specified scale values are pixels.
			# Computed size value is mm.
			# 25.4 mm per inch.
			hsize = self.scale[0] / 90 * 25.4
			vsize = self.scale[1] / 90 * 25.4
			hscale = w / hsize
			vscale = h / vsize
			scale = max(hscale, vscale)
		else:
			scale = self.scale
		return '''\
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width='{}mm' height='{}mm' viewBox='{} {} {} {}' xmlns="http://www.w3.org/2000/svg" version="1.1" xmlns:xlink="http://www.w3.org/1999/xlink" style="stroke:black;fill:none;stroke-width:.05;stroke-linecap:round"><g transform="translate(0 {}) scale(1 -1) translate(0 {})"><rect x="{}" y="{}" width="{}" height="{}" fill="white" stroke="none"/>
'''.format(w * scale, h * scale, x, y, w, h, ymid, -ymid, x - w, y - h, w * 3, h * 3) + ret + '</g></svg>'
	# }}}
# }}}

class Wire: # {{{
	'''Wires connect two terminals.
	A wire has a direction. Positive current flows in the direction of the wire.
	'''
	debug = False
	def __init__(self, schematic, A, B): # {{{
		assert isinstance(schematic, Schematic)
		assert isinstance(A, Terminal)
		assert isinstance(B, Terminal)
		assert B not in A.connection
		assert A not in B.connection
		self.schematic = schematic
		self.terminal = (A, B)
		A.connection[B] = self
		B.connection[A] = self
		self.I = None
		self.R = 0	# overridden by component wires.
		self.computed_U = 0	# overridden for component wires.
		self.component = None
		self.draw_I = self.debug
		self.draw_V = self.debug
		self.computed_I = None
		self.digits = None
		self.charge = []
	# }}}
	def hunger(self, terminal): # {{{
		I = self.get_I(terminal)
		if I >= 0:
			return 0
		return self.num_per_wire / (len(self.charge) + 1)
	# }}}
	def get_I(self, terminal): # {{{
		'''Compute current that flows to the given terminal from this wire'''
		if self.computed_I is None:
			return None
		if terminal is self.terminal[0]:
			return -self.computed_I
		else:
			assert terminal is self.terminal[1]
			return self.computed_I
	# }}}
	def get_U(self, terminal): # {{{
		if self.computed_U is None:
			return None
		return self.computed_U if terminal is self.terminal[0] else -self.computed_U
	# }}}
	def get_other(self, terminal): # {{{
		if terminal is self.terminal[0]:
			return self.terminal[1]
		else:
			assert terminal is self.terminal[1]
			return self.terminal[0]
	# }}}
	def compute_I(self): # {{{
		I = [t.compute_I(self) for t in self.terminal]	# I are currents flowing into this wire.
		if all(i is None for i in I):
			return
		debug('computing I for {}: {}'.format(self.name, I), type = 'compute')
		#self.schematic.dump()
		assert I[0] is None or I[1] is None or abs(I[0] + I[1]) < epsilon
		self.computed_I = -I[1] if I[0] is None else I[0]
		debug('computed wire I {} set to {}'.format(self.name, self.computed_I), type = 'compute')
	# }}}
	def svg(self): # {{{
		pos = tuple(self.schematic.getanim(b.position) for b in self.terminal)
		ret = '<path d="M{},{}L{},{}"/>'.format(pos[0][0], pos[0][1], pos[1][0], pos[1][1])
		draw_I = self.schematic.getanim(self.draw_I)
		draw_V = self.schematic.getanim(self.draw_V)
		if draw_I or draw_V:
			x, y = ((pos[0][c] + pos[1][c]) / 2 for c in range(2))
			dx, dy = (pos[1][c] - pos[0][c] for c in range(2))
			angle = math.degrees(math.atan2(dy, dx))
			if draw_I:
				ret += '<text x="0" y="-.1" stroke="none" fill="green" font-size=".3" text-anchor="middle" transform="translate({},{}) rotate({}) scale(1,-1)">{}</text>'.format(x, y, angle, self.schematic.display_value(self.computed_I, self.schematic.getanim(self.digits), 'A'))
			if draw_V:
				ret += '<text x="0" y=".3" stroke="none" fill="blue" font-size=".3" text-anchor="middle" transform="translate({},{}) rotate({}) scale(1,-1)">{}</text>'.format(x, y, angle, self.schematic.display_value(self.terminal[0].computed_V, self.schematic.getanim(self.digits), 'V'))
		return ret
	# }}}
# }}}

class Terminal: # {{{
	'''A terminal is a point in the schematic. It has at most 4 connections.
	Each connection can be a wire or a component.
	'''
	def __init__(self, schematic, position): # {{{
		assert isinstance(schematic, Schematic)
		self.schematic = schematic
		self.connection = {}
		self.V = None
		self.position = position
		self.component = None
		self.computed_V = None
	# }}}
	def compute_V(self, force = False): # {{{
		if not force and self.computed_V is not None:
			return
		for wire in self.connection.values():
			other = wire.get_other(self)
			if other.computed_V is None:
				continue
			I = wire.get_I(self)	# I is current going out of the wire.
			R = self.schematic.getanim(wire.R)
			if wire.computed_U is not None:
				U = wire.get_U(self)	# U is voltage drop over the wire from this terminal.
				debug('use computed U', self.name, other.name, U, I, R, type = 'U')
			elif I is not None and R is not None and (I != 0 or not math.isinf(R)):
				U = -I * R
				debug('use I*R', self.name, other.name, U, I, R, type = 'U')
			elif R == 0:
				U = 0
				debug('use 0 R', self.name, other.name, U, I, R, type = 'U')
			elif R is not None and not math.isinf(R) and I == 0:
				U = 0
				debug('use 0 I', self.name, other.name, U, I, R, type = 'U')
			else:
				# Cannot compute V yet.
				debug('skip', self.name, other.name, I, R, type = 'U')
				continue
			assert self.computed_V is None or abs(self.computed_V - other.computed_V - U) < epsilon
			self.computed_V = other.computed_V + U
			debug(self.name, wire.name, "set myV to", other.computed_V, '+', U, '=', self.computed_V, type = 'compute')
	# }}}
	def compute_I(self, wire): # {{{
		'''Compute I that flows from this terminal into wire.'''
		total = 0
		debug('computing I for {}'.format(self.name), type = 'compute')
		assert wire is None or wire in self.connection.values()
		for w in self.connection.values():
			if w is wire:
				debug('own wire', type = 'compute')
				continue
			I = w.get_I(self)	# I is current flowing into this terminal.
			debug('I = {}'.format(I), type = 'compute')
			if I is None:
				return None
			total += I
		debug('total from {} into {} = {}'.format(self.name, wire.name, total), type = 'compute')
		return total
	# }}}
# }}}

class Component: # {{{
	'''A component has a shape and contains one or more terminals and wires.
	It has an angle.
	This is a base class from which actual components are derived.
	'''
	name = 'Component' # derived classes should replace this.
	def __init__(self, schematic, position, angle): # {{{
		assert isinstance(schematic, Schematic)
		self.schematic = schematic
		self.position = position
		self.angle = angle
		# The following members need to be created by the derived class.
		self.terminal = None	# This must be a list of new Terminals.
		# This needs to be filled in compute_I() of the derived class.
		self.computed_I = None
		#debug('initialized I {} to {}'.format(self.name, self.computed_I))
		self.wire = []
	# }}}
	def terminal_pos(self, offset): # {{{
		def tpos(time): # {{{
			angle = self.schematic.getanim(self.angle)
			sina = math.sin(math.radians(angle))
			cosa = math.cos(math.radians(angle))
			mypos = self.schematic.getanim(self.position)
			dx, dy = offset
			return ((cosa * dx - sina * dy) + mypos[0], (sina * dx + cosa * dy) + mypos[1])
		# }}}
		return tpos
	# }}}
	def __getitem__(self, item): # {{{
		'''Syntactic sugar for terminal access'''
		return self.terminal[item]
	# }}}
	def get_other(self, terminal): # {{{
		return None
	# }}}
	def handle_time(self):
		pass
	def compute_I(self): # {{{
		# This should be implemented by component classes.
		raise NotImplementedError('Component should implement compute_I')
	# }}}
	def svg(self): # {{{
		'''return the svg code for this component.'''
		raise NotImplementedError('Component should implement svg')
	# }}}
# }}}


class Dipole(Component): # {{{
	'''Component with one wire and two terminals'''
	name = 'Dipole'
	def __init__(self, schematic, position, angle, I = None, U = None, V = None, R = None): # {{{
		Component.__init__(self, schematic, position, angle)
		self.terminal = [schematic.Terminal(self.terminal_pos((offset * self.size[0] / 2, 0))) for offset in (-1, 1)]
		schematic.Wire(*self.terminal)
		self.wire = (self.terminal[0].connection[self.terminal[1]],)
		for w, wire in enumerate(self.wire):
			wire.component = (self, w)
			wire.computed_U = None
			wire.U = U
			wire.I = I
			wire.R = R
			wire.computed_I = I
		for t, terminal in enumerate(self.terminal):
			terminal.component = (self, t)
		self.terminal[1].V = V
		self.draw_I = True
		self.draw_U = True
		self.digits = None
	# }}}
	def get_other(self, terminal): # {{{
		if terminal is self.terminal[0]:
			return self.terminal[1]
		assert terminal is self.terminal[1]
		return self.terminal[0]
	# }}}
	def compute_I(self): # {{{
		self.wire[0].computed_U = self.schematic.getanim(self.wire[0].U)
		#debug('initial computed U for {} set to {}'.format(self.name, self.computed_U))
		if self.wire[0].computed_U is None and all(t.computed_V is not None for t in self.terminal):
			self.wire[0].computed_U = self.terminal[0].computed_V - self.terminal[1].computed_V
			#debug('computed U for {} set to {}'.format(self.name, self.wire[0].computed_U))
		#else:
		#	if self.wire[0].computed_U is None:
		#		debug('not setting U for {} with terminals {}, {}'.format(self.name, self.terminal[0].computed_V, self.terminal[1].computed_V))
		#		self.schematic.dump()
		R = self.schematic.getanim(self.wire[0].R)
		I = self.schematic.getanim(self.wire[0].I)
		debug('I for {} = {}'.format(self.name, I), type = 'compute')
		if I is not None:
			self.wire[0].computed_I = I
			debug('compute initialized dipole I {} to {}'.format(self.name, self.computed_I), type = 'compute')
		elif R is not None and self.wire[0].computed_U is not None and self.wire[0].computed_U != 0:
			self.wire[0].computed_I = self.wire[0].computed_U / R
			debug('computed dipole I {} from UR to {}'.format(self.name, self.computed_I), type = 'compute')
		else:
			debug('compute I', self.wire, type = 'compute')
			I = [t.compute_I(self.wire[0]) for t in self.terminal]	# I are currents flowing into the component.
			if all(i is None for i in I):
				self.computed_I = None
				return
			assert any(i is None for i in I) or abs(I[0] + I[1]) < epsilon
			self.wire[0].computed_I = -I[1] if I[0] is None else I[0]
			debug('computed dipole I {} from terminals to {}'.format(self.name, self.computed_I), type = 'compute')
		if self.wire[0].computed_U is None and self.wire[0].computed_I is not None and R is not None and (self.wire[0].computed_I != 0 or not math.isinf(R)):
			self.wire[0].computed_U = self.wire[0].computed_I * R
			debug('computed U for {} set to {}'.format(self.name, self.wire[0].computed_U), type = 'compute')
		elif self.wire[0].computed_U is None:
			debug('not computing U for {} I {} R {}'.format(self.name, self.computed_I, R), type = 'compute')
	# }}}
	def svg(self): # {{{
		ret = self.svg_shape()
		pos = self.schematic.getanim(self.position)
		if self.schematic.getanim(self.draw_I):
			ret += '<text x="0" y="0" stroke="none" fill="green" font-size=".3" text-anchor="middle" transform="translate({},{}) scale(1,-1)">{}</text>'.format(pos[0], pos[1] + self.size[1] / 2 + .2, self.schematic.display_value(self.wire[0].computed_I, self.schematic.getanim(self.digits), 'A'))
		if self.schematic.getanim(self.draw_U):
			ret += '<text x="0" y="0" stroke="none" fill="red" font-size=".3" text-anchor="middle" transform="translate({},{}) scale(1,-1)">{}</text>'.format(pos[0], pos[1] - self.size[1] / 2 - .6, self.schematic.display_value(self.terminal[0].computed_V - self.terminal[1].computed_V, self.schematic.getanim(self.digits), 'V'))
		return ret
	# }}}
# }}}

class Battery(Dipole): # {{{
	typename = 'Battery'
	size = (2, 2)
	def svg_shape(self): # {{{
		pos = self.schematic.getanim(self.position)
		return '<path d="M{},{}v2m.4,-1.5v1m0,-.5h.8m-2,0h.8"/>'.format(pos[0] - .2, pos[1] - 1)
	# }}}
# }}}

class Circle(Dipole): # {{{
	typename = 'Circle'
	size = (2, 1)
	def svg_shape(self): # {{{
		pos = self.schematic.getanim(self.position)
		return '<path d="M{},{}h.5a.5,.5,0,0,0,1,0a.5,.5,0,0,0,-1,0m1,0h.5"/>'.format(pos[0] - 1, pos[1])
	# }}}
# }}}

class Lamp(Circle): # {{{
	typename = 'Lamp'
	def svg_shape(self): # {{{
		pos = self.schematic.getanim(self.position)
		s2 = 2 ** .5 / 2
		h2 = s2 / 2
		return Circle.svg_shape(self) + '<path d="M{},{}l{},{}m0,{}l{},{}"/>'.format(pos[0] + h2, pos[1] - h2, -s2, s2, -s2, s2, s2)
	# }}}
# }}}

class Switch(Dipole): # {{{
	typename = 'Switch'
	size = (2, 1)
	def __init__(self, *args, **kwargs): # {{{
		if 'open' not in kwargs:
			self.open = True
		else:
			self.open = kwargs.pop('open')
		Dipole.__init__(self, *args, **kwargs)
	# }}}
	def handle_time(self): # {{{
		if self.schematic.getanim(self.open):
			debug('open', self.schematic.current_time, type = 'U')
			self.wire[0].I = 0
			self.wire[0].U = None
			self.wire[0].R = math.inf
		else:
			debug('closed', self.schematic.current_time, type = 'U')
			self.wire[0].I = None
			self.wire[0].U = 0
			self.wire[0].R = 0
	# }}}
	def svg_shape(self): # {{{
		pos = self.schematic.getanim(self.position)
		if self.schematic.getanim(self.open):
			return '<path d="M{},{}h.5l1.0,.4m0,-.3v-.1h.5"/>'.format(pos[0] - 1, pos[1])
		else:
			return '<path d="M{},{}h.5l1.1,.1m-.1,0v-.1h.5"/>'.format(pos[0] - 1, pos[1])
	# }}}
# }}}

class Resistor(Dipole): # {{{
	typename = 'Resistor'
	size = (3, 1)
	def __init__(self, *args, **kwargs): # {{{
		self.draw_R = True
		Dipole.__init__(self, *args, **kwargs)
	# }}}
	def svg_shape(self): # {{{
		pos = self.schematic.getanim(self.position)
		ret = '<path d="M{},{}h.5v.5h2v-1h-2v.5m2,0h.5"/>'.format(pos[0] - 1.5, pos[1])
		if self.schematic.getanim(self.draw_R):
			ret += '<text x="0" y="0" stroke="none" fill="black" font-size=".3" text-anchor="middle" transform="translate({},{}) scale(1,-1)">{}</text>'.format(pos[0], pos[1] - .1, self.schematic.display_value(self.schematic.getanim(self.wire[0].R), self.schematic.getanim(self.digits), 'Ω'))
		return ret
	# }}}
# }}}

class Diode(Dipole): # {{{
	typename = 'Diode'
	size = (2, 1)
	U_drop = .7
	def __init__(self, *args, **kwargs): # {{{
		if 'U' not in kwargs:
			kwargs['U'] = self.U_drop
		Dipole.__init__(self, *args, **kwargs)
	# }}}
	def svg_shape(self): # {{{
		pos = self.schematic.getanim(self.position)
		return '<path d="M{},{}h.5m1,0l-1,.5l0,-1l1,.5h.5m-.5,-.5v1"/>'.format(pos[0] - 1, pos[1])
	# }}}
# }}}

class LED(Diode): # {{{
	typename = 'LED'
	size = (2, 1)
	U_drop = 1.8
	def svg_shape(self): # {{{
		pos = self.schematic.getanim(self.position)
		arrow = "l.5,.5l-.2,-.1m.1,-.1l.1,.2"
		return Diode.svg_shape(self) + '<path d="M{},{}'.format(pos[0], pos[1] + .5) + arrow + 'm-.2,-.5' + arrow + '"/>'
	# }}}
# }}}

class Meter(Circle): # {{{
	typename = 'Meter'
	symbol = '?'
	def svg_shape(self): # {{{
		pos = self.schematic.getanim(self.position)
		return Circle.svg_shape(self) + '<text x="0" y="0" stroke="none" fill="black" font-size=".5" text-anchor="middle" transform="translate({},{}) scale(1,-1)">{}</text>'.format(pos[0], pos[1] - .2, self.symbol)
	# }}}
# }}}

class Ameter(Meter): # {{{
	typename = 'Ameter'
	symbol = 'A'
	def __init__(self, *args, **kwargs): # {{{
		kwargs['U'] = 0
		kwargs['R'] = 0
		Dipole.__init__(self, *args, **kwargs)
		self.draw_I = True
		self.draw_U = False
	# }}}
# }}}

class Vmeter(Meter): # {{{
	typename = 'Vmeter'
	symbol = 'V'
	def __init__(self, *args, **kwargs): # {{{
		kwargs['I']= 0
		Dipole.__init__(self, *args, **kwargs)
		self.draw_I = False
		self.draw_U = True
	# }}}
# }}}

# vim: set foldmethod=marker :
